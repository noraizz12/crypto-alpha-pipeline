import gzip
import io
import logging
import json
import os
from io import BytesIO
from typing import Optional, Dict, Any, List

import pandas as pd

# AWS dependencies - optional for local development
try:
    import boto3
    import botocore.config
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    ClientError = Exception  # Fallback for type hints


# Configuration via environment variables with sensible defaults
BUCKET = os.environ.get('S3_BUCKET', 'crypto-market-data')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
STATARB_SECRETID = os.environ.get('AWS_SECRET_ID', 'statarb/binance-ro')
BOTO_CLIENT_CONFIG = botocore.config.Config(max_pool_connections=50) if AWS_AVAILABLE else None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_env_credentials(secret_id: str) -> Optional[dict]:
    """Load credentials from environment variables as fallback.

    Maps common secret IDs to environment variable patterns:
    - statarb/binance-ro -> BINANCE_API_KEY, BINANCE_SECRET
    - statarb/binance-trading -> BINANCE_TRADING_API_KEY, BINANCE_TRADING_SECRET
    - statarb/clickhouse -> CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
    """
    env_mappings = {
        'statarb/binance-ro': {
            'API': os.environ.get('BINANCE_API_KEY', ''),
            'SECRET': os.environ.get('BINANCE_SECRET', ''),
        },
        'statarb/binance-trading': {
            'api_key': os.environ.get('BINANCE_TRADING_API_KEY', os.environ.get('BINANCE_API_KEY', '')),
            'secret_key': os.environ.get('BINANCE_TRADING_SECRET', os.environ.get('BINANCE_SECRET', '')),
            'API': os.environ.get('BINANCE_TRADING_API_KEY', os.environ.get('BINANCE_API_KEY', '')),
            'SECRET': os.environ.get('BINANCE_TRADING_SECRET', os.environ.get('BINANCE_SECRET', '')),
        },
        'statarb/clickhouse': {
            'host': os.environ.get('CLICKHOUSE_HOST', 'localhost'),
            'port': os.environ.get('CLICKHOUSE_PORT', '9000'),
            'user': os.environ.get('CLICKHOUSE_USER', 'default'),
            'password': os.environ.get('CLICKHOUSE_PASSWORD', ''),
        },
        'statarb/ops_genie_api_key': {
            'ops_genie_api_key': os.environ.get('OPSGENIE_API_KEY', ''),
        },
        'statarb/tardis_key': {
            'tardis_key': os.environ.get('TARDIS_API_KEY', ''),
        },
    }

    if secret_id in env_mappings:
        creds = env_mappings[secret_id]
        # Return only if at least one value is set
        if any(v for v in creds.values()):
            return creds

    return None


def load_aws_secrets(region: str = REGION, statarb_secretid: str = STATARB_SECRETID) -> Optional[dict]:
    """Load secrets from AWS Secrets Manager, falling back to environment variables.

    For local development without AWS access, set credentials via environment:
    - BINANCE_API_KEY, BINANCE_SECRET for Binance API access
    - CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD for database
    - TARDIS_API_KEY for Tardis market data
    - OPSGENIE_API_KEY for alerting
    """
    # Try environment variables first (for local development)
    env_creds = load_env_credentials(statarb_secretid)
    if env_creds:
        logger.info(f'Loaded credentials for {statarb_secretid} from environment variables')
        return env_creds

    # Fall back to AWS Secrets Manager
    if not AWS_AVAILABLE:
        logger.warning(f'AWS SDK not available and no environment credentials for {statarb_secretid}')
        return None

    try:
        client = boto3.client('secretsmanager', region_name=region, config=BOTO_CLIENT_CONFIG)
        response = client.get_secret_value(SecretId=statarb_secretid)
        secrets = json.loads(response['SecretString'])
        return secrets
    except ClientError as ce:
        logger.warning(f'Failed to load AWS secrets for {statarb_secretid}: {ce}')
    except Exception as e:
        logger.warning(f'Failed to load AWS secrets for {statarb_secretid}: {e}')

    return None


def _get_s3_client(region: str = REGION):
    return boto3.client('s3', region_name=region, config=BOTO_CLIENT_CONFIG)


def key_exists(key: str, bucket_name: str = BUCKET) -> bool:
    try:
        _get_s3_client().head_object(Bucket=bucket_name, Key=key)
        return True
    except Exception as e:
        pass
    return False


def get_dataframe_from_bucket(file_name: str, bucket_name: str = BUCKET,
                              float_types: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    logger.info(f"Loading remote file {file_name} from bucket {bucket_name}")

    if file_name.endswith('.parquet'):
        download_file_name = file_name.split('/')[-1]
        boto3.resource('s3').Bucket(bucket_name).download_file(file_name, download_file_name)
        try:
            df = pd.read_parquet(download_file_name)
        except:
            logger.error(f"Could not read {download_file_name}")
            raise
        finally:
            os.remove(download_file_name)
    elif file_name.endswith('.csv') or file_name.endswith('.csv.gz'):
        obj = boto3.resource('s3').Object(bucket_name, file_name)
        data = obj.get()['Body'].read()
        compression = 'gzip' if file_name.endswith('.gz') else None
        df = pd.read_csv(io.BytesIO(data), header=0, delimiter=",", low_memory=False, index_col=None,
                         compression=compression, engine='c', dtype=float_types)
    else:
        logger.error(f"Unknown file format {file_name}")
        raise RuntimeError()
    return df


def get_bucket(bucket_name: str):
    return boto3.resource('s3').Bucket(bucket_name)


def get_bucket_objs(bucket, filter_pattern: Optional[str] = None) -> list:
    if filter_pattern is not None:
        logging.info(f"Getting files from bucket {bucket} with pattern {filter_pattern}")
        objs = bucket.objects.filter(Prefix=filter_pattern)
    else:
        logging.info(f"Getting all files from {bucket=}")
        objs = bucket.objects.all()

    files = [obj.key for obj in objs]
    logging.info(f"Found {len(files)} files...")
    return files


def upload_file(file_name: str, object_name: str, bucket_name: str = BUCKET):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    logger.info(f"Uploading {file_name} to {object_name} on {bucket_name}")
    s3_client = _get_s3_client()
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        logging.error(f"Unable to upload {file_name}")
        logging.error(e)
        return False
    return True


def write_text_array_to_s3_gzip(text_array: List[str], key: str, bucket: str = BUCKET, region_name: str = REGION):
    """
    Writes an array of text data to an S3 bucket, compressed using gzip.

    Parameters:
    - text_array: List of text data
    - bucket: S3 bucket name
    - key: S3 key including filename and extension e.g., 'folder/filename.txt.gz'
    - region_name: AWS region name (default: None, uses boto's default behavior)
    """

    buffer = BytesIO()

    # Compress the text data using gzip and write to the buffer
    with gzip.GzipFile(fileobj=buffer, mode='w') as gzipped_file:
        for text in text_array:
            gzipped_file.write(text.encode('utf-8'))

    s3 = _get_s3_client(region_name)

    # Reset buffer's position
    buffer.seek(0)

    # Write buffer contents to S3
    s3.put_object(Body=buffer.getvalue(), Bucket=bucket, Key=key + ".gz", ContentEncoding='gzip')


def write_df_to_s3_parquet(df: pd.DataFrame, key: str, bucket: str = BUCKET, region_name: str = REGION):
    """
    Writes a pandas DataFrame to an S3 bucket as a Parquet file from an EC2 instance with an appropriate IAM role.

    Parameters:
    - df: pandas DataFrame
    - bucket: S3 bucket name
    - key: S3 key including filename and extension e.g., 'folder/filename.parquet'
    - region_name: AWS region name (default: None, uses boto's default behavior)
    """

    buffer = BytesIO()
    df.to_parquet(buffer)
    s3 = _get_s3_client(region_name)
    s3.put_object(Body=buffer.getvalue(), Bucket=bucket, Key=key)
