"""ClickHouse database interface for high-performance time-series data storage.

This module provides a Python wrapper around ClickHouse, a columnar database optimized
for analytical queries on large datasets. It handles the system's storage needs for:
- Market data (OHLCV bars, trades, order book snapshots)
- Computed features and model outputs
- Trading activity logs and performance metrics
- Research and backtesting data

The module uses connection parameters stored in AWS Secrets Manager for security.

Constants:
    CLICKHOUSE_INSERT_BATCH_SIZE: Default batch size for bulk insertions (1M rows)
    CLICKHOUSE_HOST: Database host address from AWS secrets
    CLICKHOUSE_PORT: Connection port from AWS secrets
    CLICKHOUSE_USER: Authentication username from AWS secrets
    CLICKHOUSE_PASSWORD: Authentication password from AWS secrets
    CLICKHOUSE_DATABASE: Default database name ('statarb')

Example:
    Basic usage with context manager:
    ```python
    with ClickHouseDB() as db:
        df = db.fetch_df("SELECT * FROM bars WHERE symbol = 'BTCUSDT'")
    ```
"""
import logging
from typing import Any, Optional, Dict, List

import clickhouse_connect
import pandas as pd

from lib.util.aws import load_aws_secrets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default batch size for bulk insertions to balance memory usage and performance
CLICKHOUSE_INSERT_BATCH_SIZE = 1000000

# Load connection parameters from AWS Secrets Manager
CH_SECRET = load_aws_secrets(statarb_secretid='statarb/clickhouse')
CLICKHOUSE_HOST = CH_SECRET['host']
CLICKHOUSE_PORT = int(CH_SECRET['port'])
CLICKHOUSE_USER = CH_SECRET['user']
CLICKHOUSE_PASSWORD = CH_SECRET['password']
CLICKHOUSE_DATABASE = 'statarb'


class ClickHouseDB:
    """ClickHouse database connection and operations manager.

    Provides methods for connecting to ClickHouse, executing queries, and managing data.
    Supports context manager protocol for automatic connection cleanup.

    Attributes:
        host: Database host address
        port: Connection port number
        user: Authentication username
        password: Authentication password
        database: Target database name
        client: Internal ClickHouse client instance
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, user: Optional[str] = None, password: Optional[str] = None, database: Optional[str] = None):
        """Initialize ClickHouse connection parameters.

        Args:
            host: Database host (defaults to AWS secret value)
            port: Connection port (defaults to AWS secret value)
            user: Username (defaults to AWS secret value)
            password: Password (defaults to AWS secret value)
            database: Database name (defaults to 'statarb')
        """
        self.host = host if host is not None else CLICKHOUSE_HOST
        self.port = port if port is not None else CLICKHOUSE_PORT
        self.user = user if user is not None else CLICKHOUSE_USER
        self.password = password if password is not None else CLICKHOUSE_PASSWORD
        self.database = database if database is not None else CLICKHOUSE_DATABASE
        logger.info(f"Connecting to clickhouse: {self.host}:{self.port} {self.user}@{self.database}")

        self.client = None

    def connect(self) -> None:
        """Establish connection to ClickHouse server.

        Creates a client instance using configured connection parameters.

        Raises:
            Exception: If connection fails (network error, auth failure, etc.)
        """
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    def close(self) -> bool:
        """Close the database connection.

        Safely closes the ClickHouse client connection and cleans up resources.

        Returns:
            bool: True if successful, False if error occurred during close
        """
        if self.client is None:
            logger.info("ClickHouse client was already closed")
            return True
        try:
            self.client.close()
            self.client = None
            logger.info("ClickHouse client closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close ClickHouse client: {e}")
            return False

    def __enter__(self):
        """Context manager entry - establishes database connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()

    def fetch(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries.

        Args:
            query: SQL query string (can include parameter placeholders)
            params: Optional dict of query parameters for safe substitution

        Returns:
            List[Dict[str, Any]]: Query results where each dict represents a row

        Example:
            results = db.fetch(
                "SELECT * FROM bars WHERE symbol = {symbol:String} AND ts > {ts:DateTime}",
                {"symbol": "BTCUSDT", "ts": "2024-01-01 00:00:00"}
            )

        Raises:
            Exception: If query execution fails
        """
        logger.info(query)
        try:
            result = self.client.query(query, parameters=params)
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        except Exception as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}")
            raise

    def fetch_df(self, query: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Execute query and return results as pandas DataFrame.

        Uses Arrow format for efficient data transfer and streams results in chunks
        to handle large datasets without memory issues.

        Args:
            query: SQL query string (can include parameter placeholders)
            params: Optional dict of query parameters for safe substitution

        Returns:
            pd.DataFrame: Query results as DataFrame, or None if no results

        Example:
            df = db.fetch_df(
                "SELECT ts, symbol, close_mid, volume FROM bars "
                "WHERE ts BETWEEN {start:DateTime} AND {end:DateTime}",
                {"start": "2024-01-01", "end": "2024-01-31"}
            )

        Raises:
            Exception: If query execution fails
        """
        logger.info(query)
        try:
            chunk_list = []
            with self.client.query_arrow_stream(query, parameters=params) as stream:
                for arrow_table in stream:
                    df_chunk = arrow_table.to_pandas()
                    chunk_list.append(df_chunk)

            df = pd.concat(chunk_list)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}")
            raise

    def execute(self, query: str, params: Optional[dict] = None) -> None:
        """Execute a command without returning results (DDL, DML operations).

        Use for CREATE, DROP, ALTER, INSERT, UPDATE, DELETE statements.

        Args:
            query: SQL command string
            params: Optional dict of command parameters

        Example:
            db.execute("CREATE TABLE test (id UInt64, name String) ENGINE = MergeTree() ORDER BY id")
            db.execute("DROP TABLE IF EXISTS old_table")

        Raises:
            Exception: If command execution fails
        """
        try:
            self.client.command(query, parameters=params)
        except Exception as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}")
            raise

    def insert_df(self, df: pd.DataFrame, table_name: str, batch_size: int = CLICKHOUSE_INSERT_BATCH_SIZE) -> bool:
        """Insert pandas DataFrame into ClickHouse table using batched insertion.

        Splits large DataFrames into batches to avoid memory issues and provides
        progress logging at 10% intervals.

        Args:
            df: DataFrame to insert (column names must match table schema)
            table_name: Target table name
            batch_size: Number of rows per batch (default 1M)

        Returns:
            bool: True if successful, False if any error occurred

        Example:
            success = db.insert_df(
                df=price_data,
                table_name='bars',
                batch_size=500000  # Custom batch size for smaller chunks
            )

        Notes:
            - Column names in DataFrame must match table column names
            - Data types are automatically converted where possible
            - Progress is logged at 10% completion intervals
            - Memory-efficient batch processing prevents OOM errors
        """
        total_rows = len(df)
        column_names = list(df.columns)
        last_percentage_logged = -1

        try:
            # Insert in smaller batches to avoid memory issues
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                # Convert to tuples for the current batch only
                data_tuples = [tuple(row) for row in batch_df.itertuples(index=False, name=None)]
                # Insert the current batch
                self.client.insert(table_name, data_tuples, column_names=column_names)
                # Log at 10% increments
                current_percentage = int((end_idx / total_rows) * 10)
                if current_percentage > last_percentage_logged:
                    logger.info(f"Progress: {current_percentage * 10}% - Inserted {end_idx}/{total_rows} rows into {table_name}")
                    last_percentage_logged = current_percentage
            logger.info(f"Successfully inserted all {total_rows} rows into {table_name}.")
        except Exception as e:
            logger.warning(f"Error inserting data into ClickHouse: {e}")
            return False

        return True

    def insert_list(self, table: str, data: List[Dict[str, Any]], column_keys: Optional[List[str]] = None) -> int:
        """Insert list of dictionaries into ClickHouse table.

        Converts list of dicts to tuples and performs bulk insertion.

        Args:
            table: Target table name
            data: List of dictionaries to insert
            column_keys: Optional list specifying column order (defaults to keys from first dict)

        Returns:
            int: Number of rows inserted

        Example:
            rows = db.insert_list(
                table='trades',
                data=[
                    {'symbol': 'BTCUSDT', 'price': 50000, 'volume': 1.5},
                    {'symbol': 'ETHUSDT', 'price': 3000, 'volume': 10.2}
                ]
            )

        Raises:
            Exception: If insertion fails

        Notes:
            - All dictionaries must have the same keys
            - If column_keys provided, dictionaries must contain all specified keys
        """
        if len(data) == 0:
            return 0

        try:
            columns = list(data[0].keys()) if column_keys is None else column_keys
            values = [tuple(row[col] for col in columns) for row in data]
            self.client.insert(table, values, column_names=columns)
        except Exception as e:
            logger.error(f"Error inserting data into {table}: {e}")
            raise

        return len(data)
