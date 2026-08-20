import argparse
import logging.config
import sys
from datetime import timedelta

from lib.bars.tardis import Tardis, TARDIS_BAR_START_DATE
from lib.data import find_latest_prebar_date
from lib.universe import Universe
from lib.util.config import get_config
from lib.util.logging_util import get_logging_config
from lib.util.slack import JobMonitor
from lib.util.tardis_analyzer import TardisDataAnalyzer
from lib.util.time_util import yesterday_date, date_str_to_date
from lib.util.util import TARDIS_API_KEY

logging.config.dictConfig(get_logging_config("download_tardis"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-u', '--update', help='use existing bar file and update', dest='update', action='store_true', required=False)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-b', '--backfill', dest='backfill', action='store_true', required=False)
    parser.add_argument('-bm', '--backfill-missing', dest='backfill_missing', action='store_true', required=False, 
                        help='Find and backfill symbols with missing data')
    parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', required=False)
    parser.add_argument('-s', '--s3', dest='s3', action='store_true', required=False)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=8)
    parser.add_argument('-fc', '--force', dest='force', action='store_true', required=False)
    parser.add_argument('-dt', '--data-types', help='tardis data types', required=False, type=str)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(debug=False, s3=False, overwrite=False, force=False, backfill=False, backfill_missing=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))

    update = args['update']
    s3 = args['s3']
    pool_size = args['pool_size']
    debug = args['debug']
    backfill = args['backfill']
    backfill_missing = args['backfill_missing']
    overwrite = args['overwrite']
    force = args['force']
    notify_channel = args['notify']
    quiet = args['quiet']

    data_types = args.get('data_types')
    if data_types is not None:
        data_types = data_types.split(',')

    with JobMonitor("Download tardis", notify_channel, debug, quiet):
        # Handle different modes of operation
        if backfill_missing:
            # When backfill-missing is specified, we skip the regular download
            # and go straight to finding and backfilling missing symbols
            pass  # Skip regular download
        elif update:
            end_date = yesterday_date()
            start_date = find_latest_prebar_date(venue='tardis')
            logger.info(f"Updating bar files from {start_date} to {end_date}")
            if start_date == end_date:
                logger.warning(f"preBar files already generated to {start_date}...")
                sys.exit(1)
            
            # Do the regular download for update mode
            Tardis(
                config=config,
                api_key=TARDIS_API_KEY,
                s3=s3,
                pool_size=pool_size,
                debug=debug,
                backfill=backfill,
                overwrite=overwrite,
                force=force,
                data_types=data_types,
            ).download_files(start_date=start_date, end_date=end_date)
        else:
            # Regular mode with explicit dates
            start_date = date_str_to_date(args['from'])
            end_date = date_str_to_date(args['to'])
            
            if start_date is None or end_date is None:
                logger.error("Must specify --from and --to dates, or use --update or --backfill-missing mode")
                sys.exit(1)
            
            # Do the regular download
            Tardis(
                config=config,
                api_key=TARDIS_API_KEY,
                s3=s3,
                pool_size=pool_size,
                debug=debug,
                backfill=backfill,
                overwrite=overwrite,
                force=force,
                data_types=data_types,
            ).download_files(start_date=start_date, end_date=end_date)

        # Determine symbols that need backfilling
        new_symbols = []
        
        if backfill_missing:
            # Use TardisDataAnalyzer to find symbols with missing data
            logger.info("Analyzing tardis data coverage to find symbols with missing data...")
            analyzer = TardisDataAnalyzer(config=config)
            coverage_data = analyzer.find_symbol_coverage()
            
            # Find symbols with poor coverage (e.g., < 90%) or missing recent dates
            symbols_to_backfill = []
            for symbol_info in coverage_data:
                # Skip symbols with no data at all (they might be new)
                if symbol_info['start_date'] is None:
                    continue
                    
                # Check if symbol has poor coverage
                if symbol_info['coverage_pct'] < 90:
                    symbols_to_backfill.append(symbol_info['symbol'])
                    logger.info(f"Symbol {symbol_info['symbol']} has {symbol_info['coverage_pct']:.1f}% coverage with {symbol_info['missing_count']} missing days")
                    
                # Also check for symbols missing recent dates
                elif symbol_info['missing_dates']:
                    # Check if any missing dates are recent (within last 7 days)
                    recent_cutoff = yesterday_date() - timedelta(days=7)
                    recent_missing = [d for d in symbol_info['missing_dates'] if d >= recent_cutoff]
                    if recent_missing:
                        symbols_to_backfill.append(symbol_info['symbol'])
                        logger.info(f"Symbol {symbol_info['symbol']} missing recent dates: {recent_missing}")
            
            new_symbols = symbols_to_backfill
            logger.info(f"Found {len(new_symbols)} symbols needing backfill based on coverage analysis")
        else:
            # Use original logic to get new symbols from universe
            new_symbols = Universe(config=config, debug=debug).get_todays_new_symbols()
        
        if len(new_symbols) > 0:
            logger.info(f"Backfilling {new_symbols} from {TARDIS_BAR_START_DATE} to {yesterday_date()}")
            # download bars only for the new symbols
            bar_data_types = ['trades', 'book_snapshot_5']
            Tardis(
                config=config,
                api_key=TARDIS_API_KEY,
                s3=s3,
                pool_size=pool_size,
                symbols=new_symbols,
                debug=debug,
                backfill=True,
                overwrite=overwrite,
                force=force,
                data_types=bar_data_types
            ).download_files(start_date=TARDIS_BAR_START_DATE, end_date=yesterday_date())

            # download fundings for the whole universe since the funding file structured with all symbols inside parquet file
            logger.info(f"Backfilling fundings from {TARDIS_BAR_START_DATE} to {yesterday_date()}")
            funding_data_types = ['derivative_ticker']
            Tardis(
                config=config,
                api_key=TARDIS_API_KEY,
                s3=s3,
                pool_size=pool_size,
                symbols=new_symbols,
                debug=debug,
                backfill=True,
                overwrite=overwrite,
                force=force,
                data_types=funding_data_types
            ).download_files(start_date=TARDIS_BAR_START_DATE, end_date=yesterday_date())
        
        # Final log message
        if backfill_missing:
            if len(new_symbols) > 0:
                logger.info(f"Done backfilling {len(new_symbols)} symbols with missing data")
            else:
                logger.info("No symbols found that need backfilling")
        elif update or (args['from'] and args['to']):
            logger.info(f"Done downloading tardis")
