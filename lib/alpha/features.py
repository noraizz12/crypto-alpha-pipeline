"""Feature engineering pipeline for cryptocurrency trading signals.

This module implements a comprehensive feature engineering system that transforms
raw bar data into technical indicators and market microstructure features used
for alpha generation. It supports multiple time horizons and parallel processing
for efficient feature calculation across large universes of trading pairs.

Key Components:
    - Features class: Main pipeline for feature calculation and persistence
    - generate_rolling_features(): Parallel processing across time horizons
    - make_cx_features(): Cross-sectional (market-wide) feature generation

Feature Categories:
    - Price-based: Returns, volatility, min/max, trailing statistics
    - Volume-based: Dollar volume, trade counts, volume patterns
    - Microstructure: Spreads, bid-ask imbalance, order book metrics
    - Market-adjusted: Residual returns, beta, market-neutral signals
    - Technical indicators: RSI, moving averages, semi-variance
    - Time-based: Hour of day, day of week effects
    - News sentiment: Decayed news impact features

The pipeline handles:
    - Automatic lookback period management for rolling calculations
    - Memory-efficient chunked processing for large date ranges
    - Integration with multiple data sources (TARDIS, live)
    - Filtering for tradeable/fittable instruments
    - Cross-sectional feature normalization
    - Risk metric calculation at appropriate horizons

Typical Usage:
    # Generate features for multiple frequencies
    generate_rolling_features(
        config=config,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        frequencies=[15, 60, 1440],
        output_dir='/data/features'
    )
    
    # Or use Features class directly
    features = Features(config=config, frequency=60)
    features.run(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
"""

import os
import logging
from datetime import datetime as dt, date
from datetime import timedelta as td
from multiprocessing import Pool
from typing import List, Optional, Set, Literal, Tuple

import numpy as np
import pandas as pd

from lib.bars.bar_generator import FIXED_COLS_TO_PERSIST, BAR_FLDS_AGG_DICT

from lib.calcs.calcs import Calcs
from lib.data import dump_parquet_files
from lib.data.data_news import load_news
from lib.util.dataframes import check_df, make_date, merge_on_index, safe_del, set_index, shrink_floats
from lib.data.dataloader import DataLoader
from lib.data.loaders import load_mktcap_and_groupings_range
from lib.universe import Universe
from lib.util.directory import dir_manager, DirectoryManager
from lib.util.logging_util import KeyLogger
from lib.util.time_util import date_to_start_dt, today, yesterday_date
from lib.util.util import log_and_raise, unique_list, SYMBOL_VENUE
from lib.util.config import extract_feature_lookback_periods

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


def get_available_features_for_horizons(horizons: List[int]) -> List[str]:
    """
    Get list of available features by inspecting the directory structure.
    
    Args:
        horizons: List of horizons to check (e.g., [1440, 720])
        
    Returns:
        List of feature names available for ALL specified horizons
    """
    try:
        features_dir = dir_manager.FEATURES_DIR

        # Collect features available for each horizon
        horizon_features = {}
        for horizon in horizons:
            horizon_dir = os.path.join(features_dir, str(horizon))
            if os.path.exists(horizon_dir):
                # List all subdirectories (each represents a feature)
                features = []
                for item in os.listdir(horizon_dir):
                    item_path = os.path.join(horizon_dir, item)
                    if os.path.isdir(item_path):
                        # Each subdirectory name is a feature name
                        features.append(item)
                horizon_features[horizon] = set(features)
            else:
                logger.warning(f"Features directory not found for horizon {horizon}: {horizon_dir}")
                horizon_features[horizon] = set()

        # Return features that are available for ALL horizons
        if not horizon_features:
            return []

        # Find intersection of all feature sets
        common_features = None
        for horizon, features_set in horizon_features.items():
            if common_features is None:
                common_features = features_set
            else:
                common_features = common_features.intersection(features_set)

        # Sort and return
        return sorted(list(common_features)) if common_features else []

    except Exception as e:
        logger.error(f"Error getting available features: {e}")
        return []


def _run_feature(
        start_date: date,
        end_date: date,
        config: dict,
        frequency: int,
        universe: Universe,
        features: Optional[List[str]] = None,
        debug: bool = False,
        output_dir: Optional[str] = None
) -> None:
    """Worker function to generate features for a specific frequency.
    
    This function is designed to be called by multiprocessing pool workers
    to parallelize feature generation across different time horizons.
    
    Args:
        start_date: Start date for feature generation (inclusive).
        end_date: End date for feature generation (inclusive).
        config: Configuration dictionary containing all system parameters.
        frequency: Time horizon in minutes (e.g., 15, 60, 1440).
        features: Optional list of specific features to generate. If None,
            uses default features for the frequency.
        debug: If True, enables debug mode (no file output). Defaults to False.
        output_dir: Directory for output files. Required if not debug/prod mode.
    
    Raises:
        Exception: If output_dir is not specified when required.
    """

    symbol_venues = universe.load_universe_symbols(
        universe_source='bars',
        universe_date=start_date,
        symbol_type=SYMBOL_VENUE,
        frequency=frequency
    )
    if symbol_venues is None or len(symbol_venues) == 0:
        raise log_and_raise(f"Can't generate features on empty universe")

    logger.info(f"Generating features on universe size {len(symbol_venues)} {start_date}")
    logger.info(f"Uni symbols: {sorted(symbol_venues)}")

    Features(
        config=config,
        frequency=frequency,
        features=features,
        prod=False,
        debug=debug,
        output_dir=output_dir,
        symbol_venues=symbol_venues
    ).run(start_date=start_date, end_date=end_date)


def generate_rolling_features(
        config: dict,
        start_date: date,
        end_date: date,
        frequencies: List[int],
        pool_size: Optional[int] = None,
        chunk_days: int = 1,
        debug: bool = False,
        output_dir: Optional[str] = None,
        features: Optional[List[str]] = None
):
    """Generate features for multiple frequencies in parallel using chunked processing.
    
    This function orchestrates feature generation across multiple time horizons
    by dividing the date range into chunks and processing each chunk in parallel.
    This approach manages memory usage and enables efficient processing of large
    date ranges.
    
    Args:
        config: Configuration dictionary containing all system parameters.
        start_date: Start date for feature generation (inclusive).
        end_date: End date for feature generation (inclusive). Must be <= today.
        frequencies: List of time horizons in minutes (e.g., [15, 60, 1440]).
        pool_size: Number of parallel processes. If None, uses number of frequencies.
            Set to 1 to disable multiprocessing. Defaults to None.
        chunk_days: Number of days to process in each chunk. Helps manage memory
            usage for large date ranges. Defaults to 30.
        debug: If True, enables debug mode (no file output). Defaults to False.
        output_dir: Directory for output files. Required if not debug/prod mode.
        features: Optional list of specific features to generate. If None,
            uses default features for each frequency.
    
    Raises:
        AssertionError: If end_date > today or end_date < start_date.
        
    Notes:
        - Processes date range in chunks to manage memory usage
        - Each chunk is processed in parallel across frequencies
        - Chunks are processed sequentially to avoid overwhelming the system
        - Automatically limits end_date to yesterday if needed
    """
    assert end_date <= today().date()
    assert end_date >= start_date

    pool_size = len(frequencies) if pool_size is None else pool_size
    logger.info(f"Generating rolling features from {start_date} to {end_date} {pool_size=} {chunk_days=}")
    if features is not None:
        logger.info(f"Generating specific features: {features}")

    universe = Universe(config=config, debug=debug)

    window_start_date = start_date
    window_end_date = min(end_date, window_start_date + td(days=chunk_days), yesterday_date())
    while window_end_date <= end_date:
        logger.info(f"Generating chunk between {window_start_date} and {window_end_date}")
        if pool_size == 1:
            for frequency in frequencies:
                _run_feature(
                    start_date=window_start_date,
                    end_date=window_end_date,
                    config=config,
                    frequency=frequency,
                    universe=universe,
                    features=features,
                    debug=debug,
                    output_dir=output_dir,
                )
        else:
            pool = Pool(processes=pool_size)
            for _ in pool.starmap(
                    _run_feature,
                    [(window_start_date, window_end_date, config, frequency, universe, features, debug, output_dir) for frequency in frequencies]):
                pass
            pool.close()
            pool.join()

        if window_end_date >= end_date:
            break
        window_start_date = window_end_date
        window_end_date = min(end_date, window_start_date + td(days=chunk_days))


class Features:
    """Feature engineering pipeline for cryptocurrency trading signals.
    
    This class orchestrates the calculation of technical indicators, market
    microstructure features, and derived signals used for alpha generation.
    It handles data loading, feature computation, and persistence across
    multiple time horizons.
    
    The feature pipeline includes:
    - Price-based features (returns, volatility, min/max)
    - Volume features (dollar volume, trade counts)
    - Microstructure features (spreads, order book imbalance)
    - Market-adjusted features (residual returns, beta)
    - Technical indicators (RSI, moving averages)
    - News sentiment features
    - Time-based features
    
    Attributes:
        config: Configuration dictionary with all system parameters.
        frequency: Time horizon in minutes for feature calculation.
        prod: If True, only calculates production features. Defaults to False.
        debug: If True, enables debug mode (no file output). Defaults to False.
        bars_type: Type of bar data source ('live_prefer', 'tardis', 'new').
        output_dir: Directory for saving feature files.
        symbol_venues: List of symbol-venue pairs to process.
        features: List of feature names to calculate.
        calcs: Calcs instance for feature computation logic.
        features_calc_map: Dictionary mapping feature names to calculation configs.
    """

    def __init__(
            self,
            config: dict,
            frequency: int,
            prod: bool = False,
            debug: bool = False,
            features: Optional[List[str]] = None,
            output_dir: Optional[str] = None,
            features_dir_manager: DirectoryManager = dir_manager,
            symbol_venues: Optional[List[str]] = None
    ):
        """Initialize the Features pipeline.
        
        Args:
            config: Configuration dictionary containing all system parameters.
            frequency: Time horizon in minutes (e.g., 15, 60, 1440).
            prod: If True, only calculates production features. Defaults to False.
            debug: If True, enables debug mode (no file output). Defaults to False.
            features: Optional list of specific features to generate. If None,
                uses default features for the frequency from config.
            output_dir: Directory for output files. Required if not debug/prod mode.
            features_dir_manager: DirectoryManager instance for path management.
                Defaults to global dir_manager.
                
        Raises:
            Exception: If output_dir is not specified when required (not debug/prod).
            
        Notes:
            - Initializes feature calculation mappings for all supported features
            - Sets up data loader and calculation engine (Calcs)
        """
        self.config = config
        self.frequency = frequency
        self.prod = prod
        self.debug = debug
        self.symbol_venues = symbol_venues

        if output_dir is None and not self.debug and not self.prod:
            raise log_and_raise("Must set output_dir if not debug or prod")
        self.dir_manager = features_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.universe = Universe(config=self.config, debug=debug, universe_dir_manager=self.dir_manager)

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = self.dir_manager.FEATURES_DIR

        self.adv_lookback_days: int = self.config['ADV_LOOKBACK_DAYS']
        self.lookback_periods: int = extract_feature_lookback_periods(self.config, self.frequency)
        self.reopt_interval_mins: int = self.config['REOPTIMIZE_INTERVAL_MINS']
        self.opt_horizon: int = self.config['OPT_HORIZON']
        self.risk_horizon = self.config['RISK_HORIZON']
        self.st_risk_horizon = self.config['ST_RISK_HORIZON']
        self.similarity_threshold = self.config['NEWS_SIMILARITY_THRESHOLD']

        self.OTHER_FLDS: List[str] = ['close_mid']
        self.FREQ_FLDS = [
            'logret', 'logret_funding_adj', 'logret_resid_eqmkt', 'logret_resid_wgtmkt',
            'trade_cnt', 'update_cnt',
            'volume', 'dvolume',
            'spread_avg', 'bid_sz_avg', 'ask_sz_avg',
            'open_interest'
        ]

        self.recalc_all = features is None
        if features is not None:
            # Apply HORIZON replacement to custom features list
            self.features = [feature.replace('HORIZON', str(self.frequency)) for feature in features]
        else:
            self.features = self._get_default_features_list(prod)
        self.calcs = Calcs(config=self.config, frequency=frequency, prod=prod)

        self.features_calc_map = {
            f'logret_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
                'fld_feature_dependencies': [f'logret_{self.frequency}_trmean', f'logret_{self.frequency}_trstd']
            },
            f'dvolume_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'dvolume_{self.frequency}',
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'dvolume_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'dvolume_{self.frequency}',
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'dvolume_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'dvolume_{self.frequency}',
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'logret_resid_eqmkt_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_eqmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_eqmkt_{self.frequency}'],
            },
            f'logret_resid_eqmkt_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_eqmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_eqmkt_{self.frequency}'],
            },
            f'logret_resid_eqmkt_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_eqmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_eqmkt_{self.frequency}'],
            },
            f'logret_resid_wgtmkt_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_wgtmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_wgtmkt_{self.frequency}'],
            },
            f'logret_resid_wgtmkt_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_wgtmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_wgtmkt_{self.frequency}'],
            },
            f'logret_resid_wgtmkt_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_resid_wgtmkt_{self.frequency}',
                'fld_bar_dependencies': [f'logret_resid_wgtmkt_{self.frequency}'],
            },
            f'logret_{self.frequency}_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_lz_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'logret_{self.frequency}_lz',
                'fld_feature_dependencies': [f'logret_{self.frequency}_lz'],
            },
            f'logret_{self.frequency}_trstd_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'logret_{self.frequency}_trstd',
                'fld_feature_dependencies': [f'logret_{self.frequency}_trstd'],
            },
            f'logret_{self.frequency}_trstd_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}_trstd',
                'fld_feature_dependencies': [f'logret_{self.frequency}_trstd'],
            },
            f'logret_{self.frequency}_trstd_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}_trstd',
                'fld_feature_dependencies': [f'logret_{self.frequency}_trstd'],
            },
            f'logret_{self.frequency}_trstd_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'logret_{self.frequency}_trstd',
                'fld_feature_dependencies': [f'logret_{self.frequency}_trstd',
                                              f'logret_{self.frequency}_trstd_trmean',
                                              f'logret_{self.frequency}_trstd_trstd'],
            },
            f'logret_{self.frequency}_trstd_lz_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'logret_{self.frequency}_trstd_lz',
                'fld_feature_dependencies': [f'logret_{self.frequency}_trstd_lz'],
            },
            f'dvolume_{self.frequency}_trmean_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'dvolume_{self.frequency}_trmean',
                'fld_feature_dependencies': [f'dvolume_{self.frequency}_trmean'],
            },
            f'dvolume_{self.frequency}_lz_cz': {
                'func': self.calcs.calculate_risk_factor,
                'base_feature': f'dvolume_{self.frequency}_lz',
                'fld_feature_dependencies': [f'dvolume_{self.frequency}_lz'],
            },
            f'relative_spread_{self.frequency}': {
                'func': self.calcs.calculate_spread_factor,
                'fld_bar_dependencies': [f'spread_avg_{self.frequency}', 'close_mid'],
                'func_dependencies': [self.calcs.calculate_trailing_mean],
            },
            f'relative_spread_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'relative_spread_{self.frequency}',
                'fld_feature_dependencies': [f'relative_spread_{self.frequency}'],
                'fld_bar_dependencies': [f'spread_avg_{self.frequency}'],
            },
            f'relative_spread_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'relative_spread_{self.frequency}',
                'fld_feature_dependencies': [f'relative_spread_{self.frequency}'],
                'fld_bar_dependencies': [f'spread_avg_{self.frequency}'],
            },
            f'relative_spread_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'relative_spread_{self.frequency}',
                'fld_feature_dependencies': [f'relative_spread_{self.frequency}'],
                'fld_bar_dependencies': [f'spread_avg_{self.frequency}'],
            },
            f'logret_{self.frequency}_abs': {
                'func': self.calcs.calculate_abs_factor,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_lz_abs': {
                'func': self.calcs.calculate_abs_factor,
                'base_feature': f'logret_{self.frequency}_lz',
                'fld_feature_dependencies': [f'logret_{self.frequency}_lz'],
            },
            f'ba_imbal_{self.frequency}': {
                'func': self.calcs.calculate_bid_ask_imbalance,
                'fld_bar_dependencies': [f'bid_sz_avg_{self.frequency}', f'ask_sz_avg_{self.frequency}'],
            },
            f'ba_imbal_{self.frequency}_d': {
                'func': self.calcs.calculate_bid_ask_imbalance,
                'fld_bar_dependencies': [f'bid_sz_avg_{self.frequency}', f'ask_sz_avg_{self.frequency}'],
            },
            f'ba_imbal_{self.frequency}_dp': {
                'func': self.calcs.calculate_bid_ask_imbalance,
                'fld_bar_dependencies': [f'bid_sz_avg_{self.frequency}', f'ask_sz_avg_{self.frequency}'],
            },
            'news_event': {
                'func': self.calcs.calc_decayed_news,
                'fld_feature_dependencies': ['news_title'],
                'func_dependencies': [self.calcs.calculate_trailing_mean],
            },
            f'news_event_decayed_{self.frequency}': {
                'func': self.calcs.calc_decayed_news,
                'fld_feature_dependencies': ['news_title'],
                'func_dependencies': [self.calcs.calculate_trailing_mean],
            },
            f'news_event_decayed_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'news_event_decayed_{self.frequency}',
                'fld_feature_dependencies': [f'news_event_decayed_{self.frequency}'],
            },
            f'open_interest_{self.frequency}_d': {
                'func': self.calcs.calculate_delta,
                'base_feature': f'open_interest_{self.frequency}',
                'fld_bar_dependencies': [f'open_interest_{self.frequency}'],
            },
            f'open_interest_{self.frequency}_dp': {
                'func': self.calcs.calculate_delta,
                'base_feature': f'open_interest_{self.frequency}',
                'fld_bar_dependencies': [f'open_interest_{self.frequency}'],
            },
            f'rsi_{self.frequency}': {
                'func': self.calcs.calculate_rsi,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'trade_sz_{self.frequency}': {
                'func': self.calcs.calculate_trade_size,
                'fld_bar_dependencies': [f'volume_{self.frequency}', f'trade_cnt_{self.frequency}'],
            },
            f'trade_sz_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'trade_sz_{self.frequency}',
                'fld_feature_dependencies': [f'trade_sz_{self.frequency}']
            },
            f'trade_sz_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'trade_sz_{self.frequency}',
                'fld_feature_dependencies': [f'trade_sz_{self.frequency}']
            },
            f'trade_sz_{self.frequency}_lz': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'trade_sz_{self.frequency}',
                'fld_feature_dependencies': [f'trade_sz_{self.frequency}']
            },
            f'relative_updates_{self.frequency}': {
                'func': self.calcs.calculate_update_size,
                'fld_bar_dependencies': [f'update_cnt_{self.frequency}', f'volume_{self.frequency}'],
                'func_dependencies': [self.calcs.calculate_trailing_mean],
            },
            f'relative_updates_{self.frequency}_trmean': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'relative_updates_{self.frequency}',
                'fld_feature_dependencies': [f'relative_updates_{self.frequency}'],
            },
            f'relative_updates_{self.frequency}_trstd': {
                'func': self.calcs.calculate_trailing_mean,
                'base_feature': f'relative_updates_{self.frequency}',
                'fld_feature_dependencies': [f'relative_updates_{self.frequency}'],
            },
            f'relative_updates_{self.frequency}_lz': {
                'base_feature': f'relative_updates_{self.frequency}',
                'fld_feature_dependencies': [f'relative_updates_{self.frequency}'],
                'func': self.calcs.calculate_trailing_mean,
            },
            'hour_of_day': {
                'func': self.calcs.calculate_time_factors,
            },
            'day_of_week': {
                'func': self.calcs.calculate_time_factors,
            },
            f'dvolume_{self.frequency}_d': {
                'func': self.calcs.calculate_delta,
                'base_feature': f'dvolume_{self.frequency}',
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'dvolume_{self.frequency}_dp': {
                'func': self.calcs.calculate_delta,
                'base_feature': f'dvolume_{self.frequency}',
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'logret_{self.frequency}_min': {
                'func': self.calcs.calculate_minmax,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_max': {
                'func': self.calcs.calculate_minmax,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_min_cz': {
                'func': self.calcs.calculate_minmax,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_max_cz': {
                'func': self.calcs.calculate_minmax,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'beta_{self.frequency}': {
                'func': self.calcs.calculate_beta,
                'fld_bar_dependencies': [f"logret_funding_adj_{self.frequency}", f'logret_funding_adj_resid_wgtmkt_{self.frequency}', 'advp'],
                'func_dependencies': [self.calcs.calculate_weighted_mkt_return],
            },
            f'beta_t_{self.frequency}': {
                'func': self.calcs.calculate_beta,
                'fld_bar_dependencies': [f"logret_funding_adj_{self.frequency}", f'logret_funding_adj_resid_wgtmkt_{self.frequency}', 'advp'],
                'func_dependencies': [self.calcs.calculate_weighted_mkt_return],
            },
            f'logret_{self.frequency}_trstd_u': {
                'func': self.calcs.calculate_trailing_semi_variance,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_trstd_d': {
                'func': self.calcs.calculate_trailing_semi_variance,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_trstd_udratio': {
                'func': self.calcs.calculate_trailing_semi_variance,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_rj': {
                'func': self.calcs.calculate_extreme_metrics,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_exceedance': {
                'func': self.calcs.calculate_extreme_metrics,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'logret_{self.frequency}_csk': {
                'func': self.calcs.calculate_extreme_metrics,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            f'median_time_bucket_dvolume_{self.frequency}': {
                'func': self.calcs.calculate_volume_buckets,
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'pca_resid_{self.frequency}': {
                'func': self.calcs.calculate_pca,
                'base_feature': f'logret_{self.frequency}',
                'fld_bar_dependencies': [f'logret_{self.frequency}'],
            },
            'mdvp': {
                'func': self.calcs.calculate_mdvp,
                'fld_bar_dependencies': [f'dvolume_{self.frequency}'],
            },
            f'mktcap_turnover_{self.frequency}': {
                'func': self.calcs.calculate_mktcap_turnover,
                'fld_feature_dependencies': [f'dvolume_{self.frequency}_trmean'],
            },
        }

    def _generate_unique_function_calls(self, df_flds: Set[str]):
        """Generate ordered list of unique feature calculation function calls.
        
        This method analyzes the requested features and their dependencies to
        create an optimized execution plan. It ensures parent features are
        calculated before dependent features and eliminates duplicate calculations.
        
        Args:
            df_flds: Set of field names already present in the DataFrame.
                Used to skip calculations for existing features.
                
        Returns:
            List of tuples (function, base_feature) representing unique function
            calls in dependency order. Each tuple contains:
            - function: The calculation function to call
            - base_feature: The base feature name (or None for derived features)
            
        Notes:
            - Uses topological ordering to respect feature dependencies
            - Skips features already present in df_flds
            - Warns about features not found in calculation map
            - Eliminates duplicate function calls for efficiency
        """
        unique_calls = []
        seen_calls = set()
        visited = set()

        def _process_feature(feature: str, flds: Set[str]):
            logger.info(f"calculating {feature}")

            # Skip if already processed or not in map
            if feature in visited or feature not in self.features_calc_map:
                if feature not in flds and feature not in self.features_calc_map:
                    logger.warning(f"Not seeing {feature} in current df flds or features_calc_map")
                return

            # Process parent features first
            for parent in self.features_calc_map[feature].get('fld_feature_dependencies', []):
                _process_feature(parent, flds)

            config = self.features_calc_map[feature]
            func = config['func']
            base_feature = config.get('base_feature', None)
            call_key = (func, base_feature)
            if call_key not in seen_calls:
                unique_calls.append(call_key)
                seen_calls.add(call_key)
            visited.add(feature)

        for ff in self.features:
            _process_feature(ff, df_flds)

        return unique_calls

    def _get_default_features_list(self, prod: bool) -> List[str]:
        """Get the default feature list for the current frequency.
        
        Retrieves the appropriate feature list from the configuration based on
        the frequency and production mode. Features are defined in the config
        under FEATURES[frequency]['prod'] and FEATURES[frequency]['non_prod'].
        
        Args:
            prod: If True, returns only production features. If False, returns
                both production and non-production features.
                
        Returns:
            List of feature names with 'HORIZON' placeholder replaced by the
            actual frequency value. Returns empty list for 1-minute frequency.
            
        Notes:
            - 1-minute frequency returns empty list (raw data only)
            - Non-prod mode includes both prod and non_prod features
            - 'HORIZON' in feature names is replaced with actual frequency
            - Example: 'logret_HORIZON_lz' becomes 'logret_60_lz' for 60-min
        """
        if self.frequency == 1:
            logger.info("No features for 1-minute frequency!")
            return []

        features_list = self.config["FEATURES"].get(str(self.frequency), {}).get('prod', [])
        logger.info(f"Prod Feature list for {self.frequency} {prod=}: {features_list}")
        if not prod:
            non_prod_feature_list = self.config["FEATURES"].get(str(self.frequency), {}).get('non_prod', [])
            features_list += non_prod_feature_list
            logger.info(f"Non-Prod Feature list for {self.frequency} {prod=}: {non_prod_feature_list}")

        features_list = [feature.replace('HORIZON', str(self.frequency)) for feature in features_list]
        return features_list

    def _get_freq_flds(self, fld_type: Literal['bar', 'feature']) -> List[str]:
        """Get list of required fields from bar data for this frequency.
        
        Constructs the list of column names needed from the bar data based on
        the frequency. Frequency-specific fields have the frequency appended
        (e.g., 'logret_60' for 60-minute log returns).
        
        Returns:
            List of field names required from bar data, including:
            - OTHER_FLDS: Non-frequency fields like 'close_mid', 'advp'
            - FREQ_FLDS with frequency suffix: 'logret_60', 'volume_60', etc.
            
        Notes:
            - Used to specify which columns to load from bar files
            - Minimizes memory usage by loading only required fields
            - Frequency suffix matches the bar file column naming convention
        """

        dep_features = []
        if fld_type == 'bar':
            dep_features += ['advp', f'update_cnt_{self.frequency}', 'close_mid', 'priceable']

        fld = f'fld_{fld_type}_dependencies'
        for feature in self.features:
            logger.info(f"Getting {fld_type} parent features for {feature}")
            try:
                feature_dep_features = self.features_calc_map[feature]
                # gets either feature or bar dependencies
                if fld not in feature_dep_features:
                    logger.info(f"No {fld_type} dependencies for {feature}")
                    continue
                feature_dep_features = feature_dep_features[fld]
            except KeyError:
                logger.error(f"Could not look up {feature} in the function map!")
                raise
            logger.info(f"Adding dependent cols {feature_dep_features} for {feature}")
            dep_features += feature_dep_features

        flds = unique_list(dep_features)
        return flds

    def run_frequency(self, features_df: pd.DataFrame, start_dt: dt) -> pd.DataFrame:
        """Calculate all features for the current frequency.
        
        This is the core feature calculation method that orchestrates the
        computation of all requested features. It handles filtering, lookback
        period adjustment, feature calculation, and risk metrics.
        
        Args:
            features_df: DataFrame with bar data including all fields from
                get_freq_flds(). Must have MultiIndex (ts, symbol_venue).
            start_dt: Start datetime for feature calculation. Features before
                this date are used only for lookback calculations.
                
        Returns:
            DataFrame with all calculated features added. Only includes rows
            where fittable=True. Columns are shrunk to float32 for memory
            efficiency.
            
        Raises:
            Exception: If frequency is not in supported list.
            
        Notes:
            - Filters to fittable symbols only
            - Adjusts lookback periods based on frequency
            - Executes feature calculations in dependency order
            - Adds risk metrics for specific horizons (opt/st_risk)
            - All numeric columns converted to float32
        """
        logger.info(f"Calculating {self.frequency} features from start {start_dt}...")

        df_flds = set(features_df.columns)
        unique_function_calls = self._generate_unique_function_calls(df_flds)

        logger.info(f"starting with columns: {features_df.columns}...")
        while len(unique_function_calls) > 0:
            redo_calls = []
            for func, base_feature in unique_function_calls:
                if base_feature is None or base_feature in features_df.columns:
                    logger.info(f"Calculating {func.__name__}({base_feature})")
                    features_df = func(df=features_df, feature=base_feature, lookback_periods=self.lookback_periods, start_dt=start_dt, ffill_na=True)
                else:
                    logger.info(f"{base_feature} not in features_df, skipping...")
                    redo_calls.append((func, base_feature))
            logger.info(f"remaining {len(unique_function_calls)} function calls")
            if len(redo_calls) == len(unique_function_calls):
                logger.warning("Unable to calculate remaining features, breaking...")
                break
            unique_function_calls = redo_calls

        features_df = shrink_floats(features_df)
        check_df(features_df)

        # Only calculate risk if we're generating all features (not specific ones)
        # Risk calculation requires logret_HORIZON_trstd which may not be available when generating specific features
        if self.recalc_all:
            if self.frequency == self.risk_horizon:
                features_df = self.calcs.calculate_risk(features_df, risk_frequency=self.opt_horizon, sample_frequency=self.risk_horizon)
            elif self.frequency == self.st_risk_horizon:
                features_df = self.calcs.calculate_risk(features_df, risk_frequency=self.st_risk_horizon, sample_frequency=self.st_risk_horizon)
            else:
                logger.info(f"Not calculating risk for {self.frequency} features")
        else:
            logger.info(f"Not calculating risk for this run.  you are just calculating a single specified feature?")

        return features_df

    def _get_feature_columns(self) -> List[str]:
        """Get list of all feature columns created by this pipeline.

        Returns the complete list of feature columns that will be present
        in the output after running the feature pipeline. This is used to
        specify which columns to save to parquet files.

        Returns:
            List of feature column names from the Calcs instance. These are
            all the new columns added during feature calculation, not including
            the original bar data columns.

        Notes:
            - Delegates to self.calcs.get_new_cols()
            - Used by run() to determine which columns to persist
            - Does not include index columns (ts, symbol_venue)
            - Includes all features regardless of prod/non-prod mode
        """
        return self.calcs.get_new_cols()

    def run(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Execute the complete feature generation pipeline.
        
        This is the main entry point that orchestrates the entire feature
        calculation process: loading data, calculating features, and saving
        results. It handles lookback periods, news data integration, and
        output file generation.
        
        Args:
            start_date: Start date for feature generation (inclusive).
            end_date: End date for feature generation (inclusive).
            
        Returns:
            DataFrame containing all calculated features if successful.
            Returns None if debug=True (no DataFrame needed for debugging).
            
        Notes:
            - Automatically extends start_date backward for lookback calculations
            - Lookback period is min(lookback_periods * frequency, 90 days)
            - Special handling for 60-min frequency (45-day lookback for volume buckets)
            - Integrates news data if available
            - Adds delisting dates for risk management
            - Saves results to parquet files unless debug=True
            - For 1440-min frequency, recalculates ADVP within features
        """

        lookback_days = min(max(self.lookback_periods * self.frequency / 1440, 1),  90) + 1

        # maximum of 90 day lookback for features (otherwise 4320 forecasts needs too much)
        lookback_start_date = start_date - td(days=lookback_days)

        logger.info(f"Creating features for {self.frequency=} from {start_date} to {end_date} with lookback to {lookback_start_date} with universe size {self.symbol_venues}...")

        bars_df = self.data_loader.load_bars(
            horizon=self.frequency,
            start_date=lookback_start_date,
            end_date=end_date,
            cols=self._get_freq_flds(fld_type='bar'),
            symbol_venues=self.symbol_venues
        )

        if not self.recalc_all:
            feature_flds = self._get_freq_flds(fld_type='feature')
            if len(feature_flds) > 0:
                features_df = self.data_loader.load_features(
                    horizons=[self.frequency],
                    start_date=lookback_start_date,
                    end_date=end_date,
                    cols=feature_flds,
                    symbol_venues=self.symbol_venues
                )
                bars_df = merge_on_index(bars_df, features_df)

        bars_df = self.calcs.calculate_fittable_filter(bars_df)

        bars_df = make_date(bars_df)
        mkt_df = load_mktcap_and_groupings_range(
            start_date=lookback_start_date,
            end_date=end_date,
            mktcap_dir=self.dir_manager.MARKETCAP_DIR,
            categories_only=False,
            config=self.config
        )
        bars_df = set_index(pd.merge(bars_df.reset_index(), mkt_df, on=['symbol_venue', 'date'], how='left'))

        news_df = load_news(similarity_threshold=self.similarity_threshold, start_date=lookback_start_date, end_date=end_date, news_dir=self.dir_manager.NEWS_DIR)
        if news_df is not None:
            bars_df = merge_on_index(bars_df, news_df[['news_title']])
        else:
            bars_df['news_title'] = np.float32(0)
            logger.warning("Could not load news data!")

        if self.frequency == 1440:
            bars_df = self.data_loader.add_delisting_dates(bars_df, start_date=start_date, end_date=end_date)

        features_df = make_date(bars_df)
        start_dt = date_to_start_dt(start_date)

        features_df = self.run_frequency(features_df, start_dt)

        # raw titles not needed once attribute has been calced
        safe_del(features_df, 'news_title')

        cols = self._get_feature_columns()
        cols = list(set(cols) & set(self.features))
        dump_parquet_files(
            file_type='features',
            df=features_df,
            directory=f"{self.output_dir}/{self.frequency}",
            name=f"features.{self.frequency}",
            cols=cols,
            start_date=start_dt.date(),
            debug=self.debug
        )
        return features_df


def classify_features(features: List[str]) -> Tuple[List[str], List[str]]:
    """Classify features as bar file fields or feature file fields.

    This function determines whether each feature in a list comes from bar files
    or feature files. Bar file fields are identified using BAR_FLDS_AGG_DICT
    and FIXED_COLS_TO_PERSIST. Fields not found in these are assumed to be
    from feature files.

    Args:
        features: List of feature names to classify

    Returns:
        Tuple of (bar_features, feature_features) where:
            - bar_features: List of features from bar files
            - feature_features: List of features from feature files

    Notes:
        - Handles features with HORIZON placeholders or horizon suffixes
        - Removes _HORIZON or _{number} suffixes to find base field names
    """
    bar_features = []
    feature_features = []

    # Create a set of all bar field names for faster lookup
    bar_field_names = set(FIXED_COLS_TO_PERSIST)
    bar_field_names.update(BAR_FLDS_AGG_DICT.keys())
    
    # Add special calculated bar fields that are derived from bar data
    bar_field_names.add('vwap')
    bar_field_names.add('index_price_vwap')

    # Common horizon values
    horizons = ['15', '60', '120', '360', '720', '1440', '4320', '10080', '43200']

    for feature in features:
        is_bar_field = False

        # Check if it's a fixed column directly
        if feature in bar_field_names:
            is_bar_field = True
        else:
            base_field = feature

            # Remove _HORIZON suffix if present
            if '_HORIZON' in base_field:
                base_field = base_field.replace('_HORIZON', '')

            # Remove numeric horizon suffixes
            for horizon in horizons:
                if base_field.endswith(f'_{horizon}'):
                    base_field = base_field[:-len(f'_{horizon}')]
                    break

            # Check if the base field is a bar field
            if base_field in bar_field_names:
                is_bar_field = True
            else:
                # Check if this is an aggregated field (e.g., index_mid_price_diff_mean)
                # These come from fields in BAR_FLDS_AGG_DICT that have multiple aggregation methods
                for suffix in ['_mean', '_std', '_min', '_max', '_sum']:
                    if base_field.endswith(suffix):
                        potential_base = base_field[:-len(suffix)]
                        if potential_base in bar_field_names:
                            is_bar_field = True
                            break

        # Classify the feature
        if is_bar_field:
            bar_features.append(feature)
        else:
            feature_features.append(feature)

    logger.info(f"Classified {len(bar_features)} bar features and {len(feature_features)} feature file features")
    logger.debug(f"Bar features: {bar_features}")
    logger.debug(f"Feature file features: {feature_features}")

    return bar_features, feature_features
