"""Server-side horizon-specific calculations for real-time trading.

This module provides the ServerHorizonCalcs class which handles real-time
feature and model calculations for a specific time horizon in the live trading
system. It manages incremental updates to features and models as new market
data arrives, ensuring efficient computation without recalculating historical data.

The class is designed to be used by the AlphaServer for each configured horizon,
maintaining state and performing calculations only on new data points to minimize
latency in the live trading environment.

Example:
    >>> config = {'REOPTIMIZE_INTERVAL_MINS': 120, 'FCASTS': {'60': {'models': [...]}}}
    >>> features_df = pd.DataFrame(...)  # Initial features with MultiIndex (ts, symbol_venue)
    >>> calc = ServerHorizonCalcs(config, horizon=60, features_df=features_df)
    >>> 
    >>> # Update with new market data
    >>> new_data = pd.DataFrame(...)  # New 1-minute bars
    >>> calc.update_df(new_data)
    >>> 
    >>> # Calculate features and models for the new data
    >>> calc.calculate_features(carry_forward_map={60: ['fittable', 'tradeable']})
    >>> calc.calculate_and_lag_models()
"""

import logging
from datetime import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd

from lib.calcs import Calcs
from lib.util.config import extract_tree_features
from lib.util.dataframes import concat, get_min_max_ts, carry_forward, set_index, check_df_column_changes
from lib.alpha.features import Features
from lib.calcs.calc_util import make_cx_features
from lib.alpha.models import Models
from lib.util.directory import dir_manager, DirectoryManager
from lib.util.util import log_and_raise
from lib.util.logging_util import KeyLogger
from lib.alpha.model_calcs import generate_model_lags

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

BOOL_FEATURE_COLS = ['tradeable', 'expandable', 'priceable']

class ServerHorizonCalcs:
    """Manages horizon-specific calculations for real-time trading server.
    
    This class handles incremental feature and model calculations for a specific
    time horizon in the live trading system. It maintains state to track what has
    been processed and only calculates on new data to minimize latency.
    
    The class is designed to work with the AlphaServer, processing new market data
    as it arrives and updating features and models incrementally. It handles:
    - Feature calculation for the specific horizon
    - Model signal generation using configured alpha models
    - Lag generation for autoregressive model features
    - Carry-forward of certain fields like liquidity flags
    
    Attributes:
        config (dict): Configuration dictionary containing model specs and parameters
        horizon (int): Time horizon in minutes (e.g., 15, 60, 120, 1440)
        features_df (pd.DataFrame): DataFrame with MultiIndex (ts, symbol_venue) containing
            features and models for this horizon
        last_processed_ts (dt): Timestamp of last processed data, used to identify new data
        calcs (Calcs): Calculator instance for volume forecasts and other calculations
        features (Features): Feature engineering instance for this horizon
        models (Models): Model calculation instance for alpha signal generation
        reopt_interval (int): Interval in minutes for reoptimization calculations
    """
    
    def __init__(self, config: dict, horizon: int, features_df: pd.DataFrame, server_horizon_calcs_dir_manager: DirectoryManager = dir_manager):
        """Initialize ServerHorizonCalcs for a specific time horizon.
        
        Args:
            config: Configuration dictionary containing:
                - FCASTS: Forecast model configurations by horizon
                - REOPTIMIZE_INTERVAL_MINS: Reoptimization interval
            horizon: Time horizon in minutes (must match a key in config['FCASTS'])
            features_df: Initial features DataFrame with MultiIndex (ts, symbol_venue)
            server_horizon_calcs_dir_manager: Directory manager for file paths
        """
        self.config = config
        self.horizon = horizon

        self.features_df = features_df
        self.last_processed_ts = self.features_df.index.get_level_values('ts').max()

        self.calcs = Calcs(config=config, prod=True)
        self.features = Features(config=self.config, frequency=self.horizon, prod=True, features_dir_manager=server_horizon_calcs_dir_manager)
        self.models = Models(config=self.config)

        self.reopt_interval = self.config['REOPTIMIZE_INTERVAL_MINS']

    def get_max_ts(self) -> dt:
        """Get the maximum timestamp in the features DataFrame.
        
        Returns:
            Maximum timestamp from the 'ts' level of the MultiIndex.
        """
        return self.features_df.index.get_level_values('ts').max()

    def update_df(self, new_df: pd.DataFrame):
        """Update the features DataFrame with new market data.
        
        Concatenates new data to the existing features DataFrame, performing
        column validation to ensure consistency. This is typically called
        when new bar data arrives in the live trading system.
        
        Args:
            new_df: DataFrame with new data to append. Must have the same
                MultiIndex structure (ts, symbol_venue) and compatible columns.
        
        Raises:
            ValueError: If new_df has incompatible columns with features_df.
        """
        old_ts = self.get_max_ts()
        old_cnt = len(self.features_df)
        check_df_column_changes(self.features_df, new_df)

        self.features_df = concat([self.features_df, new_df], quiet=False)
        self.features_df['fittable'] = True
        new_ts = self.get_max_ts()
        new_cnt = len(self.features_df)
        logger.info(f"Updating dataframes at {self.horizon=} from {old_ts} to {new_ts}, rows {old_cnt} -> {new_cnt}")

    def update_last_processed_ts(self):
        """Update the last processed timestamp to the current maximum.
        
        This should be called after processing features and models to mark
        the data as processed. The next calculation cycle will only process
        data newer than this timestamp.
        """
        self.last_processed_ts = self.get_max_ts()

    def calculate_features(self, carry_forward_map: Dict[int, List[str]]) -> None:
        """Calculate features for new data points at this horizon.
        
        Performs feature calculation including:
        - Carrying forward specified fields (e.g., liquidity flags)
        - Calculating volume forecasts at reoptimization intervals
        - Running horizon-specific feature engineering
        
        The 1-minute horizon is special and only performs carry-forward
        operations without additional feature calculation.
        
        Args:
            carry_forward_map: Dictionary mapping horizon to list of fields
                to carry forward. Fields like 'fittable', 'tradeable', 
                'expandable' are forward-filled from previous timestamps.
        
        Note:
            Boolean feature columns are explicitly cast back to bool type
            after carry-forward operations to maintain type consistency.
        """

        logger.info(f"Calculating features for horizon:{self.horizon}")
        carry_forward_flds = carry_forward_map[self.horizon]

        for fld in self.features_df.columns:
            if fld.startswith("category_"):
                carry_forward_flds.append(fld)

        logger.info(f"Carrying forward {carry_forward_flds} for horizon {self.horizon}")
        self.features_df = carry_forward(self.features_df, carry_forward_flds)
        for fld in carry_forward_flds:
            if fld in BOOL_FEATURE_COLS:
                self.features_df[fld] = self.features_df[fld].astype(bool)

        if self.horizon == 1:
            return

        if self.horizon == self.reopt_interval:
            self.features_df = self.calcs.calculate_volume_forecast(self.features_df, horizon=self.horizon)

        cx_features = extract_tree_features(self.config, [self.horizon])
        cx_features = [feature for feature in cx_features if feature.startswith("cx.")]
        self.features_df = self.features_df.drop(columns=cx_features, errors='ignore')
        self.features_df = make_cx_features(self.features_df, cx_features)

        self.features_df['fittable'] = True
        self.features_df = self.features.run_frequency(features_df=self.features_df, start_dt=self.last_processed_ts)
        self.features_df['fittable'] = self.features_df['fittable'].astype(bool)

    def calculate_and_lag_models(self) -> None:
        """Calculate model signals and generate lagged features.
        
        Convenience method that runs both calculate_models() and lag_models()
        in sequence. This is the typical workflow for updating models when
        new data arrives.
        """
        self.calculate_models()
        self.lag_models()

    def calculate_models(self):
        """Calculate model signals for new data points.
        
        Processes only data newer than last_processed_ts to calculate alpha
        model signals. For each configured model at this horizon:
        - Skips models with zero weight
        - Calculates the model signal using the Models class
        - Creates an absolute value version of the signal
        - Updates the features_df with the new signals
        
        The models are calculated based on the configuration in 
        config['FCASTS'][horizon]['models'], where each model specifies:
        - name: Model name (e.g., 'hl', 'c2vwap', 'slz')
        - weight: Model weight in final alpha (0 weight models are skipped)
        - lags: Number of lags to generate for autoregressive features
        
        Raises:
            ValueError: If a model calculation results in all null values,
                indicating a potential configuration or data issue.
        """
        recent_idx = self.features_df.index.get_level_values('ts') > self.last_processed_ts
        latest_models_df = self.features_df[recent_idx]
        min_ts, max_ts = get_min_max_ts(latest_models_df)
        logger.info(f"Calculating models from {min_ts} - {max_ts}")

        latest_models_df = latest_models_df.reset_index()
        for forecast in self.config['FCASTS'][str(self.horizon)]['models']:
            name = forecast['name']
            weight = float(forecast['weight'])
            if weight == 0:
                logger.info(f"Skipping zero weight model {name} {self.horizon}")
                continue
            lags = int(forecast['lags'])
            model_fld = f'{name}_{self.horizon}'
            lagged_model_fld = f'{model_fld}_L0'

            logger.info(f"Computing model {model_fld}:{lags}")
            latest_models_df = self.models.calculate(latest_models_df, name, self.horizon)
            # XXX dump df here to debug why the model generated from features all null
            if latest_models_df[lagged_model_fld].isnull().values.all():
                raise log_and_raise(f"Could not calculate, all values null {name}", df=latest_models_df)

            abs_fld = f'{lagged_model_fld}_abs'
            latest_models_df[abs_fld] = latest_models_df[lagged_model_fld].abs()
            if abs_fld not in self.features_df.columns:
                self.features_df[abs_fld] = np.nan

        latest_models_df = set_index(latest_models_df)
        self.features_df.loc[recent_idx, latest_models_df.columns] = latest_models_df

    def lag_models(self):
        """Generate lagged versions of model signals for autoregressive features.
        
        For each model configured at this horizon, generates lagged columns
        that will be used as features in the final alpha calculation. This
        allows models to use their own historical values as inputs.
        
        The number of lags is specified per model in the configuration.
        Models with zero weight are skipped since they won't contribute
        to the final alpha anyway.
        
        Updates self.features_df in place with the new lagged columns.
        In production mode (prod=True), only the most recent data is
        processed for efficiency.
        """
        horizon = self.horizon
        features_df = self.features_df
        for forecast in self.config['FCASTS'][str(horizon)]['models']:
            name = forecast['name']
            lags = int(forecast['lags'])
            weight = float(forecast['weight'])
            if weight == 0:
                logger.info(f"Not generating model for zero weight {name}")
                continue
            model_fld = f"{name}_{horizon}"
            logger.info(f"Computing model lags {model_fld}:{lags} {weight=}")
            features_df, _ = generate_model_lags(df=features_df, model_name=model_fld, horizon=horizon, lags=lags, prod=True)
        self.features_df = features_df
