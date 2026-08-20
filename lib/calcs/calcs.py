# pylint: disable=too-many-lines,unused-argument
"""Feature calculation module for statistical arbitrage trading system.

This module provides comprehensive feature engineering capabilities for the trading system,
implementing calculations for technical indicators, market microstructure features, risk
metrics, and cross-sectional statistics. It handles:

1. **Return Calculations**: Log returns, residualized returns, funding-adjusted returns
2. **Risk Metrics**: Volatility, beta, semi-variance, PCA residuals
3. **Market Microstructure**: Bid-ask spreads, trade sizes, order book imbalances
4. **Technical Indicators**: RSI, trailing statistics, volume patterns
5. **Cross-Sectional Features**: Z-scores, rankings, market-relative metrics
6. **Time-Based Features**: Hour of day, day of week, volume buckets

The module is designed for both batch processing (historical data) and real-time
incremental updates in production. Key design principles:
- Efficient rolling window calculations
- Proper handling of multi-horizon features
- Careful treatment of lookahead bias
- Memory-efficient sparse calculations for production

Example:
    >>> config = load_config()
    >>> calcs = Calcs(config, frequency=60, prod=False)
    >>> df = load_bar_data()
    >>> 
    >>> # Calculate returns and market-adjusted returns
    >>> df = calcs.calc_returns(df, start_dt=datetime(2023, 1, 1))
    >>> 
    >>> # Add technical features
    >>> df = calcs.calculate_trailing_mean(df, feature='logret')
    >>> df = calcs.calculate_rsi(df)
    >>> 
    >>> # Add risk metrics
    >>> df = calcs.calculate_beta(df)
    >>> df = calcs.calculate_risk(df, risk_frequency=1440, sample_frequency=60)
"""

import logging
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ..data import dump_market_returns
from lib.util.config import extract_max_lags, extract_reopt_times, extract_feature_lookback_periods
from lib.util.dataframes import get_df_idx
from lib.util.logging_util import KeyLogger
from lib.util.time_util import to_datetime
from lib.util.util import unique_list
from .calc_filters_and_bounds import calculate_fittable_filter, calculate_tradeable_filter, calculate_expandable_filter, calculate_exclusions_and_bounds, calculate_price_filter
from .calc_ob import calculate_spread_factor, calculate_bid_ask_imbalance, calculate_trade_size, calculate_update_size
from .calc_returns import (calc_logret, calc_vwap, calc_funding_adjusted_logret, calculate_resid_return,
                              calculate_weighted_mkt_return, calculate_rsi, calculate_weighted_resid_return)
from .calc_extreme import calculate_extreme_metrics
from .calc_risk import calculate_trailing_semi_variance, calculate_risk_factor, calculate_pca, calculate_risk, calculate_beta
from .calc_util import calculate_trailing_z, calculate_abs_factor, calculate_time_factors, calculate_delta, calculate_weighted_feature, calculate_trailing_mean, calculate_minmax, calculate_quintiles
from .calc_volume import calculate_advp, calculate_mdvp, calculate_volume_forecast, calculate_volume_buckets, calculate_mktcap_turnover

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

pd.options.display.max_rows = 1500

MIN_BETA_POINTS = 30
BETA_RET_BOUND = .1
MAX_BETA = 10
NEWS_DECAY_MINS = 360
DEFAULT_LONG_TERM_NEWS_LOOKBACK_PERIODS = 30
DEFAULT_MID_TERM_NEWS_LOOKBACK_PERIODS = 7
DEFAULT_SHORT_TERM_NEWS_LOOKBACK_PERIODS = 3
MIN_NEWS_DT = to_datetime("20240405")


class Calcs:
    """Feature calculation engine for market data processing.
    
    This class handles all feature engineering calculations including returns,
    market adjustments, technical indicators, microstructure features, and risk metrics.
    It processes bar data to generate features used in alpha models and trading decisions.
    
    Attributes:
        config: Configuration dictionary with all system parameters
        prod: Whether running in production mode (affects optimization)
        frequency: Time frequency in minutes for calculations (15, 60, 1440, etc)
        new_cols: List tracking newly created columns
        horizon_to_max_lags: Maximum lags for each time horizon
    """

    def __init__(self, config: dict, frequency: Optional[int] = None, prod: bool = False):
        """Initialize the Calcs feature calculation engine.
        
        Args:
            config: System configuration dictionary containing:
                - MIN_ADVP_* thresholds for liquidity filtering
                - Portfolio and risk parameters
                - Feature calculation lookback periods
                - Optimization settings
            frequency: Time frequency in minutes for calculations
            prod: Whether running in production mode
        """
        self.config = config
        self.prod = prod
        self.frequency = frequency
        self.new_cols = []
        self.horizon_to_max_lags = extract_max_lags(config)

        self.min_advp_priceable = self.config['MIN_ADVP_PRICEABLE']

        self.min_advp_fittable = self.config['MIN_ADVP_FITTABLE']
        self.min_advp_tradeable = self.config['MIN_ADVP_TRADEABLE']
        self.min_advp_expandable = self.config['MIN_ADVP_EXPANDABLE']
        self.max_market_cap_expandable_frac = self.config['MAX_MARKET_CAP_EXPANDABLE_FRAC']
        self.max_portfolio_notional = self.config['MAX_PORTFOLIO_NOTIONAL']
        self.max_volume_fraction = self.config['MAX_POSITION_VOLUME_FRACTION']
        self.feature_lookback_periods = extract_feature_lookback_periods(self.config, self.frequency)

        self.max_move_filter = self.config['MAX_MOVE_FILTER']
        self.adv_lookback_days = self.config['ADV_LOOKBACK_DAYS']
        self.max_position_pct = self.config['MAX_POSITION_PCT']
        self.exclude_nonalpha_trades = self.config['EXCLUDE_NON_ALPHA_TRADES']
        self.opt_horizon = self.config['OPT_HORIZON']
        self.risk_fld = self.config['RISK_FLD']
        self.filter_delisting = self.config['FILTER_DELISTING']
        self.delisting_buffer_days = self.config['DELISTING_BUFFER_DAYS']
        self.feature_sigma_bound = self.config['FEATURE_SIGMA_BOUND']
        self.position_bounds = self.config['MAX_POSITION_BOUNDS']
        self.max_dvol_sigma = self.config['MAX_DVOL_SIGMA']
        self.min_funding_rate = self.config['MIN_FUNDING_RATE']
        self.vol_bound_coeff = self.config.get('VOL_BOUND_COEFF', 0.0)
        self.vol_bound_floor = self.config.get('VOL_BOUND_FLOOR', 0.4)

        self.long_term_news_lookback_periods = self.config.get('LONG_TERM_NEWS_LOOKBACK_PERIODS', DEFAULT_LONG_TERM_NEWS_LOOKBACK_PERIODS)
        self.mid_term_news_lookback_periods = self.config.get('MID_TERM_NEWS_LOOKBACK_PERIODS', DEFAULT_MID_TERM_NEWS_LOOKBACK_PERIODS)
        self.short_term_news_lookback_periods = self.config.get('SHORT_TERM_NEWS_LOOKBACK_PERIODS', DEFAULT_SHORT_TERM_NEWS_LOOKBACK_PERIODS)

        self.old_position_days = self.config["OLD_POSITION_DAYS"]
        self.old_position_risk_mult = self.config["OLD_POSITION_RISK_MULT"]

        # reopt times for the past 24 hours
        self.reopt_times = extract_reopt_times(self.config, dt.now(timezone.utc) - td(days=1))

    def get_new_cols(self) -> List[str]:
        """Get list of newly created column names.
        
        Returns:
            List of unique column names created during feature calculation
        """
        return unique_list(self.new_cols)

    def get_lagged_lookback_minutes(self) -> int:
        """Calculate total lookback period needed for lagged features.
        
        Returns:
            Total minutes of lookback required based on maximum lags and frequency
        """
        return self.horizon_to_max_lags[self.frequency] * self.frequency

    def calc_vwap(self, df: pd.DataFrame, horizon: Optional[int] = None, vwap_fld: str = "vwap", dvolume_fld: str = "dvolume", volume_fld: str = "volume") -> pd.DataFrame:
        """Calculate volume-weighted average price (VWAP).
        
        Args:
            df: Input dataframe with volume and dollar volume columns
            horizon: Optional time horizon for rolling VWAP calculation
            vwap_fld: Base name for VWAP column
            dvolume_fld: Base name for dollar volume column
            volume_fld: Base name for volume column
        
        Returns:
            DataFrame with added VWAP column
        
        Notes:
            VWAP = Dollar Volume / Volume
        """
        df, new_cols = calc_vwap(df, horizon, vwap_fld, dvolume_fld, volume_fld)
        self.new_cols += new_cols
        return df

    def calc_logret(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns from close prices.
        
        Computes log(close_t / close_{t-1}) for each symbol. Handles missing data
        by forward-filling prices and validates bar continuity.
        
        Args:
            df: Input dataframe with 'close_mid' prices
        
        Returns:
            DataFrame with added 'logret' column
        
        Notes:
            - Checks that open_mid equals previous close_mid for bar validation
            - Only calculates returns for bars with update_cnt > 0
        """
        df, new_cols = calc_logret(df)
        self.new_cols += new_cols
        return df

    def calc_funding_adjusted_logret(self, df: pd.DataFrame, keep_realized_funding: bool = False):
        """Calculate funding rate adjusted log returns.
        
        Adjusts log returns by subtracting realized funding rates at funding times
        (typically every 8 hours). This gives the true P&L from holding futures.
        
        Args:
            df: Input dataframe with 'logret' and funding rate columns
            keep_realized_funding: Whether to keep the realized funding column
        
        Returns:
            DataFrame with added 'logret_funding_adj' column
        
        Notes:
            - Funding is only realized when ts equals next_funding_time
            - Funding adjustment = log(1 + funding_rate)
        """
        df, new_cols = calc_funding_adjusted_logret(df, keep_realized_funding)
        self.new_cols += new_cols
        return df

    def calculate_resid_return(self, df: pd.DataFrame, filter_col: str, ret_col: str = 'logret', market_ret_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate equal-weighted market-residualized returns.

        Computes returns minus the equal-weighted average return of all fittable
        instruments, removing market-wide movements.

        Args:
            df: Input dataframe with return column
            ret_col: Name of return column to residualize
            market_ret_df: Optional pre-calculated market returns

        Returns:
            DataFrame with added residual return column
        """
        df, new_cols = calculate_resid_return(df, filter_col=filter_col, ret_col=ret_col, market_ret_df=market_ret_df)
        self.new_cols += new_cols
        return df

    def calculate_weighted_mkt_return(self, df: pd.DataFrame, ret_col: str = 'logret', frequency: Optional[int] = None) -> pd.DataFrame:
        """Calculate volume-weighted market return.

        Wrapper around calculate_weighted_mkt_return function.

        Args:
            df: Input dataframe with returns and volume
            ret_col: Return column to weight
            frequency: Optional frequency for column naming

        Returns:
            DataFrame with weighted market return column
        """
        df, mkt_col = calculate_weighted_mkt_return(df=df, filter_col='fittable', ret_col=ret_col, frequency=frequency)
        self.new_cols.append(mkt_col)
        return df

    def calculate_weighted_resid_return(self, df: pd.DataFrame, filter_col: str, ret_col: str = 'logret', market_ret_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate volume-weighted market-residualized returns.

        Computes returns minus the volume-weighted market return, where weights
        are based on log(ADVP) as a market cap proxy.

        Args:
            df: Input dataframe with return column
            ret_col: Name of return column to residualize
            market_ret_df: Optional pre-calculated weighted market returns

        Returns:
            DataFrame with added weighted residual return column
        """
        df, new_cols = calculate_weighted_resid_return(df, filter_col=filter_col, ret_col=ret_col, market_ret_df=market_ret_df)
        self.new_cols += new_cols
        return df

    def calc_returns(
            self,
            df: pd.DataFrame,
            start_dt: Optional[dt] = None,
            market_ret_df: Optional[pd.DataFrame] = None,
            chunk_start_dt: Optional[dt] = None,
            mkt_ret_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calculate all return-based features.
        
        Main entry point for return calculations including log returns, market
        residualized returns, funding adjustments, and VWAP.
        
        Args:
            df: Input dataframe with price and volume data
            start_dt: Optional start datetime for calculations
            market_ret_df: Pre-calculated market returns (for efficiency)
            chunk_start_dt: Start of current processing chunk
            mkt_ret_dir: Directory to save market return files
        
        Returns:
            DataFrame with all return features added
        
        Notes:
            Calculates:
            - Basic log returns
            - Equal and value-weighted market residuals
            - Funding adjusted returns and residuals
            - VWAP
        """
        df = self.calc_logret(df)

        if 'priceable' not in df.columns:
            df = self.calculate_price_filter(df=df, frequency=self.frequency, start_dt=start_dt)
        else:
            logger.info("Using existing priceable filter for calculating resid returns")

        df = self.calculate_resid_return(df, filter_col='priceable', market_ret_df=market_ret_df)
        df = self.calculate_weighted_resid_return(df, filter_col='priceable', market_ret_df=market_ret_df)
        df = self.calc_funding_adjusted_logret(df)
        df = self.calculate_resid_return(df, filter_col='priceable', ret_col='logret_funding_adj', market_ret_df=market_ret_df)
        df = self.calculate_weighted_resid_return(df, filter_col='priceable', ret_col='logret_funding_adj', market_ret_df=market_ret_df)

        mkt_ret_cols = ['logret_mkt', 'logret_wgtmkt', 'logret_funding_adj_mkt', 'logret_funding_adj_wgtmkt']
        if mkt_ret_dir is not None:
            logger.info(f"Dumping {mkt_ret_cols=} to {mkt_ret_dir=} since {chunk_start_dt=}")
            mkt_ret_df = df[mkt_ret_cols].reset_index().drop_duplicates('ts').set_index('ts').drop('symbol_venue', axis=1)
            mkt_ret_df = mkt_ret_df.loc[mkt_ret_df.index > chunk_start_dt]
            dump_market_returns(mkt_ret_df, mkt_ret_dir=mkt_ret_dir, start_date=start_dt.date(), end_date=chunk_start_dt.date())
        df = df.drop(columns=mkt_ret_cols, errors='ignore')

        df = self.calc_vwap(df)
        return df

    def calculate_volume_forecast(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast expected volume over a horizon.

        Estimates future volume based on time-of-day patterns and recent activity.

        Args:
            df: Input dataframe with volume bucket data
            horizon: Forecast horizon in minutes

        Returns:
            DataFrame with volume forecast column
        """
        df, new_cols = calculate_volume_forecast(df, horizon)
        self.new_cols += new_cols
        return df

    def calculate_volume_buckets(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            lookback_days: Optional[int] = None,
            ffill_na: bool = False,
    ) -> pd.DataFrame:
        """Calculate median volume by time of day buckets.
        
        Computes rolling median dollar volume for each time period (e.g., 9:30am)
        to capture intraday volume patterns. Used for volume normalization.
        
        Args:
            df: Input dataframe with dollar volume data
            feature: Unused, kept for API consistency
            start_dt: Start datetime for calculation
            lookback_periods: Unused, uses lookback_days instead
            lookback_days: Days of history for median calculation (default: 45)
            ffill_na: Unused, kept for API consistency
        
        Returns:
            DataFrame with added median time bucket volume column
        
        Notes:
            - Creates time-of-day buckets at frequency granularity
            - Forward fills missing values
            - Handles midnight rollover correctly
        """

        df, new_cols = calculate_volume_buckets(
            df=df,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_days=self.adv_lookback_days if lookback_days is None else lookback_days
        )
        self.new_cols += new_cols
        return df

    def calculate_advp(self, df: pd.DataFrame, lookback_days: int, allow_short_history: bool = False) -> pd.DataFrame:
        """Calculate average daily volume in USD (ADVP).
        
        Computes rolling average of daily dollar volume, accounting for missing
        data by dividing total volume by actual trading days.
        
        Args:
            df: Input dataframe with 'dvolume' column
            lookback_days: Number of days for rolling average
        
        Returns:
            DataFrame with added 'advp' column
        
        Notes:
            - Handles missing data by counting actual trading days
            - Critical metric for liquidity filtering
        """
        df, new_cols = calculate_advp(df, lookback_days, allow_short_history)
        self.new_cols += new_cols
        return df

    def calculate_mdvp(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            ffill_na: bool = False,
            lookback_days: Optional[int] = None,
            allow_short_history: bool = False
    ) -> pd.DataFrame:
        """Calculate median daily volume in USD (MDVP).

        Computes rolling median of daily dollar volume, more robust to
        volume spikes than the mean-based ADVP.

        Args:
            df: Input dataframe with 'dvolume' column
            feature: Unused, kept for feature pipeline API consistency
            start_dt: Unused, kept for feature pipeline API consistency
            lookback_periods: Unused, kept for feature pipeline API consistency
            ffill_na: Unused, kept for feature pipeline API consistency
            lookback_days: Number of days for rolling median window.
                Defaults to ADV_LOOKBACK_DAYS from config.
            allow_short_history: If True, skip length check

        Returns:
            DataFrame with added 'mdvp' column
        """
        lookback_days = self.adv_lookback_days if lookback_days is None else lookback_days
        assert self.frequency == 1440, f"mdvp only supports frequency=1440, got {self.frequency}"
        df, new_cols = calculate_mdvp(df, lookback_days, allow_short_history)
        self.new_cols += new_cols
        return df

    def calculate_mktcap_turnover(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            ffill_na: bool = False
    ) -> pd.DataFrame:
        """Calculate market cap turnover ratio.

        Computes the ratio of trailing mean dollar volume to market cap,
        representing how quickly the market cap turns over through trading.

        Args:
            df: Input dataframe with dvolume_{frequency}_trmean and market_cap columns
            feature: Unused, kept for API consistency
            start_dt: Unused, kept for API consistency
            lookback_periods: Unused, kept for API consistency
            ffill_na: Unused, kept for API consistency

        Returns:
            DataFrame with mktcap_turnover column

        Notes:
            - Higher values indicate more liquid/actively traded assets
            - Useful for identifying trading activity relative to size
        """
        df, new_cols = calculate_mktcap_turnover(
            df=df,
            frequency=self.frequency,
            feature_sigma_bound=self.feature_sigma_bound
        )
        self.new_cols += new_cols
        return df

    def calculate_beta(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate rolling market beta for all symbols.

        Computes beta coefficients by regressing funding-adjusted returns against
        volume-weighted market returns over a rolling window.

        Args:
            df: Input dataframe with return data
            feature: Unused, kept for API consistency
            start_dt: Start datetime for beta calculation
            lookback_periods: Number of periods for rolling window (default: 90)
            ffill_na: Unused, kept for API consistency

        Returns:
            DataFrame with added beta and beta t-statistic columns

        Notes:
            - Uses funding-adjusted returns for accurate beta estimation
            - Forward fills beta values between calculation points
        """

        df, new_cols = calculate_beta(
            df=df,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            beta_lookback_periods=self.feature_lookback_periods
        )
        self.new_cols += new_cols
        return df

    def calculate_trailing_z(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Calculate trailing z-score (standardized value).
        
        Standardizes feature by subtracting trailing mean and dividing by trailing
        standard deviation. Clips extreme values for stability.
        
        Args:
            df: Input dataframe with feature and its trailing statistics
            feature: Base feature name (expects {feature}_trmean and {feature}_trstd)
        
        Returns:
            DataFrame with added {feature}_lz column
        
        Notes:
            - Z-scores are clipped to [-5, 5] by default
            - Missing values filled with 0
        """
        df, new_cols = calculate_trailing_z(df, feature, self.feature_sigma_bound)
        self.new_cols += new_cols
        return df

    def calculate_minmax(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate rolling minimum and maximum values.

        Tracks extreme values over a lookback window, useful for identifying
        support/resistance levels and mean reversion opportunities.

        Args:
            df: Input dataframe with feature data
            feature: Feature name to calculate min/max for
            start_dt: Start datetime for calculation
            lookback_periods: Number of periods for rolling window
            ffill_na: Whether to forward fill missing values

        Returns:
            DataFrame with min/max columns
        """
        df, new_cols = calculate_minmax(
            df=df,
            feature=feature,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None
        )
        self.new_cols += new_cols
        return df

    def calculate_time_factors(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Add time-based features for seasonality modeling.

        Extracts hour of day and day of week features to capture intraday
        and weekly patterns in trading behavior.

        Args:
            df: Input dataframe with timestamp index
            feature: Unused
            start_dt: Unused
            lookback_periods: Unused
            ffill_na: Unused

        Returns:
            DataFrame with hour_of_day and day_of_week columns
        """
        df, new_cols = calculate_time_factors(df, get_df_idx(df, start_dt))
        self.new_cols += new_cols
        return df

    def calculate_abs_factor(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate absolute value of standardized returns.

        Creates volatility-like features from absolute z-scores, capturing
        magnitude of moves regardless of direction.

        Args:
            df: Input dataframe with z-score columns
            feature: Feature prefix for abs calculation
            start_dt: Unused
            lookback_periods: Unused
            ffill_na: Unused

        Returns:
            DataFrame with absolute z-score columns
        """
        df, new_cols = calculate_abs_factor(df, feature, get_df_idx(df, start_dt))
        self.new_cols += new_cols
        return df

    def calculate_weighted_feature(self, df: pd.DataFrame, feature: str, weight: str) -> pd.DataFrame:
        """Calculate weighted version of a feature.

        Creates market-wide weighted features using specified weights, typically
        for constructing custom market indices.

        Args:
            df: Input dataframe with feature and weight columns
            feature: Feature column to weight
            weight: Weight column name

        Returns:
            DataFrame with weighted feature column
        """
        df, new_cols = calculate_weighted_feature(df, feature, weight)
        self.new_cols += new_cols
        return df

    def calculate_delta(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate absolute and percentage changes for a feature.

        Computes the difference between current and previous values (lagged by frequency).
        Creates two new columns:
        - {feature}_{frequency}_d: Absolute change
        - {feature}_{frequency}_dp: Percentage change

        Args:
            df: Input dataframe with feature data
            feature: Base feature name to calculate deltas for
            start_dt: Optional start datetime for calculations
            lookback_periods: Not used in this function (kept for consistency)
            ffill_na: Not used in this function (kept for consistency)

        Returns:
            DataFrame with delta columns added

        Example:
            >>> calcs.calculate_delta(df, feature='dvolume')
            # Creates dvolume_60_d and dvolume_60_dp columns
        """
        df, new_cols = calculate_delta(df, feature, self.frequency, start_dt)
        self.new_cols += new_cols
        return df

    def calculate_trailing_mean(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            resample_frequency: Optional[int] = None,
            ffill_na: bool = False,
    ) -> pd.DataFrame:
        """Calculate trailing mean and standard deviation for a feature.
        
        Computes rolling statistics over a lookback window, essential for
        z-score normalization and outlier detection.
        
        Args:
            df: Input dataframe with feature data
            feature: Feature name to calculate statistics for
            start_dt: Start datetime for calculation
            lookback_periods: Number of periods for rolling window (default: 90)
            resample_frequency: Frequency for resampling (default: self.frequency)
            ffill_na: Whether to forward fill missing values
        
        Returns:
            DataFrame with added trailing mean and std columns
        
        Notes:
            - Creates {feature}_trmean and {feature}_trstd columns
            - In production mode, only calculates at specific timestamps
            - Handles both bar-level and resampled calculations
        """
        df, new_cols = calculate_trailing_mean(
            df=df,
            feature=feature,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            resample_frequency=resample_frequency,
            ffill_na=ffill_na,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None,
            feature_sigma_bound=self.feature_sigma_bound
        )
        self.new_cols += new_cols
        return df

    def calculate_trailing_semi_variance(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            resample_frequency: Optional[int] = None,
            ffill_na: bool = False,
    ) -> pd.DataFrame:
        """Calculate trailing upside and downside semi-variance.
        
        Computes separate volatility measures for positive and negative deviations
        from the mean, useful for asymmetric risk assessment.
        
        Args:
            df: Input dataframe with feature data
            feature: Feature name to calculate semi-variance for
            start_dt: Start datetime for calculation
            lookback_periods: Number of periods for rolling window
            resample_frequency: Frequency for resampling
            ffill_na: Whether to forward fill missing values
        
        Returns:
            DataFrame with upside/downside std and ratio columns
        
        Notes:
            - Creates {feature}_trstd_u (upside) and {feature}_trstd_d (downside)
            - Ratio indicates skewness of return distribution
        """
        df, new_cols = calculate_trailing_semi_variance(
            df=df,
            feature=feature,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            resample_frequency=resample_frequency,
            ffill_na=ffill_na,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None
        )
        self.new_cols += new_cols
        return df

    def calculate_extreme_metrics(
            self,
            df: pd.DataFrame,
            feature: Optional[str] = None,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            resample_frequency: Optional[int] = None,
            ffill_na: bool = False,
    ) -> pd.DataFrame:
        """Calculate extreme/spikiness metrics (RJ, exceedance, CSK).

        Wrapper around calc_extreme.calculate_extreme_metrics() that supplies
        instance-level config (frequency, lookback, prod settings).

        Args:
            df: Input dataframe with feature data
            feature: Feature name to compute metrics on
            start_dt: Start datetime for calculation
            lookback_periods: Number of periods for rolling window
            resample_frequency: Frequency for resampling
            ffill_na: Whether to forward fill missing values

        Returns:
            DataFrame with _rj, _exceedance, _csk columns added
        """
        df, new_cols = calculate_extreme_metrics(
            df=df,
            feature=feature,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            resample_frequency=resample_frequency,
            ffill_na=ffill_na,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None
        )
        self.new_cols += new_cols
        return df

    def calculate_update_size(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate relative order book update frequency.

        Normalizes update count by volume to measure market activity intensity
        independent of trading volume.

        Args:
            df: Input dataframe with update_cnt and volume columns
            feature: Unused, kept for API consistency
            start_dt: Start datetime for calculation
            lookback_periods: Unused
            ffill_na: Whether to forward fill missing values
        
        Returns:
            DataFrame with relative_updates column
        """
        df, new_cols = calculate_update_size(
            df=df,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None,
            feature_sigma_bound=self.feature_sigma_bound
        )
        self.new_cols += new_cols
        return df

    def calculate_trade_size(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate average trade size.
        
        Computes volume per trade as an indicator of institutional vs retail
        trading activity.
        
        Args:
            df: Input dataframe with volume and trade_cnt columns
            feature: Unused, kept for API consistency
            start_dt: Start datetime for calculation
            lookback_periods: Unused
            ffill_na: Whether to forward fill missing values
        
        Returns:
            DataFrame with trade_sz column
        """
        df, new_cols = calculate_trade_size(
            df=df,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None,
            feature_sigma_bound=self.feature_sigma_bound
        )
        self.new_cols += new_cols
        return df

    def calculate_spread_factor(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate relative bid-ask spread.

        Normalizes spread by mid price to get a comparable liquidity metric
        across different price levels.

        Args:
            df: Input dataframe with spread_avg column
            feature: Unused
            start_dt: Unused
            lookback_periods: Unused
            ffill_na: Whether to forward fill missing values

        Returns:
            DataFrame with relative_spread column
        """

        df, new_cols = calculate_spread_factor(
            df=df,
            frequency=self.frequency,
            lookback_periods=self.feature_lookback_periods if lookback_periods is None else lookback_periods,
            start_dt=start_dt,
            ffill_na=ffill_na,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            reopt_times=self.reopt_times if self.prod else None,
            max_lags=self.horizon_to_max_lags[self.frequency] if self.prod else None,
            feature_sigma_bound=self.feature_sigma_bound
        )
        self.new_cols += new_cols
        return df

    def calculate_bid_ask_imbalance(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate bid-ask size imbalance and its changes.
        
        Computes the log ratio of bid size to ask size as a measure of order book
        imbalance. Also calculates the delta (change) in this imbalance.
        
        Creates columns:
        - ba_imbal_{frequency}: Log(bid_size / ask_size)
        - ba_imbal_{frequency}_d: Absolute change in imbalance
        - ba_imbal_{frequency}_dp: Percentage change in imbalance
        
        Args:
            df: Input dataframe with bid_sz_avg and ask_sz_avg columns
            feature: Not used (kept for consistency)
            start_dt: Optional start datetime for calculations
            lookback_periods: Not used (kept for consistency)
            ffill_na: Not used (kept for consistency)
            
        Returns:
            DataFrame with bid-ask imbalance columns added
            
        Note:
            - Positive values indicate more bid pressure (buyers)
            - Negative values indicate more ask pressure (sellers)
            - Values are fillna(0) to handle cases where sizes are 0
        """
        df, new_cols = calculate_bid_ask_imbalance(df, self.frequency, start_dt)
        self.new_cols += new_cols
        return df

    def calculate_rsi(
            self,
            df: pd.DataFrame,
            feature: str,
            start_dt: Optional[dt] = None,
            lookback_periods: Optional[int] = None,
            resample_frequency: Optional[int] = None,
            ffill_na: bool = False,
    ) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI) technical indicator.
        
        RSI is a momentum oscillator that measures the speed and magnitude of price
        changes. Values range from 0 to 100, with traditional overbought/oversold
        levels at 70/30.
        
        The implementation handles multi-day resampling carefully to avoid lookahead
        bias by calculating RSI values day-by-day when resample_frequency > 1440.
        
        Args:
            df: Input dataframe with price/return data
            feature: Base feature to calculate RSI on (default: 'logret')
            start_dt: Optional start datetime for calculations
            lookback_periods: Number of periods for RSI calculation window
            resample_frequency: Frequency in minutes for resampling before RSI calc
            ffill_na: Not used (kept for consistency)
            
        Returns:
            DataFrame with RSI column added (rsi_{frequency})
            
        Note:
            - RSI > 70 traditionally indicates overbought conditions
            - RSI < 30 traditionally indicates oversold conditions
            - Uses rolling mean for average gain/loss calculation
            - Forward fills missing values up to frequency limit
        """
        df, new_cols = calculate_rsi(
            df=df,
            frequency=self.frequency,
            feature=feature,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            resample_frequency=resample_frequency,
            feature_lookback_periods=self.feature_lookback_periods,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes()
        )
        self.new_cols += new_cols
        return df

    def calculate_risk(self, df: pd.DataFrame, risk_frequency: int, sample_frequency: int) -> pd.DataFrame:
        """Calculate risk metrics from trailing volatility.

        Computes annualized risk measure based on return standard deviation,
        properly scaled for the measurement frequency.

        Args:
            df: Input dataframe with return std columns
            risk_frequency: Risk horizon in minutes
            sample_frequency: Data sampling frequency in minutes

        Returns:
            DataFrame with risk column

        Notes:
            - Risk = volatility * sqrt(horizon/year_minutes)
            - Clipped at 1e-6 minimum to avoid division by zero
        """
        df, new_cols = calculate_risk(
            df=df,
            risk_frequency=risk_frequency,
            sample_frequency=sample_frequency,
            risk_fld=self.risk_fld,
            old_position_days=self.old_position_days,
            old_position_risk_mult=self.old_position_risk_mult
        )
        self.new_cols += new_cols
        return df

    def calculate_pca(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate PCA loadings for dimensionality reduction.
        
        Performs Principal Component Analysis to extract market-wide factors
        from returns data. Currently extracts first 3 components.
        
        Args:
            df: Input dataframe with return data
            feature: Return feature for PCA (default: 'logret_funding_adj')
            start_dt: Start datetime for calculation
            lookback_periods: Window for PCA calculation
            ffill_na: Whether to forward fill missing values
        
        Returns:
            DataFrame with PCA loading columns
        
        Notes:
            - PC1 typically represents market factor
            - PC2 often captures sector/style effects
            - PC3 may represent idiosyncratic factors
        """
        df, new_cols = calculate_pca(
            df=df,
            frequency=self.frequency,
            start_dt=start_dt,
            lookback_periods=lookback_periods,
            beta_lookback_periods=self.feature_lookback_periods,
            prod=self.prod,
            eigenvalue_threshold=self.config.get('PCA_EIGENVALUE_THRESHOLD', 0.01)
        )
        self.new_cols += new_cols
        return df

    def calculate_risk_factor(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False, filter_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate risk metrics for position sizing.
        
        Computes volatility-based risk measures scaled by square root of time
        for use in portfolio optimization.
        
        Args:
            df: Input dataframe with volatility columns
            feature: Unused
            start_dt: Unused
            lookback_periods: Unused
            ffill_na: Whether to forward fill missing values
            filter: Column name to use for filtering (default: 'fittable')
        
        Returns:
            DataFrame with risk columns for each horizon
        
        Notes:
            - Risk is annualized volatility scaled by sqrt(horizon/525600)
            - Used as denominator in position sizing
        """
        # Use fittable as default filter for risk calculations
        if filter_col is None:
            filter_col = 'fittable'
        
        df, new_cols = calculate_risk_factor(
            df=df,
            feature=feature,
            frequency=self.frequency,
            start_dt=start_dt,
            prod=self.prod,
            lagged_lookback_minutes=self.get_lagged_lookback_minutes() if self.prod else None,
            feature_sigma_bound=self.feature_sigma_bound,
            filter_col=filter_col
        )
        self.new_cols += new_cols
        return df

    def calc_decayed_news(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Calculate exponentially decayed news impact features.

        Creates features that capture the lingering effect of news events with
        exponential decay over time.

        Args:
            df: Input dataframe with news_event column
            feature: Unused
            start_dt: Start datetime for calculation
            lookback_periods: Decay lookback period
            ffill_na: Whether to forward fill

        Returns:
            DataFrame with decayed news features

        Notes:
            - Creates multiple decay horizons (short, mid, long term)
            - Halflife = horizon/4 for exponential decay
            - Only processes data after MIN_NEWS_DT
        """
        if start_dt < MIN_NEWS_DT:
            logger.info(f"Not calculating news prior to {MIN_NEWS_DT}")
            return df
        halflife = int(self.frequency / 4)
        fld_base = "news_event"
        fld_decayed = f"{fld_base}_decayed_{self.frequency}"
        df[fld_base] = np.float32(0.0)
        df.loc[~df['news_title'].isna(), fld_base] = np.float32(1.0)
        df[fld_decayed] = df[fld_base].unstack().ewm(halflife=halflife).sum().stack(future_stack=True).fillna(0).astype(np.float32)
        logger.info(f"Calculating decayed news with {halflife=} for {self.frequency}")
        if self.frequency in (43200, 10080, 4320, 1440):
            lookback_periods = self.long_term_news_lookback_periods // (self.frequency // 1440)
        elif self.frequency in (720, 360, 120):
            lookback_periods = self.mid_term_news_lookback_periods
        else:
            lookback_periods = self.short_term_news_lookback_periods

        df = self.calculate_trailing_mean(df=df, feature=fld_decayed, start_dt=start_dt, lookback_periods=lookback_periods)

        self.new_cols += [fld_base, fld_decayed]
        return df

    def calculate_quintiles(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Calculate cross-sectional quintile rankings.
        
        Assigns quintile labels (1-5) based on cross-sectional ranking at each
        timestamp, useful for relative value strategies.
        
        Args:
            df: Input dataframe with feature column
            col: Column name to calculate quintiles for
        
        Returns:
            DataFrame with quintile column added as {col}_q
        """
        df, new_cols = calculate_quintiles(df, col)
        self.new_cols += new_cols
        return df

    def calculate_exclusions_and_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading exclusions and position bounds.
        
        Delegates to the imported function from calc_filters_and_bounds.
        
        Args:
            df: Input dataframe with market data and filters
        
        Returns:
            DataFrame with exclusion flags and position bounds
        """
        return calculate_exclusions_and_bounds(
            df=df,
            max_position_pct=self.max_position_pct,
            max_portfolio_notional=self.max_portfolio_notional,
            max_volume_fraction=self.max_volume_fraction,
            exclude_nonalpha_trades=self.exclude_nonalpha_trades,
            max_move_filter=self.max_move_filter,
            delisting_buffer_days=self.delisting_buffer_days,
            position_bounds=self.position_bounds,
            max_dvol_sigma=self.max_dvol_sigma,
            min_funding_rate=self.min_funding_rate,
            vol_bound_coeff=self.vol_bound_coeff,
            vol_bound_floor=self.vol_bound_floor,
        )

    def calculate_fittable_filter(self, df: pd.DataFrame, advp_col: str = 'advp', start_dt: Optional[dt] = None, error_threshold: Optional[float] = None) -> pd.DataFrame:
        """Determine which instruments meet liquidity threshold for model fitting.
        
        Args:
            df: Input dataframe with ADVP data
            advp_col: Column name for average daily volume
            start_dt: Start datetime for filter calculation
        
        Returns:
            DataFrame with 'fittable' boolean column
        
        Notes:
            - Requires ADVP > MIN_ADVP_FITTABLE (typically $15M)
            - Critical for ensuring model quality
        """
        df, new_cols = calculate_fittable_filter(df, self.min_advp_fittable, advp_col, start_dt=start_dt, error_threshold=error_threshold)
        self.new_cols.extend(new_cols)
        return df

    def calculate_tradeable_filter(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Determine which instruments meet liquidity threshold for trading.
        
        Args:
            df: Input dataframe with market data
            feature: Unused
            start_dt: Start datetime for filter
            lookback_periods: Unused
            ffill_na: Unused
        
        Returns:
            DataFrame with 'tradeable' boolean column
        """
        df, new_cols = calculate_tradeable_filter(df, self.min_advp_tradeable, start_dt)
        self.new_cols.extend(new_cols)
        return df

    def calculate_price_filter(self, df: pd.DataFrame, frequency: int, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Determine which instruments meet liquidity threshold for trading.

        Args:
            df: Input dataframe with market data
            feature: Unused
            start_dt: Start datetime for filter
            lookback_periods: Unused
            ffill_na: Unused

        Returns:
            DataFrame with 'tradeable' boolean column
        """

        assert self.frequency is None

        df, new_cols = calculate_price_filter(df=df, min_dvolume=self.min_advp_priceable, frequency=frequency, start_dt=start_dt)
        self.new_cols.extend(new_cols)
        return df

    def calculate_expandable_filter(self, df: pd.DataFrame, feature: Optional[str] = None, start_dt: Optional[dt] = None, lookback_periods: Optional[int] = None, ffill_na: bool = False) -> pd.DataFrame:
        """Determine which instruments have liquidity for position expansion.
        
        Higher liquidity threshold for instruments where we can increase positions
        without significant market impact.
        
        Args:
            df: Input dataframe with market data
            feature: Unused
            start_dt: Start datetime for filter
            lookback_periods: Unused
            ffill_na: Unused
        
        Returns:
            DataFrame with 'expandable' boolean column
        
        Notes:
            - Requires ADVP > MIN_ADVP_EXPANDABLE (typically $55M)
        """
        df, new_cols = calculate_expandable_filter(
            df, self.min_advp_expandable,
            self.max_market_cap_expandable_frac,
            self.filter_delisting,
            self.delisting_buffer_days,
            start_dt
        )
        self.new_cols.extend(new_cols)
        return df
