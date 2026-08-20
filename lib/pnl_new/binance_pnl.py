import logging
from datetime import datetime as dt, timezone
from datetime import timedelta as td
from typing import Optional, List

import numpy as np
import pandas as pd

from lib.calcs.calcs import Calcs
from lib.data import load_binance_fills
from lib.data.dataloader import DataLoader
from lib.data.live_bars import LiveBars
from lib.data.loaders import load_binance_positions, load_funding_income, load_balances
from lib.pnl_new.pnl_util import compute_commissions
from lib.util import DirectoryManager, dir_manager, TRADING_START_DT, set_index, log_and_raise, DF_INDEX, today
from lib.util.util import MKT_SYMBOL, TARDIS_EXCHANGE
from lib.util.dataframes import merge_on_index, make_date, make_symbol, flatten_cols, concat
from lib.util.time_util import date_to_end_dt, date_to_start_dt, date_to_str
from lib.data.data_files import read_parquet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinancePnl:
    """Calculate PnL using Binance fills and position snapshots.

    This class loads Binance fill data and position snapshots to analyze
    trading performance and reconcile with exchange data.

    Attributes:
        start_dt (datetime): Start datetime for PnL calculation
        end_dt (datetime): End datetime for PnL calculation
        fills_df (pd.DataFrame): DataFrame containing Binance fill data
        positions_df (pd.DataFrame): DataFrame containing Binance position snapshots
        funding_df (pd.DataFrame): DataFrame containing Binance funding income data

    Args:
        start: Start datetime for PnL calculation
        end: End datetime for PnL calculation
        pnl_dir_manager: Directory manager for data access
    """

    def __init__(
            self,
            config: dict,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
            pnl_dir_manager: DirectoryManager = dir_manager
    ):
        self.config = config
        self.start_dt = start if start is not None else TRADING_START_DT
        self.end_dt = end if end is not None else dt.now(timezone.utc)
        assert self.start_dt <= self.end_dt

        self.start_date = self.start_dt.date()
        self.end_date = (self.end_dt - td(minutes=1)).date()

        self.dir_manager = pnl_dir_manager
        self.data_loader = DataLoader(config=self.config, data_loader_dir_manager=self.dir_manager)
        self.live_bars = LiveBars(live_bar_dir=self.dir_manager.LIVE_DATA_DIR, use_new=True)
        self.calcs = Calcs(config=self.config, prod=False)

        logger.info(f"Creating BinancePnl from {self.start_dt} to {self.end_dt}")

        self.balances_df = None
        self.live_bars_df = None
        self.prebars_df = None
        self.fills_df = None
        self.positions_df = None
        self.funding_df = None

        self.security_ts_pnl_df: Optional[pd.DataFrame] = None
        self.security_date_pnl_df: Optional[pd.DataFrame] = None
        self.portfolio_ts_pnl_df: Optional[pd.DataFrame] = None
        self.portfolio_date_pnl_df: Optional[pd.DataFrame] = None

        self.load_data()
        self.merge_data()

    def load_data(self):
        """Load all Binance data: fills, positions, and funding."""
        self.live_bars_df = self._load_live_bars()
        self.prebars_df = self._load_prebars()
        self.market_return_df = self._load_market_return()
        self.fills_df = self._load_binance_fills()
        self.positions_df = self._load_binance_positions()
        self.funding_df = self._load_binance_funding()
        self.balances_df = self._load_balances()

    def _load_prebars(self) -> Optional[pd.DataFrame]:
        prebars_df = self.data_loader.load_prebar_files(
            start_date=self.start_date,
            end_date=self.end_date,
            bars_type='tardis'
        )
        if prebars_df is None:
            return None
        prebars_df = prebars_df[['close_mid']]
        return prebars_df

    def _load_market_return(self) -> Optional[pd.DataFrame]:
        """Load pre-calculated market return data from market index parquet files.

        Returns:
            DataFrame indexed by ts with logret_cum_wgtmkt column, or None if no files found
        """
        mkt_ret_dir = f"{self.dir_manager.BAR_DIR}/1/{TARDIS_EXCHANGE}"
        dfs = []
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = date_to_str(current_date)
            filename = f"{mkt_ret_dir}/{date_str}/bars.1.{TARDIS_EXCHANGE}.{date_str}.{MKT_SYMBOL}.parquet"
            try:
                df = read_parquet(filename)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Market return file not found: {filename}")
            current_date += td(days=1)

        if not dfs:
            return None

        mkt_df = concat(dfs, fast=True)
        mkt_df = mkt_df[~mkt_df.index.duplicated(keep='first')]

        # Calculate cumulative market return from per-minute log returns
        mkt_df['logret_cum_wgtmkt'] = mkt_df['logret_wgtmkt'].cumsum()
        logger.info(f"Loaded market return data: {len(mkt_df)} rows")

        return mkt_df

    def _load_live_bars(self) -> Optional[pd.DataFrame]:
        if self.start_dt < date_to_start_dt(today()):
            logger.info(f"Start date prior to today... not getting live bars")
            return None
        live_bars_df = self.live_bars.load_live_bars(start_dt=date_to_start_dt(today()))
        live_bars_df = live_bars_df[['dvolume', 'update_cnt', 'close_mid', 'index_price']]
        live_bars_df = self.calcs.calc_logret(live_bars_df)
        del live_bars_df['update_cnt']

        for fld in ['logret', 'dvolume']:
            live_bars_df[f'{fld}_cum'] = live_bars_df[[fld]].unstack().fillna(0).cumsum().stack(future_stack=True)

        ts_cnt = len(live_bars_df.index.get_level_values('ts').unique())
        live_bars_df['advp'] = live_bars_df['dvolume_cum'] * (1440 / ts_cnt)
        live_bars_df['fittable'] = True

        if 'advp' not in live_bars_df.columns or live_bars_df['advp'].isna().all() or (live_bars_df['advp'] <= 0).all():
            logger.warning("No valid advp values for market return calculation")
        else:
            live_bars_df['advp'] = live_bars_df['advp'].clip(lower=1e3)  # prevent log(0)=-inf in weight calc
            live_bars_df = self.calcs.calculate_weighted_mkt_return(live_bars_df, ret_col='logret')
            if 'logret_wgtmkt' in live_bars_df.columns:
                live_bars_df['logret_cum_wgtmkt'] = live_bars_df[['logret_wgtmkt']].unstack().fillna(0).cumsum().stack(future_stack=True)

        return live_bars_df

    def _load_balances(self) -> Optional[pd.DataFrame]:
        """Load Binance balances for the specified time period."""
        balances_df = load_balances(start_date=self.start_date, end_date=self.end_date, balances_dir=self.dir_manager.BALANCES_DIR)
        balances_df = make_date(balances_df)
        return balances_df

    def _load_binance_fills(self) -> Optional[pd.DataFrame]:
        """Load Binance fill data for the specified time period.

        Returns:
            DataFrame with Binance fill data, or None if not found
        """

        logger.info(f"Loading Binance fills from {self.start_date} to {self.end_date}")
        fills_df = load_binance_fills(
            start_date=self.start_date,
            end_date=self.end_date,
            fills_dir=self.dir_manager.BINANCE_FILLS_DIR
        )
        if fills_df is None:
            logger.warning(f"No Binance fill data found for {self.start_date} to {self.end_date}")
            return None

        fills_df = fills_df.loc[fills_df['ts'] <= self.end_dt]
        fills_df['fill_dollars_buy'] = fills_df['fill_dollars'].clip(lower=0).fillna(0.0)
        fills_df['fill_dollars_sell'] = fills_df['fill_dollars'].clip(upper=0).abs().fillna(0.0)
        logger.info(f"Loaded {len(fills_df)} Binance fills for {fills_df['symbol_venue'].nunique()} symbols")

        if self.live_bars_df is None:
            bars_df = self.prebars_df
        elif self.prebars_df is None:
            bars_df = self.live_bars_df
        else:
            bars_df = pd.concat([self.prebars_df[['close_mid']], self.live_bars_df[['close_mid']]]) if self.prebars_df is not None else self.live_bars_df

        fills_df = compute_commissions(fills_df, bars_df)
        return fills_df

    def _load_binance_positions(self) -> Optional[pd.DataFrame]:
        """Load Binance position snapshots for the specified time period.

        Loads positions starting from the previous day to enable proper delta calculation
        for the first record of the analysis period.

        Returns:
            DataFrame with Binance position data including symbol, quantity, cost basis,
            unrealized PnL, and daily unrealized PnL, or None if not found
        """
        logger.info(f"Loading Binance positions from {self.start_date} to {self.end_date}")
        positions_df = load_binance_positions(
            start_date=self.start_date,
            end_date=self.end_date,
            pos_dir=self.dir_manager.BINANCE_POSITION_DIR
        )
        if positions_df is None:
            logger.warning(f"No Binance position data found for {self.start_date} to {self.end_date}")
            return None

        # Drop columns not needed for PnL analysis
        positions_df = positions_df.drop(columns=['entry_price', 'max_notional_value', 'leverage'], errors='ignore')

        logger.info(f"Loaded {len(positions_df)} Binance position snapshots for {positions_df['symbol'].nunique()} symbols")
        return positions_df

    def _load_binance_funding(self) -> Optional[pd.DataFrame]:
        """Load Binance funding income data for the specified time period.

        Returns:
            DataFrame with Binance funding income data including timestamp, symbol,
            and funding income amount, or None if not found

        Notes:
            - Funding income is paid/received every 8 hours on perpetual contracts
            - Positive values indicate income received, negative indicates payments made
        """
        logger.info(f"Loading Binance funding from {self.start_date} to {self.end_date}")
        funding_df = load_funding_income(
            start_date=self.start_date,
            end_date=self.end_date,
            funding_dir=self.dir_manager.FUNDING_INCOME_DIR
        )

        if funding_df is None:
            logger.warning(f"No Binance funding data found for {self.start_date} to {self.end_date}")
            return None

        funding_df = funding_df.loc[funding_df['ts'] <= self.end_dt]
        logger.info(f"Loaded {len(funding_df)} Binance funding payments for {funding_df['symbol_venue'].nunique()} symbols")
        return funding_df

    def _calculate_position_age(self, merged_df: pd.DataFrame) -> pd.Series:
        """Calculate position age in days for each symbol based on positions history.

        For each symbol, determines when the current position was established by
        finding the last time the position crossed zero or changed direction.
        Uses vectorized operations with groupby for efficiency.

        Args:
            merged_df: DataFrame with merged positions and fills data

        Returns:
            Series with position age in days, indexed same as merged_df
        """
        position_age = pd.Series(np.float32(0.0), index=merged_df.index)
        lookback_days = 30
        zero_tol = 0.0001

        hist_positions_df = load_binance_positions(
            start_date=self.end_date - td(days=lookback_days),
            end_date=self.end_date,
            pos_dir=self.dir_manager.BINANCE_POSITION_DIR
        )

        if hist_positions_df is None or hist_positions_df.empty:
            logger.warning("No historical positions data available for position age calculation")
            return position_age

        latest_ts = merged_df.index.get_level_values('ts').max()

        # Get current positions at latest timestamp (non-zero only)
        latest_mask = merged_df.index.get_level_values('ts') == latest_ts
        current_positions = merged_df.loc[latest_mask, ['qty']].copy()
        current_positions = current_positions[
            (current_positions['qty'] != 0) & (~current_positions['qty'].isna())
        ]

        if current_positions.empty:
            return position_age

        # Prepare historical data with crossing detection
        hist_df = hist_positions_df.reset_index().sort_values(['symbol_venue', 'ts'])
        hist_df['prev_qty'] = hist_df.groupby('symbol_venue')['qty'].shift(1).fillna(0)

        # Detect crossings vectorized: sign change or from near-zero to significant
        is_near_zero = np.abs(hist_df['qty']) < zero_tol
        prev_near_zero = np.abs(hist_df['prev_qty']) < zero_tol
        hist_df['is_crossing'] = (
            ((hist_df['prev_qty'] * hist_df['qty'] < 0) & ~is_near_zero & ~prev_near_zero) |
            (prev_near_zero & ~is_near_zero)
        )

        # Build current qty lookup
        current_qty_map = {}
        for idx in current_positions.index:
            symbol = idx[1] if isinstance(idx, tuple) else idx
            current_qty_map[symbol] = current_positions.loc[idx, 'qty']

        # Find position start for each symbol using groupby
        def find_position_start(group: pd.DataFrame, symbol: str) -> pd.Timestamp:
            if symbol not in current_qty_map:
                return pd.NaT

            current_qty = current_qty_map[symbol]
            crossings = group[group['is_crossing']]

            if crossings.empty:
                return group.iloc[0]['ts']

            # Get crossings matching current direction
            if current_qty > 0:
                direction_crossings = crossings[crossings['qty'] > zero_tol]
            else:
                direction_crossings = crossings[crossings['qty'] < -zero_tol]

            if not direction_crossings.empty:
                return direction_crossings.iloc[-1]['ts']
            return group.iloc[0]['ts']

        position_starts = hist_df.groupby('symbol_venue').apply(
            lambda g: find_position_start(g, g.name), include_groups=False
        )

        # Calculate age for all symbols at once
        for symbol_venue, start_ts in position_starts.items():
            if pd.isna(start_ts):
                continue
            symbol_mask = merged_df.index.get_level_values('symbol_venue') == symbol_venue
            age_days = max(0, (latest_ts - start_ts).days)
            position_age.loc[symbol_mask] = np.float32(age_days)

        logger.info("Calculated position age for %d positions", (position_age > 0).sum())
        return position_age

    def merge_data(self) -> None:
        """Merge position snapshots, fills, and funding data on [ts, symbol_venue].

        Aggregates fills and funding to the next minute boundary and aligns positions to the
        following minute boundary, then forward-fills position values before merging.

        Returns:
            DataFrame with merged positions, fills, and funding data, indexed by [ts, symbol_venue],
            or None if positions_df is None

        Notes:
            - Fills and funding are aggregated by symbol_venue to the next minute boundary (ceiling)
            - Positions are aligned to the following minute boundary and forward-filled
            - Join is performed on [ts, symbol_venue]
            - If no fills or funding data, returns positions data only
            - Returns cached copy if data hasn't changed
        """
        if self.fills_df is not None:
            # Build aggregation dictionary based on available columns
            fills_grouped_df = self.fills_df.groupby(DF_INDEX).agg({
                'fill_px': 'mean',  # Average fill price
                'fill_qty': 'sum',  # Total quantity filled
                'fill_dollars': 'sum',
                'fill_dollars_buy': 'sum',
                'fill_dollars_sell': 'sum',
                'commission': 'sum',
                'symbol': 'count',  # Count of fills (will be renamed to fill_count)
                'realized_pnl': 'sum'
            })
            fills_grouped_df = fills_grouped_df.rename(columns={'symbol': 'fill_count'})
            logger.info(f"Aggregated {len(self.fills_df)} fills to {len(fills_grouped_df)} minute-level records")
        else:
            fills_grouped_df = set_index(pd.DataFrame(columns=[
                'ts', 'symbol_venue', 'fill_px', 'fill_qty', 'fill_dollars', 'fill_dollars_buy', 'fill_dollars_sell', 'commission', 'fill_count', 'realized_pnl'
            ]))
            logger.info("No fills data to aggregate, will use positions data only")

        # Aggregate funding to next minute boundary (if funding data exists)
        funding_grouped_df = None
        if self.funding_df is not None:
            funding_agg_df = self.funding_df.copy()
            funding_grouped_df = funding_agg_df.groupby(DF_INDEX).agg({
                'funding_income': 'sum'  # Sum funding payments at the same timestamp
            })
            logger.info(f"Aggregated {len(self.funding_df)} funding payments to {len(funding_grouped_df)} minute-level records")
        else:
            logger.info("No funding data to aggregate, will use positions and fills data only")

        # Filter to ensure we don't go past end_dt (resampling can extend beyond)
        positions_df = self.positions_df[self.positions_df.index.get_level_values('ts') <= self.end_dt]

        merged_df = merge_on_index(positions_df, fills_grouped_df, suffixes=('_pos', '_fill'))
        logger.info(f"Merged {len(positions_df)} positions with {len(fills_grouped_df)} fill records")
        for fld in fills_grouped_df.columns:
            merged_df[fld] = merged_df[fld].fillna(0.0)

        live_bar_flds = ['logret_cum', 'logret_cum_wgtmkt', 'close_mid', 'dvolume_cum', 'index_price']
        if self.live_bars_df is not None:
            merged_df = merge_on_index(merged_df, self.live_bars_df[live_bar_flds])
        else:
            # Set default values for live bar fields
            for fld in live_bar_flds:
                merged_df[fld] = 0.0

            # Use historical market return data from market index parquets
            if self.market_return_df is not None:
                ts_values = merged_df.index.get_level_values('ts')
                mkt_ret_map = self.market_return_df['logret_cum_wgtmkt'].to_dict()
                merged_df['logret_cum_wgtmkt'] = ts_values.map(mkt_ret_map)

        # Merge funding data if it exists
        if funding_grouped_df is not None:
            merged_df = merge_on_index(merged_df, funding_grouped_df)
            merged_df['funding_income'] = merged_df['funding_income'].fillna(0.0)
            logger.info(f"Merged with {len(funding_grouped_df)} funding records")
        else:
            merged_df['funding_income'] = 0.0

        merged_df = merged_df.sort_index()

        # Calculate net_pnl: unrealized_pnl + realized_pnl - commission + funding_income
        # Fill missing values with 0 for the calculation
        merged_df['net_pnl'] = (
                merged_df.get('unrealized_pnl', pd.Series(0.0, index=merged_df.index)).fillna(0.0) +
                merged_df.get('realized_pnl', pd.Series(0.0, index=merged_df.index)).fillna(0.0) -
                merged_df.get('commission', pd.Series(0.0, index=merged_df.index)).fillna(0.0) +
                merged_df.get('funding_income', pd.Series(0.0, index=merged_df.index)).fillna(0.0)
        )

        # Filter to analysis period (remove previous day's data used for delta calculation)
        # Filter on ts index level
        merged_df = merged_df[merged_df.index.get_level_values('ts') >= self.start_dt]
        logger.info(f"Filtered to analysis period starting {self.start_dt}")

        #cumsum **after** cutting off previous day
        merged_df['unrealized_pnl_cum'] = merged_df['unrealized_pnl'].unstack().cumsum().stack(future_stack=True)
        merged_df['net_pnl_cum'] = merged_df['net_pnl'].unstack().cumsum().stack(future_stack=True)
        merged_df['realized_pnl_cum'] = merged_df[['realized_pnl']].unstack().cumsum().stack(future_stack=True)
        merged_df = make_date(merged_df)

        # Calculate returns: net_pnl / abs(value) and unrealized_pnl_cum / abs(value)
        merged_df = self.calculate_returns(
            merged_df,
            numerators=['net_pnl', 'realized_pnl', 'unrealized_pnl'],
            denominator='notional'
        )

        # Add buy/sell tracking for fills (needed by trading_reports)
        # Positive fill_dollars = buy, negative = sell
        merged_df['dollars_buy_daily'] = merged_df['fill_dollars'].clip(lower=0).fillna(0.0)
        merged_df['dollars_sell_daily'] = merged_df['fill_dollars'].clip(upper=0).abs().fillna(0.0)
        merged_df['position_age'] = self._calculate_position_age(merged_df)
        merged_df['fill_dollars_abs'] = merged_df['fill_dollars'].abs()

        logger.info(f"Final merged data: {len(merged_df)} total records for {merged_df.index.get_level_values('symbol_venue').nunique()} symbols")
        self.security_ts_pnl_df = merged_df


    def aggregate_by_security_date(self, df: Optional[pd.DataFrame] = None, interval_minutes: int = 1440) -> Optional[pd.DataFrame]:
        """Aggregate merged positions and fills data to specified time resolution.

        Takes the minute-level merged data and aggregates to the specified time interval
        for each symbol_venue, calculating summaries for positions and fills.

        Args:
            df: Merged DataFrame from merge_positions_and_fills(). If None,
                will use self.security_ts_pnl_df.
            interval_minutes: Time interval in minutes for aggregation (default: 1440 for daily).
                Examples: 60 (hourly), 120 (2 hours), 1440 (daily)

        Returns:
            DataFrame with aggregated data, indexed by [ts, symbol_venue] where ts represents
            the start of each time bin, or None if no data available

        Notes:
            - Aggregates fills: sums quantities, notionals, commissions, and realized PnL
            - Aggregates positions: takes end-of-period values for position metrics
            - Uses pd.Grouper to bin timestamps into intervals of specified minutes
            - Time bins align to natural boundaries (e.g., midnight for daily, hour starts, etc.)
        """
        logger.info(f"Aggregating symbols into {interval_minutes} minute intervals")

        persist = False
        if df is None:
            logger.info("No merged_df provided, using self.security_ts_pnl_df...")
            df = self.security_ts_pnl_df
            if df is None:
                logger.warning("No merged data available for aggregation")
                return None
            persist = True

        df = df.reset_index()

        # Filter to only include timestamps up to end_date
        df = df[
            (df['ts'] < date_to_end_dt(self.end_date)) &
            (df['ts'] >= date_to_start_dt(self.start_date))
            ]

        interval_hours = interval_minutes / 60
        logger.info(f"Aggregating {len(df)} minute-level records to {interval_minutes}-minute ({interval_hours:.1f}-hour) resolution")

        df['notional_avg'] = df['notional']

        # Group by time interval and symbol_venue using pd.Grouper
        freq_str = f'{interval_minutes}min'
        agg_df = df.groupby([pd.Grouper(key='ts', freq=freq_str), 'symbol_venue']).agg({
            # PnL components
            'unrealized_pnl_cum': 'last',  # End of period snapshot
            'unrealized_pnl': 'sum',  # Sum of deltas
            'unrealized_pnl_tot_cum': 'last',
            'realized_pnl_cum': 'sum',  # Sum of realized PnL
            'realized_pnl': 'sum',  # Sum of realized PnL
            'commission': 'sum',  # Sum of commissions
            'funding_income': 'sum',  # Sum of funding
            'net_pnl': 'sum',  # Sum of net PnL
            'net_pnl_cum': 'last',  # End of period cumulative net PnL

            # Position metrics - use last value of the period
            'notional': 'last',  # End of period notional value
            'notional_avg': 'mean',

            # Fill metrics
            'fill_count': 'sum',
            'fill_qty': 'sum',
            'fill_dollars': 'sum',
            'fill_dollars_abs': 'sum',
            'fill_px': 'mean',
            'fill_dollars_buy': 'sum',
            'fill_dollars_sell': 'sum',

            'logret_cum': 'last',  # Cumulative - take end of period value
            'logret_cum_wgtmkt': 'last',  # Cumulative market return - take end of period value
            'dvolume_cum': 'last',  # Cumulative volume - take end of period value
            'close_mid': 'last',
            'index_price': 'last',
            'position_age': 'last'  # Position age in days
        })
        # Rename fill_px to avg_fill_px to clarify it's a weighted average
        agg_df = agg_df.rename(columns={'fill_px': 'avg_fill_px'})

        # Reset index to make ts and symbol_venue columns, then set them back as index
        agg_df = agg_df.reset_index().set_index(['ts', 'symbol_venue']).sort_index()
        agg_df = make_symbol(agg_df)

        agg_df = self.calculate_returns(
            agg_df,
            numerators=['net_pnl', 'unrealized_pnl_tot_cum'],
            denominator='notional_avg'
        )

        num_symbols = agg_df.index.get_level_values('symbol_venue').nunique()
        ts_min = agg_df.index.get_level_values('ts').min()
        ts_max = agg_df.index.get_level_values('ts').max()
        logger.info(f"Aggregated to {len(agg_df)} records ({interval_minutes}-minute intervals) for {num_symbols} symbols")
        logger.info(f"Time range: {ts_min} to {ts_max}")

        if persist:
            self.security_date_pnl_df = agg_df

        agg_df = make_date(agg_df)

        # Flag symbols with missing data in key displayed columns
        key_columns = ['close_mid', 'notional', 'alpha_opt']
        missing_mask = pd.Series(False, index=agg_df.index)
        for col in key_columns:
            if col in agg_df.columns:
                missing_mask |= agg_df[col].isna() | (agg_df[col] == 0)
        agg_df['missing_data'] = missing_mask.astype(int)

        return agg_df


    def aggregate_daily_portfolio(self) -> Optional[pd.DataFrame]:
        if self.security_date_pnl_df is None:
            self.aggregate_by_security_date()

        portfolio_date_pnl_df = self.aggregate_portfolio(self.security_date_pnl_df)

        if portfolio_date_pnl_df is None:
            logger.warning("No portfolio data available for daily aggregation")
            self.portfolio_date_pnl_df = None
            return None

        balance_cols = ['balance', 'net_balance', 'bnb_balance', 'bnb_amount']
        daily_balances_df = self.balances_df.groupby('date')[balance_cols].last()
        portfolio_date_pnl_df = merge_on_index(portfolio_date_pnl_df, daily_balances_df)
        portfolio_date_pnl_df = make_date(portfolio_date_pnl_df)
        self.portfolio_date_pnl_df = portfolio_date_pnl_df
        return self.portfolio_date_pnl_df


    def aggregate_portfolio(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Aggregate data across all symbol_venues to portfolio level.

        Takes a DataFrame with MultiIndex [ts, symbol_venue] or [date, symbol_venue]
        and aggregates across all symbols at each timestamp/date to produce
        portfolio-level metrics.

        Args:
            df: DataFrame with MultiIndex containing symbol_venue level

        Returns:
            DataFrame indexed by timestamp or date with portfolio-level aggregates,
            or None if input is None or empty

        Notes:
            - Sums unrealized PnL, realized PnL, commissions, fills across all symbols
            - Calculates total long and short notional values
            - Counts number of positions and symbols traded
        """
        persist = False
        if df is None:
            df = self.security_ts_pnl_df
            persist = True

        if df is None or len(df) == 0:
            logger.warning("No data available for portfolio aggregation")
            return None

        # Determine the time index name (ts or date)
        index_names = df.index.names
        if 'ts' in index_names:
            time_index = 'ts'
        elif 'date' in index_names:
            time_index = 'date'
        else:
            raise log_and_raise(f"DataFrame must have 'ts' or 'date' in index, got: {index_names}")

        logger.info(f"Aggregating {len(df)} records across symbol_venues to portfolio level")

        # Calculate long/short notional from position value
        # Positive value = long position, negative value = short position
        df['long_notional'] = df['notional'].clip(lower=0).fillna(0)
        df['short_notional'] = df['notional'].clip(upper=0).fillna(0)
        df['gross_notional'] = df['notional'].abs().fillna(0)

        portfolio_agg_df = flatten_cols(df.reset_index().groupby(time_index).agg({
            # PnL components (sum across all symbols)
            'unrealized_pnl_cum': 'sum',  # Total portfolio unrealized PnL
            'unrealized_pnl': 'sum',  # Sum of deltas
            'unrealized_pnl_tot_cum': 'sum',
            'realized_pnl_cum': 'sum',  # Total realized PnL
            'realized_pnl': 'sum',  # Total realized PnL
            'commission': 'sum',  # Total commissions
            'funding_income': 'sum',  # Total funding
            'net_pnl': 'sum',  # Total net PnL
            'net_pnl_cum': 'sum',  # Total portfolio cumulative net PnL

            # Position metrics
            'long_notional': 'sum',  # Total long notional
            'short_notional': 'sum',  # Total short notional (negative)
            'gross_notional': 'sum',  # Total gross notional

            # Trading activity
            'fill_count': 'sum',  # Total fills across all symbols
            'fill_dollars_abs': 'sum',  # Total traded notional
            'fill_dollars_buy': 'sum',
            'fill_dollars_sell': 'sum',

            # Symbol counts
            'symbol_venue': 'nunique',  # Number of unique symbols

            # Market return - use 'first' since it's the same value for all symbols at each timestamp
            'logret_cum_wgtmkt': 'first'
        }))

        # Rename columns - remove _sum suffix from main columns and rename specific columns
        portfolio_agg_df = portfolio_agg_df.rename(columns={
            'ts_last': 'ts',
            'unrealized_pnl_cum_sum': 'unrealized_pnl_cum',
            'unrealized_pnl_sum': 'unrealized_pnl',
            'realized_pnl_cum_sum': 'realized_pnl_cum',
            'realized_pnl_sum': 'realized_pnl',
            'commission_sum': 'commission',
            'funding_income_sum': 'funding_income',
            'net_pnl_sum': 'net_pnl',
            'net_pnl_cum_sum': 'net_pnl_cum',
            'long_notional_sum': 'long_notional',
            'short_notional_sum': 'short_notional',
            'gross_notional_sum': 'gross_notional',
            'fill_count_sum': 'fill_count',
            'fill_dollars_abs_sum': 'fill_dollars_abs',
            'symbol_venue_nunique': 'num_symbols',
            'fill_dollars_buy_sum': 'fill_dollars_buy',
            'fill_dollars_sell_sum': 'fill_dollars_sell',
            'logret_cum_wgtmkt_first': 'logret_cum_wgtmkt'
        })

        # Calculate rolling average of gross_notional
        portfolio_agg_df['gross_notional_avg'] = portfolio_agg_df['gross_notional'].expanding().mean()

        logger.info(f"Aggregated to {len(portfolio_agg_df)} portfolio-level records")
        logger.info(f"Time range: {portfolio_agg_df.index.min()} to {portfolio_agg_df.index.max()}")

        # Calculate returns for period-specific PnL metrics using gross_notional as denominator
        portfolio_agg_df = BinancePnl.calculate_returns(
            portfolio_agg_df,
            numerators=['realized_pnl', 'unrealized_pnl', 'net_pnl'],
            denominator='gross_notional_avg'
        )

        # Calculate running standard deviation of net_pnl_return
        portfolio_agg_df['net_pnl_return_std'] = portfolio_agg_df['net_pnl_return'].expanding().std()

        # Calculate Sharpe ratio: (mean daily return / std of returns) * sqrt(365)
        net_pnl_return_mean = portfolio_agg_df['net_pnl_return'].expanding().mean()
        portfolio_agg_df['sharpe'] = (net_pnl_return_mean / portfolio_agg_df['net_pnl_return_std']) * (365 ** 0.5)
        portfolio_agg_df['fill_dollars_buy_cum'] = portfolio_agg_df['fill_dollars_buy'].cumsum()
        portfolio_agg_df['fill_dollars_sell_cum'] = portfolio_agg_df['fill_dollars_sell'].cumsum()
        portfolio_agg_df['unrealized_period_pnl_cum'] = portfolio_agg_df['unrealized_pnl'].cumsum()

        if persist:
            if time_index == 'ts':
                self.portfolio_ts_pnl_df = portfolio_agg_df
            elif time_index == 'date':
                self.portfolio_date_pnl_df = portfolio_agg_df

        return portfolio_agg_df

    @staticmethod
    def calculate_returns(df: pd.DataFrame, numerators: List[str], denominator: str) -> pd.DataFrame:
        """Calculate return metrics by dividing numerator columns by a common denominator.

        Generic return calculation that creates new columns named '{numerator}_return' for each
        numerator column divided by the same denominator column.

        Args:
            df: DataFrame with PnL columns
            numerators: List of column names to use as numerators (e.g., ['total_pnl_daily', 'unrealized_pnl_cum'])
            denominator: Single column name to use as denominator for all calculations (e.g., 'value')

        Returns:
            DataFrame with added columns named '{numerator}_return' for each numerator

        Raises:
            ValueError: If denominator column or any numerator column is not found in DataFrame

        Notes:
            - All returns use the same denominator for consistency
            - Returns are calculated as numerator / abs(denominator)
            - Returns are set to 0 when denominator is 0 or missing
            - Uses absolute value of denominator for directional neutrality

        Example:
            >>> df = calculate_returns(df, numerators=['total_pnl_daily', 'unrealized_pnl_cum'], denominator='value')
            >>> # Creates columns: 'total_pnl_daily_return' and 'unrealized_pnl_cum_return'
            >>> # Both use 'value' as the denominator
        """
        logger.info(f"Calculating returns using {numerators} / {denominator}")
        # Calculate returns for each numerator using the same denominator
        for numerator in numerators:
            # Replace zeros with NaN to avoid division by zero, then fill NaN with 0
            denominator_abs = df[denominator].abs()
            # Create a safe denominator that replaces 0 with NaN
            safe_denominator = denominator_abs.replace(0, np.nan)
            # Perform division (will produce NaN where denominator was 0)
            returns = df[numerator] / safe_denominator
            # Fill NaN values with 0.0
            df[f"{numerator}_return"] = returns.fillna(0.0).astype(np.float32)

        logger.info(f"Calculated {len(numerators)} return columns using '{denominator}' as denominator")
        return df

    @property
    def as_of(self) -> Optional[dt]:
        """Get the timestamp of the latest data available.

        Returns the maximum timestamp from the merged data if available,
        otherwise None.

        Returns:
            Datetime of the most recent data point, or None if no merged data available
        """
        if self.security_ts_pnl_df is None or len(self.security_ts_pnl_df) == 0:
            return None
        # Get the maximum timestamp from the merged data index
        return self.security_ts_pnl_df.index.get_level_values('ts').max()

    def get_fills_df(self) -> Optional[pd.DataFrame]:
        """Get a copy of the fills DataFrame.

        Returns:
            Copy of fills DataFrame with 'expanding', 'date', and 'realized_pnl' columns,
            or None if no fills data available
        """
        if self.fills_df is None:
            return None
        return self.fills_df.copy()

    def get_top_winners(self, n: int = 5) -> pd.DataFrame:
        """Get top N winning securities by net_pnl.

        Args:
            n: Number of winners to return

        Returns:
            DataFrame with top winners sorted by net_pnl descending
        """
        if self.security_date_pnl_df is None or self.security_date_pnl_df.empty:
            return pd.DataFrame()

        symbol_pnl_df = self.security_date_pnl_df.groupby('symbol_venue').agg({
            'net_pnl': 'sum', 'realized_pnl': 'sum', 'unrealized_pnl': 'sum'
        }).sort_values('net_pnl', ascending=False)
        symbol_pnl_df.index = symbol_pnl_df.index.str.replace('_binance-futures', '', regex=False)
        return symbol_pnl_df.head(n)

    def get_top_losers(self, n: int = 5) -> pd.DataFrame:
        """Get top N losing securities by net_pnl.

        Args:
            n: Number of losers to return

        Returns:
            DataFrame with top losers sorted by net_pnl ascending
        """
        if self.security_date_pnl_df is None or self.security_date_pnl_df.empty:
            return pd.DataFrame()

        symbol_pnl_df = self.security_date_pnl_df.groupby('symbol_venue').agg({
            'net_pnl': 'sum', 'realized_pnl': 'sum', 'unrealized_pnl': 'sum'
        }).sort_values('net_pnl', ascending=False)
        symbol_pnl_df.index = symbol_pnl_df.index.str.replace('_binance-futures', '', regex=False)
        return symbol_pnl_df.nsmallest(n, 'net_pnl')
