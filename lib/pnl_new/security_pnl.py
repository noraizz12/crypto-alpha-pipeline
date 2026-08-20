"""Calculate PnL for a single security using FIFO accounting."""

import logging
from datetime import datetime as dt, time
from typing import Optional, Set, Dict

import pandas as pd
from lib.pnl_new.pnl_util import calc_pnl_returns

logger = logging.getLogger(__name__)


class SecurityPnl:
    """Calculate FIFO PnL for a single security.

    This class processes fills, prices, and funding data for one security
    to calculate profit and loss metrics. Designed to be run in parallel
    across multiple securities.

    Attributes:
        symbol_venue (str): Symbol and venue identifier (e.g., "BTCUSDT_binance-futures")
        start_dt (datetime): Start datetime for PnL calculation
        end_dt (datetime): End datetime for PnL calculation
        fills_df (pd.DataFrame): Fill data for this security
        bars_df (pd.DataFrame): Price/bar data for this security
        fundings_df (pd.DataFrame): Funding payment data for this security
        initial_position (dict): Initial position state (qty, cost_basis, value)
        pnl_df (pd.DataFrame): Calculated PnL results by timestamp
    """

    def __init__(
        self,
        symbol_venue: str,
        start_dt: dt,
        end_dt: dt,
        fills_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        fundings_df: Optional[pd.DataFrame] = None,
        initial_position: Optional[Dict[str, float]] = None,
        pnl_times: Optional[Set[time]] = None
    ):
        """Initialize SecurityPnl calculator for a single security.

        Args:
            symbol_venue: Symbol and venue identifier
            start_dt: Start datetime for PnL calculation
            end_dt: End datetime for PnL calculation
            fills_df: DataFrame with fills for this security
                Required columns: ts, side, fill_px, fill_qty, commission
            bars_df: DataFrame with bars for this security
                Required columns: close_mid, volume, dvolume
                Index: ts (datetime)
            fundings_df: Optional DataFrame with funding payments
                Required columns: ts, funding_income
            initial_position: Optional dict with initial position state
                Keys: qty, cost_basis, value
            pnl_times: Optional set of times (HH:MM:SS) at which to calculate PnL
                If None, calculate PnL at all bar timestamps
        """
        self.symbol_venue = symbol_venue
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.bars_df = bars_df
        self.fundings_df = fundings_df
        self.initial_position = initial_position or {'qty': 0.0, 'cost_basis': 0.0, 'value': 0.0}
        self.pnl_times = pnl_times

        # Standardize fill timestamps
        self.fills_df = self._standardize_fill_timestamps(fills_df)

        self.pnl_df = None

        # Validate input data
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input data has required columns and structure."""
        # Validate fills
        required_fill_cols = ['ts', 'side', 'fill_px', 'fill_qty', 'commission']
        optional_fill_cols = ['realized_pnl']
        if self.fills_df is not None and len(self.fills_df) > 0:
            missing_cols = set(required_fill_cols) - set(self.fills_df.columns)
            if missing_cols:
                raise ValueError(f"fills_df missing required columns: {missing_cols}")

            # Add realized_pnl column if missing (for test data or OMS fills)
            if 'realized_pnl' not in self.fills_df.columns:
                self.fills_df['realized_pnl'] = 0.0

        # Validate bars
        required_bar_cols = ['close_mid', 'volume', 'dvolume']
        if self.bars_df is not None and len(self.bars_df) > 0:
            missing_cols = set(required_bar_cols) - set(self.bars_df.columns)
            if missing_cols:
                raise ValueError(f"bars_df missing required columns: {missing_cols}")
            if 'ts' not in self.bars_df.index.names:
                raise ValueError("bars_df must have 'ts' as index")

        # Validate fundings
        if self.fundings_df is not None and len(self.fundings_df) > 0:
            required_funding_cols = ['ts', 'funding_income']
            missing_cols = set(required_funding_cols) - set(self.fundings_df.columns)
            if missing_cols:
                raise ValueError(f"fundings_df missing required columns: {missing_cols}")

        # Validate initial position
        required_pos_keys = ['qty', 'cost_basis', 'value']
        missing_keys = set(required_pos_keys) - set(self.initial_position.keys())
        if missing_keys:
            raise ValueError(f"initial_position missing required keys: {missing_keys}")

    def _extend_bars_to_end_dt(self, bars_df: pd.DataFrame, end_dt: dt) -> pd.DataFrame:
        """Extend bars with forward-filled prices to cover period through end_dt.

        Args:
            bars_df: DataFrame with bars data (must have ts index or column and close_mid column)
            end_dt: End datetime to extend bars to

        Returns:
            DataFrame with bars extended to end_dt using forward-fill of last available price
        """
        bars_with_ts_df = bars_df.reset_index() if 'ts' in bars_df.index.names else bars_df.copy()

        last_bar_ts = bars_with_ts_df['ts'].max()

        if end_dt > last_bar_ts:
            # Forward-fill bars to end_dt
            last_bar = bars_with_ts_df.iloc[-1].copy()
            last_price = last_bar['close_mid']

            # Create minute-by-minute bars from last bar to end_dt
            missing_timestamps = pd.date_range(
                start=last_bar_ts + pd.Timedelta(minutes=1),
                end=end_dt,
                freq='1min'
            )

            if len(missing_timestamps) > 0:
                # Create forward-filled bars
                extended_bars = pd.DataFrame({'ts': missing_timestamps})
                extended_bars['close_mid'] = last_price

                # Add other required columns with forward-filled values
                for col in bars_with_ts_df.columns:
                    if col not in ['ts', 'close_mid']:
                        extended_bars[col] = last_bar[col]

                # Append extended bars
                bars_with_ts_df = pd.concat([bars_with_ts_df, extended_bars], ignore_index=True)

                logger.info(
                    f"{self.symbol_venue}: Forward-filled {len(missing_timestamps)} bars "
                    f"from {last_bar_ts} to {end_dt} at price ${last_price:.2f}"
                )

        return bars_with_ts_df

    def _standardize_fill_timestamps(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize fill timestamps to closest next PnL time.

        If pnl_times is None, rounds to the next minute.
        Otherwise, rounds to the next PnL time (rolling to next day if needed).

        Args:
            fills_df: DataFrame with fill data

        Returns:
            DataFrame with standardized 'ts' column (replaces original ts)
        """
        if self.pnl_times is None:
            # Calculate every minute - ceil to next minute
            fills_df['ts'] = fills_df['ts'].dt.ceil('1min')
        else:
            # Round to next PnL time
            def get_next_pnl_time(ts: dt) -> dt:
                ts_time = ts.time()
                # Find next pnl_time after current time
                next_times = [t for t in self.pnl_times if t > ts_time]

                if next_times:
                    # Use next time today
                    next_time = min(next_times)
                    return dt.combine(ts.date(), next_time, tzinfo=ts.tzinfo)
                else:
                    # Roll to first time tomorrow
                    next_time = min(self.pnl_times)
                    next_day = ts.date() + pd.Timedelta(days=1)
                    return dt.combine(next_day, next_time, tzinfo=ts.tzinfo)

            fills_df['ts'] = fills_df['ts'].apply(get_next_pnl_time)

        return fills_df

    def _aggregate_fills(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fills at each standardized timestamp.

        Sums quantity, calculates total dollar value, and derives average price.

        Args:
            fills_df: DataFrame with 'ts' column

        Returns:
            DataFrame with aggregated fills: ts, fill_qty, fill_dollars, fill_px_avg, fill_count, commission, realized_pnl
        """
        # Calculate dollar value for each fill
        fills_df['fill_dollars'] = fills_df['fill_px'] * fills_df['fill_qty']

        # Aggregate at each timestamp
        agg_fills_df = fills_df.groupby('ts').agg({
            'fill_qty': 'sum',
            'fill_dollars': 'sum',
            'commission': 'sum',
            'realized_pnl': 'sum',
            'side': 'count'  # Count of fills
        }).reset_index()

        # Rename count column
        agg_fills_df = agg_fills_df.rename(columns={'side': 'fill_count'})

        # Calculate average price as total_dollars / total_qty
        agg_fills_df['fill_px_avg'] = agg_fills_df['fill_dollars'] / agg_fills_df['fill_qty']

        return agg_fills_df

    def calculate_pnl(self) -> pd.DataFrame:
        """Calculate FIFO PnL for this security.

        Returns:
            DataFrame with PnL metrics by timestamp
        """
        # Step 1: Aggregate fills at each timestamp
        fills_agg_df = self._aggregate_fills(self.fills_df)

        # Step 2: Extend bars to end_dt with forward-fill if needed
        bars_with_ts_df = self._extend_bars_to_end_dt(self.bars_df, self.end_dt)

        # Step 3: Merge fills and fundings into bars data to get complete timestamp index
        pnl_df = bars_with_ts_df[['ts', 'close_mid']].merge(
            fills_agg_df,
            on='ts',
            how='left'
        )
        pnl_df = pnl_df.rename(columns={'close_mid': 'mark_price'})

        # Merge funding income
        if self.fundings_df is not None and len(self.fundings_df) > 0:
            pnl_df = pnl_df.merge(
                self.fundings_df[['ts', 'funding_income']],
                on='ts',
                how='left'
            )

        # Step 4: Fill NaN values from merge
        # Flow variables (fills, commissions, fundings) should be 0 when no activity
        flow_columns = ['fill_qty', 'fill_dollars', 'commission', 'realized_pnl',
                       'fill_count', 'fill_px_avg', 'funding_income']
        for col in flow_columns:
            if col in pnl_df.columns:
                pnl_df[col] = pnl_df[col].fillna(0)

        # Step 5: Calculate cumulative quantity
        # Start with initial position quantity, then accumulate fill quantities
        pnl_df['qty'] = self.initial_position['qty'] + pnl_df['fill_qty'].cumsum()

        # Step 6: Calculate cumulative metrics
        # Cumulative commissions
        pnl_df['commission_cum'] = pnl_df['commission'].cumsum()

        # Cumulative funding income
        if 'funding_income' in pnl_df.columns:
            pnl_df['funding_income_cum'] = pnl_df['funding_income'].cumsum()
        else:
            pnl_df['funding_income_cum'] = 0.0

        # Cumulative realized PnL
        pnl_df['realized_pnl_cum'] = pnl_df['realized_pnl'].cumsum()

        # Position value (qty * mark_price)
        pnl_df['position'] = pnl_df['qty'] * pnl_df['mark_price']

        # Cumulative absolute dollar volume
        pnl_df['abs_dollars_cum'] = pnl_df['fill_dollars'].abs().cumsum()

        # Step 7: Calculate average cost basis
        # Cumulative cost = initial cost + cumulative fill dollars
        initial_cost = self.initial_position['cost_basis']
        pnl_df['cost_cum'] = initial_cost + pnl_df['fill_dollars'].cumsum()

        # Average cost basis = cumulative cost / cumulative qty
        pnl_df['cost_basis'] = pnl_df['cost_cum'] / pnl_df['qty']

        # Step 8: Calculate PnL
        # Cumulative gross PnL at each timestamp = position - cost_cum
        pnl_df['pnl_gross_cum'] = pnl_df['position'] - pnl_df['cost_cum']

        # Cumulative net PnL = Gross PnL - commissions + funding income
        pnl_df['pnl_net_cum'] = pnl_df['pnl_gross_cum'] - pnl_df['commission_cum'] + pnl_df['funding_income_cum']

        # Cumulative unrealized PnL = Net PnL - Realized PnL (realized_pnl from Binance is already net of fees)
        pnl_df['unrealized_pnl_cum'] = pnl_df['pnl_net_cum'] - pnl_df['realized_pnl_cum']

        # Period PnL (change from prior period)
        pnl_df['pnl_gross'] = pnl_df['pnl_gross_cum'].diff().fillna(pnl_df['pnl_gross_cum'])
        pnl_df['pnl_net'] = pnl_df['pnl_net_cum'].diff().fillna(pnl_df['pnl_net_cum'])
        pnl_df['unrealized_pnl'] = pnl_df['unrealized_pnl_cum'].diff().fillna(pnl_df['unrealized_pnl_cum'])

        # Notional = absolute value of position
        pnl_df['notional'] = pnl_df['position'].abs()

        # Step 8: Calculate returns on gross, net, realized, and unrealized PnL
        for pnl_col in ['pnl_gross', 'pnl_net', 'realized_pnl_cum', 'unrealized_pnl']:
            pnl_df = calc_pnl_returns(pnl_df, pnl_col=pnl_col)

        return pnl_df
