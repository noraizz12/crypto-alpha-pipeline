from collections import defaultdict
from datetime import datetime as dt
from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
import pandas as pd

from lib.pnl.pnl_util import round_dust_position, get_commission_px_dict
from lib.trader import Side, Fill
from lib.util import merge_on_index, set_index, concat, unique_list, to_datetime
from lib.util.dataframes import make_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PNL_DF_COLUMNS = [
    'date', 'realized_daily', 'unrealized_daily', 'total_pnl_daily', 'total_pnl_cumulative', 'unrealized_pnl',
    'qty', 'notional', 'position_age', 'daily_return', 'unrealized_return', 'fees_daily', 'fees_cumulative', 'fees_usd_daily', 'fees_usd_cumulative',
    'funding_income_daily', 'funding_income_cumulative', 'mark_price', 'fill_cnt_daily',
    'dollars_traded_daily', 'dollars_buy_daily', 'dollars_sell_daily', 'dollars_buy_cumulative', 'dollars_sell_cumulative']

DAILY_PNL_DF_COLUMNS = [
    'long', 'short', 'bias', 'realized_daily', 'unrealized_daily', 'total_pnl_daily',
    'total_pnl_mtd', 'total_pnl_ytd', 'total_pnl_lifetime',
    'return_daily', 'return_mtd', 'return_ytd', 'return_lifetime',
    'risk_lifetime', 'sharpe_lifetime', 'fees_daily', 'fees_lifetime', 'fees_usd_daily', 'fees_usd_lifetime',
    'funding_income_daily', 'funding_income_lifetime', 'fill_cnt_daily', 'fill_cnt_lifetime', 'dollars_traded_daily', 'dollars_traded_lifetime',
    'dollars_buy_daily', 'dollars_buy_lifetime', 'dollars_sell_daily', 'dollars_sell_lifetime', 'notional_abs_daily']


def parse_fill_data(fill_data: np.ndarray) -> Fill:
    """Convert numpy array fill data to Fill object.

    Args:
        fill_data: Numpy array with fill information

    Returns:
        Fill object with parsed data
    """
    return Fill(
        symbol=fill_data[1],
        side=Side.from_string(fill_data[4]),
        px=fill_data[3],
        qty=fill_data[2],
        exch_ts=fill_data[0],
        recv_ts=fill_data[0],
        commission=fill_data[5],
        commission_asset=fill_data[6],
    )


class CalcMultiSymbolFillPnl:
    """Core FIFO PnL calculation engine for multiple symbols.

    Implements First-In-First-Out (FIFO) accounting methodology to calculate
    realized and unrealized PnL across multiple trading symbols. Handles
    inventory tracking, funding income, and commission calculations.

    Attributes:
        start_dt (datetime): Start datetime for calculations
        end_dt (datetime): End datetime for calculations
        bars_df (pd.DataFrame): Price bar data
        fills_df (pd.DataFrame): Trading fill data
        positions_df (pd.DataFrame): Initial position data
        fundings_df (pd.DataFrame): Funding income data
        fills_pnl_df (pd.DataFrame): Fill-level PnL breakdown
        entry_fills_pnl_df (pd.DataFrame): Entry trade PnL attribution
        fills_dict (Dict): Fill data organized by symbol_venue
        commission_px_dict (Dict): Commission asset prices by timestamp
        bars_dict (Dict): Bar data organized by symbol_venue
        funding_income_dict (Dict): Funding data organized by symbol_venue
        symbol_venues (List[str]): List of all traded symbol_venue pairs
        last_processed_fill_idx (Dict): Track processing progress by symbol
        last_processed_funding_idx (Dict): Track funding processing by symbol
        realized_pnl (Dict): Cumulative realized PnL by symbol
        fees (Dict): Cumulative fees by symbol
        fees_usd (Dict): Cumulative fees in USD by symbol
        funding_income (Dict): Cumulative funding income by symbol
        fill_cnt (Dict): Fill count by symbol
        dollars_traded (Dict): Total dollars traded by symbol
        dollars_case (Dict[Dict]): Dollars traded by side and symbol
        fills_pnl_list (List): List of fill PnL records
        entry_fills_pnl_list (List): List of entry fill PnL records
        inventory_case (Dict[Dict]): FIFO inventory by side and symbol
    """

    def __init__(self, start_dt: dt, end_dt: dt):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.bars_df = None
        self.fills_df = None
        self.positions_df = None
        self.fundings_df = None
        self.fills_pnl_df = None
        self.entry_fills_pnl_df = None
        self.fills_dict = {}
        self.commission_px_dict = {}
        self.bars_dict = {}
        self.funding_income_dict = {}
        self.symbol_venues = []
        self.last_processed_fill_idx = {}
        self.last_processed_funding_idx = {}
        self.last_processed_bars_idx = {}
        self.realized_pnl = {}
        self.fees = {}
        self.fees_usd = {}
        self.funding_income = {}
        self.fill_cnt = {}
        self.dollars_traded = {}
        self.dollars_case = defaultdict(dict)
        self.fills_pnl_list = []
        self.entry_fills_pnl_list = []
        self.inventory_case = defaultdict(dict)
        self._cached_pnl_list = None
        self._last_processed_pnl_df = None

    def load_data(self, fills_df: Optional[pd.DataFrame], bars_df: pd.DataFrame, positions_df: pd.DataFrame, fundings_df: pd.DataFrame, update: bool):
        logger.info(f"Loading data with update={update}")

        if not update:
            self._get_initial_symbol_venues(fills_df, bars_df, positions_df, fundings_df)

        if fills_df is not None:
            fills_df = self._generate_data_after_init_pos(fills_df.sort_values(['ts', 'symbol_venue']))
        self.fills_df = fills_df
        self.bars_df = bars_df.sort_index()
        self.positions_df = positions_df.sort_index() if positions_df is not None else None
        if fundings_df is not None:
            fundings_df = self._generate_data_after_init_pos(fundings_df.sort_values(['ts', 'symbol_venue']), inclusive=True)
        self.fundings_df = fundings_df

        self.get_initial_fills_bars_dict()
        if not update:
            self._get_initial_inventory()

    def get_initial_fills_bars_dict(self):
        """Convert DataFrames to numpy arrays organized by symbol for efficient processing.

        Also handles cases where fills or positions occur after the latest bar data
        by using their prices for PnL calculation.
        """
        logger.info("get_initial_fills_bars_dict")

        #this is stupid
        # Convert DataFrames to numpy arrays and split by symbol_venue
        def build_data_dict(data_df: pd.DataFrame, cols: List[str]) -> Dict:
            return {sv: df[cols].to_numpy() for sv, df in data_df.groupby('symbol_venue', observed=False)}

        if self.fills_df is not None:
            self.fills_dict = build_data_dict(self.fills_df, ['ts', 'symbol', 'fill_qty', 'fill_px', 'side', 'commission', 'commission_asset'])
        if self.fundings_df is not None:
            self.funding_income_dict = build_data_dict(self.fundings_df, ['ts', 'symbol', 'funding_income'])

        # In some cases, especially live data, we could see fills happened after we have bars data, so we need to include the fills px to update our pnl calculation
        # otherwise, if we only use bars data, we will miss the pnl from those fills
        # The same thing could also happen for positions data, so we need to include positions if we see it after all bars updates
        close_mid_df_list = [self.bars_df[['close_mid']]]
        latest_bar_ts = self.bars_df.reset_index().groupby('symbol_venue', observed=False)['ts'].max()

        def filter_last_df(df: pd.DataFrame, latest_bar_ts: pd.DataFrame) -> pd.DataFrame:
            last_df = df.groupby('symbol_venue', observed=False).last()
            last_df = merge_on_index(last_df, latest_bar_ts, how='left', suffixes=('', '_bar'))
            last_df = last_df.loc[last_df['ts'] > last_df['ts_bar']]
            return last_df

        if self.fills_df is not None:
            logger.info(f"Getting latest fills")
            last_fills_df = filter_last_df(self.fills_df, latest_bar_ts)
            last_fills_df = last_fills_df.loc[~last_fills_df['fill_px'].isna()]
            last_fills_df = last_fills_df.rename(columns={'fill_px': 'close_mid'})
            close_mid_df_list.append(set_index(last_fills_df[['ts', 'close_mid']].reset_index()))
        if self.positions_df is not None:
            logger.info(f"Getting latest positions")
            last_positions_df = filter_last_df(self.positions_df.reset_index(), latest_bar_ts)
            last_positions_df = last_positions_df.loc[last_positions_df['qty'] != 0]
            last_positions_df['close_mid'] = last_positions_df['cost_basis'] / last_positions_df['qty']
            close_mid_df_list.append(set_index(last_positions_df[['ts', 'close_mid']].reset_index()))

        # After collecting all price updates, we should do a forward fill to ensure the latest ts slice get price information
        close_mid_df = concat(close_mid_df_list, fast=True).sort_index()
        close_mid_df = close_mid_df.unstack().ffill().stack(future_stack=True)
        close_mid_df = make_date(close_mid_df)

        self.commission_px_dict = get_commission_px_dict(close_mid_df)
        self.bars_dict = {sv: df.reset_index()[['ts', 'date', 'close_mid']].to_numpy() for sv, df in close_mid_df.groupby('symbol_venue', observed=False)}

    def _get_initial_symbol_venues(
            self,
            fills_df: Optional[pd.DataFrame],
            bars_df: pd.DataFrame,
            positions_df: Optional[pd.DataFrame],
            fundings_df: Optional[pd.DataFrame],
    ) -> None:
        """Extract unique symbol_venue pairs and initialize tracking dictionaries.

        Args:
            fills_df: Fill data
            bars_df: Bar data
            positions_df: Position data
            fundings_df: Funding data
        """
        fills_symbol_venues = fills_df['symbol_venue'].to_list() if fills_df is not None else []
        bars_symbol_venues = bars_df.index.get_level_values('symbol_venue').to_list()
        positions_symbol_venues = positions_df.index.get_level_values('symbol_venue').to_list() if positions_df is not None else []
        fundings_symbol_venues = fundings_df['symbol_venue'].to_list() if fundings_df is not None else []
        self.symbol_venues = unique_list(fills_symbol_venues + bars_symbol_venues + positions_symbol_venues + fundings_symbol_venues)

        # Use idx as use numpy array
        for sv in self.symbol_venues:
            self.last_processed_fill_idx[sv] = -1
            self.last_processed_funding_idx[sv] = -1
            self.last_processed_bars_idx[sv] = -1
            self.realized_pnl[sv] = 0.0
            self.funding_income[sv] = 0.0
            self.fees[sv] = 0.0
            self.fees_usd[sv] = 0.0
            self.fill_cnt[sv] = 0
            self.dollars_traded[sv] = 0.0
            self.dollars_case[Side.BUY][sv] = 0.0
            self.dollars_case[Side.SELL][sv] = 0.0

    def _generate_data_after_init_pos(self, df: pd.DataFrame, inclusive: bool = False) -> pd.DataFrame:
        """Filter data to only include records after initial position timestamp.

        Args:
            df: DataFrame to filter
            inclusive: If True, includes data at position timestamp

        Returns:
            Filtered DataFrame with only data after initial positions
        """
        df['data_after_init_pos'] = False
        for sv in self.symbol_venues:
            pos_ts = self.start_dt
            if self.positions_df is not None:
                sv_idx = self.positions_df.index.get_level_values('symbol_venue') == sv
                if not self.positions_df.loc[sv_idx].empty:
                    pos_ts = self.positions_df.loc[sv_idx].index.get_level_values('ts')[0]
            ts_condition = df['ts'] >= pos_ts if inclusive else df['ts'] > pos_ts
            df.loc[(df['symbol_venue'] == sv) & ts_condition, 'data_after_init_pos'] = True

        df = df.loc[df['data_after_init_pos']]
        df = df.drop(columns=['data_after_init_pos'])
        return df

    def _get_initial_inventory(self):
        """Initialize FIFO inventory structures from initial positions.

        Creates separate inventory arrays for long and short positions
        using the initial positions as the starting point.
        """
        if self.positions_df is not None:
            # Get the latest position for each symbol_venue
            latest_positions = self.positions_df.groupby(level='symbol_venue').last()
            # Calculate average price
            latest_positions['avg_price'] = np.where(
                latest_positions['qty'] != 0,
                latest_positions['cost_basis'] / latest_positions['qty'],
                0,
            )
            # Create structured arrays for long and short positions
            position_case = {Side.BUY: latest_positions[latest_positions['qty'] > 0], Side.SELL: latest_positions[latest_positions['qty'] < 0]}
            for case, sign in [(Side.BUY, 1), (Side.SELL, -1)]:
                if not position_case[case].empty:
                    self.inventory_case[case] = {
                        sv: np.array(list(zip(group['avg_price'], sign * group['qty'], np.full(len(group), self.start_dt))), dtype=[('price', float), ('quantity', float), ('ts', dt)])
                        for sv, group in position_case[case].groupby(level='symbol_venue')
                    }

        # Fill in empty arrays for symbol_venues not in positions
        for sv in self.symbol_venues:
            for case in [Side.BUY, Side.SELL]:
                if sv not in self.inventory_case[case]:
                    self.inventory_case[case][sv] = np.array([], dtype=[('price', float), ('quantity', float), ('ts', dt)])

    def _match_fills_with_inventory(self, fill: Fill, symbol_venue: str):
        """Match a new fill against existing inventory using FIFO methodology.

        Processes fills to calculate realized PnL when positions are closed,
        and updates inventory for expanding positions.

        Args:
            fill: Fill object representing the trade
            symbol_venue: Symbol and venue identifier
        """
        oppsite_side = Side.SELL if fill.side == Side.BUY else Side.BUY
        self.fill_cnt[symbol_venue] += 1
        self.dollars_traded[symbol_venue] += fill.px * fill.qty
        fills_pnl_dict = {'symbol': fill.symbol, 'ts': fill.exch_ts, 'side': fill.side, 'fill_px': fill.px, 'fill_qty': fill.qty, 'expanding': True, 'realized_pnl': 0}
        pre_realized_pnl = self.realized_pnl[symbol_venue]

        def diff_by_side(p1: float, p2: float, side):
            return p1 - p2 if side == Side.BUY else p2 - p1

        self.dollars_case[fill.side][symbol_venue] += fill.px * fill.qty
        remaining_qty = fill.qty
        match_inv = self.inventory_case[oppsite_side][symbol_venue]
        while remaining_qty > 0 and len(match_inv) > 0:
            fills_pnl_dict['expanding'] = False
            inv_price, inv_qty, entry_ts = match_inv[0]
            if inv_qty <= remaining_qty:
                # Full match - use entire inventory position
                realized_pnl = diff_by_side(inv_price, fill.px, fill.side) * inv_qty
                self.entry_fills_pnl_list.append({'symbol_venue': symbol_venue, 'ts': entry_ts, 'realized_pnl': realized_pnl})
                self.realized_pnl[symbol_venue] += realized_pnl
                remaining_qty -= inv_qty
                match_inv = match_inv[1:]
            else:
                # Partial match - use portion of inventory position
                realized_pnl = diff_by_side(inv_price, fill.px, fill.side) * remaining_qty
                self.entry_fills_pnl_list.append({'symbol_venue': symbol_venue, 'ts': entry_ts, 'realized_pnl': realized_pnl})
                self.realized_pnl[symbol_venue] += realized_pnl
                match_inv[0] = (inv_price, inv_qty - remaining_qty, entry_ts)
                remaining_qty = 0

        self.inventory_case[oppsite_side][symbol_venue] = match_inv
        if remaining_qty > 0:
            self.inventory_case[fill.side][symbol_venue] = np.append(
                self.inventory_case[fill.side][symbol_venue], np.array([(fill.px, remaining_qty, fill.exch_ts)], dtype=self.inventory_case[fill.side][symbol_venue].dtype))

        fills_pnl_dict['realized_pnl'] = self.realized_pnl[symbol_venue] - pre_realized_pnl
        self.fills_pnl_list.append(fills_pnl_dict)


    # Update realized pnl and inventories
    def _update_pnl_from_new_fills(self, symbol_venue: str, current_time: dt):
        """Process new fills up to current time and update PnL.

        Args:
            symbol_venue: Symbol to process
            current_time: Process fills up to this timestamp
        """
        fills_array = self.fills_dict.get(symbol_venue, np.array([]))
        while self.last_processed_fill_idx[symbol_venue] + 1 < len(fills_array):
            fill_data = fills_array[self.last_processed_fill_idx[symbol_venue] + 1]
            fill = parse_fill_data(fill_data)
            if fill.exch_ts > current_time:
                break
            self.last_processed_fill_idx[symbol_venue] += 1
            self.fees[symbol_venue] += fill.commission
            fill.calc_commission_usd(self.commission_px_dict.get(current_time, {}))
            self.fees_usd[symbol_venue] += fill.commission_usd
            self._match_fills_with_inventory(fill, symbol_venue)

    # Update funding income
    def _update_pnl_from_fundings(self, symbol_venue: str, current_time: dt):
        """Process funding income up to current time.

        Args:
            symbol_venue: Symbol to process
            current_time: Process funding up to this timestamp
        """
        fundings_array = self.funding_income_dict.get(symbol_venue, np.array([]))
        while self.last_processed_funding_idx[symbol_venue] + 1 < len(fundings_array):
            funding_data = fundings_array[self.last_processed_funding_idx[symbol_venue] + 1]
            if funding_data[0] > current_time:
                break
            self.last_processed_funding_idx[symbol_venue] += 1
            self.funding_income[symbol_venue] += funding_data[2]


    def _calculate_position_info_at_bar(self, symbol_venue: str, close_price: float, current_time: dt, calculate_position_age: bool = False) -> Tuple[float, float, int]:
        """Calculate unrealized PnL and position info at a specific price/time.

        Args:
            symbol_venue: Symbol to calculate
            close_price: Current market price
            current_time: Current timestamp
            calculate_position_age: Whether to calculate weighted average position age

        Returns:
            Tuple of (unrealized_pnl, net_position, position_age_days)
        """
        unrealized_pnl = np.sum((close_price - self.inventory_case[Side.BUY][symbol_venue]['price']) * self.inventory_case[Side.BUY][symbol_venue]['quantity'])
        unrealized_pnl += np.sum((self.inventory_case[Side.SELL][symbol_venue]['price'] - close_price) * self.inventory_case[Side.SELL][symbol_venue]['quantity'])
        current_long_position = np.sum(self.inventory_case[Side.BUY][symbol_venue]['quantity'])
        current_short_position = np.sum(self.inventory_case[Side.SELL][symbol_venue]['quantity'])
        current_position = current_long_position - current_short_position
        if current_long_position != 0 and current_short_position != 0:
            logger.warning(f"seeing {symbol_venue=} {current_long_position=} and {current_short_position=} at the same time")

        if not calculate_position_age:
            return unrealized_pnl, current_position, 0

        if current_position > 0:
            weighted_sum = sum(trade['quantity'] * pd.Timestamp(trade['ts']).timestamp() for trade in self.inventory_case[Side.BUY][symbol_venue])
            position_age = to_datetime(int(weighted_sum / current_long_position), unit='s')
        elif current_position < 0:
            weighted_sum = sum(trade['quantity'] * pd.Timestamp(trade['ts']).timestamp() for trade in self.inventory_case[Side.SELL][symbol_venue])
            position_age = to_datetime(int(weighted_sum / current_short_position), unit='s')
        else:
            position_age = pd.NaT
        return unrealized_pnl, current_position, (current_time - position_age).days

    def calculate_pnl_performance_metrics(self, start_dt: dt, record_position_age: Optional[str] = None, update: bool = False) -> pd.DataFrame:
        """Main PnL calculation loop processing all bars and fills chronologically.

        Args:
            start_dt: Start datetime for calculations
            record_position_age: Position age calculation mode ('sim', 'latest', or None)
            update: If True, use cached data and only process new timestamps

        Returns:
            DataFrame with detailed PnL metrics for each symbol and timestamp
        """
        calculate_position_age = record_position_age == 'sim'

        if update:
            logger.info("Using cached state for incremental PnL calculation")
            if self._cached_pnl_list is not None:
                pnl_list = self._cached_pnl_list.copy()
            else:
                logger.info("No cached state found for pnl_list")
                pnl_list = []
            new_pnl_count = 0
        else:
            logger.info(f"Starting full PnL calculation from scratch  {update=}")
            pnl_list = []
            self.fills_pnl_list = []
            self.entry_fills_pnl_list = []
            new_pnl_count = 0

        for symbol_venue in sorted(self.symbol_venues):
            bars_array = self.bars_dict.get(symbol_venue, np.array([]))
            symbol_record_cnt = 0
            logger.info(f"Calculating pnl on {symbol_venue} for {len(bars_array)} prices")
            for idx, bars in enumerate(bars_array):
                current_time, current_date, close_price = bars[0], bars[1], bars[2]

                # Skip bars we've already processed (for update mode)
                if update and idx <= self.last_processed_bars_idx.get(symbol_venue, -1):
                    continue

                # Process fills and funding up to this time
                self._update_pnl_from_new_fills(symbol_venue, current_time)
                self._update_pnl_from_fundings(symbol_venue, current_time)

                # Calculate position age only for the last bar if requested
                if idx == len(bars_array) - 1 and record_position_age == 'latest':
                    calculate_position_age = True

                # Calculate position info
                unrealized_pnl, current_position, position_age = self._calculate_position_info_at_bar(
                    symbol_venue, close_price, current_time, calculate_position_age
                )

                # Create PnL record
                new_record = {
                    'ts': current_time,
                    'symbol_venue': symbol_venue,
                    'date': current_date,
                    'realized_pnl': self.realized_pnl[symbol_venue],
                    'unrealized_pnl': unrealized_pnl,
                    'qty': current_position,
                    'fees_cumulative': self.fees[symbol_venue],
                    'fees_usd_cumulative': self.fees_usd[symbol_venue],
                    'funding_income_cumulative': self.funding_income[symbol_venue],
                    'mark_price': close_price,
                    'position_age': position_age,
                    'fill_cnt_cumulative': self.fill_cnt[symbol_venue],
                    'dollars_traded_cumulative': self.dollars_traded[symbol_venue],
                    'dollars_buy_cumulative': self.dollars_case[Side.BUY][symbol_venue],
                    'dollars_sell_cumulative': self.dollars_case[Side.SELL][symbol_venue],
                }

                pnl_list.append(new_record)
                self.last_processed_bars_idx[symbol_venue] = idx
                symbol_record_cnt += 1

            logger.info(f"Created {symbol_record_cnt} records for {symbol_venue}")
            new_pnl_count += symbol_record_cnt

        # Cache the current state if we processed any new data
        if new_pnl_count > 0:
            total_final_realized = sum(self.realized_pnl.values())
            total_final_fills = sum(self.fill_cnt.values())
            logger.info(f"Final state - Realized PnL: {total_final_realized:.2f}, Total Cnt: {total_final_fills:.2f}")

            self._cached_pnl_list = pnl_list.copy()
            logger.info(f"Added {new_pnl_count} new PnL records, total: {len(pnl_list)}")
        elif update:
            logger.info(f"No new data to process, returning {len(pnl_list)} cached PnL records")

        # Create DataFrames (these now contain all data - cached + new)
        self.fills_pnl_df = pd.DataFrame(self.fills_pnl_list)
        self.entry_fills_pnl_df = pd.DataFrame(self.entry_fills_pnl_list)

        # Filter entry fills DataFrame
        if not self.entry_fills_pnl_df.empty:
            self.entry_fills_pnl_df = self.entry_fills_pnl_df.loc[self.entry_fills_pnl_df['ts'] > start_dt]

        # Sort and process fills DataFrame
        if not self.fills_pnl_df.empty:
            self.fills_pnl_df = self.fills_pnl_df.sort_values(['ts', 'symbol']).reset_index(drop=True)
            self.fills_pnl_df = make_date(self.fills_pnl_df)

        if update and new_pnl_count == 0:
            # No new data, return the last processed result if available
            if self._last_processed_pnl_df is not None:
                logger.info("No new data")
                return self._last_processed_pnl_df
            else:
                logger.info("No cached processed DataFrame, running full processing")

        # Process the PnL DataFrame
        pnl_df = self._process_pnl_dataframe(pd.DataFrame(pnl_list), start_dt)

        # Cache the processed result
        if not update or new_pnl_count > 0:
            self._last_processed_pnl_df = pnl_df.copy()

        return pnl_df

    def _process_pnl_dataframe(self, pnl_df: pd.DataFrame, start_dt: dt) -> pd.DataFrame:
        """Ultra-fast processing using NumPy operations"""
        pnl_df = set_index(pnl_df)
        pnl_df = pnl_df.sort_index()
        max_ts = pnl_df.index.get_level_values('ts').max()
        pnl_df['notional'] = pnl_df['qty'] * pnl_df['mark_price']
        pnl_df = round_dust_position(df=pnl_df, round_dt=max_ts, adjust_pos_age=True)

        # Pre-allocate arrays for results
        cumulative_cols = ["fees", "fees_usd", "funding_income", "fill_cnt", "dollars_traded", "dollars_buy", "dollars_sell"]

        # Calculate diffs using vectorized operations within each group
        def calculate_diffs_for_group(group_df):
            """Process a single symbol group with NumPy"""
            result = {}

            # Calculate diffs for cumulative columns
            for col in cumulative_cols:
                values = group_df[f'{col}_cumulative'].values
                diffs = np.concatenate([[values[0]], np.diff(values)])  # First value is original, rest are diffs
                result[f'{col}_diff'] = diffs

            # Handle realized PnL
            realized_values = group_df['realized_pnl'].values
            realized_diffs = np.concatenate([[realized_values[0]], np.diff(realized_values)])
            result['realized_pnl_diff'] = realized_diffs

            # Handle unrealized PnL
            unrealized_values = group_df['unrealized_pnl'].values
            if start_dt.day != 1:
                unrealized_diffs = np.concatenate([[unrealized_values[0]], np.diff(unrealized_values)])
            else:
                unrealized_diffs = np.concatenate([[0], np.diff(unrealized_values)])
            result['unrealized_pnl_diff'] = unrealized_diffs

            return pd.DataFrame(result, index=group_df.index)

        # Apply to all groups at once
        diff_results = pnl_df.groupby('symbol_venue', group_keys=False).apply(calculate_diffs_for_group)
        pnl_df = pnl_df.join(diff_results)

        # Calculate daily cumulative sums in one operation
        diff_col_names = [f'{col}_diff' for col in cumulative_cols] + ['realized_pnl_diff', 'unrealized_pnl_diff']
        daily_cumsum = pnl_df.groupby(['symbol_venue', 'date'])[diff_col_names].cumsum()

        # Rename to _daily
        daily_rename_map = {
            'realized_pnl_diff': 'realized_daily',
            'unrealized_pnl_diff': 'unrealized_daily'
        }
        for col in cumulative_cols:
            daily_rename_map[f'{col}_diff'] = f'{col}_daily'

        daily_cumsum = daily_cumsum.rename(columns=daily_rename_map)
        pnl_df = pnl_df.join(daily_cumsum)

        # Final calculations
        pnl_df['total_pnl_daily'] = pnl_df['realized_daily'] + pnl_df['unrealized_daily'] - pnl_df['fees_usd_daily'] + pnl_df['funding_income_daily']
        pnl_df['total_pnl_cumulative'] = pnl_df['realized_pnl'] + pnl_df['unrealized_pnl'] - pnl_df['fees_usd_cumulative'] + pnl_df['funding_income_cumulative']

        pnl_df['notional_abs'] = pnl_df['notional'].abs()
        pnl_df['avg_abs_notional_daily'] = pnl_df.groupby(['symbol_venue', 'date'])['notional_abs'].transform(lambda x: x.expanding().mean())

        pnl_df['daily_return'] = pnl_df['total_pnl_daily'] / pnl_df['avg_abs_notional_daily']
        pnl_df['unrealized_return'] = pnl_df['unrealized_pnl'] / pnl_df['avg_abs_notional_daily']

        return pnl_df[PNL_DF_COLUMNS]

    def aggregate_pnl_timeslice(self, pnl_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate PnL data by timestamp across all symbols.

        Args:
            pnl_df: Symbol-level PnL data

        Returns:
            Portfolio-level PnL aggregated by timestamp
        """
        # aggregate into ts level, which is what pnl file stored from simulate
        aggregated_pnl_df = pnl_df.groupby('ts').agg(
            pnl=('total_pnl_cumulative', 'sum'),
            long=('notional', lambda x: x[x > 0].sum()),
            short=('notional', lambda x: x[x < 0].sum()),
            traded_long=('dollars_buy_cumulative', 'sum'),
            traded_short=('dollars_sell_cumulative', 'sum'),
            fees_usd=('fees_usd_cumulative', 'sum'),
            funding_income=('funding_income_cumulative', 'sum'),
        ).sort_index().reset_index()
        aggregated_pnl_df['traded_short'] = -aggregated_pnl_df['traded_short']
        for col in ['traded_long', 'traded_short']:
            aggregated_pnl_df[col] = aggregated_pnl_df[col].diff().fillna(aggregated_pnl_df[col])
        return aggregated_pnl_df

    def aggregate_pnl_performance_metrics(self, pnl_df: pd.DataFrame) -> pd.DataFrame:
        """Fast aggregation of PnL to daily level with comprehensive performance metrics."""

        # Step 1: Get daily last pnl record for each symbol_venue (vectorized)
        daily_pnl_df = pnl_df.groupby(['date', 'symbol_venue'], sort=False).last().reset_index()

        # Step 2: Aggregate to daily level - combine positive/negative notional in single pass
        def fast_aggregate(group):
            notional = group['notional']
            return pd.Series({
                'long': notional[notional > 0].sum(),
                'short': notional[notional < 0].sum(),
                'realized_daily': group['realized_daily'].sum(),
                'unrealized_daily': group['unrealized_daily'].sum(),
                'total_pnl_daily': group['total_pnl_daily'].sum(),
                'fees_daily': group['fees_daily'].sum(),
                'fees_usd_daily': group['fees_usd_daily'].sum(),
                'funding_income_daily': group['funding_income_daily'].sum(),
                'fill_cnt_daily': group['fill_cnt_daily'].sum(),
                'dollars_traded_daily': group['dollars_traded_daily'].sum(),
                'dollars_buy_daily': group['dollars_buy_daily'].sum(),
                'dollars_sell_daily': group['dollars_sell_daily'].sum(),
            })

        daily_pnl_df = daily_pnl_df.groupby('date', sort=False).apply(fast_aggregate).reset_index()

        # Step 3: Pre-calculate all cumulative sums at once (vectorized)
        cumsum_cols = ['realized_daily', 'unrealized_daily', 'fees_usd_daily', 'funding_income_daily']

        # Create all base calculations
        daily_pnl_df['bias'] = daily_pnl_df['long'] + daily_pnl_df['short']
        daily_pnl_df['notional_abs_daily'] = daily_pnl_df['long'] - daily_pnl_df['short']

        # Lifetime calculations (simple cumsum)
        daily_pnl_df['total_pnl_lifetime'] = (
                daily_pnl_df['realized_daily'].cumsum() +
                daily_pnl_df['unrealized_daily'].cumsum() -
                daily_pnl_df['fees_usd_daily'].cumsum() +
                daily_pnl_df['funding_income_daily'].cumsum()
        )
        daily_pnl_df['notional_abs_lifetime'] = daily_pnl_df['notional_abs_daily'].expanding().mean()

        # Step 4: MTD/YTD calculations - do both in single loop
        daily_pnl_df = daily_pnl_df.set_index('date').sort_index()

        # Vectorized period calculations
        freq_configs = {
            'mtd': 'MS',
            'ytd': 'YS'
        }

        for freq_name, freq_code in freq_configs.items():
            # Create period grouper
            period_groups = daily_pnl_df.groupby(pd.Grouper(freq=freq_code, label='left', closed='left'))

            # Calculate period cumulative sums efficiently
            daily_pnl_df[f'total_pnl_{freq_name}'] = (
                    period_groups['realized_daily'].cumsum() +
                    period_groups['unrealized_daily'].cumsum() -
                    period_groups['fees_usd_daily'].cumsum() +
                    period_groups['funding_income_daily'].cumsum()
            )

            # Period average notional
            daily_pnl_df[f'notional_abs_{freq_name}'] = period_groups['notional_abs_daily'].transform(lambda x: x.expanding().mean())

        # Step 5: Calculate all returns at once (vectorized)
        for freq in ['daily', 'mtd', 'ytd', 'lifetime']:
            daily_pnl_df[f'return_{freq}'] = daily_pnl_df[f'total_pnl_{freq}'] / daily_pnl_df[f'notional_abs_{freq}']

        # Step 6: Risk and Sharpe (vectorized)
        daily_returns = daily_pnl_df['return_daily']
        daily_pnl_df['risk_lifetime'] = daily_returns.expanding().std()
        daily_pnl_df['sharpe_lifetime'] = daily_returns.expanding().mean() / daily_pnl_df['risk_lifetime'] * np.sqrt(365)

        # Step 7: Lifetime cumulative columns (vectorized)
        lifetime_cols = ["fees", "fees_usd", "funding_income", "fill_cnt", "dollars_traded", "dollars_buy", "dollars_sell"]
        for col in lifetime_cols:
            daily_pnl_df[f'{col}_lifetime'] = daily_pnl_df[f'{col}_daily'].cumsum()

        # Validation (vectorized)
        assert (daily_pnl_df['long'] >= 0).all(), 'daily_pnl_df got negative long positions'
        assert (daily_pnl_df['short'] <= 0).all(), 'daily_pnl_df got positive short positions'

        return daily_pnl_df[DAILY_PNL_DF_COLUMNS]

    def dump_breakdown_file(self, breakdown_file: str):
        """Write fill-level PnL breakdown to CSV file.

        Args:
            breakdown_file: Path to output CSV file
        """
        logger.info(f"Writing breakdown file {breakdown_file}")
        self.fills_pnl_df.to_csv(breakdown_file)
