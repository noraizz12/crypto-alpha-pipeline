import logging
from datetime import datetime as dt, timezone, date, time
from multiprocessing import Pool
from typing import Optional, Literal, List, Dict, Set, Tuple

import pandas as pd
from lib.util.dataframes import set_index

from lib.data import load_binance_fills, load_oms_fills
from lib.data.dataloader import DataLoader
from lib.data.live_bars import LiveBars
from lib.data.loaders import load_funding_income
from lib.pnl_new.security_pnl import SecurityPnl
from lib.pnl_new.pnl_util import aggregate_to_daily, calc_pnl_returns
from lib.util import DirectoryManager, dir_manager, TRADING_START_DT, log_and_raise, shrink_floats, TARDIS_EXCHANGE
from lib.util.dataframes import concat, make_symbol_venue, to_datetime
from lib.util.time_util import today_date, date_to_start_dt, date_to_end_dt, yesterday_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BINANCE_FILLS_COLUMN_MAPPING = {'orderId': 'oid', 'commissionAsset': 'commission_asset', 'realizedPnl': 'realized_pnl'}
STANDARDIZED_FILLS_COLUMNS = ['date', 'ts', 'symbol', 'symbol_venue', 'fill_source', 'side', 'fill_px', 'fill_qty', 'fill_dollars', 'commission', 'commission_asset', 'realized_pnl']


class Pnl:
    """Calculate PnL from trading fills.

    This class manages the end-to-end PnL calculation process, including loading
    fills, bars, positions, and funding data, then calculating various PnL metrics
    using weighted average cost basis accounting.

    Attributes:
        start_dt (datetime): Start datetime for PnL calculation
        end_dt (datetime): End datetime for PnL calculation
        fills_df (pd.DataFrame): DataFrame containing fill data
        bars_df (pd.DataFrame): DataFrame containing bar/price data
        pnl_df (pd.DataFrame): Calculated PnL results by symbol and timestamp

    Args:
        config: Configuration dictionary
        fills_source: Source of fills data ('binance', 'oms', or 'simulation')
        start: Start datetime for PnL calculation
        end: End datetime for PnL calculation
        pnl_dir_manager: Directory manager for data access
        symbol_venues: List of symbol_venues to calculate PnL for
        pnl_times: Set of specific times to calculate PnL at
        initial_positions: Initial positions for each symbol_venue
        fills_df: Pre-loaded fills DataFrame (optional)
        bars_df: Pre-loaded bars DataFrame (optional)
        fundings_df: Pre-loaded funding income DataFrame (optional)
    """
    def __init__(
            self,
            config: dict,
            fills_source: Optional[Literal["binance", "oms", "simulation"]] = None,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
            pnl_dir_manager: DirectoryManager = dir_manager,
            symbol_venues: Optional[List[str]] = None,
            pnl_times: Optional[Set[time]] = None,
            initial_positions: Optional[Dict[str, Dict[str, float]]] = None,
            fills_df: Optional[pd.DataFrame] = None,
            bars_df: Optional[pd.DataFrame] = None,
            fundings_df: Optional[pd.DataFrame] = None
    ):
        logger.info(f"Creating Pnl from {start} to {end} {fills_source=} ")
        self.config = config
        self.fills_source = fills_source
        self.simulation = self.fills_source == 'simulation'
        self.start_dt = start if start is not None else TRADING_START_DT
        self.end_dt = end if end is not None else dt.now(timezone.utc)
        assert self.start_dt <= self.end_dt
        self.as_of = None
        self.pnl_times = pnl_times
        self.initial_positions = initial_positions or {}

        self.dir_manager = pnl_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)

        if self.fills_source == "oms":
            self.fills_dir = self.dir_manager.FILLS_DIR
        elif self.fills_source == "binance":
            self.fills_dir = self.dir_manager.BINANCE_FILLS_DIR

        # Load fills only if not provided
        if fills_df is None:
            self.fills_df = self._load_fills(update=False)
        else:
            self.fills_df = fills_df

        # Identify symbol_venues from fills and initial_positions, or use provided list
        if symbol_venues is not None:
            self.symbol_venues = symbol_venues
        else:
            self.symbol_venues = self._identify_symbol_venues()

        # Add commission asset symbol_venues
        commission_symbol_venues = self._get_commission_asset_symbol_venues(self.fills_df)
        for sv in commission_symbol_venues:
            if sv not in self.symbol_venues:
                self.symbol_venues.append(sv)

        logger.info(f"Loading prices for {len(self.symbol_venues)} symbol_venues")

        # Load bars only if not provided
        if bars_df is None:
            self.live_bars_manager = LiveBars(universe=self.symbol_venues)
            self.bars_df = self._load_prices(start_date=self.start_dt.date(), end_date=self.end_dt.date())
        else:
            self.bars_df = bars_df

        # Compute commissions adjusted by commission asset mark prices
        self.fills_df = compute_commissions(self.fills_df, self.bars_df)

        # Load fundings only if not provided
        if fundings_df is None:
            self.fundings_df = self._load_fundings_df(update=False)
        else:
            self.fundings_df = fundings_df

        # Validate all required data is present
        self._validate_data()

        self.pnl_df = None
        self.agg_pnl_df = None
        self.daily_agg_pnl_df = None

    def update_data(self) -> None:
        """Update fills, bars, and fundings with latest intraday data.

        Reloads data from end_dt to current time and appends to existing data.
        Only works for intraday updates (end_dt must be today).

        Raises:
            Exception: If end_dt is not today (can only update intraday)

        Note:
            After calling this, you must recalculate PnL by calling calculate()
        """
        now = dt.now(timezone.utc)

        # Validate we can only update intraday
        if self.end_dt.date() < today_date():
            raise log_and_raise(
                f"Cannot update data: end_dt {self.end_dt.date()} is prior to today {today_date()}. "
                f"Can only update intraday data."
            )

        old_end_dt = self.end_dt
        logger.info(f"Updating data from {old_end_dt} to {now}")

        # Update fills
        logger.info("Updating fills...")
        old_fills_count = len(self.fills_df) if self.fills_df is not None else 0
        self.fills_df = self._load_fills(update=True)
        new_fills_count = len(self.fills_df) - old_fills_count if self.fills_df is not None else 0

        if new_fills_count > 0:
            new_fills_dollars = self.fills_df.iloc[-new_fills_count:]['fill_dollars'].abs().sum()
            new_commissions = self.fills_df.iloc[-new_fills_count:]['commission'].sum()
            logger.info(f"Loaded {new_fills_count} new fills (${new_fills_dollars:,.2f} notional, ${new_commissions:.2f} commissions)")

        # Update bars
        logger.info("Updating bars...")
        if self.live_bars_manager is not None:
            new_bars_df = self.live_bars_manager.load_live_bars(start_dt=old_end_dt, end_dt=now)

            if new_bars_df is not None and len(new_bars_df) > 0:
                logger.info(f"Loaded {len(new_bars_df)} new bar rows")
                self.bars_df = concat([self.bars_df, new_bars_df])
                self.as_of = self.bars_df.index.get_level_values('ts').max()
                logger.info(f"Data as_of: {self.as_of}")
            else:
                logger.info("No new bars found")
        else:
            logger.warning("No live_bars_manager, skipping bars update")

        # Compute commissions only on new fills (handles commission_asset logic)
        if new_fills_count > 0:
            logger.info("Computing commissions on new fills...")
            new_fills_only = self.fills_df.iloc[-new_fills_count:].copy()
            new_fills_adjusted = compute_commissions(new_fills_only, self.bars_df)
            # Replace the new fills with adjusted ones
            self.fills_df.iloc[-new_fills_count:] = new_fills_adjusted.values

        # Update fundings
        logger.info("Updating fundings...")
        old_fundings_count = len(self.fundings_df) if self.fundings_df is not None else 0
        self.fundings_df = self._load_fundings_df(update=True)

        if self.fundings_df is not None:
            new_fundings_count = len(self.fundings_df) - old_fundings_count
            if new_fundings_count > 0:
                new_funding_income = self.fundings_df.iloc[-new_fundings_count:]['funding_income'].sum()
                logger.info(f"Loaded {new_fundings_count} new funding events (${new_funding_income:,.2f} total)")

        # Update end_dt
        self.end_dt = now
        logger.info(f"Data updated: {old_end_dt} -> {self.end_dt}")

        # Clear stale PnL calculations
        if self.pnl_df is not None:
            logger.info("Clearing stale PnL - call calculate() to recompute")
            self.pnl_df = None
            self.agg_pnl_df = None
            self.daily_agg_pnl_df = None


    def _validate_data(self) -> None:
        """Validate that we have all required data for PnL calculation.

        Checks for missing bars and logs each date/symbol combination where fills exist but bars are missing.
        Forward-filling will be handled in SecurityPnl when joining fills with bars.

        Raises:
            Exception: If fills or bars are missing entirely
        """
        if self.fills_df is None or len(self.fills_df) == 0:
            raise log_and_raise("No fills data available for PnL calculation")

        if self.bars_df is None or len(self.bars_df) == 0:
            raise log_and_raise("No bars data available for PnL calculation")

        # Check for symbols that have fills but no bars at all
        bars_symbols = set(self.bars_df.index.get_level_values('symbol_venue').unique())
        identified_symbols = set(self.symbol_venues)
        missing_bars_symbols = identified_symbols - bars_symbols

        if missing_bars_symbols:
            logger.warning(f"Found {len(missing_bars_symbols)} symbols with fills but no bars at all: {sorted(list(missing_bars_symbols))[:10]}")
            for symbol in sorted(list(missing_bars_symbols))[:10]:
                symbol_fills = self.fills_df[self.fills_df['symbol_venue'] == symbol]
                if len(symbol_fills) > 0:
                    min_date = symbol_fills['date'].min()
                    max_date = symbol_fills['date'].max()
                    fill_count = len(symbol_fills)
                    total_notional = abs(symbol_fills['fill_qty'] * symbol_fills['fill_px']).sum()
                    logger.warning(
                        f"  {symbol}: {fill_count} fills from {min_date} to {max_date}, "
                        f"${total_notional:,.0f} notional - NO BARS FOUND, will use forward-fill"
                    )

        # Check for date/symbol combinations where we have fills but missing bars
        fill_dates_by_symbol = self.fills_df.groupby('symbol_venue')['date'].apply(set).to_dict()

        missing_count = 0
        for symbol in self.symbol_venues:
            if symbol not in bars_symbols:
                continue  # Already logged above

            # Get dates where we have bars for this symbol
            symbol_bars = self.bars_df.loc[self.bars_df.index.get_level_values('symbol_venue') == symbol]
            bar_dates = set(symbol_bars.index.get_level_values('ts').normalize())

            # Get dates where we have fills for this symbol
            fill_dates = fill_dates_by_symbol.get(symbol, set())

            # Find dates with fills but no bars
            missing_dates = fill_dates - bar_dates

            if missing_dates:
                missing_count += len(missing_dates)
                for missing_date in sorted(missing_dates):
                    date_fills = self.fills_df[
                        (self.fills_df['symbol_venue'] == symbol) &
                        (self.fills_df['date'] == missing_date)
                    ]
                    date_notional = abs(date_fills['fill_qty'] * date_fills['fill_px']).sum()
                    logger.warning(
                        f"  {symbol} on {missing_date}: {len(date_fills)} fills, "
                        f"${date_notional:,.0f} notional - MISSING BARS, will use forward-fill"
                    )

        if missing_count > 0:
            logger.warning(f"Total: {missing_count} date/symbol combinations with fills but missing bars")

        logger.info(f"Validated: {len(self.fills_df)} fills, {len(self.bars_df)} bars for {len(self.symbol_venues)} symbols")

    @staticmethod
    def _get_commission_asset_symbol_venues(fills_df: Optional[pd.DataFrame]) -> List[str]:
        """Extract commission asset symbol_venues from fills.

        Args:
            fills_df: DataFrame containing fill data with optional commission_asset column

        Returns:
            List of symbol_venues for commission assets
        """
        if fills_df is None or len(fills_df) == 0:
            return []

        if 'commission_asset' not in fills_df.columns:
            return []

        commission_assets = fills_df['commission_asset'].dropna().unique()
        if len(commission_assets) == 0:
            return []

        # Convert commission assets to symbol_venues (e.g., 'BNB' -> 'BNBUSDT_binance-futures')
        commission_symbol_venues = [f"{asset}USDT_{TARDIS_EXCHANGE}" for asset in commission_assets]
        logger.info(f"Found {len(commission_symbol_venues)} commission asset symbol_venues: {commission_symbol_venues}")
        return commission_symbol_venues

    def _identify_symbol_venues(self) -> List[str]:
        """Identify symbol_venues from fills and initial_positions.

        Returns:
            List of symbol_venues that need price data
        """
        symbol_venues_set = set()

        if self.fills_df is not None and len(self.fills_df) > 0:
            fills_symbols = set(self.fills_df['symbol_venue'].unique())
            symbol_venues_set.update(fills_symbols)
            logger.info(f"Found {len(fills_symbols)} symbol_venues in fills")

        if self.initial_positions:
            position_symbol_venues = set(self.initial_positions.keys())
            symbol_venues_set.update(position_symbol_venues)
            logger.info(f"Found {len(position_symbol_venues)} symbol_venues in initial_positions")

        return sorted(list(symbol_venues_set))

    def _load_prices(self, start_date: date, end_date: date) -> pd.DataFrame:
        logger.info(f"Loading prices from {start_date} to {end_date}")
        prebars_df = None
        if start_date < today_date():
            prebar_end_date = min(end_date, yesterday_date())
            prebars_df = self.data_loader.load_prebar_files(start_date, prebar_end_date, symbol_venues=self.symbol_venues, bars_type='tardis_prefer')

        live_bars_df = None
        if end_date == today_date():
            live_bars_df = self.live_bars_manager.load_live_bars(date_to_start_dt(today_date()), date_to_end_dt(today_date()))

        if prebars_df is not None and live_bars_df is not None:
            df = pd.concat([prebars_df, live_bars_df])
        elif prebars_df is not None:
            df = prebars_df
        elif live_bars_df is not None:
            df = live_bars_df
        else:
            raise log_and_raise(f"No prebars or live bars found for {start_date} to {end_date}")

        self.as_of = df.index.get_level_values('ts').max()
        return df

    def _load_fundings_df(self, update: bool = False) -> Optional[pd.DataFrame]:
        """Load funding income data from configured source.

        For file source, loads historical funding income with caching support.
        For simulation source, aggregates funding data from update calls.

        Args:
            update: If True, only load new data since last load (file source only)

        Returns:
            DataFrame with funding income data containing:
                - ts: Funding timestamp
                - symbol: Trading symbol
                - symbol_venue: Symbol with venue
                - funding_income: Income amount (positive = received)

        Raises:
            Exception: If fundings_type is not recognized

        Note:
            File source groups by symbol and time, taking last value
        """
        end_date = today_date() if update and self.real_time else self.end_dt.date()
        start_date = self.start_dt.date() if not update or self.fundings_df is None else self.end_dt.date()

        fundings_df = load_funding_income(start_date=start_date, end_date=end_date, funding_dir=self.dir_manager.FUNDING_INCOME_DIR)
        if fundings_df is None:
            logger.warning(f"Could not load fundings income between {start_date} and {end_date}")
            return None

        fundings_df = fundings_df.groupby(['symbol', 'time'])['funding_income'].last().reset_index()
        fundings_df = make_symbol_venue(fundings_df)
        fundings_df['ts'] = to_datetime(fundings_df['time'], format='ISO8601')
        if update and self.fundings_df is not None:
            #XXX? do i really need to cache?
            cached_fundings_df = self.fundings_df.loc[fundings_df['ts'] < date_to_start_dt(end_date)]
            fundings_df = concat([cached_fundings_df, fundings_df], fast=True).reset_index(drop=True)

        if fundings_df is None:
            logger.warning(f"Could not load fundings income between {start_date} and {end_date}")
        return fundings_df



    def _load_fills(self, update: bool = False, fills_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Load raw fill data from configured source.

        Handles loading from different sources with optional incremental updates
        and caching. For Binance fills, maintains a cache to support efficient
        incremental loading.

        Args:
            update: If True, only load new data since last load (where supported)

        Returns:
            DataFrame with raw fill data, or None if no data available

        Raises:
            Exception: If fills_source is not recognized

        Note:
            For simulation source, returns aggregated fills from update_fill_from_sim_timeslice calls
            For trades source, returns fills set via update_fill_from_trades
        """

        start_date = self.start_dt.date() if not update else self.end_dt.date()
        end_date = today_date() if update else self.end_dt.date()

        logger.info(f"Loading raw fills from {self.fills_source} {start_date} to {end_date} {update=}")

        if self.fills_source == 'binance':
            fills_df = load_binance_fills(start_date=start_date, end_date=end_date, fills_dir=self.fills_dir)
            if fills_df is None:
                return None

            # Filter to only new fills if updating
            if update and self.fills_df is not None:
                old_ts = self.fills_df['ts'].max()
                fills_df = fills_df.loc[fills_df['ts'] > old_ts]

                if len(fills_df) == 0:
                    logger.info("No new fills found")
                    return self.fills_df

                logger.info(f"Processing {len(fills_df)} new fills from {fills_df['ts'].min()} to {fills_df['ts'].max()}")
            else:
                # Only filter if end_dt is before current time (e.g., backtesting)
                # If end_dt is close to current time (real-time monitoring), don't filter
                fills_df = fills_df.loc[fills_df['ts'] <= self.end_dt]

            standardized_fills_df = fills_df.rename(columns=BINANCE_FILLS_COLUMN_MAPPING)
            standardized_fills_df['side'] = standardized_fills_df['side'].str[0]
            # Make fill_qty negative for sells (like OMS fills)
            sell_idx = standardized_fills_df['side'] == 'S'
            standardized_fills_df.loc[sell_idx, 'fill_qty'] = -standardized_fills_df.loc[sell_idx, 'fill_qty']
            standardized_fills_df = make_symbol_venue(standardized_fills_df)
        elif self.fills_source == 'oms':
            fills_df = load_oms_fills(start_date=start_date, end_date=end_date, fills_dir=self.fills_dir)
            standardized_fills_df = fills_df
            sell_idx = standardized_fills_df['side'] == 'S'
            standardized_fills_df.loc[sell_idx, 'fill_qty'] = -standardized_fills_df.loc[sell_idx, 'fill_qty']
            standardized_fills_df = make_symbol_venue(standardized_fills_df)
        elif self.fills_source == 'simulation':
            assert fills_df is not None, "Simulation fills df must be provided for simulation fills source"
            fills_df['commission_asset'] = 'USDT'
            standardized_fills_df = fills_df.rename(columns={'executed_qty': 'fill_qty', 'trade_price': 'fill_px'})
            standardized_fills_df['side'] = 'B'
            standardized_fills_df.loc[standardized_fills_df['fill_qty'] < 0, 'side'] = 'S'
            # we use abs qty in pnl calculator, so need to convert
            sell_idx = standardized_fills_df['side'] == 'S'
            standardized_fills_df.loc[sell_idx, 'fill_qty'] = -standardized_fills_df.loc[sell_idx, 'fill_qty']
        else:
            raise log_and_raise(f"Unknown fill type {self.fills_source}")

        if standardized_fills_df is None:
            raise log_and_raise(f"Could not load {self.fills_source} fills from {self.start_dt} to {self.end_dt}!")

        # Common preprocessing for all fills
        standardized_fills_df['fill_source'] = self.fills_source.upper()

        # Calculate fill_dollars if not already present
        if 'fill_dollars' not in standardized_fills_df.columns:
            standardized_fills_df['fill_dollars'] = standardized_fills_df['fill_px'] * standardized_fills_df['fill_qty']

        standardized_fills_df = standardized_fills_df[STANDARDIZED_FILLS_COLUMNS]
        standardized_fills_df = shrink_floats(standardized_fills_df)
        standardized_fills_df['ts'] = standardized_fills_df['ts'].dt.ceil('1min')

        # Append to existing if updating
        if update and self.fills_df is not None:
            standardized_fills_df = concat([self.fills_df, standardized_fills_df], fast=True).reset_index(drop=True)

        return standardized_fills_df

    @staticmethod
    def _calculate_security_pnl(
        symbol_venue: str,
        start_dt: dt,
        end_dt: dt,
        fills_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        fundings_df: Optional[pd.DataFrame],
        initial_position: Optional[Dict[str, float]],
        pnl_times: Optional[Set[time]]
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """Static method to calculate PnL for a single security (for multiprocessing).

        Args:
            symbol_venue: Symbol and venue identifier
            start_dt: Start datetime
            end_dt: End datetime
            fills_df: Fills for this security
            bars_df: Bars for this security
            fundings_df: Funding payments for this security
            initial_position: Initial position dict

        Returns:
            Tuple of (symbol_venue, pnl_df)
        """
        try:
            sec_pnl = SecurityPnl(
                symbol_venue=symbol_venue,
                start_dt=start_dt,
                end_dt=end_dt,
                fills_df=fills_df,
                bars_df=bars_df,
                fundings_df=fundings_df,
                initial_position=initial_position,
                pnl_times=pnl_times
            )
            pnl_df = sec_pnl.calculate_pnl()
            return symbol_venue, pnl_df
        except Exception as e:
            logger.error(f"Error calculating PnL for {symbol_venue}: {e}")
            return symbol_venue, None

    def calculate_security_pnls(self, pool_size: int = 8) -> Dict[str, pd.DataFrame]:
        """Calculate PnL for all securities in parallel.

        Partitions data by symbol_venue and uses multiprocessing to calculate
        PnL for each security independently.

        Args:
            pool_size: Number of parallel processes to use

        Returns:
            Dictionary mapping symbol_venue -> PnL DataFrame
            Each DataFrame contains columns: ts, qty, cost_basis, value,
            cumulative_fees, cumulative_funding_income
        """
        logger.info(f"Starting parallel PnL calculation with {pool_size} processes")
        logger.info(f"Calculating PnL for {len(self.symbol_venues)} securities")

        # Prepare data partitions for each security
        tasks = []
        skipped_symbols = []
        for symbol_venue in self.symbol_venues:
            # Partition fills
            symbol_fills_df = self.fills_df[self.fills_df['symbol_venue'] == symbol_venue].copy()

            # Partition bars (bars_df has MultiIndex with symbol_venue)
            # Handle case where symbol has no bars
            if symbol_venue not in self.bars_df.index.get_level_values('symbol_venue'):
                # No bars for this symbol - log and skip
                fill_count = len(symbol_fills_df)
                fill_notional = abs(symbol_fills_df['fill_qty'] * symbol_fills_df['fill_px']).sum()
                logger.warning(
                    f"Skipping {symbol_venue} - no bars available for PnL calculation. "
                    f"Unable to mark {fill_count} fills with ${fill_notional:,.2f} notional"
                )
                skipped_symbols.append((symbol_venue, fill_count, fill_notional))
                continue

            symbol_bars_df = self.bars_df.xs(symbol_venue, level='symbol_venue').copy()

            # Partition fundings
            if self.fundings_df is not None and len(self.fundings_df) > 0:
                symbol_fundings_df = self.fundings_df[self.fundings_df['symbol_venue'] == symbol_venue].copy()
            else:
                symbol_fundings_df = None

            # Get initial position for this symbol from initial_positions dict
            initial_pos = self.initial_positions.get(
                symbol_venue,
                {'qty': 0.0, 'cost_basis': 0.0, 'value': 0.0}
            )

            tasks.append((
                symbol_venue,
                self.start_dt,
                self.end_dt,
                symbol_fills_df,
                symbol_bars_df,
                symbol_fundings_df,
                initial_pos,
                self.pnl_times
            ))

        # Execute in parallel using Pool.starmap
        results_dict = {}
        pool = Pool(processes=pool_size)
        for symbol_venue, pnl_df in pool.starmap(self._calculate_security_pnl, tasks):
            if pnl_df is not None:
                results_dict[symbol_venue] = pnl_df
            else:
                logger.warning(f"No PnL calculated for {symbol_venue}")
        pool.close()
        pool.join()

        # Log summary of skipped symbols
        if skipped_symbols:
            total_skipped_fills = sum(count for _, count, _ in skipped_symbols)
            total_skipped_notional = sum(notional for _, _, notional in skipped_symbols)
            logger.warning(
                f"Skipped {len(skipped_symbols)} symbols due to missing bars. "
                f"Total unmarked fills: {total_skipped_fills}, "
                f"Total unmarked notional: ${total_skipped_notional:,.2f}"
            )

        logger.info(f"Completed PnL calculation for {len(results_dict)}/{len(self.symbol_venues)} securities")
        return results_dict

    def calculate(self, pool_size: int = 8) -> pd.DataFrame:
        """Calculate PnL for all securities and aggregate into single DataFrame.

        Args:
            pool_size: Number of parallel processes to use

        Returns:
            Aggregated PnL DataFrame with MultiIndex (ts, symbol_venue)
        """

        # Calculate PnL for each security in parallel
        self.sec_pnls = self.calculate_security_pnls(pool_size=pool_size)

        # Aggregate all security PnL dataframes into single dataframe
        pnl_dfs = []
        for symbol_venue, pnl_df in self.sec_pnls.items():
            pnl_df['symbol_venue'] = symbol_venue
            pnl_df = set_index(pnl_df, ['ts', 'symbol_venue'])
            pnl_dfs.append(pnl_df)

        # Concatenate all dataframes
        self.pnl_df = pd.concat(pnl_dfs, ignore_index=False)

        logger.info(f"Aggregated PnL for {len(self.sec_pnls)} securities into single DataFrame with {len(self.pnl_df)} rows")

        return self.pnl_df

    def aggregate_pnl(self) -> pd.DataFrame:
        """Aggregate PnL across all securities into single row per timestamp.

        Stores result in self.agg_pnl_df and returns it.

        Returns:
            DataFrame with aggregated PnL metrics by timestamp:
                - pnl_gross: Incremental gross PnL
                - pnl_gross_cum: Cumulative gross PnL
                - pnl_net: Incremental net PnL
                - pnl_net_cum: Cumulative net PnL
                - commission: Total commissions
                - commission_cum: Cumulative commissions
                - long_value: Total long position value
                - short_value: Total short position value
        """
        if self.pnl_df is None:
            raise log_and_raise("Must call calculate() before aggregate_pnl()")

        # Reset index to group by timestamp
        pnl_reset_df = self.pnl_df.reset_index()

        # Aggregate by timestamp
        agg_df = pnl_reset_df.groupby('ts').agg({
            'pnl_gross': 'sum',         # Sum gross PnL across securities
            'pnl_gross_cum': 'sum',     # Sum cumulative gross PnL
            'pnl_net': 'sum',           # Sum net PnL
            'pnl_net_cum': 'sum',       # Sum cumulative net PnL
            'commission': 'sum',        # Sum commissions
            'commission_cum': 'sum',    # Sum cumulative commissions
            'funding_income': 'sum',    # Sum funding income
            'funding_income_cum': 'sum', # Sum cumulative funding income
            'realized_pnl': 'sum',      # Sum realized PnL
            'realized_pnl_cum': 'sum',  # Sum cumulative realized PnL
            'unrealized_pnl': 'sum',    # Sum unrealized PnL
            'unrealized_pnl_cum': 'sum', # Sum cumulative unrealized PnL
            'notional': 'sum',          # Sum notional across securities
            'abs_dollars_cum': 'sum',   # Sum cumulative traded dollars
            'fill_count': 'sum'         # Sum fill count across securities
        }).reset_index()

        logger.info(f"Aggregated PnL: fill_count sum = {agg_df['fill_count'].sum()}, max = {agg_df['fill_count'].max()}")

        # Calculate long and short position values
        # Long positions have positive qty (position > 0), short have negative qty (position < 0)
        # Use absolute values for both since we want dollar amounts
        pnl_reset_df['long_value'] = pnl_reset_df['position'].where(pnl_reset_df['qty'] > 0, 0).abs()
        pnl_reset_df['short_value'] = pnl_reset_df['position'].where(pnl_reset_df['qty'] < 0, 0).abs()

        position_agg_df = pnl_reset_df.groupby('ts').agg({
            'long_value': 'sum',
            'short_value': 'sum'
        }).reset_index()

        # Merge position aggregates
        agg_df = agg_df.merge(position_agg_df, on='ts')

        # Set index (just ts for portfolio-level data)
        agg_df = agg_df.set_index('ts')

        # Calculate returns at portfolio level for gross, net, realized, and unrealized PnL
        for pnl_col in ['pnl_gross', 'pnl_net', 'realized_pnl_cum', 'unrealized_pnl']:
            agg_df = calc_pnl_returns(agg_df, pnl_col=pnl_col)

        logger.info(f"Aggregated PnL across all securities: {len(agg_df)} timestamps")

        self.agg_pnl_df = agg_df
        return self.agg_pnl_df

    def calc_daily_pnl(self) -> pd.DataFrame:
        """Calculate daily aggregated PnL from intraday aggregate PnL.

        Uses aggregate_to_daily() from pnl_util to convert intraday timestamps to daily,
        then calculates returns on the daily data.
        Stores result in self.daily_agg_pnl_df and returns it.

        Returns:
            DataFrame with daily aggregated PnL metrics indexed by date

        Raises:
            Exception: If aggregate_pnl() hasn't been called yet
        """
        if self.agg_pnl_df is None:
            raise log_and_raise("Must call aggregate_pnl() before calc_daily_pnl()")

        # Aggregate to daily (returns are already correctly summed in aggregate_to_daily)
        self.daily_agg_pnl_df = aggregate_to_daily(self.agg_pnl_df)

        logger.info(f"Calculated daily PnL: {len(self.daily_agg_pnl_df)} days")

        return self.daily_agg_pnl_df


def compute_commissions(fills_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Compute commissions adjusted by commission asset mark price.

    If commission_asset column exists in fills_df, this function:
    1. Converts commission_asset to symbol_venue format (e.g., 'BNB' -> 'BNBUSDT_binance-futures')
    2. Joins mark prices from bars_df for commission assets
    3. Overwrites the commission column with commission * commission_asset_mark_price

    Args:
        fills_df: DataFrame containing fill data with optional commission_asset column
        bars_df: DataFrame with MultiIndex (ts, symbol_venue) containing close_mid prices

    Returns:
        Updated fills_df with adjusted commissions
    """
    if 'commission_asset' not in fills_df.columns:
        logger.info("No commission_asset column in fills_df, returning unchanged")
        return fills_df

    # Convert commission_asset in place: 'BNB' -> 'BNBUSDT_binance-futures'
    fills_df['commission_asset'] = fills_df['commission_asset'] + f"USDT_{TARDIS_EXCHANGE}"

    # Extract mark prices from bars_df (close_mid is the mid price field)
    bars_prices_df = bars_df.reset_index()[['ts', 'symbol_venue', 'close_mid']].copy()
    bars_prices_df = bars_prices_df.rename(columns={'close_mid': 'commission_asset_mark_price', 'symbol_venue': 'commission_asset'})

    # Merge fills with commission asset prices
    fills_with_commission_prices_df = fills_df.merge(
        bars_prices_df,
        on=['ts', 'commission_asset'],
        how='left'
    )

    # Update commission column: commission * commission_asset_mark_price
    # Only update rows where we have a commission asset and mark price
    mask = fills_with_commission_prices_df['commission_asset_mark_price'].notna()
    fills_with_commission_prices_df.loc[mask, 'commission'] = (
        fills_with_commission_prices_df.loc[mask, 'commission'] *
        fills_with_commission_prices_df.loc[mask, 'commission_asset_mark_price']
    )

    # Log adjustment statistics
    adjusted_count = mask.sum()
    logger.info(f"Adjusted {adjusted_count}/{len(fills_df)} fills with commission asset mark prices")

    # Clean up temporary columns
    fills_with_commission_prices_df = fills_with_commission_prices_df.drop(columns=['commission_asset_mark_price'])

    return fills_with_commission_prices_df






