"""Simulation-only PnL calculator using FIFO accounting methodology."""
import logging
from datetime import datetime as dt, timezone, date
from typing import Optional, List

import pandas as pd

from lib.data.dataloader import DataLoader
from lib.pnl.fill_pnl_symbol import CalcMultiSymbolFillPnl
from lib.util import DirectoryManager, dir_manager, TRADING_START_DT, log_and_raise, shrink_floats
from .pnl_util import calculate_top_drawdowns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STANDARDIZED_FILLS_COLUMNS = ['date', 'ts', 'symbol', 'symbol_venue', 'fill_source', 'side', 'fill_px', 'fill_qty', 'commission', 'commission_asset']


class FillPnl:
    """Calculate FIFO (First-In-First-Out) PnL from simulation fills.

    This class manages the end-to-end PnL calculation process for simulation,
    including loading fills, bars, positions, and funding data, then calculating
    various PnL metrics using FIFO accounting methodology.

    Attributes:
        start_dt (datetime): Start datetime for PnL calculation
        end_dt (datetime): End datetime for PnL calculation
        fills_df (pd.DataFrame): DataFrame containing fill data
        bars_df (pd.DataFrame): DataFrame containing bar/price data
        positions_df (pd.DataFrame): DataFrame containing position data
        pnl_df (pd.DataFrame): Calculated PnL results by symbol and timestamp
        ts_pnl_df (pd.DataFrame): PnL aggregated by timestamp
        daily_pnl_df (pd.DataFrame): PnL aggregated by day
        calculator (CalcMultiSymbolFillPnl): Core PnL calculation engine
    """
    def __init__(
            self,
            config: dict,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
            pnl_dir_manager: DirectoryManager = dir_manager,
            symbol_venues: Optional[List[str]] = None
    ):
        logger.info(f"Creating FillPnl for simulation from {start} to {end}")
        self.config = config
        self.start_dt = start if start is not None else TRADING_START_DT
        self.end_dt = end if end is not None else dt.now(timezone.utc)
        assert self.start_dt <= self.end_dt

        self.symbol_venues = symbol_venues

        self.raw_fills_df = None
        self.fills_df = None

        self.initialize_sim_positions = False
        self.existing_sim_positions_df = None

        self.positions_df = None
        self.fundings_df = None
        self.pnl_df = None
        self.ts_pnl_df = None
        self.daily_pnl_df = None

        self.dir_manager = pnl_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)

        self.calculator = CalcMultiSymbolFillPnl(start, end)

        self.bars_df = None
        self.as_of = None
        self._load_prices(start_date=self.start_dt.date(), end_date=self.end_dt.date())


    def _load_prices(self, start_date: date, end_date: date) -> None:
        """Load price data for simulation PnL calculation.

        Args:
            start_date: Start date for price data
            end_date: End date for price data
        """
        logger.info(f"Loading prices from {start_date} to {end_date}")
        # For simulation, load all data without resampling - we'll filter later
        prebars_df = self.data_loader.load_prebar_files(
            start_date, end_date,
            symbol_venues=self.symbol_venues,
            bars_type='tardis_prefer',
            resample_freq=None
        )

        if prebars_df is None or len(prebars_df) == 0:
            raise log_and_raise(f"No prebars found for {start_date} to {end_date}")

        self.as_of = prebars_df.index.get_level_values('ts').max()
        self.bars_df = prebars_df

    def _load_positions_df(self) -> Optional[pd.DataFrame]:
        """Load position data for simulation.

        Returns:
            DataFrame with position data, or None if not available

        Note:
            Logs warning if no positions found and initialize_positions is True
        """
        positions_df = self.existing_sim_positions_df
        if positions_df is None and self.initialize_sim_positions:
            logger.warning("Could not load latest sim positions - check if initial positions exist")
        return positions_df

    def _load_raw_fills(self, fills_df: pd.DataFrame) -> pd.DataFrame:
        """Load and standardize raw fill data from simulation.

        Args:
            fills_df: DataFrame with simulation fills containing executed_qty and trade_price

        Returns:
            DataFrame with standardized fill data

        Raises:
            RuntimeError: If fills_df is None
        """
        if fills_df is None:
            raise log_and_raise("Simulation fills_df must be provided")

        fills_df['commission_asset'] = 'USDT'
        standardized_fills_df = fills_df.rename(columns={'executed_qty': 'fill_qty', 'trade_price': 'fill_px'})
        standardized_fills_df['side'] = 'B'
        standardized_fills_df.loc[standardized_fills_df['fill_qty'] < 0, 'side'] = 'S'
        # we use abs qty in pnl calculator, so need to convert
        sell_idx = standardized_fills_df['side'] == 'S'
        standardized_fills_df.loc[sell_idx, 'fill_qty'] = -standardized_fills_df.loc[sell_idx, 'fill_qty']

        standardized_fills_df['fill_source'] = 'SIMULATION'
        standardized_fills_df = standardized_fills_df[STANDARDIZED_FILLS_COLUMNS]
        standardized_fills_df = shrink_floats(standardized_fills_df)

        self.raw_fills_df = standardized_fills_df.copy()

        # align fills on the minute boundary
        standardized_fills_df['ts'] = standardized_fills_df['ts'].dt.ceil('1min')
        return standardized_fills_df

    def get_last_day_fill_cnt(self) -> int:
        if self.fills_df is None or len(self.fills_df) == 0:
            return 0
        last_date = self.fills_df['date'].max()
        cnt = len(self.fills_df[self.fills_df['date'] == last_date])
        return cnt

    def run_pnl_calculation(
            self,
            fills_df: pd.DataFrame,
            fundings_df: Optional[pd.DataFrame] = None,
            record_position_age: Optional[str] = None) -> pd.DataFrame:
        """Execute main PnL calculation for simulation.

        Loads all required data and runs FIFO PnL calculation for all symbols.

        Args:
            fills_df: DataFrame with simulation fills
            fundings_df: Optional DataFrame with funding data
            record_position_age: How to record position age:
                - 'sim': Calculate position age for all timestamps
                - 'latest': Calculate position age only for latest timestamp
                - None: Don't calculate position age

        Returns:
            DataFrame with detailed PnL metrics by symbol and timestamp
        """
        self.fills_df = self._load_raw_fills(fills_df)

        if self.bars_df is None:
            raise log_and_raise("Could not load prebars for pnl calc")

        # Filter bars to only timestamps with fills
        self.bars_df = self.bars_df[self.bars_df.index.get_level_values('ts').isin(self.fills_df['ts'].unique())]
        if len(self.bars_df) == 0:
            raise log_and_raise("No bars found for pnl calc")

        # Load initial position if not already set
        if self.positions_df is None:
            self.positions_df = self._load_positions_df()

        self.fundings_df = fundings_df

        self.calculator.load_data(self.fills_df, self.bars_df, self.positions_df, self.fundings_df, update=False)
        self.pnl_df = self.calculator.calculate_pnl_performance_metrics(self.start_dt, record_position_age, update=False)
        return self.pnl_df

    def run_pnl_aggregation_by_ts(self) -> pd.DataFrame:
        """Aggregate PnL data by timestamp across all symbols.

        Returns:
            DataFrame with portfolio-level PnL aggregated by timestamp
        """
        self.ts_pnl_df = self.calculator.aggregate_pnl_timeslice(self.pnl_df)
        return self.ts_pnl_df

    def run_pnl_aggregation(self) -> pd.DataFrame:
        """Aggregate PnL data to daily level with performance metrics.

        Calculates daily, MTD, YTD, and lifetime metrics including returns,
        Sharpe ratio, and other performance indicators.

        Returns:
            DataFrame with comprehensive daily performance metrics
        """
        self.daily_pnl_df = self.calculator.aggregate_pnl_performance_metrics(self.pnl_df)
        return self.daily_pnl_df

    def update_position_from_sim_timeslice(self, initialize_positions: bool, existing_positions_df: Optional[pd.DataFrame] = None):
        self.initialize_sim_positions = initialize_positions
        self.existing_sim_positions_df = existing_positions_df

    def run_pnl_drawdown(self) -> pd.DataFrame:
        """Calculate top drawdown periods.

        Identifies and ranks the largest drawdown periods by percentage loss.

        Returns:
            DataFrame with top 3 drawdown periods and their statistics
        """
        return calculate_top_drawdowns(self.daily_pnl_df)
