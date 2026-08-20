"""Position tracking and management for cryptocurrency futures trading.

This module provides classes and functions for managing trading positions,
tracking fills, calculating cost basis, and handling position reconciliation.
It supports real-time position updates, P&L calculations, and dust position
management for the statistical arbitrage trading system.

Classes:
    SecPos: Represents a single security position with fill tracking
    PositionRecorder: Manages multiple positions and persistence

Functions:
    round_dust_position: Rounds small positions to zero based on exchange precision
"""

import logging
import math
import os
from datetime import datetime as dt, timezone
from typing import Optional, Tuple, Dict, List

import pandas as pd

from lib.util.directory import dir_manager
from lib.util.time_util import beginning_of_day, dt_to_str, date_to_str, today_date
from .trading import Fill
from ..util import shrink_floats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SecPos:
    """Represents a single security position with comprehensive tracking.
    
    This class maintains position state including quantity, cost basis, fees,
    and mark-to-market values. It processes fills to update position metrics
    and provides methods for position reconciliation and P&L calculations.
    
    Attributes:
        symbol: Trading symbol identifier
        abs_dvolume: Absolute dollar volume traded
        abs_qty: Absolute quantity traded (always positive)
        qty: Net position quantity (positive=long, negative=short)
        execution_qty: Net executed quantity including all fills
        cost_basis: Dollar cost basis (negative=long, positive=short)
        fill_cnt: Number of fills processed
        fees: Total trading fees in USD
        mark: Current mark price
        mark_ts: Timestamp of last mark price update
        unrealized_profit: Optional unrealized P&L
        vwap: Optional volume-weighted average price
    
    Example:
        >>> pos = SecPos("BTCUSDT")
        >>> fill = Fill(symbol="BTCUSDT", qty=0.1, px=50000, side="BUY", ...)
        >>> pos.add_fill(fill)
        >>> print(f"Position: {pos.qty} @ avg price {pos.avg_px()}")
    """
    
    def __init__(self, symbol: str):
        """Initialize a new security position.
        
        Args:
            symbol: The trading symbol for this position
        """
        self.symbol: str = symbol
        self.abs_dvolume: float = 0.0
        self.abs_qty: float = 0.0
        self.qty: float = 0.0
        self.execution_qty: float = 0.0
        self.cost_basis: float = 0.0
        self.fill_cnt: int = 0
        self.fees: float = 0.0
        self.mark: float = 0.0
        self.mark_ts: dt = dt.now(timezone.utc)

        self.unrealized_profit: Optional[float] = None
        self.vwap: Optional[float] = None

    def add_fill(self, fill: Fill):
        """Process a new fill and update position metrics.
        
        Updates quantity, cost basis, fees, and trading statistics based on
        the fill details. Maintains FIFO cost accounting.
        
        Args:
            fill: Fill object containing trade execution details
            
        Raises:
            AssertionError: If fill symbol doesn't match position symbol
        """
        assert fill.symbol == self.symbol
        signed_fill_notional = fill.notional(include_fees=False)
        self.abs_dvolume += abs(signed_fill_notional)
        self.abs_qty += fill.qty
        self.qty += fill.signed_qty()
        self.execution_qty += fill.signed_qty()
        self.cost_basis -= signed_fill_notional
        self.fill_cnt += 1
        self.fees += fill.commission_usd
        self.mark = fill.px
        self.mark_ts = fill.exch_ts

    def notional(self) -> float:
        """Calculate current notional value of the position.
        
        Returns:
            Current mark price multiplied by position quantity
        """
        return self.mark * self.qty

    def avg_px(self) -> float:
        """Calculate average execution price.
        
        Returns:
            Average price per unit, or 0.0 if no trades executed
        """
        if self.abs_qty == 0:
            return 0.0
        return self.abs_dvolume / self.abs_qty

    def update_px(self, px: float):
        """Update mark price with validation.
        
        Args:
            px: New price to set (must be positive)
            
        Note:
            Invalid prices (<=0) are logged and ignored
        """
        if px <= 0:
            logger.error(f"Bad price update on {self.symbol} {self.mark} -/-> {px}")
            return
        self.mark = px
        self.mark_ts = dt.now(timezone.utc)

    def refresh_qty(self, qty: float):
        """Synchronize position quantity with external source.
        
        Used for reconciliation with exchange data. Logs discrepancies
        if values are not mathematically close.
        
        Args:
            qty: New quantity from external source
        """
        if qty != self.qty:
            if not math.isclose(qty, self.qty):
                logger.info(f"Refreshing qty on {self.symbol} {self.qty} -> {qty}")
            self.qty = qty

    def refresh_cost_basis(self, cost_basis: float):
        """Synchronize cost basis with external source.
        
        Used for accounting reconciliation. Logs discrepancies if values
        are not mathematically close.
        
        Args:
            cost_basis: New cost basis from external source
        """
        if cost_basis != self.cost_basis:
            if not math.isclose(cost_basis, self.cost_basis):
                logger.info(f"Refreshing cost basis on {self.symbol} ${self.cost_basis} -> ${cost_basis}")
            self.cost_basis = cost_basis

    def reset_avg_cost(self):
        """Reset average cost calculations.
        
        Clears volume and quantity tracking. Used when starting a new
        accounting period while maintaining position.
        """
        self.abs_dvolume = 0
        self.abs_qty = 0
        self.execution_qty = 0

    def get_list_for_df_row(self) -> Tuple:
        """Get position data as tuple for DataFrame construction.
        
        Returns:
            Tuple of (symbol, qty, cost_basis, mark, mark_ts, notional)
        """
        return self.symbol, self.qty, self.cost_basis, self.mark, self.mark_ts, self.notional()


class PositionRecorder:
    def __init__(self, prod: bool, start: Optional[dt] = None, end: Optional[dt] = None):
        self.prod = prod
        if self.prod or start is None:
            start = beginning_of_day()
        if self.prod or end is None:
            end = dt.now(timezone.utc)

        self.start: dt = start
        self.end: dt = end
        logger.info(f"Running pnl prod: {self.prod} from {self.start} to {self.end}")

        self.secs: Dict[str, SecPos] = {}

    def get_security(self, symbol: str) -> SecPos:
        sec_pos = self.secs.get(symbol, SecPos(symbol=symbol))
        self.secs[symbol] = sec_pos
        return sec_pos

    def refresh_qty(self, symbol: str, qty: float):
        sec_pos = self.get_security(symbol)
        sec_pos.refresh_qty(qty)

    def add_fill(self, fill: Fill):
        sec_pos = self.get_security(fill.symbol)
        sec_pos.add_fill(fill)

    def get_symbols(self) -> List[str]:
        return list(self.secs.keys())

    def get_pos(self) -> List[SecPos]:
        return list(self.secs.values())

    def get_qty(self, symbol: str) -> float:
        if symbol not in self.secs:
            return 0.0
        return self.get_security(symbol).qty

    def reset_avg_cost(self):
        for sec_pos in self.secs.values():
            sec_pos.reset_avg_cost()

    def dump_positions(self):
        df_rows = []
        for sec_pos in self.get_pos():
            df_rows.append(sec_pos.get_list_for_df_row())

        if len(df_rows) == 0:
            logger.warning(f"No positions from binance!")
            return

        pos_df = pd.DataFrame(df_rows, columns=['symbol', 'positionAmt', 'cost_basis', 'price', 'ts', 'notional'])
        pos_df = shrink_floats(pos_df)
        pos_dir = f"{dir_manager.POSITION_DIR}/{date_to_str(today_date())}"
        filename = f"{pos_dir}/pos.{dt_to_str()}.parquet"
        logger.info(f"Saving positions to {filename}")
        os.makedirs(pos_dir, exist_ok=True)
        pos_df.to_parquet(filename)


