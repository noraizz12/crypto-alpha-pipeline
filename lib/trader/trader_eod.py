"""EOD Reversal execution overlay.

Modulates existing trade execution around the daily close (00:00 UTC)
based on the last-15-minute return reversal signal.

Two phases:
1. CAPTURE (23:45 UTC): Snapshot reference prices + risk metrics for all symbols
2. ACTIVE (00:00-00:15 UTC): Compute reversal signal, return execution modifiers

Research basis: FM t=5.91, Sharpe 5.67 in backtest.
Signal: -rank_cs(logret_15 / risk_15) at 00:00 UTC.
"""

import logging
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class EODReversalManager:
    """EOD reversal execution overlay.

    Two phases:
    1. CAPTURE (23:45 UTC): Snapshot reference prices + risk_1440 for all symbols
    2. ACTIVE (00:00-00:15 UTC): Compute reversal signal, return execution modifiers
    """

    def __init__(self, config: dict) -> None:
        self.enabled: bool = config.get('EOD_REVERSAL_ENABLED', False)
        self.capture_hour: int = config.get('EOD_REVERSAL_CAPTURE_HOUR', 23)
        self.capture_minute: int = config.get('EOD_REVERSAL_CAPTURE_MINUTE', 45)
        self.window_mins: int = config.get('EOD_REVERSAL_WINDOW_MINS', 15)
        self.max_agg_boost: int = config.get('EOD_REVERSAL_MAX_AGG_BOOST', 3)
        self.max_speed_mult: float = config.get('EOD_REVERSAL_MAX_SPEED_MULT', 2.0)
        self.min_speed_mult: float = config.get('EOD_REVERSAL_MIN_SPEED_MULT', 0.1)
        self.min_magnitude_pctile: float = config.get('EOD_REVERSAL_MIN_MAGNITUDE_PCTILE', 0.25)

        # state
        self._reference_prices: Dict[str, float] = {}
        self._risk_1440: Dict[str, float] = {}
        self._risk_15: Dict[str, float] = {}
        self._reference_date: Optional[date] = None
        self._signal: Dict[str, float] = {}
        self._magnitude_rank: Dict[str, float] = {}
        self._signal_ts: Optional[datetime] = None
        self._n_captured: int = 0
        self._n_signaled: int = 0

        if self.enabled:
            logger.info(
                f"EODReversalManager enabled: capture={self.capture_hour}:{self.capture_minute:02d}, "
                f"window={self.window_mins}m, max_agg_boost={self.max_agg_boost}, "
                f"speed=[{self.min_speed_mult}, {self.max_speed_mult}]"
            )

    def should_capture(self, now: datetime) -> bool:
        """True if time to capture reference prices and not yet captured today."""
        if not self.enabled:
            return False
        now_utc = now.astimezone(timezone.utc)
        if now_utc.hour != self.capture_hour:
            return False
        if now_utc.minute < self.capture_minute:
            return False
        # already captured today
        if self._reference_date == now_utc.date():
            return False
        return True

    def capture_reference_prices(
        self,
        symbols: List[str],
        get_mid_fn: Callable[[str], Optional[float]],
        target_df: pd.DataFrame,
    ) -> int:
        """Snapshot mid prices and risk metrics for all symbols.

        Args:
            symbols: list of symbol strings
            get_mid_fn: callable returning mid price for a symbol (or None)
            target_df: current target dataframe with 'symbol', 'risk_1440', 'risk_15' columns

        Returns:
            count of symbols successfully captured
        """
        self._reference_prices.clear()
        self._risk_1440.clear()
        self._risk_15.clear()
        self._signal.clear()
        self._magnitude_rank.clear()
        self._signal_ts = None

        risk_1440_map: Dict[str, float] = {}
        risk_15_map: Dict[str, float] = {}

        if 'risk_1440' in target_df.columns:
            for _, row in target_df[['symbol', 'risk_1440']].iterrows():
                risk_1440_map[row['symbol']] = float(row['risk_1440'])

        if 'risk_15' in target_df.columns:
            for _, row in target_df[['symbol', 'risk_15']].iterrows():
                risk_15_map[row['symbol']] = float(row['risk_15'])

        count = 0
        for symbol in symbols:
            mid = get_mid_fn(symbol)
            if mid is None or np.isnan(mid) or mid <= 0:
                continue
            risk_1440_val = risk_1440_map.get(symbol)
            risk_15_val = risk_15_map.get(symbol)
            if risk_1440_val is None or risk_15_val is None:
                continue
            if risk_1440_val <= 0 or risk_15_val <= 0:
                continue

            self._reference_prices[symbol] = mid
            self._risk_1440[symbol] = risk_1440_val
            self._risk_15[symbol] = risk_15_val
            count += 1

        self._reference_date = datetime.now(timezone.utc).date()
        self._n_captured = count
        logger.info(f"EOD Reversal: captured {count}/{len(symbols)} reference prices")
        return count

    def should_compute_signal(self, now: datetime) -> bool:
        """True if 00:00-00:window, reference captured, signal not yet computed."""
        if not self.enabled:
            return False
        now_utc = now.astimezone(timezone.utc)
        if now_utc.hour != 0:
            return False
        if now_utc.minute >= self.window_mins:
            return False
        if not self._reference_prices:
            return False
        if self._signal_ts is not None:
            return False
        return True

    def compute_signal(
        self,
        symbols: List[str],
        get_mid_fn: Callable[[str], Optional[float]],
    ) -> int:
        """Compute reversal signal from 15-min returns vs reference prices.

        Signal: -rank_cs(logret_15 / risk_15) -> values in [-0.5, 0.5]
        Magnitude: rank_cs(risk_1440) -> values in [0, 1]

        Args:
            symbols: list of symbol strings
            get_mid_fn: callable returning current mid price

        Returns:
            count of symbols with valid signal
        """
        eod_returns: Dict[str, float] = {}
        valid_symbols: List[str] = []

        for symbol in symbols:
            ref_price = self._reference_prices.get(symbol)
            risk_15 = self._risk_15.get(symbol)
            if ref_price is None or risk_15 is None:
                continue

            current_mid = get_mid_fn(symbol)
            if current_mid is None or np.isnan(current_mid) or current_mid <= 0:
                continue

            logret = np.log(current_mid / ref_price)
            vol_adj_ret = logret / risk_15
            eod_returns[symbol] = vol_adj_ret
            valid_symbols.append(symbol)

        if not valid_symbols:
            logger.warning("EOD Reversal: no valid symbols for signal computation")
            return 0

        # cross-sectional rank of vol-adjusted returns -> [0, 1]
        ret_values = np.array([eod_returns[s] for s in valid_symbols])
        n_symbols = len(valid_symbols)
        ret_ranks = _rank_array(ret_values)

        # reversal signal: negate the rank, center to [-0.5, 0.5]
        reversal_ranks = -(ret_ranks - 0.5)

        # magnitude rank based on risk_1440
        risk_1440_values = np.array([self._risk_1440.get(s, 0.0) for s in valid_symbols])
        mag_ranks = _rank_array(risk_1440_values)

        self._signal.clear()
        self._magnitude_rank.clear()

        for i, symbol in enumerate(valid_symbols):
            self._signal[symbol] = float(reversal_ranks[i])
            self._magnitude_rank[symbol] = float(mag_ranks[i])

        self._signal_ts = datetime.now(timezone.utc)
        self._n_signaled = n_symbols

        # log top/bottom 5
        sorted_symbols = sorted(valid_symbols, key=lambda s: self._signal[s], reverse=True)
        top_5 = sorted_symbols[:5]
        bottom_5 = sorted_symbols[-5:]

        top_str = ", ".join(
            f"{s}={self._signal[s]:+.3f}" for s in top_5
        )
        bottom_str = ", ".join(
            f"{s}={self._signal[s]:+.3f}" for s in bottom_5
        )
        logger.info(f"EOD Reversal signal computed: {n_symbols} symbols")
        logger.info(f"  Top (buy pressure): {top_str}")
        logger.info(f"  Bottom (sell pressure): {bottom_str}")

        return n_symbols

    def is_active(self) -> bool:
        """True if signal has been computed and we are within the active window."""
        if not self.enabled:
            return False
        if self._signal_ts is None:
            return False

        now_utc = datetime.now(timezone.utc)
        elapsed_mins = (now_utc - self._signal_ts).total_seconds() / 60.0
        return elapsed_mins < self.window_mins

    def get_execution_modifier(
        self, symbol: str, trade_amt: float
    ) -> Tuple[float, int]:
        """Return (speed_factor, aggression_boost) for a symbol's trade.

        speed_factor: multiplier on trade_amt (min_speed_mult to max_speed_mult)
        aggression_boost: integer to add to aggression (-max_agg_boost to +max_agg_boost)

        Returns (1.0, 0) if not active, symbol not in signal, or below magnitude threshold.
        """
        if not self.is_active():
            return (1.0, 0)

        reversal_sig = self._signal.get(symbol)
        mag_rank = self._magnitude_rank.get(symbol)

        if reversal_sig is None or mag_rank is None:
            return (1.0, 0)

        # skip low-magnitude symbols
        if mag_rank < self.min_magnitude_pctile:
            return (1.0, 0)

        if trade_amt == 0:
            return (1.0, 0)

        # alignment: positive means trade and reversal agree
        alignment = np.sign(trade_amt) * reversal_sig

        # scale alignment by magnitude rank (stronger effect for high-vol names)
        # alignment is in [-0.5, 0.5], mag_rank in [0, 1]
        # weighted_alignment in [-0.5, 0.5]
        weighted_alignment = alignment * mag_rank / max(1.0 - self.min_magnitude_pctile, 0.01)
        weighted_alignment = np.clip(weighted_alignment, -0.5, 0.5)

        # map weighted_alignment [-0.5, 0.5] to speed_factor
        # 0 -> 1.0, +0.5 -> max_speed_mult, -0.5 -> min_speed_mult
        if weighted_alignment >= 0:
            speed_factor = 1.0 + weighted_alignment * 2.0 * (self.max_speed_mult - 1.0)
        else:
            speed_factor = 1.0 + weighted_alignment * 2.0 * (1.0 - self.min_speed_mult)

        speed_factor = float(np.clip(speed_factor, self.min_speed_mult, self.max_speed_mult))

        # map to aggression boost: round weighted_alignment * 2 * max_agg_boost
        agg_boost = int(np.round(weighted_alignment * 2.0 * self.max_agg_boost))
        agg_boost = int(np.clip(agg_boost, -self.max_agg_boost, self.max_agg_boost))

        return (speed_factor, agg_boost)

    def get_signal_summary(self) -> str:
        """Human-readable summary for Slack/logging."""
        if not self.enabled:
            return "EOD Reversal: disabled"

        if self._signal_ts is None:
            if self._n_captured > 0:
                return f"EOD Reversal: {self._n_captured} prices captured, awaiting signal"
            return "EOD Reversal: no data captured"

        active_str = "ACTIVE" if self.is_active() else "INACTIVE"

        # top aligned (strongest buy reversal) and anti-aligned (strongest sell reversal)
        sorted_symbols = sorted(self._signal.keys(), key=lambda s: self._signal[s], reverse=True)
        top_3 = sorted_symbols[:3]
        bottom_3 = sorted_symbols[-3:]

        top_str = ", ".join(f"{s}={self._signal[s]:+.3f}" for s in top_3)
        bottom_str = ", ".join(f"{s}={self._signal[s]:+.3f}" for s in bottom_3)

        return (
            f"EOD Reversal [{active_str}]: {self._n_signaled} symbols, "
            f"signal_ts={self._signal_ts.strftime('%H:%M:%S')}  "
            f"Buy: {top_str}  Sell: {bottom_str}"
        )

    def reset(self) -> None:
        """Clear all state. Called on date rollover."""
        self._reference_prices.clear()
        self._risk_1440.clear()
        self._risk_15.clear()
        self._reference_date = None
        self._signal.clear()
        self._magnitude_rank.clear()
        self._signal_ts = None
        self._n_captured = 0
        self._n_signaled = 0


def _rank_array(values: np.ndarray) -> np.ndarray:
    """Cross-sectional rank scaled to [0, 1].

    Ties get average rank. NaN values get rank 0.5.
    """
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.array([0.5])

    # argsort of argsort gives ranks (0-indexed)
    order = np.argsort(np.argsort(values))
    # scale to [0, 1]: rank 0 -> 0/(n-1), rank n-1 -> 1
    ranks = order.astype(np.float64) / max(n - 1, 1)
    return ranks
