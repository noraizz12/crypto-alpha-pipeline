import logging
from datetime import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd

from lib.data import load_exchange_info
from lib.util.util import ONE_PENNY, TARDIS_EXCHANGE
from lib.trader import DEFAULT_COMMISSION_ASSET, USDT

DEFAULT_COMMISSION_ASSET_SYMBOL_VENUE = DEFAULT_COMMISSION_ASSET + USDT + "_" + TARDIS_EXCHANGE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def round_dust_position(
        df: pd.DataFrame,
        round_dt: dt,
        px_col: str = 'mark_price',
        qty_col: str = 'qty',
        pos_col: str = 'notional',
        pos_age_col: str = 'position_age',
        adjust_pos_age: bool = False,
) -> pd.DataFrame:
    round_date = round_dt.date()
    round_idx = df.index.get_level_values('ts') == round_dt
    df.loc[round_idx, 'min_notional'] = np.nan
    round_df = df.loc[round_idx]

    # query symbol to its qty precision point in time
    exch_info_df = load_exchange_info(info_date=round_date, skip_log=True)
    if exch_info_df is not None:
        round_df = round_df.join(exch_info_df[['quantityPrecision']], on='symbol_venue', how='left')
        mask = round_df['quantityPrecision'].notna()
        round_df.loc[mask, 'min_notional'] = (10.0 ** -round_df.loc[mask, 'quantityPrecision']) * round_df.loc[mask, px_col]
    round_df['min_notional'] = round_df['min_notional'].fillna(value=ONE_PENNY)

    # round position/qty
    round_case = round_df[pos_col].abs() < round_df['min_notional']
    logger.info(f"Rounding position at {round_dt} for {round_case.sum()} positions")
    for col in [qty_col, pos_col]:
        round_df.loc[round_case, col] = 0
        df.loc[round_idx, col] = round_df[col]
    if adjust_pos_age:
        round_df.loc[round_case, pos_age_col] = np.nan
        df.loc[round_idx, pos_age_col] = round_df[pos_age_col]

    return df


def calculate_top_drawdowns(df: pd.DataFrame) -> pd.DataFrame:
    """Fast identification and ranking of largest drawdown periods."""

    # Step 1: Calculate all base metrics in one pass (vectorized)
    pnl_values = df['total_pnl_lifetime'].values
    notional_values = df['notional_abs_daily'].values
    dates = df.index.values

    # Vectorized peak and drawdown calculation
    peaks = np.maximum.accumulate(pnl_values)
    drawdowns = pnl_values - peaks

    # Step 2: Find drawdown periods efficiently using NumPy
    is_drawdown = drawdowns < 0

    # Find start and end of each drawdown period
    # Use diff to find transitions
    drawdown_changes = np.diff(np.concatenate(([False], is_drawdown, [False])).astype(int))
    drawdown_starts = np.where(drawdown_changes == 1)[0]  # Start of drawdown
    drawdown_ends = np.where(drawdown_changes == -1)[0] - 1  # End of drawdown

    if len(drawdown_starts) == 0:
        # No drawdowns found
        return pd.DataFrame(columns=['start_date', 'end_date', 'drawdown_days', 'dollar_loss', 'percent_loss'])

    # Step 3: Calculate metrics for each drawdown period (vectorized)
    drawdown_metrics = []

    for start_idx, end_idx in zip(drawdown_starts, drawdown_ends):
        # Get the period data
        period_drawdowns = drawdowns[start_idx:end_idx + 1]
        period_notional = notional_values[start_idx:end_idx + 1]

        # Find trough (minimum drawdown in this period)
        trough_relative_idx = np.argmin(period_drawdowns)
        trough_idx = start_idx + trough_relative_idx

        # Calculate metrics
        start_date = dates[start_idx]
        end_date = dates[trough_idx]  # End at trough
        drawdown_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        dollar_loss = period_drawdowns[trough_relative_idx]  # Most negative value
        avg_notional = np.mean(period_notional)
        percent_loss = dollar_loss / avg_notional if avg_notional != 0 else 0

        drawdown_metrics.append({
            'start_date': pd.Timestamp(start_date).date(),
            'end_date': pd.Timestamp(end_date).date(),
            'drawdown_days': drawdown_days,
            'dollar_loss': dollar_loss,
            'percent_loss': percent_loss,
        })

    # Step 4: Create result DataFrame and sort
    if not drawdown_metrics:
        return pd.DataFrame(columns=['start_date', 'end_date', 'drawdown_days', 'dollar_loss', 'percent_loss'])

    result_df = pd.DataFrame(drawdown_metrics)
    return result_df.sort_values('percent_loss', ascending=True).head(3)


def get_commission_px_dict(close_mid_df: pd.DataFrame) -> Dict[dt, Dict[str, float]]:
    """Extract commission asset prices for fee calculation in USD.

    Args:
        close_mid_df: DataFrame with close mid prices including commission assets

    Returns:
        Dict mapping timestamp to dict of commission asset prices
    """
    # use this to estimate commission fee as usd
    commission_px_df = close_mid_df.loc[close_mid_df.index.get_level_values('symbol_venue') == DEFAULT_COMMISSION_ASSET_SYMBOL_VENUE, ['close_mid']].unstack()
    commission_px_df = commission_px_df.droplevel(0, axis=1)
    commission_px_df = commission_px_df.rename(columns={DEFAULT_COMMISSION_ASSET_SYMBOL_VENUE: DEFAULT_COMMISSION_ASSET})
    # we could load true tether price here in the future
    commission_px_df[USDT] = 1.0
    return commission_px_df.to_dict('index')
