from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from lib.pnl import FillBreakdown


def calc_return_metrics(pnl_df: pd.DataFrame, daily_scaler: float = 4, pnl_fillbreakdown: FillBreakdown = None) -> Dict[str, float]:
    """Calculate comprehensive return and risk metrics from PnL data.

    Args:
        pnl_df: DataFrame with PnL time series data
        daily_scaler: Scaling factor for annualization (default 4 for 6-hour bars)
        pnl_fillbreakdown: Optional FillBreakdown object for win ratio calculation

    Returns:
        Dictionary containing:
            - cum_pnl: Cumulative PnL in dollars
            - cum_ret: Cumulative return percentage
            - max_drawdown: Maximum drawdown in dollars
            - max_drawdown_perc: Maximum drawdown percentage
            - annualized_ret: Annualized return percentage
            - annualized_risk: Annualized volatility
            - annualized_sharpe: Sharpe ratio
            - avg_trading_volume: Average daily trading volume
            - avg_notional: Average position size
            - cum_fees: Cumulative fees in USD
            - avg_fees: Average daily fees
            - daily_fees_bps: Daily fees in basis points
            - cum_funding: Cumulative funding income
            - avg_funding: Average daily funding
            - daily_funding_bps: Daily funding in basis points
            - daily_turnover: Daily turnover ratio
            - win_ratio: Proportion of winning trades
            - num_profit_trades: Number of profitable trades
            - num_contracting_trades: Number of position-reducing trades
            - num_trades: Total number of trades
            - gain_per_fill: Average gain on winning trades
            - loss_per_fill: Average loss on losing trades
    """
    if 'notional' not in pnl_df.columns:
        pnl_df['notional'] = pnl_df['long'] - pnl_df['short']
    if 'funding_income' not in pnl_df.columns:
        pnl_df['funding_income'] = 0
    pnl_df['pnl_dff'] = pnl_df['pnl'].diff().fillna(pnl_df['pnl'])
    pnl_df['fees_usd_dff'] = pnl_df['fees_usd'].diff().fillna(pnl_df['fees_usd'])
    pnl_df['funding_income_dff'] = pnl_df['funding_income'].diff().fillna(pnl_df['funding_income'])
    pnl_df['ret'] = pnl_df['pnl_dff'] / pnl_df['notional']
    pnl_df['traded_dollars'] = pnl_df['traded_long'] + pnl_df['traded_short'].abs()

    def calc_mdd(df: pd.DataFrame, fld: str, notional_fld: str) -> Tuple[float, float]:
        notional_series = df[notional_fld]
        cumulative_max = df[fld].cummax()
        drawdown = df[fld] - cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_perc = max_drawdown / notional_series[drawdown.idxmin()]
        return max_drawdown, max_drawdown_perc

    max_drawdown, max_drawdown_perc = calc_mdd(pnl_df, 'pnl', 'notional')
    annualized_ret_perc = pnl_df['ret'].mean() * 365 * daily_scaler
    annualized_risk_perc = pnl_df['ret'].std() * np.sqrt(365) * np.sqrt(daily_scaler)
    win_ratio, num_profit_trades, num_contracting_trades, num_trades, gain_per_fill, loss_per_fill = 0, 0, 0, 0, 0, 0
    if pnl_fillbreakdown is not None:
        win_ratio, num_profit_trades, num_contracting_trades, num_trades, gain_per_fill, loss_per_fill = pnl_fillbreakdown.win_ratio()

    return {
        'cum_pnl': pnl_df['pnl'].tail(1).iloc[0],
        'cum_ret': pnl_df['ret'].sum(),
        'max_drawdown': max_drawdown,
        'max_drawdown_perc': max_drawdown_perc,
        'annualized_ret': annualized_ret_perc,
        'annualized_risk': annualized_risk_perc,
        'annualized_sharpe': annualized_ret_perc / annualized_risk_perc if annualized_risk_perc > 0 else 0,
        'avg_trading_volume': pnl_df['traded_dollars'].mean() * daily_scaler,
        'avg_notional': pnl_df['notional'].mean(),
        'cum_fees': pnl_df['fees_usd'].tail(1).iloc[0],
        'avg_fees': pnl_df['fees_usd_dff'].mean() * daily_scaler,
        'daily_fees_bps': (pnl_df['fees_usd_dff'] / pnl_df['traded_dollars']).mean() * 10000,
        'cum_funding': pnl_df['funding_income'].tail(1).iloc[0],
        'avg_funding': pnl_df['funding_income_dff'].mean() * daily_scaler,
        'daily_funding_bps': (pnl_df['funding_income_dff'] / pnl_df['notional']).mean() * daily_scaler * 10000,
        'daily_turnover': (pnl_df['traded_dollars'] / pnl_df['notional']).mean() * daily_scaler,
        'win_ratio': win_ratio,
        'num_profit_trades': num_profit_trades,
        'num_contracting_trades': num_contracting_trades,
        'num_trades': num_trades,
        'gain_per_fill': gain_per_fill,
        'loss_per_fill': loss_per_fill,
    }


def get_return_metrics_str(pnl_df: pd.DataFrame, pnl_name: str, daily_scaler: float = 4, pnl_fillbreakdown: Optional['FillBreakdown'] = None) -> str:
    """Format return metrics as a human-readable string.

    Args:
        pnl_df: DataFrame with PnL time series data
        pnl_name: Name/label for the PnL series
        daily_scaler: Scaling factor for annualization
        pnl_fillbreakdown: Optional FillBreakdown object for win ratio

    Returns:
        Formatted string with key performance metrics
    """
    return_metrics_dict = calc_return_metrics(pnl_df, daily_scaler, pnl_fillbreakdown)
    metrics = f"{pnl_name}:annualized_sharpe:{return_metrics_dict['annualized_sharpe']:.1f}\n"
    metrics += f"{pnl_name}:annualized_ret_perc:{return_metrics_dict['annualized_ret'] * 100:.2f}%\n"
    metrics += f"{pnl_name}:annualized_risk_perc:{return_metrics_dict['annualized_risk'] * 100:.2f}%\n"
    metrics += f"{pnl_name}:lifetime_pnl:${return_metrics_dict['cum_pnl']:.1f}\n"
    metrics += f"{pnl_name}:lifetime_ret_perc:{return_metrics_dict['cum_ret'] * 100:.2f}%\n"
    metrics += f"{pnl_name}:win_ratio:{return_metrics_dict['win_ratio'] * 100:.2f}%\n"
    metrics += f"{pnl_name}:profit_trades:{return_metrics_dict['num_profit_trades']} / {return_metrics_dict['num_contracting_trades']}\n"
    metrics += f"{pnl_name}:gain_per_fill:${return_metrics_dict['gain_per_fill']:.1f}\n"
    metrics += f"{pnl_name}:loss_per_fill:${return_metrics_dict['loss_per_fill']:.1f}\n"
    metrics += f"{pnl_name}:max_drawdown:${return_metrics_dict['max_drawdown']:.1f}\n"
    metrics += f"{pnl_name}:max_drawdown_perc:{return_metrics_dict['max_drawdown_perc'] * 100:.2f}%\n"
    metrics += f"{pnl_name}:daily_traded_dollars:${return_metrics_dict['avg_trading_volume']:.1f}\n"
    metrics += f"{pnl_name}:daily_fees:${return_metrics_dict['avg_fees']:.1f}\n"
    metrics += f"{pnl_name}:daily_fees_bps:{return_metrics_dict['daily_fees_bps']:.1f}\n"
    return metrics
