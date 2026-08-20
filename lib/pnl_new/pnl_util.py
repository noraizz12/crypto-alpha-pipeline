"""Utility functions for PnL analysis and statistics."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from lib.util import TARDIS_EXCHANGE
from lib.util.dataframes import make_date, remove_infs

logger = logging.getLogger(__name__)


def calc_pnl_returns(pnl_df: pd.DataFrame, pnl_col: str = 'pnl_net') -> pd.DataFrame:
    """Calculate PnL returns for a DataFrame indexed by timestamp.

    Takes a PnL DataFrame (typically the output of Pnl.aggregate_pnl()) and
    calculates returns at each timestamp.

    Args:
        pnl_df: DataFrame indexed by 'ts' with PnL metrics
            Required columns: specified pnl_col, notional
        pnl_col: Name of the PnL column to use for return calculation (default: 'pnl_net')

    Returns:
        DataFrame with original columns plus:
            - {pnl_col}_ret: Period return (pnl_col / notional)

    Note:
        Returns are calculated as pnl_col / notional.
        When notional is 0, return is NaN (meaningless metric - system shutdown).
        Return column is named {pnl_col}_ret to avoid conflicts when calculating multiple returns.
    """
    # Calculate period return = pnl_col / notional
    # Set to NaN when notional is 0 - metric is meaningless (zero notional = system shutdown)
    ret_col = f'{pnl_col}_ret'
    pnl_df[ret_col] = np.where(
        pnl_df['notional'] != 0,
        pnl_df[pnl_col] / pnl_df['notional'],
        np.nan
    )

    logger.info(f"Calculated returns for {len(pnl_df)} timestamps using {pnl_col} -> {ret_col}")

    return pnl_df


def compute_commissions(fills_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Compute commissions adjusted by commission asset mark price.

    This function requires commission_asset column in fills_df and:
    1. Converts commission_asset to symbol_venue format (e.g., 'BNB' -> 'BNBUSDT_binance-futures')
    2. Joins mark prices from bars_df for commission assets on timestamp
    3. Forward-fills missing prices to use last known price for fills after bar data
    4. Overwrites the commission column with commission * commission_asset_mark_price

    Args:
        fills_df: DataFrame containing fill data with required commission_asset column
        bars_df: DataFrame with MultiIndex (ts, symbol_venue) containing close_mid prices.
                 Must not be None or empty.

    Returns:
        Updated fills_df with adjusted commissions

    Raises:
        ValueError: If commission_asset column is missing from fills_df
        ValueError: If bars_df is None or empty (indicates data loading failure)

    Notes:
        - Uses price at time of fill when available
        - Forward-fills to use last known price for fills after bar data ends
    """
    logger.info("Computing commissions")

    if 'commission_asset' not in fills_df.columns:
        raise ValueError("fills_df missing required 'commission_asset' column - check data loader")

    if bars_df is None:
        raise ValueError("bars_df is None - cannot compute commissions without price data. "
                        "Check that both live_bars_df and prebars_df are loaded.")

    if len(bars_df) == 0:
        raise ValueError("bars_df is empty - cannot compute commissions without price data. "
                        "Check bar data loading pipeline.")

    # Convert commission_asset in place: 'BNB' -> 'BNBUSDT_binance-futures'
    fills_df['commission_asset'] = fills_df['commission_asset'] + f"USDT_{TARDIS_EXCHANGE}"

    # Extract mark prices from bars_df (close_mid is the mid price field)
    bars_prices_df = bars_df.reset_index()[['ts', 'symbol_venue', 'close_mid']].copy()
    bars_prices_df = bars_prices_df.rename(columns={'close_mid': 'commission_asset_mark_price', 'symbol_venue': 'commission_asset'})

    # Use merge_asof to get the most recent bar price for each fill timestamp
    # This handles fills that occur after bar data ends (uses last known price)
    fills_df = fills_df.sort_values('ts')
    bars_prices_df = bars_prices_df.sort_values('ts')

    # Merge asof for each commission asset separately
    fills_with_commission_prices_list = []
    for commission_asset in fills_df['commission_asset'].unique():
        asset_fills = fills_df[fills_df['commission_asset'] == commission_asset].copy()
        asset_bars = bars_prices_df[bars_prices_df['commission_asset'] == commission_asset].copy()

        if len(asset_bars) > 0:
            # merge_asof: for each fill timestamp, use the most recent bar price (backward)
            asset_fills_with_prices = pd.merge_asof(
                asset_fills,
                asset_bars[['ts', 'commission_asset_mark_price']],
                on='ts',
                direction='backward'
            )
            fills_with_commission_prices_list.append(asset_fills_with_prices)
        else:
            asset_fills['commission_asset_mark_price'] = np.nan
            fills_with_commission_prices_list.append(asset_fills)

    fills_with_commission_prices_df = pd.concat(fills_with_commission_prices_list, ignore_index=True)

    # Update commission column: commission * commission_asset_mark_price
    # Only update rows where we have a commission asset and mark price
    mask = fills_with_commission_prices_df['commission_asset_mark_price'].notna()
    fills_with_commission_prices_df.loc[mask, 'commission'] = (
        fills_with_commission_prices_df.loc[mask, 'commission_raw'] *
        fills_with_commission_prices_df.loc[mask, 'commission_asset_mark_price']
    )

    # Log adjustment statistics
    adjusted_count = mask.sum()
    logger.info(f"Adjusted {adjusted_count}/{len(fills_df)} fills with commission asset mark prices")

    # Clean up temporary columns
    fills_with_commission_prices_df = fills_with_commission_prices_df.drop(columns=['commission_asset_mark_price'])
    return fills_with_commission_prices_df


def aggregate_to_daily(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday PnL time series to daily.

    Takes a PnL DataFrame indexed by 'ts' and aggregates it to daily frequency.
    Asserts that the input time interval is <= 1 day.

    Args:
        pnl_df: DataFrame indexed by 'ts' with PnL metrics from SecurityPnl.calculate_pnl()
            Expected columns: qty, mark_price, position, cost_basis, cost_cum, notional,
                            pnl_gross, pnl_net, realized_pnl_cum, unrealized_pnl,
                            commission_cum, funding_income_cum, abs_dollars_cum, and *_ret columns

    Returns:
        DataFrame indexed by date with daily aggregated PnL metrics:
            - Last values: qty, mark_price, position, cost_basis, cost_cum, notional,
                          pnl_gross_cum, pnl_net_cum, realized_pnl_cum, unrealized_pnl_cum,
                          commission_cum, funding_income_cum, abs_dollars_cum
            - Summed values: fill_qty, fill_dollars, commission, funding_income, realized_pnl,
                           and all *_ret columns

    Raises:
        AssertionError: If time interval between rows exceeds 1 day
    """
    if len(pnl_df) == 0:
        return pnl_df

    # Assert time interval is <= 1 day
    if len(pnl_df) > 1:
        time_diffs = pnl_df.index.to_series().diff()
        max_interval = time_diffs.max()
        assert max_interval <= pd.Timedelta(days=1), \
            f"Time interval {max_interval} exceeds 1 day maximum"

    # Convert ts index to date
    daily_df = pnl_df.copy()
    daily_df['date'] = daily_df.index.date

    # Define aggregation rules
    agg_dict = {
        # Last values (end-of-day snapshots)
        'qty': 'last',
        'mark_price': 'last',
        'position': 'last',
        'cost_basis': 'last',
        'cost_cum': 'last',
        'notional': 'last',
        'long_value': 'last',
        'short_value': 'last',
        'pnl_gross_cum': 'last',
        'pnl_net_cum': 'last',
        'realized_pnl_cum': 'last',
        'unrealized_pnl_cum': 'last',
        'commission_cum': 'last',
        'funding_income_cum': 'last',
        'abs_dollars_cum': 'last',

        # Summed values (daily totals - flow variables)
        'fill_qty': 'sum',
        'fill_dollars': 'sum',
        'fill_count': 'sum',
        'commission': 'sum',
        'funding_income': 'sum',
        'realized_pnl': 'sum',
        'pnl_gross': 'sum',  # Period PnL (flow)
        'pnl_net': 'sum',    # Period PnL (flow)
        'unrealized_pnl': 'sum',  # Period PnL (flow)
    }

    # Filter agg_dict to only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in pnl_df.columns}

    # Aggregate by date
    daily_df = daily_df.groupby('date').agg(agg_dict)

    logger.info(f"Daily aggregation: fill_count sum = {daily_df['fill_count'].sum() if 'fill_count' in daily_df.columns else 'N/A'}")

    # Convert date index to datetime
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df.index.name = 'ts'

    # Recalculate returns at daily level
    # Returns should NOT be summed - they must be recalculated as pnl / notional
    for pnl_col in ['pnl_gross', 'pnl_net', 'realized_pnl_cum', 'unrealized_pnl']:
        if pnl_col in daily_df.columns:
            daily_df = calc_pnl_returns(daily_df, pnl_col=pnl_col)

    logger.info(f"Aggregated {len(pnl_df)} intraday rows to {len(daily_df)} daily rows")

    return daily_df


def calculate_performance_statistics(daily_pnl_df: pd.DataFrame, pnl_col: str = 'pnl_net') -> Optional[dict]:
    """Calculate annualized performance statistics from daily PnL.

    Args:
        daily_pnl_df: DataFrame indexed by date with daily PnL metrics
            Required columns: {pnl_col}_ret (daily returns), notional, commission,
                            abs_dollars_cum, {pnl_col}
        pnl_col: Name of the PnL column used for returns (default: 'pnl_net')

    Returns:
        Dictionary with performance statistics, or None if daily_pnl_df is empty:
            - annualized_return: Annualized return (365 trading days)
            - annualized_std: Annualized standard deviation of returns
            - sharpe_ratio: Sharpe ratio (annualized_return / annualized_std)
            - total_days: Number of trading days
            - mean_daily_return: Average daily return
            - std_daily_return: Daily return standard deviation
            - avg_notional: Average daily notional position size
            - avg_daily_traded: Average daily traded amount (abs dollar volume)
            - pct_positive_days: Percentage of days with positive returns
            - avg_turnover_pct: Average daily turnover as % of notional
            - avg_daily_commission: Average daily commission
            - avg_daily_pnl: Average daily PnL (dollar amount)
            - avg_daily_fill_count: Average number of fills per day
            - commission_pct_of_traded: Commission as % of absolute dollars traded
            - total_funding_income: Total cumulative funding income over period
            - avg_daily_funding_income: Average daily funding income (dollar amount)
            - daily_funding_income_pct_notional: Daily funding income as % of notional

    Raises:
        ValueError: If required return column doesn't exist
    """
    ret_col = f'{pnl_col}_ret'

    if ret_col not in daily_pnl_df.columns:
        raise ValueError(f"Required column '{ret_col}' not found in daily_pnl_df. "
                        f"Make sure calc_pnl_returns() was called with pnl_col='{pnl_col}'")

    # Get daily returns
    daily_returns = daily_pnl_df[ret_col]

    # Calculate statistics
    n_days = len(daily_returns)
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    # Annualize (crypto markets trade 365 days per year)
    trading_days_per_year = 365
    annualized_return = mean_daily_return * trading_days_per_year
    annualized_std = std_daily_return * np.sqrt(trading_days_per_year)

    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / annualized_std
    if np.isinf(sharpe_ratio):
        sharpe_ratio = np.nan

    # Additional statistics
    avg_notional = daily_pnl_df['notional'].mean() if 'notional' in daily_pnl_df.columns else 0.0

    # Average daily traded amount (from abs_dollars_cum which is cumulative, so take diff)
    if 'abs_dollars_cum' in daily_pnl_df.columns:
        daily_traded = daily_pnl_df['abs_dollars_cum'].diff().fillna(daily_pnl_df['abs_dollars_cum'].iloc[0])
        avg_daily_traded = daily_traded.mean()
    else:
        daily_traded = pd.Series([0.0] * n_days)
        avg_daily_traded = 0.0

    # Percentage of positive return days
    pct_positive_days = (daily_returns > 0).sum() / n_days if n_days > 0 else 0.0

    # Average daily turnover as % of notional
    if 'notional' in daily_pnl_df.columns and avg_notional > 0 and 'abs_dollars_cum' in daily_pnl_df.columns:
        daily_turnover_pct = (daily_traded / daily_pnl_df['notional']) * 100
        avg_turnover_pct = daily_turnover_pct.mean()
    else:
        avg_turnover_pct = 0.0

    # Average daily commission
    if 'commission' in daily_pnl_df.columns:
        avg_daily_commission = daily_pnl_df['commission'].mean()
    else:
        avg_daily_commission = 0.0

    # Average daily PnL (dollar amount)
    if pnl_col in daily_pnl_df.columns:
        avg_daily_pnl = daily_pnl_df[pnl_col].mean()
    else:
        avg_daily_pnl = 0.0

    # Average daily fill count
    if 'fill_count' in daily_pnl_df.columns:
        avg_daily_fill_count = daily_pnl_df['fill_count'].mean()
    else:
        avg_daily_fill_count = 0.0

    # Commission as % of absolute dollars traded
    if avg_daily_traded > 0 and avg_daily_commission > 0:
        commission_pct_of_traded = (avg_daily_commission / avg_daily_traded) * 100
    else:
        commission_pct_of_traded = 0.0

    # Funding income statistics
    if 'funding_income_cum' in daily_pnl_df.columns:
        total_funding_income = daily_pnl_df['funding_income_cum'].iloc[-1] if len(daily_pnl_df) > 0 else 0.0
    else:
        total_funding_income = 0.0

    if 'funding_income' in daily_pnl_df.columns:
        avg_daily_funding_income = daily_pnl_df['funding_income'].mean()
    else:
        avg_daily_funding_income = 0.0

    # Daily funding income as % of notional
    if avg_notional > 0 and avg_daily_funding_income != 0:
        daily_funding_income_pct_notional = (avg_daily_funding_income / avg_notional) * 100
    else:
        daily_funding_income_pct_notional = 0.0

    # Return Base: simple arithmetic mean of daily returns (time-weighted, each day has equal weight)
    # This differs from PnL Base which weights by dollars (notional)
    annualized_unlev_ret_perc_from_ret = daily_pnl_df['net_pnl_unlev_ret'].mean() * 365
    annualized_lev_ret_perc_from_ret = daily_pnl_df['net_pnl_lev_ret'].mean() * 365
    annualized_unlev_ret_perc = daily_pnl_df['net_pnl'].mean() / daily_pnl_df['gross_notional'].mean() * 365
    annualized_risk_perc = daily_pnl_df['net_pnl_unlev_ret'].std() * np.sqrt(365)

    # Calculate sharpe ratios - convert inf to NaN for cleaner display
    sharpe_unlev = annualized_unlev_ret_perc / annualized_risk_perc
    if np.isinf(sharpe_unlev):
        sharpe_unlev = np.nan

    sharpe_unlev_from_ret = annualized_unlev_ret_perc_from_ret / annualized_risk_perc
    if np.isinf(sharpe_unlev_from_ret):
        sharpe_unlev_from_ret = np.nan

    stats = {
        'annualized_return': annualized_return,
        'annualized_std': annualized_std,
        'sharpe_ratio': sharpe_ratio,
        'total_days': n_days,
        'mean_daily_return': mean_daily_return,
        'std_daily_return': std_daily_return,
        'avg_notional': avg_notional,
        'avg_daily_traded': avg_daily_traded,
        'pct_positive_days': pct_positive_days,
        'avg_turnover_pct': avg_turnover_pct,
        'avg_daily_commission': avg_daily_commission,
        'avg_daily_pnl': avg_daily_pnl,
        'avg_daily_fill_count': avg_daily_fill_count,
        'commission_pct_of_traded': commission_pct_of_traded,
        'total_funding_income': total_funding_income,
        'avg_daily_funding_income': avg_daily_funding_income,
        'daily_funding_income_pct_notional': daily_funding_income_pct_notional,

        'start_dt': daily_pnl_df['date'].iloc[0],
        'cum_pnl': daily_pnl_df['net_pnl'].sum(),
        'cum_unlev_ret': daily_pnl_df['net_pnl'].sum() / daily_pnl_df['gross_notional'].mean(),
        'cum_lev_ret': daily_pnl_df['net_pnl'].sum() / daily_pnl_df['balance'].mean(),
        'annualized_unlev_ret': annualized_unlev_ret_perc,
        'annualized_lev_ret': daily_pnl_df['net_pnl'].mean() / daily_pnl_df['balance'].mean() * 365,
        'annualized_unlev_ret_from_ret': annualized_unlev_ret_perc_from_ret,
        'annualized_lev_ret_from_ret': annualized_lev_ret_perc_from_ret,
        'annualized_risk': annualized_risk_perc,
        'annualized_sharpe': sharpe_unlev,
        'annualized_sharpe_from_ret': sharpe_unlev_from_ret,
        'volume': daily_pnl_df['fill_dollars_abs'].mean(),
        'turnover': daily_pnl_df['turnover'].mean(),
        'fees': daily_pnl_df['commission'].mean(),
        'fees_bps': daily_pnl_df['fees_bps_daily'].mean(),
        'cum_fundings_income': daily_pnl_df['funding_income'].sum(),
        'avg_fundings_income': daily_pnl_df['funding_income'].mean(),
        'avg_fundings_income_bps': daily_pnl_df['funding_income_bps_daily'].mean(),
    }

    logger.info(f"Performance statistics ({pnl_col}): "
                f"ann_ret={annualized_return:.4f}, "
                f"ann_std={annualized_std:.4f}, "
                f"sharpe={sharpe_ratio:.2f}, "
                f"days={n_days}, "
                f"win_rate={pct_positive_days:.2%}")

    return stats

def calc_return_metrics(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate return-based performance metrics.

    Args:
        pnl_df: Daily PNL DataFrame

    Returns:
        DataFrame with calculated return metrics
    """
    # Use remove_infs to handle division by zero
    # Leave as NaN when denominator is 0 (e.g. zero notional = system shut down)
    pnl_df['net_pnl_unlev_ret'] = remove_infs(pnl_df['net_pnl'] / pnl_df['gross_notional'])
    pnl_df['net_pnl_ret'] = pnl_df['net_pnl_unlev_ret']
    pnl_df['net_pnl_lev_ret'] = remove_infs(pnl_df['net_pnl'] / pnl_df['balance'])
    pnl_df['turnover'] = remove_infs(pnl_df['fill_dollars_abs'] / pnl_df['gross_notional'])
    pnl_df['fees_bps_daily'] = remove_infs(pnl_df['commission'] / pnl_df['fill_dollars_abs'] * 10000)
    pnl_df['funding_income_bps_daily'] = remove_infs(pnl_df['funding_income'] / pnl_df['gross_notional'] * 10000)
    return pnl_df

def calculate_performance_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly performance metrics.

    Args:
        df: Daily PNL DataFrame

    Returns:
        Monthly aggregated performance DataFrame
    """
    # Add return_daily column if not present
    if 'return_daily' not in df.columns:
        df['return_daily'] = remove_infs(df['net_pnl'] / df['gross_notional'])

    # Use ts directly for portfolio data (no 1-minute offset shift)
    # make_date() shifts by 1 minute which is correct for bar data but wrong for portfolio snapshots
    if 'ts' in df.columns:
        df['date'] = df['ts'].dt.normalize()
    else:
        df = make_date(df)
    df['year_month'] = df['date'].dt.strftime('%Y-%m')

    monthly_perf_df = df.groupby('year_month').agg({
        'net_pnl': ['mean', 'sum'],
        'gross_notional': 'mean',
        'balance': ['first', 'last'],
        'return_daily': 'std',  # Uses ddof=1 (default) - single-day months return NaN (metrics are meaningless)
        'commission': 'sum',
        'funding_income': 'sum',
        'logret_cum_wgtmkt': ['first', 'last'],
    })
    monthly_perf_df.columns = ['_'.join(col).strip() for col in monthly_perf_df.columns.values]

    # Use remove_infs to handle division by zero
    # Leave as NaN when denominator is 0 - metrics are meaningless without notional
    monthly_perf_df['capital_delta'] = monthly_perf_df['balance_last'] - monthly_perf_df['balance_first']
    monthly_perf_df['cum_unlev_return'] = remove_infs(monthly_perf_df['net_pnl_sum'] / monthly_perf_df['gross_notional_mean'])
    monthly_perf_df['annualized_unlev_return'] = remove_infs(monthly_perf_df['net_pnl_mean'] / monthly_perf_df['gross_notional_mean'] * 365)
    monthly_perf_df['annualized_unlev_return_std'] = monthly_perf_df['return_daily_std'] * np.sqrt(365)
    monthly_perf_df['sharpe'] = remove_infs(monthly_perf_df['annualized_unlev_return'] / monthly_perf_df['annualized_unlev_return_std'])

    # Market return for the month (convert log return to simple return)
    log_return = monthly_perf_df['logret_cum_wgtmkt_last'] - monthly_perf_df['logret_cum_wgtmkt_first']
    monthly_perf_df['market_return'] = np.exp(log_return) - 1

    # Backing out gross from net
    monthly_perf_df['gross_pnl'] = monthly_perf_df['net_pnl_sum'] + monthly_perf_df['commission_sum'] - monthly_perf_df['funding_income_sum']

    monthly_perf_df = monthly_perf_df.reset_index()
    return monthly_perf_df
