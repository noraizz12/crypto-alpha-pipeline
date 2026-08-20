"""Main trading reports orchestrator module.

This module provides the main orchestrator class for all trading reports.
Integrates all reporting components into a unified interface for
generating comprehensive trading analytics. Manages report updates,
data refresh, and dashboard generation for both historical and
real-time trading analysis.

Classes:
    TradingReports: Main orchestrator for all trading reports
"""

import logging
import os
from datetime import datetime as dt, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import html
from dash.dash_table import Format

from lib.calcs.calcs import Calcs
from lib.data.live_bars import LiveBars
from lib.data.loaders import load_single_live_bar_df
from lib.data.dataloader import DataLoader
from lib.data.loaders import load_targets
from lib.pnl import FillBreakdown
from lib.pnl_new.binance_pnl import BinancePnl
from lib.trader.trading import Side
from lib.util import get_sim_dirs
from lib.util.config import extract_horizon_models, get_config
from lib.util.dataframes import clean_poop, remove_infs, merge_on_index, make_symbol_venue
from lib.util.directory import DirectoryManager, SIM_DIR
from lib.util.time_util import today, date_to_start_dt, date_to_end_dt, today_date
from lib.util.util import fmoney

logger = logging.getLogger(__name__)


# Column definitions for PNL tables
PNL_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'notional', 'name': 'Position', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'portfolio_pct', 'name': 'Portfolio %', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'trading', 'name': 'Trading', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'unrealized_daily', 'name': 'Unrealized Pnl', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'realized_daily', 'name': 'Realized Pnl', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'total_pnl_daily', 'name': 'Total Pnl', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'daily_return', 'name': 'Daily Return %', 'type': 'numeric', 'format': Format.Format(precision=4, scheme=Format.Scheme.percentage)},
    {'id': 'unrealized_return', 'name': 'Unrealized Return %', 'type': 'numeric', 'format': Format.Format(precision=4, scheme=Format.Scheme.percentage)},
    {'id': 'funding_income', 'name': 'Funding Income', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'alpha_opt', 'name': 'Alpha Opt (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'risk_1440', 'name': 'Risk 1440 (%)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'last_funding_rate', 'name': 'Last Funding Rate (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'alpha_15', 'name': 'Alpha 15 (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'alpha_60', 'name': 'Alpha 60 (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'alpha_st', 'name': 'Alpha ST (bps)', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'position_age', 'name': 'Position Age (days)', 'type': 'numeric', 'format': Format.Format(precision=1, scheme=Format.Scheme.fixed)},
    {'id': 'no_trade', 'name': 'No Trade', 'type': 'numeric'},
]

PNL_CONT_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'notional', 'name': 'Position', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'opt_position', 'name': 'Opt Position', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'orig_trade_amt', 'name': 'Orig Trade Amt', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'dollars_done', 'name': 'Dollars Done', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'fraction_done', 'name': 'Fraction Done', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.percentage)},
    {'id': 'lbound', 'name': 'LBound', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'ubound', 'name': 'UBound', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
]


class TradingReports:
    """Main orchestrator class for all trading reports.

    Integrates all reporting components into a unified interface for
    generating comprehensive trading analytics. Manages report updates,
    data refresh, and dashboard generation for both historical and
    real-time trading analysis.

    Attributes:
        pnl_calculator: PNL calculator instance or ts_pnl_df directly
        data_loader: DataLoader instance for data access
        calcs: Calcs instance for calculations
        config: Configuration dictionary
        interval_secs: Update interval in seconds
        today_start_dt: Today's start datetime
        horizon_models: List of (horizon, model) tuples
        alpha_cols: List of alpha column names
        corr_checks_cols: List of correlation check column names
        sim_lookback_dirs: List of simulation directories
        update_ts: Last update timestamp

    Example:
        >>> pnl_calc = FillPnl(config, start, fills_source, bars_source)
        >>> tr = TradingReports(pnl_calculator=pnl_calc, config=config, today_start_dt=today)
        >>> tr.update_data()
        >>> plots = tr.get_trading_plots()
    """

    def __init__(self, config: dict, debug: bool = False) -> None:
        """Initialize with required data sources

        Args:
            config: Configuration dictionary
            interval_secs: Update interval in seconds
        """
        self.config = config
        self.debug = debug
        self.dir_manager = DirectoryManager()
        self.data_loader = DataLoader(config=self.config)
        self.calcs = Calcs(self.config, prod=False)
        self.today_start_dt = today()

        self.pnl_calculator = None
        self.live_bars = LiveBars()

        self.sim_lookback_dirs = [sim_dir for sim_dir in get_sim_dirs(f"{SIM_DIR}/long") if sim_dir.startswith('sim_lookback_')]
        self.as_of = None

        self.notrade_list = []

        self.horizon_models = extract_horizon_models(config, exclude_zero_weight=True)
        self.alpha_cols = [f"alpha_{m}_{h}" for h, m in self.horizon_models]
        self.horizons = sorted({h for h, _ in self.horizon_models})
        self.corr_checks_cols = []
        for horizon in self.horizons:
            self.corr_checks_cols.append(f"alpha_{horizon}")
        for col in self.alpha_cols:
            self.corr_checks_cols.append(col)

        self.security_ts_pnl_df = None
        self.security_daily_pnl_df = None
        self.portfolio_ts_pnl_df = None
        self.portfolio_daily_pnl_df = None
        self.today_trading_df = None
        self.st_alpha_df = None

        self.perf = {}
        self.pnl_opt_target_ts = None
        self.today_fill_breakdown = None
        self.sim_daily_ret = 0
        self.sim_daily_vol = 0

        self.load_data()

    def _load_and_update_targets(self) -> pd.DataFrame:
        target_cols = ['target_position', 'opt_position', 'lbound', 'ubound', 'dollars_done', 'fraction_done', 'orig_trade_amt', 'alpha_opt', 'trading', 'risk_1440']
        try:
            targets_df = load_targets(latest=True, opt=True).rename(columns={'position': 'opt_position'})
        except ValueError as e:
            logger.warning(f"No opt targets available, continuing without target data: {e}")
            for col in target_cols:
                self.security_daily_pnl_df[col] = 0
            self.security_daily_pnl_df['fraction_done'] = 1.0
            return self.security_daily_pnl_df

        if targets_df.empty:
            logger.warning("No targets found")
            for col in target_cols:
                self.security_daily_pnl_df[col] = 0
            self.security_daily_pnl_df['fraction_done'] = 1.0
            return self.security_daily_pnl_df

        target_max_ts = targets_df['ts'].max()
        targets_df = targets_df.loc[targets_df['ts'] == target_max_ts]
        targets_df['orig_trade_amt'] = targets_df['target_position'] - targets_df['opt_position']
        targets_df = pd.merge(
            self.security_daily_pnl_df,
            targets_df[['symbol', 'opt_position', 'lbound', 'ubound', 'orig_trade_amt', 'alpha_opt', 'risk_1440', 'target_position']],
            on='symbol', how='left'
        )
        targets_df['dollars_done'] = targets_df['notional'] - targets_df['opt_position']
        targets_df['fraction_done'] = remove_infs(targets_df['dollars_done'] / targets_df['orig_trade_amt']).fillna(1.0).clip(lower=0.0, upper=1.0)
        targets_df['trading'] = targets_df['target_position'] - targets_df['notional']
        targets_df['alpha_opt'] *= 10000
        targets_df['risk_1440'] *= 100
        self.pnl_opt_target_ts = target_max_ts
        return targets_df

    def _load_st_alpha(self) -> Optional[pd.DataFrame]:
        try:
            # Load all target files for today to get historical intraday alphas
            today_dt = today_date()
            st_alpha_df = load_targets(opt=False, latest=False, start_dt=date_to_start_dt(today_dt), end_dt=date_to_end_dt(today_dt))
        except ValueError as e:
            logger.warning(f"No targets available, continuing without target data: {e}")
            return None
        if st_alpha_df is not None:
            st_alpha_df = st_alpha_df.rename(columns={'position': 'st_position'})
            st_alpha_df['alpha_15'] *= 10000
            st_alpha_df['alpha_60'] *= 10000
            st_alpha_df['alpha_st'] = st_alpha_df['alpha_15'] + st_alpha_df['alpha_60']
            st_alpha_df = st_alpha_df[['ts', 'symbol', 'alpha_15', 'alpha_60', 'alpha_st']]
            st_alpha_df = make_symbol_venue(st_alpha_df)
            logger.info(f"Loaded st alpha until {st_alpha_df['ts'].max()}")

        return st_alpha_df

    def _make_current_trading_df(self) -> pd.DataFrame:

        # Filter for meaningful PNL or position
        trading_df = self.security_daily_pnl_df.copy()
        trading_df = trading_df.loc[(trading_df['net_pnl'].abs() > 1) | (trading_df['notional'].abs() > 1)]

        # Add short-term alpha (may be None if trading is stopped)
        if self.st_alpha_df is not None:
            st_alpha_df = self.st_alpha_df[self.st_alpha_df['ts'] == self.st_alpha_df['ts'].max()]
            trading_df = clean_poop(pd.merge(trading_df, st_alpha_df[['symbol', 'alpha_15', 'alpha_60', 'alpha_st']], on='symbol', how='left'))
        else:
            logger.warning("st_alpha_df not available, alpha columns will be NaN")
            trading_df['alpha_15'] = np.nan
            trading_df['alpha_60'] = np.nan
            trading_df['alpha_st'] = np.nan

        # Handle no-trade list first, before column selection
        self._read_notrade_list()
        trading_df['no_trade'] = 0
        trading_df.loc[trading_df.symbol.isin(self.notrade_list), 'no_trade'] = 1

        return trading_df

    def _load_financing_rates(self) -> pd.DataFrame:
        # Add financing rates
        financing_rates_df = load_single_live_bar_df(latest=True)
        if financing_rates_df is None:
            self.security_daily_pnl_df['last_funding_rate'] = np.float32(0)
            return self.security_daily_pnl_df
        security_daily_pnl_df = pd.merge(self.security_daily_pnl_df, financing_rates_df[['symbol', 'last_funding_rate']], on='symbol', how='left')
        security_daily_pnl_df['last_funding_rate'] *= 10000
        return security_daily_pnl_df

    def load_data(self):
        """Load and process data for trading reports"""
        logger.info("Loading MOST data...")

        self.pnl_calculator = BinancePnl(config=self.config, start=today(), end=dt.now(timezone.utc))
        security_ts_pnl_df = self.pnl_calculator.security_ts_pnl_df
        security_ts_pnl_df['unrealized_daily'] = security_ts_pnl_df['unrealized_pnl'].cumsum()
        security_ts_pnl_df['realized_daily'] = security_ts_pnl_df['realized_pnl'].cumsum()
        security_ts_pnl_df['total_pnl_daily'] = security_ts_pnl_df['net_pnl'].cumsum()
        security_ts_pnl_df['fees_usd_daily'] = security_ts_pnl_df['commission'].cumsum()
        security_ts_pnl_df['funding_income_daily'] = security_ts_pnl_df['funding_income'].cumsum()
        security_ts_pnl_df['dollars_traded_daily'] = security_ts_pnl_df['fill_dollars_abs'].cumsum()
        security_ts_pnl_df = security_ts_pnl_df.rename(columns={
            'gross_notional': 'notional_daily',
        })
        self.security_ts_pnl_df = security_ts_pnl_df

        self.st_alpha_df = self._load_st_alpha()

        self.security_daily_pnl_df = self.pnl_calculator.aggregate_by_security_date()
        self.security_daily_pnl_df = self._load_and_update_targets()
        self.security_daily_pnl_df = self._load_financing_rates()

        # Calculate portfolio percentage for each position
        total_gross_notional = self.security_daily_pnl_df['notional'].abs().sum()
        self.security_daily_pnl_df['portfolio_pct'] = self.security_daily_pnl_df['notional'].abs() / total_gross_notional * 100

        self.today_trading_df = self._make_current_trading_df()

        self.portfolio_ts_pnl_df = self.pnl_calculator.aggregate_portfolio(self.security_ts_pnl_df)
        self.portfolio_daily_pnl_df = self.pnl_calculator.aggregate_daily_portfolio()

        self.today_fill_breakdown = FillBreakdown(start=self.today_start_dt)
        self.today_fill_breakdown.load_fills(self.pnl_calculator.get_fills_df())

        self._load_long_term_sim_performance()
        self._update_performance()

        self.as_of = self.pnl_calculator.as_of

    def _load_long_term_sim_performance(self):
        """Load long-term simulation performance metrics"""
        if not self.sim_lookback_dirs:
            self.sim_daily_ret = 0
            self.sim_daily_vol = 1
            return

        sim_name = self.sim_lookback_dirs[-1]
        # Use /sims/long/ directory for sim_lookback_* simulations
        sim_path = f"{SIM_DIR}/long/{sim_name}"
        pnl_file = f"{sim_path}/pnl.calculator.csv"

        # Verify file exists
        if not os.path.exists(pnl_file):
            logger.warning(
                f"Simulation PnL file not found at '{pnl_file}'. "
                f"Setting default simulation metrics."
            )
            self.sim_daily_ret = 0
            self.sim_daily_vol = 1
            return

        sim_pnl_df = pd.read_csv(pnl_file, index_col=0)

        # Avoid division by zero or near-zero denominators when long equals short
        denom = sim_pnl_df['long'] - sim_pnl_df['short']
        # Filter out rows where denominator is too small (absolute value < 1.0)
        valid_mask = denom.abs() >= 1.0
        sim_pnl_df = sim_pnl_df[valid_mask]
        denom = denom[valid_mask]
        sim_pnl_df['ret'] = sim_pnl_df['pnl'].diff() / denom

        _, sim_config = get_config(config_file=f"{sim_path}/config.json")

        # Validate REOPTIMIZE_INTERVAL_MINS to avoid division by zero
        reopt_interval = sim_config.get('REOPTIMIZE_INTERVAL_MINS')
        if not reopt_interval or reopt_interval <= 0:
            logger.error(
                f"Invalid or missing REOPTIMIZE_INTERVAL_MINS in config: {reopt_interval}. "
                f"Using daily_scaler=1."
            )
            daily_scaler = 1
        else:
            daily_scaler = 1440 / reopt_interval

        # Clean NaN and infinite values before calculating statistics
        clean_ret = remove_infs(sim_pnl_df['ret']).dropna()
        self.sim_daily_ret = clean_ret.mean() * daily_scaler
        self.sim_daily_vol = clean_ret.std() * np.sqrt(daily_scaler)

    def _read_notrade_list(self):
        """Read list of symbols not to trade"""
        self.notrade_list = []
        try:
            with open(self.dir_manager.NOTRADE_FILE) as ff:
                for symbol in ff.readlines():
                    self.notrade_list.append(symbol.strip())
        except FileNotFoundError:
            logger.warning(f"No-trade file not found: {self.dir_manager.NOTRADE_FILE}")

    def update_symbol_dropdown(self, n_state: str) -> Tuple[List[dict], Optional[str]]:
        """Update dropdown menu for symbol selection"""
        # Use symbol_venue from index, not just symbol
        symbol_venue_list = sorted(self.security_ts_pnl_df.index.get_level_values('symbol_venue').unique())
        options = [{'label': sv, 'value': sv} for sv in symbol_venue_list]
        # Default to BTCUSDT_binance-futures if available
        value = next((sv for sv in symbol_venue_list if 'BTCUSDT' in sv), symbol_venue_list[0] if symbol_venue_list else None)
        return options, value

    def get_ts_display(self, n_state: str) -> str:
        """Get timestamp display text"""
        update_ts_str = self.as_of.strftime("%Y%m%d %H:%M") if self.as_of else 'Unknown'
        return f'Data as of: {update_ts_str}'

    @staticmethod
    def calc_return_metrics(pnl_df: pd.DataFrame) -> pd.DataFrame:
        # Handle division by zero by replacing 0 with NaN first (see binance_pnl.py pattern)
        safe_gross_notional = pnl_df['gross_notional'].replace(0, np.nan)
        safe_fill_dollars = pnl_df['fill_dollars_abs'].replace(0, np.nan)
        safe_balance = pnl_df['balance'].replace(0, np.nan)

        pnl_df['unlev_return_daily'] = (pnl_df['net_pnl'] / safe_gross_notional).fillna(0)
        pnl_df['lev_return_daily'] = (pnl_df['net_pnl'] / safe_balance).fillna(0)
        pnl_df['turnover'] = (pnl_df['fill_dollars_abs'] / safe_gross_notional).fillna(0)
        pnl_df['fees_bps_daily'] = (pnl_df['commission'] / safe_fill_dollars * 10000).fillna(0)
        pnl_df['funding_income_bps_daily'] = (pnl_df['funding_income'] / safe_gross_notional * 10000).fillna(0)
        return pnl_df

    def _update_performance(self):
        logger.info("Updating Performance metrics")
        if self.portfolio_daily_pnl_df is None:
            logger.warning("No portfolio daily PnL data available, setting default performance metrics")
            self.perf = {
                'today_unlev_ret': 0.0, 'today_lev_ret': 0.0, 'today_total_pnl': 0.0,
                'today_realized_pnl': 0.0, 'today_unrealized_pnl': 0.0, 'today_balance': 0.0,
                'today_notional': 0.0, 'today_leverage': 0.0, 'today_trading_volume': 0.0,
                'today_turnover': 0.0
            }
            return
        daily_pnl = self.portfolio_daily_pnl_df[
            ['commission', 'funding_income', 'fill_dollars_abs', 'net_pnl', 'gross_notional', 'balance']
        ].reset_index()

        daily_pnl = self.calc_return_metrics(daily_pnl)
        self.perf['today_unlev_ret'] = daily_pnl['unlev_return_daily'].tail(1).iloc[0]
        self.perf['today_lev_ret'] = daily_pnl['lev_return_daily'].tail(1).iloc[0]
        self.perf['today_total_pnl'] = daily_pnl['net_pnl'].tail(1).iloc[0]
        self.perf['today_realized_pnl'] = self.portfolio_daily_pnl_df['realized_pnl'].tail(1).iloc[0]
        self.perf['today_unrealized_pnl'] = self.portfolio_daily_pnl_df['unrealized_pnl'].tail(1).iloc[0]
        self.perf['today_balance'] = self.portfolio_daily_pnl_df['balance'].iloc[-1]
        self.perf['today_notional'] = daily_pnl['gross_notional'].iloc[-1]
        self.perf['today_leverage'] = self.perf['today_notional'] / self.perf['today_balance']
        self.perf['today_trading_volume'] = daily_pnl['fill_dollars_abs'].iloc[-1]
        self.perf['today_turnover'] = daily_pnl['turnover'].iloc[-1]
        logger.info(f"Perf {self.perf}")

    def get_performance_stats(self, n_state: str) -> Tuple[List[dict], List[dict], List[str], List[str]]:
        logger.info("Get Performance Stats")

        if self.portfolio_daily_pnl_df is None or self.portfolio_daily_pnl_df.empty:
            today_fundings = 0.0
            cum_market_ret = 0.0
            long_notional = 0.0
            short_notional = 0.0
        else:
            today_fundings = self.portfolio_daily_pnl_df['funding_income'].sum()
            # Use iloc[-1] to get the last cumulative value, not sum of all days
            cum_market_ret = self.portfolio_daily_pnl_df['logret_cum_wgtmkt'].iloc[-1] if 'logret_cum_wgtmkt' in self.portfolio_daily_pnl_df.columns else 0.0
            long_notional = self.portfolio_daily_pnl_df.iloc[-1]['long_notional']
            short_notional = self.portfolio_daily_pnl_df.iloc[-1]['short_notional']

        # Calculate funding bps with zero division protection
        today_notional = self.perf['today_notional']
        funding_bps = (today_fundings / today_notional * 10000) if today_notional != 0 else 0.0

        # Calculate sigma level with zero division protection
        sim_vol = self.sim_daily_vol if self.sim_daily_vol != 0 else 1.0
        sigma_level = (self.perf['today_unlev_ret'] - self.sim_daily_ret) / sim_vol
        win_ratio = self.today_fill_breakdown.win_ratio()

        # Calculate weighted average position age (weighted by absolute notional)
        # Use only the latest date to reflect current positions
        weighted_avg_position_age = 0.0
        required_cols = ['position_age', 'notional', 'date']
        if (self.security_daily_pnl_df is not None and
                not self.security_daily_pnl_df.empty and
                all(col in self.security_daily_pnl_df.columns for col in required_cols)):
            latest_date = self.security_daily_pnl_df['date'].max()
            latest_df = self.security_daily_pnl_df[self.security_daily_pnl_df['date'] == latest_date]
            abs_notional = latest_df['notional'].abs()
            total_abs_notional = abs_notional.sum()
            if total_abs_notional > 0:
                weighted_avg_position_age = (latest_df['position_age'] * abs_notional).sum() / total_abs_notional

        logger.info(f"Perf; {self.perf}")
        res = {
            'metrics': [
                "Today's Pnl",
                "Today's Realized Pnl",
                "Today's Unrealized Pnl",
                "Today's Levered Return",
                "Today's un-Levered Return",
                "Today's Market Return",
                "Current Binance Balance",
                "Current Notional",
                "Current Leverage",
                "Today's Funding Income",
                "Today's Funding Income bps",
            ],
            'today': [
                f"{fmoney(self.perf['today_total_pnl'])}",
                f"{fmoney(self.perf['today_realized_pnl'])}",
                f"{fmoney(self.perf['today_unrealized_pnl'])}",
                f"{self.perf['today_lev_ret'] * 100:.2f}%",
                f"{self.perf['today_unlev_ret'] * 100:.2f}%",
                f"{cum_market_ret:.2f}%",
                f"{fmoney(self.perf['today_balance'])}",
                f"{fmoney(self.perf['today_notional'])}",
                f"{self.perf['today_leverage']:.2f}",
                f"{fmoney(today_fundings)}",
                f"{funding_bps:.2f}",
            ],
        }
        res_right = {
            'metrics': [
                "Today's Win Ratio",
                "Today's Total Trades",
                "Today's Profitable Contracting Trades",
                "Today's Avg Gain per Trade",
                "Today's Avg Loss per Trade",
                "Today's Trading Volume",
                "Today's Turnover",
                "Today's Long / Short",
                "Today's Return Sigma Level",
                "Long-term Sim Daily Ret/Vol",
                "Weighted Avg Position Age (30d)",
            ],
            'today': [
                f"{win_ratio[0] * 100:.2f}%",
                f"{win_ratio[3]}",
                f"{win_ratio[1]} / {win_ratio[2]}",
                f"{fmoney(win_ratio[4])}",
                f"{fmoney(win_ratio[5])}",
                f"{fmoney(self.perf['today_trading_volume'])}",
                f"{self.perf['today_turnover']:.2f}",
                f"{fmoney(long_notional)} / {fmoney(short_notional)}",
                f"{sigma_level:.2f}",
                f"Ret: {self.sim_daily_ret * 100:.2f}%, Vol: {self.sim_daily_vol * 100:.2f}%",
                f"{weighted_avg_position_age:.1f} days",
            ],
        }
        data = [dict(zip(res, t)) for t in zip(*res.values())]
        data_right = [dict(zip(res_right, t)) for t in zip(*res_right.values())]
        pnl_headline = [
            'Pnl',
            *[html.Br() for _ in range(2)]
        ]
        trading_headline = [
            'Trading',
            *[html.Br() for _ in range(2)],
            f'Trader Target ts {self.pnl_opt_target_ts.round("s") if self.pnl_opt_target_ts else "N/A"}',
            *[html.Br() for _ in range(2)]
        ]
        return data, data_right, pnl_headline, trading_headline

    def intraday_pnl_figure(self, n_state: str) -> go.Figure:
        if self.portfolio_ts_pnl_df is None:
            logger.info("self.portfolio_ts_pnl_df not set yet for intraday pnl figure")
            return go.Figure()

        df = self.portfolio_ts_pnl_df.copy()

        if 'logret_cum_wgtmkt' not in df.columns:
            raise ValueError("logret_cum_wgtmkt column missing from portfolio_ts_pnl_df - check market return calculation")

        df = df.rename(columns={
            'realized_pnl_cum': 'realized_daily',
            'unrealized_pnl_cum': 'unrealized_daily',
            'net_pnl_cum': 'total_pnl_daily',
            'logret_cum_wgtmkt': 'market_ret_daily',
        })

        mkt_df = df[['market_ret_daily']].copy()
        df = df[['realized_daily', 'unrealized_daily', 'total_pnl_daily']]

        df = df.stack().rename('pnl').reset_index()
        df = df.rename(columns={'level_1': 'pnl_type'})

        fig = px.line(df, x='ts', y='pnl', title="Intraday Pnl", color='pnl_type')

        if not mkt_df['market_ret_daily'].isna().all():
            fig.add_trace(go.Scatter(
                x=mkt_df.index,  # Simple Index named 'ts', not MultiIndex
                y=mkt_df["market_ret_daily"],
                name="market_cum_ret",
                yaxis="y2",
            ))

        y_limit = 1.2 * df["pnl"].abs().max()
        y2_limit = 1.2 * mkt_df["market_ret_daily"].abs().max()

        fig.update_layout(
            yaxis={"range": [-y_limit, y_limit]},
            yaxis2={"overlaying": "y", "range": [-y2_limit, y2_limit], "side": "right", "ticksuffix": '%'},
            xaxis={"title": "ts"},
        )
        return fig

    def get_today_single_symbol_result(self, selected_symbol_venue: str, n_state: str) -> Tuple[List[Dict], List[Dict], go.Figure]:
        fills_df = self.pnl_calculator.get_fills_df()

        # Empty fills template for early returns
        empty_fills_df_side = {
            Side.BUY: pd.DataFrame(columns=['ts', 'fill_qty', 'fill_px']),
            Side.SELL: pd.DataFrame(columns=['ts', 'fill_qty', 'fill_px']),
        }

        # Handle no fills (trading stopped) - return early with empty figure
        if fills_df is None:
            logger.warning("fills_df not available for single symbol result")
            single_symbol_fig = self.today_single_symbol_figure(selected_symbol_venue, empty_fills_df_side, 0, 0)
            return [], [], single_symbol_fig

        if fills_df.empty:
            single_symbol_fig = self.today_single_symbol_figure(selected_symbol_venue, empty_fills_df_side, 0, 0)
            return [], [], single_symbol_fig

        # Filter for selected symbol
        fills_df = fills_df.loc[fills_df['symbol_venue'] == selected_symbol_venue]
        if fills_df.empty:
            single_symbol_fig = self.today_single_symbol_figure(selected_symbol_venue, empty_fills_df_side, 0, 0)
            return [], [], single_symbol_fig

        max_qty = fills_df['fill_qty'].max()
        min_qty = fills_df['fill_qty'].min()

        fills_df_side = {}
        # Compare with string values 'BUY'/'SELL' instead of enum (fills data uses strings)
        fills_df_side[Side.BUY] = fills_df.loc[fills_df['side'] == 'BUY', ['ts', 'fill_qty', 'fill_px']].sort_values('ts')
        fills_df_side[Side.SELL] = fills_df.loc[fills_df['side'] == 'SELL', ['ts', 'fill_qty', 'fill_px']].sort_values('ts')
        single_symbol_fig = self.today_single_symbol_figure(selected_symbol_venue, fills_df_side, max_qty, min_qty)
        return fills_df_side[Side.BUY].to_dict('records'), fills_df_side[Side.SELL].to_dict('records'), single_symbol_fig

    def today_single_symbol_figure(self, selected_symbol_venue: str, fills_df_side: Dict, max_qty: float, min_qty: float) -> go.Figure:
        if self.portfolio_ts_pnl_df is None:
            return go.Figure()

        if self.st_alpha_df is not None:
            st_alpha_df = self.st_alpha_df.loc[self.st_alpha_df['symbol_venue'] == selected_symbol_venue]
        else:
            st_alpha_df = pd.DataFrame()
        filtered_df = self.security_ts_pnl_df[self.security_ts_pnl_df.index.get_level_values('symbol_venue') == selected_symbol_venue].copy()

        # Calculate cumulative P&L for this symbol (after filtering to single symbol)
        if 'net_pnl' in filtered_df.columns:
            filtered_df['total_pnl_daily'] = filtered_df['net_pnl'].cumsum()
        if 'realized_pnl' in filtered_df.columns:
            filtered_df['realized_daily'] = filtered_df['realized_pnl'].cumsum()
        if 'unrealized_pnl' in filtered_df.columns:
            filtered_df['unrealized_daily'] = filtered_df['unrealized_pnl'].cumsum()

        # mark_price and index_price are time series from BinancePnl (close_mid is Binance mark price)
        filtered_df['mark_price'] = filtered_df['close_mid']

        # Calculate cumulative funding rate from funding_income (already in filtered_df from PnL data)
        # Only calculate where we have valid data - don't fill with fake values!
        if 'funding_income' in filtered_df.columns and 'notional' in filtered_df.columns:
            # Only calculate funding rate where we have valid notional (not zero, not NaN)
            # Don't fill NaN or zeros - that would be working with bad data
            valid_notional = filtered_df['notional'].abs() > 0
            if valid_notional.any():
                # Calculate funding rate only for valid rows
                funding_rate = pd.Series(index=filtered_df.index, dtype=float)
                funding_rate[valid_notional] = filtered_df.loc[valid_notional, 'funding_income'] / filtered_df.loc[valid_notional, 'notional'].abs()
                # Use log(1 + rate) for valid rates only
                funding_log_rate = np.log1p(funding_rate)
                # Cumsum will have NaN gaps where data is invalid (fail fast - don't fill with zeros)
                filtered_df['cum_realized_funding_log_rate'] = funding_log_rate.cumsum() * 10000
            # If no valid notional, don't create the column - chart won't plot (fail fast!)

        trace_attribute = {}
        # Use cumulative P&L columns (calculated above for single symbol)
        for col in ['total_pnl_daily', 'realized_daily', 'unrealized_daily']:
            trace_attribute[col] = {'line': {'dash': 'solid'}, 'row': 1, 'col': 1, 'secondary': False}
        for col in ['notional']:
            trace_attribute[col] = {'line': {'dash': 'dash'}, 'row': 1, 'col': 1, 'secondary': True}
        for col in ['mark_price', 'index_price']:
            trace_attribute[col] = {'line': {'dash': 'solid'}, 'row': 2, 'col': 1, 'secondary': False}
        for col in ['alpha_15', 'alpha_60', 'alpha_st']:
            trace_attribute[col] = {'line': {'dash': 'solid'}, 'row': 3, 'col': 1, 'secondary': False}
        for col in ['cum_realized_funding_log_rate']:
            trace_attribute[col] = {'line': {'dash': 'solid'}, 'row': 4, 'col': 1, 'secondary': False}

        fig = sp.make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
            subplot_titles=('Pnl & Notional', 'Mark px & Index Px & Funding adj Ret (%) & Fill Executions', 'Short Term Alphas (bps)', 'Cumulative Funding Rate (bps)'),
        )

        # Plot cumulative P&L, notional, and prices
        for col in ['notional', 'total_pnl_daily', 'realized_daily', 'unrealized_daily', 'mark_price', 'index_price']:
            # Only plot if column exists in filtered_df
            if col not in filtered_df.columns:
                continue
            trace = go.Scatter(
                x=filtered_df.index.get_level_values('ts'),
                y=filtered_df[col],
                name=f'{col.capitalize().replace("_", " ")}',
                mode='lines',
                line=trace_attribute[col]['line'],
                showlegend=True,
                legendgroup=trace_attribute[col]['row'],
            )
            fig.add_trace(trace, row=trace_attribute[col]['row'], col=trace_attribute[col]['col'], secondary_y=trace_attribute[col]['secondary'])

        # Only plot alphas if st_alpha_df has data for this symbol
        if not st_alpha_df.empty and 'ts' in st_alpha_df.columns:
            for col in ['alpha_15', 'alpha_60', 'alpha_st']:
                # Only plot if column exists in st_alpha_df
                if col not in st_alpha_df.columns:
                    continue
                trace = go.Scatter(
                    x=st_alpha_df['ts'],
                    y=st_alpha_df[col],
                    name=f'{col.capitalize()}',
                    mode='lines',
                    line=trace_attribute[col]['line'],
                    showlegend=True,
                    legendgroup=trace_attribute[col]['row'],
                )
                fig.add_trace(trace, row=trace_attribute[col]['row'], col=trace_attribute[col]['col'], secondary_y=trace_attribute[col]['secondary'])
        for side in [Side.BUY, Side.SELL]:
            # Only plot fills if this side has data
            if not fills_df_side[side].empty:
                fig.add_trace(go.Scatter(
                    x=fills_df_side[side]['ts'],
                    y=fills_df_side[side]['fill_px'],
                    mode='markers',
                    marker={
                        "size": 5 + (fills_df_side[side]['fill_qty'] - min_qty) / (max_qty - min_qty) * 20 if max_qty != min_qty else 5,
                        "color": 'green' if side == Side.BUY else 'red',
                        "opacity": 0.7,
                        "line": {"width": 1, "color": 'darkgreen' if side == Side.BUY else 'darkred'},
                    },
                    name=f'{side.capitalize()} Fills',
                    legendgroup=2,
                ), row=2, col=1)
                fig.update_traces(hovertemplate='<b>Time</b>: %{x}<br><b>Price</b>: %{y}<br><b>Quantity</b>: %{marker.size:.2f}<br>', selector={"name": f'{side.capitalize()} Fills'})

        col = 'cum_realized_funding_log_rate'
        # Only plot if column exists in filtered_df
        if col in filtered_df.columns:
            trace = go.Scatter(
                x=filtered_df.index.get_level_values('ts'),
                y=filtered_df[col],
                name=f'{col.capitalize()}',
                mode='lines',
                line=trace_attribute[col]['line'],
                showlegend=True,
                legendgroup=trace_attribute[col]['row'],
            )
            fig.add_trace(trace, row=trace_attribute[col]['row'], col=trace_attribute[col]['col'], secondary_y=trace_attribute[col]['secondary'])

        primary_traces = [trace for trace in fig.data if trace.yaxis is None or trace.yaxis == 'y']
        secondary_traces = [trace for trace in fig.data if trace.yaxis == 'y2']

        primary_values = []
        for trace in primary_traces:
            primary_values.extend([float(y) for y in trace.y if y is not None and not pd.isna(y)])

        secondary_values = []
        for trace in secondary_traces:
            secondary_values.extend([float(y) for y in trace.y if y is not None and not pd.isna(y)])

        if len(primary_values) >= 2 and len(secondary_values) >= 2:
            try:
                p_min = min(min(primary_values), 0)
                p_max = max(max(primary_values), 0)
                s_min = min(min(secondary_values), 0)
                s_max = max(max(secondary_values), 0)

                if p_min == p_max:
                    p_delta = 0.1 if p_min == 0 else abs(p_min) * 0.1
                    p_min -= p_delta
                    p_max += p_delta

                if s_min == s_max:
                    s_delta = 0.1 if s_min == 0 else abs(s_min) * 0.1
                    s_min -= s_delta
                    s_max += s_delta

                padding_perc = 0.3
                p_range_size = p_max - p_min
                s_range_size = s_max - s_min
                p_range = [p_min - p_range_size * padding_perc, p_max + p_range_size * padding_perc]
                s_range = [s_min - s_range_size * padding_perc, s_max + s_range_size * padding_perc]
                p_range_size = p_range[1] - p_range[0]
                s_range_size = s_range[1] - s_range[0]
                if p_range_size > 0:
                    p_domain_zero = -p_range[0] / p_range_size
                    p_domain_zero = max(0, min(1, p_domain_zero))
                else:
                    p_domain_zero = 0.5

                if s_range_size > 0:
                    s_domain_zero = -s_range[0] / s_range_size
                    s_domain_zero = max(0, min(1, s_domain_zero))
                else:
                    s_domain_zero = 0.5

                # Adjust secondary range to align zeros if needed
                if abs(p_domain_zero - s_domain_zero) > 0.01 and p_range_size > 0 and s_range_size > 0:
                    if p_domain_zero > s_domain_zero and p_domain_zero < 1:
                        s_range[0] = -s_range[1] * p_domain_zero / (1 - p_domain_zero)
                    elif p_domain_zero < s_domain_zero and p_domain_zero > 0:
                        s_range[1] = -s_range[0] * (1 - p_domain_zero) / p_domain_zero

                fig.update_yaxes(range=p_range, zeroline=True, row=1, col=1, secondary_y=False)
                fig.update_yaxes(range=s_range, zeroline=True, row=1, col=1, secondary_y=True)
                logger.info(f"Updated secondary yaxis ranges: Primary {p_range}, Secondary {s_range}")
            except Exception as e:
                logger.error(f"Error in updateing secondary yaxis ranges: {e}")

        fig.update_layout(
            title='Symbol PnL Over Time with Fill Executions',
            xaxis_title='Time',
            height=1200,
            legend1={"yanchor": "top", "xanchor": "left", "x": 1.02, "y": 1.0, "tracegroupgap": 250},
            legend2={"yanchor": "top", "xanchor": "left", "x": 1.02, "y": 0.75, "tracegroupgap": 250},
            legend3={"yanchor": "top", "xanchor": "left", "x": 1.02, "y": 0.5, "tracegroupgap": 250},
            legend4={"yanchor": "top", "xanchor": "left", "x": 1.02, "y": 0.25, "tracegroupgap": 250},
        )
        return fig

    def today_realized_by_trades_figure(self, n_state: str) -> go.Figure:
        if not self.pnl_calculator:
            logger.warning("PNL calculator not available for realized trades figure")
            return go.Figure()

        try:
            fills_df = self.pnl_calculator.get_fills_df()
            if fills_df is None:
                logger.warning("fills_df not available for realized trades figure")
                return go.Figure()

            contracting_fills_idx = ~fills_df['expanding']
            today_fills_idx = fills_df['date'] >= self.today_start_dt if self.today_start_dt else True
            contracting_fills_df = fills_df.loc[contracting_fills_idx & today_fills_idx]
            fig = px.histogram(
                contracting_fills_df,
                x='realized_pnl',
                nbins=50,
                labels={'realized_pnl': 'Realized PnL'},
            )
            fig.update_layout(title="Histogram of Today's Realized PnL", xaxis_title="Realized PnL", yaxis_title="Frequency")
        except Exception as e:
            logger.error(f"Error creating realized trades figure: {e}")
            fig = go.Figure()
        return fig

    def today_pnl_by_symbol_figure(self, n_state: str) -> Tuple[go.Figure, go.Figure, go.Figure]:
        pnl_figure_dict = {}

        # Handle empty or missing data
        if self.security_daily_pnl_df is None or self.security_daily_pnl_df.empty:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig

        title_dict = {'unrealized_pnl_tot_cum': "Total Unrealized Pnl By Symbol"}
        # Use BinancePnl column names (unrealized_pnl, realized_pnl)
        for case in ['unrealized_pnl', 'realized_pnl']:
            title_dict[case] = f"Today's {case.replace('_', ' ').title()}"

        for case in ['unrealized_pnl', 'realized_pnl', 'unrealized_pnl_tot_cum']:
            if case not in self.security_daily_pnl_df.columns:
                pnl_figure_dict[case] = go.Figure()
                continue
            df = self.security_daily_pnl_df.loc[self.security_daily_pnl_df[case] != 0].sort_values(by=case)
            if df.empty:
                pnl_figure_dict[case] = go.Figure()
            else:
                pnl_figure_dict[case] = px.bar(df, x='symbol', y=case, title=title_dict[case])

        return pnl_figure_dict['unrealized_pnl'], pnl_figure_dict['realized_pnl'], pnl_figure_dict['unrealized_pnl_tot_cum']

    @staticmethod
    def extract_model_alpha_value(alpha_str, alpha_name):
        alpha_dict = dict(pair.split(':') for pair in alpha_str.split(','))
        value = alpha_dict.get(alpha_name)
        return float(value) if value is not None else np.nan

    def get_alpha_breakdown_for_symbol(self, symbol_venue: str) -> List[Dict]:
        """Get alpha breakdown table for a specific symbol from targets.opt model_alpha_col."""
        targets_df = load_targets(latest=True, opt=True)
        if targets_df is None or targets_df.empty:
            return []

        symbol_data = targets_df[targets_df['symbol_venue'] == symbol_venue]
        if symbol_data.empty:
            return []

        model_alpha_col = symbol_data.iloc[0]['model_alpha_col']
        alpha_dict = dict(pair.split(':') for pair in model_alpha_col.split(','))

        breakdown = []
        for col, val_str in alpha_dict.items():
            val = float(val_str)
            if np.isnan(val):
                continue
            parts = col.split('_')
            breakdown.append({
                'horizon': int(parts[2]),
                'model': parts[1],
                'value_bps': round(val * 10000, 4),
            })

        breakdown.sort(key=lambda x: (-abs(x['value_bps']), x['horizon'], x['model']))
        return breakdown

    def today_alpha_by_symbol_figure(self, value: str, n_state: str) -> Tuple[go.Figure, List[dict], List[dict]]:
        try:
            alpha_df = load_targets(latest=True, opt=True)
        except ValueError as e:
            logger.warning(f"No opt targets available for alpha figure: {e}")
            alpha_df = None
        fig = go.Figure()
        data: List[dict] = []
        columns = [
            {'id': 'symbol', 'name': 'Symbol'},
            {
                'id': value,
                'name': ' '.join(value.split('_')).capitalize() + ' bps',
                'type': 'numeric',
                'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)
            },
        ]
        if alpha_df is None:
            return fig, data, columns

        if value not in alpha_df.columns:
            alpha_df[value] = alpha_df['model_alpha_col'].apply(lambda x: self.extract_model_alpha_value(x, value))
        alpha_df[value] *= 10000
        if not alpha_df[value].isna().all():
            fig = px.histogram(alpha_df, x=value, nbins=50, labels={value: value.capitalize()})
            fig.update_layout(title=f"Histogram of Latest Alpha at {alpha_df.ts.max()}", xaxis_title=value.capitalize(), yaxis_title="Frequency")
        return fig, alpha_df[['symbol', value]].sort_values(by=value).to_dict('records'), columns

    def buy_sell_figure(self, n_state: str) -> go.Figure:
        if self.portfolio_ts_pnl_df is None or self.as_of is None:
            return go.Figure()
        try:
            df = self.portfolio_ts_pnl_df.groupby('ts').agg({'fill_dollars_buy_cum': 'sum', 'fill_dollars_sell_cum': 'sum'})
            df = df.stack().rename('fill_dollars').reset_index()
            df = df.rename(columns={'level_1': 'side'})
            return px.line(df, x='ts', y='fill_dollars', color='side', title="Today's Buys vs Sells")
        except Exception as e:
            logger.error(f"Error creating buy/sell figure: {e}")
            return go.Figure()
