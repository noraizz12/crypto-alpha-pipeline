import logging
from datetime import datetime as dt, timedelta as td
from datetime import timezone
from typing import Any, Dict, Optional

import pandas as pd

from lib.util import fmoney, fpct, date_to_str
from lib.util.config import get_config
from lib.util.logging_util import KeyLogger
from lib.util.time_util import today
from lib.util.util import PNL_START_DATE
from .fill_counter import FillCounter
from .fill_pnl import FillPnl
from lib.data.loaders import load_balances, load_funding_income

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = KeyLogger(logger)

LARGE_LOSS_ALERT_PERC = -2


class PnlMonitor:
    def __init__(self, alert_mins: int, debug: bool = False):
        self.debug = debug
        self.fill_counter = FillCounter(no_fill_alert_mins=alert_mins)
        self.config = get_config()[1]
        self.pnl_calculator = FillPnl(config=self.config, start=PNL_START_DATE, end=today(), fills_source='binance')
        self.pnl_calculator.run_pnl_calculation(update=False)

    def calc_pnl(self, update: bool, fundings_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get the latest PnL records as a dictionary.

        Runs PnL calculation from PNL_START_DATE to today and returns
        the most recent daily PnL record.

        Returns:
            Dictionary with latest daily PnL metrics including cumulative unrealized PnL
        """
        logger.info(f"Calculating pnl {update=}")
        self.pnl_calculator.run_pnl_calculation(update=update, fundings_df=fundings_df)
        daily_pnl_df = self.pnl_calculator.run_pnl_aggregation()
        latest_pnl_dict = daily_pnl_df.tail(1).to_dict('records')[0]
        latest_pnl_dict['cum_unrealized_pnl'] = daily_pnl_df['unrealized_daily'].sum()
        return latest_pnl_dict

    @staticmethod
    def _get_intraday_pnl_str(latest_pnl_dict: dict, total_balance: float, funding: float) -> str:
        total_notional = latest_pnl_dict['long'] - latest_pnl_dict['short']
        ret_lev = latest_pnl_dict['total_pnl_daily'] / total_balance if total_balance > 0 else -1
        ret_unlev = latest_pnl_dict['total_pnl_daily'] / total_notional
        ret_unlev_unrealized_all = latest_pnl_dict['cum_unrealized_pnl'] / total_notional

        msgs = (
            f"Daily Pnl: {fmoney(latest_pnl_dict['total_pnl_daily'], thousands_sep=True)}, Realized: {fmoney(latest_pnl_dict['realized_daily'], thousands_sep=True)}, Unrealized: {fmoney(latest_pnl_dict['unrealized_daily'], thousands_sep=True)}\n"
            f"Daily Return Levered: {fpct(ret_lev)}, Un-levered: {fpct(ret_unlev)}\n"
            f"Total Unrealized Pnl: {fmoney(latest_pnl_dict['cum_unrealized_pnl'], thousands_sep=True)}, Return: {fpct(ret_unlev_unrealized_all)}\n"
            f"Size: {fmoney(latest_pnl_dict['long'], thousands_sep=True)} / {fmoney(latest_pnl_dict['short'], thousands_sep=True)}\n"
            f"Balance: {fmoney(total_balance, thousands_sep=True)}\n"
            f"Fees: {fmoney(latest_pnl_dict['fees_usd_daily'], thousands_sep=True)}, Funding income: {fmoney(funding, thousands_sep=True)}\n"
            f"Today's Fills {fmoney(latest_pnl_dict['dollars_traded_daily'], thousands_sep=True)}, Count: {int(latest_pnl_dict['fill_cnt_daily'])}\n"
        )

        if ret_unlev < LARGE_LOSS_ALERT_PERC:
            logger.error(msgs, key='seeing large daily loss')

        return msgs

    def run(self, update: bool = False):
        now = dt.now(timezone.utc)
        today_str = date_to_str(now.date())
        logger.info(f"Updating Binance Data and running Pnl Numbers {today_str} {update=}")

        balances_df = load_balances(start_date=today() - td(days=1))
        total_balance = float(balances_df['balance'].tail(1).iloc[0])

        fundings_df = load_funding_income(start_date=today())
        if fundings_df is None or len(fundings_df) == 0:
            logger.warning(f"No funding income data available for today")
            funding = 0.0
        else:
            funding = float(fundings_df['funding_income'].sum())

        latest_pnl_dict = self.calc_pnl(update=update, fundings_df=fundings_df)
        fill_cnt = self.pnl_calculator.get_last_day_fill_cnt()
        self.fill_counter.update_fill_cnt(fill_cnt, now)
        msgs = self._get_intraday_pnl_str(latest_pnl_dict=latest_pnl_dict, total_balance=total_balance, funding=funding)
        return msgs
