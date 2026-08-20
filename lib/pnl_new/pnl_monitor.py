import logging
from datetime import datetime as dt, timezone
from typing import Any, Dict

from lib.pnl_new.binance_pnl import BinancePnl
from lib.util import fmoney, fpct, date_to_str
from lib.util.logging_util import KeyLogger
from lib.util.time_util import date_to_start_dt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = KeyLogger(logger)

LARGE_LOSS_ALERT_PERC = -2


def _get_empty_pnl_dict() -> Dict[str, Any]:
    """Return an empty PnL dictionary with all fields set to 0."""
    return {
        'total_pnl_daily': 0.0,
        'realized_daily': 0.0,
        'unrealized_daily': 0.0,
        'fees_usd_daily': 0.0,
        'long': 0.0,
        'short': 0.0,
        'notional': 0.0,
        'dollars_traded_daily': 0.0,
        'fill_cnt_daily': 0,
        'cum_unrealized_pnl': 0.0,
        'funding_income': 0.0,
        'balance': 0.0,
        'net_balance': 0.0,
        'bnb_balance': 0.0,
        'bnb_amount': 0.0,
        'mtd_net_pnl': 0.0
    }

def _get_intraday_pnl_str(latest_pnl_dict: dict) -> str:
    total_notional = latest_pnl_dict['notional']
    total_balance = latest_pnl_dict['balance']

    ret_lev = latest_pnl_dict['total_pnl_daily'] / total_balance if total_balance > 0 else -1
    ret_unlev = latest_pnl_dict['total_pnl_daily'] / total_notional if total_notional != 0 else 0
    ret_unlev_unrealized_all = latest_pnl_dict['unrealized_pnl_tot_cum'] / total_notional if total_notional != 0 else 0

    bnb_balance = latest_pnl_dict.get('bnb_balance', 0.0)
    bnb_amount = latest_pnl_dict.get('bnb_amount', 0.0)
    msgs = (
        f"Daily Pnl: {fmoney(latest_pnl_dict['total_pnl_daily'], thousands_sep=True)}, Realized: {fmoney(latest_pnl_dict['realized_daily'], thousands_sep=True)}, Unrealized: {fmoney(latest_pnl_dict['unrealized_daily'], thousands_sep=True)}\n"
        f"MTD Net Pnl: {fmoney(latest_pnl_dict['mtd_net_pnl'], thousands_sep=True)}\n"
        f"Daily Return Levered: {fpct(ret_lev)}, Un-levered: {fpct(ret_unlev)}\n"
        f"Total Unrealized Pnl: {fmoney(latest_pnl_dict['unrealized_pnl_tot_cum'], thousands_sep=True)}, Return: {fpct(ret_unlev_unrealized_all)}\n"
        f"Size: {fmoney(latest_pnl_dict['long'], thousands_sep=True)} / {fmoney(latest_pnl_dict['short'], thousands_sep=True)}\n"
        f"Balance: {fmoney(total_balance, thousands_sep=True)}, Net Balance: {fmoney(latest_pnl_dict['net_balance'], thousands_sep=True)}, BNB: {bnb_amount:.2f} ({fmoney(bnb_balance, thousands_sep=True)})\n"
        f"Fees: {fmoney(latest_pnl_dict['fees_usd_daily'], thousands_sep=True)}, Funding income: {fmoney(latest_pnl_dict['funding_income'], thousands_sep=True)}\n"
        f"Today's Fills {fmoney(latest_pnl_dict['dollars_traded_daily'], thousands_sep=True)}, Count: {int(latest_pnl_dict['fill_cnt_daily'])}\n"
    )

    if ret_unlev < LARGE_LOSS_ALERT_PERC:
        logger.error(msgs, key='seeing large daily loss')

    return msgs

class PnlMonitorNew:
    def __init__(self, config: dict, alert_mins: int, debug: bool = False):
        self.config = config
        self.debug = debug
        self.alert_mins = alert_mins


    def calc_pnl(self) -> Dict[str, Any]:
        """Get the latest PnL records as a dictionary.

        Runs PnL calculation for today and returns the daily PnL record.

        Returns:
            Dictionary with latest daily PnL metrics including MTD net PnL
        """
        logger.info("Calculating pnl")

        end = dt.now(timezone.utc)
        month_start = date_to_start_dt(end.replace(day=1))

        # Load MTD data for month-to-date net PnL calculation
        mtd_binance_pnl = BinancePnl(config=self.config, start=month_start, end=end)
        portfolio_df = mtd_binance_pnl.aggregate_daily_portfolio()
        if portfolio_df is None or len(portfolio_df) == 0:
            logger.warning("No portfolio data available")
            return _get_empty_pnl_dict()

        # Calculate MTD net PnL (sum of all days in the month)
        mtd_net_pnl = portfolio_df['net_pnl'].sum()

        # Get latest daily record for today's metrics
        latest_pnl_dict = portfolio_df.tail(1).to_dict('records')[0]

        if self.debug:
            print(latest_pnl_dict)

        # Map column names from BinancePnl to expected format
        mapped_dict = {
            'total_pnl_daily': latest_pnl_dict.get('net_pnl', 0.0),
            'realized_daily': latest_pnl_dict.get('realized_pnl', 0.0),
            'unrealized_daily': latest_pnl_dict.get('unrealized_pnl', 0.0),
            'fees_usd_daily': latest_pnl_dict.get('commission', 0.0),
            'long': latest_pnl_dict.get('long_notional', 0.0),
            'short': latest_pnl_dict.get('short_notional', 0.0),
            'notional': latest_pnl_dict.get('gross_notional', 0.0),
            'dollars_traded_daily': latest_pnl_dict.get('fill_dollars_abs', 0.0),
            'fill_cnt_daily': latest_pnl_dict.get('fill_count', 0),
            'unrealized_pnl_tot_cum': latest_pnl_dict.get('unrealized_pnl_tot_cum', 0.0),
            'funding_income': latest_pnl_dict.get('funding_income', 0.0),
            'balance': latest_pnl_dict.get('balance', 0.0),
            'net_balance': latest_pnl_dict.get('net_balance', 0.0),
            'bnb_balance': latest_pnl_dict.get('bnb_balance', 0.0),
            'bnb_amount': latest_pnl_dict.get('bnb_amount', 0.0),
            'mtd_net_pnl': mtd_net_pnl
        }
        return mapped_dict

    def run(self):
        today_str = date_to_str(dt.now(timezone.utc))
        logger.info(f"Updating Binance Data and running Pnl Numbers {today_str}")
        latest_pnl_dict = self.calc_pnl()
        msgs = _get_intraday_pnl_str(latest_pnl_dict=latest_pnl_dict)
        return msgs
