from datetime import datetime as dt, timezone
import logging

from lib.util import clear_alert, raise_alert, HIGH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FillCounter:
    def __init__(self, no_fill_alert_mins: int):
        self.no_fill_alert_mins = no_fill_alert_mins
        self.fill_cnt = 0
        self.fill_cnt_increase_ts = dt.now(timezone.utc)
        self.no_fill_alert = False

    def update_fill_cnt(self, new_fill_cnt: int, new_fill_cnt_ts: dt):
        logger.info(f"At {new_fill_cnt_ts}, new order cnt {new_fill_cnt} with previous order cnt {self.fill_cnt} updated at {self.fill_cnt_increase_ts}")
        if (new_fill_cnt > self.fill_cnt) or (new_fill_cnt_ts.day != self.fill_cnt_increase_ts.day):
            self.fill_cnt = new_fill_cnt
            self.fill_cnt_increase_ts = new_fill_cnt_ts
            if self.no_fill_alert:
                clear_alert(key=f'No fills over {self.no_fill_alert_mins} mins')
                self.no_fill_alert = False

        elif (new_fill_cnt_ts - self.fill_cnt_increase_ts).total_seconds() > self.no_fill_alert_mins * 60:
            msg = f"At {new_fill_cnt_ts}, no new fills since {self.fill_cnt_increase_ts} with order cnt {self.fill_cnt}"
            raise_alert(key=f'No fills over {self.no_fill_alert_mins} mins', priority=HIGH, description=msg)
            self.no_fill_alert = True
