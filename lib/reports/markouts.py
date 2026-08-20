import logging
from datetime import datetime as dt, date
from datetime import timedelta as td
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from lib.data.live_bars import LiveBars
from lib.util.config import extract_horizons
from lib.data import load_raw_targets_alpha, load_sim_data, load_binance_fills, load_orders
from lib.data.dataloader import DataLoader
from lib.util.dataframes import DF_INDEX, concat, get_min_max_ts, make_date, make_symbol_venue, merge_on_index, remove_infs, set_index
from lib.util.time_util import to_datetime, yesterday, yesterday_date, date_to_start_dt, date_to_end_dt, datetime_series_to_int64
from lib.trader.trading import Side
from lib.util.util import TARDIS_EXCHANGE
from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

MARKOUT_MINUTES = [-60, -5, 5, 15, 60, 120, 240, 360]
MARKOUT_MINUTES_SIM = [5, 15, 60, 120, 240]


class FillMarkouts:
    def __init__(self, config: dict, existing_bars_df: Optional[pd.DataFrame] = None, existing_orders_df: Optional[pd.DataFrame] = None, existing_fills_df: Optional[pd.DataFrame] = None) -> None:
        self.start_dt = None
        self.end_dt = None
        self.save_file = False
        self.bars_df = existing_bars_df
        self.orders_df = existing_orders_df
        self.raw_orders_df = None
        self.fills_df = existing_fills_df
        self.alpha_df = None
        self.markouts_df = None
        self.model_alpha_markouts_df = None
        self.config = config
        self.opt_offset_mins = config['OPT_OFFSET_MINS']
        self.reoptimize_interval_mins = config['REOPTIMIZE_INTERVAL_MINS']
        self.opt_horizon = config['OPT_HORIZON']
        self.max_alpha = config['MAX_ALPHA']
        self.horizons = extract_horizons(config)
        self.alpha_str_cols = []
        self.model_alpha_cols = []
        self.alpha_str_weights = {}
        self.live_bars = LiveBars(use_new=True)

    def set_start_end(self, start: Optional[dt] = None, end: Optional[dt] = None) -> None:
        if end is not None:
            self.end_dt = end
        else:
            self.end_dt = yesterday()

        if start is not None:
            self.start_dt = start
        else:
            self.start_dt = self.end_dt - td(days=7)

        assert self.start_dt <= self.end_dt

    def calculate_markouts(self, start: Optional[dt] = None, end: Optional[dt] = None, aggression_horizon: int = 240, save: bool = False, debug: bool = False) -> None:
        self.save_file = save
        self.set_start_end(start, end)
        self.prepare_markouts_data()
        self.show_markouts_stats(aggression_horizon, debug)

    def load_bars_df(self, start_date: date, end_date: date, data_source: Literal["file", "preload"] = "file", preload_bars_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if data_source == "preload" and preload_bars_df is not None:
            return preload_bars_df
        data_loader = DataLoader()
        # Use horizon-suffixed column names for horizon=1 bars (e.g., vwap_1, volume_1)
        # close_mid doesn't need suffix as it's not horizon-specific
        cols = ['vwap_1', 'close_mid', 'volume_1', 'dvolume_1', 'high_trade_1', 'low_trade_1', 'update_cnt_1']

        # Load historical bars
        hist_end_date = min(end_date + td(days=1), yesterday_date())
        hist_start_date = start_date - td(days=1)

        bars_df = data_loader.load_bars(
            horizon=1,
            start_date=hist_start_date,
            end_date=hist_end_date,
            cols=cols,
        )

        # Rename columns to remove _1 suffix for downstream compatibility
        if bars_df is not None:
            rename_map = {
                'vwap_1': 'vwap',
                'volume_1': 'volume',
                'dvolume_1': 'dvolume',
                'high_trade_1': 'high_trade',
                'low_trade_1': 'low_trade',
                'update_cnt_1': 'update_cnt'
            }
            bars_df = bars_df.rename(columns=rename_map)

        # Determine the start date for live bars
        # If historical bars exist, start from where they end; otherwise start from request start
        if bars_df is not None:
            hist_max_date = bars_df.index.get_level_values('ts').max().date()
            live_start_date = hist_max_date + td(days=1)
        else:
            live_start_date = start_date

        # Load live bars if we need data beyond what historical bars provide
        if live_start_date <= end_date:
            live_start_dt = date_to_start_dt(live_start_date)
            live_end_dt = date_to_end_dt(end_date)
            live_bars_df = self.live_bars.load_live_bars(start_dt=live_start_dt, end_dt=live_end_dt)
            if live_bars_df is not None:
                # Live bars use non-suffixed column names
                live_cols = ['vwap', 'close_mid', 'volume', 'dvolume', 'high_trade', 'low_trade', 'update_cnt']
                available_cols = [c for c in live_cols if c in live_bars_df.columns]
                live_bars_df = live_bars_df[available_cols]

                if bars_df is not None:
                    bars_df = concat([bars_df, live_bars_df])
                else:
                    bars_df = live_bars_df

        return bars_df

    def get_alpha_cols(self) -> None:
        self.model_alpha_cols = []
        self.alpha_str_cols = []
        self.alpha_str_weights = {}
        for horizon in self.horizons:

            #XXX REMOVE ME
            if horizon == 15:
                continue


            for fcast in self.config['FCASTS'][str(horizon)]['models']:
                weight = float(fcast['weight'])
                if weight == 0:
                    continue
                name = fcast['name']
                mult = 1.0
                if horizon > self.opt_horizon:
                    mult = self.opt_horizon / horizon
                alpha_str_col = f'alpha_{name}_{horizon}'
                self.alpha_str_cols.append(alpha_str_col)
                self.alpha_str_weights[alpha_str_col] = (weight, mult)
                if weight > 0:
                    self.model_alpha_cols.append(alpha_str_col + '_mom')
                    self.model_alpha_cols.append(alpha_str_col + '_rev')

    def load_orders_df(self, start_date: date, end_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        orders_df = load_orders(start_date=start_date, end_date=end_date)
        orders_df['ts'] = orders_df['exch_ts'].dt.ceil('min')
        orders_df['venue'] = TARDIS_EXCHANGE
        orders_df = make_symbol_venue(orders_df)

        ### XXX @wenyu -- i'm adding this to get the report to run, but it should not drop duplicates, but just aggregate them.
        orders_df = orders_df.drop_duplicates(subset=DF_INDEX)

        # Convert acked_ts to numeric (nanoseconds) BEFORE setting index - datetime can't be unstacked
        orders_df['acked_ts_ns'] = datetime_series_to_int64(orders_df['acked_ts'], context='acked_ts')

        orders_df = set_index(orders_df)
        # keep raw order records
        raw_orders_df = orders_df.copy()

        # Get alpha columns before unstack so we can select only needed columns
        self.get_alpha_cols()

        # Select only needed columns before unstack (use numeric acked_ts, not datetime)
        needed_cols = ['aggression', 'acked_ts_ns'] + [col for col in self.alpha_str_cols if col in orders_df.columns]
        orders_df = orders_df[needed_cols]

        # this will forward fill order information till the next order, use it when we merge orders with fills
        if len(orders_df) > 0:
            orders_df = orders_df.unstack().resample('Min').ffill().stack(future_stack=True)
            # Convert acked_ts back from nanoseconds to datetime
            if 'acked_ts_ns' in orders_df.columns:
                orders_df['acked_ts'] = pd.to_datetime(orders_df['acked_ts_ns'], unit='ns', utc=True)
                orders_df = orders_df.drop(columns=['acked_ts_ns'])
            else:
                # Column name might be different after stack - check all columns
                logger.warning(f"acked_ts_ns not in columns after stack. Available: {orders_df.columns.tolist()}")
                orders_df['acked_ts'] = pd.NaT
                orders_df['acked_ts'] = orders_df['acked_ts'].dt.tz_localize('UTC')
        else:
            logger.warning("No orders data to process")
            orders_df['acked_ts'] = pd.NaT
            orders_df['acked_ts'] = orders_df['acked_ts'].dt.tz_localize('UTC')
        for alpha_str_col in self.alpha_str_cols:
            if alpha_str_col not in orders_df.columns:
                orders_df[alpha_str_col] = np.nan
            orders_df[alpha_str_col] = orders_df[alpha_str_col].replace('None', np.nan).astype(float)
        raw_orders_df['orig_qty'] = raw_orders_df['orig_qty'].replace('None', np.nan).astype(float).fillna(0)
        raw_orders_df['order_signed_qty'] = raw_orders_df['orig_qty']
        raw_orders_df.loc[raw_orders_df.order_side == Side.SELL, 'order_signed_qty'] = -raw_orders_df.loc[raw_orders_df.order_side == Side.SELL, 'orig_qty']
        return orders_df[['aggression', 'acked_ts'] + self.alpha_str_cols], raw_orders_df[['aggression', 'order_signed_qty']]

    def load_fills_df(
            self,
            start_date: date,
            end_date: date,
            data_source: Literal["file", "preload"] = "file",
            preload_fills_df: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        if  data_source == 'preload' and preload_fills_df is not None:
            logger.warning("Using existing fills")
            fills_df = preload_fills_df
            # XXX TO MATCH LOAD BINANCE LOAD FILLS.  just getting this working
            fills_df['sign'] = 1
            fills_df.loc[fills_df['side'] == 'SELL', 'sign'] = -1.0
            fills_df['fill_signed_qty'] = fills_df['fill_qty'] * fills_df['sign']
            fills_df['fill_dollars'] = fills_df['fill_signed_qty'] * fills_df['fill_px']
        else:
            fills_df = load_binance_fills(start_date=start_date, end_date=end_date)
            if fills_df is None:
                logger.warning(f"Could not load fills from {data_source}")
                return None

        fills_df = fills_df[['ts', 'symbol', 'fill_px', 'fill_signed_qty', 'fill_dollars', 'sign', 'date']]
        fills_df['raw_ts'] = fills_df['ts']
        fills_df['ts'] = fills_df['ts'].dt.ceil('min')
        fills_df['venue'] = TARDIS_EXCHANGE
        fills_df = make_symbol_venue(fills_df)
        # aggregate first as there could be multiple fills at the same ts for one symbol venue
        fills_df = fills_df.groupby(DF_INDEX, observed=False).agg({
            'fill_signed_qty': 'sum',
            'fill_dollars': 'sum',
        })
        fills_df['sign'] = 1
        fills_df.loc[fills_df['fill_signed_qty'] < 0, 'sign'] = -1

        fills_df = merge_on_index(fills_df, self.orders_df)
        fills_df['time_to_fill'] = (fills_df.index.get_level_values('ts') - fills_df['acked_ts']).dt.total_seconds() / 60

        # not exact, doesn't cover buy/sell within a same minute of same sec
        agg_dict = {
            'fill_signed_qty': 'sum',
            'fill_dollars': 'sum',
            'sign': 'last',
            'aggression': 'last',
            'time_to_fill': 'last',
        }
        for alpha_str in self.alpha_str_cols:
            agg_dict[alpha_str] = 'last'

        fills_df = fills_df.groupby(DF_INDEX, observed=False).agg(agg_dict)
        fills_df = make_date(fills_df)
        fills_df['avg_fill_px'] = fills_df['fill_dollars'] / fills_df['fill_signed_qty']
        return fills_df

    def load_model_alpha_df(self, start_dt: Optional[dt] = None, end_dt: Optional[dt] = None) -> None:
        start_dt = start_dt if start_dt is not None else self.start_dt
        end_dt = end_dt if end_dt is not None else self.end_dt
        alpha_df = load_raw_targets_alpha(start_dt, end_dt, cols=['close_mid'] + self.model_alpha_cols, skip_log=False)
        if alpha_df is None:
            logger.warning(f"No alpha data found for {start_dt} to {end_dt}")
            self.alpha_df = None
            return
        alpha_df = alpha_df.rename(columns={'close_mid': 'alpha_model_px'})
        alpha_df[self.model_alpha_cols] = alpha_df[self.model_alpha_cols].apply(np.sign)
        self.alpha_df = make_date(alpha_df)

    @staticmethod
    def _get_min_existing_start_dt(min_existing_start_dt: dt, df: pd.DataFrame) -> dt:
        if df is not None:
            _, max_ts = get_min_max_ts(df)
            min_existing_start_dt = min(max_ts - td(days=1), min_existing_start_dt)
        return min_existing_start_dt

    @staticmethod
    def _generate_updated_markouts_data(start_dt: dt, df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if new_df is None:
            return df
        if df is not None:
            hist_df = df.loc[df.index.get_level_values('ts') < start_dt]
            return concat([hist_df, new_df])
        return new_df

    def get_markouts_start_dt(self) -> dt:
        if self.bars_df is not None and self.orders_df is not None and self.fills_df is not None:
            min_existing_start_dt = self.end_dt
            for df in [self.bars_df, self.orders_df, self.fills_df]:
                min_existing_start_dt = self._get_min_existing_start_dt(min_existing_start_dt, df)
            start_dt = max(self.start_dt, min_existing_start_dt)
        else:
            start_dt = self.start_dt

        return start_dt

    def prepare_markouts_data(self, data_source: Literal["file", "preload"] = "file", preload_bars_df: Optional[pd.DataFrame] = None, preload_fills_df: Optional[pd.DataFrame] = None) -> None:
        end_date = self.end_dt.date()
        start_dt = self.get_markouts_start_dt()
        start_date = start_dt.date()

        new_bars_df = self.load_bars_df(start_date, end_date, data_source, preload_bars_df)
        self.bars_df = self._generate_updated_markouts_data(start_dt, self.bars_df, new_bars_df)
        if self.bars_df is not None:
            self.bars_df = make_date(self.bars_df)

        new_orders_df, new_raw_orders_df = self.load_orders_df(start_date, end_date)
        self.orders_df = self._generate_updated_markouts_data(start_dt, self.orders_df, new_orders_df)
        self.raw_orders_df = self._generate_updated_markouts_data(start_dt, self.raw_orders_df, new_raw_orders_df)

        new_fills_df = self.load_fills_df(start_date, end_date, data_source, preload_fills_df)
        self.fills_df = self._generate_updated_markouts_data(start_dt, self.fills_df, new_fills_df)

    def get_markouts_px(
            self,
            signal_df: pd.DataFrame,
            bars_df: pd.DataFrame,
            mins: int,
    ) -> pd.DataFrame:
        signal_plus_mins = signal_df.reset_index().copy()
        signal_plus_mins[f'{mins}_min_ahead_ts'] = signal_plus_mins['ts'] + td(minutes=mins)
        signal_plus_mins['ts'] = signal_plus_mins[f'{mins}_min_ahead_ts']
        signal_plus_mins = set_index(signal_plus_mins)
        markouts_plus_mins_df = merge_on_index(bars_df, signal_plus_mins, how='inner')
        return markouts_plus_mins_df

    def run_model_alpha_markouts(self) -> None:
        if self.alpha_df is None:
            logger.warning("No alpha data available for model alpha markouts")
            self.model_alpha_markouts_df = None
            return
        if self.bars_df is None:
            logger.warning("No bars data available for model alpha markouts")
            self.model_alpha_markouts_df = None
            return
        rows = []
        for dd in pd.date_range(self.alpha_df['date'].min(), self.alpha_df['date'].max()):
            alpha_by_date_df = self.alpha_df[self.alpha_df['date'] == dd]
            bars_by_date_df = self.bars_df[self.bars_df['date'] == dd]
            logger.info(f"Running for {dd}, alpha cnt {len(alpha_by_date_df)}")
            if len(alpha_by_date_df) == 0:
                continue
            for mins in MARKOUT_MINUTES:
                markouts_plus_mins_df = self.get_markouts_px(alpha_by_date_df, bars_by_date_df, mins)
                markouts_plus_mins_df[f'{mins}_min_ret'] = markouts_plus_mins_df['close_mid'] / markouts_plus_mins_df['alpha_model_px'] - 1
                for alpha_col in self.model_alpha_cols:
                    ret_bps = 10000.0 * (markouts_plus_mins_df[alpha_col] * markouts_plus_mins_df[f'{mins}_min_ret']).mean()
                    rows.append([f"mins:{mins}", alpha_col, ret_bps, dd.date()])
        model_alpha_markouts_df = pd.DataFrame.from_records(rows, columns=['name', 'model_type', 'ret_bps', 'date'])
        self.model_alpha_markouts_df = model_alpha_markouts_df

    def markout_mins(
            self,
            fills_df: pd.DataFrame,
            bars_df: pd.DataFrame,
            mins: int,
            name: str,
    ) -> List[Any]:
        date = fills_df['date'].max()
        markouts_plus_mins_df = self.get_markouts_px(fills_df, bars_df, mins)
        markouts_plus_mins_df[f'{mins}_min_price_shortfall'] = (markouts_plus_mins_df['avg_fill_px'] - markouts_plus_mins_df['close_mid']) * markouts_plus_mins_df['sign']
        markouts_plus_mins_df[f'{mins}_min_price_shortfall_dollars'] = markouts_plus_mins_df[f'{mins}_min_price_shortfall'] * markouts_plus_mins_df['fill_signed_qty'].abs()
        markouts_plus_mins_df[f'{mins}_min_price_shortfall_bps'] = markouts_plus_mins_df[f'{mins}_min_price_shortfall_dollars'] / markouts_plus_mins_df['fill_dollars'].abs()
        shortfall_dollars = markouts_plus_mins_df[f'{mins}_min_price_shortfall_dollars'].fillna(0).sum()
        fill_dollars = markouts_plus_mins_df['fill_dollars'].fillna(0).abs().sum()
        shortfall_dollars_bps = 10000.0 * shortfall_dollars / fill_dollars if fill_dollars > 0 else 0.0
        return [name, shortfall_dollars, shortfall_dollars_bps, fill_dollars, date]

    def run_markouts(
            self,
            markouts_df: pd.DataFrame,
            fills_by_date_df: pd.DataFrame,
            bars_by_date_df: pd.DataFrame,
            aggression_horizon: Optional[int] = None,
            debug: bool = False,
    ) -> List[List[Any]]:
        if debug:
            markouts_df['vwap_shortfall'] = (markouts_df['avg_fill_px'] - markouts_df['vwap']) * markouts_df['sign']
            markouts_df['vwap_shortfall_dollars'] = markouts_df['vwap_shortfall'] * markouts_df['fill_signed_qty'].abs()
            markouts_df['hl_pct'] = remove_infs(markouts_df['sign'] * (markouts_df['avg_fill_px'] - markouts_df['low_trade']) / (markouts_df['high_trade'] - markouts_df['low_trade']).abs()).fillna(0)
            markouts_df['hl_pct_wgt'] = markouts_df['hl_pct'] * markouts_df['fill_signed_qty'].abs()

            fill_dollars = markouts_df['fill_dollars'].fillna(0).abs().sum()
            vwap_shortfall_dollars = markouts_df['vwap_shortfall_dollars'].fillna(0).sum()
            vwap_shortfall_bps = 10000.0 * vwap_shortfall_dollars / fill_dollars if fill_dollars != 0 else 0
            hl_pct_wgt = markouts_df['hl_pct_wgt'].fillna(0).sum()
            hl_pct = hl_pct_wgt / fill_dollars if fill_dollars != 0 else 0

            logger.info(f"1Min Vwap Shortfall ${vwap_shortfall_dollars:.0f} bps: {vwap_shortfall_bps:.2f} fill dollars: ${fill_dollars:.0f}")
            logger.info(f"1Min HLpct: {hl_pct:.2f}")

        lns = []
        for mins in MARKOUT_MINUTES:
            lns.append(self.markout_mins(fills_by_date_df, bars_by_date_df, mins=mins, name=f"mins:{mins}"))
        for aggression in sorted(markouts_df['aggression'].unique()):
            agg_fills_df = fills_by_date_df[fills_by_date_df['aggression'] == aggression]
            if aggression_horizon is not None:
                lns.append(self.markout_mins(agg_fills_df, bars_by_date_df, mins=aggression_horizon, name=f"aggression:{aggression}"))
            else:
                for mins in MARKOUT_MINUTES:
                    lns.append(self.markout_mins(agg_fills_df, bars_by_date_df, mins=mins, name=f"aggression:{aggression};mins:{mins}"))
        return lns

    def show_markouts_stats(self, aggression_horizon: Optional[int] = None, debug: bool = False) -> None:
        markouts_df = merge_on_index(self.bars_df, self.fills_df, how='left', suffixes=('_bars', ''))
        if debug:
            self.run_markouts(markouts_df[:], self.fills_df[:], self.bars_df[:], aggression_horizon, debug)
        rows = []
        for dd in pd.date_range(self.fills_df['date'].min(), self.fills_df['date'].max()):
            day_markouts_df = markouts_df[markouts_df['date'] == dd]
            day_fills_df = self.fills_df[self.fills_df['date'] == dd]
            day_bars_df = self.bars_df[self.bars_df['date'] == dd]
            logger.info(f"Running for {dd}, fills cnt {len(day_fills_df)}")
            if len(day_fills_df) == 0:
                logger.info(f"No fills for {dd}")
                continue
            rows += self.run_markouts(day_markouts_df, day_fills_df, day_bars_df, aggression_horizon, debug)

        df = pd.DataFrame.from_records(rows, columns=['name', 'shortfall_dollars', 'shortfall_dollars_bps', 'fill_dollars', 'date'])
        df = df[df['fill_dollars'] != 0]
        if self.save_file:
            df.to_csv('markouts.csv')
        self.markouts_df = df

    def get_order_alpha_pnl(self, fills_pnl_df: pd.DataFrame) -> pd.DataFrame:
        fills_pnl_df['ts'] = fills_pnl_df['ts'].dt.ceil('min')
        fills_pnl_df = fills_pnl_df.groupby(DF_INDEX, observed=False).agg({'realized_pnl': 'sum'})
        fills_pnl_df = make_date(fills_pnl_df)

        order_fills_df = self.fills_df[['fill_dollars', 'fill_signed_qty', 'sign'] + self.alpha_str_cols]
        order_fills_df = order_fills_df.loc[order_fills_df.fill_signed_qty != 0]
        order_fills_df['fill_dollars_abs'] = order_fills_df['fill_dollars'].abs()
        order_fills_df = pd.merge(order_fills_df, fills_pnl_df[['date', 'realized_pnl']], how='left', left_index=True, right_index=True)

        # calculate total alpha from alpha_model_horizon elements
        order_fills_df['alpha_sided_total'] = 0
        for alpha_str in self.alpha_str_cols:
            weight, mult = self.alpha_str_weights[alpha_str]
            order_fills_df[f'{alpha_str}_sided'] = (order_fills_df['sign'] * order_fills_df[alpha_str].fillna(0).clip(-self.max_alpha, self.max_alpha) * weight * mult).clip(lower=0)
            order_fills_df['alpha_sided_total'] += order_fills_df[f'{alpha_str}_sided']

        # calculate alpha_model_horizon weight from alpha_sided_total and distribute realized pnl
        for alpha_str in self.alpha_str_cols:
            order_fills_df[f'{alpha_str}_pnl'] = order_fills_df[f'{alpha_str}_sided'] / order_fills_df['alpha_sided_total'] * order_fills_df['realized_pnl']
            order_fills_df[f'{alpha_str}_dollars'] = order_fills_df[f'{alpha_str}_sided'] / order_fills_df['alpha_sided_total'] * order_fills_df['fill_dollars_abs']

        order_alpha_pnl_df = order_fills_df.groupby('date')[[f'{alpha_str}_pnl' for alpha_str in self.alpha_str_cols] + [f'{alpha_str}_dollars' for alpha_str in self.alpha_str_cols]].sum().reset_index()
        order_alpha_pnl_df['date'] = order_alpha_pnl_df['date'].dt.date
        return order_alpha_pnl_df

    @staticmethod
    def resample_to_target_period(
            df: pd.DataFrame,
            timestamp_col: str = 'target_ts',
            offset: int = 2,
            opt_int: int = 360,
    ) -> pd.DataFrame:
        df[timestamp_col] = to_datetime(df[timestamp_col])
        hours_in_day = 24
        intervals = [f"{str(h).zfill(2)}:{str(offset).zfill(2)}" for h in range(0, hours_in_day, opt_int // 60)]
        custom_times = pd.DataFrame({'time': intervals})
        custom_times['time'] = to_datetime(custom_times['time'], format='%H:%M').dt.time
        df['shifted_ts'] = df[timestamp_col] - pd.Timedelta(minutes=offset)
        df['resampled_ts'] = (df['shifted_ts'].dt.floor(f'{opt_int}min') + pd.Timedelta(minutes=offset))
        early_morning_mask = df[timestamp_col].dt.time < to_datetime(f'00:{str(offset).zfill(2)}').time()
        df.loc[early_morning_mask, 'resampled_ts'] = (
                df.loc[early_morning_mask, timestamp_col].dt.floor('D')
                - pd.Timedelta(minutes=opt_int)
                + pd.Timedelta(minutes=offset)
        )
        df[timestamp_col] = df['resampled_ts']
        del df['resampled_ts']
        del df['shifted_ts']
        return df

    def calculate_vwap_shortfall(self, aggression_level: Optional[int] = None, compare_px_type: str = 'vwap') -> pd.DataFrame:
        if aggression_level is None:
            fills_df = self.fills_df
            orders_df = self.raw_orders_df
        else:
            fills_df = self.fills_df.loc[self.fills_df['aggression'] == aggression_level]
            orders_df = self.raw_orders_df.loc[self.raw_orders_df['aggression'] == aggression_level]
        vwap_df = merge_on_index(
            self.bars_df[['close_mid', 'volume', 'dvolume']],
            fills_df[['fill_signed_qty', 'fill_dollars']],
            how='left',
        )
        vwap_df = merge_on_index(vwap_df, orders_df[['order_signed_qty']], how='left')
        vwap_df['fill_dollars_abs'] = vwap_df['fill_dollars'].abs()
        # aggregate to target period and then calculate
        vwap_df['target_ts'] = vwap_df.index.get_level_values('ts')
        vwap_df = self.resample_to_target_period(vwap_df, 'target_ts', self.opt_offset_mins, self.reoptimize_interval_mins)
        period_agg_dict = {
            'dvolume': 'sum',
            'volume': 'sum',
            'fill_signed_qty': 'sum',
            'order_signed_qty': 'sum',
            'fill_dollars_abs': 'sum',
            'fill_dollars': 'sum',
        }
        period_vwap_df = vwap_df.groupby(['target_ts', 'symbol_venue'], observed=False).agg(period_agg_dict)
        period_vwap_df['close_mid'] = vwap_df.groupby(['target_ts', 'symbol_venue'], observed=False)['close_mid'].last().values
        period_vwap_df['close_mid_start'] = vwap_df.groupby(['target_ts', 'symbol_venue'], observed=False)['close_mid'].first().values

        period_vwap_df['target_period_vwap'] = period_vwap_df['dvolume'] / period_vwap_df['volume']
        period_vwap_df['avg_fill_px'] = period_vwap_df['fill_dollars'] / period_vwap_df['fill_signed_qty']
        period_vwap_df['order_dollars_abs'] = period_vwap_df['target_period_vwap'] * period_vwap_df['order_signed_qty'].abs()
        period_vwap_df['unfill_signed_qty'] = period_vwap_df['order_signed_qty'] - period_vwap_df['fill_signed_qty']
        slip_compare_px_col = 'target_period_vwap' if compare_px_type == 'vwap' else 'close_mid_start'
        # fill slippage defined as fill px - period vwap * filled qty
        period_vwap_df['fill_slip'] = (period_vwap_df['avg_fill_px'] - period_vwap_df[slip_compare_px_col]) * period_vwap_df['fill_signed_qty']
        # opportunity slippage defined as close px - period vwap * unfilled qty
        period_vwap_df['opp_slip'] = (period_vwap_df['close_mid'] - period_vwap_df[slip_compare_px_col]) * period_vwap_df['unfill_signed_qty']
        # fill na here in case opp slip is nan while fill slip is not nan so the total slip shouldn't be nan, or the opposite
        period_vwap_df['total_slip'] = period_vwap_df['opp_slip'].fillna(0) + period_vwap_df['fill_slip'].fillna(0)
        period_vwap_df = period_vwap_df.reset_index()
        period_vwap_df['ts'] = period_vwap_df['target_ts']
        period_vwap_df = make_date(period_vwap_df)
        return period_vwap_df


class SimMarkouts:
    def __init__(self):
        self.sim_name = None
        self.alpha_column = None
        self.start_dt = None
        self.end_dt = None
        self.sim_trades_df = None
        self.bars_df = None

    def calculate_markouts(
            self,
            sim_name: str,
            alpha_column: str,
            start: Optional[dt] = None,
            end: Optional[dt] = None,
    ):
        self.sim_name = sim_name
        self.alpha_column = alpha_column

        if end is not None:
            self.end_dt = end
        else:
            self.end_dt = yesterday()

        if start is not None:
            self.start_dt = start
        else:
            self.start_dt = self.end_dt - td(days=7)

        self.prepare_markouts_data()
        self.get_markouts_stats()

    def prepare_markouts_data(self):
        sim_trades_df = load_sim_data(self.sim_name)
        sim_trades_df = sim_trades_df[['trade_price', self.alpha_column]]

        self.sim_trades_df = sim_trades_df.loc[
            (sim_trades_df.index.get_level_values('ts') >= self.start_dt) & (sim_trades_df.index.get_level_values('ts') < self.end_dt)
            ]
        data_loader = DataLoader()
        self.bars_df = data_loader.load_bars(horizon=1, start_date=self.start_dt.date() - td(days=1), end_date=self.end_dt.date() + td(days=1), cols=['close_mid'])

    def get_markouts_stats(self):
        for mins in MARKOUT_MINUTES_SIM:
            trades_df = self.sim_trades_df.reset_index()
            trades_df['ts'] = trades_df['ts'] + td(minutes=mins)
            trades_df = set_index(trades_df)
            trades_df = merge_on_index(trades_df, self.bars_df, how='left')

            trades_df['ret'] = (trades_df['close_mid'] - trades_df['trade_price']) / trades_df['trade_price']
            trades_df['winner'] = 'NA'

            trades_df.loc[(trades_df[self.alpha_column] < 0) & (trades_df['ret'] < 0), 'winner'] = 'W'
            trades_df.loc[(trades_df[self.alpha_column] > 0) & (trades_df['ret'] > 0), 'winner'] = 'W'
            trades_df.loc[(trades_df[self.alpha_column] > 0) & (trades_df['ret'] < 0), 'winner'] = 'L'
            trades_df.loc[(trades_df[self.alpha_column] < 0) & (trades_df['ret'] > 0), 'winner'] = 'L'

            win_case = len(trades_df[trades_df['winner'] == 'W'])
            lose_case = len(trades_df[trades_df['winner'] == 'L'])
            win_ratio = (win_case / (win_case + lose_case)) if (win_case + lose_case) > 0 else 0
            logger.info(f"{self.alpha_column} shows win % {win_ratio * 100:.2f} with {mins} lookout for sim {self.sim_name} from {self.start_dt} to {self.end_dt}")

class MomentumMarkouts:
    def __init__(self, start_dt: dt, end_dt: dt, cooldown: bool = False, lookout_mins: int = 360, logret_1440_lz_filter: int = 6):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.cooldown = cooldown
        self.lookout_mins = lookout_mins
        self.logret_1440_lz_filter = logret_1440_lz_filter
        self.data_loader = DataLoader()

        self.bars_df = None
        self.features_df = None
        self.filtered_features_df = None

    def prepare_data(self):
        start_date = self.start_dt.date()
        end_date = self.end_dt.date()

        self.bars_df = self.data_loader.load_bars(horizon=1, start_date=start_date, end_date=end_date, cols=['close_mid'])
        self.features_df = self.data_loader.load_features(horizons=[1440], start_date=start_date, end_date=end_date, cols=['logret_1440_lz'])

    def filter_features_df(self):
        if self.cooldown:
            def get_breakthroughs_with_cooldown(x):
                mask = (x > self.logret_1440_lz_filter) & (x.shift() <= self.logret_1440_lz_filter)
                if not mask.any():
                    return mask
                timestamps = pd.Series(index=x.index, data=[idx[0] for idx in x.index])
                # Get all breakthrough times
                breakthrough_times = timestamps[mask]
                # Create a Series of time differences between each point and all previous breakthroughs
                time_diffs = pd.DataFrame(
                    {t: timestamps - t for t in breakthrough_times},
                    index=timestamps.index,
                )
                # Find points that are within 360 minutes after any previous breakthrough
                invalid_mask = (time_diffs > pd.Timedelta(minutes=0)) & (time_diffs <= pd.Timedelta(minutes=self.lookout_mins))
                invalid_mask = invalid_mask.any(axis=1)

                # Final mask: initial breakthroughs excluding those within cooldown periods
                return mask & ~invalid_mask

            mask = self.features_df.groupby(level='symbol_venue', group_keys=False).logret_1440_lz.apply(
                get_breakthroughs_with_cooldown
            )
            self.filtered_features_df = self.features_df.loc[mask].sort_index()
        else:
            mask = self.features_df.groupby(level='symbol_venue', group_keys=False).logret_1440_lz.apply(
                lambda x: (x > self.logret_1440_lz_filter) & (x.shift() <= self.logret_1440_lz_filter)
            )
            self.filtered_features_df = self.features_df.loc[mask].sort_index()

    def run_momentum(self):
        filtered_features_df = merge_on_index(self.filtered_features_df, self.bars_df[['close_mid']], how='left')

        features_plus_mins = filtered_features_df.reset_index().copy()
        features_plus_mins[f'{self.lookout_mins}_min_ahead_ts'] = features_plus_mins['ts'] + td(minutes=self.lookout_mins)
        features_plus_mins['ts'] = features_plus_mins[f'{self.lookout_mins}_min_ahead_ts']
        features_plus_mins = set_index(features_plus_mins)

        markouts_plus_mins_df = merge_on_index(features_plus_mins, self.bars_df[['close_mid']], how='left', suffixes=('', '_lookout'))
        markouts_plus_mins_df[f'{self.lookout_mins}_min_price_change'] = (markouts_plus_mins_df['close_mid_lookout'] / markouts_plus_mins_df['close_mid']) - 1
        logger.info(f"run case: {self.start_dt=}, {self.end_dt=}, {self.cooldown=}, {self.lookout_mins=}, {self.logret_1440_lz_filter=}")
        logger.info(f"\n{markouts_plus_mins_df[f'{self.lookout_mins}_min_price_change'].describe().to_markdown()}")

    def run(self):
        self.prepare_data()
        self.filter_features_df()
        self.run_momentum()
