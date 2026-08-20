import abc
import asyncio
import logging
from typing import Any, Callable, Dict, Tuple

import util.log as log
import util.object
import util.thread as thread
from exchange import downtime_manager
from market_data.md_api import EXCH, MONGODB_ID, STRATEGY_ID, TOP_DELTA, alerts, os, other_side
from market_data.md_constants import BALANCE
from ops.configure import get_doc_attr_or_default, set_attribute_from_doc
from trade.derivative import Derivative, Future, Option, PerpetualContract
from trade.oms_util import OrderRejectKey, order_reject_alert_cache
from trade.trade_api import (ACK, ASSET, AVG_PX, BPS, BUY, CANCELLED, CUM, CURRENCY, DESCENDING, EVENT_TYPE, EXCHANGE, EXECUTED_FEE_CURRENCY,
                             EXECUTED_FEE_MISC_RATE, EXECUTED_PROCEEDS, EXECUTION, GOOD_FOR_SECONDS, INTERNAL_FEE_IF_CROSSED, IOC, LEAVES, LIMIT_PX, NEW, ONE,
                             ORDER_EVENT, ORDER_ID, ORDER_PARAMS, ORDER_STATE, POST_ONLY, PROCEEDS, PX, REDUCE_ONLY, REJECT_RESPONSE, REJECT_TYPE, REJECTED,
                             SECONDS_IN_MINUTE, SELL, SIDE, SIDES, SIMULATED, SIZE, TEN, THOUSAND, TIMESTAMP, TWO, VERSION, Decimal, MarketType,
                             OrderRejectType, calc_amount_more_aggressive, calc_both_tobs, calc_quoted_size_all_inclusive, convert_size_to_usd,
                             convert_usd_to_size, db_singleton, decimal, float_to_decimal_transformer, floor_to, make_px_more_aggressive,
                             remove_open_orders_from_book, sim_randint, sim_timestamp, time, zero_if_none,)
from trade.trade_constants import SHARED_INSTRUMENTS
from util.collection import list_safe_remove
from util.io_text import environment, exception_stack_trace_as_str
from util.number import SATOSHI_DECIMAL, ZERO_DECIMAL as ZERO, decimal_to_float_transformer, dict_safe_remove, safely_add_to_key

INIT_SUSPEND_REASON = "init-suspension"
ORDER_REJECT_SUSPEND_REASON = "order-reject"
PARAM_RELOAD_SUSPEND_REASON = "param-reload"
HUGE_VALUE = decimal(1e20)
# sustained rejection time window for alerting
# 300 bc exponential backoff  \sum_{x=1}^{19} min(2^(x-1) , 32) = 479 < 500


class HasPlacerProperties(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_trading_pair(self) -> Tuple[Any, Any]:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_sort_key(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_position_mgr(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_position_mgr_new(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_max_exposure_above_baseline(self):
        raise NotImplementedError()

    def get_baseline_size(self):
        self.get_position_mgr().get_baseline_size()

    @abc.abstractmethod
    def get_ideal_allocation(self, get_price: Callable[[str, str], Decimal]):
        pass


class OrderPlacer(HasPlacerProperties):
    dont_send_orders = False
    all_order_placers = []
    # Signify whether all strats trading in this process have initialized their op's
    all_order_placers_loaded = False

    def __init__(self):
        super().__init__()
        OrderPlacer.all_order_placers.append(self)
        self.pos_mgr = None
        self.pos_mgr_new = None
        self._open_oids = set()
        self._oid_to_placement_time = {}
        self._cxl_requested_ids_to_last_req_tstamp = {}
        self._cxl_requested_ids_to_first_req_tstamp = {}
        self._oms = None
        self._book = None
        self.default_asset = None
        self.default_currency = None
        self._max_open_orders = 4  # safety feature
        self.preferred_min_cxl_time = None
        self.prevent_all_orders_side = None
        self.suspend_trading_reasons = [INIT_SUSPEND_REASON]
        self.liquidate_only_reasons = []
        self.liquidate_only = False  # needs to be enforced in each implementation, cannot be enforced here
        self._last_order_tstamp = None
        self._consecutive_rejects = 0
        self._exponentially_increasing_pause_after_reject = 1
        self._consecutive_post_only_reject_count = 0
        self._exchange_currency_unavailable_reject_count = 0
        self._exchange_currency_unavailable_reject_time = 0
        self._consecutive_taker_reject_count = 0
        self._last_reject_side = None
        self._pause_after_reject_end = 0
        self.last_material_book_change_time = 0
        self.rebate_flag = False
        self.flags = {}
        self.cash_buffer_frac = None
        self._last_dont_place_log_time = 0
        self._pacer = None
        self.maker_fee = None
        self.taker_fee = None
        self.trading_account = None
        self.param_adjuster = None
        self.firm_min_cxl_time = 0.5
        self.not_sent_oid = 1
        self._oid_to_scheduled_cancellation_time = {}
        self.prevent_unreal_pnl = False
        self._strategy_doc = None
        self.dont_take = False
        self.should_accept_assigned_position = False
        self.related_order_placers = {}
        self.cxl_order_book_delay_threshold = 1
        self.is_upside_down_pair = False
        self._side_to_most_recent_cxl_limit_size_time = {
            BUY: (ZERO, ZERO, 0),
            SELL: (ZERO, ZERO, 0),
        }
        self._strategy_id = None
        self.use_ideal_allocation = False
        self.top_delta_range = 80 * BPS
        self._ideal_allocation = None
        self._ideal_allocation_doc = None
        self.active_trader = True
        self.allocate_excess = True
        self.sim_fills = 0
        self.store_order_updates_in_postgres = False
        self.store_order_updates_in_postgres_async = None
        self.store_order_updates_in_file = True
        self.is_flex_placer = False
        self.is_statarb_placer = False
        self.orderbook_received_timestamp_acting_on = None
        thread.call_periodically_without_initial_delay(SECONDS_IN_MINUTE, self.monitor_exchange_downtime)
        self.min_quote_update_dispatch_interval = 2
        self.material_update_throttled = False
        self.min_tob_acted_on_delta = None
        self.last_tob_buy_acted_on = None
        self.last_tob_sell_acted_on = None
        self._market_data_provider = None

    def __repr__(self):
        return util.object.basic_representation(self)

    @property
    def new_pos_mgr_mode(self):
        if not self.uses_new_pos_mgr:
            return None
        return self.pos_mgr_new._mode

    @property
    def uses_new_pos_mgr(self):
        return self.pos_mgr_new is not None

    def on_assigned_order(self, order_update):
        asset, currency = order_update[ORDER_PARAMS][ASSET], order_update[ORDER_PARAMS][CURRENCY]
        if asset == self.default_asset and currency == self.default_currency:
            self._open_oids.add(order_update[ORDER_ID])
            self._oid_to_placement_time[order_update[ORDER_ID]] = order_update[TIMESTAMP]
            log.info("{} received assigned order: {}", self, order_update)

    def set_position_mgr(self, pos_mgr):
        """
        This gets set by the Strategy.
        :param pos_mgr:
        :return: nothing
        """
        self.pos_mgr = pos_mgr

    def set_position_mgr_new(self, pos_mgr):
        self.pos_mgr_new = pos_mgr

    def get_position_mgr(self):
        return self.pos_mgr

    def get_position_mgr_new(self):
        return self.pos_mgr_new

    def set_baseline_size(self, baseline_asset_size):
        self.get_position_mgr().set_baseline_size(baseline_asset_size)
        if self.get_position_mgr_new() is not None:
            self.get_position_mgr_new().set_baseline_size(baseline_asset_size)

    def set_top_delta_range(self, top_delta_range):
        self.top_delta_range = top_delta_range

    def set_min_tob_acted_on_delta(self, min_tob_acted_on_delta):
        self.min_tob_acted_on_delta = min_tob_acted_on_delta

    def set_tobs_acted_on(self, tob_buy, tob_sell):
        self.last_tob_buy_acted_on = tob_buy
        self.last_tob_sell_acted_on = tob_sell

    def set_oms(self, oms):
        self._oms = oms

    def get_oms(self):
        return self._oms

    def get_exchange(self):
        if self._oms and self._oms.exchange:
            return self._oms.exchange
        if self.trading_account:
            return self.trading_account.get(EXCHANGE)
        return None

    def set_param_adjuster(self, pa):
        self.param_adjuster = pa

    def get_param_adjuster(self):
        return self.param_adjuster

    def get_trading_pair(self):
        return self.default_asset, self.default_currency

    def set_suspend_trading(self, suspend, reason="unknown", cxl_all=True, level=None, side=None):
        if suspend:
            if reason not in self.suspend_trading_reasons:
                self.suspend_trading_reasons.append(reason)
                log.info("{} suspending trading for reason {}".format(self, reason))
                if cxl_all:
                    self.cxl_all()
        else:
            if list_safe_remove(self.suspend_trading_reasons, reason):
                if reason != INIT_SUSPEND_REASON:
                    log.info("{} trading no longer suspended for reason {}", self, reason)

    def get_trading_account(self):
        return self.trading_account

    def set_trading_account(self, account):
        self.trading_account = account

    def set_pacer(self, pacer):
        self._pacer = pacer

    def get_pacer(self):
        return self._pacer

    def is_trading_suspended(self):
        return len(self.suspend_trading_reasons) > 0 or (self._oms and self._oms.is_trading_halted())

    def should_allocate_first(self):
        return False

    def get_max_allocated_size_equiv_override(self, asset_size):
        return None

    def is_active_trader(self):
        return self.active_trader

    def set_liquidate_only(self, liquidate_only, reason="unknown"):
        if liquidate_only:
            if reason not in self.liquidate_only_reasons:
                self.liquidate_only_reasons.append(reason)
                log.info("{} liquidate only activated for reason {}", self, reason)
            else:
                log.debug("{} was already liquidate only for reason {}", self, reason)
        else:
            if list_safe_remove(self.liquidate_only_reasons, reason):
                log.info("{} liquidate only deactivated for reason {}, remaining reasons: {}", self, reason, self.liquidate_only_reasons)

    def is_liquidate_only(self):
        return len(self.liquidate_only_reasons) > 0

    def suspend_for_time_period(self, time_period, reason="unknown", cxl_all=True, level=None, side=None):
        if (level != None or side != None) or reason not in self.suspend_trading_reasons:
            self.set_suspend_trading(True, reason, cxl_all, level, side)
            asyncio.get_event_loop().call_later(time_period, self.set_suspend_trading, False, reason, cxl_all, level, side)

    def liquidate_only_for_time_period(self, time_period, reason="unknown"):
        if reason not in self.liquidate_only_reasons:
            self.set_liquidate_only(True, reason)
            asyncio.get_event_loop().call_later(time_period, self.set_liquidate_only, False, reason)

    def prior_orders_have_been_linked(self):
        """
        Called by start_trading after orders have been put into self._open_oids and before trading has started.
        These orders put in here are from the prior session, they existed upon startup of start trading
        """
        # default implementation is no-op
        pass

    def get_market_type(self):
        market_type = MarketType.SPOT
        if isinstance(self.default_asset, PerpetualContract):
            market_type = MarketType.PERPETUAL
        elif isinstance(self.default_asset, Future):
            market_type = MarketType.FUTURES
        elif isinstance(self.default_asset, Derivative):
            market_type = MarketType.DERIVATIVE
        elif isinstance(self.default_asset, Option):
            market_type = MarketType.OPTION
        return market_type

    def monitor_exchange_downtime(self):
        if self._oms:
            exch = self._oms.exchange
            market_type = self.get_market_type()
            if exch:
                if downtime_manager.singleton().is_downtime_upcoming_or_ongoing(exch, mins_upcoming=5, market_type=market_type):
                    self.set_suspend_trading(True, "exchange downtime suspend", cxl_all=True)
                else:
                    self.set_suspend_trading(False, "exchange downtime suspend")

    def on_material_book_change(self):
        pass

    def on_quotes_update(self, book):
        if (self.default_asset is None or book[ASSET] == self.default_asset) and (self.default_currency is None or book[CURRENCY] == self.default_currency):
            self._book = book
            now = time.time()
            if now - self.last_material_book_change_time > self.min_quote_update_dispatch_interval or self.is_top_delta_important(book):
                self.last_material_book_change_time = now
                self.on_material_book_change()

    def is_top_delta_important(self, book):
        if TOP_DELTA not in book or book[TOP_DELTA] is None or not book[TOP_DELTA] or self.material_update_throttled:
            # if top-delta isnt provided (e.g. not here or empty map), then it's an important update
            return True
        top_delta = book[TOP_DELTA]
        if self.min_tob_acted_on_delta is None:
            for side in SIDES:
                if side in top_delta and top_delta[side] is not None:
                    if side not in book or len(book[side]) == 0 or top_delta[side] is None:
                        return True
                    tob_on_side = next(iter(book[side]))[PX]
                    if calc_amount_more_aggressive(side, tob_on_side, top_delta[side]) / tob_on_side < self.top_delta_range:
                        return True
        else:
            curr_tob_buy = next(iter(book[BUY]))[PX]
            curr_tob_sell = next(iter(book[SELL]))[PX]
            if self.last_tob_buy_acted_on is None or self.last_tob_sell_acted_on is None:
                return True
            if abs(curr_tob_buy - self.last_tob_buy_acted_on) / curr_tob_buy > self.min_tob_acted_on_delta:
                return True
            if abs(curr_tob_sell - self.last_tob_sell_acted_on) / curr_tob_sell > self.min_tob_acted_on_delta:
                return True
        return False

    def get_latest_book(self):
        return self._book

    def get_mark_to_market_price(self):
        if self.prevent_unreal_pnl:
            return None
        book = self.get_book_for_unrealized_pnl()
        if book is None:
            log.error("{} have relative position on {}/{} but no book available", self, self.default_asset, self.default_currency)
            return None
        bid, ask = calc_both_tobs(book)
        if not bid or not ask or abs(ask - bid) / bid > decimal("0.25"):
            if bid and ask:
                log.info("{} Skipping unreal because bid ask spread too big: {} by {}", self, bid, ask)
            else:
                log.info("{} Skipping unreal because don't have bid or ask: {} by {}", self, bid, ask)
            return None
        else:
            return (bid + ask) / TWO

    def get_book_for_unrealized_pnl(self):
        if self.prevent_unreal_pnl:
            return None
        return self._book

    def get_book_less_our_orders(self, max_usd_notional=100 * THOUSAND):
        now = sim_timestamp()
        open_orders = self.create_open_order_info_list()
        recently_cancelled = []
        for side in SIDES:
            cxl_limit_size_time = self._side_to_most_recent_cxl_limit_size_time[side]
            if cxl_limit_size_time[2] > now - self.cxl_order_book_delay_threshold:
                recently_cancelled.append(
                    {
                        SIDE: side,
                        LIMIT_PX: cxl_limit_size_time[0],
                        SIZE: cxl_limit_size_time[1],
                    }
                )
        return remove_open_orders_from_book(self._book, open_orders, recently_cancelled, max_usd_notional=max_usd_notional)

    def on_market_trades(self, trade_list):
        """Override me"""

    def apply_proceeds_as_fee_rebate(self, order_update):
        """This method will not be called, here as a utility.  When all proceeds are a fee rebate (for Binance BNB), we
        replace the self.modify_order_update method with this method."""
        order_event = order_update[ORDER_EVENT]
        if order_event[EVENT_TYPE] == EXECUTION and order_update[ORDER_PARAMS][SIDE] == SELL:
            executed_proceeds = order_event[EXECUTED_PROCEEDS]
            safely_add_to_key(order_event, EXECUTED_FEE_CURRENCY, -executed_proceeds)
            log.info("{} adjusted executed fee currency by -executed_proceeds for {}, {}", self, order_update, executed_proceeds)

    def modify_order_update(self, order_update):
        pass

    def on_order_update(self, order_update):
        log.debug("{} processing order update {}", self, order_update)
        self.modify_order_update(order_update)
        self.update_internal_state(order_update)
        order_event = order_update[ORDER_EVENT]

        alert_key = OrderRejectKey(self.get_exchange(), self.get_market_type(), order_event.get(REJECT_TYPE, OrderRejectType.EXCHANGE_ERROR), "")
        if order_event[EVENT_TYPE] == REJECTED and not self.suspend_trading_reasons:
            need_to_back_off = order_event.get(REJECT_TYPE) not in (
                OrderRejectType.INTERNAL_REJECTION,
                OrderRejectType.POST_ONLY_REJECTION,
                OrderRejectType.SYSTEM_OVERLOAD_REJECTION,
                OrderRejectType.INTERNAL_INSUFFICIENT_MARGIN,
            )
            if order_event.get(REJECT_TYPE) == OrderRejectType.POST_ONLY_REJECTION:
                self._consecutive_post_only_reject_count += 1
                if self._consecutive_post_only_reject_count > 5:
                    log.info("{} {} consecutive post-only rejects, need to do exponential back-off", self, self._consecutive_post_only_reject_count)
                    need_to_back_off = True

            if need_to_back_off:
                # if fund unavailable buy rejections, we need add quote currency information added into alert key
                if (
                    order_event.get(REJECT_TYPE) == OrderRejectType.FUNDS_UNAVAILABLE
                    and order_update[ORDER_PARAMS][SIDE] == BUY
                    and order_update[ORDER_PARAMS][CURRENCY] in self.trading_account.get(SHARED_INSTRUMENTS, [])
                ):
                    alert_key = OrderRejectKey(
                        self.get_exchange(),
                        self.get_market_type(),
                        order_event.get(REJECT_TYPE, OrderRejectType.EXCHANGE_ERROR),
                        order_update[ORDER_PARAMS][CURRENCY],
                    )

                if order_event.get(REJECT_TYPE) != OrderRejectType.POST_ONLY_REJECTION:
                    desc = f"{self}_{self.get_oms().account_id} received rejection on order: {order_update}"
                    if order_event.get(REJECT_RESPONSE):
                        desc += f" with api response: {order_event[REJECT_RESPONSE]}"
                    order_reject_alert_cache.cache_and_alert(alert_key, desc)
                    # as quote currency unavailable is at exchange level, we can use it to notify which currency unavailable and adjust
                    self._exchange_currency_unavailable_reject_count = order_reject_alert_cache.key_to_failure_count.get(alert_key, 0)

                # only need to do backoff if we are not placing order from TradingVenue, it uses level side suspension and backoff
                self._last_reject_side = order_update[ORDER_PARAMS][SIDE]
                if not self.is_flex_placer and not self.is_statarb_placer:
                    log.warning("{} Suspending trading for {} seconds after new order reject".format(self, self._exponentially_increasing_pause_after_reject))
                    cxl_all = self._exponentially_increasing_pause_after_reject > 20
                    self.suspend_for_time_period(self._exponentially_increasing_pause_after_reject, ORDER_REJECT_SUSPEND_REASON, cxl_all)
                    self._exponentially_increasing_pause_after_reject *= 2

        elif order_update[ORDER_EVENT][EVENT_TYPE] == ACK:
            alert_key_fund_unavailable_buy = OrderRejectKey(
                self.get_exchange(), self.get_market_type(), OrderRejectType.FUNDS_UNAVAILABLE, order_update[ORDER_PARAMS][CURRENCY]
            )
            self._exchange_currency_unavailable_reject_time = order_reject_alert_cache.key_to_last_failure_time.get(alert_key_fund_unavailable_buy, 0)
            self._consecutive_taker_reject_count = 0
            self._consecutive_post_only_reject_count = 0
            if order_update[ORDER_PARAMS][SIDE] == self._last_reject_side:
                order_reject_alert_cache.reset_key(alert_key)
                self._exponentially_increasing_pause_after_reject = 1

        elif order_update[ORDER_EVENT][EVENT_TYPE] == CANCELLED:
            order_params = order_update[ORDER_PARAMS]
            log.info(
                "{} order {} to {} {} of {} @ {} cancelled".format(
                    self, order_update[ORDER_ID], order_params[SIDE], order_params[SIZE], order_params[ASSET], order_params[LIMIT_PX]
                )
            )

    def place_order(self, side, size, limit_px, post_only=True, good_for_seconds=None, reduce_only=False):
        if not self.is_trading_suspended() and limit_px > 0:
            order_params = {
                SIDE: side,
                SIZE: size,
                LIMIT_PX: limit_px,
                ASSET: self.default_asset,
                CURRENCY: self.default_currency,
            }
            if not post_only:
                order_params[POST_ONLY] = False
            if good_for_seconds is not None:
                if good_for_seconds == IOC:
                    order_params[GOOD_FOR_SECONDS] = 0
                else:
                    order_params[GOOD_FOR_SECONDS] = good_for_seconds
            if reduce_only:
                order_params[REDUCE_ONLY] = True
            return self.place_order_from_params(order_params)
        if limit_px <= 0:
            log.error("Limit price: {} <=0, for {} {} {} {} {}".format(limit_px, self._oms, side, size, self.default_asset, self.default_currency))
        return False

    def new_event_simulated_order(self, oid):
        order_params = self.get_order_params(oid)
        px = order_params.get(LIMIT_PX)
        log.info(f"{self} simulating new and ack for {order_params[SIDE]} {order_params[ASSET]} @ {px} oid {oid}")
        self._oms.dispatch_order_event(oid, NEW, addl_event_key_vals={SIMULATED: True})
        self._oms.dispatch_order_event(oid, ACK, addl_event_key_vals={SIMULATED: True})

    def fill_simulated_order(self, oid):
        order_state = self.get_oms().get_order_state(oid)
        order_params = self.get_order_params(oid)
        if order_state.get(LEAVES) == order_params[SIZE]:
            trade_size = order_params[SIZE]
            px = order_params.get(LIMIT_PX)
            fee_rate = self.get_maker_fee() if order_params.get(POST_ONLY, True) else self.get_taker_fee()
            log.info("{} simulating fill for {} {} {} @ {}", self, order_params[SIDE], trade_size, order_params[ASSET], px)

            partial_fill = sim_randint(1, 3) == 2
            if partial_fill:
                min_trade_increment = self.get_min_trade_increment()
                if not min_trade_increment:
                    min_trade_increment = SATOSHI_DECIMAL
                steps = trade_size // min_trade_increment
                if steps > 2:
                    trade_size = sim_randint(1, steps) * self.get_min_trade_increment()
            order_state[CUM] = trade_size
            order_state[LEAVES] = order_params[SIZE] - trade_size
            order_state[AVG_PX] = order_params[LIMIT_PX]
            order_state[PROCEEDS] = order_state[CUM] * order_state[AVG_PX]
            proceeds = trade_size * px
            self._oms.dispatch_order_event(oid, EXECUTION, trade_size, proceeds, px, addl_event_key_vals={SIMULATED: True, EXECUTED_FEE_MISC_RATE: fee_rate})

            self.sim_fills += 1

    def place_order_from_params(self, order_params):
        order_params[INTERNAL_FEE_IF_CROSSED] = self._get_internal_cross_fee(order_params)
        if self._pacer is None or self._pacer.can_i_go(self):
            now = sim_timestamp()
            if self._max_open_orders is not None and len(self._open_oids) >= self._max_open_orders:
                log.info("{} - Not placing order because max open orders exceeded".format(self))
            else:
                self._last_order_tstamp = now
                if not self.suspend_trading_reasons:
                    if self.prevent_all_orders_side is not None and self.prevent_all_orders_side == order_params[SIDE]:
                        return False

                    if not OrderPlacer.dont_send_orders:
                        oid = self._oms.place_order(order_params, orderbook_received_timestamp_acting_on=self.orderbook_received_timestamp_acting_on)
                        if oid is not None:
                            self._open_oids.add(oid)
                            self._oid_to_placement_time[oid] = sim_timestamp()
                        return oid
                    else:
                        oid = "{}{}".format(self.__str__(), self.not_sent_oid)
                        log.info(
                            "OrderPlacer {} faking order sent of {} {:f} @ {:f}, oid {}",
                            self,
                            order_params[SIDE],
                            order_params[SIZE],
                            order_params[LIMIT_PX],
                            oid,
                        )
                        self.not_sent_oid += 1
                        self._open_oids.add(oid)
                        self._oms.oid_to_params[oid] = order_params
                        self._oms.oid_to_state[oid] = {
                            LEAVES: order_params[SIZE],
                            CUM: ZERO,
                            VERSION: 1,
                        }
                        self._oid_to_placement_time[oid] = sim_timestamp()

                        thread.call_later(0.2, self.new_event_simulated_order, oid)

                        sim_fill = sim_randint(1, 10) == 1
                        sim_delay = 5
                        if not order_params.get(POST_ONLY, True):
                            sim_fill = sim_randint(1, 5) != 2
                            sim_delay = 0.5
                        if os.environ.get("SIMULATE_FILLS", True) not in (True, 1, "1", "True"):
                            sim_fill = False
                        if sim_fill and environment.env != environment.PROD:
                            thread.call_later(sim_delay, self.fill_simulated_order, oid)
                        return oid

        return False

    def get_placed_tstamp(self, oid):
        return self.get_oms().get_placed_tstamp(oid)

    def is_our_order_id(self, oid):
        return oid in self._open_oids or oid in self._oid_to_placement_time

    def get_order_context(self, oid, standard_side, on_level_num=None, if_cxl=False):
        # this maintained in trading venue child class
        return None

    def get_tick_size(self, ref_px):
        return self._oms.get_tick_size(self.default_asset, self.default_currency, ref_px)

    def get_min_trade_size(self, ref_px=None):
        return self._oms.get_min_trade_size(self.default_asset, self.default_currency, ref_px)

    def get_max_trade_size(self, ref_px=None):
        return self._oms.get_max_trade_size(self.default_asset, self.default_currency, ref_px)

    def get_min_trade_increment(self):
        return self._oms.get_min_trade_increment(self.default_asset, self.default_currency)

    def get_time_since_placed(self, oid):
        if oid in self._oid_to_placement_time:
            return sim_timestamp() - self._oid_to_placement_time[oid]
        # this shouldn't happen unless this order is dead.
        return 0

    def update_internal_state(self, order_update):
        """Cleans up self.open_oids and self.cxl_requested_ids"""
        oid = order_update[ORDER_ID]
        if oid in self._open_oids or oid in self._oid_to_placement_time:
            order_state = order_update[ORDER_STATE]
            log.debug("{} received order update of {}".format(self, order_update))
            if order_state[LEAVES] <= ZERO:
                # order is done
                self._open_oids.discard(oid)
                dict_safe_remove(self._cxl_requested_ids_to_last_req_tstamp, oid)
                # delay removing from _oid_to_placement_time as this map is used in is_our_order, and I want
                # to be able to identify an order update as ours for 10 mins after the order is dead in case
                # we get a late execution or correction
                thread.call_later(10 * SECONDS_IN_MINUTE, dict_safe_remove, self._oid_to_placement_time, oid)
                thread.call_later(SECONDS_IN_MINUTE, dict_safe_remove, self._cxl_requested_ids_to_first_req_tstamp, oid)

    def cxl_order_after_firm_time(self, oid, log_msg=None):
        self.cxl_order_after_time_passed(oid, self.firm_min_cxl_time, log_msg)

    def cxl_order_if_time_passed(self, oid, time_to_pass, log_msg=None):
        if not self.is_cxl_pending(oid):
            time_passed = self.get_time_since_placed(oid)
            if time_passed >= time_to_pass:
                self.cxl_order(oid, log_msg)
        else:
            now = sim_timestamp()
            if now - self._cxl_requested_ids_to_first_req_tstamp.get(oid, now) > 60:
                try:
                    order_state = self._oms.get_order_state(oid)
                except KeyError:
                    order_state = "[no order state exists!]"
                log.log_occassionally(
                    "{} trying to cancel {}".format(self, oid), 60, "{} trying to cancel {} and still waiting w/ state: {}", self, oid, order_state
                )

    def cxl_order_stuck_over_threshold(self, time_to_pass, log_msg=None):
        now = sim_timestamp()
        stuck_oid = []
        for oid in self._cxl_requested_ids_to_first_req_tstamp:
            if now - self._cxl_requested_ids_to_first_req_tstamp.get(oid, now) > time_to_pass:
                log.log_occassionally(
                    "{} order cancel stuck check {}".format(self, oid), 30, "{} order cancel stuck check {} for {} seconds", self, oid, time_to_pass
                )
                stuck_oid.append(oid)
        return stuck_oid

    def cxl_order_after_time_passed(self, oid, time_to_pass, log_msg=None):
        if not self.is_cxl_pending(oid):
            time_passed = self.get_time_since_placed(oid)
            if time_passed >= time_to_pass:
                if self.get_order_leaves(oid) > ZERO:
                    self.cxl_order(oid, log_msg)
            else:
                wait_time = time_to_pass - time_passed
                now = time.time()
                scheduled_time = now + wait_time
                if scheduled_time + 0.01 < self._oid_to_scheduled_cancellation_time.get(oid, 10 * now):
                    order_params = self.get_order_params(oid)
                    log.info(
                        "{} will cancel oid {} {} {} {}/{} @ {} in {:.2f} seconds cause {}",
                        self,
                        oid,
                        order_params[SIDE],
                        order_params[SIZE],
                        order_params[ASSET],
                        order_params[CURRENCY],
                        order_params[LIMIT_PX],
                        wait_time,
                        (log_msg if log_msg else "no msg provided"),
                    )
                    self._oid_to_scheduled_cancellation_time[oid] = scheduled_time
                    thread.call_later(wait_time, self.cxl_order, oid, log_msg, False)

    def cxl_these_orders_if_firm_time_passed(self, order_ids, log_msg=None):
        for oid in order_ids:
            self.cxl_order_if_time_passed(oid, self.firm_min_cxl_time, log_msg)

    def cxl_these_orders(self, order_ids, log_msg):
        log.log_occassionally_with_level(logging.DEBUG, f"{self} cxl_these_orders {log_msg}", 2, f"{self} cxl_these_orders {order_ids}: {log_msg}")
        for oid in order_ids:
            self.cxl_order(oid, log_msg)

    def cxl_order(self, oid, log_msg=None, log_if_already_done=True):
        now = sim_timestamp()
        order_state = self._oms.get_order_state(oid)

        if oid not in self._cxl_requested_ids_to_last_req_tstamp:
            if order_state[LEAVES] == ZERO:
                if log_if_already_done:
                    log.warning("{} OMS knows this order as cancelled, why is cancellation being requested by OP/algo {} {}".format(self, oid, log_msg))
                return

            self._cxl_requested_ids_to_last_req_tstamp[oid] = now
            self._cxl_requested_ids_to_first_req_tstamp[oid] = now
            order_params = self.get_order_params(oid)
            if log_msg:
                log.info("{} cancelling order {} {} {} @ {} because {}", self, oid, order_params[SIDE], order_params[SIZE], order_params[LIMIT_PX], log_msg)
            self._oms.cancel_order(oid, orderbook_received_timestamp_acting_on=self.orderbook_received_timestamp_acting_on)
            self._side_to_most_recent_cxl_limit_size_time[order_params[SIDE]] = (order_params[LIMIT_PX], order_params[SIZE], now)
            if OrderPlacer.dont_send_orders:
                thread.call_later(0.02, lambda: self._oms.dispatch_order_event(oid, CANCELLED, addl_event_key_vals={SIMULATED: True}))
        else:
            order_state = self._oms.get_order_state(oid)
            if order_state[LEAVES] == ZERO:
                if log_if_already_done:
                    log.warning("{} OMS knows this order as cancelled, why is cancellation being requested by OP/algo {}".format(self, oid))
                if oid in self._open_oids:
                    self._open_oids.remove(oid)
                if oid in self._cxl_requested_ids_to_last_req_tstamp:
                    self._cxl_requested_ids_to_last_req_tstamp.pop(oid)
            else:
                if now - self._cxl_requested_ids_to_last_req_tstamp[oid] > 1:
                    self._cxl_requested_ids_to_last_req_tstamp[oid] = now
                    self._oms.cancel_order(oid, orderbook_received_timestamp_acting_on=self.orderbook_received_timestamp_acting_on)

    def should_wait_for_book_on_stratup(self):
        return True

    def get_first_cxl_req_tstamp(self, oid):
        return self._cxl_requested_ids_to_first_req_tstamp.get(oid)

    def is_cxl_scheduled(self, oid):
        return oid in self._oid_to_scheduled_cancellation_time

    def is_cxl_pending(self, oid):
        return oid in self._cxl_requested_ids_to_last_req_tstamp

    def cxl_all(self):
        for oid in self._open_oids.copy():
            self.cxl_order(oid)

    def cxl_all_level_and_side(self, level=None, side=None):
        if level and not side:
            oids = [context.oid for context in level.standard_side_to_order_context.values() if context and context.oid]
            for oid in oids:
                if not self.is_cxl_pending(oid):
                    self.cxl_order(oid)
        elif side and level:
            oids_on_side_and_level = [
                context.oid for context in level.standard_side_to_order_context.values() if context and context.oid and self.get_side(context.oid) == side
            ]
            for oid in oids_on_side_and_level:
                if not self.is_cxl_pending(oid):
                    self.cxl_order(oid)
        elif not level and side:
            oids_on_level_and_side = [oid for oid in self._open_oids if self.get_side(oid) == side]
            for oid in oids_on_level_and_side:
                if not self.is_cxl_pending(oid):
                    self.cxl_order(oid)

    def have_pending_cxls(self, side):
        for oid in self._open_oids:
            if self.get_side(oid) == side and self.is_cxl_pending(oid):
                return True
        return False

    def cancel_all_orders_older_than(self, seconds_age, log_msg="canceling old order", side=None):
        for oid in self._open_oids.copy():
            if self.get_time_since_placed(oid) > seconds_age:
                if not self.is_cxl_pending(oid):
                    if side is None or self.get_side(oid) == side:
                        self.cxl_order(oid)
                        log.info("{} cancelling {} {}", self, log_msg, oid)

    def create_open_order_info_list(self, side=None):
        """Creates a list with one dict for each open order.  The dicts have SIDE, LEAVES, and LIMIT_PX populated.
        Useful for calling calc_both_tobs_excluding_orders()"""
        info_list = []
        for oid in self._open_oids:
            order_params = self._oms.get_order_params(oid)
            if side is None or order_params[SIDE] == side:
                order_state = self._oms.get_order_state(oid)
                info = {SIDE: order_params[SIDE], LEAVES: order_state[LEAVES], LIMIT_PX: order_params[LIMIT_PX]}
                info_list.append(info)
        return info_list

    def make_order_size(self, size, ref_px=None):
        size = floor_to(size, self._oms.get_min_trade_increment(self.default_asset, self.default_currency))
        if abs(size) < self._oms.get_min_trade_size(self.default_asset, self.default_currency, ref_px):
            return ZERO
        return size

    def get_expected_roundtrip_fee(self):
        return TWO * self.get_maker_fee()

    def get_maker_fee(self):
        if self.maker_fee is not None:
            return self.maker_fee
        if self._oms.default_fee is None:
            return ZERO
        return self._oms.default_fee

    def set_maker_fee(self, fee_rate):
        self.maker_fee = fee_rate

    def get_taker_fee(self):
        if self.taker_fee is not None:
            return self.taker_fee
        if self._oms.default_fee is None:
            return ZERO
        return self._oms.default_fee

    def convert_usd_to_size(self, usd_value, ref_px):
        if ref_px == ZERO:
            return ZERO
        asset = self.default_asset
        currency = self.default_currency
        return convert_usd_to_size(asset, currency, usd_value, ref_px)

    def convert_size_to_usd(self, size, ref_px):
        if ref_px == ZERO:
            return ZERO
        asset = self.default_asset
        currency = self.default_currency
        return convert_size_to_usd(asset, currency, size, ref_px)

    def cxl_open_orders(self, side, log_msg=None):
        for oid in self._open_oids.copy():
            if self.get_side(oid) == side:
                self.cxl_order(oid, log_msg=log_msg)

    def get_order_params(self, oid):
        return self._oms.get_order_params(oid)

    def get_limit_px(self, oid):
        if oid not in self._oms.oid_to_params:
            msg = "{} oid not in oid_to_params: {}".format(self, oid)
            raise ValueError(msg)
        return self._oms.get_order_params(oid)[LIMIT_PX]

    def get_order_size(self, oid):
        return self._oms.get_order_params(oid)[SIZE]

    def get_order_leaves(self, oid):
        return self._oms.get_order_state(oid)[LEAVES]

    def get_order_cum(self, oid):
        return self._oms.get_order_state(oid).get(CUM, ZERO)

    def get_side(self, oid):
        return self._oms.get_order_params(oid)[SIDE]

    def have_open_orders(self):
        return len(self._open_oids) > 0

    def get_open_orders_for_side(self, side):
        return [oid for oid in self._open_oids if self.get_side(oid) == side]

    def get_open_orders(self):
        return self._open_oids

    def get_total_open_order_size_for_side(self, side):
        size = ZERO
        for oid in self.get_open_orders_for_side(side):
            size += self.get_order_size(oid)
        return size

    def get_ideal_allocation(self, get_price: Callable[[str, str], Decimal]):
        return None

    def sort_oids_by_aggression(self, oids, side, decreasing_aggression=True):
        """
        Sorts oid and returns a new list, does not sort the provided list
        :param oids:
        :param side:
        :param decreasing_aggression: if true, then the most aggressive order is first
        :return: a sorted list
        """
        if side == BUY:
            decreasing_price = decreasing_aggression
        else:
            decreasing_price = not decreasing_aggression
        return sorted(oids, key=lambda oid: self.get_limit_px(oid), reverse=decreasing_price)

    def more_aggressive_than_everything(self, side):
        """Returns a price that will be more aggressive than any other price -- eg huge price if buying and
        zero if selling"""
        return HUGE_VALUE if side == BUY else ZERO

    def less_aggressive_than_everything(self, side):
        """Returns a price that will be less aggressive than any other price -- eg huge price if selling and
        zero if buying"""
        return HUGE_VALUE if side == SELL else ZERO

    def do_we_have_order_at_limit(self, limit_px):
        for oid in self._open_oids:
            if self.get_limit_px(oid) == limit_px:
                return True
        return False

    def determine_outbid_amount(self, book, side, px_to_outbid):
        # debug data gathering
        if px_to_outbid == ZERO:
            log.warning("{} determine_outbid_amount the px_to_outbid is zero with side {} and book {}", self, side, book)
            # note that we will still fail and throw with a divide by zero error later

        if self.do_we_have_order_at_limit(px_to_outbid):
            # don't outbid ourselves
            # log.warning("{} we have order at {}, don't outbid ourself", self, px_to_outbid)
            return ZERO

        max_exposure = self.get_max_exposure_above_baseline()
        tick_size = self.get_tick_size(px_to_outbid)
        tick_size_as_frac = zero_if_none(tick_size) / px_to_outbid
        if tick_size_as_frac >= decimal("0.10") * BPS:
            # the tick size is material, so consider joining instead of outbidding:
            # doing this on a range because i want to see include size that's a bp or two above the quote size as well
            a_litte_more_aggressive = make_px_more_aggressive(side, px_to_outbid, ONE * BPS * px_to_outbid)
            size = calc_quoted_size_all_inclusive(book, side, a_litte_more_aggressive, px_to_outbid)
            if max_exposure and size < max_exposure:
                # dont outbid because small size there and material tick size
                return ZERO
            try:
                opp_tob = book[other_side(side)][0][PX]
            except (ValueError, IndexError, StopIteration):
                return ZERO
            spread_frac = abs(opp_tob - px_to_outbid) / opp_tob
            if tick_size_as_frac > 2 * BPS or tick_size_as_frac >= spread_frac / TEN:
                return ZERO
        return tick_size

    # NOTE: Implement me in the algo specific OrderPlacer subclass
    def get_max_exposure_above_baseline(self):
        raise NotImplementedError()

    # NOTE: long term it would be nice to also have a load_params() method which takes a dict and applies the
    # parameter dictionary to the OP.
    def dump_params(self):
        """
        This will be overridden by the individual placers to help provide whatever parameters that they want to
        persist.
        :return: dictionary with parameters
        """
        raise NotImplementedError()

    @classmethod
    def get_sort_key(cls) -> str:
        """Key used in sorting order placers with each other. Primarily for allocation purposes."""
        return "M"

    def set_attribute_from_strat_doc(self, attr_name: str, default_val, strat_doc: Dict[Any, Any], convert_to_decimal: bool = True) -> None:
        # If the attribute is in the strat_doc, always use that
        # if not, if the attribute is set on the object (not None), leave that value there
        # else, set the attribute to default_val
        try:
            if not hasattr(self, attr_name):
                log.warning("{} this attribute '{}' isn't support by this class".format(self, attr_name))
            elif attr_name in strat_doc:
                # attribute defined in the doc so apply it
                val = strat_doc.get(attr_name)
                if convert_to_decimal:
                    val = decimal(strat_doc.get(attr_name))
                setattr(self, attr_name, val)
            else:
                # only apply the default argument if the current attribute state is None
                current_val = getattr(self, attr_name)
                if current_val is None:
                    setattr(self, attr_name, default_val)
        except:
            log.exception("Exception applying {} on {}", attr_name, self)
            alerts.raise_alert(
                "MM3 exception setting attribute on {}".format(self),
                alerts.HIGH,
                "Attribute {} on {}:\n{}".format(attr_name, self, exception_stack_trace_as_str()),
            )

    def reload_configure(self, strat_doc: Dict[Any, Any], **kwargs) -> None:
        """
        THIS FUNCTION IS CALLED MULTIPLE TIMES
        """
        context = {STRATEGY_ID: self._strategy_id, EXCH: self.get_exchange()}

        if get_doc_attr_or_default("liquidate_only", strat_doc, context=context, obj_log_context=self):
            self.set_liquidate_only(True, "strat-doc params")

        set_attribute_from_doc(self, "preferred_min_cxl_time", self.preferred_min_cxl_time, strat_doc, convert_to_decimal=False, context=context)

        set_attribute_from_doc(self, "store_order_updates_in_postgres_async", False, strat_doc, convert_to_decimal=False, context=context)
        set_attribute_from_doc(self, "store_order_updates_in_file", True, strat_doc, convert_to_decimal=False, context=context)

        set_attribute_from_doc(self, "firm_min_cxl_time", self.firm_min_cxl_time, strat_doc, convert_to_decimal=False, context=context)
        self.use_ideal_allocation = get_doc_attr_or_default(
            "use_ideal", strat_doc, context=context, default_val=self.use_ideal_allocation, obj_log_context=self
        )
        set_attribute_from_doc(self, "prevent_unreal_pnl", self.prevent_unreal_pnl, strat_doc, context=context)

        allocate_excess = get_doc_attr_or_default("allocate_excess", strat_doc, context=context, obj_log_context=self)
        if allocate_excess is not None:
            self.allocate_excess = allocate_excess
        elif self.allocate_excess is None and "account_doc" in kwargs and "allocate_excess_default" in kwargs["account_doc"]:
            self.allocate_excess = kwargs["account_doc"]["allocate_excess_default"]

    def configure(self, strat_doc: Dict[Any, Any], **kwargs) -> None:
        """
        Go through the possible parameters that we accept that are in the strategy doc and applied them to myself.
        Example, if 'wide_spread' is specified then we extract that and specify it on `set_attribute_from_strat_doc()`.
        """
        self._strategy_id = strat_doc[MONGODB_ID]
        context = {STRATEGY_ID: self._strategy_id, EXCH: self.get_exchange()}
        OrderPlacer.reload_configure(self, strat_doc)
        startup_delay_mins = get_doc_attr_or_default("startup_delay_mins", strat_doc, context=context, obj_log_context=self)
        if startup_delay_mins is not None:
            log.info("{} will have {} minutes startup delay as per strat-doc", self, startup_delay_mins)
            delay_seconds = startup_delay_mins * SECONDS_IN_MINUTE
            suspend_key = "startup-delay-strat-doc"
            self.set_suspend_trading(True, suspend_key)
            thread.call_later(delay_seconds - 0.1, log.info, "{} strat-doc startup delay over", self)
            thread.call_later(delay_seconds, self.set_suspend_trading, False, suspend_key)

    def _get_size_to_record_for_allocation_history(self):
        return self.get_position_mgr().get_asset_size()

    def _get_balance_to_record_for_allocation_history(self):
        return self.get_position_mgr().get_currency_balance()

    def record_allocation(self):
        if environment.env == environment.PROD and not OrderPlacer.dont_send_orders:
            if self._strategy_id and self.get_position_mgr() and self.get_position_mgr().get_asset_size() is not None and environment.env == environment.PROD:
                db = db_singleton()
                asset_size = self._get_size_to_record_for_allocation_history()
                ccy_balance = self._get_balance_to_record_for_allocation_history() if not isinstance(self.default_asset, Derivative) else ZERO
                db.AllocationHistory.insert_one(
                    {
                        TIMESTAMP: time.time(),
                        STRATEGY_ID: self._strategy_id,
                        EXCH: self.get_exchange(),
                        ASSET: str(self.default_asset),
                        CURRENCY: str(self.default_currency),
                        SIZE: decimal_to_float_transformer(asset_size),
                        BALANCE: decimal_to_float_transformer(ccy_balance),
                    }
                )
                log.info("{} Recorded ending allocation of {}", self, asset_size)

    def get_ideal_asset_allocation(self):
        if not self.use_ideal_allocation:
            return None
        if self._ideal_allocation is None:
            self._ideal_allocation = ZERO
            if self._strategy_id:
                db = db_singleton()
                query = {
                    STRATEGY_ID: self._strategy_id,
                    EXCH: self.get_exchange(),
                    ASSET: str(self.default_asset),
                    CURRENCY: str(self.default_currency),
                }
                cursor = db.AllocationHistory.find(query).sort([(TIMESTAMP, DESCENDING)])
                for entry in cursor:
                    self._ideal_allocation_doc = float_to_decimal_transformer(6, entry)
                    self._ideal_allocation = float_to_decimal_transformer(6, entry.get(SIZE))
                    log.info("{} loaded ideal-allocation of {}", self, self._ideal_allocation)
                    break
        return self._ideal_allocation

    def _get_internal_cross_fee(self, order_params):
        """Return the fee to use if this order is matched internally instead of on the exchange."""
        # When both maker and taker fee are positive, the simple case, we use a fee of zero for internal crosses.
        # When the maker fee is negative, make sure the maker gets that rebate because there are algo implications.
        # If the taker can't afford to pay the maker's rebate, that's an error; alert and use a fee of zero.
        internal_maker_fee = min(self.get_maker_fee(), ZERO)
        minimum_taker_fee = min(Decimal(-1) * internal_maker_fee, ZERO)
        if self.get_taker_fee() < minimum_taker_fee:
            alerts.raise_alert(
                "The taker can't afford to pay the maker rebate for %s (affects internal crossing fees)" % self,
                alerts.MEDIUM,
                "If there's an internal cross, fee will be zero. Maker fee: {}, taker fee: {}, order params: {}".format(
                    self.get_maker_fee(), self.get_taker_fee(), order_params
                ),
            )
            return ZERO
        return internal_maker_fee if order_params.get(POST_ONLY, True) else minimum_taker_fee

    def test_suspend_levels(self, suspend, reason="test", level=0, side=BUY, all_level_higher_than=False):
        self.set_suspend_trading(suspend, reason=reason, cxl_all=False, level_num=level, side=side, all_level_higher_than=all_level_higher_than)  # pylint: disable=E1123

    def test_rejected_order(self, enable=0):  # 0 = NO EFFECT, 1 = ACK, 2 = REJECTED
        self.test_order_rejection(enable)  # pylint: disable=E1123,E1101

    def set_market_data_provider(self, market_data_provider):
        self._market_data_provider = market_data_provider

    def refresh_latest_book(self):
        if self._market_data_provider:
            book = self._market_data_provider.get_latest_book()
            if book:
                log.log_occassionally_with_level(
                    logging.DEBUG,
                    f"Refreshed latest book on order placer {self.get_exchange} ({self.default_asset}/{self.default_currency})",
                    60,
                    "Refreshed latest book on order placer {} ({}/{})",
                    self.get_exchange(),
                    self.default_asset,
                    self.default_currency,
                )
                self._book = book

    def disconnect_market_data_provider(self):
        if self._market_data_provider:
            self._market_data_provider.disconnect_md()
            self._market_data_provider = None
