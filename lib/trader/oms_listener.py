import logging
import threading
import time
import json
import traceback
from datetime import datetime as dt, timezone
from typing import Any, Dict, List, Tuple

import zmq

from lib.util.util import STATARB_OMS_IP, STATARB_OMS_PORT
from lib.util.util import STATARB_CPP_OMS_IP, STATARB_CPP_OMS_USER_DATA_PORT, STATARB_CPP_OMS_REPLY_PORT
from lib.util.logging_util import KeyLogger
from .zmq_util import add_subscriber

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

NEW_ORDER_MAPPING_MSG_LENGTH = 7


class OMSListener(threading.Thread):
    def __init__(self, config: dict):
        threading.Thread.__init__(self, name="OMSListenerThread")
        self.config = config
        self.ws = None
        self.please_kill_me = False
        self.dead = False
        self.last_oms_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)
        self.oms_msg_alert_threshold_mins = 30
        self.oms_sub_socket = None
        self.msgs: List[Tuple[dt, Any]] = []

    def close_socket(self):
        if self.oms_sub_socket is not None:
            self.oms_sub_socket.setsockopt(zmq.LINGER, 0)
            self.oms_sub_socket.close()

    # return here to indicate success or failure
    def subscribe(self) -> bool:
        logger.info("Subscribing to OMS....")
        try:
            # close before try adding a new subscriber to avoid duplicates
            self.close_socket()
            self.oms_sub_socket = add_subscriber(connect_url=f"tcp://{STATARB_OMS_IP}:{STATARB_OMS_PORT}", async_ctx=False)
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to OMS: {e}", key="oms subscription failed")
            return False

    def kill_me(self):
        self.please_kill_me = True

    def get_msgs(self) -> List[Tuple[dt, Dict[str, Any]]]:
        msgs = self.msgs.copy()
        self.msgs = []
        return msgs

    def run(self):
        if not self.subscribe():
            logger.error("Initial OMS subscription failed", key="oms subscription failed")
            self.dead = True
            return

        self.last_oms_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)
        while True:
            if self.please_kill_me:
                break

            time.sleep(0.01)
            now = dt.now(timezone.utc)
            try:
                msg = self.oms_sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again as e:
                # we use this block below to check if zma.Again related to oms listener lost track of popping new msg, could remove after done
                total_secs_since_last_oms_msg = (now - self.last_oms_msg_ts).total_seconds()
                if total_secs_since_last_oms_msg > self.oms_msg_alert_threshold_mins * 60:
                    # log this error every minute if it keeps, otherwise it will spam the whole logging file, also try reconnect the oms by close and subscribe
                    if (now - self.last_error_log_ts).total_seconds() >= 60:
                        logger.error(
                            f"Long time no update over {total_secs_since_last_oms_msg} seconds since {self.last_oms_msg_ts} in oms_listener: {e}",
                            key="no oms msg in oms_listener")
                        self.last_error_log_ts = now
                        self.subscribe()
                continue
            except Exception as e:
                logger.error(f"Error in oms_listener: {e}", key="fail to listen to oms fill")
                print(traceback.format_exc())
                continue
            self.last_oms_msg_ts = now
            logger.info(f"RAW OMS MSG: {msg}")

            if isinstance(msg, str):
                # get order id mapping info from statarb-oms
                # mapping msg format as ORDER|1730915289.7855918|32184039123|AAVEUSDT|NEW|b622e2cd09e24354b615a62abefd8a67|
                msg_lst = msg.split('|')
                if 'ORDER' in msg and 'NEW' in msg and len(msg_lst) == NEW_ORDER_MAPPING_MSG_LENGTH:
                    mapping_msg_dict = {
                        'typ': 'NEW_ORDER_OID_MAPPING',
                        'koid': msg_lst[2],
                        'symbol': msg_lst[3],
                        'oid': msg_lst[5]
                    }
                    self.msgs.append((dt.now(timezone.utc), mapping_msg_dict))
                # ignore these message for now and just use direct binance ones
                continue

            if 'e' in msg:
                event = msg['e']
                if event == 'ORDER_TRADE_UPDATE':
                    self.msgs.append((dt.now(timezone.utc), msg))
                else:
                    logger.error(f"Unknown OMS Event Message: {msg}", key="see unknown oms event msg")
            elif 'code' in msg:
                logger.warning(f"Exchange error: {msg}")
                msg['typ'] = 'EXCHANGE_REJECTION_MSG'
                self.msgs.append((dt.now(timezone.utc), msg))
            else:
                logger.error(f"Could not decode OMS message {msg}", key="fail to decode oms event msg")
        logger.info("Jump out of OMS listener while loop")
        # close the socket after finish while loop
        self.close_socket()
        self.dead = True


class CppOMSListener(threading.Thread):
    def __init__(self, config: dict):
        threading.Thread.__init__(self, name="CppOMSListenerThread")
        self.config = config
        self.ws = None
        self.please_kill_me = False
        self.dead = False
        self.last_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)
        self.msg_alert_threshold_mins = 30
        self.socket = None
        self.msgs: List[Tuple[dt, Any]] = []

    def close_socket(self):
        if self.socket is not None:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()

    # return here to indicate success or failure
    def subscribe(self) -> bool:
        logger.info("Subscribing to C++ OMS....")
        try:
            # close before try adding a new subscriber to avoid duplicates
            self.close_socket()
            connect_url = f"tcp://{STATARB_CPP_OMS_IP}:{STATARB_CPP_OMS_USER_DATA_PORT}"
            self.socket = add_subscriber(connect_url=connect_url, async_ctx=False)
            logger.info(f"Subscribed to C++ OMS on {connect_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to C++ OMS: {e}", key="oms subscription failed")
            return False

    def kill_me(self):
        self.please_kill_me = True

    def get_msgs(self) -> List[Tuple[dt, Dict[str, Any]]]:
        msgs = self.msgs.copy()
        self.msgs = []
        return msgs

    def run(self):
        if not self.subscribe():
            logger.error("Initial C++ OMS subscription failed", key="oms subscription failed")
            self.dead = True
            return

        self.last_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)

        while True:
            if self.please_kill_me:
                break

            events = self.socket.poll(1000)

            if events & zmq.POLLIN:
                now = dt.now(timezone.utc)
                try:
                    raw_msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                    msg = json.loads(raw_msg)
                except zmq.Again as e:
                    # we use this block below to check if zma.Again related to oms listener lost track of popping new msg, could remove after done
                    total_secs_since_last_msg = (now - self.last_msg_ts).total_seconds()
                    if total_secs_since_last_msg > self.msg_alert_threshold_mins * 60:
                        # log this error every minute if it keeps, otherwise it will spam the whole logging file, also try reconnect the oms by close and subscribe
                        if (now - self.last_error_log_ts).total_seconds() >= 60.0:
                            logger.error(
                                f"CppOMSListener: long time no update over {total_secs_since_last_msg} seconds "
                                f"since {self.last_msg_ts} in oms_listener: {e}",
                                key="no oms msg in oms_listener"
                            )
                            self.last_error_log_ts = now
                            self.subscribe()
                    continue
                except Exception as e:
                    logger.error(f"CppOMSListener: error {e}", key="fail to listen to oms fill")
                    print(traceback.format_exc())
                    continue
                self.last_msg_ts = now

                logger.info(f"RAW OMS MSG: {msg}")

                if 'e' in msg:
                    event = msg['e']
                    if event == 'ORDER_TRADE_UPDATE':
                        order_update = msg['o']
                        symbol = order_update['s']
                        order_id = order_update['i']
                        client_order_id = order_update['c']

                        # special order ids represent generated orders, e.g. liquidation and ADL
                        if client_order_id.startswith('autoclose'):
                            logger.info(
                                f"CppOMSListener: autoclose ORDER_TRADE_UPDATE - cannot synthesize order id mapping"
                            )
                        elif client_order_id.startswith('adl_autoclose'):
                            logger.info(
                                f"CppOMSListener: adl_autoclose ORDER_TRADE_UPDATE - cannot synthesize order id mapping"
                            )
                        elif client_order_id.startswith('adl_close'):
                            logger.info(
                                f"CppOMSListener: adl_close ORDER_TRADE_UPDATE - cannot synthesize order id mapping"
                            )
                        else:
                            if order_update['x'] == "NEW":
                                mapping_msg_dict = {
                                    'typ': 'NEW_ORDER_OID_MAPPING',
                                    'koid': client_order_id,
                                    'symbol': symbol,
                                    'oid': order_id
                                }
                                logger.info(f"CppOMSListener: synthesize NEW_ORDER_OID_MAPPING: {mapping_msg_dict}")
                                self.msgs.append((dt.now(timezone.utc), mapping_msg_dict))
                        self.msgs.append((dt.now(timezone.utc), msg))
                    elif event == 'ACCOUNT_UPDATE':
                        logger.info(f"CppOMSListener: ACCOUNT_UPDATE: {msg}")
                    elif event == 'ACCOUNT_CONFIG_UPDATE':
                        logger.info(f"CppOMSListener: ACCOUNT_CONFIG_UPDATE: {msg}")
                    elif event == 'liabilityChange':
                        logger.info(f"CppOMSListener: liabilityChange: {msg}")
                    elif event == 'riskLevelChange':
                        logger.info(f"CppOMSListener: riskLevelChange: {msg}")
                    elif event == 'balanceUpdate':
                        logger.info(f"CppOMSListener: balanceUpdate: {msg}")
                    elif event == 'executionReport':
                        logger.info(f"CppOMSListener: executionReport: {msg}")
                    elif event == 'outboundAccountPosition':
                        logger.info(f"CppOMSListener: outboundAccountPosition: {msg}")
                    elif event == 'listenKeyExpired':
                        # this can also fire if we drop an old key and is per se not a failure but we cannot
                        # determine here if it is the currently used listening key, here we do nothing just
                        # make sure we do not get into an error
                        logger.info(f"CppOMSListener: listenKeyExpired: {msg}")
                    else:
                        logger.error(
                            f"CppOMSListener: unknown OMS Event: event={event} msg={msg}",
                            key="see unknown oms event msg"
                        )
                elif 'code' in msg:
                    logger.warning(f"CppOMSListener: exchange error {msg}")
                    msg['typ'] = 'EXCHANGE_REJECTION_MSG'
                    self.msgs.append((dt.now(timezone.utc), msg))
                else:
                    logger.error(f"Could not decode OMS message {msg}", key="fail to decode oms event msg")

        logger.info("Jump out of OMS listener while loop")

        # close the socket after finish while loop
        self.close_socket()
        self.dead = True


class CppOMSRestReplyListener(threading.Thread):
    def __init__(self, config: dict):
        threading.Thread.__init__(self, name="CppOMSRestReplyListenerThread")
        self.config = config
        self.ws = None
        self.please_kill_me = False
        self.dead = False
        self.last_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)
        self.msg_alert_threshold_mins = 30
        self.socket = None
        self.msgs: List[Tuple[dt, Any]] = []

    def close_socket(self):
        if self.socket is not None:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()

    # return here to indicate success or failure
    def subscribe(self) -> bool:
        logger.info("Subscribing to C++ OMS REST Replies....")
        try:
            # close before try adding a new subscriber to avoid duplicates
            self.close_socket()
            connect_url = f"tcp://{STATARB_CPP_OMS_IP}:{STATARB_CPP_OMS_REPLY_PORT}"
            self.socket = add_subscriber(connect_url=connect_url, async_ctx=False)
            logger.info("Subscribed to C++ OMS REST Replies on {connect_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to C++ OMS Rest reply: {e}", key="oms subscription failed")
            return False

    def kill_me(self):
        self.please_kill_me = True

    def get_msgs(self) -> List[Tuple[dt, Dict[str, Any]]]:
        msgs = self.msgs.copy()
        self.msgs = []
        return msgs

    def run(self):
        if not self.subscribe():
            logger.error("Initial C++ OMS Rest reply subscription failed", key="oms subscription failed")
            self.dead = True
            return

        self.last_msg_ts = dt.now(timezone.utc)
        self.last_error_log_ts = dt.now(timezone.utc)

        while True:
            if self.please_kill_me:
                break

            events = self.socket.poll(1000)

            if events & zmq.POLLIN:
                now = dt.now(timezone.utc)
                try:
                    raw_msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                    msg = json.loads(raw_msg)
                except zmq.Again as e:
                    # we use this block below to check if zma.Again related to oms listener lost track of popping new msg, could remove after done
                    total_secs_since_last_msg = (now - self.last_msg_ts).total_seconds()
                    if total_secs_since_last_msg > self.msg_alert_threshold_mins * 60:
                        # log this error every minute if it keeps, otherwise it will spam the whole logging file, also try reconnect the oms by close and subscribe
                        if (now - self.last_error_log_ts).total_seconds() >= 60.0:
                            logger.error(
                                f"CppOMSRestReplyListener: long time no update over {total_secs_since_last_msg} seconds "
                                f"since {self.last_msg_ts}: {e}",
                                key="no oms msg in oms_listener"
                            )
                            self.last_error_log_ts = now
                            self.subscribe()
                    continue
                except Exception as e:
                    logger.error(f"CppOMSRestReplyListener: error {e}", key="fail to listen to oms fill")
                    print(traceback.format_exc())
                    continue
                self.last_msg_ts = now

                logger.info(f"RAW OMS REST REPLY MSG: {msg}")

        logger.info("Jump out of OMS listener while loop")

        # close the socket after finish while loop
        self.close_socket()
        self.dead = True
