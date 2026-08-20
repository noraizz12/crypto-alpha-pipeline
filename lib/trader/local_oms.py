"""Local OMS adapters for direct Binance REST/WebSocket communication.

Provides LocalOMSSender and LocalOMSListener that replace ZMQ-based OMS
communication when trader.py is run with --local-oms. No ZMQ sockets are
ever created when these adapters are used.

LocalOMSSender: Parses the JSON envelope format from cpp_oms_create_order_msg,
signs with HMAC, and submits synchronously via requests to Binance REST.

LocalOMSListener: Runs Binance user data WebSocket in a background thread,
accumulates messages via get_msgs() with the same format as CppOMSListener.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
import traceback
from datetime import datetime as dt, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import websockets

from lib.util.aws import load_aws_secrets
from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

REST_ENDPOINT = "https://papi.binance.com"
WS_ENDPOINT = "wss://fstream.binance.com/pm/ws"
DEFAULT_SECRET_ID = "statarb/binance-trading"
LISTEN_KEY_REFRESH_SECS = 30 * 60
RECONNECT_DELAYS = [1, 2, 5, 10, 15]

# Maps envelope payload_type to (HTTP method, path)
ENDPOINT_MAP = {
    "UM_CREATE_ORDER": ("POST", "/papi/v1/um/order"),
    "UM_CANCEL_ORDER": ("DELETE", "/papi/v1/um/order"),
    "UM_CANCEL_ALL_OPEN_ORDERS": ("DELETE", "/papi/v1/um/allOpenOrders"),
}

KNOWN_EVENTS = {
    "ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE", "ACCOUNT_CONFIG_UPDATE",
    "listenKeyExpired", "liabilityChange", "riskLevelChange",
    "balanceUpdate", "executionReport", "outboundAccountPosition",
}


def _load_credentials(aws_secret_id: str = DEFAULT_SECRET_ID) -> Tuple[str, str]:
    """Load API key and secret from AWS Secrets Manager."""
    secrets = load_aws_secrets(statarb_secretid=aws_secret_id)
    if secrets is None:
        raise RuntimeError(f"Failed to load AWS secrets for {aws_secret_id}")
    api_key = secrets.get("api_key") or secrets.get("API")
    api_secret = secrets.get("secret_key") or secrets.get("SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            f"Missing API/SECRET keys in secret {aws_secret_id}. "
            f"Available keys: {list(secrets.keys())}"
        )
    return api_key, api_secret


def _sign_request(api_secret: str, query_string: str) -> str:
    """HMAC-SHA256 signature for Binance REST API."""
    return hmac.new(
        api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class LocalOMSSender:
    """Replaces ZMQ PUSH socket. Sends orders directly to Binance REST.

    Parses the JSON envelope format produced by cpp_oms_create_order_msg,
    cpp_oms_cancel_order_msg, and cpp_oms_cancel_all_open_orders_msg.
    """

    def __init__(self, aws_secret_id: str = DEFAULT_SECRET_ID):
        self.api_key, self.api_secret = _load_credentials(aws_secret_id)
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})
        logger.info("LocalOMSSender initialized")

    def send(self, msg: str) -> bool:
        """Parse JSON envelope and dispatch to Binance REST.

        Args:
            msg: JSON string in the cpp_oms envelope format.

        Returns:
            True if the request was accepted (2xx), False otherwise.
        """
        try:
            envelope = json.loads(msg)
        except json.JSONDecodeError:
            logger.error("LocalOMSSender: failed to parse message JSON", key="local_oms_parse_error")
            return False

        payload_type = envelope.get("payload_type")
        if payload_type not in ENDPOINT_MAP:
            logger.error(
                "LocalOMSSender: unknown payload_type=%s", payload_type,
                key="local_oms_unknown_payload"
            )
            return False

        http_method, path = ENDPOINT_MAP[payload_type]
        payload = dict(envelope.get("payload", {}))

        return self._submit(http_method, path, payload, payload_type)

    def _submit(self, method: str, path: str, params: dict, payload_type: str) -> bool:
        """Sign and submit a request to Binance REST."""
        # Refresh timestamp to stay within recvWindow
        params["timestamp"] = str(int(time.time() * 1000))

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Convert non-string values
        params = {k: str(v) if not isinstance(v, str) else v for k, v in params.items()}

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = _sign_request(self.api_secret, query_string)
        query_string += f"&signature={signature}"

        url = f"{REST_ENDPOINT}{path}?{query_string}"

        try:
            if method == "POST":
                resp = self._session.post(url, timeout=10)
            elif method == "DELETE":
                resp = self._session.delete(url, timeout=10)
            else:
                logger.error("LocalOMSSender: unsupported HTTP method %s", method)
                return False

            body = resp.text
            success = 200 <= resp.status_code < 300

            if success:
                logger.info(
                    "LocalOMSSender: %s %s -> %d", payload_type, path, resp.status_code
                )
            else:
                logger.error(
                    "LocalOMSSender: %s %s -> %d body=%s",
                    payload_type, path, resp.status_code, body[:500],
                    key="local_oms_rest_error"
                )
            return success
        except requests.RequestException as exc:
            logger.error(
                "LocalOMSSender: request exception for %s: %s",
                payload_type, exc,
                key="local_oms_request_exception"
            )
            return False

    def close(self) -> None:
        """Close the HTTP session. Interface compat with ZMQ socket."""
        self._session.close()
        logger.info("LocalOMSSender closed")

    @property
    def closed(self) -> bool:
        """Interface compat with ZMQ socket closed check."""
        return False


class LocalOMSListener(threading.Thread):
    """Replaces CppOMSListener. Receives fills directly from Binance WebSocket.

    Runs an asyncio event loop in a background thread, connecting to the
    Binance user data WebSocket stream. Accumulates messages in the same
    format as CppOMSListener so trader.py's process_oms_messages() works
    unchanged.
    """

    def __init__(self, aws_secret_id: str = DEFAULT_SECRET_ID):
        threading.Thread.__init__(self, name="LocalOMSListenerThread")
        self.daemon = True
        self._aws_secret_id = aws_secret_id
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self.please_kill_me = False
        self.dead = False
        self._lock = threading.Lock()
        self._msgs: List[Tuple[dt, Any]] = []
        self._reconnect_attempt = 0
        self._listen_key: Optional[str] = None

    def kill_me(self) -> None:
        """Signal the listener to shut down."""
        self.please_kill_me = True

    def get_msgs(self) -> List[Tuple[dt, Dict[str, Any]]]:
        """Drain accumulated messages. Thread-safe."""
        with self._lock:
            msgs = self._msgs.copy()
            self._msgs = []
        return msgs

    def _append_msg(self, msg: Dict[str, Any]) -> None:
        """Thread-safe message append."""
        with self._lock:
            self._msgs.append((dt.now(timezone.utc), msg))

    def run(self) -> None:
        """Thread entry point: load credentials and run asyncio loop."""
        try:
            self.api_key, self.api_secret = _load_credentials(self._aws_secret_id)
            logger.info("LocalOMSListener credentials loaded")
        except Exception as exc:
            logger.error("LocalOMSListener: failed to load credentials: %s", exc)
            self.dead = True
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main_loop())
        except Exception as exc:
            logger.error("LocalOMSListener: main loop crashed: %s", exc)
        finally:
            loop.close()
            self.dead = True
            logger.info("LocalOMSListener exited")

    async def _main_loop(self) -> None:
        """Outer reconnection loop, mirrors binance_oms.py."""
        while not self.please_kill_me:
            try:
                self._listen_key = await self._create_listen_key()
                if self._listen_key is None:
                    logger.error("LocalOMSListener: failed to create listen key, retrying")
                    await self._backoff_delay()
                    continue

                self._reconnect_attempt = 0
                await self._run_ws_session()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "LocalOMSListener: error in main loop: %s\n%s",
                    exc, traceback.format_exc()
                )
                await self._backoff_delay()

    async def _create_listen_key(self) -> Optional[str]:
        """POST /papi/v1/listenKey to get ephemeral WS token."""
        url = f"{REST_ENDPOINT}/papi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            resp = requests.post(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.error(
                    "LocalOMSListener: listen key creation failed: status=%d body=%s",
                    resp.status_code, resp.text
                )
                return None
            data = resp.json()
            listen_key = data.get("listenKey")
            logger.info("LocalOMSListener: listen key created: %s...", listen_key[:16])
            return listen_key
        except Exception as exc:
            logger.error("LocalOMSListener: exception creating listen key: %s", exc)
            return None

    async def _refresh_listen_key(self) -> bool:
        """PUT /papi/v1/listenKey to keep it alive."""
        url = f"{REST_ENDPOINT}/papi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            resp = requests.put(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.error(
                    "LocalOMSListener: listen key refresh failed: status=%d body=%s",
                    resp.status_code, resp.text
                )
                return False
            logger.info("LocalOMSListener: listen key refreshed")
            return True
        except Exception as exc:
            logger.error("LocalOMSListener: exception refreshing listen key: %s", exc)
            return False

    async def _keep_alive_loop(self) -> None:
        """Periodically refresh the listen key."""
        while not self.please_kill_me:
            await asyncio.sleep(LISTEN_KEY_REFRESH_SECS)
            if self.please_kill_me:
                break
            await self._refresh_listen_key()

    async def _run_ws_session(self) -> None:
        """Connect to WS and run message + keep-alive concurrently."""
        ws_url = f"{WS_ENDPOINT}/{self._listen_key}"
        logger.info("LocalOMSListener: connecting to WebSocket: %s...", ws_url[:60])

        async with websockets.connect(
            ws_url, ping_interval=60, ping_timeout=30, close_timeout=5,
        ) as ws_conn:
            logger.info("LocalOMSListener: WebSocket connected")
            keep_alive_task = asyncio.create_task(self._keep_alive_loop())
            try:
                async for raw_msg in ws_conn:
                    if self.please_kill_me:
                        break
                    self._handle_message(raw_msg)
            finally:
                keep_alive_task.cancel()
                try:
                    await keep_alive_task
                except asyncio.CancelledError:
                    pass

    def _handle_message(self, raw_msg: str) -> None:
        """Parse WS message and append in CppOMSListener-compatible format."""
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error("LocalOMSListener: failed to parse JSON: %s", raw_msg[:200])
            return

        event_type = msg.get("e", "unknown")

        if event_type == "listenKeyExpired":
            logger.warning("LocalOMSListener: listen key expired, will reconnect")
            self._listen_key = None
            return

        logger.info("LocalOMSListener RAW MSG: %s", msg)

        if event_type == "ORDER_TRADE_UPDATE":
            order_update = msg.get("o", {})
            client_order_id = order_update.get("c", "")
            symbol = order_update.get("s", "")
            order_id = order_update.get("i", "")

            # Skip auto-close / ADL orders (same logic as CppOMSListener)
            if client_order_id.startswith(("autoclose", "adl_autoclose", "adl_close")):
                logger.info(
                    "LocalOMSListener: %s ORDER_TRADE_UPDATE - cannot synthesize mapping",
                    client_order_id.split("_")[0]
                )
            else:
                if order_update.get("x") == "NEW":
                    mapping_msg = {
                        "typ": "NEW_ORDER_OID_MAPPING",
                        "koid": client_order_id,
                        "symbol": symbol,
                        "oid": order_id,
                    }
                    logger.info(
                        "LocalOMSListener: synthesize NEW_ORDER_OID_MAPPING: %s",
                        mapping_msg
                    )
                    self._append_msg(mapping_msg)

            self._append_msg(msg)

        elif event_type in KNOWN_EVENTS:
            logger.info("LocalOMSListener: %s: %s", event_type, msg)

        elif "code" in msg:
            logger.warning("LocalOMSListener: exchange error %s", msg)
            msg["typ"] = "EXCHANGE_REJECTION_MSG"
            self._append_msg(msg)

        else:
            logger.error(
                "LocalOMSListener: could not decode message %s",
                msg,
                key="local_oms_decode_error"
            )

    async def _backoff_delay(self) -> None:
        """Exponential backoff before reconnection."""
        idx = min(self._reconnect_attempt, len(RECONNECT_DELAYS) - 1)
        delay = RECONNECT_DELAYS[idx]
        self._reconnect_attempt += 1
        logger.info(
            "LocalOMSListener: reconnect attempt %d, waiting %ds",
            self._reconnect_attempt, delay
        )
        await asyncio.sleep(delay)
