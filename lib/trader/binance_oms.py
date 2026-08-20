"""Binance Futures OMS - User data streaming and order execution.

This module provides a Python implementation for:
1. WebSocket connection to Binance user data stream
2. Real-time account event processing (orders, fills, positions)
3. ZMQ PUB publishing of events for downstream consumption
4. Manual order submission via command line

Key Features:
- Automatic reconnection with exponential backoff
- Listen key management and refresh
- HMAC-SHA256 request signing for authenticated endpoints
- Safety limits on manual orders

Usage:
    python -m lib.trader.binance_oms [--debug] [--listen] [--secret-id SECRET] [--zmq-port PORT]
    python -m lib.trader.binance_oms --manual "B ETHUSDT 0.01 @ 2500.0 LIMIT GTC"

Examples:
    # Debug mode - print to stdout, no ZMQ
    python -m lib.trader.binance_oms --debug

    # Listen mode - read-only, logs all raw network traffic to stdout
    python -m lib.trader.binance_oms --listen

    # Production mode - publish via ZMQ on port 5555
    python -m lib.trader.binance_oms

    # Manual order - submit a single order and exit
    python -m lib.trader.binance_oms --manual "B ETHUSDT 0.01 @ 2500.0"
    python -m lib.trader.binance_oms --manual "S BTCUSDT 0.001 @ 100000.0 LIMIT GTC"
    python -m lib.trader.binance_oms --manual "B ETHUSDT 0.5 @ 2500.0 REDUCE GTC"
    python -m lib.trader.binance_oms --manual "S SOLUSDT 10 @ 200.0 MARKET"
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import signal
import time
from datetime import datetime as dt, timezone
from typing import Dict, Optional

import aiohttp
import websockets
import zmq

from lib.util.aws import load_aws_secrets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WS_ENDPOINT = "wss://fstream.binance.com/pm/ws"
REST_ENDPOINT = "https://papi.binance.com"
LISTEN_KEY_REFRESH_SECS = 30 * 60  # 30 minutes
RECONNECT_DELAYS = [1, 2, 5, 10, 15]  # exponential backoff schedule in seconds
DEFAULT_SECRET_ID = "statarb/binance-trading"
DEFAULT_ZMQ_PORT = 5555
MAX_MANUAL_NOTIONAL = 10000  # safety limit for manual orders
UM_ORDER_ENDPOINT = "/papi/v1/um/order"
RECV_WINDOW_MS = 5000
DEFAULT_WAIT_TIMEOUT_SECS = 60
TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}

# Known event types from Binance user data stream
KNOWN_EVENTS = {
    "ORDER_TRADE_UPDATE",
    "ACCOUNT_UPDATE",
    "ACCOUNT_CONFIG_UPDATE",
    "listenKeyExpired",
    "liabilityChange",
    "riskLevelChange",
    "balanceUpdate",
    "executionReport",
    "outboundAccountPosition",
}


class BinanceOMS:
    """Python backup OMS for Binance futures user data streaming.

    Connects to Binance user data WebSocket stream and publishes
    received events via ZMQ PUB, matching the C++ OMS interface.
    """

    def __init__(
        self,
        aws_secret_id: str = DEFAULT_SECRET_ID,
        ws_endpoint: str = WS_ENDPOINT,
        rest_endpoint: str = REST_ENDPOINT,
        zmq_pub_port: int = DEFAULT_ZMQ_PORT,
        debug: bool = False,
        listen: bool = False,
    ):
        self.aws_secret_id = aws_secret_id
        self.ws_endpoint = ws_endpoint
        self.rest_endpoint = rest_endpoint
        self.zmq_pub_port = zmq_pub_port
        self.debug = debug
        self.listen = listen

        # Credentials (loaded in start())
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None

        # WebSocket state
        self.listen_key: Optional[str] = None
        self.ws_connection = None
        self._reconnect_attempt = 0
        self._shutdown = False

        # ZMQ publisher
        self._zmq_ctx: Optional[zmq.Context] = None
        self._zmq_socket: Optional[zmq.Socket] = None

        # Stats
        self.stats = {
            "messages_received": 0,
            "messages_published": 0,
            "reconnections": 0,
            "listen_key_refreshes": 0,
            "start_time": None,
        }

    async def start(self) -> None:
        """Main entry point - load secrets, set up ZMQ, connect WS."""
        self.stats["start_time"] = dt.now(timezone.utc).isoformat()
        mode = "listen" if self.listen else ("debug" if self.debug else "production")
        logger.info("BinanceOMS starting (mode=%s)", mode)
        if self.listen:
            logger.info("LISTEN MODE: read-only, logging all raw network traffic")

        self._load_credentials()
        if not self.debug and not self.listen:
            self._setup_zmq_publisher()

        while not self._shutdown:
            try:
                self.listen_key = await self._create_listen_key()
                if self.listen_key is None:
                    logger.error("Failed to create listen key, retrying...")
                    await self._backoff_delay()
                    continue

                self._reconnect_attempt = 0
                await self._run_ws_session()

            except asyncio.CancelledError:
                logger.info("BinanceOMS cancelled, shutting down")
                break
            except Exception as exc:
                logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
                await self._backoff_delay()

        self._cleanup()

    def _load_credentials(self) -> None:
        """Load API credentials from AWS Secrets Manager."""
        logger.info("Loading credentials from AWS secret: %s", self.aws_secret_id)
        secrets = load_aws_secrets(statarb_secretid=self.aws_secret_id)
        if secrets is None:
            raise RuntimeError(
                f"Failed to load AWS secrets for {self.aws_secret_id}"
            )

        self.api_key = secrets.get("api_key") or secrets.get("API")
        self.api_secret = secrets.get("secret_key") or secrets.get("SECRET")

        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                f"Missing API/SECRET keys in secret {self.aws_secret_id}. "
                f"Available keys: {list(secrets.keys())}"
            )
        logger.info("Credentials loaded successfully")

    def _sign_request(self, query_string: str) -> str:
        """HMAC-SHA256 signature for Binance REST API."""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _create_listen_key(self) -> Optional[str]:
        """POST /papi/v1/listenKey to get ephemeral WS token."""
        url = f"{self.rest_endpoint}/papi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    body = await resp.text()
                    if self.listen:
                        self._log_rest("POST", url, resp.status, body)
                    if resp.status != 200:
                        logger.error(
                            "Failed to create listen key: status=%d body=%s",
                            resp.status, body,
                        )
                        return None
                    data = json.loads(body)
                    listen_key = data.get("listenKey")
                    logger.info("Listen key created: %s...", listen_key[:16])
                    return listen_key
        except Exception as exc:
            logger.error("Exception creating listen key: %s", exc)
            return None

    async def _refresh_listen_key(self) -> bool:
        """PUT /papi/v1/listenKey to keep it alive."""
        url = f"{self.rest_endpoint}/papi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    body = await resp.text()
                    if self.listen:
                        self._log_rest("PUT", url, resp.status, body)
                    if resp.status != 200:
                        logger.error(
                            "Failed to refresh listen key: status=%d body=%s",
                            resp.status, body,
                        )
                        return False
                    self.stats["listen_key_refreshes"] += 1
                    logger.info(
                        "Listen key refreshed (total: %d)",
                        self.stats["listen_key_refreshes"],
                    )
                    return True
        except Exception as exc:
            logger.error("Exception refreshing listen key: %s", exc)
            return False

    async def _keep_alive_loop(self) -> None:
        """Background task to refresh listen key periodically."""
        while not self._shutdown:
            await asyncio.sleep(LISTEN_KEY_REFRESH_SECS)
            if self._shutdown:
                break
            success = await self._refresh_listen_key()
            if not success:
                logger.warning("Listen key refresh failed, will retry next cycle")

    async def _run_ws_session(self) -> None:
        """Connect to WS and run message loop + keep-alive concurrently."""
        ws_url = f"{self.ws_endpoint}/{self.listen_key}"
        logger.info("Connecting to WebSocket: %s...", ws_url[:60])

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=60,
                ping_timeout=30,
                close_timeout=5,
            ) as ws_conn:
                self.ws_connection = ws_conn
                logger.info("WebSocket connected")

                keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                try:
                    await self._ws_message_loop(ws_conn)
                finally:
                    keep_alive_task.cancel()
                    try:
                        await keep_alive_task
                    except asyncio.CancelledError:
                        pass
                    self.ws_connection = None

        except websockets.exceptions.InvalidStatusCode as exc:
            logger.error("WebSocket connection rejected: %s", exc)
            raise
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning("WebSocket connection closed: %s", exc)
            self.stats["reconnections"] += 1
            raise
        except Exception as exc:
            logger.error("WebSocket error: %s", exc, exc_info=True)
            self.stats["reconnections"] += 1
            raise

    async def _ws_message_loop(self, ws_conn) -> None:
        """Receive and process WebSocket messages."""
        async for raw_msg in ws_conn:
            if self._shutdown:
                break
            await self._handle_message(raw_msg)

    async def _handle_message(self, raw_msg: str) -> None:
        """Parse message, log it, publish via ZMQ."""
        self.stats["messages_received"] += 1
        recv_ts = dt.now(timezone.utc)

        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON: %s", raw_msg[:200])
            return

        event_type = msg.get("e", "unknown")
        timestamp_str = recv_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if event_type == "listenKeyExpired":
            logger.warning("Listen key expired, will reconnect with new key")
            self.listen_key = None
            if self.ws_connection:
                await self.ws_connection.close()
            return

        if event_type in KNOWN_EVENTS:
            log_msg = self._format_log_message(event_type, msg, timestamp_str)
            logger.info(log_msg)
        else:
            logger.warning(
                "[%s] Unknown event type '%s': %s",
                timestamp_str, event_type, raw_msg[:500],
            )

        if self.listen:
            self._log_ws(timestamp_str, raw_msg)
        elif self.debug:
            print(f"[{timestamp_str}] {event_type}: {json.dumps(msg, indent=2)}")
        else:
            self._publish_zmq(raw_msg)

    def _format_log_message(
        self, event_type: str, msg: dict, timestamp_str: str
    ) -> str:
        """Format a log message based on event type."""
        if event_type == "ORDER_TRADE_UPDATE":
            order = msg.get("o", {})
            return (
                f"[{timestamp_str}] ORDER_TRADE_UPDATE: "
                f"symbol={order.get('s')} side={order.get('S')} "
                f"type={order.get('o')} status={order.get('X')} "
                f"exec_type={order.get('x')} qty={order.get('q')} "
                f"price={order.get('p')} filled={order.get('z')} "
                f"client_oid={order.get('c')}"
            )
        if event_type == "ACCOUNT_UPDATE":
            return f"[{timestamp_str}] ACCOUNT_UPDATE: {json.dumps(msg)[:300]}"
        return f"[{timestamp_str}] {event_type}: {json.dumps(msg)[:300]}"

    @staticmethod
    def _log_rest(method: str, url: str, status: int, body: str) -> None:
        """Log raw REST response to stdout."""
        ts_str = dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{ts_str}] REST {method} {url} -> {status}")
        print(f"  {body}")

    @staticmethod
    def _log_ws(timestamp_str: str, raw_msg: str) -> None:
        """Log raw WebSocket message to stdout."""
        try:
            formatted = json.dumps(json.loads(raw_msg), indent=2)
        except (json.JSONDecodeError, TypeError):
            formatted = raw_msg
        print(f"[{timestamp_str}] WS <<< {formatted}")

    # ── Order Submission (Phase 2) ──────────────────────────────────────

    @staticmethod
    def parse_order_string(order_str: str) -> Dict[str, str]:
        """Parse a manual order string into order parameters.

        Supported formats:
            "B ETHUSDT 0.01 @ 2500.0"                  -> LIMIT GTX default
            "B ETHUSDT 0.01 @ 2500.0 LIMIT GTC"        -> explicit type + TIF
            "S BTCUSDT 0.001 @ 100000.0 REDUCE GTC"    -> reduce-only limit
            "B SOLUSDT 10 @ 200.0 MARKET"              -> market order (no TIF)
        """
        toks = order_str.strip().split()
        if len(toks) < 5 or toks[3] != "@":
            raise ValueError(
                f"Invalid order format: '{order_str}'. "
                "Expected: SIDE SYMBOL QTY @ PRICE [TYPE] [TIF]"
            )

        side_str = toks[0].upper()
        symbol = toks[1].upper()
        qty_str = toks[2]
        price_str = toks[4]

        if side_str in ("B", "BUY"):
            side = "BUY"
        elif side_str in ("S", "SELL"):
            side = "SELL"
        else:
            raise ValueError(f"Unknown side '{side_str}', use B/S or BUY/SELL")

        qty = float(qty_str)
        price = float(price_str)
        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Defaults
        order_type = "LIMIT"
        tif = "GTX"
        reduce_only = False

        remaining = [t.upper() for t in toks[5:]]

        if remaining:
            raw_type = remaining[0]
            if raw_type == "MARKET":
                order_type = "MARKET"
                tif = None
            elif raw_type in ("REDUCE", "REDUCE_ONLY"):
                order_type = "LIMIT"
                reduce_only = True
            elif raw_type in ("POST_ONLY", "POST_ONLY_REDUCE"):
                order_type = "LIMIT"
                tif = "GTX"
                if "REDUCE" in raw_type:
                    reduce_only = True
            elif raw_type == "LIMIT":
                order_type = "LIMIT"
            else:
                raise ValueError(f"Unknown order type '{raw_type}'")

        if len(remaining) >= 2 and order_type != "MARKET":
            raw_tif = remaining[1]
            if raw_tif in ("GTC", "GTX", "IOC", "FOK"):
                tif = raw_tif
            else:
                raise ValueError(f"Unknown TIF '{raw_tif}'")

        notional = qty * price
        if notional > MAX_MANUAL_NOTIONAL:
            raise ValueError(
                f"Order notional ${notional:,.0f} exceeds "
                f"safety limit ${MAX_MANUAL_NOTIONAL:,}"
            )

        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": qty_str,
            "positionSide": "BOTH",
            "recvWindow": str(RECV_WINDOW_MS),
        }
        if order_type != "MARKET":
            params["price"] = price_str
        if tif is not None:
            params["timeInForce"] = tif
        if reduce_only:
            params["reduceOnly"] = "true"

        return params

    async def _submit_order(self, params: Dict[str, str]) -> dict:
        """Submit order to Binance PAPI REST endpoint with HMAC signature."""
        url = f"{self.rest_endpoint}{UM_ORDER_ENDPOINT}"
        headers = {"X-MBX-APIKEY": self.api_key}

        params["timestamp"] = str(int(time.time() * 1000))

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = self._sign_request(query_string)
        query_string += f"&signature={signature}"

        full_url = f"{url}?{query_string}"
        logger.info("Submitting order: %s", full_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                full_url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()
                logger.info("Order response: status=%d body=%s", resp.status, body)
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return {"status": resp.status, "raw": body}

    @staticmethod
    def _make_client_order_id() -> str:
        """Generate a unique client order ID."""
        ts = dt.now(timezone.utc).strftime("%y%m%d%H%M%S%f")
        raw = f"oms_manual_{ts}"
        return hashlib.sha1(raw.encode()).hexdigest()[:22]

    async def execute_manual_order(
        self, order_str: str, wait: bool = False, wait_timeout: int = DEFAULT_WAIT_TIMEOUT_SECS
    ) -> None:
        """Parse, confirm, submit a manual order, optionally wait for fill."""
        self._load_credentials()

        params = self.parse_order_string(order_str)
        client_oid = self._make_client_order_id()
        params["newClientOrderId"] = client_oid

        side = params["side"]
        symbol = params["symbol"]
        qty = params["quantity"]
        price = params.get("price", "MARKET")
        order_type = params["type"]
        tif = params.get("timeInForce", "N/A")
        reduce_only = params.get("reduceOnly", "false")
        notional = float(qty) * float(price) if price != "MARKET" else 0

        print("\n=== MANUAL ORDER ===")
        print(f"  Side:         {side}")
        print(f"  Symbol:       {symbol}")
        print(f"  Quantity:     {qty}")
        print(f"  Price:        {price}")
        print(f"  Type:         {order_type}")
        print(f"  TIF:          {tif}")
        print(f"  Reduce Only:  {reduce_only}")
        print(f"  Client OID:   {client_oid}")
        if notional > 0:
            print(f"  Notional:     ${notional:,.2f}")
        if wait:
            print(f"  Wait:         {wait_timeout}s timeout")
        print("====================\n")

        confirm = input("Confirm order? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Order cancelled.")
            return

        if wait:
            await self._submit_and_wait(params, client_oid, wait_timeout)
        else:
            result = await self._submit_order(params)
            print(f"\n=== RESULT ===\n{json.dumps(result, indent=2)}\n")

    async def _submit_and_wait(
        self, params: Dict[str, str], client_oid: str, timeout_secs: int
    ) -> None:
        """Submit order, connect WS, and wait for terminal status."""
        # Connect WebSocket first so we don't miss the ack
        self.listen_key = await self._create_listen_key()
        if self.listen_key is None:
            print("ERROR: Failed to create listen key, cannot wait for fill.")
            return

        ws_url = f"{self.ws_endpoint}/{self.listen_key}"

        async with websockets.connect(
            ws_url, ping_interval=60, ping_timeout=30, close_timeout=5,
        ) as ws_conn:
            print("WebSocket connected, submitting order...")

            result = await self._submit_order(params)
            rest_status = result.get("status", result.get("ordStatus", ""))
            print(f"\n=== REST RESPONSE ===\n{json.dumps(result, indent=2)}")

            # If REST already shows terminal status (e.g. market fill), we're done
            if rest_status in TERMINAL_ORDER_STATUSES:
                print(f"\nOrder immediately {rest_status}")
                return

            if "code" in result:
                print(f"\nOrder rejected by exchange: {result}")
                return

            print(f"\nWaiting up to {timeout_secs}s for order updates...\n")

            try:
                final_status = await asyncio.wait_for(
                    self._wait_for_order_updates(ws_conn, client_oid),
                    timeout=timeout_secs,
                )
                print(f"\nOrder reached terminal status: {final_status}")
            except asyncio.TimeoutError:
                print(f"\nTimeout after {timeout_secs}s. Order may still be open.")

    async def _wait_for_order_updates(
        self, ws_conn, client_oid: str
    ) -> str:
        """Listen on WS for ORDER_TRADE_UPDATE matching our client OID."""
        async for raw_msg in ws_conn:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            if msg.get("e") != "ORDER_TRADE_UPDATE":
                continue

            order_data = msg.get("o", {})
            if order_data.get("c") != client_oid:
                continue

            status = order_data.get("X", "")
            exec_type = order_data.get("x", "")
            filled_qty = order_data.get("z", "0")
            avg_price = order_data.get("ap", "0")
            last_qty = order_data.get("l", "0")
            last_price = order_data.get("L", "0")

            ts_str = dt.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

            if exec_type == "TRADE":
                print(
                    f"  [{ts_str}] FILL: {last_qty} @ {last_price}  "
                    f"(total filled: {filled_qty}, avg: {avg_price})"
                )
            else:
                print(f"  [{ts_str}] {exec_type}: status={status}")

            if status in TERMINAL_ORDER_STATUSES:
                return status

        return "DISCONNECTED"

    def _setup_zmq_publisher(self) -> None:
        """Create ZMQ PUB socket binding on configured port."""
        bind_url = f"tcp://*:{self.zmq_pub_port}"
        self._zmq_ctx = zmq.Context()
        self._zmq_socket = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        self._zmq_socket.bind(bind_url)
        logger.info("ZMQ PUB socket bound to %s", bind_url)

    def _publish_zmq(self, msg: str) -> None:
        """Publish raw JSON string via ZMQ PUB."""
        if self._zmq_socket is None:
            return
        try:
            self._zmq_socket.send_string(msg)
            self.stats["messages_published"] += 1
        except zmq.ZMQError as exc:
            logger.error("ZMQ publish error: %s", exc)

    async def _backoff_delay(self) -> None:
        """Wait with exponential backoff before reconnection."""
        idx = min(self._reconnect_attempt, len(RECONNECT_DELAYS) - 1)
        delay = RECONNECT_DELAYS[idx]
        self._reconnect_attempt += 1
        logger.info(
            "Reconnect attempt %d, waiting %ds...",
            self._reconnect_attempt, delay,
        )
        await asyncio.sleep(delay)

    def _cleanup(self) -> None:
        """Clean up ZMQ resources."""
        logger.info("Cleaning up resources. Stats: %s", self.stats)
        if self._zmq_socket is not None:
            self._zmq_socket.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()

    def shutdown(self, loop: asyncio.AbstractEventLoop) -> None:
        """Signal the OMS to shut down gracefully."""
        if self._shutdown:
            logger.info("Forced shutdown")
            loop.stop()
            return
        logger.info("Shutdown requested")
        self._shutdown = True
        if self.ws_connection is not None:
            asyncio.ensure_future(self.ws_connection.close(), loop=loop)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Python backup OMS for Binance futures user data streaming"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print messages to stdout, skip ZMQ publishing",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Read-only mode: connect to live account, log all raw network traffic, no ZMQ",
    )
    parser.add_argument(
        "--manual",
        type=str,
        default=None,
        help='Submit a single order and exit. Format: "B ETHUSDT 0.01 @ 2500.0 [LIMIT GTC]"',
    )
    parser.add_argument(
        "--wait",
        nargs="?",
        const=DEFAULT_WAIT_TIMEOUT_SECS,
        type=int,
        default=None,
        help=f"With --manual: wait for order fill via WebSocket (default timeout: {DEFAULT_WAIT_TIMEOUT_SECS}s)",
    )
    parser.add_argument(
        "--secret-id",
        default=DEFAULT_SECRET_ID,
        help=f"AWS Secrets Manager secret ID (default: {DEFAULT_SECRET_ID})",
    )
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=DEFAULT_ZMQ_PORT,
        help=f"ZMQ PUB port (default: {DEFAULT_ZMQ_PORT})",
    )
    parser.add_argument(
        "--ws-endpoint",
        default=WS_ENDPOINT,
        help=f"WebSocket endpoint (default: {WS_ENDPOINT})",
    )
    parser.add_argument(
        "--rest-endpoint",
        default=REST_ENDPOINT,
        help=f"REST endpoint (default: {REST_ENDPOINT})",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Python backup OMS."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    oms = BinanceOMS(
        aws_secret_id=args.secret_id,
        ws_endpoint=args.ws_endpoint,
        rest_endpoint=args.rest_endpoint,
        zmq_pub_port=args.zmq_port,
        debug=args.debug,
        listen=args.listen,
    )

    if args.manual:
        wait = args.wait is not None
        wait_timeout = args.wait if wait else DEFAULT_WAIT_TIMEOUT_SECS
        asyncio.run(oms.execute_manual_order(args.manual, wait=wait, wait_timeout=wait_timeout))
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, oms.shutdown, loop)

    try:
        loop.run_until_complete(oms.start())
    finally:
        loop.close()
        logger.info("BinanceOMS exited")


if __name__ == "__main__":
    main()
