# lib/trader/ - Live Trading Execution System

**Purpose:** Live trading execution for Binance futures. Reads target positions from the alpha server, runs short-term portfolio optimization per wave, places limit orders via OMS, monitors fills, and tracks positions/PnL.

## Architecture Overview

```
Target Files (targets/)
        │
        ▼
   ┌─────────┐     ┌──────────────────┐     ┌──────────────┐
   │ Trader   │────▶│ OMS (C++ or Py)  │────▶│ Binance API  │
   │ (loop)   │◀────│ via ZMQ PUB/PUSH │◀────│ (Futures)    │
   └─────────┘     └──────────────────┘     └──────────────┘
        │  ▲
        │  │  fills/acks via ZMQ SUB
        ▼  │
   ┌─────────────────────────────────────────────┐
   │ TradeQuotePoller ─▶ DataAggregator          │
   │ (Binance WS: depth10@100ms + aggTrade)      │
   │ + OpenInterestUpdater (REST, every 30 min)  │
   └─────────────────────────────────────────────┘
```

### Two OMS Backends

The system supports two OMS backends, selected via the `--old-oms` flag:

1. **C++ OMS (default):** Orders sent as JSON via ZMQ PUSH to `STATARB_CPP_OMS_INGEST_PORT`. Fill/status updates received via ZMQ SUB from `STATARB_CPP_OMS_USER_DATA_PORT` and `STATARB_CPP_OMS_REPLY_PORT`. Messages use `OMSPayloadType` envelopes.
2. **Legacy Python OMS (`--old-oms`):** Orders sent as pipe-delimited strings via ZMQ PUB. Fill updates received via ZMQ SUB from the Python `gateway.py` server. Message format: `PLACE {koid} {side} {symbol} {qty} @ {px} {order_type} {tif}`.

### Trading Loop (`run_autotrader`)

Each iteration of the main async loop:

1. **Load targets** — read latest `opt` or alpha-update file from `targets/` directory
2. **Update alphas** — merge short-term alpha columns into current target dataframe
3. **Report unfilled orders** — increase aggression on stale orders, remove orders > 60 min old
4. **Cancel all outstanding orders** — clean slate each wave
5. **Refresh positions** — from Binance API (live) or local file (paper/debug)
6. **Update prices** — mark positions with current mid prices from quote book
7. **EOD reversal overlay** — capture reference prices at 23:45, compute signal at 00:00 UTC
8. **Optimize and send orders** — run short-term CVXPY optimization, generate orders
9. **Between waves** — process OMS messages (fills/acks/cancels), paper-fill orders, sleep until next wave
10. **Report** — fill summaries, position snapshots, VWAP shortfall reports

Wave interval is configurable via `WAVE_INTERVAL_MINS` (typically 1 minute). Reoptimization interval via `REOPTIMIZE_INTERVAL_MINS`.

## Key Files

### trader.py (~1830 lines)
Main trading orchestrator.

**Key class:** `Trader`

**Constructor parameters:**
- `config` — production config dict
- `debug` / `paper_trade` / `no_trade` / `market_data_only` — operating modes
- `trade_on_startup` — trade immediately on first target (vs. waiting for next opt)
- `no_limits` — disable dollar limits (max aggression 6)
- `old_oms` — use legacy Python OMS instead of C++ OMS
- `eod` — enable EOD reversal overlay
- `slack_client` — Slack webhook for notifications

**Config parameters read:**
- `WAVE_INTERVAL_MINS`, `REOPTIMIZE_INTERVAL_MINS`, `ORDER_ACCELERATION`
- `MIN_TRADE_DOLLARS`, `MAX_ORDER_DOLLARS`, `MAX_TOTAL_ORDER_DOLLARS`
- `MAX_PORTFOLIO_NOTIONAL`, `MAX_TURNOVER`, `MAX_MOVE_FILTER`
- `MAX_DOLLARS_PER_OPT_TO_TRADE_MULT`, `MAX_TRADING_BIAS`
- `ST_FACTOR_SIGMAS`, `ST_KAPPA`, `SHORT_TERM_MODEL_HORIZONS`

**Key methods:**
- `run_autotrader()` — main async trading loop (see flow above)
- `optimize_and_send_orders()` — compute trades from targets, run ST optimizer, send orders
- `optimize(target_df, wave_notional)` — CVXPY short-term portfolio optimization with factor risk
- `create_order(symbol, aggression, ...)` — build `Order` with limit price from book, qty clipping, tick rounding
- `send_order(order)` — send to OMS (or paper-trade locally)
- `fill_order(fill)` — process fill, update positions, track slippage, write to fills file
- `process_oms_messages()` — handle NEW (ack), TRADE (fill), CANCELED messages from OMS listener
- `paper_fill_orders()` — simulate fills for paper trading when market crosses order price
- `get_limit_price(symbol, side, aggression)` — look up price from 10-level order book by aggression level
- `get_latest_targets(opt_only)` — read latest target CSV from `targets/` directory
- `cancel_all(symbols)` / `cancel_oid(oid)` — send cancel messages to OMS
- `refresh_positions(from_local_file)` — sync positions from Binance API or local file
- `update_df_positions_and_prices()` — mark target_df with live prices, compute unrealized PnL
- `update_alphas(df)` — merge fresh short-term alpha values into target_df
- `report_positions()` / `report_unfilled_orders()` / `report_current_wave_fills()` — Slack reporting
- `get_max_portfolio_size()` — calculate max notional from Binance balance * leverage

**Aggression model:**
- Range: MIN_AGGRESSION (-9) to max_aggression (6)
- Negative aggression = passive (posts in book at deeper levels)
- 0 = best bid/ask, 1 = mid, 2+ = crosses spread
- Per-symbol aggression starts at BASE_AGGRESSION (-9)
- Increased on unfilled orders, decreased on fills
- Aggression determines order type: negative → POST_ONLY/GTX, positive → LIMIT/GTC
- Reducing position → REDUCE or POST_ONLY_REDUCE variant

**Control files:**
- `notrade.txt` — symbols listed here are skipped during order generation
- `liquidate.txt` — symbols listed here are gradually liquidated over N hours. Format: `SYMBOL, hours`

**Liquidation logic:** `update_liquidate_target_positions()` overrides target_position with `init_pos * (1 - elapsed/hours)`.

### trading.py (~1135 lines)
Trading primitives and data classes.

**Enums:**
- `Side` (BUY/SELL) — with `from_string()`, `from_float()`, `sign()` converters
- `OrderType` (LIMIT, REDUCE, POST_ONLY, POST_ONLY_REDUCE, MARKET)
- `TimeInForce` (GTX/GTC/IOC/FOK)
- `PositionSide` (BOTH/LONG/SHORT) — for hedge mode support
- `FillType` (NORMAL/FAKE) — real vs paper-trade fills
- `Venue` (BINANCE)
- `OMSPayloadType` — C++ OMS message types (UM_CREATE_ORDER, UM_CANCEL_ORDER, etc.)

**`Fill` class:**
- Created from `Fill.from_binance_json(msg)` or directly
- `calc_commission_usd(price_dict)` — convert commission to USD
- `fill_slip()` — calculate slippage vs optimal price in dollars
- `notional(include_fees)` — signed notional value
- `__str__()` — pipe-delimited format for fills file persistence

**`Order` class:**
- Created via `Trader.create_order()` (live), `Order.from_str()` (manual), `Order.from_binance_json()` (exchange ack), or `Order.from_file_line()` (replay)
- `oms_create_order_msg(precision)` — legacy OMS format: `PLACE {koid} {side} {symbol} {qty} @ {px} {type} {tif}`
- `cpp_oms_create_order_msg(precision)` — C++ OMS JSON envelope with `UM_CREATE_ORDER` payload type
- `cpp_oms_cancel_order_msg(oid)` — cancel single order
- `cpp_oms_cancel_all_open_orders_msg(symbol)` — cancel all open orders for symbol
- `order_file_ln()` — pipe-delimited format for orders file, includes all alpha values at order time
- Tick size rounding: buys rounded up (ceil), sells rounded down (floor)
- `expanding()` — True if order increases position, determines REDUCE vs LIMIT
- `make_paper_order()` — simulate ack for paper trading
- `update_from_pending_order(pending)` — transfer metadata from pending to acked order
- `MODEL_HORIZONS` = [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]
- `MODELS` = ["hl", "c2vwap", "slz", "vadj", "ba", "badj", "oi", "rsi"]

**`Cancel` class:**
- Created from `Cancel.from_binance_json(msg)`
- `update_from_order(order)` — enrich with koid and remaining_qty
- `order_file_ln()` — pipe-delimited cancel record

**`get_agg_side_fills_info(symbol, fills, side)`** — aggregate fill stats (qty, avg_cost, notional, slippage_bps) for reporting

### binance_oms.py (~757 lines)
Python OMS for Binance futures — direct WebSocket connection to Binance user data stream.

**Key class:** `BinanceOMS`

**Functionality:**
- Connects to Binance user data WebSocket (`wss://fstream.binance.com/pm/ws`)
- Creates and refreshes listen keys via REST API (`/papi/v1/listenKey`)
- Receives ORDER_TRADE_UPDATE, ACCOUNT_UPDATE, listenKeyExpired events
- Publishes events via ZMQ PUB (port 5555) for the Trader to consume
- Credentials loaded from AWS Secrets Manager
- Auto-reconnect with exponential backoff

**Manual order mode (`--manual`):**
- `parse_order_string(order_str)` — parses "B ETHUSDT 0.01 @ 2500.0 LIMIT GTC" format
- `execute_manual_order(order_str)` — submit via Binance REST API with HMAC signature
- Safety limit: MAX_MANUAL_NOTIONAL = $10,000

**Modes:** `--debug` (stdout only), `--listen` (read-only raw traffic log), production (ZMQ publish)

### oms_listener.py (~367 lines)
OMS fill/status listeners — threaded consumers of ZMQ messages from the OMS.

**Three listener classes:**

1. **`OMSListener`** — legacy Python OMS listener
   - Subscribes via ZMQ SUB to `STATARB_OMS_IP:STATARB_OMS_PORT`
   - Receives `recv_pyobj()` messages
   - Parses ORDER pipe-delimited strings for OID mapping
   - Passes ORDER_TRADE_UPDATE events to Trader
   - Auto-reconnects if no messages for 30 minutes

2. **`CppOMSListener`** — C++ OMS user data listener
   - Subscribes via ZMQ SUB to `STATARB_CPP_OMS_IP:STATARB_CPP_OMS_USER_DATA_PORT`
   - Receives `recv_string()` JSON messages
   - Synthesizes NEW_ORDER_OID_MAPPING from ORDER_TRADE_UPDATE NEW events
   - Handles autoclose/ADL orders from exchange
   - Handles additional event types: ACCOUNT_UPDATE, ACCOUNT_CONFIG_UPDATE, liabilityChange, riskLevelChange, balanceUpdate, executionReport, listenKeyExpired

3. **`CppOMSRestReplyListener`** — C++ OMS REST reply listener
   - Subscribes via ZMQ SUB to `STATARB_CPP_OMS_IP:STATARB_CPP_OMS_REPLY_PORT`
   - Receives and logs REST API reply messages

All listeners run as daemon threads, poll with `zmq.NOBLOCK` or `socket.poll(1000)`, and support `get_msgs()` to drain the message queue.

### md_aggregator.py (~447 lines)
Real-time market data aggregation into 1-minute bars.

**Key classes:**

**`DataAggregator`** (Thread):
- Receives quote (depth10) and trade (aggTrade) data from `TradeQuotePoller`
- Maintains `current_book` dict — latest 10-level order book per symbol
- Aggregates into 1-minute bars: OHLC mid, spread, bid/ask sizes, VWAP, volume, trade count
- Tracks VWAP per symbol via `dollar_tracker`/`qty_tracker` (reset each opt period)
- Merges funding rates (queried 1 sec before bar close), open interest, and index price
- Writes bar parquet files to `live/{yyyymmdd}/{unix_ts}.parquet`
- Uploads raw trade/quote data and bar files to S3 (production only)
- Latency monitoring: alerts if quote latency exceeds 60ms for 5 consecutive checks (10-min intervals)

**`OpenInterestUpdater`** (Thread):
- Polls Binance open interest REST API every 30 minutes
- Delayed start by 5 minutes

### positions.py (~243 lines)
Position tracking and persistence.

**`SecPos`** — single security position:
- Tracks: qty, cost_basis, abs_dvolume, abs_qty, execution_qty, fill_cnt, fees, mark price
- `add_fill(fill)` — update position from Fill object
- `avg_px()` — average execution price (abs_dvolume / abs_qty)
- `refresh_qty(qty)` / `refresh_cost_basis(cost_basis)` — reconciliation with exchange data
- `reset_avg_cost()` — clear execution tracking (new opt period)

**`PositionRecorder`** — manages all positions:
- `dump_positions()` — writes position parquet to `positions/{yyyymmdd}/pos.{timestamp}.parquet`
- `add_fill(fill)` — routes fill to correct SecPos
- `get_qty(symbol)` — current position quantity

### trade_quote_poller.py (~181 lines)
WebSocket market data subscriber.

**Key class:** `TradeQuotePoller` (Thread)

**Functionality:**
- Connects to `wss://fstream.binance.com/ws`
- Subscribes to `{symbol}@depth10@100ms` (quotes) and `{symbol}@aggTrade` (trades)
- Routes messages to `DataAggregator.add_quote_data()` / `add_trade_data()`
- Subscribes in chunks of 100 symbols (Binance limit)
- Auto-resubscribes at UTC midnight for updated universe
- Loads symbols from Universe (tradeable filter) plus any currently-held positions
- `wait_for_book(wait_seconds)` — blocks until book populated for all symbols

### trader_eod.py (~342 lines)
End-of-day reversal execution overlay.

**Key class:** `EODReversalManager`

**Two-phase approach:**
1. **CAPTURE (23:45 UTC):** Snapshot reference mid prices + risk_1440 + risk_15 for all symbols
2. **ACTIVE (00:00-00:15 UTC):** Compute reversal signal, modulate execution speed and aggression

**Signal:** `-rank_cs(logret_15 / risk_15)` — cross-sectional rank of vol-adjusted 15-min return, negated for reversal

**Execution modifiers via `get_execution_modifier(symbol, trade_amt)`:**
- `speed_factor` (multiplier on trade_amt): `min_speed_mult` to `max_speed_mult`
- `aggression_boost` (added to aggression): `-max_agg_boost` to `+max_agg_boost`
- Only applied to symbols above `min_magnitude_pctile` in risk_1440 rank

**Config keys:** `EOD_REVERSAL_ENABLED`, `EOD_REVERSAL_CAPTURE_HOUR/MINUTE`, `EOD_REVERSAL_WINDOW_MINS`, `EOD_REVERSAL_MAX_AGG_BOOST`, `EOD_REVERSAL_MAX_SPEED_MULT`, `EOD_REVERSAL_MIN_SPEED_MULT`, `EOD_REVERSAL_MIN_MAGNITUDE_PCTILE`

### zmq_util.py (~79 lines)
ZeroMQ socket utilities.

**Key class:** `ZmqPubSub` — singleton managing async and blocking ZMQ contexts

**Functions:**
- `add_publisher(connect_url)` — returns ZMQ PUB socket
- `add_pusher(connect_url)` — returns ZMQ PUSH socket (for C++ OMS)
- `add_subscriber(connect_url, msg_filter)` — returns ZMQ SUB socket

## Key Data Flows

### Order Lifecycle
```
Trader.create_order() → Order (pending, koid assigned)
    → send_order() → OMS (via ZMQ PUB/PUSH)
    → pending_orders[koid] = order

OMS ACK → OMSListener:
    → NEW_ORDER_OID_MAPPING: oid_to_koid[oid] = koid
    → Order.from_binance_json(): acked_order
    → acked_order.update_from_pending_order(pending)
    → outstanding_orders[oid] = acked_order

Exchange Fill → OMSListener:
    → Fill.from_binance_json()
    → Trader.fill_order(fill): update position, write fills file, track slippage

Cancel → Trader.cancel_oid():
    → send CANCEL to OMS
    → OMS confirms → remove from outstanding_orders
```

### File Outputs
- **Orders:** `orders/{yyyymmdd}/orders.[paper.]{yyyymmdd}.csv` — pipe-delimited ORDER/CANCEL records with alpha values
- **Fills:** `fills/fills.[paper.]{yyyymmdd}.csv` — pipe-delimited FILL records
- **Positions:** `positions/{yyyymmdd}/pos.{timestamp}.parquet` — position snapshots every wave
- **Raw OMS:** `raw_oms/oms.{yyyymmdd}.txt` — raw OMS messages (production only)
- **Live bars:** `live/{yyyymmdd}/{unix_ts}.parquet` — 1-minute market data bars
- **Reports:** `REPORT_DIR/pnl/pnl.csv`, `wave/wave.csv`, `vwap_shortfall/shortfall.csv`

## Short-Term Optimization

The `optimize()` method runs a CVXPY optimization each wave:

- **Objective:** maximize `alpha_st @ w - kappa * (resid_risk + factor_risk)`
- **Constraints:** position bounds (`lbound`, `ubound`), max traded dollars (`wave_notional`)
- **Risk model:** diagonal residual covariance + factor model (using `ST_FACTOR_SIGMAS`)
- **Solver:** SCS
- **Kappa:** `ST_KAPPA` from config

The result `st_trade_amt` replaces the naive `trade_amt` for symbols where the optimizer has a non-zero trade. For symbols without optimizer trades, the system falls back to time-pacing (fraction of period elapsed * original trade amount).

## Common Patterns

- **KeyLogger** used throughout for rate-limited error logging (`logger.error(msg, key="unique_key")`)
- **`_message()`** sends to both log and Slack (with `@channel` ping for first occurrence of keyed messages)
- **Async main loop** with sync `time.sleep()` between waves (not fully async)
- **Paper trade mode** simulates fills locally when market price crosses order price
- **Debug mode** disables OMS publishing, file persistence, and most Slack messages
- **Symbol management:** loaded from Universe (tradeable filter) + any open positions
