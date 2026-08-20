# lib/external/ - External Data Sources and APIs

**Purpose:** Integration with external data sources and APIs (Binance, CoinGecko, news providers).

## Key Files

### binance_utils.py (500+ lines)
Binance API utilities

**Key Functions:**
- `get_exchange_info()` - Contract specifications
- `get_positions()` - Current position query
- `get_balances()` - Account balance query
- Order placement and cancellation wrappers

### download_binance.py (400+ lines)
Binance data downloader

**Key Components:**
- Downloads historical fills and positions
- Reconciliation utilities

### news_server.py (300+ lines)
Cryptocurrency news ingestion

**Key Components:**
- Fetches news articles from CryptoPanic API
- Deduplicates similar articles
- Saves to news/ directory

### coingecko.py
CoinGecko API integration

**Key Components:**
- Market cap data retrieval

### defi_llama.py
DeFi Llama data integration

## Key Functionality

- **Exchange Integration:** Binance futures API (REST + WebSocket)
- **News Data:** Real-time crypto news ingestion and processing
- **Market Data:** Historical and real-time price/volume data
- **Metadata:** Contract specs, market cap, DeFi metrics
