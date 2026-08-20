# lib/util/ - Shared Utilities

**Purpose:** Shared utility modules used across the entire system. Provides common functionality for configuration, data manipulation, time handling, and infrastructure.

## Key Files

### config.py (400+ lines)
Configuration management

**Key Functions:**
- `get_config()` - Load config.json with validation
- `extract_horizons()`, `extract_models()` - Extract model configurations
- `extract_reopt_times()` - Reoptimization schedule parsing

### dataframes.py (800+ lines)
DataFrame utilities

**Key Functions:**
- `merge_on_index()` - Safe multi-index merging
- `concat()` - Fast DataFrame concatenation
- `check_df()` - DataFrame validation
- `get_min_max_ts()` - Timestamp range extraction
- `safe_del()` - Safe column deletion
- `shrink_floats()` - Memory optimization (float64 → float32)

### time_util.py (500+ lines)
Time handling utilities

**Key Functions:**
- `date_to_start_dt()`, `date_to_end_dt()` - Date conversions
- `compute_lookback_days()` - Lookback period calculation
- `date_range()` - Iterable date ranges
- `to_datetime()` - Robust datetime parsing

### directory.py (130 lines)
Directory path management

**Key Components:**
- Key class: `DirectoryManager` - Singleton for all paths
- Defines ROOT_DIR, DATA_DIR, TRADING_DIR, REPORT_DIR, etc.
- All data and trading paths managed centrally

### util.py (600+ lines)
General utilities

**Key Functions:**
- `unique_list()` - Remove duplicates preserving order
- `log_and_raise()` - Error logging with context
- `fpct()`, `fmoney()` - Number formatting
- `log_mem_usage()` - Memory profiling
- Constants: SYMBOL_PAIR, SYMBOL_VENUE, TARDIS_EXCHANGE

### logging_util.py
Enhanced logging

**Key Components:**
- Key class: `KeyLogger` - Adds key-value pair logging
- Structured logging for better searchability

### slack.py
Slack notification utilities

**Key Functions:**
- `send_slack_async()` - Async message sending

### aws.py (400+ lines)
AWS S3 integration

**Key Functions:**
- S3 file upload/download
- Parquet file operations on S3

### files.py
File system utilities

**Key Functions:**
- `safe_mkdir()` - Create directories safely
- File path manipulation

## Key Functionality

- **Configuration:** Single source of truth (config.json) with validation
- **Data Manipulation:** DataFrame operations, merging, validation
- **Time Handling:** Date/datetime conversions, ranges, UTC timezone management
- **Path Management:** Centralized directory structure via DirectoryManager
- **Infrastructure:** AWS S3, Slack, logging, memory profiling
- **Error Handling:** Consistent error logging with context

## Common Patterns

- **Config-Driven:** All modules accept `config` dict from config.json
- **DirectoryManager:** All paths via singleton `dir_manager`
- **DataLoader:** Unified data access via `DataLoader` class
- **Calcs:** Feature calculations via `Calcs` class
- **Logging:** Structured logging with `KeyLogger`
- **Error Handling:** `log_and_raise()` for context-rich errors
