# Data Pipeline Documentation

## Overview

The NSE Data Pipeline is a production-grade system for fetching, validating, and storing market data from NSE (National Stock Exchange of India) using OpenBB Platform v4 and DuckDB.

## Architecture

```

  OpenBB Client    Fetches data from Yahoo Finance/NSE




   Validator       Validates data quality using Pandera




  DuckDB Storage   Stores in optimized OLAP database




Technical Indicators  Generates features (optional)

```

## Components

### 1. OpenBB Data Fetcher (`openbb_client.py`)

**Features**:
- OpenBB Platform v4 integration
- NSE symbol format handling (.NS suffix)
- Rate limiting (60 requests/minute default)
- Exponential backoff retry logic
- Response caching with TTL
- Batch fetching support

**Usage**:
```python
from src.data_pipeline import OpenBBDataFetcher

fetcher = OpenBBDataFetcher()

# Fetch single symbol
df = fetcher.fetch_equity_ohlcv(
    symbols=["RELIANCE"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Fetch NIFTY 50 constituents
symbols = fetcher.get_nifty50_constituents()
```

### 2. Data Storage Manager (`data_storage.py`)

**Features**:
- DuckDB for high-performance OLAP
- Optimized schema for time-series data
- Automatic deduplication (upsert)
- Metadata tracking
- Data quality logging

**Database Schema**:
```sql
-- Main OHLCV table
ohlcv (
    symbol VARCHAR,
    date DATE,
    open/high/low/close DECIMAL(18,4),
    volume BIGINT,
    PRIMARY KEY (symbol, date)
)

-- Market metadata
market_metadata (
    symbol VARCHAR PRIMARY KEY,
    first_data_date DATE,
    last_data_date DATE,
    total_records INTEGER
)
```

**Usage**:
```python
from src.data_pipeline import DataStorageManager

storage = DataStorageManager()
storage.create_schema()

# Insert data
storage.insert_ohlcv(df)

# Query data
data = storage.query_ohlcv(
    symbols=["RELIANCE", "TCS"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### 3. Data Validator (`data_validator.py`)

**Features**:
- Pandera schema validation
- OHLC relationship checks
- Outlier detection (IQR, Z-score)
- Missing date identification
- Corporate action adjustments
- Comprehensive quality reports

**Usage**:
```python
from src.data_pipeline import MarketDataValidator

validator = MarketDataValidator()

# Validate data
validated_df, report = validator.validate_ohlcv(df, fix_errors=True)

# Check missing dates
missing = validator.check_missing_dates(df, "RELIANCE")

# Generate quality report
quality_report = validator.generate_quality_report(df)
```

### 4. Technical Indicators (`technical_indicators.py`)

**Features**:
- Returns (simple and log)
- Moving averages (SMA, EMA)
- Volatility (historical, Parkinson)
- RSI, MACD, Bollinger Bands
- ATR, Stochastic, Volume indicators
- All vectorized for performance

**Usage**:
```python
from src.feature_engineering import TechnicalIndicators

indicators = TechnicalIndicators()

# Generate all features
df_with_features = indicators.generate_all_features(df)

# Or generate specific indicators
df = indicators.calculate_rsi(df, period=14)
df = indicators.calculate_macd(df)
```

### 5. Pipeline Orchestrator (`pipeline.py`)

**Features**:
- Coordinates all components
- Parallel processing (multithreading)
- Incremental updates
- Backfill missing data
- Progress tracking with tqdm
- Comprehensive error handling

**Usage**:
```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Initialize database
pipeline.initialize_database()

# Fetch historical data
stats = pipeline.fetch_and_store_historical(
    symbols=["RELIANCE", "TCS", "HDFCBANK"],
    start_date="2023-01-01",
    end_date="2024-12-31",
    batch_size=10,
    max_workers=3
)

# Incremental update
update_stats = pipeline.update_data()

# Health check
health = pipeline.health_check()
```

## Quick Start

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Initialize and fetch data
python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2
```

### Command-Line Usage

```bash
# Fetch NIFTY 50 data (2 years)
python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2

# Fetch specific symbols
python scripts/setup_data_pipeline.py --symbols "RELIANCE,TCS,HDFCBANK" --years 1

# Fetch with date range
python scripts/setup_data_pipeline.py \
    --symbols NIFTY50 \
    --start-date 2022-01-01 \
    --end-date 2024-12-31
```

### Programmatic Usage

```python
from src.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()
pipeline.initialize_database()

# Fetch data
pipeline.fetch_and_store_historical(
    symbols=pipeline.data_fetcher.get_nifty50_constituents(),
    start_date="2023-01-01",
    end_date="2024-12-31"
)

# Query data
df = pipeline.storage.query_ohlcv(
    symbols=["RELIANCE"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

## Configuration

### Environment Variables (.env)

```bash
# Database
DUCKDB_PATH=data/trading_db.duckdb

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/trading.log
```

### Configuration File (config/data_sources.yaml)

```yaml
openbb:
  fetch_config:
    max_retries: 3
    retry_delay: 5
    timeout: 30
    cache_ttl: 3600
    requests_per_minute: 60

duckdb:
  path: "data/trading_db.duckdb"
  connection_pool_size: 5

data_quality:
  outlier_threshold: 5
  min_volume: 1000
  max_price_gap_pct: 20
```

## Data Quality

### Validation Checks

1. **Schema Validation**: Type checking, null handling
2. **OHLC Relationships**: High >= Open/Close, Low <= Open/Close
3. **Outlier Detection**: IQR and Z-score methods
4. **Missing Dates**: Compare against trading calendar
5. **Duplicate Detection**: Symbol + date uniqueness
6. **Zero Volume**: Flag days with no trading
7. **Extreme Movements**: Price changes > 20%

### Quality Reports

```python
# Generate quality report
report = pipeline.storage.get_data_quality_report()

# Check data coverage
coverage = pipeline.storage.get_data_coverage("RELIANCE")

# Get database statistics
stats = pipeline.storage.get_database_stats()
```

## Performance

### Benchmarks

- **Fetch Speed**: ~10-15 symbols/minute (with rate limiting)
- **Storage**: ~1000 rows/second to DuckDB
- **Query**: Sub-second for typical queries
- **Validation**: ~10,000 rows/second

### Optimization Tips

1. **Batch Fetching**: Process 10-15 symbols per batch
2. **Parallel Workers**: Use 2-4 workers for parallel fetching
3. **Caching**: Enable caching to avoid redundant API calls
4. **Indexing**: Database indexes are created automatically
5. **Incremental Updates**: Only fetch new data daily

## Troubleshooting

### Common Issues

**1. OpenBB Import Error**
```bash
# Install OpenBB
pip install openbb==4.2.4
```

**2. No Data Returned**
```python
# Check symbol format (must include .NS for NSE)
symbol = "RELIANCE.NS"

# Verify date range
# Use recent dates (within last 5 years)
```

**3. Rate Limiting**
```python
# Adjust requests per minute in config
openbb:
  fetch_config:
    requests_per_minute: 30  # Reduce if hitting limits
```

**4. Database Locked**
```python
# Close existing connections
pipeline.storage.close()

# Or delete and recreate database
rm data/trading_db.duckdb
```

## Testing

```bash
# Run all pipeline tests
pytest tests/unit/test_data_pipeline.py -v

# Run specific test class
pytest tests/unit/test_data_pipeline.py::TestOpenBBDataFetcher -v

# Run with coverage
pytest tests/unit/test_data_pipeline.py --cov=src.data_pipeline
```

## Best Practices

1. **Always Validate**: Run validation on fetched data
2. **Incremental Updates**: Use `update_data()` for daily refreshes
3. **Monitor Quality**: Review quality reports regularly
4. **Backfill Carefully**: Check for missing dates periodically
5. **Cache Wisely**: Clear cache when data issues occur
6. **Rate Limit**: Respect API rate limits to avoid bans
7. **Error Handling**: Always catch and log exceptions
8. **Database Maintenance**: Run VACUUM periodically

## API Reference

See [`docs/api_reference.md`](../docs/api_reference.md) for detailed API documentation.

## Examples

See [`notebooks/exploratory/01_data_exploration.ipynb`](../notebooks/exploratory/01_data_exploration.ipynb) for usage examples.

## Support

For issues or questions:
- Check logs: `logs/trading.log`
- Run health check: `pipeline.health_check()`
- GitHub Issues: [link]

---

**Last Updated**: 2025-01-14

