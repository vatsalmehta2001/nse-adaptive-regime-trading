# Data Pipeline Quick Reference

##  Quick Commands

```bash
# Setup pipeline (fetch 2 years of NIFTY 50)
python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2

# Update data (incremental)
make update-data

# Run tests
pytest tests/unit/test_data_pipeline.py -v

# Check data quality
python -c "from src.data_pipeline import DataPipeline; p = DataPipeline(); print(p.storage.get_database_stats())"
```

##  Code Examples

### Fetch and Store Data

```python
from src.data_pipeline import DataPipeline

# Initialize
pipeline = DataPipeline()
pipeline.initialize_database()

# Fetch NIFTY 50 (2 years)
nifty50 = pipeline.data_fetcher.get_nifty50_constituents()
stats = pipeline.fetch_and_store_historical(
    symbols=nifty50,
    start_date="2023-01-01",
    end_date="2024-12-31",
    validate=True
)

print(f"Fetched {stats['total_rows_stored']} rows")
```

### Query Data

```python
from src.data_pipeline import DataStorageManager

storage = DataStorageManager()

# Query single symbol
df = storage.query_ohlcv(
    symbols=["RELIANCE"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Query multiple symbols
df = storage.query_ohlcv(
    symbols=["RELIANCE", "TCS", "HDFCBANK"]
)

# Get latest date
latest = storage.get_latest_date("RELIANCE")
print(f"Latest data: {latest}")
```

### Validate Data

```python
from src.data_pipeline import MarketDataValidator

validator = MarketDataValidator()

# Validate and fix errors
validated_df, report = validator.validate_ohlcv(df, fix_errors=True)

print(f"Status: {report['data_quality']['status']}")
print(f"Valid rows: {report['valid_rows']}")

# Check missing dates
missing = validator.check_missing_dates(df, "RELIANCE")
print(f"Missing dates: {len(missing)}")

# Generate quality report
quality = validator.generate_quality_report(df)
```

### Generate Features

```python
from src.feature_engineering import TechnicalIndicators

indicators = TechnicalIndicators()

# Generate all features
df_features = indicators.generate_all_features(df)

# Or specific indicators
df = indicators.calculate_rsi(df, period=14)
df = indicators.calculate_macd(df)
df = indicators.calculate_bollinger_bands(df)

print(f"Generated {len(df.columns)} features")
```

### Incremental Updates

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Update all symbols
update_stats = pipeline.update_data()

# Update specific symbols
update_stats = pipeline.update_data(
    symbols=["RELIANCE", "TCS"],
    lookback_days=7
)

print(f"Updated {update_stats['symbols_updated']} symbols")
print(f"Added {update_stats['new_rows']} new rows")
```

### Health Check

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

health = pipeline.health_check()

print(f"Status: {health['status']}")
print(f"Symbols: {health['database']['unique_symbols']}")
print(f"Data age: {health['database'].get('data_age_days', 0)} days")
```

##  Configuration

### Environment Variables

```bash
# .env
DUCKDB_PATH=data/trading_db.duckdb
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/trading.log
```

### Config File

```yaml
# config/data_sources.yaml
openbb:
  fetch_config:
    max_retries: 3
    retry_delay: 5
    requests_per_minute: 60
    cache_ttl: 3600
```

##  Database Queries

```python
from src.data_pipeline import DataStorageManager

storage = DataStorageManager()

# Get all symbols
symbols = storage.get_available_symbols()

# Get data coverage
coverage = storage.get_data_coverage("RELIANCE")

# Get quality report
quality_df = storage.get_data_quality_report()

# Get database stats
stats = storage.get_database_stats()
print(f"DB Size: {stats['db_size_mb']:.2f} MB")
print(f"Symbols: {stats['unique_symbols']}")
print(f"Total rows: {stats['ohlcv_count']}")
```

##  Testing

```bash
# Run all data pipeline tests
pytest tests/unit/test_data_pipeline.py -v

# Run specific test
pytest tests/unit/test_data_pipeline.py::TestOpenBBDataFetcher -v

# Run with coverage
pytest tests/unit/test_data_pipeline.py --cov=src.data_pipeline
```

##  Troubleshooting

### Check Logs

```python
# View latest logs
tail -f logs/trading.log

# Check for errors
grep "ERROR" logs/trading.log
```

### Clear Cache

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.clear_cache()
```

### Rebuild Database

```bash
# Delete and recreate
rm data/trading_db.duckdb
python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2
```

### Check Data Quality

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Get quality report
quality = pipeline.storage.get_data_quality_report()
print(quality)

# Check specific symbol
coverage = pipeline.storage.get_data_coverage("RELIANCE")
print(coverage)
```

##  Performance Tips

1. **Batch Fetching**: Use `batch_size=10` for optimal performance
2. **Parallel Workers**: Set `max_workers=3` to avoid rate limits
3. **Caching**: Enable caching to reduce API calls
4. **Incremental Updates**: Use `update_data()` instead of full refetch
5. **Database Indexes**: Created automatically, no action needed

##  Common Tasks

### Task 1: Setup Fresh Database

```bash
python scripts/setup_data_pipeline.py --symbols NIFTY50 --years 2
```

### Task 2: Daily Update

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.update_data()
```

### Task 3: Backfill Missing Data

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
backfilled = pipeline.backfill_missing_dates("RELIANCE")
print(f"Backfilled {backfilled} dates")
```

### Task 4: Export to CSV

```python
from src.data_pipeline import DataStorageManager

storage = DataStorageManager()
df = storage.query_ohlcv(symbols=["RELIANCE"])
df.to_csv("reliance_data.csv", index=False)
```

### Task 5: Generate Features for ML

```python
from src.data_pipeline import DataStorageManager
from src.feature_engineering import TechnicalIndicators

storage = DataStorageManager()
indicators = TechnicalIndicators()

# Get data
df = storage.query_ohlcv(symbols=["RELIANCE", "TCS"])

# Generate all features
df_ml = indicators.generate_all_features(df)

# Save for ML training
df_ml.to_parquet("ml_features.parquet")
```

##  More Information

- **Full Documentation**: `DATA_PIPELINE_README.md`
- **API Reference**: `docs/api_reference.md`
- **Examples**: `notebooks/exploratory/`
- **Tests**: `tests/unit/test_data_pipeline.py`

---

**Quick Help**: For any issues, check `logs/trading.log` first!

