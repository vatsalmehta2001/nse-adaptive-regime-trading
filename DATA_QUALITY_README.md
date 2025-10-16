# Data Quality Management - Complete Guide

Production-grade data quality framework for NSE market data validation, cleaning, and auditing.

---

## Overview

This framework provides:
- **Automated Data Validation**: Schema validation and business rule checks
- **Quality Scoring**: 0-100 scale with actionable grades
- **Audit Trails**: Complete tracking of all filtering decisions
- **Comprehensive Reporting**: JSON, CSV, and human-readable summaries
- **Production Thresholds**: Based on NSE circuit breakers and market analysis

---

## Quick Start

### 1. Import Classes

```python
from src.data_pipeline import (
    DataQualityConfig,
    MarketDataValidator,
    DataQualityReporter
)
```

### 2. Clean Data with Audit Trail

```python
# Initialize validator
validator = MarketDataValidator()

# Clean with audit trail
clean_df, audit = validator.clean_returns_with_audit(
    df,
    threshold=DataQualityConfig.MAX_DAILY_RETURN,  # 20%
    log_details=True
)

print(f"Retention Rate: {audit['retention_rate']*100:.1f}%")
print(f"Filtered: {audit['total_filtered']} rows")
```

### 3. Generate Quality Report

```python
# Initialize reporter
reporter = DataQualityReporter()

# Generate report
report = reporter.generate_quality_report(
    clean_df,
    audit,
    symbol="RELIANCE"
)

# Print summary
reporter.print_quality_summary(report)

print(f"Quality Score: {report['quality_score']:.1f}/100")
print(f"Grade: {report['quality_grade']}")
```

---

## Data Quality Configuration

### Threshold Philosophy

**Why 20% for daily returns?**
- NSE circuit breaker limits are 10%/20% depending on stock
- 20% threshold captures extreme moves while avoiding false positives
- Based on analysis of historical NSE data

**Previous Issue:**
-  Old code used 1000% threshold (10x return!)
-  Allowed unrealistic data through
-  New code uses 20% (production-grade)

### Configuration File

Edit `config/data_quality_config.yaml`:

```yaml
thresholds:
  returns:
    daily: 0.20      # 20% NSE circuit breaker
    weekly: 0.50     # 50% weekly
    monthly: 1.00    # 100% monthly

  prices:
    min_price: 1.0              # Minimum INR price
    max_intraday_jump: 0.30     # 30% intraday move

  completeness:
    min_data_points: 60              # 60 days minimum
    max_missing_ratio: 0.10          # Max 10% missing

quality_score:
  excellent: 90    # >= 90: Production ready
  good: 75         # >= 75: Minor issues
  acceptable: 60   # >= 60: Review required
  poor: 0          # < 60: Fix before use
```

---

## Quality Score Interpretation

### Scoring System (0-100)

**Components:**
- **Retention Rate** (20 points): How much data kept after filtering
- **Completeness** (30 points): Missing data percentage
- **Data Issues** (50 points): Severity and count of issues

**Calculation:**
```
Score = 100
  - (1 - retention_rate) × 20
  - (1 - completeness) × 30
  - Σ(issue_severity_penalty)

Where issue penalties:
  - Low severity: -5 points
  - Medium severity: -10 points
  - High severity: -20 points
```

### Quality Grades

| Score | Grade | Meaning | Action |
|-------|-------|---------|--------|
| 90-100 | **EXCELLENT** | Production ready |  Use for training & backtesting |
| 75-89 | **GOOD** | Minor issues, usable |  Safe to use, monitor issues |
| 60-74 | **ACCEPTABLE** | Review carefully |  Review issues before use |
| 0-59 | **POOR** | Fix data issues |  Do not use, fix issues first |

---

## Audit Trail Structure

Every cleaning operation produces a detailed audit trail:

```python
audit = {
    'initial_rows': 1000,
    'threshold_used': 0.20,
    'final_rows': 950,
    'total_filtered': 50,
    'retention_rate': 0.95,

    'filtered_by_symbol': {
        'RELIANCE': {
            'count': 50,
            'dates': ['2024-03-20', '2024-06-15', ...],
            'returns': [0.25, -0.22, ...],
            'max_return': 0.30,
            'min_return': -0.25
        }
    },

    'extreme_dates': [
        {
            'symbol': 'RELIANCE',
            'date': '2024-03-20',
            'return': 0.25,
            'close': 2850.50
        },
        ...
    ]
}
```

### What Gets Logged

 Every filtered data point
 Reason for filtering
 Symbol and date affected
 Actual value that exceeded threshold
 Statistics (max, min, count)

---

## Common Quality Issues

### 1. Extreme Returns (High Priority)

**Example:**
```
Symbol RELIANCE: Filtering 3 extreme returns (>20%)
  2024-03-20: +25.50% return
  2024-06-15: -22.80% return
  2024-09-10: +21.20% return
```

**Causes:**
- Stock splits or bonuses (not adjusted)
- Data errors from provider
- Corporate actions (mergers, demergers)
- Flash crashes

**Solution:**
```python
# Review filtered dates
for date_info in audit['extreme_dates']:
    print(f"{date_info['date']}: {date_info['return']*100:.1f}%")

# Check for corporate actions
# Adjust threshold if legitimate event
```

### 2. Low Retention Rate (<90%)

**Example:**
```
Quality Score: 65/100 (ACCEPTABLE)
Retention Rate: 82.5%
```

**Causes:**
- Too many extreme moves (volatile stock)
- Data quality issues
- Threshold too strict

**Solution:**
```python
# Check which symbols affected
print(audit['filtered_by_symbol'])

# Consider:
# 1. Using higher threshold for small caps
# 2. Checking data source quality
# 3. Corporate action adjustment
```

### 3. Missing Data (Medium Priority)

**Example:**
```
Data Issues: missing_values (count: 45, severity: medium)
Completeness: 92.3%
```

**Causes:**
- Trading halts
- Delisting periods
- Data provider gaps

**Solution:**
```python
# Identify missing dates
missing_dates = pd.date_range(
    start=df['date'].min(),
    end=df['date'].max()
).difference(df['date'])

print(f"Missing dates: {missing_dates}")

# Options:
# 1. Forward-fill for short gaps (<3 days)
# 2. Exclude symbol if too much missing data
# 3. Contact data provider
```

### 4. Zero Volume Days (Low Priority)

**Example:**
```
Data Issues: zero_volume (count: 15, severity: low)
```

**Causes:**
- Holidays (data error)
- Low liquidity stocks
- Circuit filter days

**Solution:**
- Usually safe to ignore
- Flag for monitoring
- Don't remove (valid market condition)

---

## Usage in Production

### In Training Pipeline

```python
from src.data_pipeline import DataQualityConfig, MarketDataValidator, DataQualityReporter

# Load data
factors = store.get_factors(symbols)

# Validate quality
validator = MarketDataValidator()
reporter = DataQualityReporter()

# Clean with audit
clean_factors, audit = validator.clean_returns_with_audit(
    factors,
    threshold=DataQualityConfig.MAX_DAILY_RETURN
)

# Generate report
quality_report = reporter.generate_quality_report(
    clean_factors,
    audit,
    symbol="PORTFOLIO"
)

# Check quality score
if quality_report['quality_score'] < 60:
    logger.error(" Poor data quality! Fix issues before training.")
    reporter.print_quality_summary(quality_report)
    sys.exit(1)
elif quality_report['quality_score'] < 75:
    logger.warning("  Acceptable quality. Review issues before training.")
    reporter.print_quality_summary(quality_report)
else:
    logger.info(f" Good quality ({quality_report['quality_score']:.1f}/100)")

# Proceed with training...
```

### In Backtesting Pipeline

```python
# Validate data before backtesting
clean_data, audit = validator.clean_returns_with_audit(prices)
report = reporter.generate_quality_report(clean_data, audit)

if report['quality_score'] < 60:
    raise ValueError("Data quality too poor for reliable backtest")
```

---

## Configuration Guide

### Customize Thresholds

Edit `config/data_quality_config.yaml`:

```yaml
thresholds:
  returns:
    daily: 0.15  # Stricter (15% instead of 20%)
    # Or
    daily: 0.25  # More lenient (25%)
```

### Different Thresholds by Market Cap

```python
# In code
if symbol in large_caps:
    threshold = 0.15  # Stricter for large caps
elif symbol in small_caps:
    threshold = 0.25  # More lenient for small caps
else:
    threshold = DataQualityConfig.MAX_DAILY_RETURN  # Default 20%
```

---

## Quality Report Examples

### Example 1: Excellent Quality

```
================================================================================
DATA QUALITY REPORT: RELIANCE
================================================================================

Quality Score: 95.5/100 (EXCELLENT)

DATA STATISTICS:
  Total Rows:      494
  Filtered Rows:   2
  Retention Rate:  99.6%
  Completeness:    98.5%

DATE RANGE:
  Start: 2023-10-16
  End:   2025-10-15
  Days:  730

EXTREME VALUES FILTERED:
  RELIANCE: 2 dates
    Max: +22.5%, Min: -21.8%

================================================================================
```

**Analysis:** Excellent quality. Only 2 extreme dates filtered (likely valid events). Safe to use.

### Example 2: Poor Quality

```
================================================================================
DATA QUALITY REPORT: VOLATILE_STOCK
================================================================================

Quality Score: 45.0/100 (POOR)

DATA STATISTICS:
  Total Rows:      300
  Filtered Rows:   150
  Retention Rate:  50.0%
  Completeness:    75.0%

EXTREME VALUES FILTERED:
  VOLATILE_STOCK: 150 dates
    Max: +85.2%, Min: -78.5%

DATA ISSUES (3):
   [HIGH] volume_drop: Recent volume is 8.5% of historical
    [MEDIUM] constant_price: 45 days with no price change
    [LOW] zero_volume: 12 days with zero volume

================================================================================
```

**Analysis:** Poor quality. 50% of data filtered. Multiple serious issues. Do not use without investigation.

---

## Comparison Reports

### Generate Cross-Symbol Comparison

```python
# Generate reports for multiple symbols
reports = []
for symbol in ['RELIANCE', 'TCS', 'INFY', 'ICICIBANK']:
    clean_df, audit = validator.clean_returns_with_audit(df[df['symbol']==symbol])
    report = reporter.generate_quality_report(clean_df, audit, symbol=symbol)
    reports.append(report)

# Generate comparison
comparison = reporter.generate_comparison_report(reports)
print(comparison)
```

**Output:**
```
   symbol  quality_score quality_grade  retention_rate  completeness  issues
0  RELIANCE      95.5    EXCELLENT          0.996          0.985         1
1  TCS           88.0    GOOD               0.945          0.980         2
2  INFY          82.5    GOOD               0.920          0.975         2
3  ICICIBANK     68.0    ACCEPTABLE         0.850          0.920         4
```

---

## Integration with Existing Code

### Backward Compatible

All new features are backward compatible:

```python
# Old code still works
validator = MarketDataValidator()
validated_df, report = validator.validate_ohlcv(df)

# New enhanced features available
clean_df, audit = validator.clean_returns_with_audit(df)
quality_report = reporter.generate_quality_report(clean_df, audit)
```

### Enhanced Scripts

Training and backtesting scripts now include automatic quality checking:

```bash
# Training with quality checks
python scripts/train_models.py --symbols NIFTY50 --check-quality

# Quality report automatically generated to:
# reports/data_quality/quality_report_PORTFOLIO_20241015_120000.json
```

---

## Troubleshooting

### High Filtering Rate (>10%)

**Problem:** Retention rate < 90%

**Diagnosis:**
```python
# Check which dates were filtered
for symbol, details in audit['filtered_by_symbol'].items():
    print(f"{symbol}: {details['count']} dates filtered")
    for date, ret in zip(details['dates'], details['returns']):
        print(f"  {date}: {ret*100:+.1f}%")
```

**Solutions:**
1. Check for corporate actions (splits, bonuses)
2. Verify data source quality
3. Consider higher threshold for volatile stocks
4. Manual review of filtered dates

### Low Quality Score (<75)

**Problem:** Quality score below GOOD threshold

**Diagnosis:**
```python
# Review quality report
reporter.print_quality_summary(report)

# Check specific issues
for issue in report['data_issues']:
    print(f"{issue['severity']}: {issue['description']}")
```

**Solutions:**
1. Fix data source issues
2. Add corporate action adjustments
3. Improve data completeness
4. Review and address each flagged issue

### Different Results After Filtering

**Problem:** Model performance changes after quality filtering

**Expected:** This is normal. Filtering improves data quality and should improve model reliability (though not necessarily returns).

**What to Check:**
- Ensure retention rate > 90%
- Check filtered dates aren't critical events
- Verify thresholds appropriate for stock volatility

---

## Best Practices

### 1. Always Use Audit Trails

```python
#  Don't do this
df_clean = df[df['returns'].abs() < 0.20]

#  Do this
clean_df, audit = validator.clean_returns_with_audit(df, threshold=0.20)
reporter.generate_quality_report(clean_df, audit)
```

### 2. Review Quality Scores Before Training

```python
if quality_score < 60:
    raise ValueError("Data quality too poor for training")
elif quality_score < 75:
    logger.warning("Quality acceptable but review recommended")
    # Continue with caution
```

### 3. Save Quality Reports

```python
# Reports automatically saved to:
# reports/data_quality/quality_report_SYMBOL_TIMESTAMP.json
# reports/data_quality/quality_summary_SYMBOL_TIMESTAMP.csv

# Always review before production deployment
```

### 4. Use Appropriate Thresholds

```python
# For large caps (RELIANCE, TCS, etc.)
threshold = 0.15  # Stricter (15%)

# For mid caps
threshold = 0.20  # Standard (20%)

# For small caps
threshold = 0.25  # More lenient (25%)
```

---

## API Reference

### DataQualityConfig

```python
class DataQualityConfig:
    MAX_DAILY_RETURN = 0.20      # 20% threshold
    MAX_WEEKLY_RETURN = 0.50     # 50% threshold
    MAX_MONTHLY_RETURN = 1.00    # 100% threshold
    MIN_PRICE = 1.0              # Minimum price
    MIN_DATA_POINTS = 60         # Minimum days

    @classmethod
    def get_threshold(cls, period: str) -> float:
        """Get threshold for time period."""
```

### MarketDataValidator

```python
def clean_returns_with_audit(
    self,
    df: pd.DataFrame,
    threshold: float = 0.20,
    log_details: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter extreme returns with audit trail.

    Returns:
        (cleaned_df, audit_report)
    """
```

### DataQualityReporter

```python
def generate_quality_report(
    self,
    df: pd.DataFrame,
    audit_trail: Dict,
    symbol: str = "ALL"
) -> Dict:
    """
    Generate comprehensive quality report.

    Returns quality score (0-100) and detailed metrics.
    """

def generate_comparison_report(
    self,
    reports: List[Dict]
) -> pd.DataFrame:
    """Compare quality across multiple symbols."""

def print_quality_summary(self, report: Dict):
    """Print human-readable summary."""
```

---

## Examples

### Example 1: Single Symbol Quality Check

```python
from src.data_pipeline import DataQualityConfig, MarketDataValidator, DataQualityReporter

# Load data for one symbol
df = pd.read_csv('data/RELIANCE_ohlcv.csv')

# Validate and clean
validator = MarketDataValidator()
clean_df, audit = validator.clean_returns_with_audit(
    df,
    threshold=DataQualityConfig.MAX_DAILY_RETURN
)

# Generate report
reporter = DataQualityReporter()
report = reporter.generate_quality_report(clean_df, audit, symbol='RELIANCE')

# Check score
if report['quality_score'] >= 75:
    print(" Quality is GOOD or better")
else:
    print("  Quality needs attention")
    reporter.print_quality_summary(report)
```

### Example 2: Portfolio Quality Check

```python
# Check quality for entire portfolio
symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
reports = []

for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol]
    clean_df, audit = validator.clean_returns_with_audit(symbol_df)
    report = reporter.generate_quality_report(clean_df, audit, symbol=symbol)
    reports.append(report)

# Generate comparison
comparison = reporter.generate_comparison_report(reports)

# Flag poor quality symbols
poor_quality = comparison[comparison['quality_score'] < 75]
if len(poor_quality) > 0:
    print(f"  {len(poor_quality)} symbols have quality < 75:")
    print(poor_quality[['symbol', 'quality_score', 'quality_grade']])
```

### Example 3: Automated Quality Gate

```python
def training_quality_gate(factors: pd.DataFrame) -> bool:
    """Quality gate for model training."""
    validator = MarketDataValidator()
    reporter = DataQualityReporter()

    clean_factors, audit = validator.clean_returns_with_audit(factors)
    report = reporter.generate_quality_report(clean_factors, audit)

    # Production requirements
    if report['quality_score'] < 75:
        logger.error(f" Quality too low: {report['quality_score']:.1f}/100")
        return False

    if audit['retention_rate'] < 0.90:
        logger.error(f" Too much data filtered: {audit['retention_rate']*100:.1f}%")
        return False

    logger.info(f" Quality gate passed: {report['quality_score']:.1f}/100")
    return True

# Use in pipeline
if not training_quality_gate(factors):
    sys.exit(1)
```

---

## Regulatory Compliance

### Audit Trail Requirements

For regulatory compliance, all filtering decisions must be:
- **Logged**: Complete record of what was filtered and why
- **Traceable**: Can reconstruct filtering decisions
- **Justified**: Clear threshold and rationale
- **Reviewable**: Human-readable reports

This framework provides:
 Complete audit trails
 JSON and CSV exports for archiving
 Timestamp tracking
 Configurable thresholds with documentation

---

## Testing

Run quality tests:

```bash
pytest tests/unit/test_data_quality.py -v
```

Expected results:
```
test_threshold_values PASSED
test_clean_returns_with_audit PASSED
test_audit_trail_structure PASSED
test_quality_score_calculation PASSED
test_quality_grades PASSED
test_end_to_end_quality_pipeline PASSED
```

---

## Summary

 **Production-Grade Thresholds**: 20% daily (not 1000%)
 **Complete Audit Trails**: Every filtering decision logged
 **Quality Scoring**: 0-100 scale with actionable grades
 **Comprehensive Reports**: JSON, CSV, human-readable
 **Backward Compatible**: No breaking changes
 **Well Tested**: Comprehensive test suite

**The data quality framework is production-ready and regulatory-compliant.**

