"""
Data Validator for Market Data.

Comprehensive validation and cleaning using Pandera schema validation
and custom business logic for financial data quality.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.utils.logging_config import get_logger
from src.utils.market_calendar import NSEMarketCalendar

logger = get_logger(__name__)


class DataQualityConfig:
    """
    Configuration for data quality thresholds.

    Production-grade thresholds based on market analysis and risk management.
    """

    # Return thresholds (realistic for Indian markets)
    MAX_DAILY_RETURN = 0.20  # 20% circuit breaker limit
    MAX_WEEKLY_RETURN = 0.50  # 50%
    MAX_MONTHLY_RETURN = 1.00  # 100%

    # Price thresholds
    MIN_PRICE = 1.0  # Minimum valid price (INR)
    MAX_INTRADAY_JUMP = 0.30  # 30% intraday move

    # Volume thresholds
    MIN_VOLUME = 0  # Allow zero volume (mark separately)
    MIN_AVG_VOLUME = 10000  # Minimum average volume

    # Data completeness
    MIN_DATA_POINTS = 60  # Minimum days for factor calculation
    MAX_MISSING_DAYS_RATIO = 0.10  # Max 10% missing days

    @classmethod
    def get_threshold(cls, period: str = 'daily') -> float:
        """
        Get appropriate threshold for time period.

        Args:
            period: 'daily', 'weekly', or 'monthly'

        Returns:
            Threshold value
        """
        thresholds = {
            'daily': cls.MAX_DAILY_RETURN,
            'weekly': cls.MAX_WEEKLY_RETURN,
            'monthly': cls.MAX_MONTHLY_RETURN
        }
        return thresholds.get(period, cls.MAX_DAILY_RETURN)


class MarketDataValidator:
    """
    Validate and clean market data using Pandera schemas and custom rules.

    Handles OHLCV validation, outlier detection, missing data identification,
    and corporate action adjustments.
    """

    def __init__(self):
        """Initialize market data validator."""
        self.market_calendar = NSEMarketCalendar()

        # Define OHLCV schema using Pandera
        self.ohlcv_schema = DataFrameSchema(
            {
                "symbol": Column(pa.String, nullable=False),
                "date": Column(pa.DateTime, nullable=False),
                "open": Column(pa.Float, Check.greater_than(0), nullable=False),
                "high": Column(pa.Float, Check.greater_than(0), nullable=False),
                "low": Column(pa.Float, Check.greater_than(0), nullable=False),
                "close": Column(pa.Float, Check.greater_than(0), nullable=False),
                "volume": Column(pa.Int, Check.greater_than_or_equal_to(0), nullable=False),
            },
            strict=False,  # Allow additional columns
            coerce=True,  # Try to coerce types
        )

        logger.info("Market Data Validator initialized")

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        fix_errors: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate OHLCV data structure and values.

        Args:
            df: DataFrame with OHLCV data
            fix_errors: Attempt to fix errors automatically

        Returns:
            Tuple of (validated_df, validation_report)

        Raises:
            pandera.errors.SchemaError: If validation fails and fix_errors=False
        """
        logger.info(f"Validating {len(df)} rows of OHLCV data...")

        validation_report = {
            "total_rows": len(df),
            "valid_rows": 0,
            "errors": [],
            "warnings": [],
            "fixes_applied": [],
        }

        # Make a copy to avoid modifying original
        df = df.copy()

        # 1. Schema validation
        try:
            df = self.ohlcv_schema.validate(df, lazy=True)
            validation_report["valid_rows"] = len(df)
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Schema validation found {len(e.failure_cases)} errors")
            validation_report["errors"].append({
                "type": "schema_validation",
                "count": len(e.failure_cases),
                "details": str(e),
            })

            if not fix_errors:
                raise

            # Try to fix schema errors
            df = self._fix_schema_errors(df, e)
            validation_report["fixes_applied"].append("schema_coercion")

        # 2. OHLC relationship validation
        ohlc_issues = self._validate_ohlc_relationships(df)
        if ohlc_issues:
            validation_report["warnings"].append({
                "type": "ohlc_relationships",
                "count": len(ohlc_issues),
                "details": f"Found {len(ohlc_issues)} OHLC relationship violations",
            })

            if fix_errors:
                df = self._fix_ohlc_relationships(df, ohlc_issues)
                validation_report["fixes_applied"].append("ohlc_relationships")

        # 3. Detect extreme price movements
        extreme_moves = self._detect_extreme_movements(df)
        if extreme_moves:
            validation_report["warnings"].append({
                "type": "extreme_movements",
                "count": len(extreme_moves),
                "details": f"Found {len(extreme_moves)} extreme price movements",
            })

        # 4. Check for duplicates
        duplicates = df.duplicated(subset=["symbol", "date"], keep="first")
        dup_count = duplicates.sum()

        if dup_count > 0:
            validation_report["warnings"].append({
                "type": "duplicates",
                "count": dup_count,
                "details": f"Found {dup_count} duplicate rows",
            })

            if fix_errors:
                df = df[~duplicates]
                validation_report["fixes_applied"].append("duplicate_removal")

        # 5. Check for zero volume
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > 0:
            validation_report["warnings"].append({
                "type": "zero_volume",
                "count": zero_volume,
                "details": f"Found {zero_volume} days with zero volume",
            })

        validation_report["valid_rows"] = len(df)

        logger.info(
            f"Validation complete: {validation_report['valid_rows']}/{validation_report['total_rows']} "
            f"rows valid, {len(validation_report['fixes_applied'])} fixes applied"
        )

        return df, validation_report

    @staticmethod
    def _fix_schema_errors(df: pd.DataFrame, schema_errors: pa.errors.SchemaErrors) -> pd.DataFrame:
        """
        Attempt to fix schema validation errors.

        Args:
            df: DataFrame with errors
            schema_errors: Pandera schema errors

        Returns:
            Fixed DataFrame
        """
        df = df.copy()

        # Convert date to datetime if needed
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=["symbol", "date", "close"])

        return df

    @staticmethod
    def _validate_ohlc_relationships(df: pd.DataFrame) -> List[int]:
        """
        Validate OHLC relationships (High >= Open, Close; Low <= Open, Close).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of row indices with violations
        """
        violations = []

        # High should be >= max(open, close)
        high_violations = df[df["high"] < df[["open", "close"]].max(axis=1)].index.tolist()
        violations.extend(high_violations)

        # Low should be <= min(open, close)
        low_violations = df[df["low"] > df[["open", "close"]].min(axis=1)].index.tolist()
        violations.extend(low_violations)

        return list(set(violations))

    @staticmethod
    def _fix_ohlc_relationships(df: pd.DataFrame, violation_indices: List[int]) -> pd.DataFrame:
        """
        Fix OHLC relationship violations.

        Args:
            df: DataFrame with OHLCV data
            violation_indices: Indices of rows with violations

        Returns:
            Fixed DataFrame
        """
        df = df.copy()

        for idx in violation_indices:
            if idx not in df.index:
                continue

            row = df.loc[idx]

            # Fix high: should be at least max(open, close, low)
            df.loc[idx, "high"] = max(row["open"], row["close"], row["low"], row["high"])

            # Fix low: should be at most min(open, close, high)
            df.loc[idx, "low"] = min(row["open"], row["close"], row["high"], row["low"])

        return df

    @staticmethod
    def _detect_extreme_movements(
        df: pd.DataFrame,
        threshold: float = 0.20,
    ) -> List[Tuple[int, str, float]]:
        """
        Detect extreme price movements (potential errors).

        Args:
            df: DataFrame with OHLCV data
            threshold: Threshold for extreme movement (default 20%)

        Returns:
            List of (index, symbol, change_pct)
        """
        df = df.copy()
        df = df.sort_values(["symbol", "date"])

        # Calculate daily returns
        df["returns"] = df.groupby("symbol")["close"].pct_change()

        # Find extreme movements
        extreme = df[df["returns"].abs() > threshold]

        return [(idx, row["symbol"], row["returns"]) for idx, row in extreme.iterrows()]

    def check_missing_dates(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> List[str]:
        """
        Identify missing trading dates.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            List of missing dates (YYYY-MM-DD)
        """
        symbol_data = df[df["symbol"] == symbol].copy()

        if symbol_data.empty:
            logger.warning(f"No data found for {symbol}")
            return []

        # Get date range
        start_date = symbol_data["date"].min()
        end_date = symbol_data["date"].max()

        # Get expected trading days
        expected_dates = self.market_calendar.get_trading_days(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )

        # Convert to date for comparison
        actual_dates = set(pd.to_datetime(symbol_data["date"]).dt.date)
        expected_dates_set = set([d.date() for d in expected_dates])

        # Find missing dates
        missing_dates = expected_dates_set - actual_dates

        missing_dates_list = sorted([str(d) for d in missing_dates])

        if missing_dates_list:
            logger.warning(f"Found {len(missing_dates_list)} missing trading dates for {symbol}")

        return missing_dates_list

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detect price and volume outliers.

        Args:
            df: DataFrame with OHLCV data
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        df["is_outlier"] = False

        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            symbol_data = df[symbol_mask]

            if len(symbol_data) < 30:  # Need minimum data
                continue

            if method == "iqr":
                # IQR method
                for col in ["close", "volume"]:
                    Q1 = symbol_data[col].quantile(0.25)
                    Q3 = symbol_data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    outliers = (symbol_data[col] < lower_bound) | (symbol_data[col] > upper_bound)
                    df.loc[symbol_mask & outliers.values, "is_outlier"] = True

            elif method == "zscore":
                # Z-score method
                for col in ["close", "volume"]:
                    mean = symbol_data[col].mean()
                    std = symbol_data[col].std()

                    if std == 0:
                        continue

                    z_scores = np.abs((symbol_data[col] - mean) / std)
                    outliers = z_scores > threshold

                    df.loc[symbol_mask & outliers.values, "is_outlier"] = True

        outlier_count = df["is_outlier"].sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method")

        return df

    def fix_corporate_actions(
        self,
        df: pd.DataFrame,
        actions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust prices for corporate actions (splits, dividends).

        Args:
            df: DataFrame with OHLCV data
            actions: DataFrame with corporate actions

        Returns:
            Adjusted DataFrame
        """
        if actions.empty:
            logger.info("No corporate actions to process")
            return df

        df = df.copy()
        df = df.sort_values(["symbol", "date"])

        for _, action in actions.iterrows():
            symbol = action["symbol"]
            action_date = pd.to_datetime(action["date"])
            action_type = action["action_type"]

            # Get data before action date
            mask = (df["symbol"] == symbol) & (df["date"] < action_date)

            if action_type == "split":
                ratio = action.get("ratio", 1.0)

                # Adjust prices (divide by ratio)
                df.loc[mask, "open"] = df.loc[mask, "open"] / ratio
                df.loc[mask, "high"] = df.loc[mask, "high"] / ratio
                df.loc[mask, "low"] = df.loc[mask, "low"] / ratio
                df.loc[mask, "close"] = df.loc[mask, "close"] / ratio

                # Adjust volume (multiply by ratio)
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio

                logger.info(f"Adjusted {symbol} for {ratio}:1 split on {action_date.date()}")

            elif action_type == "dividend":
                amount = action.get("amount", 0.0)

                # Adjust prices (subtract dividend)
                df.loc[mask, "open"] = df.loc[mask, "open"] - amount
                df.loc[mask, "high"] = df.loc[mask, "high"] - amount
                df.loc[mask, "low"] = df.loc[mask, "low"] - amount
                df.loc[mask, "close"] = df.loc[mask, "close"] - amount

                logger.info(f"Adjusted {symbol} for ₹{amount} dividend on {action_date.date()}")

        return df

    def generate_quality_report(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality metrics.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with quality metrics
        """
        report = {
            "total_rows": len(df),
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
            "symbols": {
                "total": df["symbol"].nunique(),
                "list": df["symbol"].unique().tolist(),
            },
            "data_quality": {},
            "issues": [],
        }

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            report["data_quality"]["missing_values"] = missing_counts[missing_counts > 0].to_dict()

        # Check for duplicates
        duplicates = df.duplicated(subset=["symbol", "date"]).sum()
        report["data_quality"]["duplicates"] = duplicates

        # Check for zero volume days
        zero_volume = (df["volume"] == 0).sum()
        report["data_quality"]["zero_volume_days"] = zero_volume

        # Check OHLC relationships
        ohlc_violations = len(self._validate_ohlc_relationships(df))
        report["data_quality"]["ohlc_violations"] = ohlc_violations

        # Detect extreme movements
        extreme_movements = len(self._detect_extreme_movements(df))
        report["data_quality"]["extreme_movements"] = extreme_movements

        # Summary
        total_issues = duplicates + zero_volume + ohlc_violations + extreme_movements

        if total_issues == 0:
            report["data_quality"]["status"] = "excellent"
        elif total_issues < len(df) * 0.01:  # Less than 1%
            report["data_quality"]["status"] = "good"
        elif total_issues < len(df) * 0.05:  # Less than 5%
            report["data_quality"]["status"] = "acceptable"
        else:
            report["data_quality"]["status"] = "poor"

        logger.info(f"Quality report generated: {report['data_quality']['status']} quality")

        return report

    def clean_returns_with_audit(
        self,
        df: pd.DataFrame,
        threshold: float = 0.20,
        log_details: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter extreme returns with comprehensive audit trail.

        Production-grade filtering with detailed logging of all removed data.

        Args:
            df: DataFrame with returns column (or close for calculation)
            threshold: Maximum absolute return (default 20% = NSE circuit breaker)
            log_details: Whether to log detailed filtering info

        Returns:
            Tuple of (cleaned_df, audit_report)
        """
        initial_rows = len(df)
        audit_report = {
            'initial_rows': initial_rows,
            'threshold_used': threshold,
            'filtered_by_symbol': {},
            'extreme_dates': [],
            'total_filtered': 0,
            'retention_rate': 0.0
        }

        # Ensure we have returns column
        df = df.copy()
        if 'returns' not in df.columns:
            if 'close' in df.columns:
                df['returns'] = df.groupby('symbol')['close'].pct_change()
            else:
                logger.error("DataFrame must have 'returns' or 'close' column")
                return df, audit_report

        # Identify extreme returns
        extreme_mask = df['returns'].abs() > threshold
        extreme_df = df[extreme_mask].copy()

        if len(extreme_df) > 0:
            logger.info(f"Found {len(extreme_df)} rows with extreme returns (>{threshold*100:.0f}%)")

            # Log by symbol
            for symbol in extreme_df['symbol'].unique():
                symbol_extreme = extreme_df[extreme_df['symbol'] == symbol]

                audit_report['filtered_by_symbol'][symbol] = {
                    'count': len(symbol_extreme),
                    'dates': symbol_extreme['date'].astype(str).tolist(),
                    'returns': symbol_extreme['returns'].tolist(),
                    'max_return': float(symbol_extreme['returns'].max()),
                    'min_return': float(symbol_extreme['returns'].min())
                }

                if log_details:
                    logger.warning(
                        f"Symbol {symbol}: Filtering {len(symbol_extreme)} extreme returns "
                        f"(>{threshold*100:.0f}%)"
                    )
                    for _, row in symbol_extreme.head(5).iterrows():
                        logger.warning(
                            f"  {row['date']:%Y-%m-%d}: {row['returns']*100:+.2f}% return"
                        )
                    if len(symbol_extreme) > 5:
                        logger.warning(f"  ... and {len(symbol_extreme)-5} more dates")

                # Track extreme dates
                for _, row in symbol_extreme.iterrows():
                    audit_report['extreme_dates'].append({
                        'symbol': symbol,
                        'date': str(row['date']),
                        'return': float(row['returns']),
                        'close': float(row['close']) if 'close' in row else None
                    })

        # Filter extreme returns
        df_clean = df[~extreme_mask].copy()

        audit_report['total_filtered'] = initial_rows - len(df_clean)
        audit_report['retention_rate'] = len(df_clean) / initial_rows if initial_rows > 0 else 0
        audit_report['final_rows'] = len(df_clean)

        logger.info(
            f"Data Quality Filter: {len(df_clean):,}/{initial_rows:,} rows retained "
            f"({audit_report['retention_rate']*100:.1f}%)"
        )

        return df_clean, audit_report

