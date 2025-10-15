"""
Tests for Data Quality Improvements.

Tests all new data quality functionality including thresholds, audit trails,
and quality reporting.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Suppress pandera deprecation warnings in tests
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'

from src.data_pipeline.data_validator import DataQualityConfig, MarketDataValidator
from src.data_pipeline.quality_reporter import DataQualityReporter


class TestDataQualityConfig:
    """Test DataQualityConfig class."""
    
    def test_threshold_values(self):
        """Test that thresholds are set correctly."""
        assert DataQualityConfig.MAX_DAILY_RETURN == 0.20
        assert DataQualityConfig.MAX_WEEKLY_RETURN == 0.50
        assert DataQualityConfig.MAX_MONTHLY_RETURN == 1.00
        assert DataQualityConfig.MIN_PRICE == 1.0
    
    def test_get_threshold_method(self):
        """Test get_threshold method."""
        assert DataQualityConfig.get_threshold('daily') == 0.20
        assert DataQualityConfig.get_threshold('weekly') == 0.50
        assert DataQualityConfig.get_threshold('monthly') == 1.00
        assert DataQualityConfig.get_threshold('invalid') == 0.20  # Default
    
    def test_data_completeness_thresholds(self):
        """Test data completeness thresholds."""
        assert DataQualityConfig.MIN_DATA_POINTS == 60
        assert DataQualityConfig.MAX_MISSING_DAYS_RATIO == 0.10


class TestDataQualityFiltering:
    """Test improved data quality filtering with audit trails."""
    
    @pytest.fixture
    def sample_data_with_extremes(self):
        """Create test data with extreme returns."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'symbol': ['TEST'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(10000, 100000, 100)
        })
        
        df['returns'] = df['close'].pct_change()
        
        # Add some extreme returns
        df.loc[10, 'returns'] = 0.50  # 50% return (extreme)
        df.loc[20, 'returns'] = -0.40  # -40% return (extreme)
        df.loc[30, 'returns'] = 0.25  # 25% return (borderline)
        
        return df
    
    def test_clean_returns_with_audit(self, sample_data_with_extremes):
        """Test returns cleaning with audit trail."""
        validator = MarketDataValidator()
        
        clean_df, audit = validator.clean_returns_with_audit(
            sample_data_with_extremes, 
            threshold=0.20,
            log_details=False
        )
        
        # Should filter 3 rows (50%, -40%, 25%)
        assert len(clean_df) == 97
        assert audit['total_filtered'] == 3
        assert 'TEST' in audit['filtered_by_symbol']
        assert audit['filtered_by_symbol']['TEST']['count'] == 3
        assert audit['retention_rate'] == 0.97
    
    def test_audit_trail_structure(self, sample_data_with_extremes):
        """Test audit trail contains all required information."""
        validator = MarketDataValidator()
        
        clean_df, audit = validator.clean_returns_with_audit(
            sample_data_with_extremes,
            threshold=0.20
        )
        
        # Check audit structure
        assert 'initial_rows' in audit
        assert 'threshold_used' in audit
        assert 'filtered_by_symbol' in audit
        assert 'extreme_dates' in audit
        assert 'total_filtered' in audit
        assert 'retention_rate' in audit
        
        # Check filtered_by_symbol details
        if 'TEST' in audit['filtered_by_symbol']:
            symbol_audit = audit['filtered_by_symbol']['TEST']
            assert 'count' in symbol_audit
            assert 'dates' in symbol_audit
            assert 'returns' in symbol_audit
            assert 'max_return' in symbol_audit
            assert 'min_return' in symbol_audit
    
    def test_different_thresholds(self, sample_data_with_extremes):
        """Test filtering with different thresholds."""
        validator = MarketDataValidator()
        
        # Strict threshold (10%)
        clean_strict, audit_strict = validator.clean_returns_with_audit(
            sample_data_with_extremes,
            threshold=0.10,
            log_details=False
        )
        
        # Lenient threshold (30%)
        clean_lenient, audit_lenient = validator.clean_returns_with_audit(
            sample_data_with_extremes,
            threshold=0.30,
            log_details=False
        )
        
        # Strict should filter more
        assert len(clean_strict) <= len(clean_lenient)
        assert audit_strict['total_filtered'] >= audit_lenient['total_filtered']
    
    def test_no_returns_column(self):
        """Test handling of missing returns column."""
        df = pd.DataFrame({
            'symbol': ['TEST'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'close': 100 + np.random.randn(10)
        })
        
        validator = MarketDataValidator()
        clean_df, audit = validator.clean_returns_with_audit(df, log_details=False)
        
        # Should calculate returns from close
        assert 'returns' in clean_df.columns


class TestQualityReporting:
    """Test quality reporting functionality."""
    
    @pytest.fixture
    def sample_audit(self):
        """Create sample audit trail."""
        return {
            'retention_rate': 0.95,
            'total_filtered': 5,
            'threshold_used': 0.20,
            'filtered_by_symbol': {
                'TEST': {
                    'count': 5,
                    'max_return': 0.30,
                    'min_return': -0.25
                }
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'symbol': ['TEST'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(100000, 1000000, 100)
        })
    
    def test_quality_report_generation(self, tmp_path, sample_data, sample_audit):
        """Test report generation."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        report = reporter.generate_quality_report(sample_data, sample_audit, symbol='TEST')
        
        # Check required fields
        assert 'quality_score' in report
        assert 'quality_grade' in report
        assert 0 <= report['quality_score'] <= 100
        assert report['quality_grade'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR']
        assert report['data_quality']['retention_rate'] == 0.95
    
    def test_quality_score_calculation(self, tmp_path):
        """Test quality score calculation logic."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        # Perfect data
        report_perfect = {
            'data_quality': {'retention_rate': 1.0},
            'missing_data': {'completeness': 1.0},
            'data_issues': []
        }
        score_perfect = reporter._calculate_quality_score(report_perfect)
        assert score_perfect == 100
        
        # Good data (95% retention, 98% completeness)
        report_good = {
            'data_quality': {'retention_rate': 0.95},
            'missing_data': {'completeness': 0.98},
            'data_issues': [{'severity': 'low'}]
        }
        score_good = reporter._calculate_quality_score(report_good)
        assert 85 <= score_good <= 95
        
        # Poor data
        report_poor = {
            'data_quality': {'retention_rate': 0.5},
            'missing_data': {'completeness': 0.7},
            'data_issues': [
                {'severity': 'high'},
                {'severity': 'high'},
                {'severity': 'medium'}
            ]
        }
        score_poor = reporter._calculate_quality_score(report_poor)
        assert score_poor < 60
    
    def test_quality_grades(self, tmp_path):
        """Test quality grade assignment."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        assert reporter._get_quality_grade(95) == "EXCELLENT"
        assert reporter._get_quality_grade(80) == "GOOD"
        assert reporter._get_quality_grade(65) == "ACCEPTABLE"
        assert reporter._get_quality_grade(50) == "POOR"
    
    def test_report_saving(self, tmp_path, sample_data, sample_audit):
        """Test that reports are saved correctly."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        report = reporter.generate_quality_report(sample_data, sample_audit, symbol='TEST')
        
        # Check files were created
        json_files = list(tmp_path.glob('quality_report_TEST_*.json'))
        csv_files = list(tmp_path.glob('quality_summary_TEST_*.csv'))
        
        assert len(json_files) >= 1
        assert len(csv_files) >= 1
        
        # Verify JSON content
        with open(json_files[0]) as f:
            saved_report = json.load(f)
            assert saved_report['symbol'] == 'TEST'
            assert 'quality_score' in saved_report
        
        # Verify CSV content
        csv_df = pd.read_csv(csv_files[0])
        assert len(csv_df) == 1
        assert csv_df['symbol'].iloc[0] == 'TEST'
    
    def test_comparison_report(self, tmp_path):
        """Test cross-symbol comparison report."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        # Create multiple reports
        reports = [
            {
                'symbol': 'SYM1',
                'quality_score': 95.0,
                'quality_grade': 'EXCELLENT',
                'data_quality': {'retention_rate': 0.98, 'filtered_rows': 2},
                'missing_data': {'completeness': 0.99},
                'data_issues': []
            },
            {
                'symbol': 'SYM2',
                'quality_score': 75.0,
                'quality_grade': 'GOOD',
                'data_quality': {'retention_rate': 0.90, 'filtered_rows': 10},
                'missing_data': {'completeness': 0.95},
                'data_issues': [{'severity': 'low'}]
            },
            {
                'symbol': 'SYM3',
                'quality_score': 50.0,
                'quality_grade': 'POOR',
                'data_quality': {'retention_rate': 0.70, 'filtered_rows': 30},
                'missing_data': {'completeness': 0.80},
                'data_issues': [{'severity': 'high'}, {'severity': 'medium'}]
            }
        ]
        
        comparison = reporter.generate_comparison_report(reports)
        
        # Check structure
        assert len(comparison) == 3
        assert 'quality_score' in comparison.columns
        assert 'quality_grade' in comparison.columns
        
        # Check sorting (best quality first)
        assert comparison.iloc[0]['symbol'] == 'SYM1'
        assert comparison.iloc[2]['symbol'] == 'SYM3'


class TestIntegration:
    """Integration tests for data quality pipeline."""
    
    def test_end_to_end_quality_pipeline(self, tmp_path):
        """Test complete quality pipeline."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'symbol': ['TEST'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(10000, 100000, 100)
        })
        
        # Add extreme returns
        df.loc[10, 'close'] = df.loc[9, 'close'] * 1.5  # 50% jump
        df.loc[20, 'close'] = df.loc[19, 'close'] * 0.6  # -40% drop
        
        # Step 1: Validate and clean
        validator = MarketDataValidator()
        clean_df, audit = validator.clean_returns_with_audit(
            df,
            threshold=DataQualityConfig.MAX_DAILY_RETURN,
            log_details=False
        )
        
        # Step 2: Generate quality report
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        report = reporter.generate_quality_report(clean_df, audit, symbol='TEST')
        
        # Verify pipeline results
        assert len(clean_df) < len(df)  # Some data filtered
        assert audit['total_filtered'] >= 2  # At least 2 extreme returns
        assert 'quality_score' in report
        assert report['quality_score'] > 0
        
        # Verify files saved
        assert len(list(tmp_path.glob('*.json'))) >= 1
        assert len(list(tmp_path.glob('*.csv'))) >= 1
    
    def test_threshold_vs_quality_score(self):
        """Test that stricter thresholds don't always mean better quality."""
        df = pd.DataFrame({
            'symbol': ['TEST'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close': 100 + np.random.randn(100) * 0.5,  # Low volatility
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        validator = MarketDataValidator()
        
        # Lenient threshold (should retain more data)
        clean_lenient, audit_lenient = validator.clean_returns_with_audit(
            df, threshold=0.30, log_details=False
        )
        
        # Strict threshold (might filter good data)
        clean_strict, audit_strict = validator.clean_returns_with_audit(
            df, threshold=0.05, log_details=False
        )
        
        # Lenient should have better retention
        assert audit_lenient['retention_rate'] >= audit_strict['retention_rate']


class TestProductionReadiness:
    """Test production-grade features."""
    
    def test_no_breaking_changes(self):
        """Ensure backward compatibility."""
        # Old code should still work
        validator = MarketDataValidator()
        
        # Old-style validation should still work
        df = pd.DataFrame({
            'symbol': ['TEST'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'open': np.random.uniform(90, 110, 10),
            'high': np.random.uniform(95, 115, 10),
            'low': np.random.uniform(85, 105, 10),
            'close': np.random.uniform(90, 110, 10),
            'volume': np.random.randint(10000, 100000, 10)
        })
        
        # This should not raise an error
        validated_df, report = validator.validate_ohlcv(df)
        assert len(validated_df) > 0
    
    def test_audit_trail_completeness(self):
        """Test that audit trail has all required fields."""
        validator = MarketDataValidator()
        
        df = pd.DataFrame({
            'symbol': ['TEST'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'close': 100 + np.random.randn(50),
            'volume': np.random.randint(10000, 100000, 50)
        })
        
        clean_df, audit = validator.clean_returns_with_audit(df, log_details=False)
        
        # Required audit fields
        required_fields = [
            'initial_rows',
            'threshold_used',
            'filtered_by_symbol',
            'extreme_dates',
            'total_filtered',
            'retention_rate',
            'final_rows'
        ]
        
        for field in required_fields:
            assert field in audit, f"Missing required audit field: {field}"
    
    def test_quality_score_bounds(self, tmp_path):
        """Test that quality scores are always in valid range."""
        reporter = DataQualityReporter(output_dir=str(tmp_path))
        
        # Test extreme cases
        test_cases = [
            {'data_quality': {'retention_rate': 0.0}, 'missing_data': {'completeness': 0.0}, 'data_issues': []},
            {'data_quality': {'retention_rate': 1.0}, 'missing_data': {'completeness': 1.0}, 'data_issues': []},
            {'data_quality': {'retention_rate': 0.5}, 'missing_data': {'completeness': 0.5}, 'data_issues': [{'severity': 'high'}] * 10}
        ]
        
        for test_case in test_cases:
            score = reporter._calculate_quality_score(test_case)
            assert 0 <= score <= 100, f"Score out of bounds: {score}"

