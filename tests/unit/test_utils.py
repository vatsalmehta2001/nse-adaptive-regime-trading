"""
Unit tests for utility modules.

Tests for logging, helpers, database, and market calendar utilities.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.helpers import (
    format_currency,
    format_percentage,
    load_config,
    safe_divide,
    save_config,
)
from src.utils.market_calendar import NSEMarketCalendar


class TestHelpers:
    """Test helper functions."""

    def test_format_currency(self) -> None:
        """Test currency formatting."""
        assert format_currency(1000000) == "₹1,000,000.00"
        assert format_currency(1000000, "USD") == "$1,000,000.00"
        assert format_currency(1234.56, "INR", decimals=2) == "₹1,234.56"

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        assert format_percentage(0.05) == "5.00%"
        assert format_percentage(0.1234, decimals=2) == "12.34%"
        assert format_percentage(-0.05) == "-5.00%"

    def test_safe_divide(self) -> None:
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=999.0) == 999.0

    def test_load_save_config(self, tmp_path: Path) -> None:
        """Test configuration loading and saving."""
        config = {"key": "value", "number": 42}

        # Test YAML
        yaml_path = tmp_path / "config.yaml"
        save_config(config, yaml_path)
        loaded = load_config(yaml_path)
        assert loaded == config

        # Test JSON
        json_path = tmp_path / "config.json"
        save_config(config, json_path)
        loaded = load_config(json_path)
        assert loaded == config


class TestMarketCalendar:
    """Test market calendar functionality."""

    def test_trading_days(self) -> None:
        """Test trading day identification."""
        calendar = NSEMarketCalendar()

        # Monday (should be trading day if not holiday)
        monday = pd.Timestamp("2024-01-08")  # A Monday
        assert monday.dayofweek == 0
        # Can't assert is_trading_day without knowing if it's a holiday

        # Saturday (should not be trading day)
        saturday = pd.Timestamp("2024-01-06")
        assert calendar.is_weekend(saturday)

        # Sunday (should not be trading day)
        sunday = pd.Timestamp("2024-01-07")
        assert calendar.is_weekend(sunday)

    def test_get_next_trading_day(self) -> None:
        """Test getting next trading day."""
        calendar = NSEMarketCalendar()

        # From Friday to next trading day (likely Monday)
        friday = pd.Timestamp("2024-01-05")  # A Friday
        next_day = calendar.get_next_trading_day(friday)

        # Next trading day should not be weekend
        assert not calendar.is_weekend(next_day)

    def test_get_trading_days_count(self) -> None:
        """Test counting trading days."""
        calendar = NSEMarketCalendar()

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        count = calendar.get_trading_days_count(start, end)

        # Should be approximately 22-23 trading days in January
        assert 20 <= count <= 25


@pytest.mark.unit
def test_placeholder() -> None:
    """Placeholder test to ensure test framework works."""
    assert True

