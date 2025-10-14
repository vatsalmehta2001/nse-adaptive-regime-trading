"""
Market Calendar Module.

Provides NSE market calendar functionality including trading days,
holidays, and market hours.
"""

from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import pytz
from holidays import India

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class NSEMarketCalendar:
    """
    NSE (National Stock Exchange of India) market calendar.

    Handles trading days, holidays, and market hours for NSE.
    """

    # IST timezone
    TIMEZONE = pytz.timezone("Asia/Kolkata")

    # NSE trading hours (IST)
    MARKET_OPEN = time(9, 15)  # 9:15 AM
    MARKET_CLOSE = time(15, 30)  # 3:30 PM
    PRE_OPEN_START = time(9, 0)  # 9:00 AM
    POST_CLOSE_END = time(16, 0)  # 4:00 PM

    # Trading days (Monday=0, Sunday=6)
    TRADING_DAYS = {0, 1, 2, 3, 4}  # Monday to Friday

    def __init__(self) -> None:
        """Initialize NSE market calendar."""
        self.india_holidays = India()
        logger.debug("NSE Market Calendar initialized")

    def is_holiday(self, date: pd.Timestamp) -> bool:
        """
        Check if date is a public holiday in India.

        Args:
            date: Date to check

        Returns:
            True if holiday, False otherwise
        """
        return date.date() in self.india_holidays

    def is_weekend(self, date: pd.Timestamp) -> bool:
        """
        Check if date is a weekend.

        Args:
            date: Date to check

        Returns:
            True if weekend, False otherwise
        """
        return date.dayofweek not in self.TRADING_DAYS

    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """
        Check if date is a trading day (not weekend or holiday).

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        return not self.is_weekend(date) and not self.is_holiday(date)

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.

        Args:
            dt: Datetime to check (uses current time if None)

        Returns:
            True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now(self.TIMEZONE)
        elif dt.tzinfo is None:
            dt = self.TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.TIMEZONE)

        date = pd.Timestamp(dt.date())

        # Check if trading day
        if not self.is_trading_day(date):
            return False

        # Check if within trading hours
        current_time = dt.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def get_next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Get next trading day after given date.

        Args:
            date: Reference date

        Returns:
            Next trading day
        """
        next_day = date + pd.Timedelta(days=1)

        while not self.is_trading_day(next_day):
            next_day += pd.Timedelta(days=1)

        return next_day

    def get_previous_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Get previous trading day before given date.

        Args:
            date: Reference date

        Returns:
            Previous trading day
        """
        prev_day = date - pd.Timedelta(days=1)

        while not self.is_trading_day(prev_day):
            prev_day -= pd.Timedelta(days=1)

        return prev_day

    def get_trading_days(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> List[pd.Timestamp]:
        """
        Get list of trading days between start and end dates (inclusive).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of trading days
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        return [date for date in date_range if self.is_trading_day(date)]

    def get_trading_days_count(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> int:
        """
        Count trading days between start and end dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of trading days
        """
        return len(self.get_trading_days(start_date, end_date))

    def get_market_hours(
        self,
        date: Optional[pd.Timestamp] = None,
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get market open and close times for a given date.

        Args:
            date: Date to check (uses current date if None)

        Returns:
            Tuple of (market_open, market_close) datetimes, or None if not trading day
        """
        if date is None:
            date = pd.Timestamp.now(tz=self.TIMEZONE).normalize()

        if not self.is_trading_day(date):
            return None

        market_open = self.TIMEZONE.localize(
            datetime.combine(date.date(), self.MARKET_OPEN)
        )
        market_close = self.TIMEZONE.localize(
            datetime.combine(date.date(), self.MARKET_CLOSE)
        )

        return market_open, market_close

    def get_time_to_market_open(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Get time remaining until market opens.

        Args:
            dt: Reference datetime (uses current time if None)

        Returns:
            Timedelta until market open, or None if market is open
        """
        if dt is None:
            dt = datetime.now(self.TIMEZONE)
        elif dt.tzinfo is None:
            dt = self.TIMEZONE.localize(dt)

        if self.is_market_open(dt):
            return None

        date = pd.Timestamp(dt.date())

        # If after market close, get next trading day
        if dt.time() > self.MARKET_CLOSE:
            date = self.get_next_trading_day(date)

        # If not a trading day, get next trading day
        if not self.is_trading_day(date):
            date = self.get_next_trading_day(date)

        market_open = self.TIMEZONE.localize(
            datetime.combine(date.date(), self.MARKET_OPEN)
        )

        return market_open - dt

    def get_time_to_market_close(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Get time remaining until market closes.

        Args:
            dt: Reference datetime (uses current time if None)

        Returns:
            Timedelta until market close, or None if market is closed
        """
        if dt is None:
            dt = datetime.now(self.TIMEZONE)
        elif dt.tzinfo is None:
            dt = self.TIMEZONE.localize(dt)

        if not self.is_market_open(dt):
            return None

        date = pd.Timestamp(dt.date())
        market_close = self.TIMEZONE.localize(
            datetime.combine(date.date(), self.MARKET_CLOSE)
        )

        return market_close - dt

    def wait_for_market_open(self, check_interval: int = 60) -> None:
        """
        Block until market opens.

        Args:
            check_interval: Time between checks in seconds

        Note:
            This is a blocking operation. Use with caution.
        """
        import time

        while not self.is_market_open():
            time_to_open = self.get_time_to_market_open()

            if time_to_open is not None:
                logger.info(f"Market closed. Opens in {time_to_open}")
                time.sleep(min(check_interval, time_to_open.total_seconds()))
            else:
                break

        logger.info("Market is now open")

    def get_holidays(self, year: int) -> List[datetime]:
        """
        Get list of holidays for a given year.

        Args:
            year: Year to get holidays for

        Returns:
            List of holiday dates
        """
        holidays_list = []

        for date in pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            freq="D",
        ):
            if self.is_holiday(date):
                holidays_list.append(date)

        return holidays_list


# Global calendar instance
_calendar_instance: Optional[NSEMarketCalendar] = None


def get_market_calendar() -> NSEMarketCalendar:
    """
    Get global market calendar instance.

    Returns:
        NSE market calendar instance
    """
    global _calendar_instance

    if _calendar_instance is None:
        _calendar_instance = NSEMarketCalendar()

    return _calendar_instance


# Convenience functions
def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if date is a trading day."""
    return get_market_calendar().is_trading_day(date)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if market is currently open."""
    return get_market_calendar().is_market_open(dt)


def get_next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Get next trading day."""
    return get_market_calendar().get_next_trading_day(date)


def get_previous_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Get previous trading day."""
    return get_market_calendar().get_previous_trading_day(date)


def get_trading_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
    """Get list of trading days."""
    return get_market_calendar().get_trading_days(start_date, end_date)

