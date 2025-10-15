"""
OpenBB Platform Data Fetcher for NSE.

Professional OpenBB Platform v4 client for fetching NSE market data
with rate limiting, retry logic, and caching.
"""

import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")

            raise last_exception

        return wrapper

    return decorator


class RateLimiter:
    """Simple rate limiter to respect API constraints."""

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class OpenBBDataFetcher:
    """
    Professional OpenBB Platform client for NSE data.

    Uses OpenBB Platform v4 with proper error handling, rate limiting,
    and caching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenBB data fetcher.

        Args:
            config: Configuration dictionary from data_sources.yaml
        """
        self.config = config or {}

        # Extract configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        self.timeout = self.config.get("timeout", 30)
        self.cache_ttl = self.config.get("cache_ttl", 3600)

        # Rate limiting
        rpm = self.config.get("requests_per_minute", 60)
        self.rate_limiter = RateLimiter(requests_per_minute=rpm)

        # Cache for API responses
        self._cache: Dict[str, tuple] = {}  # key: (data, timestamp)

        logger.info("OpenBB Data Fetcher initialized")

    def _get_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "|".join(key_parts)

    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached DataFrame or None
        """
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            age = time.time() - timestamp

            if age < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key[:50]}... (age: {age:.1f}s)")
                return data
            else:
                logger.debug(f"Cache expired for {cache_key[:50]}...")
                del self._cache[cache_key]

        return None

    def _set_cached(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        Cache data with timestamp.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        self._cache[cache_key] = (data.copy(), time.time())

    @staticmethod
    def _normalize_nse_symbol(symbol: str) -> str:
        """
        Normalize NSE symbol to OpenBB format.

        Args:
            symbol: Symbol (e.g., "RELIANCE" or "RELIANCE.NS")

        Returns:
            Normalized symbol (e.g., "RELIANCE.NS")
        """
        symbol = symbol.upper().strip()

        # Add .NS suffix if not present
        if not symbol.endswith(".NS"):
            symbol = f"{symbol}.NS"

        return symbol

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def fetch_equity_ohlcv(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for NSE stocks.

        Args:
            symbols: Symbol or list of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data

        Raises:
            ImportError: If OpenBB not installed
            ValueError: If invalid parameters
        """
        # Normalize symbols
        if isinstance(symbols, str):
            symbols = [symbols]

        symbols = [self._normalize_nse_symbol(s) for s in symbols]

        logger.info(
            f"Fetching OHLCV data for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        all_data = []

        for symbol in symbols:
            # Check cache
            cache_key = self._get_cache_key(
                "ohlcv", symbol, start_date, end_date, interval
            )
            cached_data = self._get_cached(cache_key)

            if cached_data is not None:
                all_data.append(cached_data)
                continue

            # Rate limiting
            self.rate_limiter.wait_if_needed()

            try:
                # Import OpenBB here to allow graceful degradation
                from openbb import obb

                logger.debug(f"Fetching {symbol} via OpenBB...")

                # Fetch data using OpenBB Platform v4
                result = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    provider="yfinance",  # Using yfinance for NSE data
                )

                # Convert to DataFrame
                df = result.to_df()

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue

                # Add symbol column
                df["symbol"] = symbol.replace(".NS", "")

                # Standardize column names
                df = self._standardize_ohlcv_columns(df)

                # Cache the result
                self._set_cached(cache_key, df)

                all_data.append(df)

                logger.info(f"Fetched {len(df)} rows for {symbol}")

            except ImportError:
                logger.error(
                    "OpenBB not installed. Install with: pip install openbb"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                # Don't raise, continue with other symbols
                continue

        if not all_data:
            logger.warning("No data fetched for any symbol")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        logger.info(
            f"Successfully fetched {len(combined_df)} total rows for "
            f"{len(symbols)} symbols"
        )

        return combined_df

    @staticmethod
    def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize OHLCV column names.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized columns
        """
        # Map common column name variations
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Date": "date",
        }

        df = df.rename(columns=column_mapping)

        # Ensure date is in datetime format
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.columns = ["date"] + list(df.columns[1:])

        # Add adj_close if missing (use close)
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        return df

    def fetch_index_data(
        self,
        index_name: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch NSE index data.

        Args:
            index_name: Index name (NIFTY, BANKNIFTY, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with index data
        """
        # Map index names to symbols
        index_symbols = {
            "NIFTY": "^NSEI",
            "NIFTY50": "^NSEI",
            "NSEI": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "NIFTYBANK": "^NSEBANK",
        }

        symbol = index_symbols.get(index_name.upper(), f"^{index_name}")

        logger.info(f"Fetching index data for {index_name} ({symbol})")

        return self.fetch_equity_ohlcv(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

    def get_nifty50_constituents(self) -> List[str]:
        """
        Get current NIFTY 50 constituent stocks.

        Returns:
            List of stock symbols

        Note:
            This returns a hardcoded list. In production, fetch from NSE API.
        """
        # NIFTY 50 constituents (as of 2024)
        # In production, fetch this dynamically from NSE API
        nifty50 = [
            "ADANIPORTS",
            "ASIANPAINT",
            "AXISBANK",
            "BAJAJ-AUTO",
            "BAJFINANCE",
            "BAJAJFINSV",
            "BHARTIARTL",
            "BPCL",
            "BRITANNIA",
            "CIPLA",
            "COALINDIA",
            "DIVISLAB",
            "DRREDDY",
            "EICHERMOT",
            "GRASIM",
            "HCLTECH",
            "HDFCBANK",
            "HDFCLIFE",
            "HEROMOTOCO",
            "HINDALCO",
            "HINDUNILVR",
            "ICICIBANK",
            "INDUSINDBK",
            "INFY",
            "ITC",
            "JSWSTEEL",
            "KOTAKBANK",
            "LT",
            "M&M",
            "MARUTI",
            "NESTLEIND",
            "NTPC",
            "ONGC",
            "POWERGRID",
            "RELIANCE",
            "SBIN",
            "SBILIFE",
            "SUNPHARMA",
            "TATAMOTORS",
            "TATASTEEL",
            "TCS",
            "TECHM",
            "TITAN",
            "ULTRACEMCO",
            "UPL",
            "WIPRO",
        ]

        logger.info(f"Retrieved {len(nifty50)} NIFTY 50 constituents")

        return nifty50

    def fetch_fundamentals(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """
        Fetch fundamental data for NSE stocks.

        Args:
            symbols: Symbol or list of symbols

        Returns:
            DataFrame with fundamental data

        Note:
            OpenBB Platform v4 fundamental data support varies by provider.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        symbols = [self._normalize_nse_symbol(s) for s in symbols]

        logger.info(f"Fetching fundamentals for {len(symbols)} symbols")

        fundamentals = []

        for symbol in symbols:
            self.rate_limiter.wait_if_needed()

            try:
                from openbb import obb

                # Fetch company profile
                profile = obb.equity.profile(symbol=symbol, provider="yfinance")
                profile_df = profile.to_df()

                if not profile_df.empty:
                    profile_df["symbol"] = symbol.replace(".NS", "")
                    fundamentals.append(profile_df)

            except Exception as e:
                logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
                continue

        if not fundamentals:
            return pd.DataFrame()

        return pd.concat(fundamentals, ignore_index=True)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        current_time = time.time()
        valid_entries = sum(
            1
            for _, (_, timestamp) in self._cache.items()
            if current_time - timestamp < self.cache_ttl
        )

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "cache_ttl": self.cache_ttl,
        }

