"""
Technical Indicators Module.

Generates technical indicators from OHLCV data using vectorized operations
for high performance. Provides foundation for Qlib alpha factors.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Generate technical indicators from OHLCV data.

    Uses vectorized pandas operations for efficiency. Handles NaN values
    properly and provides comprehensive technical analysis features.
    """

    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        price_col: str = "close",
        periods: List[int] = [1, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        Calculate returns over various periods.

        Args:
            df: DataFrame with OHLCV data (must have symbol and price_col)
            price_col: Column name for price (default: 'close')
            periods: List of periods for return calculation

        Returns:
            DataFrame with return columns added
        """
        df = df.copy()

        for period in periods:
            # Simple returns
            df[f"return_{period}d"] = df.groupby("symbol")[price_col].pct_change(period)

            # Log returns
            df[f"log_return_{period}d"] = df.groupby("symbol")[price_col].apply(
                lambda x: np.log(x / x.shift(period))
            )

        logger.debug(f"Calculated returns for periods: {periods}")

        return df

    @staticmethod
    def calculate_moving_averages(
        df: pd.DataFrame,
        price_col: str = "close",
        windows: List[int] = [5, 10, 20, 50, 200],
    ) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            windows: List of window sizes

        Returns:
            DataFrame with MA columns added
        """
        df = df.copy()

        for window in windows:
            # Simple Moving Average
            df[f"sma_{window}"] = df.groupby("symbol")[price_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # Exponential Moving Average
            df[f"ema_{window}"] = df.groupby("symbol")[price_col].transform(
                lambda x: x.ewm(span=window, adjust=False).mean()
            )

            # Distance from MA (as percentage)
            df[f"dist_sma_{window}"] = (
                (df[price_col] - df[f"sma_{window}"]) / df[f"sma_{window}"]
            ) * 100

        logger.debug(f"Calculated moving averages for windows: {windows}")

        return df

    @staticmethod
    def calculate_volatility(
        df: pd.DataFrame,
        price_col: str = "close",
        windows: List[int] = [10, 20, 30, 60],
    ) -> pd.DataFrame:
        """
        Calculate historical volatility (rolling standard deviation).

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            windows: List of window sizes

        Returns:
            DataFrame with volatility columns added
        """
        df = df.copy()

        # First calculate returns if not present
        if "return_1d" not in df.columns:
            df["return_1d"] = df.groupby("symbol")[price_col].pct_change()

        for window in windows:
            # Rolling standard deviation of returns (annualized)
            df[f"volatility_{window}d"] = df.groupby("symbol")["return_1d"].transform(
                lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).std()
                * np.sqrt(252)  # Annualize
            )

            # Parkinson volatility (using high-low range)
            df[f"parkinson_vol_{window}d"] = df.groupby("symbol").apply(
                lambda x: (
                    np.log(x["high"] / x["low"]) ** 2 / (4 * np.log(2))
                ).rolling(window=window, min_periods=max(1, window // 2)).mean()
                ** 0.5
                * np.sqrt(252)
            ).reset_index(level=0, drop=True)

        logger.debug(f"Calculated volatility for windows: {windows}")

        return df

    @staticmethod
    def calculate_rsi(
        df: pd.DataFrame,
        price_col: str = "close",
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            period: RSI period (default: 14)

        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()

        def calculate_rsi_for_group(group):
            delta = group[price_col].diff()

            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        df[f"rsi_{period}"] = df.groupby("symbol", group_keys=False).apply(
            calculate_rsi_for_group
        )

        logger.debug(f"Calculated RSI with period {period}")

        return df

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        price_col: str = "close",
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            period: Period for moving average
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band columns added
        """
        df = df.copy()

        # Calculate middle band (SMA)
        df[f"bb_middle_{period}"] = df.groupby("symbol")[price_col].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )

        # Calculate standard deviation
        df[f"bb_std_{period}"] = df.groupby("symbol")[price_col].transform(
            lambda x: x.rolling(window=period, min_periods=1).std()
        )

        # Calculate upper and lower bands
        df[f"bb_upper_{period}"] = (
            df[f"bb_middle_{period}"] + std_dev * df[f"bb_std_{period}"]
        )
        df[f"bb_lower_{period}"] = (
            df[f"bb_middle_{period}"] - std_dev * df[f"bb_std_{period}"]
        )

        # Calculate %B (position within bands)
        df[f"bb_percent_{period}"] = (df[price_col] - df[f"bb_lower_{period}"]) / (
            df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
        )

        # Calculate bandwidth
        df[f"bb_width_{period}"] = (
            df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
        ) / df[f"bb_middle_{period}"]

        logger.debug(f"Calculated Bollinger Bands with period {period}")

        return df

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        price_col: str = "close",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()

        # Calculate fast and slow EMAs
        df["ema_fast"] = df.groupby("symbol")[price_col].transform(
            lambda x: x.ewm(span=fast_period, adjust=False).mean()
        )

        df["ema_slow"] = df.groupby("symbol")[price_col].transform(
            lambda x: x.ewm(span=slow_period, adjust=False).mean()
        )

        # MACD line
        df["macd"] = df["ema_fast"] - df["ema_slow"]

        # Signal line
        df["macd_signal"] = df.groupby("symbol")["macd"].transform(
            lambda x: x.ewm(span=signal_period, adjust=False).mean()
        )

        # MACD histogram
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Clean up temporary columns
        df = df.drop(columns=["ema_fast", "ema_slow"])

        logger.debug(
            f"Calculated MACD ({fast_period},{slow_period},{signal_period})"
        )

        return df

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with OHLCV data
            period: ATR period

        Returns:
            DataFrame with ATR column added
        """
        df = df.copy()

        # Calculate true range components
        df["h_l"] = df["high"] - df["low"]
        df["h_pc"] = abs(df["high"] - df.groupby("symbol")["close"].shift(1))
        df["l_pc"] = abs(df["low"] - df.groupby("symbol")["close"].shift(1))

        # True range is the max of the three
        df["tr"] = df[["h_l", "h_pc", "l_pc"]].max(axis=1)

        # ATR is the moving average of true range
        df[f"atr_{period}"] = df.groupby("symbol")["tr"].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )

        # ATR as percentage of price
        df[f"atr_percent_{period}"] = (df[f"atr_{period}"] / df["close"]) * 100

        # Clean up temporary columns
        df = df.drop(columns=["h_l", "h_pc", "l_pc", "tr"])

        logger.debug(f"Calculated ATR with period {period}")

        return df

    @staticmethod
    def calculate_momentum_indicators(
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate various momentum indicators.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price

        Returns:
            DataFrame with momentum indicators added
        """
        df = df.copy()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = df.groupby("symbol")[price_col].transform(
                lambda x: ((x / x.shift(period)) - 1) * 100
            )

        # Stochastic Oscillator
        period = 14
        df["lowest_low"] = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(window=period, min_periods=1).min()
        )
        df["highest_high"] = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(window=period, min_periods=1).max()
        )

        df["stoch_k"] = (
            (df[price_col] - df["lowest_low"])
            / (df["highest_high"] - df["lowest_low"])
            * 100
        )

        df["stoch_d"] = df.groupby("symbol")["stoch_k"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        # Clean up temporary columns
        df = df.drop(columns=["lowest_low", "highest_high"])

        logger.debug("Calculated momentum indicators")

        return df

    @staticmethod
    def calculate_volume_indicators(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate volume-based indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume indicators added
        """
        df = df.copy()

        # Volume moving averages
        for window in [5, 10, 20]:
            df[f"volume_sma_{window}"] = df.groupby("symbol")["volume"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # Volume ratio (current vs average)
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_sma_{window}"]

        # Volume momentum
        df["volume_momentum_5"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.pct_change(5)
        )

        # On-Balance Volume (OBV)
        df["price_change"] = df.groupby("symbol")["close"].diff()
        df["obv_contribution"] = df["volume"] * np.sign(df["price_change"])
        df["obv"] = df.groupby("symbol")["obv_contribution"].transform(
            lambda x: x.cumsum()
        )

        # Clean up temporary columns
        df = df.drop(columns=["price_change", "obv_contribution"])

        logger.debug("Calculated volume indicators")

        return df

    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_basic: bool = True,
        include_advanced: bool = True,
    ) -> pd.DataFrame:
        """
        Generate complete technical indicator feature set.

        Args:
            df: DataFrame with OHLCV data
            include_basic: Include basic indicators (returns, MA, volatility)
            include_advanced: Include advanced indicators (RSI, MACD, etc.)

        Returns:
            DataFrame with all features
        """
        logger.info(f"Generating technical indicators for {len(df)} rows...")

        df = df.copy()

        if include_basic:
            # Basic features
            df = self.calculate_returns(df)
            df = self.calculate_moving_averages(df)
            df = self.calculate_volatility(df)

        if include_advanced:
            # Advanced technical indicators
            df = self.calculate_rsi(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_macd(df)
            df = self.calculate_atr(df)
            df = self.calculate_momentum_indicators(df)
            df = self.calculate_volume_indicators(df)

        # Count number of features added
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adj_close",
            ]
        ]

        logger.info(f"Generated {len(feature_cols)} technical indicator features")

        return df

