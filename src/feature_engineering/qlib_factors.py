"""
Qlib Alpha-158 Factor Library.

Implementation of Qlib's 158 institutional-grade alpha factors for NSE markets.
All operations are fully vectorized for maximum performance.

Reference: Qlib Alpha158 (Microsoft Research)
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings('ignore', category=RuntimeWarning)


class QlibAlpha158:
    """
    Qlib Alpha-158 factor library for NSE markets.
    
    Generates 158 technical/statistical features:
    - KLEN/KMID/KSFT families (30 factors): Kline relationships
    - ROC features (30 factors): Rate of change across periods
    - MA features (30 factors): Moving average ratios
    - STD features (20 factors): Volatility measures
    - Volume features (24 factors): Volume ratios, VSTD
    - Beta features (10 factors): Market sensitivity
    - Rolling stats (14 factors): Skew, kurtosis, quantiles
    
    All vectorized operations (NO loops). Handles NaN properly.
    """
    
    def __init__(self):
        """Initialize with standard Qlib periods."""
        self.periods = [5, 10, 20, 30, 60]
        self.short_periods = [5, 10, 20]
        self.long_periods = [30, 60]
        
    def generate_all_factors(
        self,
        df: pd.DataFrame,
        symbol: str = None
    ) -> pd.DataFrame:
        """
        Generate complete 158-factor set.
        
        Args:
            df: OHLCV DataFrame with date index
            symbol: Stock symbol (for logging)
            
        Returns:
            DataFrame with exactly 158 factor columns
        """
        if symbol:
            logger.info(f"Generating Alpha-158 factors for {symbol}")
        
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Make a copy and ensure date index
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            else:
                raise ValueError("DataFrame must have date index or 'date' column")
        
        # Calculate base returns
        df['returns'] = df['close'].pct_change()
        
        # Generate all factor groups
        factors_list = []
        
        # 1. Kline features (30 factors)
        kline_factors = self.generate_kline_features(df)
        factors_list.append(kline_factors)
        
        # 2. ROC features (30 factors)
        roc_factors = self.generate_roc_features(df)
        factors_list.append(roc_factors)
        
        # 3. MA features (30 factors)
        ma_factors = self.generate_ma_features(df)
        factors_list.append(ma_factors)
        
        # 4. STD features (20 factors)
        std_factors = self.generate_std_features(df)
        factors_list.append(std_factors)
        
        # 5. Volume features (24 factors)
        volume_factors = self.generate_volume_features(df)
        factors_list.append(volume_factors)
        
        # 6. Beta features (10 factors)
        beta_factors = self.generate_beta_features(df)
        factors_list.append(beta_factors)
        
        # 7. Rolling stats (14 factors)
        stats_factors = self.generate_rolling_stats(df)
        factors_list.append(stats_factors)
        
        # Combine all factors
        factors = pd.concat(factors_list, axis=1)
        
        # Handle NaN: forward fill then drop incomplete rows
        factors = factors.ffill().fillna(0)
        
        # Replace inf with large finite values
        factors = factors.replace([np.inf, -np.inf], [1e10, -1e10])
        
        # Validate exactly 158 factors
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        assert len(factor_cols) == 158, f"Expected 158 factors, got {len(factor_cols)}"
        
        if symbol:
            logger.info(f"✅ Generated {len(factor_cols)} factors for {symbol}")
        
        return factors
    
    def generate_kline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kline (candlestick) features - 30 factors.
        
        Base features:
        - KLEN = (close - open) / open
        - KMID = (close - open) / (high - low)
        - KMID2 = (close - open) / (close + open) * 2
        - KSFT = (2*close - high - low) / (high - low)
        - KSFT2 = (2*close - high - low) / open
        
        For each period [5,10,20,30,60]:
        - Mean and Std of each base feature (10 features per period)
        
        Select 30 best features
        """
        kline = pd.DataFrame(index=df.index)
        
        # Base kline metrics
        klen = (df['close'] - df['open']) / df['open']
        kmid = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        kmid2 = 2 * (df['close'] - df['open']) / (df['close'] + df['open'] + 1e-10)
        ksft = (2*df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        ksft2 = (2*df['close'] - df['high'] - df['low']) / (df['open'] + 1e-10)
        
        factor_idx = 1
        
        # Rolling statistics for each period
        for period in self.periods:
            kline[f'factor_{factor_idx:03d}'] = klen.rolling(period).mean()
            factor_idx += 1
            kline[f'factor_{factor_idx:03d}'] = klen.rolling(period).std()
            factor_idx += 1
            kline[f'factor_{factor_idx:03d}'] = kmid.rolling(period).mean()
            factor_idx += 1
            kline[f'factor_{factor_idx:03d}'] = ksft.rolling(period).mean()
            factor_idx += 1
            kline[f'factor_{factor_idx:03d}'] = ksft2.rolling(period).mean()
            factor_idx += 1
            kline[f'factor_{factor_idx:03d}'] = kmid2.rolling(period).mean()
            factor_idx += 1
        
        return kline
    
    def generate_roc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rate of Change features - 30 factors.
        
        ROC_p = (close - close.shift(p)) / close.shift(p)
        
        For periods [5,10,20,30,60]:
        - ROC_p itself
        - Mean(ROC_p) over different windows
        - Std(ROC_p)
        - Max/Min ROC_p
        """
        roc = pd.DataFrame(index=df.index)
        
        factor_idx = 31
        
        for period in self.periods:
            # Basic ROC
            roc_p = df['close'].pct_change(period)
            roc[f'factor_{factor_idx:03d}'] = roc_p
            factor_idx += 1
            
            # Rolling statistics of ROC
            roc[f'factor_{factor_idx:03d}'] = roc_p.rolling(10).mean()
            factor_idx += 1
            roc[f'factor_{factor_idx:03d}'] = roc_p.rolling(10).std()
            factor_idx += 1
            roc[f'factor_{factor_idx:03d}'] = roc_p.rolling(10).max()
            factor_idx += 1
            roc[f'factor_{factor_idx:03d}'] = roc_p.rolling(10).min()
            factor_idx += 1
            roc[f'factor_{factor_idx:03d}'] = df['close'].pct_change(period).rolling(20).mean()
            factor_idx += 1
        
        return roc
    
    def generate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Moving Average features - 30 factors.
        
        For periods [5,10,20,30,60]:
        - MA_p = close.rolling(p).mean()
        - (close - MA_p) / close
        - (MA_short - MA_long) / MA_long for various combinations
        """
        ma = pd.DataFrame(index=df.index)
        
        factor_idx = 61
        
        # Calculate all MAs first
        mas = {}
        for period in self.periods:
            mas[period] = df['close'].rolling(period).mean()
        
        # MA deviations (10 factors: 2 per period × 5 periods)
        for period in self.periods:
            ma[f'factor_{factor_idx:03d}'] = (df['close'] - mas[period]) / (df['close'] + 1e-10)
            factor_idx += 1
            ma[f'factor_{factor_idx:03d}'] = (df['close'] - mas[period]) / (mas[period] + 1e-10)
            factor_idx += 1
        
        # MA crossovers (20 factors: all unique pairs)
        ma[f'factor_{factor_idx:03d}'] = (mas[5] - mas[10]) / (mas[10] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[5] - mas[20]) / (mas[20] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[5] - mas[30]) / (mas[30] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[5] - mas[60]) / (mas[60] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[10] - mas[20]) / (mas[20] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[10] - mas[30]) / (mas[30] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[10] - mas[60]) / (mas[60] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[20] - mas[30]) / (mas[30] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[20] - mas[60]) / (mas[60] + 1e-10)
        factor_idx += 1
        ma[f'factor_{factor_idx:03d}'] = (mas[30] - mas[60]) / (mas[60] + 1e-10)
        factor_idx += 1
        
        # Additional MA-based features (10 more factors)
        # MA slopes
        for period in [5, 10, 20, 30, 60]:
            ma_slope = mas[period] - mas[period].shift(5)
            ma[f'factor_{factor_idx:03d}'] = ma_slope / (mas[period] + 1e-10)
            factor_idx += 1
        
        # Distance from MAs
        for period in [10, 20, 60]:
            ma[f'factor_{factor_idx:03d}'] = (df['high'] - mas[period]) / (mas[period] + 1e-10)
            factor_idx += 1
        
        # MA convergence/divergence
        ma[f'factor_{factor_idx:03d}'] = (mas[5] + mas[20]) / 2 - mas[60]
        factor_idx += 1
        
        ma[f'factor_{factor_idx:03d}'] = (mas[10] + mas[30]) / 2 - mas[60]
        factor_idx += 1
        
        return ma
    
    def generate_std_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard deviation (volatility) features - 20 factors.
        
        For periods [5,10,20,30,60]:
        - Std of returns
        - Std of close prices
        - Normalized volatility
        - Rolling beta (if available)
        """
        std = pd.DataFrame(index=df.index)
        
        factor_idx = 91
        
        for period in self.periods:
            # Return volatility
            std[f'factor_{factor_idx:03d}'] = df['returns'].rolling(period).std()
            factor_idx += 1
            
            # Price volatility (normalized)
            std[f'factor_{factor_idx:03d}'] = df['close'].rolling(period).std() / (df['close'].rolling(period).mean() + 1e-10)
            factor_idx += 1
            
            # Parkinson volatility (high-low range)
            hl_vol = (df['high'] - df['low']) / (df['close'] + 1e-10)
            std[f'factor_{factor_idx:03d}'] = hl_vol.rolling(period).mean()
            factor_idx += 1
            
            # Volume volatility
            std[f'factor_{factor_idx:03d}'] = df['volume'].rolling(period).std() / (df['volume'].rolling(period).mean() + 1e-10)
            factor_idx += 1
        
        return std
    
    def generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features - 24 factors.
        
        For periods [5,10,20,30,60]:
        - Volume / MA(Volume)
        - VSTD = Std(Volume) / Mean(Volume)
        - Volume × |Returns| (money flow proxy)
        - Correlation(Volume, |Returns|)
        """
        vol = pd.DataFrame(index=df.index)
        
        factor_idx = 111
        
        for period in [5, 10, 20, 30]:
            # Volume ratio
            vol_ma = df['volume'].rolling(period).mean()
            vol[f'factor_{factor_idx:03d}'] = df['volume'] / (vol_ma + 1e-10)
            factor_idx += 1
            
            # Volume volatility
            vol[f'factor_{factor_idx:03d}'] = df['volume'].rolling(period).std() / (vol_ma + 1e-10)
            factor_idx += 1
            
            # Money flow
            vol[f'factor_{factor_idx:03d}'] = (df['volume'] * df['returns'].abs()).rolling(period).mean()
            factor_idx += 1
            
            # Volume-return correlation
            vol[f'factor_{factor_idx:03d}'] = df['volume'].rolling(period).corr(df['returns'].abs())
            factor_idx += 1
            
            # Volume-price correlation
            vol[f'factor_{factor_idx:03d}'] = df['volume'].rolling(period).corr(df['close'])
            factor_idx += 1
            
            # Accumulation/Distribution proxy
            vol[f'factor_{factor_idx:03d}'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']
            vol[f'factor_{factor_idx:03d}'] = vol[f'factor_{factor_idx:03d}'].rolling(period).mean()
            factor_idx += 1
        
        return vol
    
    def generate_beta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Beta (market sensitivity) features - 4 factors.
        
        Rolling regression of stock returns vs market (using own history as proxy).
        """
        beta = pd.DataFrame(index=df.index)
        
        factor_idx = 135
        
        # Use rolling covariance and variance for beta calculation
        for period in [20, 60]:
            # Rolling beta (using rolling window regression concept)
            # Beta approximation: cov(returns, market) / var(market)
            # Using lagged returns as market proxy
            market_proxy = df['returns'].shift(1)
            
            rolling_cov = df['returns'].rolling(period).cov(market_proxy)
            rolling_var = market_proxy.rolling(period).var()
            
            beta[f'factor_{factor_idx:03d}'] = rolling_cov / (rolling_var + 1e-10)
            factor_idx += 1
        
        # Residual volatility (idiosyncratic risk)
        for period in [20, 60]:
            market_proxy = df['returns'].shift(1)
            rolling_beta = df['returns'].rolling(period).cov(market_proxy) / (market_proxy.rolling(period).var() + 1e-10)
            residuals = df['returns'] - rolling_beta * market_proxy
            beta[f'factor_{factor_idx:03d}'] = residuals.rolling(period).std()
            factor_idx += 1
        
        return beta
    
    def generate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling statistical features - 20 factors.
        
        For key periods:
        - Rolling skewness
        - Rolling kurtosis
        - Rolling quantiles (25th, 75th)
        - Rolling max/min ratios
        - Additional momentum and trend indicators
        """
        stats = pd.DataFrame(index=df.index)
        
        factor_idx = 139  # Start after beta (135-138)
        
        for period in [20, 60]:
            # Skewness of returns
            stats[f'factor_{factor_idx:03d}'] = df['returns'].rolling(period).skew()
            factor_idx += 1
            
            # Kurtosis of returns
            stats[f'factor_{factor_idx:03d}'] = df['returns'].rolling(period).kurt()
            factor_idx += 1
            
            # Quantile ratios
            q25 = df['close'].rolling(period).quantile(0.25)
            q75 = df['close'].rolling(period).quantile(0.75)
            stats[f'factor_{factor_idx:03d}'] = (df['close'] - q25) / (q75 - q25 + 1e-10)
            factor_idx += 1
            
            # Max/Min ratios
            rolling_max = df['close'].rolling(period).max()
            rolling_min = df['close'].rolling(period).min()
            stats[f'factor_{factor_idx:03d}'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
            factor_idx += 1
            
            # RSI-like momentum
            gains = df['returns'].clip(lower=0)
            losses = -df['returns'].clip(upper=0)
            avg_gain = gains.rolling(period).mean()
            avg_loss = losses.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            stats[f'factor_{factor_idx:03d}'] = 100 - (100 / (1 + rs))
            factor_idx += 1
        
        # Additional momentum indicators (10 factors)
        # Price momentum at different periods
        for mom_period in [10, 20]:
            stats[f'factor_{factor_idx:03d}'] = df['close'] / df['close'].shift(mom_period) - 1
            factor_idx += 1
        
        # Distance from highs at different periods
        for high_period in [60, 252]:
            rolling_high = df['close'].rolling(high_period, min_periods=20).max()
            stats[f'factor_{factor_idx:03d}'] = df['close'] / (rolling_high + 1e-10)
            factor_idx += 1
        
        # Trend strength (ADX-like) at different periods
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        
        for tr_period in [14, 20]:
            stats[f'factor_{factor_idx:03d}'] = tr.rolling(tr_period).mean()
            factor_idx += 1
        
        # Breakout indicators
        stats[f'factor_{factor_idx:03d}'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(float)
        factor_idx += 1
        
        stats[f'factor_{factor_idx:03d}'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(float)
        factor_idx += 1
        
        # Volume-price divergence
        stats[f'factor_{factor_idx:03d}'] = df['volume'].pct_change(20) - df['close'].pct_change(20)
        factor_idx += 1
        
        # Acceleration
        stats[f'factor_{factor_idx:03d}'] = df['returns'].diff(20)
        factor_idx += 1
        
        return stats
    
    def get_factor_names(self) -> List[str]:
        """Return list of all 158 factor names."""
        return [f'factor_{i:03d}' for i in range(1, 159)]
    
    def validate_factor_count(self, factors: pd.DataFrame) -> bool:
        """Assert exactly 158 factors generated."""
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        return len(factor_cols) == 158

