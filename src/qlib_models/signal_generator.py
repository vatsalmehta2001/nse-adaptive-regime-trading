"""
Alpha Signal Generator.

Converts model predictions to actionable trading signals with various strategies.
Supports rank-based, threshold-based, and long-short portfolio generation.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector


class AlphaSignalGenerator:
    """
    Generate trading signals from trained models.

    Signal types:
    - Raw predictions (continuous)
    - Rank signals (percentile)
    - Long/short signals (top/bottom quantiles)
    - Position weights
    """

    def __init__(
        self,
        models: Dict[str, Any],
        regime_detector: Optional[WassersteinRegimeDetector] = None
    ):
        """
        Initialize signal generator.

        Args:
            models: Dict of model_name -> trained_model or model_dict
            regime_detector: Optional for regime-adaptive signals
        """
        self.models = models
        self.regime_detector = regime_detector

        logger.info(f"Initialized signal generator with {len(models)} models")

    def generate_predictions(
        self,
        factors: pd.DataFrame,
        model_name: str = "default"
    ) -> pd.Series:
        """
        Generate raw return predictions.

        Args:
            factors: DataFrame with factor columns
            model_name: Model identifier

        Returns:
            Series indexed by (symbol, date) with predicted returns

        Raises:
            ValueError: If model not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        model_obj = self.models[model_name]

        # Handle both direct model and dict with 'model' key
        if isinstance(model_obj, dict):
            model = model_obj['model']
            model_type = model_obj.get('metadata', {}).get('model_type', 'lightgbm')
        else:
            model = model_obj
            # Infer model type
            model_type = 'xgboost' if hasattr(model, 'predict') and 'Booster' in str(type(model)) else 'lightgbm'

        # Get factor columns
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        X = factors[factor_cols]

        # Generate predictions
        logger.info(f"Generating predictions with {model_name} ({model_type})...")

        if model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            predictions = model.predict(dmatrix)
        else:
            # LightGBM
            predictions = model.predict(X)

        # Create Series with same index as input
        pred_series = pd.Series(predictions, index=factors.index, name='prediction')

        logger.info(f"Generated {len(pred_series)} predictions")

        return pred_series

    def generate_rank_signals(
        self,
        predictions: pd.Series,
        method: str = "percentile"
    ) -> pd.Series:
        """
        Convert predictions to rank signals.

        Args:
            predictions: Raw predictions
            method: 'percentile', 'zscore', or 'minmax'

        Returns:
            Rank signals (0-1 scale for percentile/minmax, standardized for zscore)

        Raises:
            ValueError: If method not supported
        """
        logger.info(f"Generating rank signals using {method} method...")

        if method == "percentile":
            # Rank by percentile (0-1)
            if isinstance(predictions.index, pd.MultiIndex):
                # Rank within each date
                rank_signals = predictions.groupby(level='date').rank(pct=True)
            else:
                # Simple ranking
                rank_signals = predictions.rank(pct=True)

        elif method == "zscore":
            # Z-score normalization
            if isinstance(predictions.index, pd.MultiIndex):
                # Normalize within each date
                rank_signals = predictions.groupby(level='date').transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
            else:
                rank_signals = (predictions - predictions.mean()) / (predictions.std() + 1e-10)

        elif method == "minmax":
            # Min-max scaling to [0, 1]
            if isinstance(predictions.index, pd.MultiIndex):
                # Scale within each date
                rank_signals = predictions.groupby(level='date').transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
                )
            else:
                rank_signals = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-10)

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'percentile', 'zscore', or 'minmax'")

        rank_signals.name = f'rank_{method}'

        logger.info(f"Rank signals range: [{rank_signals.min():.3f}, {rank_signals.max():.3f}]")

        return rank_signals

    def generate_long_short_signals(
        self,
        predictions: pd.Series,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        n_long: Optional[int] = None,
        n_short: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate long/short portfolio signals.

        Args:
            predictions: Predicted returns by symbol
            long_pct: Top percentile for long (0.2 = top 20%)
            short_pct: Bottom percentile for short
            n_long: Fixed number of long positions (overrides long_pct)
            n_short: Fixed number of short positions (overrides short_pct)

        Returns:
            DataFrame with [symbol, date, signal, weight]
            signal: 1 (long), -1 (short), 0 (neutral)
        """
        logger.info("Generating long/short signals...")

        results = []

        # Handle multi-index (symbol, date) vs single index
        if isinstance(predictions.index, pd.MultiIndex):
            # Group by date and generate signals for each date
            dates = predictions.index.get_level_values('date').unique()

            for date in dates:
                # Get predictions for this date only
                date_preds = predictions.xs(date, level='date')
                signals_df = self._generate_signals_for_date(
                    date_preds, date, long_pct, short_pct, n_long, n_short
                )
                results.append(signals_df)

            signals = pd.concat(results, ignore_index=True)

        else:
            # Single date
            signals = self._generate_signals_for_date(
                predictions, None, long_pct, short_pct, n_long, n_short
            )

        logger.info(f"Generated signals: {(signals['signal']==1).sum()} long, "
                   f"{(signals['signal']==-1).sum()} short")

        return signals

    def _generate_signals_for_date(
        self,
        predictions: pd.Series,
        date: Optional[pd.Timestamp],
        long_pct: float,
        short_pct: float,
        n_long: Optional[int],
        n_short: Optional[int]
    ) -> pd.DataFrame:
        """Generate signals for a single date."""
        # Get symbols
        if isinstance(predictions.index, pd.MultiIndex):
            symbols = predictions.index.get_level_values('symbol')
            pred_values = predictions.values
        else:
            symbols = predictions.index
            pred_values = predictions.values

        # Determine number of longs and shorts
        n_stocks = len(predictions)

        if n_long is None:
            n_long = int(n_stocks * long_pct)
        if n_short is None:
            n_short = int(n_stocks * short_pct)

        # Rank predictions
        sorted_indices = np.argsort(pred_values)[::-1]  # High to low

        # Initialize signals
        signals = np.zeros(n_stocks)
        weights = np.zeros(n_stocks)

        # Long positions (top predictions)
        long_indices = sorted_indices[:n_long]
        signals[long_indices] = 1

        # Equal weight for longs
        if n_long > 0:
            weights[long_indices] = 1.0 / n_long

        # Short positions (bottom predictions)
        short_indices = sorted_indices[-n_short:]
        signals[short_indices] = -1

        # Equal weight for shorts
        if n_short > 0:
            weights[short_indices] = -1.0 / n_short

        # Create DataFrame
        signals_df = pd.DataFrame({
            'symbol': symbols,
            'signal': signals,
            'weight': weights,
            'prediction': pred_values
        })

        if date is not None:
            signals_df['date'] = date

        return signals_df

    def generate_regime_adaptive_signals(
        self,
        factors: pd.DataFrame,
        current_regime: int
    ) -> pd.DataFrame:
        """
        Generate signals using regime-specific model.

        Args:
            factors: Factor data
            current_regime: Current regime identifier

        Returns:
            DataFrame with signals and metadata

        Raises:
            ValueError: If regime model not found
        """
        if self.regime_detector is None:
            raise ValueError("No regime detector provided")

        # Check for regime-specific model
        regime_model_name = f"regime_{current_regime}"

        if regime_model_name not in self.models:
            logger.warning(f"No model for regime {current_regime}, using default")
            regime_model_name = "default"

        logger.info(f"Generating signals for regime {current_regime}...")

        # Generate predictions
        predictions = self.generate_predictions(factors, model_name=regime_model_name)

        # Get regime characteristics for adaptive parameters
        regime_chars = self.regime_detector.get_regime_characteristics()
        regime_info = regime_chars[regime_chars['regime'] == current_regime].iloc[0]

        # Adaptive parameters based on regime
        if regime_info['regime_name'] == 'bull':
            # Bull: More aggressive, more positions
            long_pct = 0.3
            short_pct = 0.1
        elif regime_info['regime_name'] == 'bear':
            # Bear: Defensive, more shorts
            long_pct = 0.1
            short_pct = 0.3
        elif regime_info['regime_name'] == 'high_volatility':
            # High vol: Reduced positions
            long_pct = 0.15
            short_pct = 0.15
        elif regime_info['regime_name'] == 'crash':
            # Crash: Minimal exposure
            long_pct = 0.05
            short_pct = 0.05
        else:
            # Default
            long_pct = 0.2
            short_pct = 0.2

        # Generate signals
        signals = self.generate_long_short_signals(
            predictions,
            long_pct=long_pct,
            short_pct=short_pct
        )

        # Add regime info
        signals['regime'] = current_regime
        signals['regime_name'] = regime_info['regime_name']

        logger.info(f"Regime {current_regime} ({regime_info['regime_name']}): "
                   f"{(signals['signal']==1).sum()} long, {(signals['signal']==-1).sum()} short")

        return signals

    def combine_signals(
        self,
        signal_dfs: list[pd.DataFrame],
        weights: Optional[list[float]] = None
    ) -> pd.DataFrame:
        """
        Combine multiple signal DataFrames with optional weighting.

        Args:
            signal_dfs: List of signal DataFrames
            weights: Optional weights for each signal (must sum to 1)

        Returns:
            Combined signal DataFrame
        """
        if weights is None:
            weights = [1.0 / len(signal_dfs)] * len(signal_dfs)

        if len(weights) != len(signal_dfs):
            raise ValueError("Number of weights must match number of signal DataFrames")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        logger.info(f"Combining {len(signal_dfs)} signal sets with weights {weights}")

        # Combine signals
        combined = signal_dfs[0].copy()
        combined['weight'] = combined['weight'] * weights[0]

        for i, (sig_df, weight) in enumerate(zip(signal_dfs[1:], weights[1:]), 1):
            # Merge on symbol (and date if present)
            merge_on = ['symbol']
            if 'date' in sig_df.columns:
                merge_on.append('date')

            combined = combined.merge(
                sig_df[merge_on + ['weight', 'prediction']],
                on=merge_on,
                how='outer',
                suffixes=('', f'_{i}')
            )

            # Add weighted signal
            combined['weight'] = combined['weight'].fillna(0) + \
                                (combined[f'weight_{i}'].fillna(0) * weight)

        # Recalculate signal from combined weight
        combined['signal'] = np.sign(combined['weight'])

        return combined

