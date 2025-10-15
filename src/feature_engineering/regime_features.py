"""
Regime-Aware Feature Engineering.

Generates regime-aware features by combining alpha factors with market regimes.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.qlib_factors import QlibAlpha158
from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector


class RegimeFeatureEngineer:
    """
    Generate regime-aware features for ML.
    
    Combines:
    - 158 alpha factors
    - Regime one-hot encoding (4 dummies)
    - Regime stability (days since change)
    - Top factors × regime interactions
    """
    
    def __init__(
        self,
        factor_generator: Optional[QlibAlpha158] = None,
        regime_detector: Optional[WassersteinRegimeDetector] = None
    ):
        """
        Initialize with components.
        
        Args:
            factor_generator: Fitted Qlib Alpha-158 generator
            regime_detector: Fitted regime detector
        """
        self.factor_generator = factor_generator or QlibAlpha158()
        self.regime_detector = regime_detector
        
    def generate_complete_features(
        self,
        df: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        symbol: str = None,
        include_interactions: bool = True,
        top_n_interactions: int = 20
    ) -> pd.DataFrame:
        """
        Generate full feature set.
        
        Output:
        - 158 alpha factors
        - 4 regime dummies (bull, bear, highvol, crash)
        - 2 regime stability features
        - (Optional) 80 interaction features (20 factors × 4 regimes)
        
        Total: ~244 features
        
        Args:
            df: OHLCV DataFrame
            regime_labels: Regime labels for each date (optional)
            symbol: Stock symbol
            include_interactions: Whether to include interaction features
            top_n_interactions: Number of top factors to use for interactions
            
        Returns:
            DataFrame with complete feature set
        """
        logger.info(f"Generating regime-aware features for {symbol or 'data'}...")
        
        # Generate alpha factors
        factors = self.factor_generator.generate_all_factors(df, symbol=symbol)
        
        # Add regime features if available
        if regime_labels is not None or self.regime_detector is not None:
            if regime_labels is None:
                # Use detector to predict regimes
                regime_labels = pd.Series(
                    self.regime_detector.predict(df),
                    index=df.index
                )
            
            # Add regime dummies
            regime_dummies = pd.get_dummies(
                regime_labels,
                prefix='regime',
                dtype=float
            )
            
            factors = pd.concat([factors, regime_dummies], axis=1)
            
            # Add regime stability features
            stability = self.add_regime_stability(regime_labels)
            factors = pd.concat([factors, stability], axis=1)
            
            # Add interactions if requested
            if include_interactions:
                interactions = self._create_regime_interactions(
                    factors,
                    regime_dummies,
                    top_n=top_n_interactions
                )
                factors = pd.concat([factors, interactions], axis=1)
        
        logger.info(f"✅ Generated {len(factors.columns)} total features")
        
        return factors
    
    def add_regime_stability(
        self,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate:
        - days_in_regime: Days since last regime change
        - regime_persistence: Expected duration based on history
        
        Args:
            regime_labels: Series with regime labels
            
        Returns:
            DataFrame with stability features
        """
        stability = pd.DataFrame(index=regime_labels.index)
        
        # Days in current regime
        days_in_regime = []
        current_regime = None
        days_count = 0
        
        for regime in regime_labels:
            if regime == current_regime:
                days_count += 1
            else:
                days_count = 1
                current_regime = regime
            
            days_in_regime.append(days_count)
        
        stability['days_in_regime'] = days_in_regime
        
        # Regime persistence (average duration of this regime historically)
        regime_durations = {}
        
        for regime in regime_labels.unique():
            regime_mask = regime_labels == regime
            
            # Count consecutive days in this regime
            durations = []
            current_duration = 0
            
            for is_regime in regime_mask:
                if is_regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            if durations:
                regime_durations[regime] = np.mean(durations)
            else:
                regime_durations[regime] = 1
        
        stability['regime_persistence'] = regime_labels.map(regime_durations)
        
        return stability
    
    def _create_regime_interactions(
        self,
        factors: pd.DataFrame,
        regime_dummies: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Create interaction features between top factors and regimes.
        
        Args:
            factors: Factor DataFrame
            regime_dummies: Regime dummy variables
            top_n: Number of top factors to use
            
        Returns:
            DataFrame with interaction features
        """
        interactions = pd.DataFrame(index=factors.index)
        
        # Get top N factors (by variance as proxy for importance)
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        
        if len(factor_cols) > top_n:
            factor_vars = factors[factor_cols].var().sort_values(ascending=False)
            top_factors = factor_vars.head(top_n).index.tolist()
        else:
            top_factors = factor_cols[:top_n]
        
        # Create interactions
        for factor in top_factors:
            for regime_col in regime_dummies.columns:
                interaction_name = f"interaction_{factor}_{regime_col}"
                interactions[interaction_name] = factors[factor] * regime_dummies[regime_col]
        
        logger.info(f"Created {len(interactions.columns)} interaction features")
        
        return interactions
    
    def get_feature_importance_by_regime(
        self,
        factors: pd.DataFrame,
        returns: pd.Series,
        regime_labels: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate feature importance separately for each regime.
        
        Args:
            factors: Factor DataFrame
            returns: Target returns
            regime_labels: Regime labels
            
        Returns:
            DataFrame with importance by regime
        """
        from scipy.stats import spearmanr
        
        importance = []
        
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        
        for regime in regime_labels.unique():
            regime_mask = regime_labels == regime
            
            for factor in factor_cols[:50]:  # Limit for speed
                # Calculate correlation in this regime
                factor_vals = factors.loc[regime_mask, factor]
                returns_vals = returns[regime_mask]
                
                valid_mask = factor_vals.notna() & returns_vals.notna()
                
                if valid_mask.sum() > 10:
                    corr, pval = spearmanr(
                        factor_vals[valid_mask],
                        returns_vals[valid_mask]
                    )
                    
                    importance.append({
                        'regime': regime,
                        'factor': factor,
                        'correlation': corr,
                        'pvalue': pval,
                        'abs_corr': abs(corr)
                    })
        
        importance_df = pd.DataFrame(importance)
        
        return importance_df

