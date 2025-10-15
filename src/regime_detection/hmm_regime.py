"""
Hidden Markov Model Regime Detection (Gaussian Mixture Alternative).

Since hmmlearn has installation issues, we use Gaussian Mixture Models
from scikit-learn as an alternative for regime detection comparison.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.mixture import GaussianMixture


class HMMRegimeDetector:
    """
    Hidden Markov Model-like regime detection using Gaussian Mixture Models.
    
    Simpler than Wasserstein but useful for validation.
    Uses sklearn.mixture.GaussianMixture as alternative to hmmlearn.
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Gaussian Mixture-based regime detector.
        
        Args:
            n_regimes: Number of regimes/components (default: 4)
            n_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = GaussianMixture(
            n_components=n_regimes,
            max_iter=n_iter,
            random_state=random_state,
            covariance_type='full'
        )
        
        self._fitted = False
        self.regime_characteristics: Optional[pd.DataFrame] = None
        self.regime_names: Dict[int, str] = {}
        
    def fit(
        self,
        df: pd.DataFrame,
        features: List[str] = None
    ) -> 'HMMRegimeDetector':
        """
        Fit Gaussian Mixture Model on returns and volatility.
        
        Args:
            df: DataFrame with date index
            features: Feature columns to use (default: ['returns', 'volatility'])
            
        Returns:
            self (fitted)
        """
        logger.info("Fitting Gaussian Mixture regime detector...")
        
        # Prepare data
        data = df.copy()
        
        # Calculate features if not present
        if 'returns' not in data.columns:
            if 'close' in data.columns:
                data['returns'] = data['close'].pct_change()
            else:
                raise ValueError("DataFrame must have 'returns' or 'close' column")
        
        if 'volatility' not in data.columns:
            data['volatility'] = data['returns'].rolling(20).std()
        
        # Default features
        if features is None:
            features = ['returns', 'volatility']
        
        # Prepare feature matrix
        X = data[features].dropna()
        self._feature_names = features
        self._original_data = data
        
        # Fit model
        logger.info(f"Fitting GMM with {self.n_regimes} components...")
        self.model.fit(X.values)
        
        # Predict regimes
        regime_labels = self.model.predict(X.values)
        self._regime_labels = regime_labels
        self._X = X
        
        # Characterize regimes
        self._characterize_regimes(data, regime_labels, X.index)
        
        self._fitted = True
        logger.info("✅ GMM regime detector fitted successfully")
        
        return self
    
    def _characterize_regimes(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        dates: pd.DatetimeIndex
    ):
        """Calculate statistical characteristics for each regime."""
        characteristics = []
        
        for regime in range(self.n_regimes):
            # Get data for this regime
            regime_mask = labels == regime
            regime_dates = dates[regime_mask]
            regime_returns = df.loc[regime_dates, 'returns'].dropna()
            
            if len(regime_returns) > 0:
                char = {
                    'regime': regime,
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurt(),
                    'sharpe_ratio': regime_returns.mean() / (regime_returns.std() + 1e-10),
                    'avg_duration': len(regime_returns),
                    'frequency': len(regime_returns) / len(labels)
                }
                
                characteristics.append(char)
        
        self.regime_characteristics = pd.DataFrame(characteristics)
        
        # Assign names
        self._assign_regime_names()
    
    def _assign_regime_names(self):
        """Assign intuitive names to regimes."""
        if self.regime_characteristics is None:
            return
        
        char = self.regime_characteristics.copy()
        
        for idx, row in char.iterrows():
            regime = row['regime']
            
            if row['mean_return'] < -0.02 and row['volatility'] > 0.035:
                self.regime_names[regime] = 'crash'
            elif row['volatility'] > 0.03:
                self.regime_names[regime] = 'high_volatility'
            elif row['mean_return'] > 0.001:
                self.regime_names[regime] = 'bull'
            else:
                self.regime_names[regime] = 'bear'
        
        self.regime_characteristics['regime_name'] = \
            self.regime_characteristics['regime'].map(self.regime_names)
        
        logger.info(f"GMM regime names: {self.regime_names}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime sequence using trained model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of regime labels
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = df[self._feature_names].dropna()
        
        # Predict
        labels = self.model.predict(X.values)
        
        return labels
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of shape (n_samples, n_regimes) with probabilities
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = df[self._feature_names].dropna()
        probs = self.model.predict_proba(X.values)
        
        return probs
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probabilities.
        
        Returns:
            DataFrame with transition matrix
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        # Estimate transition matrix from data
        labels = self._regime_labels
        n = self.n_regimes
        
        transition_counts = np.zeros((n, n))
        
        for i in range(len(labels) - 1):
            from_regime = labels[i]
            to_regime = labels[i + 1]
            transition_counts[from_regime, to_regime] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_probs = transition_counts / (row_sums + 1e-10)
        
        # Create DataFrame
        regime_labels = [self.regime_names.get(i, f'regime_{i}') for i in range(n)]
        
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_labels,
            columns=regime_labels
        )
        
        return transition_df
    
    def get_emission_parameters(self) -> pd.DataFrame:
        """
        Get mean/covariance of each regime (Gaussian emission parameters).
        
        Returns:
            DataFrame with emission parameters
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        params = []
        
        for i in range(self.n_regimes):
            regime_name = self.regime_names.get(i, f'regime_{i}')
            mean = self.model.means_[i]
            
            param = {
                'regime': i,
                'regime_name': regime_name,
            }
            
            # Add mean for each feature
            for j, feature in enumerate(self._feature_names):
                param[f'mean_{feature}'] = mean[j]
            
            # Add covariance diagonal (variances)
            for j, feature in enumerate(self._feature_names):
                param[f'var_{feature}'] = self.model.covariances_[i][j, j]
            
            params.append(param)
        
        return pd.DataFrame(params)
    
    def get_regime_characteristics(self) -> pd.DataFrame:
        """Get regime statistical characteristics."""
        if self.regime_characteristics is None:
            raise ValueError("Model must be fitted first")
        
        return self.regime_characteristics

