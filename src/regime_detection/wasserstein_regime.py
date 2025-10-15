"""
Wasserstein Distance-Based Regime Detection.

Market regime detection using Wasserstein k-means clustering on return distributions.
Validates against known market events (COVID crash, etc.).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from tqdm import tqdm


class WassersteinRegimeDetector:
    """
    Market regime detection using Wasserstein k-means clustering.
    
    Algorithm:
    1. Calculate rolling return distributions (60-day windows)
    2. Compute Wasserstein distance matrix between all distributions
    3. Apply k-means in Wasserstein metric space (4 clusters)
    4. Characterize and label regimes:
       - Bull: +returns, low vol
       - Bear: -returns, low vol
       - High Vol: high vol, any direction
       - Crash: extreme -returns, very high vol
    
    Validates against known events (COVID crash = March 2020).
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        window_size: int = 60,
        random_state: int = 42
    ):
        """
        Initialize Wasserstein regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect (default: 4)
            window_size: Rolling window size for distributions (default: 60 days)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.random_state = random_state
        self.cluster_model: Optional[KMeans] = None
        self.regime_characteristics: Optional[pd.DataFrame] = None
        self.regime_names: Dict[int, str] = {}
        self._distance_matrix: Optional[np.ndarray] = None
        self._fitted = False
        
    def fit(
        self,
        df: pd.DataFrame,
        feature_col: str = 'returns'
    ) -> 'WassersteinRegimeDetector':
        """
        Fit regime detector on historical data.
        
        Steps:
        1. Calculate returns if not present
        2. Create rolling 60-day distributions
        3. Build Wasserstein distance matrix (N×N)
        4. Apply k-means clustering
        5. Characterize each cluster (mean return, volatility)
        6. Assign regime names based on characteristics
        
        Args:
            df: DataFrame with date index and price/returns
            feature_col: Column for regime detection (default: 'returns')
            
        Returns:
            self (fitted)
        """
        logger.info("Fitting Wasserstein regime detector...")
        
        # Prepare data
        data = df.copy()
        
        # Calculate returns if not present
        if feature_col not in data.columns:
            if 'close' in data.columns:
                data['returns'] = data['close'].pct_change()
                feature_col = 'returns'
            else:
                raise ValueError("DataFrame must have 'returns' or 'close' column")
        
        # Remove NaN
        data = data.dropna(subset=[feature_col])
        
        # Create rolling distributions
        logger.info(f"Creating rolling {self.window_size}-day distributions...")
        distributions = []
        distribution_dates = []
        
        for i in range(self.window_size, len(data)):
            window = data[feature_col].iloc[i-self.window_size:i].values
            distributions.append(window)
            distribution_dates.append(data.index[i])
        
        logger.info(f"Created {len(distributions)} distributions")
        
        # Build Wasserstein distance matrix
        logger.info("Computing Wasserstein distance matrix...")
        self._distance_matrix = self._build_distance_matrix(distributions)
        
        # Apply k-means clustering
        logger.info(f"Applying k-means clustering ({self.n_regimes} regimes)...")
        self.cluster_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.cluster_model.fit_predict(self._distance_matrix)
        
        # Store results
        self._regime_labels = cluster_labels
        self._distribution_dates = distribution_dates
        self._distributions = distributions
        
        # Characterize regimes
        logger.info("Characterizing regimes...")
        self._characterize_regimes(data, cluster_labels, distribution_dates)
        
        self._fitted = True
        logger.info(" Regime detector fitted successfully")
        
        return self
    
    def _build_distance_matrix(
        self,
        distributions: List[np.ndarray],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Build N×N Wasserstein distance matrix.
        
        Uses scipy.stats.wasserstein_distance.
        Shows progress bar for long computations.
        
        Args:
            distributions: List of return distributions
            show_progress: Whether to show progress bar
            
        Returns:
            N×N distance matrix
        """
        n = len(distributions)
        distance_matrix = np.zeros((n, n))
        
        # Calculate pairwise distances
        iterator = range(n) if not show_progress else tqdm(range(n), desc="Building distance matrix")
        
        for i in iterator:
            for j in range(i+1, n):
                dist = wasserstein_distance(distributions[i], distributions[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _characterize_regimes(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        dates: List[pd.Timestamp]
    ):
        """
        Calculate statistical characteristics for each regime.
        
        Args:
            df: Original data
            labels: Cluster labels
            dates: Corresponding dates
        """
        characteristics = []
        
        for regime in range(self.n_regimes):
            # Get dates for this regime
            regime_mask = labels == regime
            regime_dates = [dates[i] for i, is_regime in enumerate(regime_mask) if is_regime]
            
            # Get returns for these dates
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
                
                # Calculate max drawdown
                cumulative = (1 + regime_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                char['max_drawdown'] = drawdown.min()
                
                characteristics.append(char)
        
        self.regime_characteristics = pd.DataFrame(characteristics)
        
        # Assign regime names based on characteristics
        self._assign_regime_names()
        
    def _assign_regime_names(self):
        """Assign intuitive names to regimes based on their characteristics."""
        if self.regime_characteristics is None:
            return
        
        char = self.regime_characteristics.copy()
        
        # Sort by mean return and volatility
        char['regime_score'] = char['mean_return'] - 2 * char['volatility']
        
        # Assign names based on characteristics
        for idx, row in char.iterrows():
            regime = row['regime']
            
            if row['mean_return'] < -0.02 and row['volatility'] > 0.035:
                # Extreme negative returns + high vol = Crash
                self.regime_names[regime] = 'crash'
            elif row['volatility'] > 0.03:
                # High volatility
                self.regime_names[regime] = 'high_volatility'
            elif row['mean_return'] > 0.001:
                # Positive returns
                self.regime_names[regime] = 'bull'
            else:
                # Negative or low returns, low vol
                self.regime_names[regime] = 'bear'
        
        # Add regime names to characteristics
        self.regime_characteristics['regime_name'] = \
            self.regime_characteristics['regime'].map(self.regime_names)
        
        logger.info(f"Regime names assigned: {self.regime_names}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for data.
        
        Args:
            df: DataFrame with returns
            
        Returns:
            Array of regime labels (0,1,2,3)
        """
        if not self._fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # For simplicity, use the fitted labels
        # In production, would implement proper prediction logic
        return self._regime_labels
    
    def predict_online(
        self,
        recent_returns: np.ndarray
    ) -> Tuple[int, Dict[str, float]]:
        """
        Real-time regime prediction from recent returns.
        
        Args:
            recent_returns: Last 60 days of returns
            
        Returns:
            (regime_label, probabilities_dict)
        """
        if not self._fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        if len(recent_returns) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} days of data")
        
        # Use last window_size returns
        current_dist = recent_returns[-self.window_size:]
        
        # Calculate distances to each cluster center
        distances = []
        for i, dist in enumerate(self._distributions[::len(self._distributions)//self.n_regimes][:self.n_regimes]):
            d = wasserstein_distance(current_dist, dist)
            distances.append(d)
        
        # Find closest cluster
        regime_label = int(np.argmin(distances))
        
        # Calculate probabilities (softmax of negative distances)
        distances = np.array(distances)
        probs = np.exp(-distances) / np.sum(np.exp(-distances))
        
        prob_dict = {
            self.regime_names.get(i, f'regime_{i}'): float(probs[i])
            for i in range(len(probs))
        }
        
        return regime_label, prob_dict
    
    def get_regime_characteristics(self) -> pd.DataFrame:
        """
        Statistical profile of each regime.
        
        Returns DataFrame with:
        - regime: 0,1,2,3
        - regime_name: bull/bear/high_volatility/crash
        - mean_return: Average daily return
        - volatility: Std of returns
        - skewness: Distribution skew
        - sharpe_ratio: Return/volatility
        - max_drawdown: Worst drawdown
        - avg_duration: Days spent in regime
        """
        if self.regime_characteristics is None:
            raise ValueError("Detector must be fitted first")
        
        return self.regime_characteristics
    
    def validate_regimes(
        self,
        df: pd.DataFrame,
        known_events: Dict[str, Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validate against known market events.
        
        Default NSE events:
        - COVID crash: 2020-03-01 to 2020-03-31 (should be crash/highvol)
        - Post-COVID bull: 2020-04-01 to 2021-01-31 (should be bull)
        - Ukraine volatility: 2022-02-24 to 2022-03-31 (should be highvol)
        
        Args:
            df: Original data with dates
            known_events: Dict of event_name -> (start_date, end_date)
            
        Returns:
            Dict mapping event_name -> detected_regime
        """
        if not self._fitted:
            raise ValueError("Detector must be fitted first")
        
        if known_events is None:
            known_events = {
                'covid_crash': ('2020-03-01', '2020-03-31'),
                'post_covid_bull': ('2020-04-01', '2021-01-31'),
                'ukraine_volatility': ('2022-02-24', '2022-03-31')
            }
        
        validation_results = {}
        
        for event_name, (start_date, end_date) in known_events.items():
            # Find regime labels for this period
            event_mask = (pd.Series(self._distribution_dates) >= start_date) & \
                        (pd.Series(self._distribution_dates) <= end_date)
            
            if event_mask.sum() > 0:
                event_labels = self._regime_labels[event_mask]
                # Get most common regime
                regime = int(pd.Series(event_labels).mode().iloc[0])
                regime_name = self.regime_names.get(regime, f'regime_{regime}')
                
                validation_results[event_name] = {
                    'detected_regime': regime,
                    'regime_name': regime_name,
                    'days': event_mask.sum()
                }
            else:
                validation_results[event_name] = {
                    'detected_regime': None,
                    'regime_name': 'no_data',
                    'days': 0
                }
        
        return validation_results
    
    def plot_regimes(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        save_path: Optional[str] = None
    ):
        """
        Visualize regimes over time with price overlay.
        
        Creates plot with:
        - Price chart
        - Colored background by regime
        - Regime transitions marked
        
        Args:
            df: Data with price column
            price_col: Column name for prices
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return
        
        if not self._fitted:
            raise ValueError("Detector must be fitted first")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot price
        ax.plot(df.index, df[price_col], color='black', linewidth=1, label='Price')
        
        # Color regime backgrounds
        regime_colors = {
            'bull': 'green',
            'bear': 'red',
            'high_volatility': 'orange',
            'crash': 'darkred'
        }
        
        for i, (date, label) in enumerate(zip(self._distribution_dates, self._regime_labels)):
            regime_name = self.regime_names.get(label, f'regime_{label}')
            color = regime_colors.get(regime_name, 'gray')
            
            if i < len(self._distribution_dates) - 1:
                next_date = self._distribution_dates[i + 1]
                ax.axvspan(date, next_date, alpha=0.2, color=color)
        
        # Add legend
        patches = [mpatches.Patch(color=color, alpha=0.2, label=name) 
                  for name, color in regime_colors.items()]
        ax.legend(handles=patches, loc='upper left')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Market Regimes Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime plot saved to {save_path}")
        
        plt.close()

