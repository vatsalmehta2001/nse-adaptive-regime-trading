"""
Regime-Adaptive Trading Strategy.

Strategy that dynamically switches models and parameters based on detected market regime.
Each regime has dedicated model, optimization method, and risk parameters.
"""

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from src.qlib_models.signal_generator import AlphaSignalGenerator
from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector


class RegimeAdaptiveStrategy:
    """
    Strategy that adapts to market regimes.
    
    Each regime has:
    - Dedicated trained model
    - Custom portfolio optimization method
    - Regime-specific risk parameters
    """
    
    # Default configurations for each regime
    REGIME_CONFIGS = {
        0: {  # Bull market
            'name': 'bull',
            'long_pct': 0.3,
            'short_pct': 0.1,
            'max_position': 0.1,
            'rebalance_freq': 'weekly',
            'risk_aversion': 0.5,
            'optimization_method': 'max_sharpe'
        },
        1: {  # Bear market
            'name': 'bear',
            'long_pct': 0.1,
            'short_pct': 0.3,
            'max_position': 0.08,
            'rebalance_freq': 'daily',
            'risk_aversion': 2.0,
            'optimization_method': 'min_variance'
        },
        2: {  # High volatility
            'name': 'high_volatility',
            'long_pct': 0.15,
            'short_pct': 0.15,
            'max_position': 0.05,
            'rebalance_freq': 'daily',
            'risk_aversion': 3.0,
            'optimization_method': 'risk_parity'
        },
        3: {  # Crash
            'name': 'crash',
            'long_pct': 0.05,
            'short_pct': 0.05,
            'max_position': 0.03,
            'rebalance_freq': 'daily',
            'risk_aversion': 5.0,
            'optimization_method': 'min_variance'
        }
    }
    
    def __init__(
        self,
        regime_models: Dict[int, Any],
        regime_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        regime_detector: Optional[WassersteinRegimeDetector] = None
    ):
        """
        Initialize adaptive strategy.
        
        Args:
            regime_models: Dict of regime_id -> model or model_dict
            regime_configs: Dict of regime_id -> config (uses defaults if None)
            regime_detector: Fitted regime detector
        """
        self.regime_models = regime_models
        
        # Use provided configs or defaults
        if regime_configs is None:
            self.regime_configs = self.REGIME_CONFIGS.copy()
        else:
            self.regime_configs = regime_configs
        
        self.regime_detector = regime_detector
        
        # Create signal generator with regime models
        self.signal_generator = AlphaSignalGenerator(
            models=regime_models,
            regime_detector=regime_detector
        )
        
        logger.info(f"Initialized regime-adaptive strategy with {len(regime_models)} regime models")
    
    def generate_signals(
        self,
        factors: pd.DataFrame,
        current_regime: int
    ) -> pd.DataFrame:
        """
        Generate signals for current regime.
        
        Uses regime-specific model and parameters.
        
        Args:
            factors: Factor data
            current_regime: Current regime identifier
            
        Returns:
            DataFrame with signals, weights, and metadata
        """
        logger.info(f"Generating signals for regime {current_regime}...")
        
        # Get regime configuration
        regime_config = self.get_regime_config(current_regime)
        regime_name = regime_config.get('name', f'regime_{current_regime}')
        
        # Determine model to use
        model_name = f"regime_{current_regime}"
        if model_name not in self.regime_models:
            logger.warning(f"No model for regime {current_regime}, using default")
            model_name = "default" if "default" in self.regime_models else list(self.regime_models.keys())[0]
        
        # Generate predictions
        predictions = self.signal_generator.generate_predictions(
            factors,
            model_name=model_name
        )
        
        # Generate long/short signals based on regime parameters
        signals = self.signal_generator.generate_long_short_signals(
            predictions,
            long_pct=regime_config.get('long_pct', 0.2),
            short_pct=regime_config.get('short_pct', 0.2)
        )
        
        # Add regime metadata
        signals['regime'] = current_regime
        signals['regime_name'] = regime_name
        signals['optimization_method'] = regime_config.get('optimization_method', 'mean_variance')
        signals['risk_aversion'] = regime_config.get('risk_aversion', 1.0)
        
        logger.info(f"Regime {current_regime} ({regime_name}): "
                   f"{(signals['signal']==1).sum()} long, {(signals['signal']==-1).sum()} short")
        
        return signals
    
    def get_regime_config(self, regime: int) -> Dict[str, Any]:
        """
        Get configuration for regime.
        
        Args:
            regime: Regime identifier
            
        Returns:
            Configuration dictionary
        """
        if regime in self.regime_configs:
            return self.regime_configs[regime]
        else:
            # Return default balanced config
            logger.warning(f"No config for regime {regime}, using balanced defaults")
            return {
                'name': f'regime_{regime}',
                'long_pct': 0.2,
                'short_pct': 0.2,
                'max_position': 0.1,
                'rebalance_freq': 'daily',
                'risk_aversion': 1.0,
                'optimization_method': 'mean_variance'
            }
    
    def update_regime_config(
        self,
        regime: int,
        config_updates: Dict[str, Any]
    ):
        """
        Update configuration for a specific regime.
        
        Args:
            regime: Regime identifier
            config_updates: Dictionary of config values to update
        """
        if regime not in self.regime_configs:
            self.regime_configs[regime] = self.get_regime_config(regime)
        
        self.regime_configs[regime].update(config_updates)
        
        logger.info(f"Updated config for regime {regime}: {config_updates}")
    
    def generate_adaptive_signals_history(
        self,
        factors: pd.DataFrame,
        regime_history: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals for historical period with regime changes.
        
        Args:
            factors: Factor data with multi-index (symbol, date)
            regime_history: Series with regime per date
            
        Returns:
            DataFrame with signals for all dates
        """
        logger.info("Generating adaptive signals for historical period...")
        
        all_signals = []
        
        # Get unique dates
        if isinstance(factors.index, pd.MultiIndex):
            dates = factors.index.get_level_values('date').unique()
        else:
            dates = factors.index.unique()
        
        for date in dates:
            # Get regime for this date
            if date in regime_history.index:
                current_regime = regime_history[date]
            else:
                logger.warning(f"No regime for {date}, skipping")
                continue
            
            # Get factors for this date
            if isinstance(factors.index, pd.MultiIndex):
                date_factors = factors.xs(date, level='date')
            else:
                date_factors = factors.loc[[date]]
            
            # Generate signals
            signals = self.generate_signals(date_factors, int(current_regime))
            signals['date'] = date
            
            all_signals.append(signals)
        
        # Combine all signals
        if all_signals:
            combined_signals = pd.concat(all_signals, ignore_index=True)
            logger.info(f"Generated signals for {len(dates)} dates across {len(all_signals)} regime changes")
            return combined_signals
        else:
            return pd.DataFrame()
    
    def get_position_limits(self, regime: int) -> Dict[str, float]:
        """
        Get position limits for regime.
        
        Args:
            regime: Regime identifier
            
        Returns:
            Dictionary with position limits
        """
        config = self.get_regime_config(regime)
        
        return {
            'max_position': config.get('max_position', 0.1),
            'max_leverage': config.get('max_leverage', 1.0),
            'min_position': config.get('min_position', 0.0)
        }
    
    def get_rebalance_frequency(self, regime: int) -> str:
        """
        Get rebalancing frequency for regime.
        
        Args:
            regime: Regime identifier
            
        Returns:
            Rebalance frequency string ('daily', 'weekly', 'monthly')
        """
        config = self.get_regime_config(regime)
        return config.get('rebalance_freq', 'daily')
    
    def get_optimization_params(self, regime: int) -> Dict[str, Any]:
        """
        Get portfolio optimization parameters for regime.
        
        Args:
            regime: Regime identifier
            
        Returns:
            Dictionary with optimization parameters
        """
        config = self.get_regime_config(regime)
        
        return {
            'method': config.get('optimization_method', 'mean_variance'),
            'risk_aversion': config.get('risk_aversion', 1.0),
            'constraints': {
                'max_position': config.get('max_position', 0.1),
                'long_only': False  # Allow long-short
            }
        }
    
    def summarize_regime_strategy(self) -> pd.DataFrame:
        """
        Create summary table of regime-specific strategies.
        
        Returns:
            DataFrame with regime configurations
        """
        summary = []
        
        for regime_id, config in self.regime_configs.items():
            summary.append({
                'regime': regime_id,
                'regime_name': config.get('name', f'regime_{regime_id}'),
                'long_pct': config.get('long_pct', 0),
                'short_pct': config.get('short_pct', 0),
                'max_position': config.get('max_position', 0),
                'rebalance_freq': config.get('rebalance_freq', 'daily'),
                'risk_aversion': config.get('risk_aversion', 1.0),
                'optimization_method': config.get('optimization_method', 'mean_variance')
            })
        
        summary_df = pd.DataFrame(summary)
        
        return summary_df

