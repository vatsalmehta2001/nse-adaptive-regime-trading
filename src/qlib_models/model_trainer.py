"""
Qlib Model Trainer for Alpha Factor-Based Predictions.

Trains machine learning models on Alpha-158 factors to predict forward returns.
Supports LightGBM and XGBoost with time-series aware train/validation/test splits.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class QlibModelTrainer:
    """
    Train ML models on Qlib Alpha-158 factors.
    
    Models supported:
    - LightGBM (gradient boosting)
    - XGBoost (extreme gradient boosting)
    
    Features:
    - Time-series aware train/validation/test split
    - Hyperparameter defaults optimized for financial data
    - Early stopping to prevent overfitting
    - Feature importance analysis
    - Model serialization with metadata
    - Regime-specific model training
    """
    
    LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'verbose': -1
    }
    
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'verbosity': 0
    }
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: 'lightgbm' or 'xgboost'
            config: Model hyperparameters (uses defaults if None)
        
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in ['lightgbm', 'xgboost']:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        
        if config is None:
            self.config = (
                self.LIGHTGBM_PARAMS.copy() 
                if model_type == 'lightgbm' 
                else self.XGBOOST_PARAMS.copy()
            )
        else:
            self.config = config
        
        logger.info(f"Initialized {model_type} trainer with config: {self.config}")
    
    def prepare_data(
        self,
        factors: pd.DataFrame,
        forward_horizon: int = 5,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare train/validation/test splits with time-series integrity.
        
        CRITICAL: 
        - Split chronologically (no shuffle)
        - Calculate forward returns AFTER splitting
        - Remove NaN carefully
        
        Args:
            factors: DataFrame with 158 factors + metadata (multi-index: symbol, date)
            forward_horizon: Days ahead to predict
            train_ratio: Proportion for training (default: 0.6)
            valid_ratio: Proportion for validation (default: 0.2)
            
        Returns:
            (X_train, y_train, X_valid, y_valid, X_test, y_test)
        
        Raises:
            ValueError: If ratios don't sum to <= 1.0
        """
        if train_ratio + valid_ratio >= 1.0:
            raise ValueError("train_ratio + valid_ratio must be < 1.0")
        
        logger.info(f"Preparing data with {forward_horizon}-day forward horizon...")
        
        # Ensure data is sorted chronologically
        if isinstance(factors.index, pd.MultiIndex):
            # Multi-index (symbol, date)
            factors = factors.sort_index(level=['symbol', 'date'])
            has_multi_index = True
        else:
            # Single index (date)
            factors = factors.sort_index()
            has_multi_index = False
        
        # Get factor columns (exclude metadata)
        factor_cols = [c for c in factors.columns if c.startswith('factor_')]
        if 'close' in factors.columns:
            price_col = 'close'
        elif 'price' in factors.columns:
            price_col = 'price'
        else:
            raise ValueError("DataFrame must contain 'close' or 'price' column")
        
        # Calculate forward returns (CRITICAL: done before split to maintain order)
        logger.info("Calculating forward returns...")
        
        if has_multi_index:
            # Calculate per-symbol forward returns
            forward_returns = []
            for symbol in factors.index.get_level_values('symbol').unique():
                symbol_data = factors.loc[symbol]
                symbol_prices = symbol_data[price_col]
                
                # Forward return calculation
                fwd_ret = symbol_prices.pct_change(periods=forward_horizon).shift(-forward_horizon)
                fwd_ret.index = pd.MultiIndex.from_product(
                    [[symbol], fwd_ret.index],
                    names=['symbol', 'date']
                )
                forward_returns.append(fwd_ret)
            
            forward_returns = pd.concat(forward_returns).sort_index()
        else:
            # Simple case: single time series
            forward_returns = factors[price_col].pct_change(
                periods=forward_horizon
            ).shift(-forward_horizon)
        
        # Create feature matrix and target
        X = factors[factor_cols].copy()
        y = forward_returns.copy()
        
        # Remove NaN (forward returns will have NaN at end)
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Total samples after NaN removal: {len(X)}")
        
        # CRITICAL: Chronological split (no shuffle)
        n_total = len(X)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        
        X_valid = X.iloc[n_train:n_train + n_valid]
        y_valid = y.iloc[n_train:n_train + n_valid]
        
        X_test = X.iloc[n_train + n_valid:]
        y_test = y.iloc[n_train + n_valid:]
        
        logger.info(f"Train samples: {len(X_train)} ({train_ratio*100:.1f}%)")
        logger.info(f"Valid samples: {len(X_valid)} ({valid_ratio*100:.1f}%)")
        logger.info(f"Test samples: {len(X_test)} ({(1-train_ratio-valid_ratio)*100:.1f}%)")
        
        # Validation: ensure no date overlap
        if has_multi_index:
            train_dates = X_train.index.get_level_values('date')
            valid_dates = X_valid.index.get_level_values('date')
            test_dates = X_test.index.get_level_values('date')
            
            assert train_dates.max() <= valid_dates.min(), "Train/Valid date overlap detected"
            assert valid_dates.max() <= test_dates.min(), "Valid/Test date overlap detected"
            
            logger.info(f"Train period: {train_dates.min()} to {train_dates.max()}")
            logger.info(f"Valid period: {valid_dates.min()} to {valid_dates.max()}")
            logger.info(f"Test period: {test_dates.min()} to {test_dates.max()}")
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        early_stopping_rounds: int = 50
    ) -> Any:
        """
        Train model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            early_stopping_rounds: Patience for early stopping
            
        Returns:
            Trained model object
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'lightgbm':
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            
            # Train with early stopping
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
            
            model = lgb.train(
                self.config,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
            
            logger.info(f"Best iteration: {model.best_iteration}")
        
        elif self.model_type == 'xgboost':
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            
            # Train with early stopping
            evals = [(dtrain, 'train'), (dvalid, 'valid')]
            
            model = xgb.train(
                self.config,
                dtrain,
                num_boost_round=self.config.get('n_estimators', 1000),
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=100
            )
            
            logger.info(f"Best iteration: {model.best_iteration}")
        
        logger.info("Training complete")
        return model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: Any
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Metrics:
        - MSE (Mean Squared Error)
        - MAE (Mean Absolute Error)
        - IC (Information Coefficient - Spearman)
        - Rank IC
        - R2 Score
        - Direction accuracy (sign match percentage)
        
        Args:
            X_test: Test features
            y_test: Test targets
            model: Trained model
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Generate predictions
        if self.model_type == 'lightgbm':
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Information Coefficient (Spearman correlation)
        ic, _ = spearmanr(y_test, y_pred)
        
        # Rank IC (correlation of ranks)
        y_test_rank = pd.Series(y_test).rank(pct=True)
        y_pred_rank = pd.Series(y_pred).rank(pct=True)
        rank_ic, _ = spearmanr(y_test_rank, y_pred_rank)
        
        # Direction accuracy (sign match)
        direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'ic': float(ic),
            'rank_ic': float(rank_ic),
            'direction_accuracy': float(direction_accuracy)
        }
        
        logger.info(f"Test Metrics: IC={ic:.4f}, R2={r2:.4f}, Direction={direction_accuracy:.2%}")
        
        return metrics
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with [feature, importance, rank]
        """
        if self.model_type == 'lightgbm':
            importance = model.feature_importance(importance_type='gain')
        elif self.model_type == 'xgboost':
            importance = model.get_score(importance_type='gain')
            # XGBoost returns dict, convert to array
            importance_dict = importance
            importance = np.array([
                importance_dict.get(f'f{i}', 0) 
                for i in range(len(feature_names))
            ])
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.head(top_n)
    
    def train_regime_models(
        self,
        factors: pd.DataFrame,
        regime_labels: pd.Series,
        forward_horizon: int = 5
    ) -> Dict[int, Any]:
        """
        Train separate model for each regime.
        
        Args:
            factors: DataFrame with factors and metadata
            regime_labels: Series with regime labels per date
            forward_horizon: Days ahead to predict
            
        Returns:
            Dictionary mapping regime_id -> trained_model
        """
        logger.info("Training regime-specific models...")
        
        # Ensure regime labels are aligned with factors
        if isinstance(factors.index, pd.MultiIndex):
            # Join regime labels by date
            factors_with_regime = factors.copy()
            factors_with_regime = factors_with_regime.reset_index()
            factors_with_regime = factors_with_regime.merge(
                regime_labels.to_frame('regime'),
                left_on='date',
                right_index=True,
                how='inner'
            )
            factors_with_regime = factors_with_regime.set_index(['symbol', 'date'])
        else:
            factors_with_regime = factors.copy()
            factors_with_regime['regime'] = regime_labels
        
        regime_models = {}
        unique_regimes = factors_with_regime['regime'].dropna().unique()
        
        for regime in sorted(unique_regimes):
            logger.info(f"\nTraining model for regime {regime}...")
            
            # Filter data for this regime
            regime_data = factors_with_regime[
                factors_with_regime['regime'] == regime
            ].copy()
            
            # Remove regime column before training
            if 'regime' in regime_data.columns:
                regime_data = regime_data.drop(columns=['regime'])
            
            # Prepare data
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.prepare_data(
                regime_data,
                forward_horizon=forward_horizon,
                train_ratio=0.6,
                valid_ratio=0.2
            )
            
            # Train model
            model = self.train(X_train, y_train, X_valid, y_valid)
            
            # Evaluate
            metrics = self.evaluate(X_test, y_test, model)
            logger.info(f"Regime {regime} IC: {metrics['ic']:.4f}")
            
            regime_models[int(regime)] = {
                'model': model,
                'metrics': metrics
            }
        
        logger.info(f"\nTrained {len(regime_models)} regime-specific models")
        return regime_models
    
    def save_model(
        self,
        model: Any,
        path: str,
        metadata: Dict[str, Any]
    ):
        """
        Save model with metadata.
        
        Metadata includes:
        - Training date
        - Hyperparameters
        - Performance metrics
        - Feature names
        - Training data period
        
        Args:
            model: Trained model
            path: Save path
            metadata: Metadata dictionary
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_type'] = self.model_type
        metadata['config'] = self.config
        
        # Save model and metadata
        save_dict = {
            'model': model,
            'metadata': metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model and metadata.
        
        Args:
            path: Path to saved model
            
        Returns:
            (model, metadata)
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        model = save_dict['model']
        metadata = save_dict['metadata']
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Trained at: {metadata.get('saved_at')}")
        
        return model, metadata

