"""
Model Training Pipeline.

Complete pipeline for training ML models on alpha factors.
Supports both single model and regime-adaptive training.

Usage:
    python scripts/train_models.py --symbols NIFTY50 --model lightgbm
    python scripts/train_models.py --regime-adaptive --forward-horizon 10
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.feature_engineering.feature_store import FeatureStore
from src.qlib_models.model_trainer import QlibModelTrainer
from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML models on alpha factors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='NIFTY50',
        help='Symbol list or preset (NIFTY50, NIFTY100, etc.)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['lightgbm', 'xgboost'],
        default='lightgbm',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--forward-horizon',
        type=int,
        default=5,
        help='Forward prediction horizon in days'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Training data ratio'
    )
    
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=0.2,
        help='Validation data ratio'
    )
    
    parser.add_argument(
        '--regime-adaptive',
        action='store_true',
        help='Train separate models for each regime'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/trading_db.duckdb',
        help='Path to DuckDB database'
    )
    
    return parser.parse_args()


def get_symbol_list(symbol_arg: str) -> List[str]:
    """
    Get list of symbols from argument.
    
    Args:
        symbol_arg: Symbol preset or comma-separated list
        
    Returns:
        List of symbols
    """
    # Presets
    presets = {
        'NIFTY50': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 
                   'ICICIBANK', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC'],
        'NIFTY100': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR'],
    }
    
    if symbol_arg in presets:
        return presets[symbol_arg]
    else:
        return [s.strip() for s in symbol_arg.split(',')]


def train_single_model(
    factors: pd.DataFrame,
    model_type: str,
    forward_horizon: int,
    train_ratio: float,
    valid_ratio: float,
    output_dir: Path
):
    """
    Train a single model on all data.
    
    Args:
        factors: Factor data
        model_type: Model type
        forward_horizon: Forward prediction horizon
        train_ratio: Training ratio
        valid_ratio: Validation ratio
        output_dir: Output directory
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_type} model")
    logger.info(f"{'='*80}\n")
    
    # Initialize trainer
    trainer = QlibModelTrainer(model_type=model_type)
    
    # Prepare data
    X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
        factors,
        forward_horizon=forward_horizon,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio
    )
    
    # Train model
    model = trainer.train(X_train, y_train, X_valid, y_valid)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test, model)
    
    # Feature importance
    feature_names = X_train.columns.tolist()
    importance_df = trainer.get_feature_importance(model, feature_names, top_n=50)
    
    logger.info("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['rank']:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{model_type}_{forward_horizon}day_{timestamp}.pkl"
    
    metadata = {
        'model_type': model_type,
        'forward_horizon': forward_horizon,
        'train_samples': len(X_train),
        'valid_samples': len(X_valid),
        'test_samples': len(X_test),
        'metrics': metrics,
        'feature_names': feature_names,
        'top_features': importance_df.head(50).to_dict(orient='records')
    }
    
    trainer.save_model(model, str(model_path), metadata)
    
    # Save feature importance
    importance_path = output_dir / f"feature_importance_{model_type}_{timestamp}.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")
    
    logger.info(f"\nModel Training Summary:")
    logger.info(f"  Test IC:              {metrics['ic']:.4f}")
    logger.info(f"  Test R2:              {metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy:   {metrics['direction_accuracy']:.2%}")
    logger.info(f"  Model saved to:       {model_path}")


def train_regime_models(
    factors: pd.DataFrame,
    regime_labels: pd.Series,
    model_type: str,
    forward_horizon: int,
    output_dir: Path
):
    """
    Train separate models for each regime.
    
    Args:
        factors: Factor data
        regime_labels: Regime labels
        model_type: Model type
        forward_horizon: Forward prediction horizon
        output_dir: Output directory
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training regime-adaptive {model_type} models")
    logger.info(f"{'='*80}\n")
    
    # Initialize trainer
    trainer = QlibModelTrainer(model_type=model_type)
    
    # Train regime models
    regime_models = trainer.train_regime_models(
        factors,
        regime_labels,
        forward_horizon=forward_horizon
    )
    
    # Save each regime model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for regime_id, model_dict in regime_models.items():
        model = model_dict['model']
        metrics = model_dict['metrics']
        
        model_path = output_dir / f"{model_type}_regime_{regime_id}_{forward_horizon}day_{timestamp}.pkl"
        
        metadata = {
            'model_type': model_type,
            'regime': regime_id,
            'forward_horizon': forward_horizon,
            'metrics': metrics
        }
        
        trainer.save_model(model, str(model_path), metadata)
        
        logger.info(f"\nRegime {regime_id} Model:")
        logger.info(f"  IC:     {metrics['ic']:.4f}")
        logger.info(f"  R2:     {metrics['r2']:.4f}")
        logger.info(f"  Saved:  {model_path}")
    
    logger.info(f"\nTrained {len(regime_models)} regime-specific models")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup logging
    logger.info(f"\n{'='*80}")
    logger.info("MODEL TRAINING PIPELINE")
    logger.info(f"{'='*80}\n")
    logger.info(f"Configuration:")
    logger.info(f"  Symbols:          {args.symbols}")
    logger.info(f"  Model:            {args.model}")
    logger.info(f"  Forward horizon:  {args.forward_horizon} days")
    logger.info(f"  Regime adaptive:  {args.regime_adaptive}")
    logger.info(f"  Database:         {args.db_path}")
    logger.info(f"  Output dir:       {args.output_dir}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get symbol list
    symbols = get_symbol_list(args.symbols)
    logger.info(f"Training on {len(symbols)} symbols: {symbols[:5]}...")
    
    # Connect to feature store
    logger.info(f"\nConnecting to feature store...")
    store = FeatureStore(db_path=args.db_path)
    
    # Load factors
    logger.info("Loading factors...")
    factors = store.get_factors(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        include_regimes=args.regime_adaptive
    )
    
    if factors.empty:
        logger.error("No factor data found. Please run factor generation first.")
        return
    
    logger.info(f"Loaded {len(factors)} factor rows")
    
    # Check for price column (factor_001 is close price in Alpha-158)
    if 'close' not in factors.columns and 'factor_001' not in factors.columns:
        logger.error("No 'close' or 'factor_001' price column found. Cannot calculate forward returns.")
        return
    
    # Add close column if not present (use factor_001 which is close price in Alpha-158)
    if 'close' not in factors.columns:
        factors['close'] = factors['factor_001']
        logger.info("Using factor_001 as close price column")
    
    # Train models
    if args.regime_adaptive:
        # Check for regime labels
        if 'regime_label' not in factors.columns:
            logger.error("No regime labels found. Please run regime detection first.")
            return
        
        regime_labels = factors['regime_label']
        
        # Train regime-specific models
        train_regime_models(
            factors,
            regime_labels,
            args.model,
            args.forward_horizon,
            output_dir
        )
    else:
        # Train single model
        train_single_model(
            factors,
            args.model,
            args.forward_horizon,
            args.train_ratio,
            args.valid_ratio,
            output_dir
        )
    
    # Close feature store
    store.close()
    
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

