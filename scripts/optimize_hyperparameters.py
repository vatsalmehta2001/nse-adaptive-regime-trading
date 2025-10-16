"""
Hyperparameter Optimization with Rolling Window Cross-Validation.

Implements grid search and rolling window validation for model hyperparameters.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering.feature_store import FeatureStore
from src.qlib_models.model_trainer import QlibModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with rolling window CV"
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK',
        help='Comma-separated symbol list'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['lightgbm', 'xgboost'],
        default='lightgbm',
        help='Model type'
    )

    parser.add_argument(
        '--forward-horizon',
        type=int,
        default=5,
        help='Forward prediction horizon'
    )

    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of rolling window splits'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/optimization',
        help='Output directory for results'
    )

    return parser.parse_args()


def get_lightgbm_param_grid() -> List[Dict]:
    """Get LightGBM hyperparameter grid."""
    return [
        # Conservative (low learning rate, more trees)
        {
            'learning_rate': 0.01,
            'n_estimators': 2000,
            'num_leaves': 15,
            'max_depth': 5
        },
        # Moderate
        {
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'num_leaves': 31,
            'max_depth': -1
        },
        # Aggressive (higher learning rate, fewer trees)
        {
            'learning_rate': 0.1,
            'n_estimators': 500,
            'num_leaves': 63,
            'max_depth': -1
        },
        # Deep trees
        {
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'num_leaves': 63,
            'max_depth': 8
        },
        # Regularized
        {
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 50,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6
        }
    ]


def rolling_window_cv(
    factors: pd.DataFrame,
    forward_horizon: int,
    n_splits: int,
    param_grid: List[Dict],
    model_type: str
) -> pd.DataFrame:
    """
    Perform rolling window cross-validation.

    Args:
        factors: Factor data
        forward_horizon: Days ahead to predict
        n_splits: Number of rolling windows
        param_grid: List of hyperparameter dictionaries
        model_type: Model type

    Returns:
        DataFrame with CV results
    """
    logger.info(f"Running rolling window CV with {n_splits} splits...")

    results = []

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Get data indices (we need to split chronologically)
    if isinstance(factors.index, pd.MultiIndex):
        # Get unique dates
        dates = sorted(factors.index.get_level_values('date').unique())
        date_indices = np.arange(len(dates))
    else:
        date_indices = np.arange(len(factors))

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(date_indices)):
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")

        # Get date ranges for this fold
        if isinstance(factors.index, pd.MultiIndex):
            train_dates = [dates[i] for i in train_idx]
            val_dates = [dates[i] for i in val_idx]

            # Filter factors by dates
            fold_factors = factors[
                factors.index.get_level_values('date').isin(train_dates + val_dates)
            ]
        else:
            fold_factors = factors.iloc[list(train_idx) + list(val_idx)]

        # Try each parameter set
        for param_idx, params in enumerate(param_grid):
            logger.info(f"  Testing params {param_idx + 1}/{len(param_grid)}: {params}")

            # Update default params
            full_params = QlibModelTrainer.LIGHTGBM_PARAMS.copy() if model_type == 'lightgbm' else QlibModelTrainer.XGBOOST_PARAMS.copy()
            full_params.update(params)

            # Train model
            trainer = QlibModelTrainer(model_type=model_type, config=full_params)

            try:
                X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
                    fold_factors,
                    forward_horizon=forward_horizon,
                    train_ratio=len(train_idx) / (len(train_idx) + len(val_idx)),
                    valid_ratio=len(val_idx) / (len(train_idx) + len(val_idx))
                )

                model = trainer.train(X_train, y_train, X_valid, y_valid)
                metrics = trainer.evaluate(X_valid, y_valid, model)

                # Store results
                result = {
                    'fold': fold_idx,
                    'param_set': param_idx,
                    'ic': metrics['ic'],
                    'r2': metrics['r2'],
                    'direction_accuracy': metrics['direction_accuracy'],
                    'mse': metrics['mse']
                }
                result.update(params)
                results.append(result)

                logger.info(f"    IC: {metrics['ic']:.4f}, R2: {metrics['r2']:.4f}, Direction: {metrics['direction_accuracy']:.2%}")

            except Exception as e:
                logger.error(f"    Error with params {params}: {e}")
                continue

    results_df = pd.DataFrame(results)

    # Calculate mean metrics across folds for each param set
    logger.info("\n" + "="*80)
    logger.info("Cross-Validation Results Summary")
    logger.info("="*80)

    for param_idx in results_df['param_set'].unique():
        param_results = results_df[results_df['param_set'] == param_idx]
        mean_ic = param_results['ic'].mean()
        std_ic = param_results['ic'].std()
        mean_direction = param_results['direction_accuracy'].mean()

        logger.info(f"\nParam Set {param_idx}:")
        logger.info(f"  IC: {mean_ic:.4f} ± {std_ic:.4f}")
        logger.info(f"  Direction Accuracy: {mean_direction:.2%}")
        logger.info(f"  Params: {param_grid[param_idx]}")

    # Find best params
    mean_metrics = results_df.groupby('param_set').agg({
        'ic': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'direction_accuracy': ['mean', 'std']
    })

    best_param_idx = mean_metrics[('ic', 'mean')].idxmax()
    best_params = param_grid[best_param_idx]

    logger.info(f"\n{'='*80}")
    logger.info(f"Best Parameters (by mean IC):")
    logger.info(f"  Param Set: {best_param_idx}")
    logger.info(f"  Mean IC: {mean_metrics.loc[best_param_idx, ('ic', 'mean')]:.4f}")
    logger.info(f"  Std IC: {mean_metrics.loc[best_param_idx, ('ic', 'std')]:.4f}")
    logger.info(f"  Params: {best_params}")
    logger.info(f"{'='*80}\n")

    return results_df, best_params, mean_metrics


def main():
    """Main optimization pipeline."""
    args = parse_args()

    logger.info(f"\n{'='*80}")
    logger.info("HYPERPARAMETER OPTIMIZATION WITH ROLLING WINDOW CV")
    logger.info(f"{'='*80}\n")
    logger.info(f"Configuration:")
    logger.info(f"  Symbols:         {args.symbols}")
    logger.info(f"  Model:           {args.model}")
    logger.info(f"  Forward horizon: {args.forward_horizon} days")
    logger.info(f"  CV splits:       {args.n_splits}")
    logger.info(f"  Output dir:      {args.output_dir}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    logger.info(f"Optimizing on {len(symbols)} symbols")

    # Load factors
    logger.info("\nLoading factors...")
    store = FeatureStore()
    factors = store.get_factors(symbols=symbols, include_regimes=False)
    store.close()

    if factors.empty:
        logger.error("No factor data found")
        return

    # Add close column (use factor_001 which is close price)
    if 'close' not in factors.columns:
        factors['close'] = factors['factor_001']

    logger.info(f"Loaded {len(factors)} factor rows")

    # Get parameter grid
    if args.model == 'lightgbm':
        param_grid = get_lightgbm_param_grid()
    else:
        # For XGBoost, use similar grid
        param_grid = get_lightgbm_param_grid()  # Simplified for now

    logger.info(f"Testing {len(param_grid)} parameter sets")

    # Run rolling window CV
    results_df, best_params, mean_metrics = rolling_window_cv(
        factors,
        args.forward_horizon,
        args.n_splits,
        param_grid,
        args.model
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"cv_results_{args.model}_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"CV results saved to {results_path}")

    # Save mean metrics
    metrics_path = output_dir / f"cv_metrics_{args.model}_{timestamp}.csv"
    mean_metrics.to_csv(metrics_path)
    logger.info(f"Mean metrics saved to {metrics_path}")

    # Save best params
    best_params_path = output_dir / f"best_params_{args.model}_{timestamp}.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best parameters saved to {best_params_path}")

    # Train final model with best params
    logger.info("\nTraining final model with best parameters...")

    full_params = (
        QlibModelTrainer.LIGHTGBM_PARAMS.copy()
        if args.model == 'lightgbm'
        else QlibModelTrainer.XGBOOST_PARAMS.copy()
    )
    full_params.update(best_params)

    trainer = QlibModelTrainer(model_type=args.model, config=full_params)
    X_train, y_train, X_valid, y_valid, X_test, y_test = trainer.prepare_data(
        factors,
        forward_horizon=args.forward_horizon
    )

    final_model = trainer.train(X_train, y_train, X_valid, y_valid)
    final_metrics = trainer.evaluate(X_test, y_test, final_model)

    # Save final model
    model_path = output_dir / f"optimized_{args.model}_{timestamp}.pkl"
    trainer.save_model(final_model, str(model_path), {
        'best_params': best_params,
        'cv_metrics': mean_metrics.to_dict(),
        'final_metrics': final_metrics
    })

    logger.info(f"\nFinal Model Performance:")
    logger.info(f"  Test IC:              {final_metrics['ic']:.4f}")
    logger.info(f"  Test R2:              {final_metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy:   {final_metrics['direction_accuracy']:.2%}")
    logger.info(f"  Model saved to:       {model_path}")

    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

