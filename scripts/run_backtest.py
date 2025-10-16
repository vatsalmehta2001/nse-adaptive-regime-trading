"""
Backtesting Pipeline.

Complete pipeline for backtesting trained models on historical data.

Usage:
    python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31
    python scripts/run_backtest.py --regime-adaptive --rebalance weekly
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.feature_engineering.feature_store import FeatureStore
from src.qlib_models.signal_generator import AlphaSignalGenerator
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest on trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model (or directory for regime models)'
    )

    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='NIFTY50',
        help='Symbol list or preset'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=1000000,
        help='Initial capital in INR'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0005,
        help='Slippage rate (0.0005 = 0.05%%)'
    )

    parser.add_argument(
        '--rebalance',
        type=str,
        choices=['daily', 'weekly', 'monthly'],
        default='daily',
        help='Rebalancing frequency'
    )

    parser.add_argument(
        '--regime-adaptive',
        action='store_true',
        help='Use regime-adaptive strategy'
    )

    parser.add_argument(
        '--long-pct',
        type=float,
        default=0.2,
        help='Top percentile for long positions'
    )

    parser.add_argument(
        '--short-pct',
        type=float,
        default=0.2,
        help='Bottom percentile for short positions'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for results'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/trading_db.duckdb',
        help='Path to DuckDB database'
    )

    return parser.parse_args()


def get_symbol_list(symbol_arg: str) -> List[str]:
    """Get list of symbols from argument."""
    presets = {
        'NIFTY50': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
                   'ICICIBANK', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC'],
        'NIFTY100': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR'],
    }

    if symbol_arg in presets:
        return presets[symbol_arg]
    else:
        return [s.strip() for s in symbol_arg.split(',')]


def load_models(model_path: Optional[str], regime_adaptive: bool) -> Dict[str, any]:
    """
    Load trained models.

    Args:
        model_path: Path to model or directory
        regime_adaptive: Whether loading regime models

    Returns:
        Dictionary of models
    """
    if model_path is None:
        # Find latest model
        models_dir = Path('models')
        if regime_adaptive:
            model_files = list(models_dir.glob('*_regime_*.pkl'))
        else:
            model_files = list(models_dir.glob('lightgbm_*.pkl'))

        if not model_files:
            raise FileNotFoundError("No models found in models directory")

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model: {model_path}")

    models = {}

    if regime_adaptive:
        # Load multiple regime models
        model_dir = Path(model_path) if Path(model_path).is_dir() else Path(model_path).parent
        regime_files = list(model_dir.glob('*_regime_*.pkl'))

        for regime_file in regime_files:
            with open(regime_file, 'rb') as f:
                model_dict = pickle.load(f)
                regime_id = model_dict['metadata'].get('regime', 0)
                models[f'regime_{regime_id}'] = model_dict
                logger.info(f"Loaded regime {regime_id} model from {regime_file}")
    else:
        # Load single model
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            models['default'] = model_dict
            logger.info(f"Loaded model from {model_path}")

    return models


def run_simple_backtest(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    models: Dict[str, any],
    args
) -> pd.DataFrame:
    """
    Run simple (non-regime-adaptive) backtest.

    Args:
        factors: Factor data
        prices: Price data
        models: Loaded models
        args: Command line arguments

    Returns:
        Backtest results DataFrame
    """
    logger.info("\nRunning simple backtest...")

    # Generate signals
    signal_generator = AlphaSignalGenerator(models=models)

    predictions = signal_generator.generate_predictions(factors, model_name='default')

    signals = signal_generator.generate_long_short_signals(
        predictions,
        long_pct=args.long_pct,
        short_pct=args.short_pct
    )

    logger.info(f"Generated {len(signals)} signals")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage
    )

    results = engine.run_backtest(
        signals=signals,
        prices=prices,
        rebalance_freq=args.rebalance
    )

    return results


def run_regime_adaptive_backtest(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    regime_labels: pd.Series,
    models: Dict[str, any],
    args
) -> pd.DataFrame:
    """
    Run regime-adaptive backtest.

    Args:
        factors: Factor data
        prices: Price data
        regime_labels: Regime labels
        models: Loaded regime models
        args: Command line arguments

    Returns:
        Backtest results DataFrame
    """
    logger.info("\nRunning regime-adaptive backtest...")

    # Create regime-adaptive strategy
    strategy = RegimeAdaptiveStrategy(
        regime_models=models,
        regime_configs=None  # Use defaults
    )

    # Generate signals for historical period
    signals = strategy.generate_adaptive_signals_history(
        factors=factors,
        regime_history=regime_labels
    )

    logger.info(f"Generated {len(signals)} signals across regimes")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage
    )

    results = engine.run_backtest(
        signals=signals,
        prices=prices,
        regime_labels=regime_labels,
        rebalance_freq=args.rebalance
    )

    return results


def main():
    """Main backtesting pipeline."""
    args = parse_args()

    # Setup logging
    logger.info(f"\n{'='*80}")
    logger.info("BACKTESTING PIPELINE")
    logger.info(f"{'='*80}\n")
    logger.info(f"Configuration:")
    logger.info(f"  Period:           {args.start} to {args.end}")
    logger.info(f"  Symbols:          {args.symbols}")
    logger.info(f"  Initial capital:  {args.initial_capital:,.0f} INR")
    logger.info(f"  Commission:       {args.commission:.3%}")
    logger.info(f"  Slippage:         {args.slippage:.3%}")
    logger.info(f"  Rebalance:        {args.rebalance}")
    logger.info(f"  Regime adaptive:  {args.regime_adaptive}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get symbols
    symbols = get_symbol_list(args.symbols)
    logger.info(f"Backtesting {len(symbols)} symbols")

    # Connect to feature store
    logger.info("\nConnecting to feature store...")
    store = FeatureStore(db_path=args.db_path)

    # Load factors
    logger.info("Loading factors...")
    factors = store.get_factors(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        include_regimes=args.regime_adaptive
    )

    if factors.empty:
        logger.error("No factor data found for backtest period")
        return

    logger.info(f"Loaded {len(factors)} factor rows")

    # Load actual prices from OHLCV table (factor_001 contains returns, not prices!)
    logger.info("Loading actual prices from OHLCV table...")

    import duckdb
    conn = duckdb.connect(args.db_path)

    # Get unique dates and symbols
    if isinstance(factors.index, pd.MultiIndex):
        symbols_list = factors.index.get_level_values('symbol').unique().tolist()
        dates_list = factors.index.get_level_values('date').unique().tolist()
    else:
        # This shouldn't happen for backtesting
        logger.error("Expected multi-index factors for backtesting")
        return

    # Query OHLCV for actual prices
    symbols_str = ','.join([f"'{s}'" for s in symbols_list])
    prices_query = f"""
        SELECT date, symbol, close
        FROM ohlcv
        WHERE symbol IN ({symbols_str})
          AND date >= '{args.start}'
          AND date <= '{args.end}'
        ORDER BY date, symbol
    """

    prices_df = conn.execute(prices_query).fetchdf()
    conn.close()

    if prices_df.empty:
        logger.error("No price data found in OHLCV table for backtest period")
        return

    logger.info(f"Loaded {len(prices_df)} price rows from OHLCV table")

    # Format prices for backtest engine
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices = prices_df

    # Load models
    logger.info("\nLoading models...")
    models = load_models(args.model_path, args.regime_adaptive)

    # Run backtest
    if args.regime_adaptive:
        if 'regime_label' not in factors.columns:
            logger.error("No regime labels found. Run with --regime-adaptive requires regime detection.")
            return

        regime_labels = factors['regime_label']

        results = run_regime_adaptive_backtest(
            factors, prices, regime_labels, models, args
        )
    else:
        results = run_simple_backtest(factors, prices, models, args)

    # Analyze performance
    logger.info("\nAnalyzing performance...")
    analyzer = PerformanceAnalyzer(risk_free_rate=0.05)

    regime_labels = factors['regime_label'] if 'regime_label' in factors.columns else None

    report = analyzer.generate_report(
        returns=results['daily_return'],
        regime_labels=regime_labels
    )

    # Print summary
    analyzer.print_summary(report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"backtest_results_{timestamp}.csv"
    results.to_csv(results_path)
    logger.info(f"\nBacktest results saved to {results_path}")

    # Save performance report
    import json
    report_path = output_dir / f"performance_report_{timestamp}.json"

    # Convert DataFrame to dict for JSON serialization
    report_copy = report.copy()
    if 'regime_performance' in report_copy and isinstance(report_copy['regime_performance'], pd.DataFrame):
        report_copy['regime_performance'] = report_copy['regime_performance'].to_dict(orient='records')

    with open(report_path, 'w') as f:
        json.dump(report_copy, f, indent=2, default=str)

    logger.info(f"Performance report saved to {report_path}")

    # Close feature store
    store.close()

    logger.info(f"\n{'='*80}")
    logger.info("BACKTESTING COMPLETE")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
