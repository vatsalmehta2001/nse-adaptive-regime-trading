#!/usr/bin/env python3
"""
Generate Qlib Alpha-158 factors and detect market regimes.

Usage:
    python scripts/generate_factors.py --symbols NIFTY50 --years 2
    python scripts/generate_factors.py --symbols "RELIANCE,TCS" --years 1
    python scripts/generate_factors.py --update
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import DataStorageManager, OpenBBDataFetcher
from src.feature_engineering import (
    QlibAlpha158,
    FeatureStore,
    FactorAnalyzer,
    RegimeFeatureEngineer,
)
from src.regime_detection import WassersteinRegimeDetector, HMMRegimeDetector
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(
    log_level="INFO",
    log_file="logs/factor_generation.log",
    enable_console=True
)
logger = get_logger(__name__)


def generate_factors_pipeline(
    symbols: List[str],
    start_date: str,
    end_date: str,
    detect_regimes: bool = True,
    analyze_factors: bool = True
) -> Dict[str, Any]:
    """
    Complete factor generation pipeline.
    
    Steps:
    1. Load OHLCV from DuckDB
    2. Generate 158 factors per symbol
    3. Store factors in database
    4. Detect market regimes (Wasserstein + HMM)
    5. Store regime labels
    6. Calculate factor IC and correlations
    7. Generate analysis report
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        detect_regimes: Whether to detect market regimes
        analyze_factors: Whether to analyze factors
        
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("FACTOR GENERATION PIPELINE")
    logger.info("=" * 80)
    
    # Initialize components
    storage = DataStorageManager()
    feature_store = FeatureStore()
    factor_generator = QlibAlpha158()
    analyzer = FactorAnalyzer()
    
    # Create schema
    logger.info("Creating feature store schema...")
    feature_store.create_schema()
    
    stats = {
        'symbols_processed': 0,
        'total_factors_stored': 0,
        'regimes_detected': False,
        'avg_ic': None,
        'failed_symbols': []
    }
    
    # Process each symbol
    logger.info(f"Processing {len(symbols)} symbols...")
    
    for symbol in tqdm(symbols, desc="Generating factors"):
        try:
            # Load data
            df = storage.query_ohlcv(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                stats['failed_symbols'].append(symbol)
                continue
            
            # Reset index to get date column
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            
            # Generate factors
            logger.info(f"Generating factors for {symbol}...")
            factors = factor_generator.generate_all_factors(df, symbol=symbol)
            
            # Validate
            if not factor_generator.validate_factor_count(factors):
                logger.error(f"Factor count validation failed for {symbol}")
                stats['failed_symbols'].append(symbol)
                continue
            
            # Store
            rows = feature_store.store_factors(factors, symbol=symbol)
            stats['symbols_processed'] += 1
            stats['total_factors_stored'] += rows
            
            logger.info(f"✅ {symbol}: {rows} rows stored")
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
            stats['failed_symbols'].append(symbol)
            continue
    
    # Regime detection (market-level, not per-symbol)
    if detect_regimes:
        logger.info("=" * 80)
        logger.info("REGIME DETECTION")
        logger.info("=" * 80)
        
        # Try to use NIFTY index, fallback to first symbol
        try:
            nifty_data = storage.query_ohlcv(
                symbols=["^NSEI"],  # NIFTY 50 index
                start_date=start_date,
                end_date=end_date
            )
            
            if nifty_data.empty:
                # Fallback to first available symbol
                nifty_data = storage.query_ohlcv(
                    symbols=[symbols[0]],
                    start_date=start_date,
                    end_date=end_date
                )
                logger.info(f"Using {symbols[0]} for regime detection")
            
            if not nifty_data.empty:
                # Reset index
                if isinstance(nifty_data.index, pd.MultiIndex):
                    nifty_data = nifty_data.reset_index(level=0, drop=True)
                
                # Wasserstein detector
                logger.info("Fitting Wasserstein regime detector...")
                wasserstein_detector = WassersteinRegimeDetector(
                    n_regimes=4,
                    window_size=60
                )
                wasserstein_detector.fit(nifty_data)
                regime_labels = wasserstein_detector.predict(nifty_data)
                
                # Get characteristics
                char = wasserstein_detector.get_regime_characteristics()
                logger.info(f"Regime characteristics:\n{char}")
                
                # Store regime labels
                regime_df = pd.DataFrame({
                    'date': nifty_data.index,
                    'regime_label': regime_labels,
                    'regime_name': [wasserstein_detector.regime_names.get(l, f'regime_{l}') for l in regime_labels]
                })
                feature_store.store_regime_labels(regime_df, method='wasserstein')
                
                # Validate
                validation = wasserstein_detector.validate_regimes(nifty_data)
                logger.info(f"Regime validation: {validation}")
                
                stats['regimes_detected'] = True
                stats['regime_validation'] = validation
                
        except Exception as e:
            logger.error(f"Regime detection failed: {e}", exc_info=True)
    
    # Factor analysis
    if analyze_factors and stats['symbols_processed'] > 0:
        logger.info("=" * 80)
        logger.info("FACTOR ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Get factors for first symbol
            test_symbol = [s for s in symbols if s not in stats['failed_symbols']][0]
            factors = feature_store.get_factors(
                symbols=[test_symbol],
                include_regimes=False
            )
            
            if not factors.empty:
                # Calculate forward returns
                df = storage.query_ohlcv(symbols=[test_symbol])
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)
                
                forward_returns = df['close'].pct_change().shift(-5)  # 5-day forward
                
                # Align indices
                forward_returns = forward_returns.reindex(factors.index)
                
                # Calculate IC
                logger.info("Calculating Information Coefficient...")
                ic_df = analyzer.calculate_ic(factors, forward_returns, periods=[1, 5, 20])
                
                avg_ic = ic_df['mean_ic'].mean()
                logger.info(f"Average IC: {avg_ic:.4f}")
                
                # Top factors
                top_20 = ic_df.head(20)
                logger.info(f"Top 20 factors by IC:\n{top_20[['factor', 'ic_5d', 'mean_ic']]}")
                
                stats['avg_ic'] = avg_ic
                stats['top_factors'] = top_20['factor'].tolist()
                
        except Exception as e:
            logger.error(f"Factor analysis failed: {e}", exc_info=True)
    
    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Symbols processed: {stats['symbols_processed']}")
    logger.info(f"Total rows stored: {stats['total_factors_stored']}")
    logger.info(f"Failed symbols: {len(stats['failed_symbols'])}")
    if stats['failed_symbols']:
        logger.info(f"Failed: {', '.join(stats['failed_symbols'])}")
    logger.info(f"Regimes detected: {stats['regimes_detected']}")
    if stats['avg_ic']:
        logger.info(f"Average IC: {stats['avg_ic']:.4f}")
    logger.info("=" * 80)
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Qlib Alpha-158 factors and detect market regimes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate factors for NIFTY 50 stocks (2 years)
  python scripts/generate_factors.py --symbols NIFTY50 --years 2
  
  # Generate factors for specific symbols
  python scripts/generate_factors.py --symbols "RELIANCE,TCS,INFY" --years 1
  
  # Update existing factors (incremental)
  python scripts/generate_factors.py --update
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="NIFTY50",
        help="'NIFTY50' or comma-separated symbol list (default: NIFTY50)"
    )
    
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years of historical data (default: 2)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), overrides --years"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incremental update (fetch only new data)"
    )
    
    parser.add_argument(
        "--no-regimes",
        action="store_true",
        help="Skip regime detection"
    )
    
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip factor analysis"
    )
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols == "NIFTY50":
        try:
            fetcher = OpenBBDataFetcher()
            symbols = fetcher.get_nifty50_constituents()
            logger.info(f"Fetched {len(symbols)} NIFTY 50 symbols")
        except Exception as e:
            logger.error(f"Could not fetch NIFTY 50: {e}")
            logger.info("Using default symbols")
            symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Date range
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.now() - timedelta(days=args.years*365)).strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Processing {len(symbols)} symbols")
    
    # Run pipeline
    stats = generate_factors_pipeline(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        detect_regimes=not args.no_regimes,
        analyze_factors=not args.no_analysis
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ FACTOR GENERATION COMPLETE")
    print("=" * 80)
    print(f"Symbols processed: {stats['symbols_processed']}/{len(symbols)}")
    print(f"Total rows stored: {stats['total_factors_stored']:,}")
    if stats['avg_ic']:
        print(f"Average IC: {stats['avg_ic']:.4f}")
    if stats['regimes_detected']:
        print(f"✅ Market regimes detected and stored")
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()

