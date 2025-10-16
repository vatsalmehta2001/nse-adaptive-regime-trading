"""
Verify Qlib Implementation.

Quick script to verify all components are in place and working.
"""

import sys
from pathlib import Path


def verify_files():
    """Verify all required files exist."""
    print("\n" + "="*80)
    print("QLIB IMPLEMENTATION VERIFICATION")
    print("="*80 + "\n")

    required_files = {
        'Core Components': [
            'src/qlib_models/__init__.py',
            'src/qlib_models/model_trainer.py',
            'src/qlib_models/signal_generator.py',
            'src/portfolio/__init__.py',
            'src/portfolio/optimizer.py',
            'src/backtesting/backtest_engine.py',
            'src/backtesting/performance_analyzer.py',
            'src/strategies/__init__.py',
            'src/strategies/regime_adaptive_strategy.py',
        ],
        'Scripts': [
            'scripts/train_models.py',
            'scripts/run_backtest.py',
        ],
        'Configuration': [
            'config/qlib_model_config.yaml',
        ],
        'Tests': [
            'tests/unit/test_qlib_models.py',
        ],
        'Documentation': [
            'QLIB_IMPLEMENTATION.md',
            'QLIB_QUICK_REFERENCE.md',
            'QLIB_IMPLEMENTATION_SUMMARY.md',
        ]
    }

    all_present = True
    total_files = 0

    for category, files in required_files.items():
        print(f"\n{category}:")
        print("-" * 40)

        for file_path in files:
            path = Path(file_path)
            exists = path.exists()
            status = "✓" if exists else "✗"

            size = ""
            if exists and path.is_file():
                size_kb = path.stat().st_size / 1024
                size = f"({size_kb:.1f} KB)"

            print(f"  {status} {file_path} {size}")

            if not exists:
                all_present = False
            else:
                total_files += 1

    print("\n" + "="*80)
    print(f"\nTotal Files: {total_files}")
    print(f"Status: {'ALL FILES PRESENT ✓' if all_present else 'MISSING FILES ✗'}")

    return all_present


def verify_imports():
    """Verify imports work."""
    print("\n" + "="*80)
    print("IMPORT VERIFICATION")
    print("="*80 + "\n")

    imports = [
        ('QlibModelTrainer', 'from src.qlib_models import QlibModelTrainer'),
        ('AlphaSignalGenerator', 'from src.qlib_models import AlphaSignalGenerator'),
        ('PortfolioOptimizer', 'from src.portfolio import PortfolioOptimizer'),
        ('BacktestEngine', 'from src.backtesting import BacktestEngine'),
        ('PerformanceAnalyzer', 'from src.backtesting import PerformanceAnalyzer'),
        ('RegimeAdaptiveStrategy', 'from src.strategies import RegimeAdaptiveStrategy'),
    ]

    all_success = True

    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_success = False

    print(f"\nImport Status: {'ALL IMPORTS SUCCESSFUL ✓' if all_success else 'IMPORT ERRORS ✗'}")

    return all_success


def verify_dependencies():
    """Verify required dependencies."""
    print("\n" + "="*80)
    print("DEPENDENCY VERIFICATION")
    print("="*80 + "\n")

    dependencies = [
        'pandas',
        'numpy',
        'lightgbm',
        'xgboost',
        'scipy',
        'sklearn',
        'cvxpy',
        'loguru',
        'tqdm',
    ]

    all_installed = True

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (not installed)")
            all_installed = False

    if not all_installed:
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")

    print(f"\nDependency Status: {'ALL INSTALLED ✓' if all_installed else 'MISSING DEPENDENCIES ✗'}")

    return all_installed


def print_next_steps():
    """Print next steps."""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")

    steps = [
        "1. Install missing dependencies (if any):",
        "   pip install -r requirements.txt",
        "",
        "2. Run tests:",
        "   pytest tests/unit/test_qlib_models.py -v",
        "",
        "3. Train models:",
        "   python scripts/train_models.py --symbols NIFTY50 --model lightgbm",
        "",
        "4. Run backtest:",
        "   python scripts/run_backtest.py --start 2023-01-01 --end 2024-12-31",
        "",
        "5. Review documentation:",
        "   - QLIB_IMPLEMENTATION.md (full guide)",
        "   - QLIB_QUICK_REFERENCE.md (quick commands)",
        "   - QLIB_IMPLEMENTATION_SUMMARY.md (summary)",
        "",
        "6. Validate performance:",
        "   - Check reports/ directory for backtest results",
        "   - Verify Sharpe ratio > 0.5",
        "   - Ensure IC in range 0.03-0.08",
    ]

    for step in steps:
        print(f"  {step}")


def main():
    """Run verification."""
    files_ok = verify_files()
    imports_ok = verify_imports()
    deps_ok = verify_dependencies()

    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80 + "\n")

    print(f"  Files:        {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"  Imports:      {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"  Dependencies: {'✓ PASS' if deps_ok else '✗ FAIL'}")

    overall = files_ok and imports_ok and deps_ok
    print(f"\n  Overall:      {'✓ READY' if overall else '✗ NOT READY'}")

    print_next_steps()

    print("\n" + "="*80 + "\n")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

