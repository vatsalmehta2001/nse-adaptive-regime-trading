"""
Command-line interface for NSE Adaptive Regime Trading System.

Provides a unified CLI for all trading operations.
"""

import argparse
import sys
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NSE Adaptive Regime Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup Qlib data
  nse-trade setup-data --market NSE

  # Train Qlib models
  nse-trade train-qlib --model lightgbm

  # Train RL agent
  nse-trade train-rl --algorithm ppo --timesteps 1000000

  # Run backtest
  nse-trade backtest --start 2023-01-01 --end 2024-12-31

  # Run live trading (paper mode)
  nse-trade live --mode paper

For more information, visit: https://github.com/yourusername/nse-adaptive-regime-trading
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup data command
    setup_parser = subparsers.add_parser("setup-data", help="Setup Qlib data")
    setup_parser.add_argument("--market", default="NSE", help="Market name")
    setup_parser.add_argument("--region", default="IN", help="Region code")

    # Train Qlib command
    train_qlib_parser = subparsers.add_parser("train-qlib", help="Train Qlib models")
    train_qlib_parser.add_argument("--model", default="lightgbm", help="Model type")
    train_qlib_parser.add_argument("--config", help="Config file path")

    # Train RL command
    train_rl_parser = subparsers.add_parser("train-rl", help="Train RL agent")
    train_rl_parser.add_argument("--algorithm", default="ppo", help="RL algorithm")
    train_rl_parser.add_argument("--timesteps", type=int, default=1000000, help="Training steps")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--capital", type=float, default=1000000, help="Initial capital")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("--mode", default="paper", choices=["paper", "live"], help="Mode")
    live_parser.add_argument("--confirm", action="store_true", help="Confirm live trading")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "setup-data":
            from scripts.setup_qlib_data import setup_qlib_data

            setup_qlib_data(market=args.market, region=args.region)

        elif args.command == "train-qlib":
            from scripts.train_qlib_models import train_qlib_models

            config = args.config or "config/qlib_config.yaml"
            train_qlib_models(config_path=config, model=args.model)

        elif args.command == "train-rl":
            from scripts.train_rl_agent import train_rl_agent

            train_rl_agent(algorithm=args.algorithm, total_timesteps=args.timesteps)

        elif args.command == "backtest":
            from scripts.run_backtest import run_backtest

            run_backtest(
                start_date=args.start,
                end_date=args.end,
                initial_capital=args.capital,
                strategy="adaptive_regime",
            )

        elif args.command == "live":
            from scripts.run_live_trading import run_live_trading

            run_live_trading(mode=args.mode, confirm=args.confirm)

        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

