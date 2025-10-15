#!/usr/bin/env python
"""
Run Live Trading Script.

Executes live trading with real-time market data and order execution.
"""

import argparse
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum: int, frame: object) -> None:
    """
    Handle shutdown signals.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global shutdown_flag
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


def run_live_trading(mode: str = "paper", confirm: bool = False) -> None:
    """
    Run live trading system.

    Args:
        mode: Trading mode ('paper' or 'live')
        confirm: Confirmation flag for live trading
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 80)
    logger.info("NSE ADAPTIVE REGIME TRADING SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info("-" * 80)

    # Safety check for live trading
    if mode == "live":
        if not confirm:
            logger.error("Live trading requires explicit confirmation with --confirm flag")
            logger.error("This will trade with REAL MONEY. Use paper trading to test first.")
            sys.exit(1)

        logger.warning("  LIVE TRADING MODE - REAL MONEY AT RISK ")
        logger.warning("Press Ctrl+C to stop trading gracefully")

        # Final confirmation
        try:
            response = input("\nType 'YES' to confirm live trading: ")
            if response != "YES":
                logger.info("Live trading cancelled")
                sys.exit(0)
        except KeyboardInterrupt:
            logger.info("\nLive trading cancelled")
            sys.exit(0)

    elif mode == "paper":
        logger.info("Paper trading mode - no real money at risk")

    try:
        # TODO: Implement live trading logic
        # This would typically involve:
        # 1. Initialize all components (data, strategy, execution, risk)
        # 2. Connect to Zerodha Kite API
        # 3. Start real-time data streaming
        # 4. Run trading loop
        # 5. Monitor positions and risk
        # 6. Handle shutdown gracefully

        logger.warning("Live trading not yet implemented - placeholder only")
        logger.info(
            "To run live trading, implement the execution logic in src/execution/ "
            "and update this script"
        )

        logger.info("\nLive trading system would start here...")
        logger.info("Press Ctrl+C to stop")

        # Placeholder main loop
        import time

        while not shutdown_flag:
            time.sleep(1)
            # Trading logic would go here

    except KeyboardInterrupt:
        logger.info("\nShutdown signal received")
    except Exception as e:
        logger.error(f"Live trading system error: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down gracefully...")
        logger.info("Closing all positions...")
        logger.info("Disconnecting from APIs...")
        logger.info("Trading system stopped")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run live trading system",
        epilog="  WARNING: Live trading involves real financial risk. "
        "Test thoroughly with paper trading first.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "live"],
        help="Trading mode (default: paper)",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm live trading with real money",
    )

    args = parser.parse_args()

    run_live_trading(mode=args.mode, confirm=args.confirm)


if __name__ == "__main__":
    main()

