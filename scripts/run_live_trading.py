#!/usr/bin/env python
"""
Run live trading with real money.

WARNING: This script executes real trades with real money. Use only after
extensive testing in paper trading mode. Ensure you understand the risks
and have validated your strategy thoroughly.

Usage:
    python scripts/run_live_trading.py --broker DHAN

Safety Features:
    - Pre-trade risk checks
    - Position size limits
    - Daily loss limits
    - Circuit breaker
    - Emergency stop commands
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import signal
from datetime import datetime, time as dt_time

from src.execution.broker_factory import BrokerFactory
from src.execution.order_manager import OrderManager
from src.risk_management.risk_controller import RiskController
from src.risk_management.circuit_breaker import CircuitBreaker
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_flag
    print("\n\nReceived shutdown signal. Stopping safely...")
    shutdown_flag = True


def is_market_hours() -> bool:
    """Check if market is currently open."""
    now = datetime.now().time()
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)

    # Check if it's a weekday
    weekday = datetime.now().weekday()
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return False

    return market_open <= now <= market_close


def should_square_off() -> bool:
    """Check if it's time to square off positions (close all)."""
    now = datetime.now().time()
    square_off_time = dt_time(15, 20)  # 10 minutes before close
    return now >= square_off_time


def print_status(broker, risk_controller: RiskController, circuit_breaker: CircuitBreaker) -> None:
    """Print current trading status."""
    try:
        positions = broker.get_positions()
        balance = broker.get_account_balance()

        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - LIVE Trading Status")
        print("-" * 70)
        print(f"Account Balance: ₹{balance.total_balance:,.2f}")
        print(f"Available Cash:  ₹{balance.available_cash:,.2f}")
        print(f"Used Margin:     ₹{balance.used_margin:,.2f}")
        print(f"Open Positions:  {len(positions)}")

        if positions:
            print("\nPositions:")
            for pos in positions:
                pos_value = pos.quantity * pos.last_price
                print(
                    f"  {pos.symbol:12s} {pos.quantity:>5d} @ ₹{pos.average_price:>8,.2f} "
                    f"(LTP: ₹{pos.last_price:>8,.2f}, Value: ₹{pos_value:>10,.2f}, P&L: ₹{pos.pnl:>8,.2f})"
                )

        # Risk metrics
        risk_metrics = risk_controller.get_risk_metrics(balance.total_balance)
        print(f"\nRisk Metrics:")
        print(f"  Daily P&L:       ₹{risk_metrics['daily_pnl']:,.2f}")
        print(f"  Daily Orders:    {risk_metrics['daily_order_count']}/{risk_metrics['max_orders_per_day']}")
        print(f"  Loss Limit:      {risk_metrics['loss_limit_remaining_pct']:.2f}% remaining")

        # Circuit breaker
        cb_status = circuit_breaker.get_status()
        if cb_status["active"]:
            print(f"\nCIRCUIT BREAKER: ACTIVE")
            print(f"  Reason: {cb_status['activation_reason']}")
            if "cooldown_remaining_minutes" in cb_status:
                print(f"  Cooldown: {cb_status['cooldown_remaining_minutes']:.1f} minutes")

        print("-" * 70)

    except Exception as e:
        logger.error(f"Error printing status: {e}")


def confirm_live_trading() -> bool:
    """Get user confirmation for live trading."""
    print("\n" + "=" * 70)
    print("LIVE TRADING WARNING")
    print("=" * 70)
    print("\nYou are about to start LIVE TRADING with REAL MONEY.")
    print("\nBefore proceeding, confirm that you have:")
    print("  1. Tested your strategy extensively in paper trading")
    print("  2. Validated risk management rules")
    print("  3. Reviewed and understand all position limits")
    print("  4. Started with small position sizes")
    print("  5. Are prepared for potential losses")
    print()

    confirm1 = input("Have you completed extensive paper trading? (yes/no): ").strip().lower()
    if confirm1 != "yes":
        return False

    confirm2 = input("Do you understand you may lose money? (yes/no): ").strip().lower()
    if confirm2 != "yes":
        return False

    confirm3 = input("Type 'START LIVE TRADING' to confirm: ").strip()
    if confirm3 != "START LIVE TRADING":
        return False

    return True


def run_live_trading(broker_type: str = "DHAN", mode: str = "LIVE") -> None:
    """
    Main live trading loop.

    Args:
        broker_type: Broker to use (DHAN or KITE)
        mode: LIVE or SANDBOX
    """
    # Safety check
    if mode == "LIVE" and not confirm_live_trading():
        print("\nLive trading cancelled.")
        return

    print("\n" + "=" * 70)
    print(f"LIVE TRADING - {broker_type} ({mode})")
    print("=" * 70)
    print()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components
    try:
        print("Initializing trading components...")
        broker = BrokerFactory.create(broker_type, mode=mode)

        if not broker.authenticate():
            print("Authentication failed. Exiting.")
            return

        order_manager = OrderManager(broker, enable_monitoring=True)
        risk_controller = RiskController()
        circuit_breaker = CircuitBreaker()

        logger.info(f"Live trading initialized: {broker_type} {mode}")
        print("Trading components initialized successfully.\n")

    except Exception as e:
        print(f"Error initializing components: {e}")
        logger.error(f"Initialization failed: {e}")
        return

    print("Live trading started. Press Ctrl+C for safe shutdown.\n")

    last_status_time = time.time()
    status_interval = 300  # 5 minutes

    try:
        while not shutdown_flag:
            # Check if market is open
            if not is_market_hours():
                current_time = datetime.now().time()
                print(
                    f"\r{current_time.strftime('%H:%M:%S')} - Market closed, waiting...",
                    end="",
                )

                # Reset daily counters at end of day
                if current_time >= dt_time(15, 45):
                    risk_controller.reset_daily()
                    circuit_breaker.reset()

                time.sleep(30)
                continue

            # Square off positions near market close
            if should_square_off():
                print("\n\nSquare-off time reached. Closing all positions...")
                try:
                    broker.close_all_positions()
                    print("All positions closed.")
                except Exception as e:
                    logger.error(f"Error closing positions: {e}")

                # Wait until market close
                time.sleep(60)
                continue

            # Show status periodically
            if time.time() - last_status_time >= status_interval:
                print_status(broker, risk_controller, circuit_breaker)
                last_status_time = time.time()

            # Check circuit breaker
            if circuit_breaker.is_active():
                print("\rCircuit breaker active. Waiting...", end="")
                time.sleep(30)
                continue

            # Strategy execution would go here
            # TODO: Integrate with qlib_models.signal_generator
            # signals = signal_generator.generate_signals()
            # target_weights = optimizer.optimize(signals)
            # orders = generate_orders(target_weights)
            # for order in orders:
            #     # Get current price
            #     price = broker.get_ltp(order.symbol)
            #     positions = broker.get_positions()
            #     balance = broker.get_account_balance()
            #     
            #     # Validate with risk controller
            #     is_valid, reason = risk_controller.validate_order(
            #         order, balance.total_balance, positions, price
            #     )
            #     
            #     if is_valid:
            #         order_id = order_manager.place_order(order)
            #         risk_controller.record_order()
            #     else:
            #         logger.warning(f"Order rejected: {reason}")

            time.sleep(10)  # Main loop delay

    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        print(f"\nError: {e}")

    finally:
        # Cleanup on exit
        print("\n\nShutting down...")

        # Final status
        print_status(broker, risk_controller, circuit_breaker)

        # Emergency: close all positions if requested
        close_positions = input("\nClose all positions? (yes/no): ").strip().lower()
        if close_positions == "yes":
            print("Closing all positions...")
            try:
                broker.close_all_positions()
                print("All positions closed.")
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
                print(f"Error: {e}")

        # Cancel all pending orders
        print("Cancelling pending orders...")
        try:
            broker.cancel_all_orders()
            print("All pending orders cancelled.")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

        # Stop order manager
        order_manager.stop()

        print("\nLive trading stopped safely.")
        logger.info("Live trading stopped")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument(
        "--broker",
        type=str,
        default="DHAN",
        choices=["DHAN", "KITE"],
        help="Broker to use (default: DHAN)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="LIVE",
        choices=["LIVE", "SANDBOX"],
        help="Trading mode (default: LIVE)",
    )

    args = parser.parse_args()

    # Extra safety check
    if args.mode == "LIVE":
        print("\nWARNING: You selected LIVE mode with real money.")
        print("Consider using --mode SANDBOX for testing first.")
        confirm = input("Continue with LIVE mode? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return

    run_live_trading(broker_type=args.broker, mode=args.mode)


if __name__ == "__main__":
    main()
