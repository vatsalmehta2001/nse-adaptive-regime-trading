#!/usr/bin/env python
"""
Run paper trading with live data.

This script runs the trading system in paper trading mode, simulating
order execution with live market data. Perfect for strategy testing
without risking real money.

Usage:
    python scripts/run_paper_trading.py

Features:
    - Live market data from DhanHQ sandbox
    - Realistic order simulation (slippage, commission)
    - Risk management enforcement
    - Real-time portfolio tracking
    - Trade logging to database
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, time as dt_time
from typing import Optional

from src.execution.broker_factory import BrokerFactory
from src.execution.order_manager import OrderManager
from src.risk_management.risk_controller import RiskController
from src.risk_management.circuit_breaker import CircuitBreaker
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def is_market_hours() -> bool:
    """
    Check if market is currently open.

    Returns:
        True if market is open
    """
    now = datetime.now().time()
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)

    # Check if it's a weekday
    weekday = datetime.now().weekday()
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return False

    return market_open <= now <= market_close


def print_status(
    broker,
    risk_controller: RiskController,
    circuit_breaker: CircuitBreaker,
) -> None:
    """
    Print current trading status.

    Args:
        broker: Broker instance
        risk_controller: Risk controller
        circuit_breaker: Circuit breaker
    """
    try:
        portfolio_value = broker.get_portfolio_value()
        positions = broker.get_positions()
        balance = broker.get_account_balance()

        initial_capital = 1000000  # Default
        pnl = portfolio_value - initial_capital
        pnl_pct = (pnl / initial_capital) * 100

        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trading Status")
        print("-" * 70)
        print(f"Portfolio Value: ₹{portfolio_value:,.2f}")
        print(f"Cash Available:  ₹{balance.available_cash:,.2f}")
        print(f"P&L:             ₹{pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"Open Positions:  {len(positions)}")

        if positions:
            print("\nPositions:")
            for pos in positions:
                pos_pnl = pos.quantity * (pos.last_price - pos.average_price)
                pos_pnl_pct = (pos_pnl / (pos.average_price * abs(pos.quantity))) * 100
                print(
                    f"  {pos.symbol:12s} {pos.quantity:>5d} @ ₹{pos.average_price:>8,.2f} "
                    f"(LTP: ₹{pos.last_price:>8,.2f}, P&L: ₹{pos_pnl:>8,.2f} {pos_pnl_pct:+.1f}%)"
                )

        # Risk metrics
        risk_metrics = risk_controller.get_risk_metrics(portfolio_value)
        print(f"\nRisk Metrics:")
        print(f"  Daily P&L:       ₹{risk_metrics['daily_pnl']:,.2f}")
        print(f"  Daily Orders:    {risk_metrics['daily_order_count']}")
        print(f"  Violations:      {risk_metrics['violations_count']}")

        # Circuit breaker status
        cb_status = circuit_breaker.get_status()
        if cb_status["active"]:
            print(f"\nCIRCUIT BREAKER: ACTIVE")
            print(f"  Reason: {cb_status['activation_reason']}")
            if "cooldown_remaining_minutes" in cb_status:
                print(f"  Cooldown: {cb_status['cooldown_remaining_minutes']:.1f} minutes")
        else:
            cb_metrics = circuit_breaker.get_metrics(portfolio_value)
            print(f"\nCircuit Breaker: OK")
            print(f"  Consecutive losses: {cb_metrics['consecutive_losses']}")

        print("-" * 70)

    except Exception as e:
        logger.error(f"Error printing status: {e}")


def run_paper_trading(
    initial_capital: float = 1000000,
    status_interval: int = 300,
    enable_strategy: bool = False,
) -> None:
    """
    Main paper trading loop.

    Args:
        initial_capital: Starting capital
        status_interval: Seconds between status updates
        enable_strategy: Whether to enable automated strategy execution
    """
    print("\n" + "=" * 70)
    print("PAPER TRADING - LIVE SIMULATION")
    print("=" * 70)
    print(f"\nInitial Capital: ₹{initial_capital:,.2f}")
    print("Mode: Paper Trading (No real money)")
    print("Data Source: DhanHQ Sandbox")
    print()

    # Initialize components
    try:
        broker = BrokerFactory.create("PAPER", initial_capital=initial_capital)
        order_manager = OrderManager(broker, enable_monitoring=True)
        risk_controller = RiskController()
        circuit_breaker = CircuitBreaker()

        logger.info("Paper trading components initialized")

    except Exception as e:
        print(f"Error initializing components: {e}")
        logger.error(f"Initialization failed: {e}")
        return

    print("Paper trading started. Press Ctrl+C to stop.\n")

    last_status_time = time.time()

    try:
        while True:
            # Check if market is open
            if not is_market_hours():
                current_time = datetime.now().time()
                print(
                    f"\r{current_time.strftime('%H:%M:%S')} - Market closed, waiting...",
                    end="",
                )
                time.sleep(30)  # Check every 30 seconds
                continue

            # Show status periodically
            if time.time() - last_status_time >= status_interval:
                print_status(broker, risk_controller, circuit_breaker)
                last_status_time = time.time()

            # Update pending orders (for paper broker)
            if hasattr(broker, "update_pending_orders"):
                broker.update_pending_orders()

            # Strategy execution would go here
            if enable_strategy:
                # TODO: Integrate with qlib_models.signal_generator
                # signals = signal_generator.generate_signals()
                # target_weights = optimizer.optimize(signals)
                # orders = generate_orders(target_weights)
                # for order in orders:
                #     is_valid, reason = risk_controller.validate_order(...)
                #     if is_valid and not circuit_breaker.is_active():
                #         order_manager.place_order(order)
                pass

            time.sleep(5)  # Main loop delay

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...\n")

        # Final status
        print_status(broker, risk_controller, circuit_breaker)

        # Trade summary
        if hasattr(broker, "get_trades"):
            trades = broker.get_trades()
            print(f"\nTrade Summary:")
            print(f"  Total Trades: {len(trades)}")

            if trades:
                total_commission = sum(t.get("commission", 0) for t in trades)
                print(f"  Total Commission: ₹{total_commission:,.2f}")

                winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
                losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

                if winning_trades:
                    print(f"  Winning Trades: {len(winning_trades)}")
                if losing_trades:
                    print(f"  Losing Trades: {len(losing_trades)}")

        # Final P&L
        final_value = broker.get_portfolio_value()
        total_pnl = final_value - initial_capital
        total_pnl_pct = (total_pnl / initial_capital) * 100

        print("\n" + "=" * 70)
        print("PAPER TRADING SUMMARY")
        print("=" * 70)
        print(f"Initial Capital: ₹{initial_capital:,.2f}")
        print(f"Final Value:     ₹{final_value:,.2f}")
        print(f"Total P&L:       ₹{total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        print("=" * 70)

        # Stop order manager
        order_manager.stop()

        logger.info("Paper trading stopped")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run paper trading simulation")
    parser.add_argument(
        "--capital",
        type=float,
        default=1000000,
        help="Initial capital (default: 1000000)",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=300,
        help="Seconds between status updates (default: 300)",
    )
    parser.add_argument(
        "--enable-strategy",
        action="store_true",
        help="Enable automated strategy execution",
    )

    args = parser.parse_args()

    run_paper_trading(
        initial_capital=args.capital,
        status_interval=args.status_interval,
        enable_strategy=args.enable_strategy,
    )


if __name__ == "__main__":
    main()

