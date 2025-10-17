#!/usr/bin/env python
"""
Interactive broker setup script.

This script guides users through broker configuration and credential storage.
It supports DhanHQ (sandbox and live) and paper trading setup.

Usage:
    python scripts/setup_broker.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from datetime import datetime
from src.execution.broker_factory import BrokerFactory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_credentials(client_id: str, access_token: str, mode: str = "SANDBOX") -> bool:
    """
    Validate DhanHQ credentials before saving.

    Args:
        client_id: DhanHQ client ID
        access_token: DhanHQ access token
        mode: SANDBOX or LIVE

    Returns:
        True if valid, False otherwise
    """
    # Check format
    if not client_id or not access_token:
        print("❌ Client ID and access token are required")
        return False

    if len(client_id) < 5:
        print("❌ Client ID seems too short (minimum 5 characters)")
        return False

    if len(access_token) < 10:
        print("❌ Access token seems too short (minimum 10 characters)")
        return False

    # Test connection
    print(f"\n🔄 Testing {mode} credentials...")
    try:
        broker = BrokerFactory.create(
            "DHAN",
            mode=mode,
            client_id=client_id,
            access_token=access_token
        )

        if broker.authenticate():
            print(f"✅ {mode} credentials validated successfully!")

            # Try to fetch balance
            try:
                balance = broker.get_account_balance()
                print(f"📊 Account balance: ₹{balance.total_balance:,.2f}")
                print(f"💰 Available cash: ₹{balance.available_cash:,.2f}")
                return True
            except Exception as e:
                print(f"⚠️  Could not fetch balance: {e}")
                print("✅ Authentication works, but API may be limited")
                return True
        else:
            print("❌ Authentication failed - please check your credentials")
            return False

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def setup_dhan_sandbox() -> bool:
    """
    Guide user through DhanHQ sandbox setup.

    Returns:
        True if setup successful
    """
    print("\n" + "=" * 70)
    print("DHAN SANDBOX SETUP")
    print("=" * 70)

    print("\nSteps to get DhanHQ sandbox credentials:")
    print("1. Visit: https://developer.dhanhq.co")
    print("2. Sign up (no Dhan account needed!)")
    print("3. Navigate to 'API Credentials' section")
    print("4. Get your client ID and access token")
    print("5. Sandbox provides ₹10 lakhs virtual capital")
    print()

    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"\n📝 Attempt {attempt + 1}/{max_attempts}")
        client_id = input("Enter your DhanHQ client ID: ").strip()
        access_token = input("Enter your DhanHQ access token: ").strip()

        if validate_credentials(client_id, access_token, "SANDBOX"):
            # Save to .env file
            env_path = Path(".env")
            env_exists = env_path.exists()

            with open(env_path, "a") as f:
                if not env_exists:
                    f.write("# Environment variables for NSE trading system\n")
                f.write(f"\n# DhanHQ Sandbox Credentials (added {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write(f"DHAN_CLIENT_ID={client_id}\n")
                f.write(f"DHAN_ACCESS_TOKEN={access_token}\n")

            print(f"\n✅ Credentials saved to {env_path}")

            # Also add to .gitignore if not already there
            gitignore_path = Path(".gitignore")
            if gitignore_path.exists():
                with open(gitignore_path, "r") as f:
                    content = f.read()
                if ".env" not in content:
                    with open(gitignore_path, "a") as f:
                        f.write("\n# Environment variables\n.env\n")
                    print("📝 Added .env to .gitignore")

            return True
        else:
            if attempt < max_attempts - 1:
                print("\n⚠️  Validation failed. Please check your credentials and try again.")
                retry = input("Retry? (y/n): ").strip().lower()
                if retry != 'y':
                    return False
            else:
                print("\n❌ Setup failed after 3 attempts")
                return False

    return False


def setup_dhan_live() -> bool:
    """
    Guide user through DhanHQ live setup.

    Returns:
        True if setup successful
    """
    print("\n" + "=" * 70)
    print("DHAN LIVE TRADING SETUP")
    print("=" * 70)

    print("\nWARNING: LIVE TRADING INVOLVES REAL MONEY")
    print("\nBefore proceeding, ensure you have:")
    print("- Thoroughly tested your strategy in sandbox")
    print("- Validated risk management rules")
    print("- Are comfortable with potential losses")
    print("- Have a Dhan demat account")
    print()

    confirm = input("Type 'YES' to continue with live setup: ").strip()
    if confirm != "YES":
        print("\nSetup cancelled.")
        return False

    print("\nSteps to get DhanHQ live credentials:")
    print("1. Login to your Dhan demat account at https://dhan.co")
    print("2. Navigate to Settings > API Access")
    print("3. Generate API credentials")
    print("4. Copy your client ID and access token")
    print()
    print("⚠️  Note: Live API access may require additional verification")
    print()

    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"\n📝 Attempt {attempt + 1}/{max_attempts}")
        client_id = input("Enter your DhanHQ LIVE client ID: ").strip()
        access_token = input("Enter your DhanHQ LIVE access token: ").strip()

        if validate_credentials(client_id, access_token, "LIVE"):
            # Save to .env
            env_path = Path(".env")
            with open(env_path, "a") as f:
                f.write(f"\n# DhanHQ Live Credentials (added {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write(f"DHAN_LIVE_CLIENT_ID={client_id}\n")
                f.write(f"DHAN_LIVE_ACCESS_TOKEN={access_token}\n")

            print(f"\n✅ Live credentials saved to {env_path}")
            print("\n⚠️  IMPORTANT: Test extensively with small amounts before scaling up!")

            return True
        else:
            if attempt < max_attempts - 1:
                print("\n⚠️  Validation failed. Please check your credentials and try again.")
                retry = input("Retry? (y/n): ").strip().lower()
                if retry != 'y':
                    return False
            else:
                print("\n❌ Setup failed after 3 attempts")
                return False

    return False


def setup_paper_trading() -> bool:
    """
    Guide user through paper trading setup.

    Returns:
        True if setup successful
    """
    print("\n" + "=" * 70)
    print("PAPER TRADING SETUP")
    print("=" * 70)

    print("\nPaper trading uses custom simulation with live data.")
    print("No real money involved - perfect for strategy testing.")
    print("\nPaper trading requires DhanHQ sandbox for live data.")
    print()

    # Check if DhanHQ sandbox is set up
    if not (os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN")):
        print("DhanHQ sandbox not configured.")
        setup_sandbox = input("Would you like to set up DhanHQ sandbox now? (yes/no): ").strip().lower()
        if setup_sandbox == "yes":
            if not setup_dhan_sandbox():
                print("\nPaper trading setup incomplete.")
                return False
        else:
            print("\nPaper trading requires DhanHQ sandbox. Setup cancelled.")
            return False

    print("\nPaper trading configuration:")
    initial_capital = input("Enter initial capital (default: 1000000): ").strip()
    if not initial_capital:
        initial_capital = "1000000"

    try:
        initial_capital_float = float(initial_capital)
    except ValueError:
        print("Invalid capital amount. Using default: ₹10,00,000")
        initial_capital_float = 1000000.0

    print(f"\nPaper trading will start with ₹{initial_capital_float:,.2f}")
    print("\nTo use paper trading in your code:")
    print("  from src.execution import BrokerFactory")
    print(f"  broker = BrokerFactory.create('PAPER', initial_capital={initial_capital_float})")

    return True


def main():
    """Main setup flow."""
    print("\n" + "=" * 70)
    print("NSE TRADING SYSTEM - BROKER SETUP")
    print("=" * 70)
    print("\nThis script will help you configure broker credentials.\n")

    print("Available options:")
    print("1. DhanHQ Sandbox (Recommended for testing)")
    print("2. DhanHQ Live (Real money - use after thorough testing)")
    print("3. Paper Trading (Custom simulation with live data)")
    print("4. Exit")
    print()

    choice = input("Select option (1-4): ").strip()

    success = False

    if choice == "1":
        success = setup_dhan_sandbox()
    elif choice == "2":
        success = setup_dhan_live()
    elif choice == "3":
        success = setup_paper_trading()
    elif choice == "4":
        print("\nExiting setup.")
        return
    else:
        print("\nInvalid choice.")
        return

    if success:
        print("\n" + "=" * 70)
        print("SETUP COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Test with paper trading: python scripts/run_paper_trading.py")
        print("2. Monitor results for 2+ weeks")
        print("3. Review performance metrics")
        print("4. Only then consider live trading")
        print("\nImportant: Always start with small position sizes in live trading.")
        print()
    else:
        print("\nSetup incomplete. Please try again.")


if __name__ == "__main__":
    main()

