#!/usr/bin/env python
"""
Update DhanHQ symbol-to-security_id mappings.

This script helps add and manage symbol→security_id mappings required
for DhanHQ live trading.

Usage:
    # Interactive mode
    python scripts/update_symbol_mappings.py

    # Add single mapping
    python scripts/update_symbol_mappings.py --symbol RELIANCE --exchange NSE --security-id 1333

    # Validate existing mappings
    python scripts/update_symbol_mappings.py --validate

    # List all mappings
    python scripts/update_symbol_mappings.py --list

Note:
    Security IDs must be obtained from DhanHQ documentation or API.
    Using incorrect security IDs in live trading can cause order failures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.execution.symbol_mapper import SymbolMapper
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_mapping_interactive() -> None:
    """Interactive mode to add symbol mappings."""
    print("\n" + "=" * 70)
    print("DhanHQ Symbol Mapping Tool")
    print("=" * 70)
    print("\n⚠️  Important: Security IDs must be verified with DhanHQ")
    print("Using incorrect IDs in live trading will cause order failures!")
    print()

    mapper = SymbolMapper()

    while True:
        print("\nAdd new symbol mapping:")
        symbol = input("Symbol (e.g., RELIANCE): ").strip().upper()
        if not symbol:
            print("❌ Symbol cannot be empty")
            break

        exchange = input("Exchange (NSE/BSE) [NSE]: ").strip().upper() or "NSE"

        # Check if mapping already exists
        if mapper.has_mapping(symbol, exchange):
            current_id = mapper.get_security_id(symbol, exchange)
            print(f"⚠️  Mapping already exists: {symbol} ({exchange}) → {current_id}")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                continue

        security_id = input(f"Security ID for {symbol} on {exchange}: ").strip()

        if not security_id:
            print("❌ Security ID cannot be empty")
            continue

        if not security_id.isdigit():
            print(f"⚠️  Warning: Security ID '{security_id}' is not numeric")
            confirm = input("Add anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue

        notes = input("Notes (optional): ").strip()

        try:
            mapper.add_mapping(symbol, exchange, security_id, "manual", notes)
            print(f"✅ Added: {symbol} ({exchange}) → {security_id}")
        except Exception as e:
            print(f"❌ Error adding mapping: {e}")

        more = input("\nAdd another? (y/n): ").strip().lower()
        if more != 'y':
            break

    print("\n✅ Done! Mappings saved.")


def list_mappings() -> None:
    """List all existing mappings."""
    mapper = SymbolMapper()

    print("\n" + "=" * 70)
    print("DhanHQ Symbol Mappings")
    print("=" * 70)

    nse_symbols = sorted(mapper.get_all_symbols("NSE"))
    bse_symbols = sorted(mapper.get_all_symbols("BSE"))

    if nse_symbols:
        print(f"\nNSE ({len(nse_symbols)} symbols):")
        for symbol in nse_symbols:
            try:
                security_id = mapper.get_security_id(symbol, "NSE")
                print(f"  {symbol:15s} → {security_id}")
            except ValueError:
                pass

    if bse_symbols:
        print(f"\nBSE ({len(bse_symbols)} symbols):")
        for symbol in bse_symbols:
            try:
                security_id = mapper.get_security_id(symbol, "BSE")
                print(f"  {symbol:15s} → {security_id}")
            except ValueError:
                pass

    total = len(nse_symbols) + len(bse_symbols)
    print(f"\nTotal: {total} symbols")


def validate_mappings() -> None:
    """Validate all mappings for issues."""
    mapper = SymbolMapper()

    print("\n" + "=" * 70)
    print("Validating Symbol Mappings")
    print("=" * 70)

    results = mapper.validate_mappings()

    print(f"\n✅ Valid mappings: {len(results['valid'])}")

    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  {warning}")

    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  {error}")
    else:
        print("\n✅ No errors found!")

    print()


def bulk_import_from_csv(csv_file: str) -> None:
    """
    Import mappings from CSV file.

    Args:
        csv_file: Path to CSV file with columns: symbol,exchange,security_id
    """
    import csv

    print(f"\n📥 Importing mappings from {csv_file}...")

    try:
        mappings = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get('symbol', '').strip().upper()
                exchange = row.get('exchange', 'NSE').strip().upper()
                security_id = row.get('security_id', '').strip()
                notes = row.get('notes', '').strip()

                if symbol and security_id:
                    mappings.append((symbol, exchange, security_id, notes))

        if mappings:
            mapper = SymbolMapper()
            count = mapper.bulk_add_mappings(mappings)
            print(f"✅ Imported {count} mappings")
        else:
            print("❌ No valid mappings found in CSV")

    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
    except Exception as e:
        print(f"❌ Error importing: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update DhanHQ symbol-to-security_id mappings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/update_symbol_mappings.py

  # Add single mapping
  python scripts/update_symbol_mappings.py --symbol RELIANCE --exchange NSE --security-id 1333

  # List all mappings
  python scripts/update_symbol_mappings.py --list

  # Validate mappings
  python scripts/update_symbol_mappings.py --validate

  # Import from CSV
  python scripts/update_symbol_mappings.py --import mappings.csv
        """
    )

    parser.add_argument("--symbol", help="Trading symbol")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE/BSE)")
    parser.add_argument("--security-id", help="DhanHQ security ID")
    parser.add_argument("--notes", default="", help="Optional notes")
    parser.add_argument("--list", action="store_true", help="List all mappings")
    parser.add_argument("--validate", action="store_true", help="Validate mappings")
    parser.add_argument("--import", dest="import_csv", help="Import mappings from CSV")

    args = parser.parse_args()

    # Handle different modes
    if args.list:
        list_mappings()
    elif args.validate:
        validate_mappings()
    elif args.import_csv:
        bulk_import_from_csv(args.import_csv)
    elif args.symbol and args.security_id:
        # Add single mapping
        mapper = SymbolMapper()
        try:
            mapper.add_mapping(
                args.symbol,
                args.exchange,
                args.security_id,
                "cli",
                args.notes
            )
            print(f"✅ Mapping added: {args.symbol} ({args.exchange}) → {args.security_id}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        add_mapping_interactive()


if __name__ == "__main__":
    main()

