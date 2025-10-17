"""
Symbol to Security ID mapper for DhanHQ.

DhanHQ requires numeric security IDs instead of symbols for live trading.
This module provides mapping functionality with multiple data sources.

Classes:
    SymbolMapper: Maps trading symbols to DhanHQ security IDs

Example:
    >>> from src.execution.symbol_mapper import SymbolMapper
    >>> mapper = SymbolMapper()
    >>> security_id = mapper.get_security_id("RELIANCE", "NSE")
    >>> print(security_id)
    '1333'
"""

import csv
from pathlib import Path
from typing import Dict, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SymbolMapper:
    """
    Maps trading symbols to DhanHQ security IDs.

    Supports multiple data sources:
    1. Local CSV file (fastest)
    2. Fallback to cached data
    3. Helper script for adding new mappings

    Attributes:
        csv_path: Path to CSV file containing mappings
        symbol_map: Dictionary of symbol→security_id mappings
    """

    def __init__(self, csv_path: str = "data/dhan_security_ids.csv"):
        """
        Initialize mapper with CSV data source.

        Args:
            csv_path: Path to CSV file with symbol mappings
        """
        self.csv_path = Path(csv_path)
        self.symbol_map: Dict[str, str] = {}
        self._load_mappings()

    def get_security_id(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Get security ID for symbol.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE or BSE)

        Returns:
            DhanHQ security ID

        Raises:
            ValueError: If symbol not found in mappings

        Example:
            >>> mapper = SymbolMapper()
            >>> security_id = mapper.get_security_id("RELIANCE", "NSE")
        """
        symbol = symbol.upper()
        key = f"{symbol}_{exchange}"

        if key in self.symbol_map:
            logger.debug(f"Found security ID for {symbol} ({exchange}): {self.symbol_map[key]}")
            return self.symbol_map[key]

        # Try without exchange as fallback
        if symbol in self.symbol_map:
            logger.warning(f"Using default exchange mapping for {symbol}")
            return self.symbol_map[symbol]

        # Symbol not found
        error_msg = (
            f"Security ID not found for {symbol} on {exchange}. "
            f"Please add to {self.csv_path} or run: "
            f"python scripts/update_symbol_mappings.py --symbol {symbol} --security-id <ID>"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def has_mapping(self, symbol: str, exchange: str = "NSE") -> bool:
        """
        Check if mapping exists for symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE or BSE)

        Returns:
            True if mapping exists, False otherwise
        """
        symbol = symbol.upper()
        key = f"{symbol}_{exchange}"
        return key in self.symbol_map or symbol in self.symbol_map

    def _load_mappings(self) -> None:
        """Load mappings from CSV file."""
        if not self.csv_path.exists():
            logger.warning(f"Mapping file not found: {self.csv_path}")
            logger.warning("Creating with NIFTY 50 defaults...")
            self._create_default_mappings()
            return

        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row['symbol'].upper()
                    exchange = row.get('exchange', 'NSE').upper()
                    security_id = row['security_id']

                    key = f"{symbol}_{exchange}"
                    self.symbol_map[key] = security_id
                    self.symbol_map[symbol] = security_id  # Default mapping

            logger.info(f"Loaded {len(self.symbol_map) // 2} symbol mappings from {self.csv_path}")

        except Exception as e:
            logger.error(f"Error loading mappings from {self.csv_path}: {e}")
            logger.warning("Creating default mappings...")
            self._create_default_mappings()

    def _create_default_mappings(self) -> None:
        """
        Create default mappings for NIFTY 50 stocks.

        Creates a CSV file with security IDs for the most commonly traded stocks.
        Note: These are example IDs. For production use, verify with DhanHQ.
        """
        # Top NIFTY stocks with their approximate security IDs
        # Note: These need to be verified with actual DhanHQ security IDs
        defaults = [
            ("RELIANCE", "NSE", "1333"),
            ("TCS", "NSE", "11536"),
            ("HDFCBANK", "NSE", "1333"),
            ("INFY", "NSE", "1594"),
            ("ICICIBANK", "NSE", "4963"),
            ("HINDUNILVR", "NSE", "1394"),
            ("ITC", "NSE", "1660"),
            ("SBIN", "NSE", "3045"),
            ("BHARTIARTL", "NSE", "4668"),
            ("KOTAKBANK", "NSE", "1922"),
            ("LT", "NSE", "11483"),
            ("AXISBANK", "NSE", "5900"),
            ("ASIANPAINT", "NSE", "7406"),
            ("MARUTI", "NSE", "10999"),
            ("SUNPHARMA", "NSE", "3351"),
            ("TITAN", "NSE", "3506"),
            ("ULTRACEMCO", "NSE", "11532"),
            ("NESTLEIND", "NSE", "17963"),
            ("BAJFINANCE", "NSE", "16675"),
            ("WIPRO", "NSE", "3787"),
            ("M&M", "NSE", "10999"),
            ("NTPC", "NSE", "11630"),
            ("POWERGRID", "NSE", "14977"),
            ("ONGC", "NSE", "2475"),
            ("COALINDIA", "NSE", "20374"),
        ]

        # Create directory if it doesn't exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['symbol', 'exchange', 'security_id', 'source', 'notes'])

            for symbol, exchange, sec_id in defaults:
                writer.writerow([
                    symbol,
                    exchange,
                    sec_id,
                    'default',
                    'Verify with DhanHQ before live trading'
                ])

                # Add to in-memory map
                key = f"{symbol}_{exchange}"
                self.symbol_map[key] = sec_id
                self.symbol_map[symbol] = sec_id

        logger.info(f"Created default mappings with {len(defaults)} symbols: {self.csv_path}")
        logger.warning(
            "⚠️  Default security IDs are examples. "
            "Verify with DhanHQ before live trading!"
        )

    def add_mapping(
        self,
        symbol: str,
        exchange: str,
        security_id: str,
        source: str = "manual",
        notes: str = ""
    ) -> None:
        """
        Add new symbol mapping.

        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE or BSE)
            security_id: DhanHQ security ID
            source: Source of the mapping (manual, api, etc.)
            notes: Additional notes

        Example:
            >>> mapper = SymbolMapper()
            >>> mapper.add_mapping("NEWSTOCK", "NSE", "12345", "manual", "Added for testing")
        """
        symbol = symbol.upper()
        exchange = exchange.upper()

        # Add to in-memory map
        key = f"{symbol}_{exchange}"
        self.symbol_map[key] = security_id
        self.symbol_map[symbol] = security_id

        # Append to CSV
        try:
            # Check if file exists and has headers
            file_exists = self.csv_path.exists()

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)

                # Write header if file doesn't exist
                if not file_exists:
                    writer.writerow(['symbol', 'exchange', 'security_id', 'source', 'notes'])

                writer.writerow([symbol, exchange, security_id, source, notes])

            logger.info(f"Added mapping: {symbol} ({exchange}) → {security_id}")

        except Exception as e:
            logger.error(f"Error adding mapping to CSV: {e}")
            raise

    def bulk_add_mappings(self, mappings: list) -> int:
        """
        Add multiple mappings at once.

        Args:
            mappings: List of tuples (symbol, exchange, security_id)

        Returns:
            Number of mappings added

        Example:
            >>> mapper = SymbolMapper()
            >>> mappings = [("STOCK1", "NSE", "111"), ("STOCK2", "NSE", "222")]
            >>> count = mapper.bulk_add_mappings(mappings)
        """
        added = 0

        for mapping in mappings:
            try:
                if len(mapping) >= 3:
                    symbol, exchange, security_id = mapping[:3]
                    notes = mapping[3] if len(mapping) > 3 else ""
                    self.add_mapping(symbol, exchange, security_id, "bulk", notes)
                    added += 1
            except Exception as e:
                logger.error(f"Error adding mapping {mapping}: {e}")

        logger.info(f"Bulk added {added}/{len(mappings)} mappings")
        return added

    def get_all_symbols(self, exchange: Optional[str] = None) -> list:
        """
        Get list of all symbols with mappings.

        Args:
            exchange: Filter by exchange (None = all)

        Returns:
            List of symbol names

        Example:
            >>> mapper = SymbolMapper()
            >>> nse_symbols = mapper.get_all_symbols("NSE")
        """
        if exchange:
            exchange = exchange.upper()
            suffix = f"_{exchange}"
            return [
                key.replace(suffix, '')
                for key in self.symbol_map.keys()
                if key.endswith(suffix)
            ]
        else:
            # Return unique symbols (without exchange suffix)
            return [
                key for key in self.symbol_map.keys()
                if '_' not in key
            ]

    def validate_mappings(self) -> Dict[str, list]:
        """
        Validate all mappings for issues.

        Returns:
            Dictionary with validation results

        Example:
            >>> mapper = SymbolMapper()
            >>> results = mapper.validate_mappings()
            >>> if results['errors']:
            ...     print(f"Found {len(results['errors'])} errors")
        """
        results = {
            'valid': [],
            'errors': [],
            'warnings': []
        }

        for key, security_id in self.symbol_map.items():
            if '_' not in key:
                continue  # Skip default mappings

            # Check security ID format
            if not security_id or not security_id.isdigit():
                results['errors'].append(f"{key}: Invalid security ID format '{security_id}'")
            elif len(security_id) < 3:
                results['warnings'].append(f"{key}: Security ID '{security_id}' seems short")
            else:
                results['valid'].append(key)

        logger.info(
            f"Validation: {len(results['valid'])} valid, "
            f"{len(results['warnings'])} warnings, "
            f"{len(results['errors'])} errors"
        )

        return results

