"""
Broker factory for easy broker switching.

This module provides a factory pattern for creating broker instances.
It simplifies broker instantiation and makes switching between brokers
(DhanHQ, Kite, Paper) as simple as changing one line of code.

Classes:
    BrokerFactory: Factory for creating broker instances

Example:
    >>> from src.execution.broker_factory import BrokerFactory
    >>> 
    >>> # Development with sandbox
    >>> broker = BrokerFactory.create("DHAN", mode="SANDBOX")
    >>> 
    >>> # Paper trading
    >>> broker = BrokerFactory.create("PAPER", initial_capital=1000000)
    >>> 
    >>> # Live trading (later)
    >>> broker = BrokerFactory.create("DHAN", mode="LIVE")
    >>> 
    >>> # From configuration file
    >>> broker = BrokerFactory.from_config()
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from src.execution.broker_interface import BrokerInterface
from src.execution.dhan_broker import DhanBroker
from src.execution.kite_broker import KiteBroker
from src.execution.paper_broker import PaperTradingBroker
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BrokerFactory:
    """
    Factory for creating broker instances.

    Simplifies broker instantiation and switching. Supports creating
    brokers from configuration files or explicit parameters.
    """

    @staticmethod
    def create(broker_type: str, **kwargs) -> BrokerInterface:
        """
        Create broker instance.

        Args:
            broker_type: 'DHAN', 'KITE', or 'PAPER'
            **kwargs: Broker-specific parameters

        Returns:
            BrokerInterface implementation

        Raises:
            ValueError: If broker_type unknown

        Example:
            >>> # DhanHQ sandbox
            >>> broker = BrokerFactory.create("DHAN", mode="SANDBOX")
            >>> 
            >>> # Paper trading
            >>> broker = BrokerFactory.create("PAPER", initial_capital=1000000)
        """
        broker_type = broker_type.upper()

        logger.info(f"Creating broker: {broker_type}")

        if broker_type == "DHAN":
            return BrokerFactory._create_dhan(**kwargs)
        elif broker_type == "KITE":
            return BrokerFactory._create_kite(**kwargs)
        elif broker_type == "PAPER":
            return BrokerFactory._create_paper(**kwargs)
        else:
            raise ValueError(
                f"Unknown broker type: {broker_type}. "
                f"Supported: DHAN, KITE, PAPER"
            )

    @staticmethod
    def from_config(
        config_path: Optional[str] = None
    ) -> BrokerInterface:
        """
        Create broker from configuration file.

        Args:
            config_path: Path to broker configuration
                        (default: config/broker_config.yaml)

        Returns:
            Configured broker instance

        Example:
            >>> broker = BrokerFactory.from_config()
            >>> # Uses active_broker from config/broker_config.yaml
        """
        if config_path is None:
            config_path = "config/broker_config.yaml"

        config = load_config(config_path)

        broker_type = config.get("active_broker", "PAPER").upper()
        broker_settings = config.get("brokers", {}).get(broker_type.lower(), {})

        logger.info(f"Creating broker from config: {broker_type}")

        return BrokerFactory.create(broker_type, **broker_settings)

    @staticmethod
    def _create_dhan(
        mode: str = "SANDBOX",
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        config_path: str = "config/broker_config.yaml",
        **kwargs,
    ) -> DhanBroker:
        """
        Create DhanHQ broker.

        Args:
            mode: 'SANDBOX' or 'LIVE'
            client_id: DhanHQ client ID (if not provided, read from config/env)
            access_token: DhanHQ access token (if not provided, read from config/env)
            config_path: Path to config file
            **kwargs: Additional DhanBroker parameters

        Returns:
            DhanBroker instance

        Raises:
            ValueError: If credentials not provided and not found in config
        """
        # Try to get credentials from parameters, then config, then environment
        if client_id is None or access_token is None:
            # Try config file
            try:
                config = load_config(config_path)
                dhan_config = config.get("brokers", {}).get("dhan", {})

                if client_id is None:
                    client_id = dhan_config.get("client_id")
                if access_token is None:
                    access_token = dhan_config.get("access_token")

            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Try environment variables
        if client_id is None:
            client_id = os.getenv("DHAN_CLIENT_ID")
        if access_token is None:
            access_token = os.getenv("DHAN_ACCESS_TOKEN")

        # Expand environment variables in credentials (e.g., ${VAR_NAME})
        if client_id and client_id.startswith("${") and client_id.endswith("}"):
            env_var = client_id[2:-1]
            client_id = os.getenv(env_var)

        if access_token and access_token.startswith("${") and access_token.endswith("}"):
            env_var = access_token[2:-1]
            access_token = os.getenv(env_var)

        if not client_id or not access_token:
            raise ValueError(
                "DhanHQ credentials not found. Please provide client_id and "
                "access_token, or set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN "
                "environment variables, or configure in broker_config.yaml"
            )

        logger.info(f"Creating DhanHQ broker in {mode} mode")

        # Extract connection settings from kwargs or config
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1.0)

        if "connection" in kwargs:
            max_retries = kwargs["connection"].get("max_retries", max_retries)
            retry_delay = kwargs["connection"].get("retry_delay", retry_delay)

        return DhanBroker(
            client_id=client_id,
            access_token=access_token,
            mode=mode,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    @staticmethod
    def _create_kite(
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        config_path: str = "config/broker_config.yaml",
        **kwargs,
    ) -> KiteBroker:
        """
        Create Kite Connect broker.

        Args:
            api_key: Kite Connect API key
            api_secret: Kite Connect API secret
            access_token: Kite Connect access token
            config_path: Path to config file
            **kwargs: Additional KiteBroker parameters

        Returns:
            KiteBroker instance

        Raises:
            ValueError: If credentials not provided and not found in config
        """
        # Try to get credentials from parameters, then config, then environment
        if api_key is None or access_token is None:
            # Try config file
            try:
                config = load_config(config_path)
                kite_config = config.get("brokers", {}).get("kite", {})

                if api_key is None:
                    api_key = kite_config.get("api_key")
                if api_secret is None:
                    api_secret = kite_config.get("api_secret")
                if access_token is None:
                    access_token = kite_config.get("access_token")

            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Try environment variables
        if api_key is None:
            api_key = os.getenv("KITE_API_KEY")
        if api_secret is None:
            api_secret = os.getenv("KITE_API_SECRET")
        if access_token is None:
            access_token = os.getenv("KITE_ACCESS_TOKEN")

        # Expand environment variables in credentials
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)

        if api_secret and api_secret.startswith("${") and api_secret.endswith("}"):
            env_var = api_secret[2:-1]
            api_secret = os.getenv(env_var)

        if access_token and access_token.startswith("${") and access_token.endswith("}"):
            env_var = access_token[2:-1]
            access_token = os.getenv(env_var)

        if not api_key:
            raise ValueError(
                "Kite Connect API key not found. Please provide api_key, "
                "or set KITE_API_KEY environment variable, "
                "or configure in broker_config.yaml"
            )

        if not access_token:
            logger.warning(
                "Kite Connect access token not found. You'll need to "
                "call set_access_token() before trading."
            )

        logger.info("Creating Kite Connect broker")

        # Extract connection settings from kwargs or config
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1.0)

        if "connection" in kwargs:
            max_retries = kwargs["connection"].get("max_retries", max_retries)
            retry_delay = kwargs["connection"].get("retry_delay", retry_delay)

        return KiteBroker(
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    @staticmethod
    def _create_paper(
        initial_capital: float = 1000000,
        data_broker_type: str = "DHAN",
        data_broker_mode: str = "SANDBOX",
        commission_pct: float = 0.03,
        slippage_pct: float = 0.10,
        fill_probability: float = 0.95,
        config_path: str = "config/broker_config.yaml",
        **kwargs,
    ) -> PaperTradingBroker:
        """
        Create paper trading broker.

        Args:
            initial_capital: Starting capital (default ₹10 lakhs)
            data_broker_type: Broker to use for live data (default DHAN)
            data_broker_mode: Mode for data broker (default SANDBOX)
            commission_pct: Commission percentage (default 0.03%)
            slippage_pct: Slippage percentage (default 0.10%)
            fill_probability: Limit order fill probability (default 95%)
            config_path: Path to config file
            **kwargs: Additional parameters

        Returns:
            PaperTradingBroker instance
        """
        # Try to get simulation settings from config
        try:
            config = load_config(config_path)
            paper_config = config.get("brokers", {}).get("paper", {})
            sim_config = paper_config.get("simulation", {})

            initial_capital = kwargs.get("initial_capital", paper_config.get("initial_capital", initial_capital))
            commission_pct = sim_config.get("commission_pct", commission_pct)
            slippage_pct = sim_config.get("slippage_pct", slippage_pct)
            fill_probability = sim_config.get("fill_probability", fill_probability)
            data_broker_type = paper_config.get("data_broker", data_broker_type)

        except Exception as e:
            logger.warning(f"Could not load paper trading config: {e}")

        # Create data broker for live market data
        logger.info(f"Creating paper trading broker with ₹{initial_capital:,.0f}")
        logger.info(f"Using {data_broker_type} in {data_broker_mode} mode for live data")

        data_broker = BrokerFactory.create(
            data_broker_type,
            mode=data_broker_mode,
        )

        return PaperTradingBroker(
            initial_capital=initial_capital,
            data_broker=data_broker,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            fill_probability=fill_probability,
        )

    @staticmethod
    def list_available_brokers() -> Dict[str, Dict[str, Any]]:
        """
        List all available brokers and their status.

        Returns:
            Dictionary with broker information

        Example:
            >>> brokers = BrokerFactory.list_available_brokers()
            >>> print(brokers)
            {
                'DHAN': {'implemented': True, 'supports_live': True, ...},
                'KITE': {'implemented': False, ...},
                'PAPER': {'implemented': True, ...}
            }
        """
        return {
            "DHAN": {
                "implemented": True,
                "supports_live": True,
                "supports_sandbox": True,
                "description": "DhanHQ broker with sandbox and live trading support",
            },
            "KITE": {
                "implemented": True,
                "supports_live": True,
                "supports_sandbox": False,
                "description": "Kite Connect broker for Zerodha trading",
            },
            "PAPER": {
                "implemented": True,
                "supports_live": False,
                "supports_sandbox": False,
                "description": "Paper trading with live data simulation",
            },
        }

