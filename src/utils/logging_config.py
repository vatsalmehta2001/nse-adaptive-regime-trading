"""
Logging Configuration Module.

Provides structured logging using Loguru with JSON formatting,
rotation, and multiple output handlers.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Remove default logger
logger.remove()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "text",
    rotation: str = "500 MB",
    retention: str = "30 days",
    enable_console: bool = True,
) -> None:
    """
    Configure logging with Loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log format ("text" or "json")
        rotation: Log rotation policy
        retention: Log retention policy
        enable_console: Enable console output
    """
    # Console handler
    if enable_console:
        if log_format == "json":
            console_format = (
                "{{\"time\": \"{time:YYYY-MM-DD HH:mm:ss.SSS}\", "
                "\"level\": \"{level}\", \"module\": \"{name}\", "
                "\"function\": \"{function}\", \"line\": {line}, "
                "\"message\": \"{message}\"}}\n"
            )
        else:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>\n"
            )

        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=log_format == "text",
            backtrace=True,
            diagnose=True,
        )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_format == "json":
            file_format = (
                "{{\"time\": \"{time:YYYY-MM-DD HH:mm:ss.SSS}\", "
                "\"level\": \"{level}\", \"module\": \"{name}\", "
                "\"function\": \"{function}\", \"line\": {line}, "
                "\"message\": \"{message}\", \"extra\": {extra}}}\n"
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {message}\n"
            )

        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Async logging
        )


def get_logger(name: str, **kwargs: Any) -> Any:
    """
    Get a logger instance with optional context.

    Args:
        name: Logger name (typically __name__)
        **kwargs: Additional context to bind to logger

    Returns:
        Configured logger instance
    """
    if kwargs:
        return logger.bind(name=name, **kwargs)
    return logger.bind(name=name)


def log_trade(
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    strategy: str,
    **kwargs: Any,
) -> None:
    """
    Log a trade execution with structured data.

    Args:
        symbol: Trading symbol
        action: Trade action (BUY, SELL, etc.)
        quantity: Number of shares
        price: Execution price
        strategy: Strategy name
        **kwargs: Additional trade metadata
    """
    trade_data: Dict[str, Any] = {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "strategy": strategy,
        "value": quantity * price,
        **kwargs,
    }

    logger.bind(**trade_data).info(f"Trade executed: {action} {quantity} {symbol} @ {price}")


def log_performance(
    metric: str,
    value: float,
    period: str = "daily",
    **kwargs: Any,
) -> None:
    """
    Log performance metrics.

    Args:
        metric: Metric name (e.g., 'return', 'sharpe_ratio')
        value: Metric value
        period: Time period (daily, weekly, monthly)
        **kwargs: Additional context
    """
    perf_data: Dict[str, Any] = {
        "metric": metric,
        "value": value,
        "period": period,
        **kwargs,
    }

    logger.bind(**perf_data).info(f"Performance metric: {metric} = {value:.4f} ({period})")


def log_risk_event(
    event_type: str,
    severity: str,
    message: str,
    **kwargs: Any,
) -> None:
    """
    Log risk management events.

    Args:
        event_type: Type of risk event (e.g., 'stop_loss', 'var_breach')
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        message: Event description
        **kwargs: Additional context
    """
    risk_data: Dict[str, Any] = {
        "event_type": event_type,
        "severity": severity,
        **kwargs,
    }

    log_level = {
        "LOW": "info",
        "MEDIUM": "warning",
        "HIGH": "error",
        "CRITICAL": "critical",
    }.get(severity.upper(), "warning")

    log_func = getattr(logger.bind(**risk_data), log_level)
    log_func(f"Risk event: {event_type} - {message}")


# Initialize logging on module import
try:
    import os
    from dotenv import load_dotenv

    load_dotenv()

    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE_PATH", "logs/trading.log"),
        log_format=os.getenv("LOG_FORMAT", "text"),
        rotation=os.getenv("LOG_ROTATION", "500 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
    )
except Exception as e:
    # Fallback to basic logging
    logger.add(sys.stdout, level="INFO")
    logger.warning(f"Failed to load logging configuration from .env: {e}")

