"""
Helper Utilities Module.

Common utility functions used across the project.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file has unsupported format
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration

    Raises:
        ValueError: If config file has unsupported format
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif suffix == ".json":
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        YAML content as dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to YAML file.

    Args:
        data: Data to save
        file_path: Path to save YAML file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        JSON content as dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root() -> Path:
    """
    Get project root directory.

    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils/
    return Path(__file__).parent.parent.parent


def resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    """
    Resolve path relative to project root or specified directory.

    Args:
        path: Path to resolve
        relative_to: Base directory (defaults to project root)

    Returns:
        Resolved absolute path
    """
    path = Path(path)

    if path.is_absolute():
        return path

    base = relative_to if relative_to else get_project_root()
    return (base / path).resolve()


def format_currency(amount: float, currency: str = "INR", decimals: int = 2) -> str:
    """
    Format amount as currency string.

    Args:
        amount: Amount to format
        currency: Currency code (default: INR)
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    symbols = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
    }

    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format (0.05 = 5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


class ConfigModel(BaseModel):
    """
    Base configuration model with Pydantic validation.

    Provides common configuration loading functionality.
    """

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "ConfigModel":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Configuration instance
        """
        data = load_yaml(file_path)
        return cls(**data)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "ConfigModel":
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Configuration instance
        """
        data = load_json(file_path)
        return cls(**data)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save YAML file
        """
        save_yaml(self.dict(), file_path)

    def to_json(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            file_path: Path to save JSON file
        """
        save_json(self.dict(), file_path)


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from list.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def flatten(nested_list: list) -> list:
    """
    Flatten nested list.

    Args:
        nested_list: Nested list to flatten

    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

