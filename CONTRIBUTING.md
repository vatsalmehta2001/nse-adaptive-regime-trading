# Contributing to NSE Adaptive Regime Trading System

Thank you for your interest in contributing! This is a personal portfolio project, but suggestions and improvements are welcome.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details

### Suggesting Enhancements

For feature requests:

1. Describe the enhancement clearly
2. Explain the use case
3. Consider implementation complexity
4. Discuss potential alternatives

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Run code formatting (`black src/ tests/`)
7. Run type checking (`mypy src/`)
8. Commit with clear messages
9. Push to your branch
10. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- Virtual environment tool

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/nse-adaptive-regime-trading.git
cd nse-adaptive-regime-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -e ".[dev,test]"

# Setup pre-commit hooks
pre-commit install
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints to all functions
- Write comprehensive docstrings (Google style)

### Example

```python
def calculate_returns(
    prices: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Price series with datetime index
        method: Return calculation method ('simple' or 'log')

    Returns:
        Returns series with same index as prices

    Raises:
        ValueError: If method is not supported
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unsupported method: {method}")
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies

### Test Example

```python
import pytest
from src.utils.helpers import format_currency


class TestHelpers:
    """Test helper functions."""

    def test_format_currency_inr(self) -> None:
        """Test INR currency formatting."""
        assert format_currency(1000000) == "₹1,000,000.00"

    def test_format_currency_usd(self) -> None:
        """Test USD currency formatting."""
        assert format_currency(1000000, "USD") == "$1,000,000.00"

    @pytest.mark.parametrize(
        "amount,currency,expected",
        [
            (1000, "INR", "₹1,000.00"),
            (1000, "USD", "$1,000.00"),
            (1000.50, "INR", "₹1,000.50"),
        ],
    )
    def test_format_currency_parametrized(
        self, amount: float, currency: str, expected: str
    ) -> None:
        """Test currency formatting with multiple cases."""
        assert format_currency(amount, currency) == expected
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    Short description.

    Longer description if needed. Can span multiple lines.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> function_name(value1, value2)
        result
    """
    pass
```

### Adding Documentation

- Update relevant .md files in `docs/`
- Add inline comments for complex logic
- Create notebooks for examples
- Keep README.md up to date

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Example

```
feat(regime-detection): Add Wasserstein distance method

Implement Wasserstein distance calculation for comparing
return distributions across different time periods.

Closes #42
```

## Project Structure

Familiarize yourself with the project structure:

```
nse-adaptive-regime-trading/
 src/                    # Source code
    data_pipeline/      # Data acquisition and processing
    feature_engineering/# Feature generation
    regime_detection/   # Market regime identification
    qlib_models/        # Qlib model implementations
    rl_strategy/        # RL agent training
    risk_management/    # Risk controls
    execution/          # Order execution
    backtesting/        # Backtesting engine
    utils/              # Utilities
 tests/                  # Test suite
 config/                 # Configuration files
 scripts/                # Utility scripts
 notebooks/              # Jupyter notebooks
 docs/                   # Documentation
```

## Questions?

Feel free to:
- Open an issue for questions
- Reach out via email (see README)
- Check existing documentation

## Code of Conduct

### Our Pledge

We pledge to make participation in this project a harassment-free experience for everyone.

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Enforcement

Unacceptable behavior may result in being blocked from the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 

