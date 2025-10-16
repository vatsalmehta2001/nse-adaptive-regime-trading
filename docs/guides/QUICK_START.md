# Quick Start Guide

Get started with the NSE Adaptive Regime Trading System in minutes.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for version control)
- 8GB+ RAM recommended

## Installation

### 1. Clone the Repository (if not done)

```bash
git clone https://github.com/yourusername/nse-adaptive-regime-trading.git
cd nse-adaptive-regime-trading
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

**Important**: Update these in `.env`:
- `ZERODHA_API_KEY` - Your Zerodha API key
- `ZERODHA_API_SECRET` - Your Zerodha API secret
- Keep `TRADING_MODE=paper` for testing

### 5. Initialize Database

```bash
# This will create the DuckDB database
python -c "from src.utils.database import get_database; db = get_database(); db.initialize_schema()"
```

## Basic Usage

### Using Make Commands (Recommended)

```bash
# View all available commands
make help

# Setup Qlib data
make setup-data

# Train Qlib models
make train-qlib

# Train RL agent
make train-rl

# Run backtest
make run-backtest

# Run tests
make test
```

### Using Python Scripts Directly

```bash
# Setup Qlib data
python scripts/setup_qlib_data.py --market NSE --region IN

# Train Qlib models
python scripts/train_qlib_models.py --config config/qlib_config.yaml

# Train RL agent
python scripts/train_rl_agent.py --algorithm ppo --timesteps 1000000

# Run backtest
python scripts/run_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --initial-capital 1000000

# Paper trading
python scripts/run_live_trading.py --mode paper
```

### Using CLI

```bash
# Once installed, use the CLI
nse-trade setup-data --market NSE
nse-trade train-qlib --model lightgbm
nse-trade train-rl --algorithm ppo
nse-trade backtest --start 2023-01-01 --end 2024-12-31
nse-trade live --mode paper
```

## Development Setup

### Install Development Tools

```bash
# Install dev dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/ -m integration
pytest tests/performance/ -m performance
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ scripts/
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or use make command
make notebook
```

## Project Structure

```
nse-adaptive-regime-trading/
 config/                  # Configuration files (YAML)
 data/                    # Data storage (gitignored)
 docs/                    # Documentation
 logs/                    # Log files (gitignored)
 models/                  # Trained models (gitignored)
 notebooks/               # Jupyter notebooks
 scripts/                 # Utility scripts
 src/                     # Source code
    data_pipeline/       # Data acquisition
    feature_engineering/ # Feature generation
    regime_detection/    # Regime identification
    qlib_models/         # Qlib models
    rl_strategy/         # RL agents
    risk_management/     # Risk controls
    execution/           # Order execution
    backtesting/         # Backtesting engine
    utils/               # Utilities
 tests/                   # Test suite
```

## Configuration

### Key Configuration Files

- `config/data_sources.yaml` - Data source settings
- `config/qlib_config.yaml` - Qlib model configuration
- `config/tensortrade_config.yaml` - RL agent settings
- `config/zerodha_config.yaml` - Trading execution settings
- `config/trading_rules.yaml` - Risk management rules

### Environment Variables

Key variables in `.env`:

```bash
# Trading mode
TRADING_MODE=paper          # 'paper' or 'live'

# Database
DUCKDB_PATH=data/trading_db.duckdb

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/trading.log

# Zerodha API
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
```

## Common Tasks

### 1. Data Exploration

```bash
# Open data exploration notebook
jupyter lab notebooks/exploratory/01_data_exploration.ipynb
```

### 2. Strategy Backtesting

```bash
# Run backtest
python scripts/run_backtest.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --strategy adaptive_regime

# Analyze results
python scripts/analyze_backtest.py --results backtest_results.json
```

### 3. Model Training

```bash
# Train Qlib model
make train-qlib

# Train RL agent
make train-rl
```

### 4. Paper Trading

```bash
# Start paper trading
python scripts/run_live_trading.py --mode paper

# Monitor logs
tail -f logs/trading.log
```

## Troubleshooting

### Common Issues

**Import errors**:
```bash
# Ensure project is installed
pip install -e .
```

**Database errors**:
```bash
# Reinitialize database
rm data/trading_db.duckdb
python -c "from src.utils.database import get_database; get_database().initialize_schema()"
```

**API connection errors**:
- Check `.env` file has correct credentials
- Verify Zerodha API key is active
- Ensure you're not rate limited

### Getting Help

1. Check documentation in `docs/`
2. Review examples in `notebooks/`
3. Check existing issues on GitHub
4. Read the full README.md

## Next Steps

1. **Explore Data**: Start with data exploration notebooks
2. **Configure Strategy**: Customize `config/trading_rules.yaml`
3. **Backtest**: Run historical backtests to validate strategy
4. **Paper Trade**: Test with paper trading before live
5. **Monitor**: Set up monitoring and alerts
6. **Go Live**: Only after thorough testing!

## Safety Reminders

 **Important Safety Guidelines**:

1. Always test in paper trading mode first
2. Start with small position sizes
3. Monitor risk metrics continuously
4. Set appropriate stop losses
5. Never risk more than you can afford to lose
6. Keep `TRADING_MODE=paper` until fully tested
7. Regularly backup your database and configurations

## Support

- Documentation: `docs/`
- Examples: `notebooks/`
- Issues: GitHub Issues
- Email: your.email@example.com

Happy Trading!

