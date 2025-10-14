# Notebooks

This directory contains Jupyter notebooks for exploratory analysis and experiments.

## Structure

### exploratory/

Interactive exploration and analysis of data and features.

- `01_data_exploration.ipynb`: Initial data exploration and statistics
- `02_feature_analysis.ipynb`: Feature engineering and analysis
- `03_regime_analysis.ipynb`: Market regime characteristics

### experiments/

Experimental notebooks for testing strategies and models.

- `01_regime_detection_experiment.ipynb`: Regime detection methods comparison
- `02_qlib_models_experiment.ipynb`: Qlib model training and evaluation
- `03_rl_agent_experiment.ipynb`: RL agent training experiments
- `04_strategy_backtest.ipynb`: Strategy backtesting and analysis

## Usage

### Setup

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Navigate to the notebooks directory

### Best Practices

- Always run notebooks in order (01, 02, 03, etc.)
- Clear outputs before committing to git
- Document your findings in markdown cells
- Save important results to the `reports/` directory
- Use relative paths from project root

### Tips

- Use `%load_ext autoreload` and `%autoreload 2` for live code updates
- Leverage the project's utility modules for common operations
- Save long-running experiments as scripts in `scripts/`
- Use `.env` file for configuration (never commit credentials)

## Contributing

When creating new notebooks:

1. Use descriptive names with numbering
2. Include markdown documentation
3. Add to this README
4. Clear outputs before committing
5. Test that notebooks run from scratch

