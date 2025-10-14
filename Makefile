# Makefile for NSE Adaptive Regime Trading System

.PHONY: help install install-dev test test-cov lint format type-check clean run-backtest train-qlib train-rl

# Default target
help:
	@echo "NSE Adaptive Regime Trading System - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run linting checks"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Trading:"
	@echo "  make setup-data       Setup Qlib data"
	@echo "  make train-qlib       Train Qlib models"
	@echo "  make train-rl         Train RL agents"
	@echo "  make run-backtest     Run backtest"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Clean temporary files"
	@echo "  make clean-all        Clean all generated files"

# Setup
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,test]"
	pre-commit install

# Code quality
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/
	pylint src/

type-check:
	mypy src/

# Testing
test:
	pytest -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-performance:
	pytest tests/performance/ -v -m performance

# Trading operations
setup-data:
	python scripts/setup_qlib_data.py --market NSE --region IN

train-qlib:
	python scripts/train_qlib_models.py --config config/qlib_config.yaml

train-rl:
	python scripts/train_rl_agent.py --algorithm ppo --timesteps 1000000

run-backtest:
	python scripts/run_backtest.py \
		--start-date 2023-01-01 \
		--end-date 2024-12-31 \
		--initial-capital 1000000 \
		--strategy adaptive_regime

# Maintenance
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

clean-all: clean
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf models/*
	rm -rf reports/*
	rm -rf backtest_results/*
	@echo "Warning: This deleted all generated data, models, and results!"

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server 8000

# Jupyter
notebook:
	jupyter lab

# Docker (if using Docker in the future)
docker-build:
	docker build -t nse-trading .

docker-run:
	docker run -it --rm nse-trading

# Monitoring
logs:
	tail -f logs/trading.log

logs-trade:
	tail -f logs/trades.log

