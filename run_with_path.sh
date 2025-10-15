#!/bin/bash
# Helper script to run Python commands with correct PYTHONPATH
# Usage: ./run_with_path.sh python scripts/train_models.py --symbols NIFTY50

export PYTHONPATH="/Users/vatsalmehta/Developer/nse-adaptive-regime-trading:$PYTHONPATH"
"$@"

