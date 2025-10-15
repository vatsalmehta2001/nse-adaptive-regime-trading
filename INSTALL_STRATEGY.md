# Installation Strategy

##  Smart Installation Approach

Install packages in stages to avoid conflicts and identify issues early.

---

##  **Stage 1: Core Data Pipeline (Required)**

This gets you up and running with the data pipeline:

```bash
pip install -r requirements-minimal.txt
```

**What you get**:
-  OpenBB for market data
-  DuckDB for storage
-  Pandas/NumPy for data processing
-  Data validation (Pandera)
-  Configuration management
-  Logging and utilities

**Test it**:
```bash
python -c "from src.data_pipeline import DataPipeline; print(' Stage 1 Complete!')"
```

---

##  **Stage 2: Machine Learning (Optional)**

Add ML capabilities for Qlib models:

```bash
pip install scikit-learn>=1.5.0 lightgbm>=4.0.0 xgboost>=2.0.0 statsmodels>=0.14.0
```

**What you get**:
-  Basic ML algorithms
-  Gradient boosting models
-  Statistical models

---

##  **Stage 3: Technical Analysis (Optional)**

Add technical indicators:

```bash
pip install pandas-ta matplotlib seaborn plotly
```

**For TA-Lib (optional, requires system library)**:
```bash
# macOS
brew install ta-lib
pip install ta-lib

# Linux
sudo apt-get install ta-lib
pip install ta-lib
```

---

##  **Stage 4: Time Series & Regime Detection (Optional)**

Add advanced time series analysis:

```bash
pip install arch>=6.3.0 hmmlearn>=0.3.0
```

---

##  **Stage 5: Deep Learning & RL (Advanced, Optional)**

For reinforcement learning strategies:

```bash
# PyTorch (choose based on your system)
# CPU only:
pip install torch>=2.3.0

# With CUDA (Linux/Windows):
# Visit: https://pytorch.org/get-started/locally/

# Then install RL packages:
pip install stable-baselines3>=2.3.0 gymnasium>=0.29.0
```

---

##  **Stage 6: Qlib Framework (Advanced, Optional)**

Microsoft Qlib has complex dependencies:

```bash
# Install Qlib
pip install qlib>=0.9.0

# Note: May require additional dependencies
# Follow: https://qlib.readthedocs.io/
```

---

##  **Stage 7: Development Tools (Optional)**

For development and code quality:

```bash
pip install black isort flake8 mypy pylint pytest pytest-cov
```

---

##  **Stage 8: Jupyter & Visualization (Optional)**

For interactive development:

```bash
pip install jupyter jupyterlab ipython ipywidgets
```

---

##  **Recommended Installation Path**

### For Quick Testing:
```bash
# Just the essentials
pip install -r requirements-minimal.txt
python scripts/setup_data_pipeline.py --symbols "RELIANCE" --years 1
```

### For Full Development:
```bash
# Stage 1: Core (required)
pip install -r requirements-minimal.txt

# Stage 2: ML & Analysis
pip install scikit-learn lightgbm xgboost statsmodels pandas-ta matplotlib seaborn

# Stage 3: Development tools
pip install black pytest jupyterlab

# Stage 4: Advanced (only if needed)
# pip install torch stable-baselines3 qlib
```

---

##  **Troubleshooting**

### Issue: Package installation fails

**Solution 1: Install one by one**
```bash
# Install core packages individually
pip install openbb
pip install pandas numpy
pip install duckdb
pip install pandera
```

**Solution 2: Use conda for problematic packages**
```bash
conda install -c conda-forge numpy pandas scipy scikit-learn
pip install openbb duckdb pandera  # Install remaining via pip
```

### Issue: Version conflicts

**Solution: Use version ranges**
```bash
# Instead of exact versions, use ranges
pip install "pandas>=2.2.0,<3.0.0" "numpy>=1.26.0,<2.0.0"
```

### Issue: SSL Certificate errors (macOS)

**Solution: Fix Python certificates**
```bash
/Applications/Python*/Install\ Certificates.command
```

---

##  **Verification Checklist**

After each stage, verify installation:

### Stage 1 (Core):
```python
python -c "
from src.data_pipeline import DataPipeline, OpenBBDataFetcher
from src.feature_engineering import TechnicalIndicators
print(' Core components working!')
"
```

### Stage 2 (ML):
```python
python -c "
import lightgbm, xgboost, sklearn
print(' ML libraries installed!')
"
```

### Stage 3 (Visualization):
```python
python -c "
import matplotlib, seaborn, plotly
print(' Visualization libraries installed!')
"
```

---

##  **Quick Start After Installation**

Once Stage 1 is complete:

```bash
# 1. Fetch real data
python scripts/setup_data_pipeline.py --symbols "RELIANCE,TCS" --years 1

# 2. Query the database
python -c "
from src.data_pipeline import DataStorageManager
storage = DataStorageManager()
df = storage.query_ohlcv(symbols=['RELIANCE'])
print(f' {len(df)} rows in database')
"

# 3. Generate features
python -c "
from src.data_pipeline import DataStorageManager
from src.feature_engineering import TechnicalIndicators

storage = DataStorageManager()
df = storage.query_ohlcv(symbols=['RELIANCE'])

indicators = TechnicalIndicators()
df_features = indicators.generate_all_features(df)
print(f' Generated {len(df_features.columns)} features')
"
```

---

##  **Best Practices**

1. **Always use virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   ```

2. **Start minimal, add as needed**:
   - Install `requirements-minimal.txt` first
   - Add packages only when you need them
   - Test after each addition

3. **Keep a working state**:
   ```bash
   # After successful installation, save it
   pip freeze > requirements-working.txt
   ```

4. **Document any issues**:
   - Note which packages needed special handling
   - Save workarounds for future reference

---

##  **What Each File Does**

- **requirements-minimal.txt**: Core data pipeline only (~12 packages)
- **requirements.txt**: Full system with all features (~50+ packages)
- **requirements-working.txt**: Your actual installed versions (create with `pip freeze`)

---

##  **Success Criteria**

You're ready to proceed when:

 `pip install -r requirements-minimal.txt` succeeds  
 `from src.data_pipeline import DataPipeline` works  
 Can fetch real market data  
 Can store data in DuckDB  
 Can generate technical indicators  

**Everything else is optional enhancement!**

---

##  **Still Having Issues?**

1. Check Python version: `python --version` (needs 3.11+)
2. Try fresh virtual environment
3. Install packages one at a time to find the problematic one
4. Check system-specific requirements (macOS vs Linux vs Windows)
5. Consider using Conda for scientific packages

---

**Remember**: The data pipeline works with just `requirements-minimal.txt`. Everything else is optional!

