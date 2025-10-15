# Installation Guide

## Issue: OpenBB Version and SSL Certificate Errors

### Problem 1: OpenBB Version Not Found
**Error**: `ERROR: No matching distribution found for openbb==4.2.4`

**Solution**: The version 4.2.4 doesn't exist. I've updated `requirements.txt` to use `openbb>=4.5.0` (the latest available version).

### Problem 2: SSL Certificate Error (macOS)
**Error**: `SSLError(SSLCertVerificationError('OSStatus -26276'))`

**Solution**: This is a macOS certificate issue. Follow these steps:

## Quick Fix Installation

### Step 1: Fix SSL Certificates (macOS)

```bash
# Option A: Install Python certificates (Recommended)
cd /Applications/Python\ 3.*/
./Install\ Certificates.command

# Option B: If you're using Anaconda/Conda
conda install -c anaconda openssl

# Option C: Update certifi
pip install --upgrade certifi
```

### Step 2: Install Dependencies

**Option A: Minimal Install (Recommended for Testing)**
```bash
# Use the minimal requirements first
pip install -r requirements-minimal.txt
```

**Option B: Full Install (Complete System)**
```bash
# Install the full requirements (this takes longer)
pip install -r requirements.txt
```

### Step 3: Install Optional Dependencies

Some packages require system-level dependencies:

#### TA-Lib (Technical Analysis Library)

**macOS**:
```bash
# Install via Homebrew
brew install ta-lib

# Then install Python wrapper
pip install ta-lib
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install ta-lib
pip install ta-lib
```

#### If you get errors with specific packages, skip them:
```bash
# Install without problematic packages
pip install -r requirements.txt --no-deps
pip install <specific-package> --no-deps
```

## Installation Steps (Complete)

### 1. Create Virtual Environment (Recommended)

```bash
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### 2. Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Fix SSL (if needed)

```bash
# macOS: Install certificates
/Applications/Python*/Install\ Certificates.command

# Or update certifi
pip install --upgrade certifi
```

### 4. Install Requirements

```bash
# Start with minimal (data pipeline only)
pip install -r requirements-minimal.txt

# Test if it works
python -c "import openbb; print(openbb.__version__)"

# If successful, install full requirements
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Test imports
python -c "
from src.data_pipeline import DataPipeline
from src.feature_engineering import TechnicalIndicators
print(' All imports successful!')
"
```

## Common Issues and Solutions

### Issue: Package X not found

**Solution**: Update the version in requirements.txt or use version ranges
```bash
# Instead of: package==1.2.3
# Use: package>=1.2.0,<2.0.0
```

### Issue: Conflicting dependencies

**Solution**: Install in batches
```bash
# Install core dependencies first
pip install openbb pandas numpy duckdb

# Then install others
pip install -r requirements.txt
```

### Issue: TA-Lib installation fails

**Solution**: Skip it for now (it's optional)
```bash
# Install everything except ta-lib
grep -v "ta-lib" requirements.txt > requirements-no-talib.txt
pip install -r requirements-no-talib.txt
```

### Issue: Qlib installation fails

**Solution**: Qlib can be complex to install
```bash
# Skip Qlib initially, it's for later stages
grep -v "qlib" requirements.txt > requirements-no-qlib.txt
pip install -r requirements-no-qlib.txt
```

## Test Data Pipeline

After installation, test the data pipeline:

```bash
# Test with a single symbol
python -c "
from src.data_pipeline import OpenBBDataFetcher
fetcher = OpenBBDataFetcher()
print('Fetching RELIANCE data...')
df = fetcher.fetch_equity_ohlcv(['RELIANCE'], '2024-01-01', '2024-01-31')
print(f' Fetched {len(df)} rows')
"
```

## Minimal Working Setup

If you just want to test the data pipeline quickly:

```bash
# Install only what's needed for data pipeline
pip install openbb pandas numpy duckdb pandera pydantic pyyaml python-dotenv loguru tqdm pytz holidays

# Test it
python scripts/setup_data_pipeline.py --symbols "RELIANCE" --years 1
```

## Environment Variables

Don't forget to set up your .env file:

```bash
cp .env.example .env
# Edit .env with your settings (if needed)
```

For the data pipeline, you don't need API keys initially (OpenBB uses free sources).

## Quick Verification

Run this to verify everything is working:

```python
python -c "
import sys
print('Python version:', sys.version)
print()

packages = ['openbb', 'pandas', 'numpy', 'duckdb', 'pandera', 'pydantic']
for package in packages:
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f' {package}: {version}')
    except ImportError as e:
        print(f' {package}: NOT INSTALLED')
"
```

## Next Steps

Once installed:

1. **Test data fetch**:
   ```bash
   python scripts/setup_data_pipeline.py --symbols "RELIANCE,TCS" --years 1
   ```

2. **Run tests**:
   ```bash
   pytest tests/unit/test_data_pipeline.py -v
   ```

3. **Check documentation**:
   - `DATA_PIPELINE_README.md` - Full pipeline documentation
   - `QUICK_REFERENCE.md` - Quick command reference

## Getting Help

If you still have issues:

1. Check Python version: `python --version` (needs 3.11+)
2. Check pip version: `pip --version`
3. Try installing in a fresh virtual environment
4. Install packages one at a time to find the problematic one

## Alternative: Conda Environment

If pip continues to have issues, use Conda:

```bash
# Create conda environment
conda create -n nse-trading python=3.11

# Activate it
conda activate nse-trading

# Install via conda where possible
conda install pandas numpy scipy scikit-learn

# Then install remaining via pip
pip install openbb duckdb pandera
```

---

**Summary**: The main issue was the OpenBB version. I've fixed it to use `>=4.5.0`. If you have SSL issues, run the certificate fix command for your system first.

