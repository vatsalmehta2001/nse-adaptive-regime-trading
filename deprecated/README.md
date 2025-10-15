# ⚠️ DEPRECATED FILES

This folder contains legacy workarounds that are **no longer needed**.

---

## run_with_path.sh

**STATUS:** DEPRECATED  
**REASON:** Package is now properly installed with `pip install -e .`

### Old Way (DEPRECATED):
```bash
./run_with_path.sh python scripts/train_models.py
```

### New Way (CORRECT):
```bash
# Install package properly
pip install -e .

# Run directly
python scripts/train_models.py
```

---

## Migration Guide

If you were using `run_with_path.sh`:

### 1. Uninstall old setup:
```bash
pip uninstall nse-adaptive-regime-trading
```

### 2. Clean install:
```bash
pip install -e .
```

### 3. Verify:
```bash
python -c "from src.data_pipeline import DataPipeline; print('✅ Works!')"
```

### 4. Remove script calls:
- **Replace:** `./run_with_path.sh python script.py`
- **With:** `python script.py`

---

## Why This Change?

### Before (Bad):
- Required manual PYTHONPATH manipulation
- Different behavior in development vs production
- Not how Python packages should be installed
- Caused import errors

### After (Good):
- Standard Python package installation
- Works everywhere consistently
- Proper editable install
- No PYTHONPATH workarounds needed

---

**This file kept for historical reference only. Do not use in production.**

**Last Updated:** October 15, 2024

