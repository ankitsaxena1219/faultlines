# Faultlines: Financial Data & Risk Modeling Pipeline

Modular time-series data pipeline connecting historical US Stock/ETF price feeds with SEC metadata.

## Quickstart for Task 2 Developers

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Build Database
Run ingestion to populate `faultlines.duckdb`:
```powershell
python -m faultlines.ingestion
```

### 3. Fetch Data for Task 2 (Rolling Covariance / Risk Models)
```python
from faultlines import get_log_return_matrix, get_ticker_metadata

# Fetch log-return matrix (Days x Tickers)
returns_df = get_log_return_matrix(
    start_date="2016-01-01", 
    end_date="2016-12-31", 
    asset_type="Stock", 
    min_obs_pct=0.90
)

# Metadata lookup
metadata_df = get_ticker_metadata(tickers=list(returns_df.columns))
```

### 4. Run Data Quality Suite
```python
from faultlines import run_data_quality_checks

run_data_quality_checks()
```
