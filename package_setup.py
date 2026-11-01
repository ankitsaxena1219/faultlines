import os
import shutil

PROJECT_ROOT = os.getcwd()
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "faultlines")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# 1. Create directory structure
os.makedirs(PACKAGE_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# 2. Move existing root modules into faultlines/ package directory if present
file_mappings = {
    "ingest_to_duckdb.py": "ingestion.py",
    "data_pipeline.py": "pipeline.py",
    "data_quality_checks.py": "quality.py",
}

for src_name, dst_name in file_mappings.items():
  src_path = os.path.join(PROJECT_ROOT, src_name)
  dst_path = os.path.join(PACKAGE_DIR, dst_name)
  if os.path.exists(src_path):
    shutil.move(src_path, dst_path)
    print(f"Moved {src_name} -> faultlines/{dst_name}")

if os.path.exists(os.path.join(PROJECT_ROOT, "explore_duplicates.ipynb")):
  shutil.move(
      os.path.join(PROJECT_ROOT, "explore_duplicates.ipynb"),
      os.path.join(NOTEBOOKS_DIR, "01_data_exploration.ipynb"),
  )
  print("Moved explore_duplicates.ipynb -> notebooks/01_data_exploration.ipynb")

# 3. Create setup.py for editable installation (pip install -e .)
setup_py_content = """from setuptools import setup, find_packages

setup(
    name="faultlines",
    version="0.1.0",
    packages=find_packages(),
)
"""
with open(os.path.join(PROJECT_ROOT, "setup.py"), "w") as f:
  f.write(setup_py_content)

# 4. Create faultlines/__init__.py
init_content = """from .pipeline import get_log_return_matrix, get_ticker_metadata
from .quality import run_data_quality_checks

__all__ = ["get_log_return_matrix", "get_ticker_metadata", "run_data_quality_checks"]
"""
with open(os.path.join(PACKAGE_DIR, "__init__.py"), "w") as f:
  f.write(init_content)

# 5. Generate requirements.txt
requirements_content = """duckdb>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
kagglehub>=0.2.0
jupyter
matplotlib
setuptools
"""
with open(os.path.join(PROJECT_ROOT, "requirements.txt"), "w") as f:
  f.write(requirements_content)

# 6. Generate .gitignore
gitignore_content = """venv/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
faultlines.duckdb
data/
"""
with open(os.path.join(PROJECT_ROOT, ".gitignore"), "w") as f:
  f.write(gitignore_content)

# 7. Generate README.md safely using string concatenation
ticks = "```"
readme_content = f"""# Faultlines: Financial Data & Risk Modeling Pipeline

Modular time-series data pipeline connecting historical US Stock/ETF price feeds with SEC metadata.

## Quickstart for Task 2 Developers

### 1. Environment Setup
{ticks}powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .
{ticks}

### 2. Build Database
Run ingestion to populate `faultlines.duckdb`:
{ticks}powershell
python -m faultlines.ingestion
{ticks}

### 3. Fetch Data for Task 2 (Rolling Covariance / Risk Models)
{ticks}python
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
{ticks}

### 4. Run Data Quality Suite
{ticks}python
from faultlines import run_data_quality_checks

run_data_quality_checks()
{ticks}
"""

with open(os.path.join(PROJECT_ROOT, "README.md"), "w") as f:
  f.write(readme_content)

print("\n--- Package Assembly Complete ---")