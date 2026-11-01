import os
import duckdb
import numpy as np
import pandas as pd
from .pipeline import get_log_return_matrix, get_ticker_metadata

# Resolve path to faultlines.duckdb located in the project root
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "faultlines.duckdb")
)


def run_data_quality_checks():
  print("=" * 60)
  print("         FAULTLINES DATA PIPELINE - DATA QUALITY SUITE      ")
  print("=" * 60)

  con = duckdb.connect(DB_PATH, read_only=True)

  # 1. Primary Key Integrity & Duplicates Check
  pk_dupes = con.execute("""
        SELECT COUNT(*) 
        FROM (
            SELECT Date, Ticker, COUNT(*) 
            FROM raw_prices 
            GROUP BY Date, Ticker 
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

  print(f"\n1. Primary Key Integrity (Date, Ticker):")
  print(
      f"   -> Duplicate pairs: {pk_dupes}"
      f" {'[PASSED]' if pk_dupes == 0 else '[FAILED]'}"
  )

  # 2. Price Sanity Checks (Zero or Negative Prices)
  invalid_prices = con.execute("""
        SELECT COUNT(*) FROM raw_prices WHERE Close <= 0 OR Open <= 0
    """).fetchone()[0]

  print(f"\n2. Price Sanity Check (Close/Open <= 0):")
  print(
      f"   -> Invalid price rows: {invalid_prices}"
      f" {'[PASSED]' if invalid_prices == 0 else '[FAILED]'}"
  )

  # 3. High/Low Consistency Check (High < Low)
  hl_inconsistency = con.execute("""
        SELECT COUNT(*) FROM raw_prices WHERE High < Low
    """).fetchone()[0]

  print(f"\n3. High/Low Price Consistency Check (High < Low):")
  print(
      f"   -> Inconsistent High/Low rows: {hl_inconsistency}"
      f" {'[PASSED]' if hl_inconsistency == 0 else '[FAILED]'}"
  )

  # 4. SEC Metadata Linkage Coverage Check
  returns_df = get_log_return_matrix(
      "2016-01-01", "2016-12-31", asset_type="Stock"
  )
  meta_df = get_ticker_metadata(tickers=list(returns_df.columns))

  matched_count = meta_df["Company_Name"].notna().sum()
  total_count = len(meta_df)
  coverage_pct = (matched_count / total_count) * 100

  print(f"\n4. SEC Metadata Linkage Coverage:")
  print(
      f"   -> Mapped Tickers: {matched_count:,} / {total_count:,}"
      f" ({coverage_pct:.2f}%)"
  )

  # 5. Outlier Detection in Log Returns (|return| > 50% single-day change)
  log_returns = returns_df.values.flatten()
  extreme_pos = np.sum(log_returns > 0.405)
  extreme_neg = np.sum(log_returns < -0.693)

  print(f"\n5. Extreme Single-Day Log Return Outliers:")
  print(f"   -> Returns > +50%: {extreme_pos:,} instances")
  print(f"   -> Returns < -50%: {extreme_neg:,} instances")

  # 6. Matrix Sparsity Check
  zero_returns = np.sum(log_returns == 0)
  total_cells = log_returns.size
  zero_pct = (zero_returns / total_cells) * 100

  print(f"\n6. Matrix Sparsity Check:")
  print(
      f"   -> Filled Zero Entries: {zero_returns:,} / {total_cells:,}"
      f" ({zero_pct:.2f}%)"
  )

  con.close()
  print("\n" + "=" * 60)
  print("                  DATA QUALITY AUDIT COMPLETE               ")
  print("=" * 60)


if __name__ == "__main__":
  run_data_quality_checks()