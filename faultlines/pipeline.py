import os
import duckdb
import numpy as np
import pandas as pd

# Path resolving to faultlines.duckdb in the root project folder
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "faultlines.duckdb")
)


def get_log_return_matrix(
    start_date: str,
    end_date: str,
    asset_type: str = None,  # 'Stock', 'ETF', or None
    min_obs_pct: float = 0.90,
    fill_missing: bool = True,
) -> pd.DataFrame:
  """Fetches an aligned log-return matrix (Days x Tickers) for a specified date range.

  Parameters:
      start_date (str): Format 'YYYY-MM-DD'
      end_date (str): Format 'YYYY-MM-DD'
      asset_type (str, optional): Filter by 'Stock' or 'ETF'. Defaults to None.
      min_obs_pct (float): Minimum proportion of non-null observations required
      per ticker.

  Returns:
      pd.DataFrame: Pivot matrix where index is Date and columns are Tickers.
  """
  # MODIFIED HERE: Use DB_PATH instead of "faultlines.duckdb"
  con = duckdb.connect(DB_PATH, read_only=True)

  asset_clause = f"WHERE Asset_Type = '{asset_type}'" if asset_type else ""

  query = f"""
        WITH calculated_returns AS (
            SELECT 
                Date, 
                REPLACE(Ticker, '.us', '') AS Ticker,
                ln(Close / LAG(Close) OVER (PARTITION BY Ticker ORDER BY Date)) AS log_return
            FROM raw_prices
            {asset_clause}
        )
        SELECT Date, Ticker, log_return
        FROM calculated_returns
        WHERE Date BETWEEN '{start_date}' AND '{end_date}'
    """

  df = con.execute(query).df()
  con.close()

  pivot_df = df.pivot(index="Date", columns="Ticker", values="log_return")
  pivot_df = pivot_df.dropna(how="all", axis=0)

  min_valid_rows = int(len(pivot_df) * min_obs_pct)
  pivot_df = pivot_df.dropna(thresh=min_valid_rows, axis=1)

  # Optionally fill remaining missing returns
  if fill_missing:
    pivot_df = pivot_df.fillna(0)

  return pivot_df


def get_ticker_metadata(tickers: list = None) -> pd.DataFrame:
  """Fetches SEC metadata (Company Name, CIK, SIC and SIC Description)
  for a list of tickers."""

  con = duckdb.connect(DB_PATH, read_only=True)

  query = """
        SELECT
            REPLACE(p.Ticker, '.us', '') AS Ticker,
            p.Asset_Type,
            m.Company_Name,
            m.cik_str,
            m.sic,
            m.sicDescription
        FROM (
            SELECT DISTINCT Ticker, Asset_Type
            FROM raw_prices
        ) p
        LEFT JOIN sec_metadata m
            ON UPPER(REPLACE(p.Ticker, '.us', '')) = UPPER(m.Ticker)
    """

  df = con.execute(query).df()
  con.close()

  if tickers:
    requested_tickers = [t.upper() for t in tickers]
    df = df[df["Ticker"].str.upper().isin(requested_tickers)]

  return df



if __name__ == "__main__":
  sample_stocks = get_log_return_matrix(
      "2016-01-01", "2016-12-31", asset_type="Stock"
  )
  print("\n--- Pipeline Test Output (Stocks Only) ---")
  print("Matrix Shape (Days x Tickers):", sample_stocks.shape)

  meta = get_ticker_metadata(list(sample_stocks.columns[:5]))
  print("\n--- Sample Ticker Metadata ---")
  print(meta)