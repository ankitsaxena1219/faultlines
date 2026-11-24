"""Reusable DuckDB-backed data-fetching functions for FaultLines."""

from collections.abc import Sequence
from math import ceil
import os

import duckdb
import pandas as pd


DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "faultlines.duckdb")
)
SUPPORTED_ASSET_TYPES = {"Stock", "ETF"}


def _validate_request(
    start_date: str,
    end_date: str,
    asset_type: str | None,
    min_obs_pct: float,
) -> tuple[str, str]:
    """Validate public query arguments before sending them to DuckDB."""
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("start_date and end_date must be valid dates.") from exc

    if start > end:
        raise ValueError("start_date must be earlier than or equal to end_date.")
    if asset_type is not None and asset_type not in SUPPORTED_ASSET_TYPES:
        raise ValueError(
            f"asset_type must be one of {sorted(SUPPORTED_ASSET_TYPES)} or None."
        )
    if not 0 <= min_obs_pct <= 1:
        raise ValueError("min_obs_pct must be between 0 and 1.")

    return start.date().isoformat(), end.date().isoformat()


def get_log_return_matrix(
    start_date: str,
    end_date: str,
    asset_type: str | None = None,
    min_obs_pct: float = 0.90,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Fetch an aligned log-return matrix indexed by date.

    Missing returns remain null when ``fill_missing=False``. The default zero
    fill is retained for backwards compatibility, but callers doing statistical
    analysis should generally disable it to avoid treating missing observations
    as genuine zero returns.
    """
    start_date, end_date = _validate_request(
        start_date, end_date, asset_type, min_obs_pct
    )

    query = """
        WITH normalized_prices AS (
            SELECT
                CAST(Date AS DATE) AS Date,
                upper(replace(lower(Ticker), '.us', '')) AS Ticker,
                Close
            FROM raw_prices
            WHERE (? IS NULL OR Asset_Type = ?)
        ), calculated_returns AS (
            SELECT
                Date,
                Ticker,
                ln(Close / LAG(Close) OVER (
                    PARTITION BY Ticker ORDER BY Date
                )) AS log_return
            FROM normalized_prices
        )
        SELECT Date, Ticker, log_return
        FROM calculated_returns
        WHERE Date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        ORDER BY Date, Ticker
    """

    with duckdb.connect(DB_PATH, read_only=True) as con:
        df = con.execute(
            query, [asset_type, asset_type, start_date, end_date]
        ).df()

    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    pivot_df = df.pivot_table(
        index="Date", columns="Ticker", values="log_return", aggfunc="first"
    ).sort_index()
    pivot_df = pivot_df.dropna(how="all", axis=0)
    pivot_df = pivot_df.dropna(how="all", axis=1)

    min_valid_rows = ceil(len(pivot_df) * min_obs_pct)
    if min_valid_rows:
        pivot_df = pivot_df.dropna(thresh=min_valid_rows, axis=1)
    pivot_df = pivot_df.dropna(how="all", axis=0)

    if fill_missing:
        pivot_df = pivot_df.fillna(0)

    return pivot_df


def get_ticker_metadata(tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Fetch SEC metadata for all raw tickers or a requested ticker subset."""
    query = """
        SELECT DISTINCT
            upper(replace(lower(p.Ticker), '.us', '')) AS Ticker,
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
            ON upper(replace(lower(p.Ticker), '.us', '')) = upper(m.Ticker)
    """

    requested_tickers = None
    if tickers is not None:
        requested_tickers = pd.DataFrame({
            "Ticker": [
                str(ticker).upper().replace(".US", "")
                for ticker in tickers
            ]
        }).drop_duplicates()
        query += """
            WHERE upper(replace(lower(p.Ticker), '.us', '')) IN
                (SELECT Ticker FROM requested_tickers)
        """

    with duckdb.connect(DB_PATH, read_only=True) as con:
        if requested_tickers is not None:
            con.register("requested_tickers", requested_tickers)
        df = con.execute(query).df()
        if requested_tickers is not None:
            con.unregister("requested_tickers")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    sample_stocks = get_log_return_matrix(
        "2016-01-01", "2016-12-31", asset_type="Stock"
    )
    print("\n--- Pipeline Test Output (Stocks Only) ---")
    print("Matrix Shape (Days x Tickers):", sample_stocks.shape)
    print("\n--- Sample Ticker Metadata ---")
    print(get_ticker_metadata(list(sample_stocks.columns[:5])))
