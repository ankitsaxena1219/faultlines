"""Data quality checks and limitations report for the FaultLines dataset."""

import os

import duckdb
import numpy as np

from .pipeline import get_log_return_matrix, get_ticker_metadata


DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "faultlines.duckdb")
)

DATASET_LIMITATIONS = (
    "SEC metadata uses the current SEC ticker universe and may not resolve "
    "historical symbols.",
    "The price source is not guaranteed to include delisted firms, so results "
    "may contain survivorship bias.",
    "Missing price observations are not equivalent to zero returns; use "
    "fill_missing=False for statistical analysis.",
)


def _status(value: int) -> str:
    return "PASSED" if value == 0 else "FAILED"


def run_data_quality_checks(
    start_date: str = "2016-01-01",
    end_date: str = "2016-12-31",
    asset_type: str = "Stock",
    emit: bool = True,
) -> dict:
    """Run quality checks and return a reusable structured report."""
    with duckdb.connect(DB_PATH, read_only=True) as con:
        raw = con.execute(
            """
            SELECT
                count(*) AS row_count,
                count(DISTINCT Ticker) AS ticker_count,
                min(Date) AS min_date,
                max(Date) AS max_date,
                sum(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) AS null_dates,
                sum(CASE WHEN Open IS NULL OR High IS NULL OR Low IS NULL
                         OR Close IS NULL THEN 1 ELSE 0 END) AS null_ohlc,
                sum(CASE WHEN Close <= 0 OR Open <= 0 OR High < Low
                         THEN 1 ELSE 0 END) AS invalid_price_rows
            FROM raw_prices
            """
        ).fetchone()
        duplicate_pairs = con.execute(
            """
            SELECT count(*) FROM (
                SELECT Date, Ticker
                FROM raw_prices
                GROUP BY Date, Ticker
                HAVING count(*) > 1
            )
            """
        ).fetchone()[0]

    returns_df = get_log_return_matrix(
        start_date,
        end_date,
        asset_type=asset_type,
        fill_missing=False,
    )
    metadata_df = get_ticker_metadata(tickers=list(returns_df.columns))
    total_tickers = len(returns_df.columns)
    matched_tickers = int(metadata_df["Company_Name"].notna().sum())
    sic_tickers = int(metadata_df["sic"].notna().sum())
    total_cells = int(returns_df.size)
    missing_cells = int(returns_df.isna().sum().sum())
    finite_returns = returns_df.to_numpy(dtype=float)
    finite_returns = finite_returns[np.isfinite(finite_returns)]

    report = {
        "raw": {
            "rows": int(raw[0]),
            "tickers": int(raw[1]),
            "min_date": raw[2],
            "max_date": raw[3],
            "null_dates": int(raw[4]),
            "null_ohlc": int(raw[5]),
            "invalid_price_rows": int(raw[6]),
            "duplicate_date_ticker_pairs": int(duplicate_pairs),
        },
        "returns": {
            "rows": len(returns_df),
            "tickers": total_tickers,
            "missing_cells": missing_cells,
            "missing_cell_pct": missing_cells / total_cells * 100
            if total_cells else 0.0,
            "all_null_rows": int(returns_df.isna().all(axis=1).sum()),
            "min_tickers_observed_on_date": int(
                returns_df.notna().sum(axis=1).min()
            ) if len(returns_df) else 0,
            "max_tickers_observed_on_date": int(
                returns_df.notna().sum(axis=1).max()
            ) if len(returns_df) else 0,
            "zero_returns": int((returns_df == 0).sum().sum()),
            "extreme_positive_returns": int((finite_returns > 0.405).sum()),
            "extreme_negative_returns": int((finite_returns < -0.693).sum()),
        },
        "sec_metadata": {
            "matched_company_tickers": matched_tickers,
            "sic_tickers": sic_tickers,
            "company_coverage_pct": matched_tickers / total_tickers * 100
            if total_tickers else 0.0,
            "sic_coverage_pct": sic_tickers / total_tickers * 100
            if total_tickers else 0.0,
        },
        "limitations": list(DATASET_LIMITATIONS),
    }

    if emit:
        raw_report = report["raw"]
        returns_report = report["returns"]
        sec_report = report["sec_metadata"]
        print("=" * 60)
        print("         FAULTLINES DATA PIPELINE - DATA QUALITY SUITE")
        print("=" * 60)
        print(
            "\n1. Primary Key Integrity: "
            f"{raw_report['duplicate_date_ticker_pairs']} duplicate pairs "
            f"[{_status(raw_report['duplicate_date_ticker_pairs'])}]"
        )
        print(
            "2. Price Integrity: "
            f"{raw_report['invalid_price_rows']} invalid rows "
            f"[{_status(raw_report['invalid_price_rows'])}]"
        )
        print(
            "3. Missing OHLC Values: "
            f"{raw_report['null_ohlc']} rows; return-matrix missing cells: "
            f"{returns_report['missing_cells']:,} "
            f"({returns_report['missing_cell_pct']:.2f}%)"
        )
        print(
            "4. SEC Company Coverage: "
            f"{sec_report['matched_company_tickers']:,}/{total_tickers:,} "
            f"({sec_report['company_coverage_pct']:.2f}%)"
        )
        print(
            "5. SEC SIC Coverage: "
            f"{sec_report['sic_tickers']:,}/{total_tickers:,} "
            f"({sec_report['sic_coverage_pct']:.2f}%)"
        )
        print(
            "6. Return Coverage by Date: "
            f"{returns_report['min_tickers_observed_on_date']:,} to "
            f"{returns_report['max_tickers_observed_on_date']:,} tickers"
        )
        print(
            "7. Extreme Log Returns: "
            f"+50%: {returns_report['extreme_positive_returns']:,}; "
            f"-50%: {returns_report['extreme_negative_returns']:,}"
        )
        print("\nDataset limitations:")
        for limitation in DATASET_LIMITATIONS:
            print(f"- {limitation}")
        print("\n" + "=" * 60)

    return report


if __name__ == "__main__":
    run_data_quality_checks()
