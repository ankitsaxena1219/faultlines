"""Build the SEC metadata table for the complete raw ticker universe."""

import argparse
from pathlib import Path
import os
import time

import duckdb
import pandas as pd
import requests


DB_PATH = Path(__file__).resolve().with_name("faultlines.duckdb")
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
METADATA_COLUMNS = [
    "Ticker", "Company_Name", "cik_str", "sic", "sicDescription"
]


def _normalise_ticker(value: object) -> str:
    return str(value).upper().replace(".US", "")


def _raw_tickers(db_path: Path) -> set[str]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        values = con.execute(
            "SELECT DISTINCT Ticker FROM raw_prices"
        ).fetchall()
    return {_normalise_ticker(value[0]) for value in values}


def build_sec_metadata(
    tickers: set[str] | None = None,
    db_path: Path = DB_PATH,
    request_pause_seconds: float = 0.12,
) -> dict[str, object]:
    """Download SEC metadata and replace ``sec_metadata`` in DuckDB.

    By default, every distinct ticker in ``raw_prices`` is considered. SEC's
    current ticker file cannot resolve every historical symbol; those symbols
    remain unmatched and are included in the returned coverage summary.
    """
    sec_email = os.environ.get("SEC_EMAIL")
    if not sec_email:
        raise ValueError("Set SEC_EMAIL before requesting SEC data.")

    requested_tickers = {
        _normalise_ticker(ticker) for ticker in (tickers or _raw_tickers(db_path))
    }
    headers = {"User-Agent": f"FaultLines Project contact: {sec_email}"}

    response = requests.get(SEC_TICKER_URL, headers=headers, timeout=30)
    response.raise_for_status()
    sec = pd.DataFrame.from_dict(response.json(), orient="index")
    sec = sec.rename(columns={"ticker": "Ticker", "title": "Company_Name"})
    sec["Ticker"] = sec["Ticker"].map(_normalise_ticker)
    sec = sec[sec["Ticker"].isin(requested_tickers)]
    sec = sec.drop_duplicates("Ticker")

    records = []
    failures = {}
    for _, row in sec.iterrows():
        ticker = row["Ticker"]
        cik = str(int(row["cik_str"])).zfill(10)
        try:
            result = requests.get(
                SEC_SUBMISSIONS_URL.format(cik=cik),
                headers=headers,
                timeout=30,
            )
            result.raise_for_status()
            company = result.json()
            records.append({
                "Ticker": ticker,
                "Company_Name": row["Company_Name"],
                "cik_str": int(row["cik_str"]),
                "sic": company.get("sic"),
                "sicDescription": company.get("sicDescription"),
            })
        except requests.RequestException as exc:
            failures[ticker] = str(exc)
        time.sleep(request_pause_seconds)

    metadata = pd.DataFrame(records, columns=METADATA_COLUMNS)
    with duckdb.connect(str(db_path)) as con:
        con.execute("DROP TABLE IF EXISTS sec_metadata")
        con.register("metadata_df", metadata)
        con.execute("CREATE TABLE sec_metadata AS SELECT * FROM metadata_df")
        con.unregister("metadata_df")

    matched = set(metadata["Ticker"])
    summary = {
        "requested_tickers": len(requested_tickers),
        "sec_matches": len(sec),
        "metadata_records": len(metadata),
        "records_with_sic": int(metadata["sic"].notna().sum()),
        "unmatched_tickers": sorted(requested_tickers - matched),
        "request_failures": failures,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset")
    args = parser.parse_args()
    build_sec_metadata(set(args.tickers) if args.tickers else None)
