import time
import os
import requests
import pandas as pd
import duckdb

DB_PATH = os.path.abspath("faultlines.duckdb")

SEC_EMAIL = os.environ.get("SEC_EMAIL")

if not SEC_EMAIL:
    raise ValueError(
        "Please set the SEC_EMAIL environment variable before running this script."
    )

HEADERS = {
    "User-Agent": f"FaultLines Project contact: {SEC_EMAIL}"
}

# Load the 50 selected project tickers
selected = pd.read_csv("results/selected_tickers.csv")
selected["Ticker"] = selected["Ticker"].str.upper()
selected_tickers = set(selected["Ticker"])

print(f"Project tickers: {len(selected_tickers)}")

# Fetch SEC ticker-to-CIK mapping
print("\n1. Fetching SEC ticker mapping...")

url = "https://www.sec.gov/files/company_tickers.json"
response = requests.get(url, headers=HEADERS, timeout=30)
response.raise_for_status()

sec = pd.DataFrame.from_dict(response.json(), orient="index")
sec = sec.rename(columns={
    "ticker": "Ticker",
    "title": "Company_Name"
})
sec["Ticker"] = sec["Ticker"].str.upper()
# Exclude known historical ticker mismatches
historical_mismatches = {"ABX", "APC", "BBT"}
sec = sec[~sec["Ticker"].isin(historical_mismatches)].copy()

# Keep only the project tickers
sec = sec[sec["Ticker"].isin(selected_tickers)].copy()

print(f"Matched to current SEC ticker list: {len(sec)}")
print(f"Not matched: {len(selected_tickers) - len(sec)}")

# Retrieve SIC information for matched companies
print("\n2. Fetching SIC information...")

records = []

for _, row in sec.iterrows():
    cik = str(row["cik_str"]).zfill(10)
    submissions_url = (
        f"https://data.sec.gov/submissions/CIK{cik}.json"
    )

    try:
        r = requests.get(
            submissions_url,
            headers=HEADERS,
            timeout=30
        )
        r.raise_for_status()
        company = r.json()

        records.append({
            "Ticker": row["Ticker"],
            "Company_Name": row["Company_Name"],
            "cik_str": int(row["cik_str"]),
            "sic": company.get("sic"),
            "sicDescription": company.get("sicDescription")
        })

        print(f"Retrieved: {row['Ticker']}")

    except Exception as e:
        print(f"Failed: {row['Ticker']} - {e}")

    time.sleep(0.12)

metadata = pd.DataFrame(
    records,
    columns=[
        "Ticker",
        "Company_Name",
        "cik_str",
        "sic",
        "sicDescription"
    ]
)

# Create SEC metadata table in DuckDB
print("\n3. Creating sec_metadata table...")

con = duckdb.connect(DB_PATH)
con.execute("DROP TABLE IF EXISTS sec_metadata")
con.register("metadata_df", metadata)

con.execute("""
    CREATE TABLE sec_metadata AS
    SELECT * FROM metadata_df
""")

con.unregister("metadata_df")
con.close()

# Print summary
matched = set(metadata["Ticker"])
missing = sorted(selected_tickers - matched)

print("\n========== SUMMARY ==========")
print(f"Project tickers: {len(selected_tickers)}")
print(f"Metadata records: {len(metadata)}")
print(f"Records with SIC: {metadata['sic'].notna().sum()}")
print(f"Records without SIC: {metadata['sic'].isna().sum()}")
print(f"Unmatched project tickers: {missing}")
print("=============================")
