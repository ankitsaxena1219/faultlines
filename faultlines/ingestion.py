import os
import duckdb
import kagglehub

print("1. Locating Kaggle dataset path...")
kaggle_path = kagglehub.dataset_download(
    "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
)
glob_path = os.path.join(kaggle_path, "**", "*.txt").replace("\\", "/")

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "faultlines.duckdb")
)
con = duckdb.connect(DB_PATH)

print("\n2. Rebuilding raw_prices with quality filters...")
con.execute("DROP TABLE IF EXISTS raw_prices")

con.execute(f"""
    CREATE TABLE raw_prices AS 
    WITH source_data AS (
        SELECT *, 
               regexp_extract(filename, '([^/\\\\]+)\\.txt$', 1) AS Ticker,
               CASE 
                   WHEN filename LIKE '%ETFs%' THEN 'ETF'
                   WHEN filename LIKE '%Stocks%' THEN 'Stock'
                   ELSE 'Other'
               END AS Asset_Type
        FROM read_csv_auto('{glob_path}', filename=True, header=True)
    )
    SELECT DISTINCT Date, Ticker, Asset_Type, Open, High, Low, Close, Volume
    FROM source_data
    WHERE Close > 0 
      AND Open > 0 
      AND High >= Low
""")

row_count = con.execute("SELECT count(*) FROM raw_prices").fetchone()[0]
ticker_count = con.execute("SELECT count(DISTINCT Ticker) FROM raw_prices").fetchone()[0]

print(f"\nClean Ingestion Complete!")
print(f"Total Valid Rows: {row_count:,} | Unique Tickers: {ticker_count:,}")
con.close()