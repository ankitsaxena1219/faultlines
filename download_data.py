import json
import os
import kagglehub
import pandas as pd
import requests

print("1. Downloading Kaggle Stock Dataset...")
kaggle_path = kagglehub.dataset_download(
    "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
)
print(f"Dataset cached at: {kaggle_path}")

print("\n2. Fetching SEC Ticker & SIC Mapping...")
headers = {"User-Agent": "Ankit Saxena asaxena319@gatech.edu"}
url = "https://www.sec.gov/files/company_tickers.json"
res = requests.get(url, headers=headers)

df_sec = pd.DataFrame.from_dict(res.json(), orient="index")
df_sec.rename(
    columns={"ticker": "Ticker", "title": "Company_Name"}, inplace=True
)

os.makedirs("./data", exist_ok=True)
df_sec.to_csv("./data/sec_sic_mapping.csv", index=False)
print("SEC mapping saved to ./data/sec_sic_mapping.csv")