import httpx

HEADERS = {"User-Agent": "Mark Mark03@gmail.com"}

# Step 1: ticker -> CIK lookup
resp = httpx.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
resp.raise_for_status()
tickers = resp.json()

aapl_cik = None
for entry in tickers.values():
    if entry["ticker"] == "AAPL":
        aapl_cik = str(entry["cik_str"]).zfill(10)
        break

print("AAPL CIK:", aapl_cik)

# Step 2: pull company facts using that CIK
facts_resp = httpx.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{aapl_cik}.json", headers=HEADERS)
facts_resp.raise_for_status()
facts = facts_resp.json()

#sanity check: recent revenue filings
revenues = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
print("Most recent 3 revenue filings:")
for entry in revenues[-3:]:
    print(entry["end"], entry["val"], entry["form"])