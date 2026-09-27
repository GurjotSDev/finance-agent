import httpx
import pandas as pd
from io import StringIO

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0 (research project; mark03@gmail.com)"}

resp = httpx.get(url, headers=headers)
resp.raise_for_status()  # Raise an error if the request failed

sp500 = pd.read_html(StringIO(resp.text))[0]

tickers = sp500["Symbol"].tolist()

print("Columns:", sp500.columns.tolist())
print("Total tickers:" , len(tickers))
print("First 5:", tickers[:5])

# Sanity check: fail loudly if something is wrong
# rather than silently continueing

assert 490 <= len(tickers) <= 510, f"Unexpected number of tickers: {len(tickers)}"
assert "AAPL" in tickers, "AAPL not found in S&P 500 list"
assert "MSFT" in tickers, "MSFT not found in S&P 500 list"

print("Sanity checks passed. S&P 500 list looks good.")