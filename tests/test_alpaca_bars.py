"""
tests/test_alpaca_bars.py

Exploration script (no AI, no MCP): prove we can fetch daily price bars
for several tickers in one request, and see what the data looks like
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests  import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

load_dotenv()

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"], 
)

# BRK.B is in here on purpose: it tests the dot ticker format
symbols = ["AAPL", "MSFT", "NVDA", "JNJ", "BRK.B"]

# -100 calendar days back gives roughly 65-70 trading days
start = datetime.now(timezone.utc) - timedelta(days=100)

request = StockBarsRequest(
    symbol_or_symbols=symbols, 
    timeframe=TimeFrame.Day,
    start=start,
    feed=DataFeed.IEX,
    adjustment=Adjustment.ALL,
)

bars = client.get_stock_bars(request)
df = bars.df

print("Shape (rows, columns):", df.shape)
print("Index levels:", df.index.names)
print("Columns:", df.columns.tolist())
print()
print("Bars per symbol:")
print(df.groupby(level="symbol").size())
print()
print("First rows:")
print(df.head())
print()
print("Last close per symbol")
print(df["close"].groupby(level="symbol").last())

# What happens with a ticker that doesn't exist?
# The screener will send -500 symbols. We need to know whether one bad
# symbol breaks the whole request or is it just left out
print()
print("Testing with a fake symbol mixed in...")
try:
    bad = client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=["AAPL", "LMAO"],
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
        )
    )
    print("No error. Symbols returned:", sorted(bad.df.index.get_level_values("symbol").unique()))
except Exception as e:
    print("Request failed:", type(e).__name__, e)