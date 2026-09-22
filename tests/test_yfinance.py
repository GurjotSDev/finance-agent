import yfinance as yf

ticker = yf.Ticker("AAPL")

hist = ticker.history(period="5d")
print(hist[["Open", "High", "Low", "Close", "Volume"]])

info = ticker.info
print(info.get("longName"), "-", info.get("sector"))


print(ticker.info.get("shortName"))
print(ticker.info.get("industry"))