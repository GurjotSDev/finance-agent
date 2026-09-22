from mcp.server.mcpserver import MCPServer
import yfinance as yf

# This name identifies the server when teh agent connects to it
mcp = MCPServer("market-data")

@mcp.tool()
def get_price_history(ticker: str, period: str = "1mo") -> str:
    """
    Get historical price data for a ticker
    
    Args:
        ticker: Stock symbol, eg AAPL, MSFT, NVDA
        period: Time range - one of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return f"No data found for ticker '{ticker}'."

    # Convert to a compact string the LLM can easily read
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].round(2)

    short_periods = {"1d", "5d", "1mo"}
    if period not in short_periods and len(hist) > 40:
        monthly = hist.resample("ME").last().round(2)
        return monthly.to_csv()
    return hist.to_csv()

@mcp.tool()
def get_company_info(ticker: str) -> str:
    """
    Get basic info: name, sector, industry, market cap.
    
    Args:
        ticker: Stock symbol, e.g. AAPL, MSFT, NVDA
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or "shortName" not in info:
        return f"No company found for ticker '{ticker}'"

    return (
        f"Name: {info.get('shortName')}\n"
        f"Sector: {info.get('sector')}\n"
        f"Industry: {info.get('Industry')}\n"
        f"Market Cap: {info.get('marketCap')}\n"
        f"Current Price: {info.get('currentPrice')}"
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")