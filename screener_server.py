"""
screener_server.py

MCP server that exposes the screener as ONE tool. The model calls it once.
All fetching and math for ~500 stocks happens here in python, and the model
only receives the finished, ranked shortlist
"""

import logging
from typing import Annotated, Literal
from pydantic import Field
from mcp.server.mcpserver import MCPServer
from screener import LOOKBACK_DAYS, MAX_PER_SECTOR, MIN_VOLATILITY, screen

log = logging.getLogger(__name__)
mcp = MCPServer("screener")

# The exact GICS sector names used in the S&P 500 list. Using Literal turns
# this into an enum in the tool schema, so the model can only pick valid
# names. Otherwise "Tech" would silently exclude nothing
Sector = Literal[
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

@mcp.tool()
def screen_stocks(
    budget: Annotated[
        float,
        Field(gt=0, le=1_000_000, description="Total dollars to split across the picks"),
    ], 
    top_n: Annotated[
        int, 
        Field(ge=1, le=10, description="How many stocks to recommend"),
    ] = 5,
    max_volatility: Annotated[
        float | None,
        Field(gt=MIN_VOLATILITY, le=2.0, description=("Risk ceiling: annualized volatility as a decimal (0.30 = 30%)" "Guide: low risk 0.30, moderate 0.45. Omit for no ceiling"),),
    ] = None,
    exclude_sectors: Annotated[
        list[Sector] | None,
        Field(description="Sectors to leave out entirely"),
    ] = None,
) -> dict:
    """
    Screen all S&P 500 stocks and return a ranked, diversified shrtlist
    with a dollar split that sums to 'budget'
    
    Ranking: score = 3-month momentum / anualized volatility (gain per 
    unit of risk). Atmost 2 picks per sector. Stocks with abnormally low 
    voltility (usually pinned by a pending buyout) are always excluded.
    Use the numbers returned here as-is; do not recompute or round them.
    """
    try:
        result = screen(
            budget=budget,
            top_n=top_n,
            exclude_sectors=list(exclude_sectors) if exclude_sectors else None,
            max_volatility=max_volatility,
        )

        # Convert each row to plain python types. Pansdas gives numpy floats
        # which JSON can't always serialize, so floar()/str() are explicit
        picks = [
            {
                "symbol": str(symbol),
                "name": str(row["Security"]),
                "sector": str(row["GICS Sector"]),
                "last_close": round(float(row["last_close"]), 2),
                "momentum_pct": round(float(row["momentum"]) * 100, 1),
                "volatility_pct": round(float(row["volatility"]) * 100, 1),
                "score": round(float(row["score"]), 2),
                "dollars": round(float(row["dollars"]), 2)
            }
            for symbol, row in result.picks.iterrows()
        ]
    
        response = {
            "ok": True,
            "as_of": result.as_of,
            "picks": picks,
            "total_allocated": round(sum(p["dollars"] for p in picks), 2),
            "screened": result.scored,
            "universe_size": result.universe_size,
            "excluded": {
                "too_volatile_count": result.too_volatile,
                "abnormally_calm": result.too_calm,
                "missing_data": result.missing,
                "short_history": result.short_history,
            },
            "settings":{
                "lookback_trading_days": LOOKBACK_DAYS,
                "max_volatility_pct": round(max_volatility*100, 1) if max_volatility else None,
                "max_per_sector": MAX_PER_SECTOR,
                "excluded_sectors": list(exclude_sectors) if exclude_sectors else [],
            },
            "data_notes": (
                "Prices are split/dividend-adjusted daily closes from the IEX feed "
                "(one exchange, not the consolidated tape), completed trading days only. "
                "momentum_pct = % change over lookback window. "
                "volatility_pct = annualized standard deviation of daily returns."
            ),
        }
        if not picks:
            response["message"] = "No stocks met the criteria. Try a higher max_volatility or fewer exculded sectors."
        return response
    except Exception as e:
        log.exception("Screen failed")
        return {"ok": False, "error": f"screen failed: {e}"}

if __name__ == "__main__":
    logging.basicConfig(levels=logging.INFO)
    mcp.run()