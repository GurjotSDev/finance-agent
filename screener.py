"""
screener.py

Rank S&P 500 stocks by momentum per unit of risk, then split a budget 
across te top picks. Pure Python: no AI, no MCP. The MCP server will
import screen()

    momentum = % price change over the lookback window (3 months)
    volatility = annualized standard deviation of daily returns
    score = momentum / volatility (return per unit of risk)
"""

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from collections import Counter
from sp500 import get_sp500

load_dotenv()
log = logging.getLogger(__name__)

LOOKBACK_DAYS = 63 # About 3 months of trading days
TRADING_DAYS_PER_YEAR = 252
BATCH_SIZE = 100
MAX_PER_SECTOR = 2

_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
)

@dataclass
class ScreenResult:
    """Everything the screener found, including what went wrong"""
    picks: pd.DataFrame # the ranked shortlist with dollar amounts
    as_of: str # date of the most recent price used
    universe_size: int # how many tickers we tried to screen
    scored: int # how many had enough data to score
    missing: list[str] = field(default_factory=list) # no data returned
    short_history: list[str] = field(default_factory=list) # too few bars
    too_volatile: int = 0 # how many were removed by max_volatility

# Fetch closing prices for every symbol, in batches
def fetch_closes(symbols: list[str], lookback_days: int = LOOKBACK_DAYS) -> pd.Series:
    """Daily adjusted closes, indexed by (Symbol, timesamp), sorted"""
    # Trading days -> calendar days: x1.5 covers weekends, +10 covers holidays
    start = datetime.now(timezone.utc) - timedelta(days=int(lookback_days *1.5) + 10)
    
    frames = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        request = StockBarsRequest(
            symbol_or_symbols=batch, 
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
            adjustment=Adjustment.ALL,
        )
        bars = _client.get_stock_bars(request)
        if not bars.df.empty:
            frames.append(bars.df["close"])
        log.info("Fetched symbols %d-%d of %d", i+1, i+len(batch), len(symbols))

    if not frames:
        raise RuntimeError("Alpaca returned no price data al all")

    closes = pd.concat(frames).sort_index()

    # Keep only completed trading days, Today's bar may still be forming
    # if the maret is open, so everything is scored on finishing days.
    today_start = pd.Timestamp.now(tz="America/New_York").normalize()
    return closes[closes.index.get_level_values("timestamp") < today_start]

# Compute momentum, volatility, score for each symbol
def score_stocks(closes: pd.Series, lookback_days: int = LOOKBACK_DAYS):
    """Return (metircs DataFrame sorted best first, list of short-history symbols)"""
    # N daily returns need N+1 prices
    needed = lookback_days + 1
    window = closes.groupby(level="symbol").tail(needed)

    # Drop symbols without a full window instead of scoring them on less data
    counts = window.groupby(level="symbol").size()
    complete = counts[counts == needed].index
    short = sorted(set(counts.index) - set(complete))
    window = window[window.index.get_level_values("symbol").isin(complete)]

    by_symbol = window.groupby(level="symbol")
    first = by_symbol.first()
    last = by_symbol.last()

    # Daily return = today / yesterday -1, computed per symbol
    daily_returns = window / by_symbol.shift(1) -1
    daily_vol = daily_returns.groupby(level="symbol").std()

    metrics = pd.DataFrame(
        {
            "last_close": last,
            "momentum": last / first - 1,
            "volatility": daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR), 
        }
    )
    metrics["volatility"] = metrics["volatility"].replace(0, float("nan"))
    metrics["score"] = metrics["momentum"] / metrics["volatility"]
    metrics = metrics.dropna().sort_values("score", ascending=False)
    return metrics, short

def pick_diversified(ranked: pd.DataFrame, top_n: int, max_per_sector: int) -> pd.DataFrame:
    """Walk down the ranked list, skipping stocks whose sector is already full"""
    chosen = []
    per_sector = Counter()
    for symbol, row in ranked.iterrows():
        sector = row["GICS Sector"]
        if per_sector[sector] >= max_per_sector:
            continue
        chosen.append(symbol)
        per_sector[sector] += 1
        if len(chosen) == top_n:
            break
    return ranked.loc[chosen]



# Split the budget across the picks
def allocate(picks: pd.DataFrame, budget:float) -> pd.Series:
    """Dollars per pick, weighted by score, summing to exactly 'budget' """
    weights = picks["score"] / picks["score"].sum()
    dollars = (weights * budget).round(2)
    # Rounding each amount to cents can eave the total a few cents off,
    # Give the difference o the rop pick so that total is exact
    dollars.iloc[0] = round(dollars.iloc[0] + (budget - dollars.sum()), 2)
    return dollars

# The dunction called by other code
def screen(
        budget: float,
        top_n: int=5,
        exclude_sectors: list[str] | None = None,
        max_volatility: float | None = None,
        lookback_days: int = LOOKBACK_DAYS,
) -> ScreenResult:
    universe = get_sp500()
    if exclude_sectors:
        excluded = {s.strip().lower() for s in exclude_sectors}
        universe = universe[~universe["GICS Sector"].str.lower().isin(excluded)]

    symbols = universe["Symbol"].tolist()
    closes = fetch_closes(symbols, lookback_days)

    returned = set(closes.index.get_level_values("symbol"))
    missing = sorted(set(symbols) - returned)
    if missing:
        log.warning("No data for %d symbols: %s", len(missing), missing)

    metrics, short = score_stocks(closes, lookback_days)
    if short:
        log.warning("Not enough history for %d symbols: %s", len(short), short)
    scored = len(metrics)

    too_volatile = 0
    if max_volatility is not None:
        keep = metrics["volatility"] <= max_volatility
        too_volatile = int((~keep).sum())
        metrics = metrics[keep]

    # Only recommend stocks that actually went up.
    ranked = metrics[metrics["score"] > 0].join(
        universe.set_index("Symbol")[["Security", "GICS Sector"]]
    )
    picks = pick_diversified(ranked, top_n, MAX_PER_SECTOR)
    if not picks.empty:
        picks["dollars"] = allocate(picks, budget)

    as_of = closes.index.get_level_values("timestamp").max().date().isoformat()
    return ScreenResult(
        picks=picks,
        as_of=as_of,
        universe_size=len(symbols),
        scored=scored,
        missing=missing,
        short_history=short,
        too_volatile=too_volatile,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = screen(budget=1000, top_n=5, max_volatility=0.30)

    print()
    print(f"Prices as of: {result.as_of}")
    print(f"Screened {result.scored} of {result.universe_size} stocks")
    print(f"Removed as too volatile: {result.too_volatile}")
    print(f"Missing data: {result.missing or 'none'}")
    print(f"Short history: {result.short_history or 'none'}")
    print()

    show = result.picks.copy()
    show["momentum"] = (show["momentum"] * 100).map("{:+.1f}%".format)
    show["volatility"] = (show["volatility"] * 100).map("{:.1f}%".format)
    show["score"] = show["score"].map("{:.2f}".format)
    show["dollars"] = show["dollars"].map("${:,.2f}".format)
    print(show[["Security", "GICS Sector", "last_close", "momentum", "volatility", "score", "dollars"]].to_string())
    print(f"\nTotal allocated: ${result.picks['dollars'].sum():,.2f}")