"""
sp500.py

Provides the current S&P 500 constituent list, cashed to disk

- Scrapes Wikipedia at most once every MAX_AGE_DAYS
- Saves each scrape as data/sp500_YYY-MM-DD.csv (a dated record)
- If a scrape fails, falls back to the newest cached file
- Validates every list before returning it, whether fresh or cached
"""

import logging
import os
from datetime import date
from io import StringIO
from pathlib import Path
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_DIR = Path(__file__).resolve().parent / "data"
MAX_AGE_DAYS = 7  # Maximum age of cached data before re-scraping\
KEEP_COLUMNS = ["Symbol", "Security", "GICS Sector"]

# Logging instead of print(): this module will run inside an MCP server
# where stdout is the protocol channel. Logging writes to stderr by default
log = logging.getLogger(__name__)

def _user_agent() -> str:
    contact = os.getenv("CONTACT_EMAIL", "no-contact-provided")
    return f"Mozilla/5.0 (research prject; {contact})"

def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if te list looks wrong. Return df so calls can be chained"""
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    tickers = set(df["Symbol"])
    if not (490 <= len(tickers) <= 510):
        raise ValueError(f"Unexpected numer of tickers: {len(tickers)}")
    for must_have in ("AAPL", "MSFT"):
        if must_have not in tickers:
            raise ValueError(f"{must_have} not found in S&P 500 list")
    return df

def _fetch_from_wikipedia() -> pd.DataFrame:
    resp = httpx.get(WIKI_URL, headers={"User-Agent": _user_agent()}, timeout=15)
    resp.raise_for_status()

    # Match="symbols" picks the table that contains text, instead of
    # trusting it to always be the first table on the page
    table = pd.read_html(StringIO(resp.text), match="Symbol")[0]
    table = table[KEEP_COLUMNS].copy()
    table["Symbol"] = table["Symbol"].str.strip()
    return _validate(table)

def _latest_cache() -> tuple[Path, date] | None:
    """Newest cached file and its date, or None if there is no cache"""
    files = sorted(CACHE_DIR.glob("sp500_*.csv"))
    if not files:
        return None
    newest = files[-1]
    file_date = date.fromisoformat(newest.stem.removeprefix("sp500_"))
    return newest, file_date

def _read_cache(path: Path) -> pd.DataFrame:
    # keep_default_na = false: otherwise pandas turns string like "NA"
    # into missing values, which would silently corrupt a ticker.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return _validate(df)

def get_sp500(force_refresh: bool = False) -> pd.DataFrame:
    """Return the S&P 500 list as a Dataframe (Symbol, Security, GICS Sector)"""
    cached = _latest_cache()

    if cached and not force_refresh:
        path, file_date = cached
        age = (date.today() - file_date).days
        if age <= MAX_AGE_DAYS:
            log.info("Using cached S&P 500 list from %s (%d days old)", file_date, age)
            return _read_cache(path)

    try:
        df = _fetch_from_wikipedia()
    except Exception as e:
        if cached:
            log.warning("Scrape failed (%s). Fallinf back to %s", e, cached[0].name)
            return _read_cache(cached[0])
        raise

    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"sp500_{date.today().isoformat()}.csv"
    df.to_csv(out, index=False)
    log.info("Scraped %d tickers, saved to %s", len(df), out.name)
    return df

def get_tickers(force_refresh: bool = False) -> list[str]:
    return get_sp500(force_refresh)["Symbol"].tolist()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_sp500()
    print("Total tickers:", len(df))
    print("Sectors:", df["GICS Sector"].value_counts().to_dict())
    print(df.head())