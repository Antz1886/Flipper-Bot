"""
modules/data_feed.py — Deriv OHLC Data Fetcher
Fetches candlestick history via the Deriv ticks_history API and
returns a pandas DataFrame compatible with smartmoneyconcepts.
"""

import logging
import pandas as pd

from modules.connector import get_api
from config import N_BARS

log = logging.getLogger(__name__)


async def fetch_ohlc(
    symbol: str,
    granularity: int,
    n_bars: int = N_BARS,
) -> pd.DataFrame | None:
    """
    Fetch the last `n_bars` OHLC candles via Deriv ticks_history.

    Parameters
    ----------
    symbol      : Deriv symbol code, e.g. "R_75"
    granularity : Candle size in seconds (300 = M5, 900 = M15)
    n_bars      : Number of candles to fetch

    Returns
    -------
    DataFrame with columns: time, open, high, low, close, volume
    Compatible with the smartmoneyconcepts library.
    None on failure.
    """
    api = get_api()
    try:
        resp = await api.send({
            "ticks_history": symbol,
            "granularity":   granularity,
            "count":         n_bars,
            "end":           "latest",
            "style":         "candles",
        })
    except Exception as e:
        log.error(f"ticks_history request failed for {symbol}: {e}")
        return None

    candles = resp.get("candles")
    if not candles:
        log.error(f"No candles in response for {symbol} (granularity={granularity}s)")
        return None

    df = pd.DataFrame(candles)

    # Normalize column names
    df = df.rename(columns={"epoch": "time"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Ensure numeric types
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deriv candle history doesn't include tick volume — use row index as proxy
    # (smc.ob() uses volume to rank OB strength; non-zero values avoid div-by-zero)
    if "volume" not in df.columns:
        df["volume"] = range(1, len(df) + 1)

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna().reset_index(drop=True)

    log.debug(f"Fetched {len(df)} candles | {symbol} {granularity}s")
    return df
