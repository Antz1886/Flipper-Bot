"""
modules/smc_engine.py — Core Institutional Edge (Liquidity Sweep + OB Mitigation)
Pure structural price action engine based on Conti's institutional framework.
Avoids indicator overfitting. Emits signals strictly upon:
  1. Swing Liquidity Sweep (false breakout outside swing high/low)
  2. Market Structure Shift (displacement closing back within / breaking structure)
  3. Order Block Mitigation (price tapping into the origin impulse candle)
"""

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)


@dataclass
class SMCSignal:
    symbol: str
    direction: str         # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    sweep_level: float
    ob_top: float
    ob_bottom: float
    risk_price_distance: float
    reward_price_distance: float
    rr_ratio: float


def identify_swings(df: pd.DataFrame, swing_len: int = config.SWING_LENGTH) -> pd.DataFrame:
    """
    Identify confirmed swing highs and swing lows without lookahead bias.
    A swing high requires `swing_len` lower highs to the left and right.
    """
    df = df.copy()
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(swing_len, n - swing_len):
        current_h = highs[i]
        current_l = lows[i]

        # Check Swing High
        is_sh = True
        for offset in range(1, swing_len + 1):
            if highs[i - offset] >= current_h or highs[i + offset] > current_h:
                is_sh = False
                break
        if is_sh:
            swing_highs[i] = current_h

        # Check Swing Low
        is_sl = True
        for offset in range(1, swing_len + 1):
            if lows[i - offset] <= current_l or lows[i + offset] < current_l:
                is_sl = False
                break
        if is_sl:
            swing_lows[i] = current_l

    df["swing_high"] = swing_highs
    df["swing_low"] = swing_lows
    return df


def detect_institutional_setup(symbol: str, df: pd.DataFrame) -> Optional[SMCSignal]:
    """
    Evaluate structural price action on OHLC data:
    1. Find most recent key swing levels.
    2. Check for recent Liquidity Sweep within lookback.
    3. Identify the origin Order Block (OB) responsible for the reversal displacement.
    4. Validate if current price is actively mitigating the Order Block.
    """
    if df is None or len(df) < (config.SWING_LENGTH * 3 + 10):
        return None

    df_swings = identify_swings(df, config.SWING_LENGTH)
    current_price = float(df_swings["close"].iloc[-1])
    current_high = float(df_swings["high"].iloc[-1])
    current_low = float(df_swings["low"].iloc[-1])

    # Recent history window (lookback 30 candles)
    window = df_swings.iloc[-30:].copy()

    # Find distinct historical confirmed swing levels (excluding last swing_len bars)
    historical_bars = df_swings.iloc[:-config.SWING_LENGTH]
    recent_sh_series = historical_bars["swing_high"].dropna()
    recent_sl_series = historical_bars["swing_low"].dropna()

    if recent_sh_series.empty or recent_sl_series.empty:
        return None

    recent_sh = float(recent_sh_series.iloc[-1])
    recent_sl = float(recent_sl_series.iloc[-1])

    # ─────────────────────────────────────────────────────────────────────────────
    # BULLISH SETUP: Sell-side Liquidity Sweep + Bullish OB Mitigation
    # ─────────────────────────────────────────────────────────────────────────────
    # Condition 1: A recent candle dipped below recent_sl (sweep) but closed above it (rejection)
    # Look back over the past 15 candles for the sweep
    sweep_idx = None
    sweep_extreme_low = None

    for idx in range(len(window) - 15, len(window) - 1):
        bar = window.iloc[idx]
        if bar["low"] < recent_sl and bar["close"] > recent_sl:
            sweep_idx = idx
            sweep_extreme_low = bar["low"]
            break

    if sweep_idx is not None and sweep_extreme_low is not None:
        # Condition 2: Market Structure Shift / Displacement up after sweep
        # Find the origin bullish OB (the down candle prior to the upward displacement)
        sweep_bar = window.iloc[sweep_idx]
        ob_top = max(float(sweep_bar["open"]), float(sweep_bar["close"]))
        ob_bottom = sweep_extreme_low

        # Condition 3: Mitigation check
        # Price must have pulled back into the Order Block zone [ob_bottom, ob_top]
        if ob_bottom <= current_price <= (ob_top + (ob_top - ob_bottom) * 0.2):
            sl = ob_bottom
            risk_dist = abs(current_price - sl)
            if risk_dist > 0:
                tp = current_price + (risk_dist * config.RISK_REWARD_RATIO)
                log.info(
                    f"🎯 [SMC BULLISH EDGE] {symbol} | Sweep Low: ${recent_sl:.5f} | "
                    f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                    f"SL: ${sl:.5f} | TP: ${tp:.5f}"
                )
                return SMCSignal(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    sweep_level=recent_sl,
                    ob_top=ob_top,
                    ob_bottom=ob_bottom,
                    risk_price_distance=risk_dist,
                    reward_price_distance=abs(tp - current_price),
                    rr_ratio=config.RISK_REWARD_RATIO
                )

    # ─────────────────────────────────────────────────────────────────────────────
    # BEARISH SETUP: Buy-side Liquidity Sweep + Bearish OB Mitigation
    # ─────────────────────────────────────────────────────────────────────────────
    # Condition 1: A recent candle spiked above recent_sh (sweep) but closed below it (rejection)
    sweep_idx_bear = None
    sweep_extreme_high = None

    for idx in range(len(window) - 15, len(window) - 1):
        bar = window.iloc[idx]
        if bar["high"] > recent_sh and bar["close"] < recent_sh:
            sweep_idx_bear = idx
            sweep_extreme_high = bar["high"]
            break

    if sweep_idx_bear is not None and sweep_extreme_high is not None:
        # Condition 2: Order Block (the up candle prior to displacement down)
        sweep_bar = window.iloc[sweep_idx_bear]
        ob_top = sweep_extreme_high
        ob_bottom = min(float(sweep_bar["open"]), float(sweep_bar["close"]))

        # Condition 3: Mitigation check
        # Price is currently pulling back up into [ob_bottom, ob_top]
        if (ob_bottom - (ob_top - ob_bottom) * 0.2) <= current_price <= ob_top:
            sl = ob_top
            risk_dist = abs(sl - current_price)
            if risk_dist > 0:
                tp = current_price - (risk_dist * config.RISK_REWARD_RATIO)
                log.info(
                    f"🎯 [SMC BEARISH EDGE] {symbol} | Sweep High: ${recent_sh:.5f} | "
                    f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                    f"SL: ${sl:.5f} | TP: ${tp:.5f}"
                )
                return SMCSignal(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    sweep_level=recent_sh,
                    ob_top=ob_top,
                    ob_bottom=ob_bottom,
                    risk_price_distance=risk_dist,
                    reward_price_distance=abs(current_price - tp),
                    rr_ratio=config.RISK_REWARD_RATIO
                )

    return None
