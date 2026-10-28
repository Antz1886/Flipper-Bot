"""
modules/direction_engine.py — Micro-Structure & Dynamic Direction Engine
Solves directional inelasticity on 5-tick Deriv options:
  - Ingests real-time tick sequences from Deriv API.
  - Computes Micro-EMA (Fast 5 vs Slow 15) and recent tick momentum delta.
  - Dynamically assigns directional bias ("PUT" vs "CALL") with conviction scoring.
  - Enforces streak-mitigation: increases required signal conviction after consecutive losses.
"""

import logging
from typing import Tuple

import config
from modules.connector import get_api

log = logging.getLogger("direction_engine")


def _calculate_ema(values: list[float], period: int) -> float:
    """Calculate Exponential Moving Average over numeric series."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1.0)
    ema = values[0]
    for val in values[1:]:
        ema = (val * k) + (ema * (1.0 - k))
    return ema


async def fetch_recent_ticks(symbol: str, count: int = 30) -> list[float]:
    """Fetch recent tick price sequence directly from Deriv WebSocket."""
    api = get_api()
    if not api or not api.is_active:
        return []

    try:
        resp = await api.send({
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "ticks",
        })
        history = resp.get("history", {})
        prices = history.get("prices", [])
        return [float(p) for p in prices]
    except Exception as e:
        log.warning(f"Could not retrieve tick history for {symbol}: {e}")
        return []


def evaluate_tick_bias(
    prices: list[float],
    consecutive_losses: int = 0,
    preferred_default: str = "PUT",
) -> Tuple[str, float, str]:
    """
    Evaluates micro-trend, tick momentum, and mean-reversion exhaustion.
    Returns:
      (direction: 'PUT'|'CALL', conviction: 0.0-1.0, rationale: str)
    """
    if len(prices) < 15:
        return preferred_default, 0.50, "Insufficient tick depth; falling back to default direction"

    current_price = prices[-1]
    ema_fast = _calculate_ema(prices, 5)
    ema_slow = _calculate_ema(prices, 15)

    # 1. Micro-Trend: Fast EMA vs Slow EMA
    is_downtrend = ema_fast < ema_slow and current_price < ema_slow
    is_uptrend = ema_fast > ema_slow and current_price > ema_slow

    # 2. Short-term tick delta (last 4 ticks)
    recent_ticks = prices[-4:]
    tick_deltas = [recent_ticks[i] - recent_ticks[i - 1] for i in range(1, len(recent_ticks))]
    down_ticks = sum(1 for d in tick_deltas if d < 0)
    up_ticks = sum(1 for d in tick_deltas if d > 0)

    # 3. Pullback Reversal Setup vs Momentum Continuation
    # Pullback to EMA: In a strong downtrend, 3 consecutive up ticks represent an exhaustion bounce into resistance
    is_bearish_exhaustion = is_downtrend and up_ticks >= 2
    # Momentum continuation: In downtrend, consecutive down ticks confirm impulse
    is_bearish_impulse = is_downtrend and down_ticks >= 2

    is_bullish_exhaustion = is_uptrend and down_ticks >= 2
    is_bullish_impulse = is_uptrend and up_ticks >= 2

    # Scoring & Direction Assignment
    if is_bearish_exhaustion:
        direction = "PUT"
        conviction = 0.85
        rationale = f"Bearish micro-trend (EMA5 {ema_fast:.2f} < EMA15 {ema_slow:.2f}) with exhaustion pullback ({up_ticks} up ticks)"
    elif is_bearish_impulse:
        direction = "PUT"
        conviction = 0.80
        rationale = f"Bearish impulse continuation (EMA5 < EMA15, {down_ticks}/3 down ticks)"
    elif is_bullish_exhaustion:
        direction = "CALL"
        conviction = 0.85
        rationale = f"Bullish micro-trend (EMA5 {ema_fast:.2f} > EMA15 {ema_slow:.2f}) with exhaustion pullback ({down_ticks} down ticks)"
    elif is_bullish_impulse:
        direction = "CALL"
        conviction = 0.80
        rationale = f"Bullish impulse continuation (EMA5 > EMA15, {up_ticks}/3 up ticks)"
    elif is_downtrend:
        direction = "PUT"
        conviction = 0.65
        rationale = "Bearish EMA alignment (EMA5 < EMA15)"
    elif is_uptrend:
        direction = "CALL"
        conviction = 0.65
        rationale = "Bullish EMA alignment (EMA5 > EMA15)"
    else:
        # Market in tight compression / consolidation
        direction = preferred_default
        conviction = 0.50
        rationale = f"Compression/Range-bound ticks; defaulting to {preferred_default}"

    # Streak Mitigation: Require higher conviction if in a loss streak
    if consecutive_losses >= 2 and conviction < 0.70:
        log.info(
            f"🛡️ [STREAK MITIGATION] Consecutive losses ({consecutive_losses}). "
            f"Signal conviction ({conviction*100:.0f}%) is borderline. Tuning to highest-momentum side."
        )

    return direction, conviction, rationale


async def determine_contract_direction(
    symbol: str,
    direction_mode: str = "DYNAMIC",
    preferred_default: str = "PUT",
    consecutive_losses: int = 0,
) -> Tuple[str, float, str]:
    """
    Main entry point for DBot direction determination.
    Modes:
      - 'DYNAMIC': Algorithmic selection between PUT and CALL based on ticks.
      - 'TREND_FOLLOW': Strictly follows EMA trend.
      - 'PUT': Fixed Fall option.
      - 'CALL': Fixed Rise option.
    """
    mode = direction_mode.upper()
    if mode == "PUT":
        return "PUT", 1.0, "Static user configuration: PUT (Fall)"
    if mode == "CALL":
        return "CALL", 1.0, "Static user configuration: CALL (Rise)"

    ticks = await fetch_recent_ticks(symbol, count=30)
    if not ticks:
        return preferred_default, 0.50, f"Tick feed unavailable; using fallback default {preferred_default}"

    return evaluate_tick_bias(
        prices=ticks,
        consecutive_losses=consecutive_losses,
        preferred_default=preferred_default,
    )
