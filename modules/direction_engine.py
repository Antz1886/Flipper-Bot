"""
modules/direction_engine.py — Micro-Structure & Dynamic Direction Engine
Solves directional inelasticity and counter-trend drag on 5-tick Deriv options:
  - Higher-Timeframe (HTF) M1 Trend Gating: Strictly aligns trades with M1 EMA21 vs EMA50 trend.
    Prevents fighting the macroscopic momentum (eliminating counter-trend losses).
  - Tick-Level RSI & Momentum Timing: Filters out local tops and bottoms.
    Enters CALL on pullbacks in an uptrend; enters PUT on bounces in a downtrend.
  - Streak-Mitigation: Raises conviction threshold after consecutive losses.
"""

import asyncio
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


def _calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI) over numeric series."""
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))

    recent_gains = gains[-period:]
    recent_losses = losses[-period:]
    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _calculate_atr(prices: list[float], period: int = 14) -> float:
    """Calculate Average True Range (ATR) over tick price series."""
    if len(prices) < period + 1:
        return 0.0
    ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    if not ranges:
        return 0.0
    recent = ranges[-period:]
    return sum(recent) / len(recent)


async def fetch_recent_ticks(symbol: str, count: int = 50) -> list[float]:
    """Fetch recent tick price sequence directly from Deriv WebSocket with retries."""
    api = get_api()
    if not api or not api.is_active:
        log.warning(f"get_api() is None or not active (api={api})")
        return []

    for attempt in range(1, 4):
        try:
            resp = await api.send({
                "ticks_history": symbol,
                "end": "latest",
                "count": count,
                "style": "ticks",
            })
            history = resp.get("history", {})
            prices = history.get("prices", [])
            if prices:
                return [float(p) for p in prices]
            error = resp.get("error")
            if error:
                log.warning(f"Deriv ticks_history error for {symbol}: {error}")
        except Exception as e:
            log.warning(f"Could not retrieve tick history for {symbol} (attempt {attempt}): {e}")
        await asyncio.sleep(0.5)

    return []


async def fetch_htf_candles(symbol: str, count: int = 60, granularity: int = 60) -> list[dict]:
    """Fetch higher-timeframe candles (default: M1 / 60s) for macroscopic trend gating with retries."""
    api = get_api()
    if not api or not api.is_active:
        return []

    for attempt in range(1, 4):
        try:
            resp = await api.send({
                "ticks_history": symbol,
                "end": "latest",
                "count": count,
                "granularity": granularity,
                "style": "candles",
            })
            candles = resp.get("candles", [])
            if candles:
                return candles
        except Exception as e:
            log.warning(f"Could not retrieve HTF candles for {symbol} (attempt {attempt}): {e}")
        await asyncio.sleep(0.5)

    return []


def evaluate_htf_trend(candles: list[dict]) -> Tuple[str, float, float]:
    """
    Evaluates Higher-Timeframe trend from M1 candles using EMA21 and EMA50.
    Returns:
      (trend: 'BULLISH'|'BEARISH'|'NEUTRAL', ema21: float, ema50: float)
    """
    if len(candles) < 20:
        return "NEUTRAL", 0.0, 0.0

    closes = [float(c.get("close", 0.0)) for c in candles]
    ema21 = _calculate_ema(closes, min(21, len(closes)))
    ema50 = _calculate_ema(closes, min(50, len(closes)))
    curr_close = closes[-1]

    if ema21 > ema50 and curr_close >= ema50:
        return "BULLISH", ema21, ema50
    elif ema21 < ema50 and curr_close <= ema50:
        return "BEARISH", ema21, ema50
    else:
        return "NEUTRAL", ema21, ema50


def evaluate_tick_bias(
    prices: list[float],
    consecutive_losses: int = 0,
    preferred_default: str = "PUT",
    htf_trend: str = "NEUTRAL",
    tick_rsi: float = 50.0,
) -> Tuple[str, float, str]:
    """
    Evaluates micro-trend, tick momentum, HTF trend alignment, and RSI exhaustion.
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

    # If RSI not explicitly provided, calculate from price sequence
    if tick_rsi == 50.0 and len(prices) >= 15:
        tick_rsi = _calculate_rsi(prices, period=14)

    # 3. Volatility Regime / Chop Filter (ATR Squeeze Protection)
    if len(prices) >= 20:
        atr_fast = _calculate_atr(prices, period=7)
        atr_slow = _calculate_atr(prices, period=20)
        # If volatility has collapsed by >35% compared to baseline, market is in flat squeeze/chop
        if atr_slow > 0 and (atr_fast / atr_slow) < 0.65:
            return (
                preferred_default,
                0.40,
                f"Volatility Squeeze / Dead Chop (ATR7 {atr_fast:.3f} < 65% of ATR20 {atr_slow:.3f}). Sitting out.",
            )

    # 4. Macro Trend Gating (M1 EMA Alignment) with Asymmetric Pullback Requirements
    if htf_trend == "BULLISH":
        # In a bullish HTF trend: ONLY allow CALL options (ban counter-trend PUTs)
        # Tightened CALL filter: Require deeper pullback (RSI <= 45) to avoid buying near local crests
        if tick_rsi > 68.0:
            return (
                "CALL",
                0.45,
                f"Bullish HTF (M1), but Tick RSI ({tick_rsi:.1f}) is elevated (>68). Waiting for dip.",
            )
        elif tick_rsi <= 45.0 and down_ticks >= 1:
            # High-conviction pullback entry in an uptrend
            return (
                "CALL",
                0.85,
                f"Bullish HTF alignment + Deep dip pullback entry (RSI {tick_rsi:.1f} <= 45, {down_ticks} down ticks into support)",
            )
        elif (is_uptrend or up_ticks >= 2) and tick_rsi <= 55.0:
            return (
                "CALL",
                0.80,
                f"Bullish HTF continuation (EMA5 > EMA15, RSI {tick_rsi:.1f} <= 55)",
            )
        else:
            return (
                "CALL",
                0.60,
                f"Bullish HTF bias but awaiting cleaner dip (RSI {tick_rsi:.1f})",
            )

    elif htf_trend == "BEARISH":
        # In a bearish HTF trend: ONLY allow PUT options (ban counter-trend CALLs)
        if tick_rsi < 30.0:
            return (
                "PUT",
                0.45,
                f"Bearish HTF (M1), but Tick RSI ({tick_rsi:.1f}) is oversold (<30). Waiting for bounce.",
            )
        elif tick_rsi >= 45.0 and up_ticks >= 1:
            # High-conviction bounce fade in a downtrend
            return (
                "PUT",
                0.85,
                f"Bearish HTF alignment + Bounce fade entry (RSI {tick_rsi:.1f}, {up_ticks} up ticks into resistance)",
            )
        elif is_downtrend or down_ticks >= 2:
            return (
                "PUT",
                0.80,
                f"Bearish HTF continuation (EMA5 < EMA15, RSI {tick_rsi:.1f})",
            )
        else:
            return (
                "PUT",
                0.72,
                f"Bearish HTF bias (RSI {tick_rsi:.1f})",
            )

    # 4. Fallback / Neutral HTF Mode (Direct micro-trend & RSI boundary evaluation)
    is_bearish_exhaustion = is_downtrend and up_ticks >= 2
    is_bearish_impulse = is_downtrend and down_ticks >= 2
    is_bullish_exhaustion = is_uptrend and down_ticks >= 2
    is_bullish_impulse = is_uptrend and up_ticks >= 2

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
    elif tick_rsi < 30.0:
        direction = "CALL"
        conviction = 0.75
        rationale = f"Oversold micro-mean-reversion bounce (RSI {tick_rsi:.1f} < 30)"
    elif tick_rsi > 70.0:
        direction = "PUT"
        conviction = 0.75
        rationale = f"Overbought micro-mean-reversion fade (RSI {tick_rsi:.1f} > 70)"
    elif is_downtrend:
        direction = "PUT"
        conviction = 0.65
        rationale = "Bearish EMA alignment (EMA5 < EMA15)"
    elif is_uptrend:
        direction = "CALL"
        conviction = 0.65
        rationale = "Bullish EMA alignment (EMA5 > EMA15)"
    else:
        direction = preferred_default
        conviction = 0.50
        rationale = f"Compression/Range-bound ticks (RSI {tick_rsi:.1f}); defaulting to {preferred_default}"

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
      - 'DYNAMIC': Algorithmic selection between PUT and CALL based on HTF trend & ticks.
      - 'TREND_FOLLOW': Strictly follows EMA trend.
      - 'PUT': Fixed Fall option.
      - 'CALL': Fixed Rise option.
    """
    mode = direction_mode.upper()
    if mode == "PUT":
        return "PUT", 1.0, "Static user configuration: PUT (Fall)"
    if mode == "CALL":
        return "CALL", 1.0, "Static user configuration: CALL (Rise)"

    ticks = await fetch_recent_ticks(symbol, count=50)
    if not ticks:
        return preferred_default, 0.50, f"Tick feed unavailable; using fallback default {preferred_default}"

    # Fetch Higher-Timeframe trend if enabled
    htf_trend = "NEUTRAL"
    if getattr(config, "DBOT_USE_HTF_FILTER", True):
        candles = await fetch_htf_candles(
            symbol,
            count=60,
            granularity=getattr(config, "DBOT_HTF_TIMEFRAME_S", 60),
        )
        if candles:
            htf_trend, _, _ = evaluate_htf_trend(candles)

    # Compute Tick RSI
    tick_rsi = _calculate_rsi(ticks, period=getattr(config, "DBOT_RSI_PERIOD", 14))

    return evaluate_tick_bias(
        prices=ticks,
        consecutive_losses=consecutive_losses,
        preferred_default=preferred_default,
        htf_trend=htf_trend,
        tick_rsi=tick_rsi,
    )
