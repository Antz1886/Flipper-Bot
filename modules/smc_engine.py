"""
modules/smc_engine.py — Quantitative Institutional SMC Engine
Implements Conti's single structural edge without lookahead bias or indicator curve-fitting:
  1. Closed-Bar Structural Pivots: Swing Highs/Lows computed strictly on closed candles.
  2. Chronological Liquidity Sweep: Sweep must occur AFTER the swing pivot is confirmed.
  3. Displacement / MSS Confirmation: Impulse rejection must exceed ATR displacement threshold.
  4. Walk-Forward Validation: Verifies the Order Block has not been invalidated (stopped out)
     or already mitigated & exhausted on intermediate bars.
  5. Spread/Friction Guard: Enforces a minimum stop distance (>= MIN_STOP_ATR_RATIO * ATR)
     so Deriv multiplier spread does not degrade the 1:2 R:R expectancy.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)

# Tracks already-executed setups (symbol, direction, rounded_sweep_level) to prevent duplicate entries on the same OB
_consumed_setups: set[tuple[str, str, float]] = set()


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
    atr: float = 0.0


def mark_setup_consumed(signal: SMCSignal):
    """Mark a specific liquidity sweep Order Block as consumed so it is never traded twice."""
    _consumed_setups.add((signal.symbol, signal.direction, round(signal.sweep_level, 6)))


def reset_consumed_setups():
    """Clear consumed setups cache (used in testing or session reset)."""
    _consumed_setups.clear()


def calculate_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """Calculate Wilder's / Simple Average True Range (ATR) on OHLC DataFrame."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def identify_swings(df: pd.DataFrame, swing_len: int = config.SWING_LENGTH) -> pd.DataFrame:
    """
    Identify confirmed swing highs and swing lows without lookahead bias.
    A swing at index `i` is only confirmed once `i + swing_len` bars have closed.
    """
    df = df.copy()
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)

    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(swing_len, n - swing_len):
        current_h = highs[i]
        current_l = lows[i]

        # Check Swing High: strictly higher than `swing_len` bars to the right, >= to the left
        is_sh = True
        for offset in range(1, swing_len + 1):
            if highs[i - offset] > current_h or highs[i + offset] >= current_h:
                is_sh = False
                break
        if is_sh:
            swing_highs[i] = current_h

        # Check Swing Low: strictly lower than `swing_len` bars to the right, <= to the left
        is_sl = True
        for offset in range(1, swing_len + 1):
            if lows[i - offset] < current_l or lows[i + offset] <= current_l:
                is_sl = False
                break
        if is_sl:
            swing_lows[i] = current_l

    df["swing_high"] = swing_highs
    df["swing_low"] = swing_lows
    return df


def _check_bullish_setup(
    symbol: str,
    closed_df: pd.DataFrame,
    current_price: float,
    current_atr: float,
    swing_len: int,
) -> Optional[SMCSignal]:
    """
    Evaluate Bullish Institutional Setup:
    1. Locate most recent confirmed Swing Low at positional index `sl_pos`.
    2. Scan ONLY bars after `sl_pos + swing_len` for a liquidity sweep (`low < swing_low` and `close > swing_low`).
    3. Confirm bullish displacement (strong rejection range/body >= MIN_DISPLACEMENT_ATR * ATR).
    4. Walk forward through subsequent closed bars to ensure OB was NOT invalidated (`low <= ob_bottom`)
       and NOT already mitigated & exhausted.
    5. Trigger entry if `current_price` is inside `[ob_bottom, ob_top]` and stop size >= MIN_STOP_ATR_RATIO * ATR.
    """
    n_closed = len(closed_df)
    valid_sl_positions = np.where(~np.isnan(closed_df["swing_low"].values))[0]
    if len(valid_sl_positions) == 0:
        return None

    # Evaluate the most recent confirmed swing lows (up to last 4) so that if a sweep candle
    # itself becomes a confirmed swing low after `swing_len` bars, the swept swing low is still evaluated.
    for sl_pos in reversed(valid_sl_positions[-4:]):
        sl_pos = int(sl_pos)
        recent_sl = float(closed_df["swing_low"].iloc[sl_pos])

        if (symbol, "LONG", round(recent_sl, 6)) in _consumed_setups:
            continue

        # A swing at sl_pos is only confirmed at sl_pos + swing_len.
        scan_start = max(sl_pos + swing_len, n_closed - 30)
        if scan_start >= n_closed:
            continue

        # Ensure the swing level wasn't already broken by a full candle close before scan_start
        broken_early = False
        for pre_idx in range(sl_pos + 1, scan_start):
            if float(closed_df["close"].iloc[pre_idx]) < recent_sl:
                broken_early = True
                break
        if broken_early:
            continue

        sweep_idx = None
        disp_idx = None
        sweep_extreme_low = None

        for idx in range(scan_start, n_closed):
            bar = closed_df.iloc[idx]
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])
            bar_close = float(bar["close"])

            # If a candle closes decisively below the swing low without rejecting, the level broke (not a sweep)
            if bar_close < recent_sl and (recent_sl - bar_close) > 0.25 * current_atr:
                break

            # Check for wick sweep below recent_sl that rejects back above recent_sl
            if bar_low < recent_sl and bar_close >= recent_sl:
                rejection_tail = bar_close - bar_low
                if rejection_tail >= config.MIN_DISPLACEMENT_ATR * current_atr:
                    sweep_idx = idx
                    disp_idx = idx
                    sweep_extreme_low = bar_low
                    break
                elif idx + 1 < n_closed:
                    next_bar = closed_df.iloc[idx + 1]
                    next_body = float(next_bar["close"]) - float(next_bar["open"])
                    if float(next_bar["close"]) > bar_high and next_body >= config.MIN_DISPLACEMENT_ATR * current_atr:
                        sweep_idx = idx
                        disp_idx = idx + 1
                        sweep_extreme_low = min(bar_low, float(next_bar["low"]))
                        break

        if sweep_idx is None or disp_idx is None or sweep_extreme_low is None:
            continue

        sweep_bar = closed_df.iloc[sweep_idx]
        ob_bottom = float(sweep_extreme_low)
        ob_top = max(float(sweep_bar["open"]), float(sweep_bar["close"]))

        if ob_top <= ob_bottom:
            ob_top = float(sweep_bar["high"])

        ob_height = ob_top - ob_bottom
        if ob_height < config.MIN_STOP_ATR_RATIO * current_atr:
            continue

        # Walk forward from displacement bar + 1 to the end of closed_df
        invalidated_or_spent = False
        mitigated_previously = False
        for k in range(disp_idx + 1, n_closed):
            k_low = float(closed_df["low"].iloc[k])
            k_high = float(closed_df["high"].iloc[k])

            # 1. Invalidation check: If any subsequent closed bar broke below ob_bottom, setup is dead
            if k_low <= ob_bottom:
                invalidated_or_spent = True
                break

            # 2. Check if already mitigated and rallied away (> 2R above ob_top)
            if k_low <= ob_top:
                mitigated_previously = True
            if mitigated_previously and k_high >= ob_top + ob_height * config.RISK_REWARD_RATIO:
                invalidated_or_spent = True
                break

        if invalidated_or_spent:
            continue

        # Mitigation Entry Check on current live price:
        if ob_bottom < current_price <= ob_top:
            sl = ob_bottom
            risk_dist = current_price - sl
            if risk_dist < config.MIN_STOP_ATR_RATIO * current_atr:
                sl = current_price - (config.MIN_STOP_ATR_RATIO * current_atr)
                risk_dist = current_price - sl

            tp = current_price + (risk_dist * config.RISK_REWARD_RATIO)
            log.info(
                f"🎯 [SMC BULLISH EDGE] {symbol} | Sweep Low: ${recent_sl:.5f} | "
                f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                f"SL: ${sl:.5f} | TP: ${tp:.5f} | ATR: {current_atr:.5f}"
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
                reward_price_distance=tp - current_price,
                rr_ratio=config.RISK_REWARD_RATIO,
                atr=current_atr,
            )

    return None


def _check_bearish_setup(
    symbol: str,
    closed_df: pd.DataFrame,
    current_price: float,
    current_atr: float,
    swing_len: int,
) -> Optional[SMCSignal]:
    """
    Evaluate Bearish Institutional Setup across recent confirmed Swing Highs.
    """
    n_closed = len(closed_df)
    valid_sh_positions = np.where(~np.isnan(closed_df["swing_high"].values))[0]
    if len(valid_sh_positions) == 0:
        return None

    for sh_pos in reversed(valid_sh_positions[-4:]):
        sh_pos = int(sh_pos)
        recent_sh = float(closed_df["swing_high"].iloc[sh_pos])

        if (symbol, "SHORT", round(recent_sh, 6)) in _consumed_setups:
            continue

        scan_start = max(sh_pos + swing_len, n_closed - 30)
        if scan_start >= n_closed:
            continue

        broken_early = False
        for pre_idx in range(sh_pos + 1, scan_start):
            if float(closed_df["close"].iloc[pre_idx]) > recent_sh:
                broken_early = True
                break
        if broken_early:
            continue

        sweep_idx = None
        disp_idx = None
        sweep_extreme_high = None

        for idx in range(scan_start, n_closed):
            bar = closed_df.iloc[idx]
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])
            bar_close = float(bar["close"])

            if bar_close > recent_sh and (bar_close - recent_sh) > 0.25 * current_atr:
                break

            if bar_high > recent_sh and bar_close <= recent_sh:
                rejection_tail = bar_high - bar_close
                if rejection_tail >= config.MIN_DISPLACEMENT_ATR * current_atr:
                    sweep_idx = idx
                    disp_idx = idx
                    sweep_extreme_high = bar_high
                    break
                elif idx + 1 < n_closed:
                    next_bar = closed_df.iloc[idx + 1]
                    next_body = float(next_bar["open"]) - float(next_bar["close"])
                    if float(next_bar["close"]) < bar_low and next_body >= config.MIN_DISPLACEMENT_ATR * current_atr:
                        sweep_idx = idx
                        disp_idx = idx + 1
                        sweep_extreme_high = max(bar_high, float(next_bar["high"]))
                        break

        if sweep_idx is None or disp_idx is None or sweep_extreme_high is None:
            continue

        sweep_bar = closed_df.iloc[sweep_idx]
        ob_top = float(sweep_extreme_high)
        ob_bottom = min(float(sweep_bar["open"]), float(sweep_bar["close"]))

        if ob_top <= ob_bottom:
            ob_bottom = float(sweep_bar["low"])

        ob_height = ob_top - ob_bottom
        if ob_height < config.MIN_STOP_ATR_RATIO * current_atr:
            continue

        invalidated_or_spent = False
        mitigated_previously = False
        for k in range(disp_idx + 1, n_closed):
            k_low = float(closed_df["low"].iloc[k])
            k_high = float(closed_df["high"].iloc[k])

            if k_high >= ob_top:
                invalidated_or_spent = True
                break

            if k_high >= ob_bottom:
                mitigated_previously = True
            if mitigated_previously and k_low <= ob_bottom - ob_height * config.RISK_REWARD_RATIO:
                invalidated_or_spent = True
                break

        if invalidated_or_spent:
            continue

        if ob_bottom <= current_price < ob_top:
            sl = ob_top
            risk_dist = sl - current_price
            if risk_dist < config.MIN_STOP_ATR_RATIO * current_atr:
                sl = current_price + (config.MIN_STOP_ATR_RATIO * current_atr)
                risk_dist = sl - current_price

            tp = current_price - (risk_dist * config.RISK_REWARD_RATIO)
            log.info(
                f"🎯 [SMC BEARISH EDGE] {symbol} | Sweep High: ${recent_sh:.5f} | "
                f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                f"SL: ${sl:.5f} | TP: ${tp:.5f} | ATR: {current_atr:.5f}"
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
                reward_price_distance=current_price - tp,
                rr_ratio=config.RISK_REWARD_RATIO,
                atr=current_atr,
            )

    return None


def detect_institutional_setup(symbol: str, df: pd.DataFrame) -> Optional[SMCSignal]:
    """
    Evaluate structural price action on OHLC data:
    - Separates closed historical bars (`df.iloc[:-1]`) from the live forming bar (`df.iloc[-1]`).
    - Computes ATR and confirmed Swing Highs/Lows strictly on closed bars.
    - Checks for unmitigated, non-invalidated Bullish or Bearish Order Block mitigations.
    """
    min_bars = config.SWING_LENGTH * 3 + 15
    if df is None or len(df) < min_bars:
        return None

    # Live price from the actively forming candle
    current_price = float(df["close"].iloc[-1])

    # Closed candles only for structural swings, sweeps, and ATR
    closed_df = df.iloc[:-1].copy().reset_index(drop=True)
    closed_df["atr"] = calculate_atr(closed_df, config.ATR_PERIOD)
    current_atr = float(closed_df["atr"].iloc[-1])
    if current_atr <= 0 or np.isnan(current_atr):
        return None

    closed_swings = identify_swings(closed_df, config.SWING_LENGTH)

    # Check Bullish Setup first
    bull_sig = _check_bullish_setup(symbol, closed_swings, current_price, current_atr, config.SWING_LENGTH)
    if bull_sig is not None:
        return bull_sig

    # Check Bearish Setup
    bear_sig = _check_bearish_setup(symbol, closed_swings, current_price, current_atr, config.SWING_LENGTH)
    if bear_sig is not None:
        return bear_sig

    return None
