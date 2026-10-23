"""
modules/smc_engine.py — Quantitative Institutional SMC Engine (Forex Majors Edition)
Implements Conti's institutional structural edge with macro microstructure filters:
  1. Macro Session Windows: Restricts sweeps & signals to London Open (07:00–11:00 GMT)
     and New York Open (13:00–17:00 GMT). Asian session sweeps are discarded as low-volume fakeouts.
  2. HTF Premium/Discount Dealing Range (H4): Authorizes Longs strictly in H4 Discount (< 50% equilibrium)
     and Shorts strictly in H4 Premium (> 50% equilibrium).
  3. Macro Liquidity Pool Intersection: Breached swing levels must align within 3–5 pips of
     Previous Day High/Low (PDH/PDL), Asian Session High/Low, or Equal Highs/Lows (EQH/EQL).
  4. Closed-Bar Structural Pivots: Swings computed strictly on confirmed closed bars.
  5. Chronological Sweep & ATR Displacement: Rejection range/body must exceed `MIN_DISPLACEMENT_ATR`.
  6. Walk-Forward Validation & Deduplication: Prevents duplicate execution on previously
     consumed or invalidated Order Blocks.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time as time_cls
from typing import Optional
import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)

# Tracks already-executed setups to prevent duplicate entries on the same OB or sweep level
_consumed_setups: set[tuple] = set()


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
    macro_pool_desc: str = ""
    htf_equilibrium: float = 0.0
    displacement_ratio: float = 1.0
    htf_depth_pct: float = 0.50
    macro_pool_type: str = "SWING"


def mark_setup_consumed(signal: SMCSignal):
    """Mark both the swept swing level and the Order Block zone as consumed so it is never traded twice."""
    _consumed_setups.add(("SWEEP", signal.symbol, signal.direction, round(signal.sweep_level, 5)))
    _consumed_setups.add(("OB", signal.symbol, signal.direction, round(signal.ob_bottom, 5), round(signal.ob_top, 5)))


def reset_consumed_setups():
    """Clear consumed setups cache (used in testing or session reset)."""
    _consumed_setups.clear()


# ─── 1. Macro Session Window Verification (London & New York) ─────────────────

def is_in_macro_session(dt: Optional[datetime | pd.Timestamp] = None) -> bool:
    """
    Check if the given datetime (or current UTC time) falls within
    the London Open (07:00–11:00 GMT) or New York Open (13:00–17:00 GMT) institutional windows.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    cur_time = dt.time()
    windows = getattr(config, "MACRO_SESSION_WINDOWS_UTC", [(7, 0, 11, 0), (13, 0, 17, 0)])

    for (sh, sm, eh, em) in windows:
        if time_cls(sh, sm) <= cur_time <= time_cls(eh, em):
            return True
    return False


def _get_bar_timestamp(row_or_df) -> Optional[datetime]:
    """Extract a timezone-aware UTC datetime from a Series row or index."""
    if "time" in row_or_df:
        val = row_or_df["time"]
        if isinstance(val, (datetime, pd.Timestamp)):
            dt = val.to_pydatetime() if hasattr(val, "to_pydatetime") else val
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    if hasattr(row_or_df, "name") and isinstance(row_or_df.name, (datetime, pd.Timestamp)):
        dt = row_or_df.name.to_pydatetime() if hasattr(row_or_df.name, "to_pydatetime") else row_or_df.name
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return None


# ─── 2. Pip Size & Mathematical Microstructure Helpers ───────────────────────

def get_pip_size(symbol: str) -> float:
    """Pip size for Forex majors: 0.01 for JPY pairs, 0.0001 for all others."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def calculate_atr(df: pd.DataFrame, period: int = getattr(config, "ATR_PERIOD", 14)) -> pd.Series:
    """Calculate Wilder's / Simple Average True Range (ATR) on OHLC DataFrame."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def identify_swings(df: pd.DataFrame, swing_len: int = getattr(config, "SWING_LENGTH", 5)) -> pd.DataFrame:
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

        # Check Swing High
        is_sh = True
        for offset in range(1, swing_len + 1):
            if highs[i - offset] > current_h or highs[i + offset] >= current_h:
                is_sh = False
                break
        if is_sh:
            swing_highs[i] = current_h

        # Check Swing Low
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


# ─── 3. HTF Dealing Range & Premium/Discount Matrices ────────────────────────

def calculate_h4_dealing_range(
    df_h4: pd.DataFrame,
    swing_len: int = getattr(config, "HTF_SWING_LENGTH", 3),
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate the H4 dealing range from confirmed H4 Swing High and Swing Low:
    Returns (h4_swing_high, h4_swing_low, equilibrium_50pct).
    """
    if df_h4 is None or len(df_h4) < (swing_len * 2 + 5):
        return None, None, None

    closed_h4 = df_h4.iloc[:-1].copy().reset_index(drop=True) if len(df_h4) > 5 else df_h4.copy().reset_index(drop=True)
    swings = identify_swings(closed_h4, swing_len)

    sh_indices = np.where(~np.isnan(swings["swing_high"].values))[0]
    sl_indices = np.where(~np.isnan(swings["swing_low"].values))[0]

    if len(sh_indices) == 0 or len(sl_indices) == 0:
        return None, None, None

    recent_sh = float(swings["swing_high"].iloc[sh_indices[-1]])
    recent_sl = float(swings["swing_low"].iloc[sl_indices[-1]])

    if recent_sh <= recent_sl:
        # Scan back for the valid swing high that sits above the recent swing low
        for sh_idx in reversed(sh_indices):
            cand_sh = float(swings["swing_high"].iloc[sh_idx])
            if cand_sh > recent_sl:
                recent_sh = cand_sh
                break

    equilibrium = (recent_sh + recent_sl) / 2.0
    return recent_sh, recent_sl, equilibrium


def check_htf_premium_discount(
    direction: str,
    current_price: float,
    df_h4: Optional[pd.DataFrame] = None,
) -> tuple[bool, str, float]:
    """
    Enforce H4 dealing range equilibrium:
    - Longs authorized ONLY in Discount (price < 50% equilibrium)
    - Shorts authorized ONLY in Premium (price > 50% equilibrium)
    """
    if df_h4 is None or not getattr(config, "ENFORCE_HTF_PREMIUM_DISCOUNT", True):
        return True, "HTF filter disabled or no H4 data provided", 0.0

    h4_high, h4_low, equilibrium = calculate_h4_dealing_range(df_h4, getattr(config, "HTF_SWING_LENGTH", 3))
    if equilibrium is None or h4_high is None or h4_low is None:
        return True, "No confirmed H4 dealing range identified; proceeding on M15 structure", 0.0

    if direction.upper() == "LONG":
        if current_price > equilibrium:
            return (
                False,
                f"Price {current_price:.5f} is in H4 Premium zone (> 50% Eq: {equilibrium:.5f}, Range: [{h4_low:.5f} - {h4_high:.5f}]). Longs restricted to Discount.",
                equilibrium,
            )
        return True, f"Price {current_price:.5f} is in H4 Discount zone (< 50% Eq: {equilibrium:.5f})", equilibrium

    elif direction.upper() == "SHORT":
        if current_price < equilibrium:
            return (
                False,
                f"Price {current_price:.5f} is in H4 Discount zone (< 50% Eq: {equilibrium:.5f}, Range: [{h4_low:.5f} - {h4_high:.5f}]). Shorts restricted to Premium.",
                equilibrium,
            )
        return True, f"Price {current_price:.5f} is in H4 Premium zone (> 50% Eq: {equilibrium:.5f})", equilibrium

    return True, "Neutral", 0.0


# ─── 4. Macro Liquidity Pool Filters (PDH/PDL, Asia H/L, EQH/EQL) ────────────

def extract_macro_liquidity_pools(
    symbol: str,
    closed_df: pd.DataFrame,
    current_atr: float,
) -> list[dict]:
    """
    Extract Macro Liquidity Pools from historical price action:
      1. PDH / PDL (Previous Day High & Low)
      2. Asian Session High & Low (00:00 - 07:00 UTC)
      3. EQH / EQL (Equal Highs and Equal Lows within pip tolerance)
    """
    pools = []
    pip_size = get_pip_size(symbol)
    pip_tol = getattr(config, "MACRO_POOL_PIP_TOLERANCE", 5.0)
    price_tol = max(pip_tol * pip_size, 0.20 * current_atr)

    has_time = "time" in closed_df.columns or isinstance(closed_df.index, pd.DatetimeIndex)

    if has_time:
        times = closed_df["time"] if "time" in closed_df.columns else pd.Series(closed_df.index)
        times_utc = pd.to_datetime(times, utc=True)
        temp_df = closed_df.copy()
        temp_df["_time_utc"] = times_utc
        temp_df["_date"] = temp_df["_time_utc"].dt.date
        temp_df["_hour"] = temp_df["_time_utc"].dt.hour

        unique_dates = temp_df["_date"].unique()
        if len(unique_dates) >= 2:
            prev_date = unique_dates[-2]
            prev_bars = temp_df[temp_df["_date"] == prev_date]
            if len(prev_bars) > 0:
                pdh = float(prev_bars["high"].max())
                pdl = float(prev_bars["low"].min())
                pools.append({"type": "PDH", "price": pdh, "side": "HIGH", "desc": f"PDH (${pdh:.5f})"})
                pools.append({"type": "PDL", "price": pdl, "side": "LOW", "desc": f"PDL (${pdl:.5f})"})

        # Asian Session (00:00 - 07:00 UTC of latest date)
        latest_date = unique_dates[-1]
        asia_start = getattr(config, "ASIA_SESSION_START_UTC_HOUR", 0)
        asia_end = getattr(config, "ASIA_SESSION_END_UTC_HOUR", 7)
        asia_bars = temp_df[(temp_df["_date"] == latest_date) & (temp_df["_hour"] >= asia_start) & (temp_df["_hour"] < asia_end)]
        if len(asia_bars) >= 2:
            asia_h = float(asia_bars["high"].max())
            asia_l = float(asia_bars["low"].min())
            pools.append({"type": "ASIA_HIGH", "price": asia_h, "side": "HIGH", "desc": f"Asia High (${asia_h:.5f})"})
            pools.append({"type": "ASIA_LOW", "price": asia_l, "side": "LOW", "desc": f"Asia Low (${asia_l:.5f})"})

    # Equal Highs (EQH) and Equal Lows (EQL) from confirmed swings
    sh_vals = closed_df["swing_high"].dropna().values if "swing_high" in closed_df.columns else []
    sl_vals = closed_df["swing_low"].dropna().values if "swing_low" in closed_df.columns else []

    for i in range(len(sh_vals)):
        for j in range(i + 1, len(sh_vals)):
            if abs(sh_vals[i] - sh_vals[j]) <= price_tol:
                eqh_price = (sh_vals[i] + sh_vals[j]) / 2.0
                pools.append({"type": "EQH", "price": eqh_price, "side": "HIGH", "desc": f"EQH (${eqh_price:.5f})"})

    for i in range(len(sl_vals)):
        for j in range(i + 1, len(sl_vals)):
            if abs(sl_vals[i] - sl_vals[j]) <= price_tol:
                eql_price = (sl_vals[i] + sl_vals[j]) / 2.0
                pools.append({"type": "EQL", "price": eql_price, "side": "LOW", "desc": f"EQL (${eql_price:.5f})"})

    return pools


def is_near_macro_liquidity_pool(
    symbol: str,
    level: float,
    side: str,  # "LOW" for bullish sweep, "HIGH" for bearish sweep
    pools: list[dict],
    tolerance_pips: float = getattr(config, "MACRO_POOL_PIP_TOLERANCE", 5.0),
    current_atr: float = 0.0,
) -> tuple[bool, str]:
    """Check if a swept swing level aligns within tolerance of an existing macro liquidity pool."""
    pip_size = get_pip_size(symbol)
    price_tol = max(tolerance_pips * pip_size, 0.20 * current_atr if current_atr > 0 else 0.0)

    for p in pools:
        if p["side"] == side:
            dist = abs(level - p["price"])
            if dist <= price_tol:
                dist_pips = dist / pip_size
                return True, f"{p['desc']} within {dist_pips:.1f} pips"

    return False, "No aligning macro liquidity pool"


# ─── 5. Core SMC Setup Evaluators ─────────────────────────────────────────────

def _check_bullish_setup(
    symbol: str,
    closed_df: pd.DataFrame,
    current_price: float,
    current_atr: float,
    swing_len: int,
    macro_pools: list[dict],
) -> Optional[SMCSignal]:
    """Evaluate Bullish Institutional Setup across recent confirmed Swing Lows."""
    n_closed = len(closed_df)
    valid_sl_positions = np.where(~np.isnan(closed_df["swing_low"].values))[0]
    if len(valid_sl_positions) == 0:
        return None

    sl_buffer = getattr(config, "STRUCTURAL_SL_BUFFER_ATR", 0.35) * current_atr
    enforce_sessions = getattr(config, "ENFORCE_MACRO_SESSIONS", True)
    enforce_pools = getattr(config, "ENFORCE_MACRO_LIQUIDITY_FILTER", True)

    for sl_pos in reversed(valid_sl_positions[-4:]):
        sl_pos = int(sl_pos)
        recent_sl = float(closed_df["swing_low"].iloc[sl_pos])

        if ("SWEEP", symbol, "LONG", round(recent_sl, 5)) in _consumed_setups:
            continue

        # Filter 4: Macro Liquidity Pool Intersection (must align with PDL, Asia Low, or EQL)
        pool_desc = "Standard Swing Low"
        if enforce_pools and len(macro_pools) > 0:
            is_macro, p_desc = is_near_macro_liquidity_pool(
                symbol, recent_sl, side="LOW", pools=macro_pools, current_atr=current_atr
            )
            if not is_macro:
                continue
            pool_desc = p_desc

        scan_start = max(sl_pos + swing_len, n_closed - 30)
        if scan_start >= n_closed:
            continue

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

            if bar_close < recent_sl and (recent_sl - bar_close) > 0.25 * current_atr:
                break

            if bar_low < recent_sl and bar_close >= recent_sl:
                # Filter 1: Check macro session window of the sweep bar
                if enforce_sessions:
                    bar_time = _get_bar_timestamp(bar)
                    if bar_time and not is_in_macro_session(bar_time):
                        continue

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

        if ("OB", symbol, "LONG", round(ob_bottom, 5), round(ob_top, 5)) in _consumed_setups:
            continue

        ob_height = ob_top - ob_bottom
        if ob_height < 0.35 * current_atr:
            continue

        # Walk forward from displacement bar + 1 to end of closed_df
        invalidated_or_spent = False
        mitigated_previously = False
        for k in range(disp_idx + 1, n_closed):
            k_low = float(closed_df["low"].iloc[k])
            k_high = float(closed_df["high"].iloc[k])

            if k_low <= ob_bottom:
                invalidated_or_spent = True
                break

            if k_low <= ob_top:
                mitigated_previously = True
            if mitigated_previously and k_high >= ob_top + ob_height * config.RISK_REWARD_RATIO:
                invalidated_or_spent = True
                break

        if invalidated_or_spent:
            continue

        # Mitigation Entry Check on current live price:
        if ob_bottom < current_price <= ob_top:
            sl = ob_bottom - sl_buffer
            risk_dist = current_price - sl
            if risk_dist < config.MIN_STOP_ATR_RATIO * current_atr:
                sl = current_price - (config.MIN_STOP_ATR_RATIO * current_atr)
                risk_dist = current_price - sl

            tp = current_price + (risk_dist * config.RISK_REWARD_RATIO)
            log.info(
                f"🎯 [SMC BULLISH EDGE] {symbol} | Macro Pool: {pool_desc} | "
                f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                f"SL: ${sl:.5f} | TP: ${tp:.5f} | ATR: {current_atr:.5f}"
            )
            m_type = "SWING"
            for candidate in ("PDL", "ASIA_LOW", "EQL", "PDH", "ASIA_HIGH", "EQH"):
                if candidate in pool_desc.upper():
                    m_type = candidate
                    break

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
                macro_pool_desc=pool_desc,
                displacement_ratio=round(rejection_tail / current_atr, 2),
                macro_pool_type=m_type,
            )

    return None


def _check_bearish_setup(
    symbol: str,
    closed_df: pd.DataFrame,
    current_price: float,
    current_atr: float,
    swing_len: int,
    macro_pools: list[dict],
) -> Optional[SMCSignal]:
    """Evaluate Bearish Institutional Setup across recent confirmed Swing Highs."""
    n_closed = len(closed_df)
    valid_sh_positions = np.where(~np.isnan(closed_df["swing_high"].values))[0]
    if len(valid_sh_positions) == 0:
        return None

    sl_buffer = getattr(config, "STRUCTURAL_SL_BUFFER_ATR", 0.35) * current_atr
    enforce_sessions = getattr(config, "ENFORCE_MACRO_SESSIONS", True)
    enforce_pools = getattr(config, "ENFORCE_MACRO_LIQUIDITY_FILTER", True)

    for sh_pos in reversed(valid_sh_positions[-4:]):
        sh_pos = int(sh_pos)
        recent_sh = float(closed_df["swing_high"].iloc[sh_pos])

        if ("SWEEP", symbol, "SHORT", round(recent_sh, 5)) in _consumed_setups:
            continue

        # Filter 4: Macro Liquidity Pool Intersection (must align with PDH, Asia High, or EQH)
        pool_desc = "Standard Swing High"
        if enforce_pools and len(macro_pools) > 0:
            is_macro, p_desc = is_near_macro_liquidity_pool(
                symbol, recent_sh, side="HIGH", pools=macro_pools, current_atr=current_atr
            )
            if not is_macro:
                continue
            pool_desc = p_desc

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
                # Filter 1: Check macro session window of the sweep bar
                if enforce_sessions:
                    bar_time = _get_bar_timestamp(bar)
                    if bar_time and not is_in_macro_session(bar_time):
                        continue

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

        if ("OB", symbol, "SHORT", round(ob_bottom, 5), round(ob_top, 5)) in _consumed_setups:
            continue

        ob_height = ob_top - ob_bottom
        if ob_height < 0.35 * current_atr:
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
            sl = ob_top + sl_buffer
            risk_dist = sl - current_price
            if risk_dist < config.MIN_STOP_ATR_RATIO * current_atr:
                sl = current_price + (config.MIN_STOP_ATR_RATIO * current_atr)
                risk_dist = sl - current_price

            tp = current_price - (risk_dist * config.RISK_REWARD_RATIO)
            log.info(
                f"🎯 [SMC BEARISH EDGE] {symbol} | Macro Pool: {pool_desc} | "
                f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Entry: ${current_price:.5f} | "
                f"SL: ${sl:.5f} | TP: ${tp:.5f} | ATR: {current_atr:.5f}"
            )
            m_type = "SWING"
            for candidate in ("PDH", "ASIA_HIGH", "EQH", "PDL", "ASIA_LOW", "EQL"):
                if candidate in pool_desc.upper():
                    m_type = candidate
                    break

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
                macro_pool_desc=pool_desc,
                displacement_ratio=round(rejection_tail / current_atr, 2),
                macro_pool_type=m_type,
            )

    return None


def detect_institutional_setup(
    symbol: str,
    df: pd.DataFrame,
    df_h4: Optional[pd.DataFrame] = None,
) -> Optional[SMCSignal]:
    """
    Evaluate structural price action with Macro Microstructure constraints:
    - Verifies current execution time is within London/NY macro sessions.
    - Filters liquidity sweeps to intersections with PDH/PDL, Asia H/L, or EQH/EQL pools.
    - Evaluates unmitigated, non-invalidated Order Blocks.
    - Strictly enforces H4 dealing range equilibrium (Longs in Discount, Shorts in Premium).
    """
    min_bars = config.SWING_LENGTH * 3 + 15
    if df is None or len(df) < min_bars:
        return None

    # Filter 1: Macro Session Gate (current evaluation time)
    if getattr(config, "ENFORCE_MACRO_SESSIONS", True):
        last_bar_time = _get_bar_timestamp(df.iloc[-1])
        if not is_in_macro_session(last_bar_time):
            return None

    # Live price from actively forming candle
    current_price = float(df["close"].iloc[-1])

    # Closed candles strictly for structural swings, sweeps, ATR, and macro pools
    closed_df = df.iloc[:-1].copy().reset_index(drop=True)
    closed_df["atr"] = calculate_atr(closed_df, config.ATR_PERIOD)
    current_atr = float(closed_df["atr"].iloc[-1])
    if current_atr <= 0 or np.isnan(current_atr):
        return None

    closed_swings = identify_swings(closed_df, config.SWING_LENGTH)
    macro_pools = extract_macro_liquidity_pools(symbol, closed_swings, current_atr)

    # Calculate H4 dealing range depth if H4 data available
    h4_depth = 0.50
    if df_h4 is not None:
        h4_h, h4_l, _ = calculate_h4_dealing_range(df_h4, getattr(config, "HTF_SWING_LENGTH", 3))
        if h4_h and h4_l and h4_h > h4_l:
            h4_depth = round(min(1.0, max(0.0, (current_price - h4_l) / (h4_h - h4_l))), 4)

    # 1. Check Bullish Setup
    bull_sig = _check_bullish_setup(symbol, closed_swings, current_price, current_atr, config.SWING_LENGTH, macro_pools)
    if bull_sig is not None:
        # Filter 2: Enforce H4 Discount Matrix for Longs
        htf_ok, htf_msg, eq = check_htf_premium_discount("LONG", bull_sig.entry_price, df_h4)
        if not htf_ok:
            log.info(f"🚫 [HTF MATRIX REJECTION] {symbol} | {htf_msg}")
            return None
        bull_sig.htf_equilibrium = eq
        bull_sig.htf_depth_pct = h4_depth
        return bull_sig

    # 2. Check Bearish Setup
    bear_sig = _check_bearish_setup(symbol, closed_swings, current_price, current_atr, config.SWING_LENGTH, macro_pools)
    if bear_sig is not None:
        # Filter 2: Enforce H4 Premium Matrix for Shorts
        htf_ok, htf_msg, eq = check_htf_premium_discount("SHORT", bear_sig.entry_price, df_h4)
        if not htf_ok:
            log.info(f"🚫 [HTF MATRIX REJECTION] {symbol} | {htf_msg}")
            return None
        bear_sig.htf_equilibrium = eq
        bear_sig.htf_depth_pct = h4_depth
        return bear_sig

    return None
