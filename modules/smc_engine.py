"""
modules/smc_engine.py — Smart Money Concepts Alpha Engine
Detects Fair Value Gaps (FVG) and Order Blocks (OB) using the
smartmoneyconcepts library. Only emits a TradeSignal when both
indicators align in the same direction on the primary timeframe,
confirmed by a matching FVG on the confluence (M15) timeframe.

v2.0 — Updated AI feature vector with candle structure features,
       configurable veto threshold, and graceful model fallback.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pandas_ta as ta
import os
import joblib

from config import (
    RISK_REWARD_RATIO, HTF_EMA_PERIOD, MIN_ADX_TREND,
    FAST_EMA_PERIOD, SLOW_EMA_PERIOD, RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    ATR_MULTIPLIER, USE_AI_FILTER, AI_VETO_THRESHOLD
)

log = logging.getLogger(__name__)

# ─── AI Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/rf_setup_scorer.pkl"
rf_model = None
MODEL_FEATURES = None  # Will be set after loading model

try:
    if os.path.exists(MODEL_PATH):
        rf_model = joblib.load(MODEL_PATH)
        # Try to detect which features the model expects
        try:
            MODEL_FEATURES = rf_model.feature_names_in_.tolist()
            log.info(f"Loaded AI model with features: {MODEL_FEATURES}")
        except AttributeError:
            MODEL_FEATURES = None
            log.info(f"Loaded AI model (feature names not embedded)")
        log.info(f"Loaded AI Setup Scorer model from {MODEL_PATH}")
    else:
        log.warning("AI model not found. Running in pure SMC mode.")
except Exception as e:
    log.error(f"Failed to load AI model: {e}")

# Feature columns for the new expanded model
EXPANDED_FEATURES = [
    "direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema",
    "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
    "distance_to_ob_pct", "hour_of_day"
]

# Legacy features for backward compatibility with old model
LEGACY_FEATURES = [
    "direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema"
]

# Cache of last logged vetoes per symbol to prevent log spam
_last_logged_vetoes = {}



# ─── Signal Dataclass ─────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    symbol:       str
    direction:    str    # "BUY" or "SELL"
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    signal_type:  str    # always "LIMIT" for OB-based entries
    ob_top:       float
    ob_bottom:    float
    fvg_top:      float
    fvg_bottom:   float
    confidence:   float  # 0.0–1.0  (0.7 base + 0.3 if confluence TF confirms)

    def risk_pips(self, point: float) -> float:
        return abs(self.entry_price - self.stop_loss) / point

    def reward_pips(self, point: float) -> float:
        return abs(self.take_profit - self.entry_price) / point


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _detect_fvg(
    ohlc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (bull_fvg, bear_fvg) DataFrames of unmitigated Fair Value Gaps.
    Unmitigated = MitigatedIndex == 0 (price has not returned to fill the gap).
    """
    try:
        result = smc.fvg(ohlc, join_consecutive=False)
        if result is None or result.empty:
            return pd.DataFrame(), pd.DataFrame()

        unmitigated = result[result["MitigatedIndex"] == 0]
        bull = unmitigated[unmitigated["FVG"] == 1]   # 1  = bullish FVG
        bear = unmitigated[unmitigated["FVG"] == -1]  # -1 = bearish FVG
        return bull, bear

    except Exception as exc:
        log.warning(f"FVG detection raised: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def _detect_ob(
    ohlc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (bull_ob, bear_ob) DataFrames of unmitigated Order Blocks.
    Uses swing_highs_lows as the prerequisite for ob().
    """
    try:
        swings  = smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)
        ob_data = smc.ob(ohlc, swing_highs_lows=swings)

        if ob_data is None or ob_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        unmitigated = ob_data[ob_data["MitigatedIndex"] == 0]
        bull = unmitigated[unmitigated["OB"] == 1]   # 1  = bullish OB
        bear = unmitigated[unmitigated["OB"] == -1]  # -1 = bearish OB
        return bull, bear

    except Exception as exc:
        log.warning(f"OB detection raised: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def _compute_tp(entry: float, sl: float, direction: str) -> float:
    """Project Take Profit using the configured Risk:Reward ratio."""
    risk = abs(entry - sl)
    return (entry + risk * RISK_REWARD_RATIO) if direction == "BUY" \
        else (entry - risk * RISK_REWARD_RATIO)


def _build_ai_features(
    direction: str,
    ob_top: float,
    ob_bottom: float,
    entry: float,
    current_close: float,
    df_ai: pd.DataFrame,
    latest_atr: float,
    latest_rsi: float,
    price_vs_ema: float,
) -> pd.DataFrame:
    """
    Build the feature vector for the AI model.
    Automatically uses expanded or legacy features based on what the model expects.
    """
    direction_encoded = 1 if direction == "BUY" else 0
    ob_size_pct = abs(ob_top - ob_bottom) / current_close if current_close else 0
    atr_pct = latest_atr / current_close if current_close else 0
    
    # Candle structure features
    last_open = float(df_ai["open"].iloc[-1])
    last_high = float(df_ai["high"].iloc[-1])
    last_low = float(df_ai["low"].iloc[-1])
    
    candle_body_pct = abs(current_close - last_open) / current_close if current_close else 0
    upper_wick_pct = (last_high - max(last_open, current_close)) / current_close if current_close else 0
    lower_wick_pct = (min(last_open, current_close) - last_low) / current_close if current_close else 0
    distance_to_ob_pct = abs(current_close - entry) / current_close if current_close else 0
    hour_of_day = datetime.now(timezone.utc).hour
    
    # Determine which feature set to use
    if MODEL_FEATURES is not None:
        feature_cols = MODEL_FEATURES
    elif rf_model is not None:
        # Try to infer from model's n_features_in_
        try:
            n_features = rf_model.n_features_in_
            feature_cols = EXPANDED_FEATURES if n_features == len(EXPANDED_FEATURES) else LEGACY_FEATURES
        except AttributeError:
            feature_cols = LEGACY_FEATURES
    else:
        feature_cols = LEGACY_FEATURES
    
    # Build values dict for all possible features
    all_values = {
        "direction_encoded": direction_encoded,
        "ob_size_pct": ob_size_pct,
        "atr_pct": atr_pct,
        "rsi": latest_rsi,
        "price_vs_ema": price_vs_ema,
        "candle_body_pct": candle_body_pct,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "distance_to_ob_pct": distance_to_ob_pct,
        "hour_of_day": hour_of_day,
    }
    
    # Build DataFrame with only the features the model expects
    feature_values = [[all_values.get(f, 0) for f in feature_cols]]
    return pd.DataFrame(feature_values, columns=feature_cols)


def _ai_evaluate(
    symbol: str,
    direction: str,
    features: pd.DataFrame,
    entry_price: float,
    is_duplicate: bool,
) -> tuple[float | None, bool]:
    """
    Evaluate the setup using the AI model.
    Returns (win_probability, approved).
    Returns (None, True) if no model is available (pure SMC mode).
    """
    if not USE_AI_FILTER or rf_model is None:
        return None, True
    
    try:
        prob_win = rf_model.predict_proba(features)[0][1]
        approved = prob_win >= AI_VETO_THRESHOLD
        
        dir_tag = "BULL" if direction == "BUY" else "BEAR"
        log_func = log.debug if is_duplicate else log.info
        
        if approved:
            log.info(f"✨ {symbol} [{dir_tag}] AI Approved Setup! Win Prob: {prob_win:.2%}")
        else:
            log_func(f"🚫 {symbol} [{dir_tag}] Vetoed by AI. Win Prob: {prob_win:.2%} < {AI_VETO_THRESHOLD:.0%}")
        
        return prob_win, approved
    except Exception as e:
        log.warning(f"AI evaluation failed for {symbol}: {e} — approving by default.")
        return None, True


# ─── Public API ───────────────────────────────────────────────────────────────

async def detect_signal(
    symbol:          str,
    primary_ohlc:    pd.DataFrame,
    confluence_ohlc: pd.DataFrame,
    htf_ohlc:        pd.DataFrame,
    current_bid:     float,
    current_ask:     float,
) -> Optional[TradeSignal]:
    """
    Scan for a trend-following pullback setup (EMA + RSI + ADX).
    """

    def _analyze() -> Optional[TradeSignal]:
        # ── High Timeframe Trend Filter (EMA 200 on H1) ──────────────────────
        macro_trend = None  # "BULL" or "BEAR" or None
        if htf_ohlc is not None and len(htf_ohlc) >= HTF_EMA_PERIOD:
            try:
                htf_df = htf_ohlc.copy()
                htf_df["EMA_HTF"] = ta.ema(htf_df["close"], length=HTF_EMA_PERIOD)
                latest_htf_ema = float(htf_df["EMA_HTF"].iloc[-1])
                latest_htf_close = float(htf_df["close"].iloc[-1])
                if pd.notna(latest_htf_ema) and latest_htf_ema > 0:
                    macro_trend = "BULL" if latest_htf_close > latest_htf_ema else "BEAR"
                    log.debug(f"{symbol} Macro Trend (H1): {macro_trend} | Close: {latest_htf_close:.4f} | EMA: {latest_htf_ema:.4f}")
            except Exception as e:
                log.warning(f"Failed to calculate HTF Trend for {symbol}: {e}")

        # ── Primary TF indicators ───────────────────────────────────────────
        df = primary_ohlc.copy()
        try:
            df["EMA_FAST"] = ta.ema(df["close"], length=FAST_EMA_PERIOD)
            df["EMA_SLOW"] = ta.ema(df["close"], length=SLOW_EMA_PERIOD)
            df["RSI"] = ta.rsi(df["close"], length=14)
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            
            latest_fast_ema = float(df["EMA_FAST"].iloc[-1])
            latest_slow_ema = float(df["EMA_SLOW"].iloc[-1])
            latest_rsi = float(df["RSI"].iloc[-1])
            latest_atr = float(df["ATR"].iloc[-1])
            latest_adx = float(adx_df["ADX_14"].iloc[-1])
        except Exception as e:
            log.warning(f"Failed to calculate primary indicators for {symbol}: {e}")
            return None

        # ── Trend Strength Filter (ADX) ──────────────────────────────────────
        if pd.notna(latest_adx):
            log.debug(f"{symbol} Primary ADX: {latest_adx:.2f}")
            if latest_adx < MIN_ADX_TREND:
                log.debug(f"🚫 {symbol} Setup vetoed: Market ranging (ADX: {latest_adx:.1f} < {MIN_ADX_TREND})")
                return None

        current_close = float(df["close"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_high = float(df["high"].iloc[-1])

        # ── BULLISH Trend pullback setup (BUY) ──────────────────────────────────
        if latest_fast_ema > latest_slow_ema:
            if macro_trend == "BEAR":
                log.debug(f"🚫 {symbol} Bullish pullback setup vetoed by H1 Bearish Macro Trend.")
                return None

            # Pullback condition: low touches or goes below EMA 20, but close stays above EMA 50
            if current_low <= latest_fast_ema and current_close >= latest_slow_ema:
                if latest_rsi <= RSI_BUY_THRESHOLD:
                    entry = latest_fast_ema
                    sl = latest_slow_ema - (ATR_MULTIPLIER * latest_atr)
                    tp = _compute_tp(entry, sl, "BUY")
                    
                    sig = TradeSignal(
                        symbol=symbol, direction="BUY",
                        entry_price=round(entry, 6),
                        stop_loss=round(sl, 6),
                        take_profit=round(tp, 6),
                        signal_type="LIMIT",
                        ob_top=latest_fast_ema, ob_bottom=latest_slow_ema,
                        fvg_top=0.0, fvg_bottom=0.0,
                        confidence=1.0,
                    )
                    log.info(
                        f"[BULL PULLBACK] {symbol} | Entry {sig.entry_price} | "
                        f"SL {sig.stop_loss} | TP {sig.take_profit} | "
                        f"RSI {latest_rsi:.1f} | ADX {latest_adx:.1f}"
                    )
                    return sig

        # ── BEARISH Trend pullback setup (SELL) ──────────────────────────────────
        elif latest_fast_ema < latest_slow_ema:
            if macro_trend == "BULL":
                log.debug(f"🚫 {symbol} Bearish pullback setup vetoed by H1 Bullish Macro Trend.")
                return None

            # Pullback condition: high touches or goes above EMA 20, but close stays below EMA 50
            if current_high >= latest_fast_ema and current_close <= latest_slow_ema:
                if latest_rsi >= RSI_SELL_THRESHOLD:
                    entry = latest_fast_ema
                    sl = latest_slow_ema + (ATR_MULTIPLIER * latest_atr)
                    tp = _compute_tp(entry, sl, "SELL")
                    
                    sig = TradeSignal(
                        symbol=symbol, direction="SELL",
                        entry_price=round(entry, 6),
                        stop_loss=round(sl, 6),
                        take_profit=round(tp, 6),
                        signal_type="LIMIT",
                        ob_top=latest_slow_ema, ob_bottom=latest_fast_ema,
                        fvg_top=0.0, fvg_bottom=0.0,
                        confidence=1.0,
                    )
                    log.info(
                        f"[BEAR PULLBACK] {symbol} | Entry {sig.entry_price} | "
                        f"SL {sig.stop_loss} | TP {sig.take_profit} | "
                        f"RSI {latest_rsi:.1f} | ADX {latest_adx:.1f}"
                    )
                    return sig

        log.debug(f"No trend pullback signal | {symbol}")
        return None

    return await asyncio.to_thread(_analyze)



