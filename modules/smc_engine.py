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
from smartmoneyconcepts import smc
import os
import joblib

from config import SWING_LENGTH, RISK_REWARD_RATIO, OB_BUFFER_PCT, AI_VETO_THRESHOLD

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
    direction: str,
    features: pd.DataFrame,
) -> tuple[float | None, bool]:
    """
    Evaluate the setup using the AI model.
    Returns (win_probability, approved).
    Returns (None, True) if no model is available (pure SMC mode).
    """
    if rf_model is None:
        return None, True
    
    try:
        prob_win = rf_model.predict_proba(features)[0][1]
        approved = prob_win >= AI_VETO_THRESHOLD
        
        dir_tag = "BULL" if direction == "BUY" else "BEAR"
        if approved:
            log.info(f"✨ [{dir_tag}] AI Approved Setup! Win Prob: {prob_win:.2%}")
        else:
            log.info(f"🚫 [{dir_tag}] Vetoed by AI. Win Prob: {prob_win:.2%} < {AI_VETO_THRESHOLD:.0%}")
        
        return prob_win, approved
    except Exception as e:
        log.warning(f"AI evaluation failed: {e} — approving by default.")
        return None, True


# ─── Public API ───────────────────────────────────────────────────────────────

async def detect_signal(
    symbol:          str,
    primary_ohlc:    pd.DataFrame,
    confluence_ohlc: pd.DataFrame,
    current_bid:     float,
    current_ask:     float,
) -> Optional[TradeSignal]:
    """
    Scan for a high-confluence SMC trade setup.

    Confluence rule:
      - An unmitigated FVG **and** an unmitigated OB must both exist
        in the same direction on the PRIMARY timeframe (M5).
      - A matching unmitigated FVG on the CONFLUENCE timeframe (M15)
        adds 0.3 to the confidence score (max = 1.0).
      - Only one direction (bull or bear) is evaluated per cycle;
        the one with the freshest OB wins.

    Entry logic (pending LIMIT order):
      BUY  → limit at OB proximal (Top),  SL below OB distal (Bottom)
      SELL → limit at OB proximal (Bottom), SL above OB distal (Top)
    """

    def _analyze() -> Optional[TradeSignal]:

        # ── Primary TF signals ───────────────────────────────────────────────
        bull_fvg_p, bear_fvg_p = _detect_fvg(primary_ohlc)
        bull_ob_p,  bear_ob_p  = _detect_ob(primary_ohlc)

        # ── Confluence TF FVG (direction-aware) ──────────────────────────────
        bull_fvg_c, bear_fvg_c = _detect_fvg(confluence_ohlc)
        
        # ── Compute TA Features for AI ───────────────────────────────────────
        latest_atr, latest_rsi, price_vs_ema = 0.0, 0.0, 0.0
        df_ai = primary_ohlc.copy()
        
        df_ai["ATR"] = ta.atr(df_ai["high"], df_ai["low"], df_ai["close"], length=14)
        df_ai["RSI"] = ta.rsi(df_ai["close"], length=14)
        df_ai["EMA_50"] = ta.ema(df_ai["close"], length=50)
        
        latest_atr = df_ai["ATR"].iloc[-1]
        latest_rsi = df_ai["RSI"].iloc[-1]
        latest_ema = df_ai["EMA_50"].iloc[-1]
        current_close = float(df_ai["close"].iloc[-1])
        
        if pd.notna(latest_ema) and latest_ema != 0:
            price_vs_ema = (current_close - latest_ema) / latest_ema
        
        # Handle NaN values
        if pd.isna(latest_atr): latest_atr = 0.0
        if pd.isna(latest_rsi): latest_rsi = 50.0  # neutral default

        # ── BULLISH setup ─────────────────────────────────────────────────────
        if not bull_fvg_p.empty and not bull_ob_p.empty:
            fvg_row = bull_fvg_p.iloc[-1]
            ob_row  = bull_ob_p.iloc[-1]

            ob_top    = float(ob_row["Top"])
            ob_bottom = float(ob_row["Bottom"])

            # Proximal = Top of bullish OB (price retraces DOWN toward it)
            entry = ob_top
            # Distal = Bottom; SL placed a 5% buffer below the distal edge
            sl_buffer = (ob_top - ob_bottom) * OB_BUFFER_PCT
            sl = ob_bottom - sl_buffer
            tp = _compute_tp(entry, sl, "BUY")

            # Price must currently be above the OB (not yet tested)
            if current_bid > ob_top:
                conf = 0.7 + (0.3 if not bull_fvg_c.empty else 0.0)
                
                # AI evaluation
                features = _build_ai_features(
                    "BUY", ob_top, ob_bottom, entry, current_close,
                    df_ai, latest_atr, latest_rsi, price_vs_ema
                )
                prob_win, approved = _ai_evaluate("BUY", features)
                
                if not approved:
                    return None
                
                if prob_win is not None:
                    conf = prob_win
                
                sig = TradeSignal(
                    symbol=symbol, direction="BUY",
                    entry_price=round(entry, 6),
                    stop_loss=round(sl, 6),
                    take_profit=round(tp, 6),
                    signal_type="LIMIT",
                    ob_top=ob_top, ob_bottom=ob_bottom,
                    fvg_top=float(fvg_row["Top"]),
                    fvg_bottom=float(fvg_row["Bottom"]),
                    confidence=conf,
                )
                log.info(
                    f"[BULL] {symbol} | Entry {sig.entry_price} | "
                    f"SL {sig.stop_loss} | TP {sig.take_profit} | "
                    f"Conf {conf:.0%}"
                )
                return sig

        # ── BEARISH setup ─────────────────────────────────────────────────────
        if not bear_fvg_p.empty and not bear_ob_p.empty:
            fvg_row = bear_fvg_p.iloc[-1]
            ob_row  = bear_ob_p.iloc[-1]

            ob_top    = float(ob_row["Top"])
            ob_bottom = float(ob_row["Bottom"])

            # Proximal = Bottom of bearish OB (price retraces UP toward it)
            entry = ob_bottom
            sl_buffer = (ob_top - ob_bottom) * OB_BUFFER_PCT
            sl = ob_top + sl_buffer
            tp = _compute_tp(entry, sl, "SELL")

            if current_ask < ob_bottom:
                conf = 0.7 + (0.3 if not bear_fvg_c.empty else 0.0)
                
                # AI evaluation
                features = _build_ai_features(
                    "SELL", ob_top, ob_bottom, entry, current_close,
                    df_ai, latest_atr, latest_rsi, price_vs_ema
                )
                prob_win, approved = _ai_evaluate("SELL", features)
                
                if not approved:
                    return None
                
                if prob_win is not None:
                    conf = prob_win
                
                sig = TradeSignal(
                    symbol=symbol, direction="SELL",
                    entry_price=round(entry, 6),
                    stop_loss=round(sl, 6),
                    take_profit=round(tp, 6),
                    signal_type="LIMIT",
                    ob_top=ob_top, ob_bottom=ob_bottom,
                    fvg_top=float(fvg_row["Top"]),
                    fvg_bottom=float(fvg_row["Bottom"]),
                    confidence=conf,
                )
                log.info(
                    f"[BEAR] {symbol} | Entry {sig.entry_price} | "
                    f"SL {sig.stop_loss} | TP {sig.take_profit} | "
                    f"Conf {conf:.0%}"
                )
                return sig

        log.debug(f"No confluence signal | {symbol}")
        return None

    return await asyncio.to_thread(_analyze)
