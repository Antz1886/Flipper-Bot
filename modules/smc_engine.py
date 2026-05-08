"""
modules/smc_engine.py — Smart Money Concepts Alpha Engine
Detects Fair Value Gaps (FVG) and Order Blocks (OB) using the
smartmoneyconcepts library. Only emits a TradeSignal when both
indicators align in the same direction on the primary timeframe,
confirmed by a matching FVG on the confluence (M15) timeframe.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pandas_ta as ta
from smartmoneyconcepts import smc
import os
import joblib

from config import SWING_LENGTH, RISK_REWARD_RATIO, OB_BUFFER_PCT

log = logging.getLogger(__name__)

# ─── AI Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/rf_setup_scorer.pkl"
rf_model = None
try:
    if os.path.exists(MODEL_PATH):
        rf_model = joblib.load(MODEL_PATH)
        log.info(f"Loaded AI Setup Scorer model from {MODEL_PATH}")
    else:
        log.warning("AI model not found. Running in pure SMC mode.")
except Exception as e:
    log.error(f"Failed to load AI model: {e}")

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
        if rf_model is not None:
            # We use a copy to avoid SettingWithCopyWarnings
            df_ai = primary_ohlc.copy()
            df_ai["ATR"] = ta.atr(df_ai["high"], df_ai["low"], df_ai["close"], length=14)
            df_ai["RSI"] = ta.rsi(df_ai["close"], length=14)
            df_ai["EMA_50"] = ta.ema(df_ai["close"], length=50)
            
            latest_atr = df_ai["ATR"].iloc[-1]
            latest_rsi = df_ai["RSI"].iloc[-1]
            latest_ema = df_ai["EMA_50"].iloc[-1]
            current_close = df_ai["close"].iloc[-1]
            
            if pd.notna(latest_ema) and latest_ema != 0:
                price_vs_ema = (current_close - latest_ema) / latest_ema

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
                
                if rf_model is not None:
                    # direction_encoded, ob_size_pct, atr_pct, rsi, price_vs_ema
                    features = pd.DataFrame([[1, abs(ob_top - ob_bottom) / current_close, latest_atr / current_close, latest_rsi, price_vs_ema]], 
                                            columns=["direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema"])
                    prob_win = rf_model.predict_proba(features)[0][1]
                    if prob_win < 0.60:
                        log.info(f"[BULL] Vetoed by AI. Win Prob: {prob_win:.2%} < 60%")
                        return None
                    conf = prob_win
                    log.info(f"✨ [BULL] AI Approved Setup! Win Prob: {prob_win:.2%}")
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
                
                if rf_model is not None:
                    # direction_encoded, ob_size_pct, atr_pct, rsi, price_vs_ema
                    features = pd.DataFrame([[0, abs(ob_top - ob_bottom) / current_close, latest_atr / current_close, latest_rsi, price_vs_ema]], 
                                            columns=["direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema"])
                    prob_win = rf_model.predict_proba(features)[0][1]
                    if prob_win < 0.60:
                        log.info(f"[BEAR] Vetoed by AI. Win Prob: {prob_win:.2%} < 60%")
                        return None
                    conf = prob_win
                    log.info(f"✨ [BEAR] AI Approved Setup! Win Prob: {prob_win:.2%}")
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
