import asyncio
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from datetime import datetime

from config import PRIMARY_TF_S, SWING_LENGTH, RISK_REWARD_RATIO, OB_BUFFER_PCT
from modules.connector import initialize, shutdown, get_api
from modules.smc_engine import _compute_tp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_data_collector")

DATA_FILE = "ml_data/historical_setups.csv"
CANDLES_TO_FETCH = 30000  # 30k per symbol for fast lookahead-bias-free simulation
SYMBOLS_TO_COLLECT = ["R_75", "R_100"]

async def fetch_historical_data(api, symbol: str, granularity: int, total_candles: int) -> pd.DataFrame:
    """Fetch a large amount of historical candles by paginating backwards."""
    all_candles = []
    end_time = "latest"
    chunk_size = 5000
    
    chunks_needed = total_candles // chunk_size
    
    for i in range(chunks_needed):
        log.info(f"[{symbol}] Fetching chunk {i+1}/{chunks_needed}...")
        try:
            resp = await api.send({
                "ticks_history": symbol,
                "granularity": granularity,
                "count": chunk_size,
                "end": end_time,
                "style": "candles",
            })
            candles = resp.get("candles", [])
            if not candles: break
            
            all_candles.extend(candles)
            end_time = candles[0]["epoch"] - 1
            await asyncio.sleep(0.5)
        except Exception as e:
            log.error(f"Error fetching {symbol}: {e}")
            break

    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["epoch"]).sort_values("epoch")
    df = df.rename(columns={"epoch": "time"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    if "volume" not in df.columns:
        df["volume"] = range(1, len(df) + 1)
        
    return df.reset_index(drop=True)

def simulate_setups(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    log.info(f"[{symbol}] Simulating setups on {len(df)} candles...")
    
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    
    from smartmoneyconcepts import smc
    
    setups = []
    wins = 0
    losses = 0
    
    active_zone = None
    
    t = 500
    while t < len(df):
        if active_zone is not None:
            direction, entry, sl, tp, armed_at, features = active_zone
            
            # Check TTL
            if t - armed_at > 48:
                active_zone = None
                t += 1
                continue
            
            candle = df.iloc[t]
            triggered = (direction == "BUY" and candle["low"] <= entry) or (direction == "SELL" and candle["high"] >= entry)
            
            if triggered:
                # Trade entered at candle t
                post_entry = df.iloc[t:]
                sl_hit = post_entry["low"] <= sl if direction == "BUY" else post_entry["high"] >= sl
                tp_hit = post_entry["high"] >= tp if direction == "BUY" else post_entry["low"] <= tp
                
                first_sl = sl_hit.idxmax() if sl_hit.any() else None
                first_tp = tp_hit.idxmax() if tp_hit.any() else None
                
                exit_idx = None
                outcome = None
                if first_sl is not None and first_tp is not None:
                    if first_tp < first_sl:
                        outcome = 1
                        exit_idx = first_tp
                    else:
                        outcome = 0
                        exit_idx = first_sl
                elif first_tp is not None:
                    outcome = 1
                    exit_idx = first_tp
                elif first_sl is not None:
                    outcome = 0
                    exit_idx = first_sl
                
                if outcome is not None and exit_idx is not None:
                    if outcome == 1: wins += 1
                    else: losses += 1
                    
                    # Store features and outcome
                    setup_row = {
                        "symbol": symbol,
                        "direction": direction,
                        "ob_size_pct": features["ob_size_pct"],
                        "atr_pct": features["atr_pct"],
                        "rsi": features["rsi"],
                        "price_vs_ema": features["price_vs_ema"],
                        "candle_body_pct": features["candle_body_pct"],
                        "upper_wick_pct": features["upper_wick_pct"],
                        "lower_wick_pct": features["lower_wick_pct"],
                        "distance_to_ob_pct": features["distance_to_ob_pct"],
                        "hour_of_day": features["hour_of_day"],
                        "label": outcome
                    }
                    setups.append(setup_row)
                    
                    active_zone = None
                    t = exit_idx + 1  # jump to after exit
                    continue
                else:
                    active_zone = None
                    t += 1
                    continue
            else:
                t += 1
                continue
                
        # If no active zone, check if we should scan at t
        slice_df = df.iloc[t-499:t+1].copy().reset_index(drop=True)
        try:
            swings = smc.swing_highs_lows(slice_df, swing_length=SWING_LENGTH)
            ob_data = smc.ob(slice_df, swing_highs_lows=swings)
            fvg_data = smc.fvg(slice_df, join_consecutive=False)
            
            unmitigated = ob_data[ob_data["MitigatedIndex"] == 0]
            if not unmitigated.empty:
                last_ob = unmitigated.iloc[-1]
                ob_val = last_ob["OB"]
                if ob_val != 0 and pd.notna(ob_val):
                    direction = "BUY" if ob_val == 1 else "SELL"
                    recent_fvg = fvg_data.iloc[-5:]
                    if (recent_fvg["FVG"] == (1 if direction == "BUY" else -1)).any():
                        ob_top, ob_bottom = last_ob["Top"], last_ob["Bottom"]
                        current_close = slice_df["close"].iloc[-1]
                        
                        if direction == "BUY" and current_close > ob_top:
                            entry = ob_top
                            sl = ob_bottom - (ob_top - ob_bottom) * OB_BUFFER_PCT
                            tp = _compute_tp(entry, sl, "BUY")
                        elif direction == "SELL" and current_close < ob_bottom:
                            entry = ob_bottom
                            sl = ob_top + (ob_top - ob_bottom) * OB_BUFFER_PCT
                            tp = _compute_tp(entry, sl, "SELL")
                        else:
                            t += 1
                            continue
                            
                        # Capture features at candle t (the current closed candle)
                        feat_row = df.iloc[t]
                        price = feat_row["close"]
                        ema_val = feat_row["EMA_50"]
                        
                        features = {
                            "ob_size_pct": abs(ob_top - ob_bottom) / price,
                            "atr_pct": feat_row["ATR"] / price if pd.notna(feat_row["ATR"]) else 0,
                            "rsi": feat_row["RSI"] if pd.notna(feat_row["RSI"]) else 50.0,
                            "price_vs_ema": (price - ema_val) / ema_val if pd.notna(ema_val) and ema_val != 0 else 0,
                            "candle_body_pct": abs(feat_row["close"] - feat_row["open"]) / price,
                            "upper_wick_pct": (feat_row["high"] - max(feat_row["open"], feat_row["close"])) / price,
                            "lower_wick_pct": (min(feat_row["open"], feat_row["close"]) - feat_row["low"]) / price,
                            "distance_to_ob_pct": abs(price - entry) / price,
                            "hour_of_day": feat_row["time"].hour if hasattr(feat_row["time"], "hour") else 0
                        }
                        
                        active_zone = (direction, entry, sl, tp, t, features)
            
        except Exception as e:
            pass
        t += 1
        
    total = wins + losses
    if total > 0:
        log.info(
            f"[{symbol}] Dataset: {total} setups | "
            f"Wins: {wins} ({wins/total:.1%}) | Losses: {losses} ({losses/total:.1%})"
        )
    else:
        log.warning(f"[{symbol}] No valid setups generated!")
        
    return pd.DataFrame(setups)

async def main():
    os.makedirs("ml_data", exist_ok=True)
    if not await initialize(): return
    api = get_api()
    
    all_setups = []
    for symbol in SYMBOLS_TO_COLLECT:
        if not await api.is_connected():
            log.info(f"Deriv API connection inactive. Reconnecting before fetching {symbol}...")
            await api.connect()
        df = await fetch_historical_data(api, symbol, PRIMARY_TF_S, CANDLES_TO_FETCH)
        if df is not None and not df.empty:
            setups_df = simulate_setups(df, symbol)
            all_setups.append(setups_df)
            log.info(f"[{symbol}] Generated {len(setups_df)} setups.")
            
    await shutdown()
    
    if not all_setups: return
    
    final_df = pd.concat(all_setups)
    
    # Data quality report
    log.info("=" * 60)
    log.info("DATA QUALITY REPORT")
    log.info(f"  Total setups: {len(final_df)}")
    log.info(f"  Win rate: {final_df['label'].mean():.1%}")
    log.info(f"  Per symbol:")
    for sym in final_df['symbol'].unique():
        sub = final_df[final_df['symbol'] == sym]
        log.info(f"    {sym}: {len(sub)} setups, win rate: {sub['label'].mean():.1%}")
    log.info(f"  Feature columns: {list(final_df.columns)}")
    
    win_rate = final_df['label'].mean()
    if win_rate > 0.85:
        log.warning(f"⚠️  Dataset is heavily skewed toward wins ({win_rate:.1%}). Model may overfit!")
    elif win_rate < 0.15:
        log.warning(f"⚠️  Dataset is heavily skewed toward losses ({win_rate:.1%}). Check R:R settings!")
    else:
        log.info(f"  ✅ Class balance looks reasonable ({win_rate:.1%} wins)")
    log.info("=" * 60)
    
    final_df.to_csv(DATA_FILE, index=False)
    log.info(f"Combined extensive dataset saved to {DATA_FILE}. Total setups: {len(final_df)}")

if __name__ == "__main__":
    asyncio.run(main())
