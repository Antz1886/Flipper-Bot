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
CANDLES_TO_FETCH = 100000  # 100k per symbol for better coverage
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
    fvg_data = smc.fvg(df, join_consecutive=False)
    swings = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    ob_data = smc.ob(df, swing_highs_lows=swings)
    
    obs = ob_data[ob_data["OB"].notna() & (ob_data["OB"] != 0)].copy()
    
    setups = []
    wins = 0
    losses = 0
    
    for idx, ob_row in obs.iterrows():
        direction = "BUY" if ob_row["OB"] == 1 else "SELL"
        ob_top, ob_bottom = ob_row["Top"], ob_row["Bottom"]
        
        recent_fvg = fvg_data.iloc[max(0, idx-5):idx+5]
        if not (recent_fvg["FVG"] == (1 if direction == "BUY" else -1)).any(): continue
            
        if direction == "BUY":
            entry = ob_top
            sl = ob_bottom - (ob_top - ob_bottom) * OB_BUFFER_PCT
            tp = _compute_tp(entry, sl, "BUY")
        else:
            entry = ob_bottom
            sl = ob_top + (ob_top - ob_bottom) * OB_BUFFER_PCT
            tp = _compute_tp(entry, sl, "SELL")
            
        future = df.iloc[idx + 2:]
        if future.empty: continue
        
        # Entry
        mask = future["low"] <= entry if direction == "BUY" else future["high"] >= entry
        if not mask.any(): continue
        entry_idx = mask.idxmax()
        
        # SL before entry
        pre_entry = future.loc[:entry_idx]
        if (pre_entry["low"] <= sl if direction == "BUY" else pre_entry["high"] >= sl).any(): continue
            
        # Outcome
        post_entry = df.iloc[entry_idx:]
        sl_hit = post_entry["low"] <= sl if direction == "BUY" else post_entry["high"] >= sl
        tp_hit = post_entry["high"] >= tp if direction == "BUY" else post_entry["low"] <= tp
        
        first_sl = sl_hit.idxmax() if sl_hit.any() else None
        first_tp = tp_hit.idxmax() if tp_hit.any() else None
        
        outcome = None
        if first_sl is not None and first_tp is not None:
            outcome = 1 if first_tp < first_sl else 0
        elif first_tp is not None: outcome = 1
        elif first_sl is not None: outcome = 0
            
        if outcome is not None:
            feat = df.iloc[idx]
            price = feat["close"]
            
            # Skip if price is zero (data error)
            if price == 0:
                continue
            
            ema_val = feat["EMA_50"]
            
            # Build expanded feature set
            setup = {
                "symbol": symbol,
                "direction": direction,
                # NORMALIZED FEATURES (Percentage of price)
                "ob_size_pct": abs(ob_top - ob_bottom) / price,
                "atr_pct": feat["ATR"] / price if pd.notna(feat["ATR"]) else 0,
                "rsi": feat["RSI"] if pd.notna(feat["RSI"]) else 50.0,
                "price_vs_ema": (price - ema_val) / ema_val if pd.notna(ema_val) and ema_val != 0 else 0,
                # NEW: Candle structure features
                "candle_body_pct": abs(feat["close"] - feat["open"]) / price,
                "upper_wick_pct": (feat["high"] - max(feat["open"], feat["close"])) / price,
                "lower_wick_pct": (min(feat["open"], feat["close"]) - feat["low"]) / price,
                "distance_to_ob_pct": abs(price - entry) / price,
                "hour_of_day": feat["time"].hour if hasattr(feat["time"], "hour") else 0,
                "label": outcome
            }
            setups.append(setup)
            
            if outcome == 1:
                wins += 1
            else:
                losses += 1
    
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
