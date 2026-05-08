import asyncio
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

from config import SYMBOL, PRIMARY_TF_S, SWING_LENGTH, RISK_REWARD_RATIO, OB_BUFFER_PCT
from modules.connector import initialize, shutdown, get_api
from modules.smc_engine import _compute_tp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_data_collector")

DATA_FILE = "ml_data/historical_setups.csv"
CANDLES_TO_FETCH = 100000  # Pull extensive history (approx 1 year of M5)

async def fetch_historical_data(api, symbol: str, granularity: int, total_candles: int) -> pd.DataFrame:
    """Fetch a large amount of historical candles by paginating backwards."""
    all_candles = []
    end_time = "latest"
    chunk_size = 5000  # Deriv API max
    
    chunks_needed = total_candles // chunk_size
    
    for i in range(chunks_needed):
        log.info(f"Fetching chunk {i+1}/{chunks_needed} (end_time={end_time})...")
        try:
            resp = await api.send({
                "ticks_history": symbol,
                "granularity": granularity,
                "count": chunk_size,
                "end": end_time,
                "style": "candles",
            })
            candles = resp.get("candles", [])
            if not candles:
                log.warning(f"No more candles available at chunk {i+1}")
                break
            
            all_candles.extend(candles)
            end_time = candles[0]["epoch"] - 1
            
            await asyncio.sleep(0.5) # Reduced sleep for faster fetching
        except Exception as e:
            log.error(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates(subset=["epoch"]).sort_values("epoch")
    
    df = df.rename(columns={"epoch": "time"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    if "volume" not in df.columns:
        df["volume"] = range(1, len(df) + 1)
        
    df = df.reset_index(drop=True)
    log.info(f"Total candles fetched: {len(df)}")
    return df

def simulate_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized vectorized forward simulation of SMC setups.
    """
    log.info("Calculating SMC indicators and TA features...")
    
    # Calculate TA features
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    
    # Calculate SMC
    from smartmoneyconcepts import smc
    fvg_data = smc.fvg(df, join_consecutive=False)
    swings = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    ob_data = smc.ob(df, swing_highs_lows=swings)
    
    # Filter for valid Order Blocks (ignore NaNs and zeros)
    obs = ob_data[ob_data["OB"].notna() & (ob_data["OB"] != 0)].copy()
    
    log.info(f"Found {len(obs)} potential Order Blocks. Vectorized simulation in progress...")
    
    setups = []
    
    for idx, ob_row in obs.iterrows():
        direction = "BUY" if ob_row["OB"] == 1 else "SELL"
        ob_top = ob_row["Top"]
        ob_bottom = ob_row["Bottom"]
        
        # 1. FVG Confluence Check
        recent_fvg = fvg_data.iloc[max(0, idx-5):idx+5]
        has_fvg = (recent_fvg["FVG"] == (1 if direction == "BUY" else -1)).any()
        if not has_fvg: continue
            
        # 2. Setup Parameters
        if direction == "BUY":
            entry = ob_top
            sl = ob_bottom - (ob_top - ob_bottom) * OB_BUFFER_PCT
            tp = _compute_tp(entry, sl, "BUY")
        else:
            entry = ob_bottom
            sl = ob_top + (ob_top - ob_bottom) * OB_BUFFER_PCT
            tp = _compute_tp(entry, sl, "SELL")
            
        # 3. Vectorized Forward Search
        # Slice from 'formation + buffer' to end
        future = df.iloc[idx + 2:]
        if future.empty: continue
        
        # A. Find Entry
        if direction == "BUY":
            entry_mask = future["low"] <= entry
        else:
            entry_mask = future["high"] >= entry
            
        if not entry_mask.any(): continue
        entry_idx = entry_mask.idxmax()
        
        # B. Check for invalidation (SL hit before entry)
        pre_entry = future.loc[:entry_idx]
        if direction == "BUY":
            if (pre_entry["low"] <= sl).any(): continue
        else:
            if (pre_entry["high"] >= sl).any(): continue
            
        # C. Outcome Search
        post_entry = df.iloc[entry_idx:]
        if direction == "BUY":
            sl_hit_mask = post_entry["low"] <= sl
            tp_hit_mask = post_entry["high"] >= tp
        else:
            sl_hit_mask = post_entry["high"] >= sl
            tp_hit_mask = post_entry["low"] <= tp
            
        first_sl = sl_hit_mask.idxmax() if sl_hit_mask.any() else None
        first_tp = tp_hit_mask.idxmax() if tp_hit_mask.any() else None
        
        outcome = None
        if first_sl is not None and first_tp is not None:
            outcome = 1 if first_tp < first_sl else 0
        elif first_tp is not None:
            outcome = 1
        elif first_sl is not None:
            outcome = 0
            
        if outcome is not None:
            features = df.iloc[idx]
            setups.append({
                "time": features["time"],
                "direction": direction,
                "ob_size": abs(ob_top - ob_bottom),
                "atr": features["ATR"],
                "rsi": features["RSI"],
                "price_vs_ema": (features["close"] - features["EMA_50"]) / features["EMA_50"] if features["EMA_50"] else 0,
                "label": outcome
            })
            
    return pd.DataFrame(setups)

async def main():
    import os
    os.makedirs("ml_data", exist_ok=True)
    
    if not await initialize():
        log.error("Failed to connect to Deriv")
        return
        
    api = get_api()
    df = await fetch_historical_data(api, SYMBOL, PRIMARY_TF_S, CANDLES_TO_FETCH)
    await shutdown()
    
    if df is None or df.empty:
        log.error("No data fetched.")
        return
        
    setups_df = simulate_setups(df)
    
    if setups_df.empty:
        log.warning("No setups found during simulation.")
        return
        
    win_rate = setups_df["label"].mean()
    log.info(f"Generated {len(setups_df)} labeled setups. Baseline Win Rate: {win_rate:.2%}")
    
    setups_df.to_csv(DATA_FILE, index=False)
    log.info(f"Extensive dataset saved to {DATA_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
