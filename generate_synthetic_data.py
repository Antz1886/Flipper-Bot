import pandas as pd
import numpy as np
import os

# Generate a synthetic dataset of SMC setups for training the AI model
np.random.seed(42)
n_samples = 1000

# Features:
# direction: BUY or SELL
# ob_size: float (size of the order block)
# atr: float (average true range)
# rsi: float (relative strength index)
# price_vs_ema: float (distance from 50 EMA)

direction = np.random.choice(["BUY", "SELL"], size=n_samples)
ob_size = np.random.uniform(0.5, 5.0, size=n_samples)
atr = np.random.uniform(1.0, 10.0, size=n_samples)
rsi = np.random.uniform(20.0, 80.0, size=n_samples)
price_vs_ema = np.random.uniform(-0.02, 0.02, size=n_samples)

# Synthetic Label Generation (Adding some signal for the model to learn)
# E.g., BUYs with low RSI (oversold) and price > EMA are more likely to win
labels = []
for i in range(n_samples):
    prob_win = 0.5
    if direction[i] == "BUY":
        if rsi[i] < 40: prob_win += 0.2
        if price_vs_ema[i] > 0: prob_win += 0.1
        if ob_size[i] > 3.0: prob_win -= 0.15 # Too large OB = lower probability
    else:
        if rsi[i] > 60: prob_win += 0.2
        if price_vs_ema[i] < 0: prob_win += 0.1
        if ob_size[i] > 3.0: prob_win -= 0.15
        
    prob_win = max(0.1, min(0.9, prob_win))
    label = np.random.choice([0, 1], p=[1-prob_win, prob_win])
    labels.append(label)

df = pd.DataFrame({
    "time": pd.date_range(start="2024-01-01", periods=n_samples, freq="h"),
    "direction": direction,
    "ob_size": ob_size,
    "atr": atr,
    "rsi": rsi,
    "price_vs_ema": price_vs_ema,
    "label": labels
})

os.makedirs("ml_data", exist_ok=True)
df.to_csv("ml_data/historical_setups.csv", index=False)
print("Generated synthetic ml_data/historical_setups.csv with 1000 samples.")
