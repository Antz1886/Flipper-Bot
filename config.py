"""
config.py — Global Configuration (Conti Institutional SMC Framework)
Systematic Deriv Multipliers automated trading using Smart Money Concepts (SMC)
with single structural edge and hardcoded institutional risk mandates.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ───────────────────────────────────────────────────
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

DERIV_WS_URL    = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Symbol Selection & Account Mode ─────────────────────────────────────────
# Liquid Forex Pairs on Deriv
SYMBOLS         = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD"]
DEMO_MODE       = True   # Set to True for forward-testing validation

# ─── Core Institutional Edge: SMC Parameters ─────────────────────────────────
# 1. Structural Pivots
SWING_LENGTH    = 5      # Candles left & right to define a major swing high/low
# 2. Timeframe granularities
PRIMARY_TIMEFRAME_S = 900   # M15 execution timeframe (900 seconds)
N_BARS_INGEST       = 150   # Candles fetched for structural calculation
# 3. Minimum Risk-to-Reward Ratio
RISK_REWARD_RATIO   = 2.0   # Minimum 1:2 R:R projected target

# Multipliers available per symbol on Deriv
SYMBOL_MULTIPLIERS = {
    "frxEURUSD": 100,
    "frxGBPUSD": 100,
    "frxUSDJPY": 100,
    "frxAUDUSD": 100,
    "frxUSDCAD": 100,
}

# ─── Hardcoded Institutional Risk Mandates ────────────────────────────────────
RISK_PER_TRADE_PCT       = 0.02   # 2% of total equity risked per trade
MIN_STAKE_USD            = 1.00   # Deriv minimum multiplier stake ($1.00)
MAX_STAKE_USD            = 100.00 # Max stake ceiling per position
MAX_DAILY_DRAWDOWN_PCT   = 0.05   # 5% hard daily loss circuit breaker
CIRCUIT_BREAKER_USD      = 5.00   # Absolute account floor threshold

# ─── System & Execution Settings ──────────────────────────────────────────────
SCAN_INTERVAL_S = 15     # Scan cycle interval in seconds
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2

LOG_FILE        = "logs/bot.log"
TRADE_LOG_CSV   = "logs/trades.csv"
MAGIC_COMMENT   = "Conti-SMC-Institutional-v1"
