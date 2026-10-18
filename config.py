"""
config.py — Global Configuration (Conti Institutional SMC Framework — Quant Edition)
Systematic Deriv Multipliers automated trading using Smart Money Concepts (SMC)
with true structural position sizing, equity-synced drawdown protection,
correlation limits, and ATR-validated Order Block mitigation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ───────────────────────────────────────────────────
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

DERIV_WS_URL    = os.getenv("DERIV_WS_URL", f"wss://red.derivws.com/websockets/v3?app_id={DERIV_APP_ID}")

# ─── Symbol Selection & Account Mode ─────────────────────────────────────────
# Liquid Forex Pairs on Deriv
SYMBOLS         = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD"]
DEMO_MODE       = True   # Set to True for forward-testing validation

# ─── Core Institutional Edge: SMC Parameters ─────────────────────────────────
# 1. Structural Pivots (Confirmed without lookahead bias)
SWING_LENGTH             = 5      # Candles left & right to confirm a swing high/low
# 2. Timeframe granularities
PRIMARY_TIMEFRAME_S      = 900    # M15 execution timeframe (900 seconds)
N_BARS_INGEST            = 150    # Candles fetched for structural calculation
# 3. Minimum Risk-to-Reward Ratio
RISK_REWARD_RATIO        = 2.0    # Minimum 1:2 R:R projected target
# 4. Microstructure & Displacement Filters
ATR_PERIOD               = 14     # ATR lookback for volatility scaling
MIN_STOP_ATR_RATIO       = 0.40   # Minimum OB/stop size (0.4x ATR) to overcome multiplier spread
MIN_DISPLACEMENT_ATR     = 0.60   # Impulse displacement body/range threshold relative to ATR

# Multipliers available per symbol on Deriv
SYMBOL_MULTIPLIERS = {
    "frxEURUSD": 100,
    "frxGBPUSD": 100,
    "frxUSDJPY": 100,
    "frxAUDUSD": 100,
    "frxUSDCAD": 100,
    "frxXAUUSD": 50,
    "frxXTIUSD": 50,
}

# ─── Hardcoded Institutional Risk Mandates ────────────────────────────────────
RISK_PER_TRADE_PCT       = 0.02   # 2% of total account equity risked at structural stop-loss
MAX_MARGIN_PER_TRADE_PCT = 0.25   # Max 25% of equity committed as stake margin on a single trade
MAX_CONCURRENT_TRADES    = 2      # Max 2 simultaneous open positions (caps correlated USD risk at 4%)
MIN_STAKE_USD            = 1.00   # Deriv minimum multiplier stake ($1.00)
MAX_STAKE_USD            = 500.00 # Absolute stake ceiling per position ($500.00)
MAX_DAILY_DRAWDOWN_PCT   = 0.05   # 5% hard daily equity drawdown circuit breaker
CIRCUIT_BREAKER_USD      = 5.00   # Absolute account floor threshold

# ─── System & Execution Settings ──────────────────────────────────────────────
SCAN_INTERVAL_S = 15     # Scan cycle interval in seconds
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2

LOG_FILE        = "logs/bot.log"
TRADE_LOG_CSV   = "logs/trades.csv"
MAGIC_COMMENT   = "Conti-SMC-Quant-v2"
