"""
config.py — Global Configuration (Deriv WebSocket API edition)
Small Account Trading Bot | $10 → $100 | SMC + Kelly Compounding

DEMO_MODE = True   ← set to False when switching to real $10 account
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ────────────────────────────────────────────────────
# Get token from: https://app.deriv.com/account/api-token  (Trade + Read scope)
# Register app at: https://api.deriv.com/
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")
DERIV_WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Symbol & Timeframe ───────────────────────────────────────────────────────
# Deriv internal symbol codes:
#   Volatility 75 Index → R_75
#   Volatility 100 Index → R_100
#   Crash 1000 Index → CRASH1000
SYMBOL          = os.getenv("SYMBOL", "R_75")
PRIMARY_TF_S    = 300    # M5  — primary signal timeframe (seconds)
CONFLUENCE_TF_S = 900    # M15 — confluence timeframe (seconds)
N_BARS          = 500    # Candles to fetch per OHLC request

# ─── Demo / Live Mode ────────────────────────────────────────────────────────
# Set DEMO_MODE = True while testing on a Deriv virtual account.
# All proportions are preserved — just scaled 1000x to match the demo balance.
DEMO_MODE = True

# ─── Risk & Compounding ───────────────────────────────────────────────────────
KELLY_FRACTION      = 0.22     # 22% of balance risked per trade (Full Kelly)
MIN_STAKE_USD       = 1.00    # Minimum stake enforced by Deriv
MAX_STAKE_USD       = 100.00  # Safety cap per trade (prevents runaway sizing on demo)
RISK_REWARD_RATIO   = 2.0     # Minimum R:R for take-profit amount calculation

# Live account targets ($10 real money)
_REAL_CIRCUIT_BREAKER = 4.00
_REAL_TARGET          = 100.00

# Demo account targets (proportionally scaled: demo starts at $10,000)
_DEMO_CIRCUIT_BREAKER = 4000.00
_DEMO_TARGET          = 15000.00   # Test a 50% gain on demo

CIRCUIT_BREAKER_USD = _DEMO_CIRCUIT_BREAKER if DEMO_MODE else _REAL_CIRCUIT_BREAKER
ACCOUNT_TARGET_USD  = _DEMO_TARGET          if DEMO_MODE else _REAL_TARGET

# ─── Multiplier Contract Settings ─────────────────────────────────────────────
# Available multipliers for R_75: 10, 20, 30, 50, 100
# Higher multiplier = more sensitive P&L per price move
MULTIPLIER = 50

# ─── SMC Engine ───────────────────────────────────────────────────────────────
SWING_LENGTH  = 10    # Swing lookback for Order Block detection
OB_BUFFER_PCT = 0.05  # 5% buffer beyond OB distal edge for price zone

# ─── Loop Timing ──────────────────────────────────────────────────────────────
SCAN_INTERVAL_S  = 30.0   # Seconds between SMC scan cycles
HEALTH_CHECK_S   = 30.0   # Connection health-check interval

# ─── Execution ────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2.0
MAGIC_COMMENT   = "SMC_MICRO_BOT"   # Stored in contract purchase_time comment

# ─── Paths ────────────────────────────────────────────────────────────────────
LOG_FILE  = "bot.log"
TRADE_CSV = "trades.csv"
