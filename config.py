"""
config.py — Global Configuration (Deriv Tick-Momentum edition)
Small Account Trading Bot | $10 → $100 | Tick Auto-Correlation
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ────────────────────────────────────────────────────
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

if not DERIV_API_TOKEN or DERIV_API_TOKEN == "your_api_token_here":
    raise ValueError(
        "CRITICAL: Deriv API token is missing or invalid! "
        "Please ensure your .env file is present and contains a valid DERIV_API_TOKEN."
    )

DERIV_REST_URL  = "https://api.derivws.com"
DERIV_WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Symbol & Mode ────────────────────────────────────────────────────────────
SYMBOLS         = ["R_100"]  # Re-targeted specifically to Volatility 100 Index
DEMO_MODE       = False      # Set to True for virtual account testing

# ─── Tick-Momentum Parameters ────────────────────────────────────────────────
TICK_MOMENTUM_COUNT = 3      # Number of consecutive ticks in the same direction
TICK_DURATION       = 5      # Rise/Fall contract duration in ticks
TRADE_STAKE         = 1.00   # Fixed stake size ($1.00 minimum enforced by Deriv)

# ─── Targets & Safeguards ─────────────────────────────────────────────────────
CIRCUIT_BREAKER_USD = 5.00   # Stop trading if balance falls below $5.00
ACCOUNT_TARGET_USD  = 100.00 # Target balance to stop trading

# ─── Trading Constraints ──────────────────────────────────────────────────────
MAX_CONCURRENT_TRADES = 1    # Limit to 1 open trade for the micro-account
TRADE_COOLDOWN_S      = 10   # Cooldown seconds after a trade closes before next trade

# ─── Loop Timing & Retries ───────────────────────────────────────────────────
HEALTH_CHECK_S   = 30.0      # Connection health-check interval (seconds)
MAX_RETRIES      = 3
RETRY_DELAY_S    = 2.0

# ─── Paths ────────────────────────────────────────────────────────────────────
LOG_FILE  = "logs/bot.log"
TRADE_CSV = "trades.csv"

# ─── Legacy compatibility parameters (unused in Tick-Momentum) ───────────────
RISK_REWARD_RATIO = 1.5
HTF_EMA_PERIOD = 200
MIN_ADX_TREND = 20.0
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50
RSI_BUY_THRESHOLD = 45
RSI_SELL_THRESHOLD = 55
ATR_MULTIPLIER = 1.5
USE_AI_FILTER = False
AI_VETO_THRESHOLD = 0.45
SYMBOL_MULTIPLIERS = {"R_100": 40, "R_75": 50}
MAGIC_COMMENT = "SMC_MICRO_BOT"

