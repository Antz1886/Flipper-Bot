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
SYMBOLS         = ["R_100", "R_75"]  # Trade both Volatility 100 and Volatility 75
DEMO_MODE       = False              # Set to True for virtual account testing

# ─── Tick-Momentum Parameters ────────────────────────────────────────────────
TICK_MOMENTUM_COUNT = 3      # Number of consecutive ticks in the same direction
TICK_DURATION       = 5      # Rise/Fall contract duration in ticks
TRADE_STAKE         = 1.00   # Fixed stake size ($1.00 minimum enforced by Deriv)

# ─── Multiplier Settings (for Tick-Momentum) ──────────────────────────────────
USE_MULTIPLIERS    = False       # Set to False to trade Rise/Fall contracts (zero commission)
MULTIPLIER_SL_USD  = 0.30        # Risk $0.30 per trade max
MULTIPLIER_TP_USD  = 0.90        # Target $0.90 profit per trade (1:3 Risk-to-Reward)

# ─── Targets & Safeguards ─────────────────────────────────────────────────────
CIRCUIT_BREAKER_USD = 1.00       # Stop trading if balance falls below $1.00 (cannot stake)
ACCOUNT_TARGET_USD  = 100.00     # Target balance to stop trading

# ─── Daily Limits (for capital protection) ────────────────────────────────────
DAILY_PROFIT_TARGET_USD = 3.00   # Stop trading for the day if profit reaches +$3.00
DAILY_STOP_LOSS_USD     = 2.00   # Stop trading for the day if loss reaches -$2.00

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
SYMBOL_MULTIPLIERS = {"R_100": 400, "R_75": 50}
MAGIC_COMMENT = "SMC_MICRO_BOT"

