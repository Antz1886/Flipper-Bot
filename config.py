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

if not DERIV_API_TOKEN or DERIV_API_TOKEN == "your_api_token_here":
    raise ValueError(
        "CRITICAL: Deriv API token is missing or invalid! "
        "Please ensure your .env file is present and contains a valid DERIV_API_TOKEN."
    )

DERIV_REST_URL  = "https://api.derivws.com"
DERIV_WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Symbol & Timeframe ───────────────────────────────────────────────────────
# Deriv internal symbol codes:
#   Volatility 75 Index → R_75
#   Volatility 100 Index → R_100
#   Crash 1000 Index → CRASH1000
SYMBOLS         = ["R_75", "R_100"]
PRIMARY_TF_S    = 300    # M5  — primary signal timeframe (seconds)
CONFLUENCE_TF_S = 900    # M15 — confluence timeframe (seconds)
N_BARS          = 500    # Candles to fetch per OHLC request

# ─── Demo / Live Mode ────────────────────────────────────────────────────────
# Set DEMO_MODE = True while testing on a Deriv virtual account.
# All proportions are preserved — just scaled 1000x to match the demo balance.
DEMO_MODE = False

# ─── Risk & Compounding ───────────────────────────────────────────────────────
KELLY_FRACTION      = 0.08     # 8% of balance risked per trade (Fractional Kelly)
MAX_STAKE_PCT_OF_BALANCE = 0.50 # Protects account from total ruin, allows concurrent trades
MIN_STAKE_USD       = 1.00    # Minimum stake enforced by Deriv
MAX_STAKE_USD       = 100.00  # Safety cap per trade (prevents runaway sizing on demo)
RISK_REWARD_RATIO   = 1.5     # Optimized for high win rate and faster compounding

# Live account targets ($10 real money)
_REAL_CIRCUIT_BREAKER = 5.00
_REAL_TARGET          = 100.00

# Demo account targets (proportionally scaled: demo starts at $10,000)
_DEMO_CIRCUIT_BREAKER = 4000.00
_DEMO_TARGET          = 15000.00   # Test a 50% gain on demo

CIRCUIT_BREAKER_USD = _DEMO_CIRCUIT_BREAKER if DEMO_MODE else _REAL_CIRCUIT_BREAKER
ACCOUNT_TARGET_USD  = _DEMO_TARGET          if DEMO_MODE else _REAL_TARGET

# ─── Multiplier Contract Settings ─────────────────────────────────────────────
# These are DEFAULTS — the bot will dynamically discover valid multipliers
# at startup via the `contracts_for` API. If discovery fails, these are used.
# Available multipliers depend on the symbol:
#   R_75  : 10, 20, 30, 50, 100
#   R_100 : 40, 100, 200, 300, 400
SYMBOL_MULTIPLIERS = {
    "R_75":  50,
    "R_100": 40       # Changed from 100 → 40 as safe default
}

# ─── SMC Engine ───────────────────────────────────────────────────────────────
SWING_LENGTH  = 10    # Swing lookback for Order Block detection
OB_BUFFER_PCT = 0.05  # 5% buffer beyond OB distal edge for price zone

# ─── Zone & Trade Management ─────────────────────────────────────────────────
ZONE_TTL_S            = 4 * 3600   # 4 hours — max time a zone stays armed
TRADE_COOLDOWN_S      = 60         # Min seconds between trades on same symbol
MAX_CONCURRENT_TRADES = 2          # Max open contracts across ALL symbols

# ─── Loop Timing ──────────────────────────────────────────────────────────────
SCAN_INTERVAL_S  = 30.0   # Seconds between SMC scan cycles
HEALTH_CHECK_S   = 30.0   # Connection health-check interval

# ─── Execution ────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2.0
MAGIC_COMMENT   = "SMC_MICRO_BOT"   # Stored in contract purchase_time comment

# ─── Advanced Filters ─────────────────────────────────────────────────────────
HTF_TF_S = 3600             # 1 Hour macro trend timeframe (seconds)
HTF_EMA_PERIOD = 200        # Period for macro trend EMA
MIN_ADX_TREND = 20.0        # Minimum ADX to allow trading (M5 timeframe)
USE_REJECTION_FILTER = False # Disable tick-based rejection entry for EMA Pullback
REJECTION_TRIGGER_PCT = 0.15 

# ─── EMA-RSI Pullback Strategy Settings ───────────────────────────────────────
FAST_EMA_PERIOD = 20
SLOW_EMA_PERIOD = 50
RSI_BUY_THRESHOLD = 45
RSI_SELL_THRESHOLD = 55
ATR_MULTIPLIER = 1.5

# ─── AI Filter ────────────────────────────────────────────────────────────────
USE_AI_FILTER = False       # Disable AI veto for trend pullback strategy
AI_VETO_THRESHOLD = 0.45


# ─── Paths ────────────────────────────────────────────────────────────────────
LOG_FILE  = "logs/bot.log"
TRADE_CSV = "trades.csv"

# ─── Portfolio-Level Risk Controls ─────────────────────────────────────────────────────
MAX_TOTAL_RISK = 0.02          # 2% of equity across all open positions
MAX_OPEN_POSITIONS = 2
MAX_LONG_EXPOSURE = 0.02        # 2% of equity long side
MAX_SHORT_EXPOSURE = 0.02       # 2% of equity short side

# ─── Equity-Curve Protection ───────────────────────────────────────────────────────
EQ_CURVE_ROLLING_WINDOW = 50   # Number of recent trades to evaluate performance
EQ_CURVE_MIN_PF = 1.0          # Minimum profit factor before shutdown
EQ_CURVE_MIN_EXPECTANCY = 0.0   # Minimum expectancy before shutdown

# ─── Parameter Sensitivity Ranges ─────────────────────────────────────────────────
EMA_FAST_RANGE = (15, 25)
EMA_SLOW_RANGE = (40, 60)
RSI_RANGE = (35, 45)
ADX_RANGE = (20, 35)

# ─── Stress-Test Settings ───────────────────────────────────────────────────────
STRESS_SLIPPAGE_MULTIPLIER = 2.0
STRESS_EXECUTION_DELAY_S = 5
STRESS_MISSING_CANDLES = True
STRESS_API_DISCONNECT_PROB = 0.01
STRESS_SPREAD_WIDENING = 0.02

# ─── Trade-Score Buckets ───────────────────────────────────────────────────────
TRADE_SCORE_BUCKETS = {
    "high": (90, 100),
    "mid": (80, 90),
    "low": (70, 80)
}

# ─── Shadow-Mode Settings ─────────────────────────────────────────────────────
SHADOW_MODE = True   # Record live signals without order execution
SHADOW_DURATION_DAYS = 30

# ─── Institutional Validation Targets ─────────────────────────────────────────────
TARGET_PROFIT_FACTOR = 1.4
TARGET_SHARPE = 1.2
TARGET_MAX_DD = 0.15
TARGET_RECOVERY_FACTOR = 2.0
TARGET_MIN_TRADES = 1000

