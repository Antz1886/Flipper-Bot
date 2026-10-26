"""
config.py — Global Configuration (Conti Institutional SMC Framework — Quant Edition)
Systematic Deriv Multipliers automated trading using Smart Money Concepts (SMC)
with true structural position sizing, equity-synced drawdown protection,
DXY correlation limits, macro session gating, HTF dealing range matrices,
sub-millisecond Deal Cancellation SLA monitoring, spread widening filters,
and Tier-1 macroeconomic data news embargos.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ───────────────────────────────────────────────────
# Flipper-Bot Registered Deriv Application ID (21-char NanoID for REST+OTP API)
DERIV_APP_ID        = os.getenv("DERIV_APP_ID", "33z3Jc7fSBXsnKQzT1tXm")
# Public v3 WebSocket App ID for unauthenticated/legacy v3 feeds
DERIV_PUBLIC_APP_ID = os.getenv("DERIV_PUBLIC_APP_ID", "1089")
DERIV_API_TOKEN     = os.getenv("DERIV_API_TOKEN", "").strip()

DERIV_REST_URL      = os.getenv("DERIV_REST_URL", "https://api.derivws.com")
DERIV_WS_URL        = os.getenv(
    "DERIV_WS_URL",
    f"wss://red.derivws.com/websockets/v3?app_id={DERIV_PUBLIC_APP_ID}",
)

# ─── Symbol Selection & Account Mode ─────────────────────────────────────────
# Liquid Forex Pairs on Deriv
SYMBOLS         = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD"]
DEMO_MODE       = True   # Set to True for forward-testing validation

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

# ─── 1. Macro Session Windows (GMT / UTC) ────────────────────────────────────
# London Open (07:00–11:00 GMT) and New York Open (13:00–17:00 GMT)
MACRO_SESSION_WINDOWS_UTC = [
    (7, 0, 11, 0),   # London Open: 07:00 to 11:00 UTC
    (13, 0, 17, 0),  # New York Open: 13:00 to 17:00 UTC
]
ENFORCE_MACRO_SESSIONS = True

# ─── 2. HTF Premium / Discount Matrices (H4 Dealing Range) ────────────────────
HTF_TIMEFRAME_S               = 14400  # H4 timeframe in seconds (4 * 3600)
HTF_SWING_LENGTH              = 3      # Swing length on H4 candles
N_HTF_BARS_INGEST             = 80     # Sufficient H4 bars to establish dealing range
ENFORCE_HTF_PREMIUM_DISCOUNT  = True

# ─── 3. Micro-Mechanic Fix 1 & 2: Deal Cancellation SLA & Premium Bleed ──────
USE_DEAL_CANCELLATION             = True
DEAL_CANCELLATION_DURATION        = "15m"   # Deriv Multiplier duration
# Minute 14, second 55 trigger: 895.0s evaluation (5.0s strict SLA before 900.0s expiry)
DEAL_CANCELLATION_SLA_BUFFER_S    = 5.0     # Dispatch cancellation at least 5s before expiration
DEAL_CANCELLATION_TRIGGER_S       = 895.0   # 14m 55s trigger for 15m window
# Premium Bleed Threshold: Cancellation fee must NOT exceed 15% of the 1% risk budget
MAX_CANCELLATION_FEE_PCT_OF_RISK  = 0.15

# ─── 4. Micro-Mechanic Fix 3: Execution Spread Widening Filter ───────────────
ENFORCE_SPREAD_FILTER             = True
MAX_ALLOWED_SPREAD_PIPS = {
    "frxEURUSD": 1.5,
    "frxGBPUSD": 2.0,
    "frxUSDJPY": 1.8,
    "frxAUDUSD": 1.8,
    "frxUSDCAD": 2.0,
    "DEFAULT": 2.0,
}

# ─── 5. Micro-Mechanic Fix 4: Tier-1 Macro News Embargo ──────────────────────
ENFORCE_NEWS_EMBARGO              = True
NEWS_EMBARGO_MINUTES_BEFORE       = 30      # 30 minutes before high-impact event
NEWS_EMBARGO_MINUTES_AFTER        = 30      # 30 minutes after high-impact event
ECONOMIC_CALENDAR_URL             = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CALENDAR_CACHE_FILE               = "logs/calendar_cache.json"
CALENDAR_SYNC_INTERVAL_S          = 3600    # Refresh economic calendar hourly

# ─── 6. Macro Liquidity Pool Filters (PDH/PDL, Asia H/L, EQH/EQL) ────────────
ENFORCE_MACRO_LIQUIDITY_FILTER = True
MACRO_POOL_PIP_TOLERANCE       = 5.0    # 3-5 pips tolerance around macro liquidity pool
ASIA_SESSION_START_UTC_HOUR    = 0      # Asian session: 00:00 UTC
ASIA_SESSION_END_UTC_HOUR      = 7      # Asian session: 07:00 UTC (Tokyo/Sydney close before London)

# ─── 7. DXY Correlation & Portfolio Concurrency Trap ─────────────────────────
BASE_USD_PAIRS             = ["frxUSDJPY", "frxUSDCAD"]
QUOTE_USD_PAIRS            = ["frxEURUSD", "frxGBPUSD", "frxAUDUSD"]
MAX_BASE_USD_TRADES        = 1     # At most 1 base-USD pair active
MAX_QUOTE_USD_TRADES       = 1     # At most 1 quote-USD pair active
ENFORCE_DXY_DIRECTION_LOCK = True  # Disallows stacking same directional DXY risk
MAX_CONCURRENT_TRADES      = 2     # Global cap: 1 Base + 1 Quote USD trade

# ─── Core Institutional Edge: SMC Parameters ─────────────────────────────────
SWING_LENGTH             = 5      # Candles left & right to confirm a swing high/low
PRIMARY_TIMEFRAME_S      = 900    # M15 execution timeframe (900 seconds)
N_BARS_INGEST            = 150    # Candles fetched for structural calculation
RISK_REWARD_RATIO        = 2.0    # Minimum 1:2 R:R projected target
ATR_PERIOD               = 14     # ATR lookback for volatility scaling
MIN_STOP_ATR_RATIO       = 1.20   # Minimum stop distance (1.2x ATR) + structural buffer to beat 100x spread
STRUCTURAL_SL_BUFFER_ATR = 0.35   # Extra ATR buffer beyond OB extreme to prevent wick/spread stop-outs
MIN_DISPLACEMENT_ATR     = 0.80   # Impulse displacement body/range threshold relative to ATR
SYMBOL_COOLDOWN_S        = 900    # Minimum 15m (1 M15 bar) cooldown between entries on the same symbol

# ─── Hardcoded Institutional Risk Mandates ────────────────────────────────────
RISK_PER_TRADE_PCT       = 0.01   # 1% of total account equity risked at structural stop-loss
MAX_MARGIN_PER_TRADE_PCT = 0.05   # Max 5% of equity committed as stake margin on a single trade
MIN_STAKE_USD            = 1.00   # Deriv minimum multiplier stake ($1.00)
MAX_STAKE_USD            = 150.00 # Conservative stake ceiling ($150.00) to prevent oversized spread drag
MAX_DAILY_DRAWDOWN_PCT   = 0.05   # 5% hard daily equity drawdown circuit breaker
CIRCUIT_BREAKER_USD      = 5.00   # Absolute account floor threshold

# ─── Virtual Account Sizing & Stake Overrides ─────────────────────────────────
# Set in .env to simulate a micro-account on a standard $10,000 demo (e.g. VIRTUAL_EQUITY=20.0, FIXED_STAKE_USD=2.0)
VIRTUAL_EQUITY     = float(os.getenv("VIRTUAL_EQUITY", "0.0") or "0.0")
FIXED_STAKE_USD    = float(os.getenv("FIXED_STAKE_USD", "0.0") or "0.0")

# ─── System & Execution Settings ──────────────────────────────────────────────
SCAN_INTERVAL_S = 15     # Scan cycle interval in seconds
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2

LOG_FILE        = "logs/bot.log"
TRADE_LOG_CSV   = "logs/trades.csv"
MAGIC_COMMENT   = "Conti-SMC-Quant-v2"
