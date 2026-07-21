"""
config.py — Global Configuration (Deriv Forex, Gold, and Oil Edition)
Trading Forex (EUR/USD, GBP/USD), Gold (XAU/USD), and WTI Oil (XTI/USD) via Deriv API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Deriv API Credentials ───────────────────────────────────────────────────
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

DERIV_WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Symbol Selection & Account Mode ─────────────────────────────────────────
# Deriv symbols:
# frxEURUSD: EUR/USD Forex
# frxGBPUSD: GBP/USD Forex
# frxUSDJPY: USD/JPY Forex
# frxAUDUSD: AUD/USD Forex
# frxUSDCAD: USD/CAD Forex
SYMBOLS     = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD"]
DEMO_MODE   = True   # Set to True for virtual account testing

# ─── Strategy Indicator Parameters ──────────────────────────────────────────
EMA_TREND_PERIOD   = 200  # 1-Hour chart macro trend filter
EMA_FAST_PERIOD    = 20   # 15-Minute chart fast EMA
EMA_SLOW_PERIOD    = 50   # 15-Minute chart slow EMA

RSI_PERIOD         = 14
RSI_BUY_THRESHOLD  = 35   # Buy when RSI < 35 on M15
RSI_SELL_THRESHOLD = 65   # Sell when RSI > 65 on M15

# ─── Deriv Multiplier Contract Risk Management ──────────────────────────────
TRADE_STAKE        = 10.00  # Default stake size ($10.00)
MULTIPLIER_SL_USD  = 5.00   # Stop Loss in USD ($5.00 risk)
MULTIPLIER_TP_USD  = 10.00  # Take Profit in USD ($10.00 reward — 1:2 R:R)
CIRCUIT_BREAKER_USD= 1.00   # Minimum balance threshold before emergency stop

# Multiplier values per symbol family
SYMBOL_MULTIPLIERS = {
    "frxEURUSD": 100,
    "frxGBPUSD": 100,
    "frxXAUUSD": 50,
    "frxXTIUSD": 50,
}

# ─── System & Retry Settings ─────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY_S   = 2
SCAN_INTERVAL_S = 15

LOG_FILE        = "logs/bot.log"
TRADE_LOG_CSV   = "logs/trades.csv"
MAGIC_COMMENT   = "FlipperBot-v2"
