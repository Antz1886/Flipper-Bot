import os
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# ─── API Configuration ────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET        = True  # Always True for Testnet demo trading

# Endpoints
TESTNET_URL = "https://testnet.binancefuture.com"
PROD_URL    = "https://fapi.binance.com"
BASE_URL    = TESTNET_URL if USE_TESTNET else PROD_URL

# ─── Strategy Settings ────────────────────────────────────────────────────────
# Configure the 3 coins that allow low $2.00 trades and have good trend profiles
SYMBOLS_CONFIG = {
    "ETHUSDT": {"qty_precision": 3},  # Ethereum (Min: ~$1.90)
    "SOLUSDT": {"qty_precision": 2},  # Solana (Min: ~$1.40)
    "ADAUSDT": {"qty_precision": 0}   # Cardano (Min: ~$0.60)
}

LEVERAGE           = 5         # 5x Leverage
TRADE_STAKE_USDT   = 2.00      # Nominal trade size of $2.00 (Margin used is STAKE / LEVERAGE = $0.40 per coin)

# Risk Management
STOP_LOSS_PCT      = 1.5       # Stop Loss at 1.5% price distance
TAKE_PROFIT_PCT     = 3.0       # Take Profit at 3.0% price distance (1:2 Risk-to-Reward)

# Indicators
RSI_PERIOD         = 14
RSI_BUY_THRESHOLD  = 35        # Buy the dip when RSI < 35 on M15 chart
RSI_SELL_THRESHOLD = 65        # Sell the bounce when RSI > 65 on M15 chart

EMA_FAST_PERIOD    = 20
EMA_SLOW_PERIOD    = 50
EMA_TREND_PERIOD   = 200       # Trend filter on H1 chart

# ─── Execution Settings ──────────────────────────────────────────────────────
SCAN_INTERVAL_S    = 15        # Scan market every 15 seconds
LOG_FILE           = "logs/bot.log"
TRADE_LOG_CSV      = "logs/trades.csv"
CIRCUIT_BREAKER_USD = 1.00     # Stop trading if balance falls below $1.00
