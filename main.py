"""
main.py — Flipper-Bot Main Orchestrator (Deriv Forex, Gold, and Oil Edition)
Connects to Deriv API WebSocket, scans Forex, Gold, and Oil candles,
and executes Deriv Multiplier contracts with embedded SL/TP boundaries.
"""

import asyncio
import os
import sys
import logging
import time
import csv
from datetime import datetime

import config
from modules import connector
from modules.data_feed import fetch_ohlc
from modules.order_executor import place_multiplier_contract
from modules.risk_manager import check_account_health

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    ]
)
log = logging.getLogger("main")


def calculate_ema(series, period):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def log_trade(symbol, side, entry_price, stake, stop_loss_usd, take_profit_usd, contract_id="N/A"):
    """Log executed trade details to CSV journal."""
    try:
        os.makedirs("logs", exist_ok=True)
        file_exists = os.path.exists(config.TRADE_LOG_CSV)
        with open(config.TRADE_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "side", "entry_price", "stake", "stop_loss_usd", "take_profit_usd", "contract_id"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, entry_price, stake, stop_loss_usd, take_profit_usd, contract_id
            ])
    except Exception as e:
        log.error(f"Failed to log trade to CSV: {e}")


async def main():
    log.info("=" * 60)
    log.info("       FLIPPER-BOT (DERIV FOREX, GOLD & OIL EDITION)")
    log.info("=" * 60)
    log.info(f"   Symbols     : {', '.join(config.SYMBOLS)}")
    log.info(f"   Contract    : Deriv Multipliers (MULTUP / MULTDOWN)")
    log.info(f"   Stake Size  : ${config.TRADE_STAKE:.2f} USD")
    log.info(f"   SL / TP     : ${config.MULTIPLIER_SL_USD:.2f} / ${config.MULTIPLIER_TP_USD:.2f} USD")
    log.info(f"   Demo Mode   : {config.DEMO_MODE}")
    log.info("=" * 60)

    # 1. Connect and Authorize with Deriv WebSocket API
    success = await connector.initialize()
    if not success:
        log.critical("Failed to initialize Deriv API connection. Exiting.")
        return

    api = connector.get_api()

    # Dry Run check flag
    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Verification completed successfully. Exiting.")
        await connector.shutdown()
        return

    log.info("🚀 Starting main scanning loop...")

    try:
        while True:
            log.info("─── Starting Scan Cycle ───")

            # Check account balance & circuit breaker
            healthy = await check_account_health()
            if not healthy:
                log.warning("Account health check failed or circuit breaker hit. Pausing...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            for symbol in config.SYMBOLS:
                log.info(f"Scanning {symbol}...")

                # Fetch 1-Hour candles for Macro Trend Filter (200 EMA)
                df_h1 = await fetch_ohlc(symbol, granularity=3600, n_bars=210)
                if df_h1 is None or len(df_h1) < config.EMA_TREND_PERIOD:
                    log.warning(f"  Insufficient H1 candle data for {symbol} — skipping.")
                    continue

                df_h1["ema_trend"] = calculate_ema(df_h1["close"], config.EMA_TREND_PERIOD)
                h1_trend_ema = df_h1["ema_trend"].iloc[-1]
                current_price = df_h1["close"].iloc[-1]

                # Fetch 15-Minute candles for Entry Signals (20/50 EMA + RSI)
                df_m15 = await fetch_ohlc(symbol, granularity=900, n_bars=100)
                if df_m15 is None or len(df_m15) < config.EMA_SLOW_PERIOD:
                    log.warning(f"  Insufficient M15 candle data for {symbol} — skipping.")
                    continue

                df_m15["ema_fast"] = calculate_ema(df_m15["close"], config.EMA_FAST_PERIOD)
                df_m15["ema_slow"] = calculate_ema(df_m15["close"], config.EMA_SLOW_PERIOD)
                df_m15["rsi"] = calculate_rsi(df_m15["close"], config.RSI_PERIOD)

                m15_fast_ema = df_m15["ema_fast"].iloc[-1]
                m15_slow_ema = df_m15["ema_slow"].iloc[-1]
                m15_rsi = df_m15["rsi"].iloc[-1]

                is_bullish_trend = current_price > h1_trend_ema
                is_bearish_trend = current_price < h1_trend_ema

                log.info(f"  {symbol} Price: ${current_price:.4f} | H1 Trend EMA: ${h1_trend_ema:.4f}")
                log.info(f"  M15 Fast EMA: ${m15_fast_ema:.4f} | M15 Slow EMA: ${m15_slow_ema:.4f} | RSI: {m15_rsi:.2f}")

                # ─── LONG SETUP ──────────────────────────────────────────────────────
                if is_bullish_trend:
                    ema_aligned = m15_fast_ema > m15_slow_ema
                    price_in_pullback = current_price <= m15_fast_ema
                    rsi_oversold = m15_rsi < config.RSI_BUY_THRESHOLD

                    log.info(f"  Long Scan: EMA aligned: {ema_aligned} | Pullback: {price_in_pullback} | RSI oversold: {rsi_oversold}")

                    if ema_aligned and price_in_pullback and rsi_oversold:
                        log.info(f"🚀 [{symbol} LONG SIGNAL] Triggering Deriv MULTUP contract...")
                        res = await place_multiplier_contract(symbol, "LONG")
                        if res:
                            log_trade(symbol, "LONG", current_price, config.TRADE_STAKE, config.MULTIPLIER_SL_USD, config.MULTIPLIER_TP_USD, res.get("contract_id"))

                # ─── SHORT SETUP ─────────────────────────────────────────────────────
                elif is_bearish_trend:
                    ema_aligned = m15_fast_ema < m15_slow_ema
                    price_in_pullback = current_price >= m15_fast_ema
                    rsi_overbought = m15_rsi > config.RSI_SELL_THRESHOLD

                    log.info(f"  Short Scan: EMA aligned: {ema_aligned} | Pullback: {price_in_pullback} | RSI overbought: {rsi_overbought}")

                    if ema_aligned and price_in_pullback and rsi_overbought:
                        log.info(f"🚀 [{symbol} SHORT SIGNAL] Triggering Deriv MULTDOWN contract...")
                        res = await place_multiplier_contract(symbol, "SHORT")
                        if res:
                            log_trade(symbol, "SHORT", current_price, config.TRADE_STAKE, config.MULTIPLIER_SL_USD, config.MULTIPLIER_TP_USD, res.get("contract_id"))

            await asyncio.sleep(config.SCAN_INTERVAL_S)

    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
    except Exception as e:
        log.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        await connector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
