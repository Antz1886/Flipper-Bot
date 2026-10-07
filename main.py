"""
main.py — Conti Institutional SMC Trading Framework (Deriv Multipliers)
Orchestrates:
  1. Data Ingestion: Clean M15 OHLC candle feeds via Deriv WebSocket API.
  2. Single Structural Edge: Liquidity Sweep -> MSS -> Order Block Mitigation.
  3. Hardcoded Risk Mandates: Fixed 2% risk sizing + 5% daily circuit breaker.
  4. Execution: Automated Deriv MULTUP / MULTDOWN routing with zero manual intervention.
"""

import asyncio
import os
import sys
import logging
import csv
from datetime import datetime

import config
from modules import connector
from modules.data_feed import fetch_ohlc
from modules.smc_engine import detect_institutional_setup
from modules.risk_manager import (
    check_account_health,
    calculate_order_risk,
    can_open_symbol,
    mark_symbol_active,
    mark_symbol_closed,
)
from modules.order_executor import place_multiplier_contract

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def log_trade(symbol, side, entry_price, stake, stop_loss_usd, take_profit_usd, contract_id="N/A", sweep_level=0.0):
    """Log executed trade details into the standardized trade journal."""
    try:
        os.makedirs("logs", exist_ok=True)
        file_exists = os.path.exists(config.TRADE_LOG_CSV)
        with open(config.TRADE_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "side", "entry_price", "stake",
                    "stop_loss_usd", "take_profit_usd", "contract_id", "sweep_level"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, entry_price, stake,
                stop_loss_usd, take_profit_usd, contract_id, sweep_level
            ])
    except Exception as e:
        log.error(f"Failed to log trade to CSV: {e}")


async def main():
    log.info("=" * 65)
    log.info("      CONTI INSTITUTIONAL SMC AUTOMATED TRADING BOT (DERIV)")
    log.info("=" * 65)
    log.info(f"   Symbols       : {', '.join(config.SYMBOLS)}")
    log.info(f"   Core Edge     : Liquidity Sweep + Order Block Mitigation (M15)")
    log.info(f"   Risk Mandate  : Fixed {config.RISK_PER_TRADE_PCT*100:.1f}% equity risk per trade")
    log.info(f"   Hard Circuit  : {config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}% max daily loss halt")
    log.info(f"   Target R:R    : 1:{config.RISK_REWARD_RATIO:.1f} minimum projected reward")
    log.info(f"   Demo Testing  : {config.DEMO_MODE}")
    log.info("=" * 65)

    # 1. Connect and Authorize with Deriv WebSocket API
    success = await connector.initialize()
    if not success:
        log.critical("Failed to initialize Deriv API connection. Exiting.")
        return

    api = connector.get_api()

    # Dry Run verification check
    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Architecture and connectivity verified. Exiting.")
        await connector.shutdown()
        return

    log.info("🚀 Starting institutional scanning and execution loop...")

    try:
        while True:
            log.info("─── Starting Institutional Scan Cycle ───")

            # 1. Check account balance & daily circuit breaker
            healthy = check_account_health()
            if not healthy:
                log.warning("Daily circuit breaker active or balance floor breached. Pausing cycle...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            current_balance = api.balance if (api and api.balance is not None) else 1000.0

            for symbol in config.SYMBOLS:
                # 2. Check position locks
                if not can_open_symbol(symbol):
                    log.debug(f"  {symbol} already has an active position open — skipping scan.")
                    continue

                log.info(f"Evaluating {symbol}...")

                # 3. Data Ingestion: Fetch M15 OHLC candles
                df_m15 = await fetch_ohlc(symbol, granularity=config.PRIMARY_TIMEFRAME_S, n_bars=config.N_BARS_INGEST)
                if df_m15 is None or len(df_m15) < 30:
                    log.warning(f"  Insufficient candle data for {symbol} — skipping.")
                    continue

                # 4. Signal Generation: Single structural SMC edge
                signal = detect_institutional_setup(symbol, df_m15)

                if signal:
                    # 5. Hardcoded Risk Calculation
                    stake_usd, sl_usd, tp_usd = calculate_order_risk(
                        signal.entry_price, signal.stop_loss, current_balance
                    )

                    log.info(
                        f"⚡ [{signal.direction} SIGNAL CONFIRMED] {symbol} | "
                        f"Entry: ${signal.entry_price:.5f} | SL: ${signal.stop_loss:.5f} | TP: ${signal.take_profit:.5f} | "
                        f"Equity Risk: ${sl_usd:.2f} ({(sl_usd/current_balance)*100:.1f}%) | "
                        f"Target Profit: ${tp_usd:.2f}"
                    )

                    # 6. Execute Deriv Multiplier order
                    mark_symbol_active(symbol)
                    order_res = await place_multiplier_contract(
                        symbol=symbol,
                        direction=signal.direction,
                        stake=stake_usd,
                        stop_loss_usd=sl_usd,
                        take_profit_usd=tp_usd,
                    )

                    if order_res:
                        log_trade(
                            symbol=symbol,
                            side=signal.direction,
                            entry_price=signal.entry_price,
                            stake=stake_usd,
                            stop_loss_usd=sl_usd,
                            take_profit_usd=tp_usd,
                            contract_id=order_res.get("contract_id", "N/A"),
                            sweep_level=signal.sweep_level,
                        )
                else:
                    last_price = float(df_m15["close"].iloc[-1])
                    log.info(f"  {symbol}: ${last_price:.5f} — No unmitigated liquidity sweep setup.")

            await asyncio.sleep(config.SCAN_INTERVAL_S)

    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
    except Exception as e:
        log.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        await connector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
