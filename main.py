"""
main.py — Conti Institutional SMC Trading Framework (Quant Edition)
Orchestrates:
  1. Auto-Reconnecting WebSocket Session: Automatically restores dropped connections every cycle.
  2. Live Portfolio Sync: Queries Deriv `portfolio` every cycle to track active stakes & unlock closed pairs.
  3. Hardcoded Risk Mandates: Equity-based 5% daily drawdown circuit breaker + max 2 concurrent USD trades.
  4. Data Ingestion: Clean M15 OHLC candle feeds via Deriv WebSocket API.
  5. Quantitative SMC Engine: Closed-bar Swing Liquidity Sweep -> ATR Displacement -> Unmitigated OB Entry.
  6. True Structural Sizing: Stake and dollar SL/TP mathematically bound to |entry - stop_loss| and multiplier.
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
from modules.smc_engine import detect_institutional_setup, mark_setup_consumed
from modules.risk_manager import (
    sync_open_positions,
    check_account_health,
    calculate_order_risk,
    can_open_symbol,
    get_risk_state,
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
    log.info("=" * 68)
    log.info("    CONTI INSTITUTIONAL SMC BOT — QUANT EDITION (DERIV MULTIPLIERS)")
    log.info("=" * 68)
    log.info(f"   Symbols          : {', '.join(config.SYMBOLS)}")
    log.info(f"   Core Edge        : Closed-Bar Sweep + ATR Displacement + OB Mitigation")
    log.info(f"   Risk Mandate     : {config.RISK_PER_TRADE_PCT*100:.1f}% Equity Risk at Structural SL")
    log.info(f"   Correlation Cap  : Max {config.MAX_CONCURRENT_TRADES} Concurrent USD Positions")
    log.info(f"   Hard Circuit     : {config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}% Max Daily Equity Drawdown")
    log.info(f"   Target R:R       : 1:{config.RISK_REWARD_RATIO:.1f} Structural Projection")
    log.info(f"   Demo Testing     : {config.DEMO_MODE}")
    log.info("=" * 68)

    # 1. Connect and Authorize with Deriv WebSocket API
    success = await connector.initialize()
    if not success:
        log.warning("Initial Deriv connection failed; scan loop will automatically retry every cycle.")

    # Dry Run verification check
    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Architecture and connectivity verified. Exiting.")
        await connector.shutdown()
        return

    log.info("🚀 Starting institutional scanning and execution loop...")

    try:
        while True:
            log.info("─── Starting Institutional Scan Cycle ───")

            # 0. Ensure WebSocket connection is active (auto-reconnect if dropped)
            if not await connector.ensure_connected():
                log.warning("Deriv WebSocket unavailable after reconnect attempts. Retrying next cycle...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            api = connector.get_api()

            # 1. Sync active contracts & stakes with Deriv exchange before evaluating health
            await sync_open_positions()

            # 2. Check account equity & daily circuit breaker
            healthy = check_account_health()
            if not healthy:
                log.warning("Daily circuit breaker active or equity floor breached. Pausing cycle...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            risk_state = get_risk_state()
            current_equity = (
                risk_state.effective_equity
                if (api and api.authorized and risk_state.effective_equity > 0)
                else 1000.0
            )

            for symbol in config.SYMBOLS:
                # 3. Check symbol lock and USD correlation concurrency cap
                if not can_open_symbol(symbol):
                    if symbol in risk_state.active_symbols:
                        log.info(f"  {symbol}: Active position open on exchange — skipping.")
                    else:
                        log.info(
                            f"  {symbol}: Max concurrent USD trades ({config.MAX_CONCURRENT_TRADES}) "
                            f"reached — skipping for correlation safety."
                        )
                    continue

                log.info(f"Evaluating {symbol}...")

                # 4. Data Ingestion: Fetch M15 OHLC candles
                df_m15 = await fetch_ohlc(symbol, granularity=config.PRIMARY_TIMEFRAME_S, n_bars=config.N_BARS_INGEST)
                if df_m15 is None or len(df_m15) < (config.SWING_LENGTH * 3 + 15):
                    log.warning(f"  Insufficient candle data for {symbol} — skipping.")
                    continue

                # 5. Signal Generation: Single structural SMC edge
                signal = detect_institutional_setup(symbol, df_m15)

                if signal:
                    # 6. Structural Multiplier Position Sizing
                    stake_usd, sl_usd, tp_usd = calculate_order_risk(
                        signal.entry_price,
                        signal.stop_loss,
                        current_equity,
                        symbol=symbol,
                    )

                    log.info(
                        f"⚡ [{signal.direction} SIGNAL CONFIRMED] {symbol} | "
                        f"Entry: ${signal.entry_price:.5f} | Structural SL: ${signal.stop_loss:.5f} | "
                        f"Structural TP: ${signal.take_profit:.5f} | Stake: ${stake_usd:.2f} | "
                        f"Dollar Risk: ${sl_usd:.2f} ({(sl_usd/current_equity)*100:.2f}%) | "
                        f"Dollar TP: ${tp_usd:.2f}"
                    )

                    # 7. Execute Deriv Multiplier order (engages lock ONLY on confirmed fill)
                    order_res = await place_multiplier_contract(
                        symbol=symbol,
                        direction=signal.direction,
                        stake=stake_usd,
                        stop_loss_usd=sl_usd,
                        take_profit_usd=tp_usd,
                    )

                    if order_res:
                        mark_setup_consumed(signal)
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
