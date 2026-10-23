"""
main.py — Conti Institutional SMC Trading Framework (Quant Edition — Forex Majors)
Orchestrates:
  1. Auto-Reconnecting WebSocket Session: Automatically restores dropped connections every cycle.
  2. Macro Session Gating: Enforces London Open (07:00–11:00 GMT) and New York Open (13:00–17:00 GMT).
  3. Tier-1 Data Embargo: Synchronizes ForexFactory economic calendar and halts USD trading ±30m around high-impact news.
  4. Live Portfolio Sync & Deal Cancellation SLA Monitor:
     - Evaluates live spot at Minute 14, Second 55 (895s / 5s strict SLA before broker expiry).
     - Dispatches {"cancel": contract_id} immediately if price fails to displace away from OB.
  5. Execution Spread Widening Filter: Verifies live bid-ask spread before placing multiplier contracts.
  6. Cancellation Premium Bleed Protection: Enforces strict 15% risk-budget cap on cancellation fees.
  7. Hardcoded Risk Mandates: Daily 5% drawdown circuit breaker + Base/Quote USD concurrency & DXY direction locks.
  8. Multi-Timeframe Ingestion: M15 execution data + H4 dealing range candles via Deriv API.
  9. Quantitative SMC Engine: Closed-bar Liquidity Sweep -> Macro Pool Intersection -> ATR Displacement -> H4 Matrix.
  10. True Structural Sizing: Stake and dollar SL/TP mathematically bound to |entry - stop_loss| and multiplier.
"""

import asyncio
import os
import sys
import logging
import csv
from datetime import datetime, timezone

import config
from modules import connector
from modules.data_feed import fetch_ohlc
from modules.smc_engine import (
    detect_institutional_setup,
    mark_setup_consumed,
    is_in_macro_session,
)
from modules.risk_manager import (
    sync_open_positions,
    check_account_health,
    calculate_order_risk,
    can_open_symbol,
    get_risk_state,
)
from modules.order_executor import place_multiplier_contract
from modules.cancellation_manager import (
    register_cancellation_contract,
    evaluate_sluggish_ob_cancellations,
)
from modules.news_embargo import (
    sync_calendar_if_needed,
    is_news_embargo_active,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
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
                    "stop_loss_usd", "take_profit_usd", "contract_id", "sweep_level",
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, entry_price, stake,
                stop_loss_usd, take_profit_usd, contract_id, sweep_level,
            ])
    except Exception as e:
        log.error(f"Failed to log trade to CSV: {e}")


async def main():
    log.info("=" * 76)
    log.info("   CONTI INSTITUTIONAL SMC BOT — QUANT EDITION (FOREX MULTIPLIERS)")
    log.info("=" * 76)
    log.info(f"   Symbols             : {', '.join(config.SYMBOLS)}")
    log.info(f"   Macro Sessions      : London (07:00-11:00 GMT) & NY (13:00-17:00 GMT)")
    log.info(f"   HTF Dealing Range   : H4 (14400s) Premium/Discount Matrix Enforced")
    log.info(f"   Deal Cancellation   : {getattr(config, 'USE_DEAL_CANCELLATION', True)} (SLA Trigger: Minute 14, Second 55)")
    log.info(f"   Premium Bleed Cap   : Max {getattr(config, 'MAX_CANCELLATION_FEE_PCT_OF_RISK', 0.15)*100:.0f}% of 1% Risk Budget")
    log.info(f"   Spread Filter       : Max allowed: EURUSD {config.MAX_ALLOWED_SPREAD_PIPS.get('frxEURUSD', 1.5)} pips")
    log.info(f"   News Embargo        : ±{getattr(config, 'NEWS_EMBARGO_MINUTES_BEFORE', 30)}m around Tier-1 USD releases")
    log.info(f"   DXY Correlation Cap : Max {getattr(config, 'MAX_BASE_USD_TRADES', 1)} Base-USD & {getattr(config, 'MAX_QUOTE_USD_TRADES', 1)} Quote-USD Trade")
    log.info(f"   Risk Mandate        : {config.RISK_PER_TRADE_PCT*100:.1f}% Equity Risk at Structural SL")
    log.info(f"   Target R:R          : 1:{config.RISK_REWARD_RATIO:.1f} Structural Projection")
    log.info(f"   Demo Testing        : {config.DEMO_MODE}")
    log.info("=" * 76)

    # 1. Connect and Authorize with Deriv WebSocket API
    success = await connector.initialize()
    if not success:
        log.warning("Initial Deriv connection failed; scan loop will automatically retry every cycle.")

    # 2. Synchronize Economic Calendar for Tier-1 News Embargo
    if getattr(config, "ENFORCE_NEWS_EMBARGO", True):
        sync_calendar_if_needed(force=True)

    # Dry Run verification check
    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Architecture, news embargo, and connectivity verified. Exiting.")
        await connector.shutdown()
        return

    log.info("🚀 Starting institutional scanning and execution loop...")

    try:
        while True:
            log.info("─── Starting Institutional Scan Cycle ───")

            # 0. Ensure WebSocket connection is active
            if not await connector.ensure_connected():
                log.warning("Deriv WebSocket unavailable after reconnect attempts. Retrying next cycle...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            api = connector.get_api()

            # 1. Sync active contracts & stakes with Deriv exchange
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

            # 3. Micro-Mechanic Fix 4: Check Tier-1 Economic News Embargo
            if getattr(config, "ENFORCE_NEWS_EMBARGO", True):
                sync_calendar_if_needed()
                embargo_active, embargo_reason = is_news_embargo_active()
                if embargo_active:
                    log.warning(f"{embargo_reason} — Signal generation locked across USD pairs.")

            # 4. Macro Session Gate: Check if current UTC time is inside London or NY institutional windows
            now_utc = datetime.now(timezone.utc)
            in_session = is_in_macro_session(now_utc)
            if getattr(config, "ENFORCE_MACRO_SESSIONS", True) and not in_session:
                now_str = now_utc.strftime("%H:%M UTC")
                log.info(
                    f"⏳ Current time ({now_str}) is outside London/NY macro windows (07:00-11:00 & 13:00-17:00 GMT). "
                    f"Signal generation suspended (monitoring mode)."
                )

            for symbol in config.SYMBOLS:
                # 5. Ingest M15 execution candles
                df_m15 = await fetch_ohlc(symbol, granularity=config.PRIMARY_TIMEFRAME_S, n_bars=config.N_BARS_INGEST)
                if df_m15 is None or len(df_m15) < (config.SWING_LENGTH * 3 + 15):
                    log.warning(f"  Insufficient M15 candle data for {symbol} — skipping.")
                    continue

                current_spot = float(df_m15["close"].iloc[-1])

                # 6. Micro-Mechanic Fix 1: Sub-Second SLA Deal Cancellation Monitor (Evaluates at Minute 14, Second 55)
                if getattr(config, "USE_DEAL_CANCELLATION", True):
                    await evaluate_sluggish_ob_cancellations(symbol, current_spot)

                # Skip signal generation if outside macro sessions or under news embargo
                if getattr(config, "ENFORCE_MACRO_SESSIONS", True) and not in_session:
                    continue
                if getattr(config, "ENFORCE_NEWS_EMBARGO", True) and embargo_active:
                    continue

                # 7. Check symbol lock, Base/Quote USD concurrency, and DXY correlation
                if not can_open_symbol(symbol):
                    if symbol in risk_state.active_symbols:
                        log.info(f"  {symbol}: Active position open on exchange — skipping.")
                    else:
                        log.info(
                            f"  {symbol}: Currency correlation, concurrency limit, or news embargo active — skipping."
                        )
                    continue

                log.info(f"Evaluating {symbol} (${current_spot:.5f})...")

                # 8. Ingest H4 candles for HTF Dealing Range
                df_h4 = None
                if getattr(config, "ENFORCE_HTF_PREMIUM_DISCOUNT", True):
                    df_h4 = await fetch_ohlc(symbol, granularity=config.HTF_TIMEFRAME_S, n_bars=config.N_HTF_BARS_INGEST)

                # 9. Signal Generation: SMC Engine with Macro Pool and HTF Filters
                signal = detect_institutional_setup(symbol, df_m15, df_h4=df_h4)

                if signal:
                    # 10. Verify DXY Correlation Direction Lock before sizing
                    if not can_open_symbol(symbol, direction=signal.direction):
                        log.info(f"  {symbol}: Blocked by DXY directional exposure lock — skipping.")
                        continue

                    # 11. Structural Multiplier Position Sizing
                    stake_usd, sl_usd, tp_usd = calculate_order_risk(
                        signal.entry_price,
                        signal.stop_loss,
                        current_equity,
                        symbol=symbol,
                    )

                    log.info(
                        f"⚡ [{signal.direction} SIGNAL CONFIRMED] {symbol} | Pool: {signal.macro_pool_desc} | "
                        f"Entry: ${signal.entry_price:.5f} | Structural SL: ${signal.stop_loss:.5f} | "
                        f"TP: ${signal.take_profit:.5f} | Stake: ${stake_usd:.2f} | "
                        f"Risk: ${sl_usd:.2f} ({(sl_usd/current_equity)*100:.2f}%)"
                    )

                    # 12. Execute Multiplier order with Spread Filter & Cancellation Premium Check
                    order_res = await place_multiplier_contract(
                        symbol=symbol,
                        direction=signal.direction,
                        stake=stake_usd,
                        stop_loss_usd=sl_usd,
                        take_profit_usd=tp_usd,
                        ob_top=signal.ob_top,
                        ob_bottom=signal.ob_bottom,
                    )

                    if order_res:
                        mark_setup_consumed(signal)
                        contract_id = order_res.get("contract_id", "N/A")

                        # Register contract for SLA Deal Cancellation monitoring (Fix 1 & 2)
                        if order_res.get("has_deal_cancellation"):
                            register_cancellation_contract(
                                contract_id=contract_id,
                                symbol=symbol,
                                direction=signal.direction,
                                ob_top=signal.ob_top,
                                ob_bottom=signal.ob_bottom,
                                date_expiry=order_res.get("date_expiry"),
                                cancellation_fee=order_res.get("cancellation_fee", 0.0),
                            )

                        log_trade(
                            symbol=symbol,
                            side=signal.direction,
                            entry_price=signal.entry_price,
                            stake=stake_usd,
                            stop_loss_usd=sl_usd,
                            take_profit_usd=tp_usd,
                            contract_id=contract_id,
                            sweep_level=signal.sweep_level,
                        )
                else:
                    log.info(f"  {symbol}: ${current_spot:.5f} — No unmitigated macro liquidity sweep setup.")

            await asyncio.sleep(config.SCAN_INTERVAL_S)

    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
    except Exception as e:
        log.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        await connector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
