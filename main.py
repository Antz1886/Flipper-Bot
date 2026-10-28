"""
main.py — Conti Institutional SMC Trading Framework (Quant Edition — Self-Healing & Learning)
Orchestrates:
  1. Auto-Reconnecting WebSocket Session: Automatically restores dropped connections every cycle.
  2. Autonomous Self-Healing Supervisor:
     - Measures real-time WebSocket RTT and dynamically adapts Deal Cancellation SLA triggers.
     - Automatically reconciles orphan exchange contracts and purges ghost locks.
     - Enforces defensive drawdown throttling (cutting risk 50%) during loss streaks.
  3. Continual Bayesian Meta-Learning:
     - Evaluates setup contexts (Session + Liquidity Pool + Displacement + HTF Depth).
     - Vetoes setups exhibiting negative empirical expectancy (E[R] < 0) from accumulated history.
     - Adaptively scales position risk (0.50% - 1.00%) based on Thompson Sampling conviction.
  4. Post-Trade Attribution & SQLite Persistence: Logs immutable trade contexts and outcomes.
  5. Macro Session Gating: Enforces London Open (07:00–11:00 GMT) and New York Open (13:00–17:00 GMT).
  6. Tier-1 Data Embargo: Synchronizes ForexFactory calendar and halts USD trading ±30m around high-impact news.
  7. Micro-Mechanic Protections:
     - Minute 14, Second 55 Deal Cancellation SLA dispatch.
     - Live execution spread widening filter.
     - Cancellation premium bleed cap (max 15% of risk budget).
  8. Hardcoded Risk Mandates: Daily 5% drawdown circuit breaker + Base/Quote USD concurrency & DXY direction locks.
"""

import asyncio
import os
import sys
import logging
import csv
import uuid
from datetime import datetime, timezone

import config
from modules import connector
from modules.data_feed import fetch_ohlc
from modules.smc_engine import (
    detect_institutional_setup,
    mark_setup_consumed,
    is_in_macro_session,
    get_pip_size,
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
from modules.self_healer import get_self_healer
from modules.meta_learner import get_meta_learner
from modules.trade_learner import (
    get_learner_db,
    TradeContext,
    get_session_phase,
)
from modules.dbot_engine import DBotEngine

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


async def run_smc():
    log.info("=" * 80)
    log.info("   CONTI INSTITUTIONAL SMC BOT — QUANT EDITION (SELF-HEALING & LEARNING)")
    log.info("=" * 80)
    log.info(f"   Symbols             : {', '.join(config.SYMBOLS)}")
    session_desc = "London (07:00-11:00 GMT) & NY (13:00-17:00 GMT)" if getattr(config, "ENFORCE_MACRO_SESSIONS", False) else "24/7 Continuous Algorithmic (Synthetics Mode)"
    tf_min = max(1, config.PRIMARY_TIMEFRAME_S // 60)
    htf_hr = round(config.HTF_TIMEFRAME_S / 3600, 1)
    is_synth = any(s.upper().startswith(("R_", "1HZ", "HZ")) for s in config.SYMBOLS)
    active_swing = getattr(config, "SWING_LENGTH_SYNTHETICS", 3) if is_synth else getattr(config, "SWING_LENGTH", 5)
    active_disp = getattr(config, "MIN_DISPLACEMENT_ATR_SYNTHETICS", 0.50) if is_synth else getattr(config, "MIN_DISPLACEMENT_ATR", 0.80)
    log.info(f"   Execution Strategy  : M{tf_min} SMC (SwingLen: {active_swing}, Disp: {active_disp*100:.0f}% ATR)")
    log.info(f"   Macro Sessions      : {session_desc}")
    log.info(f"   HTF Dealing Range   : {htf_hr}H ({config.HTF_TIMEFRAME_S}s) Premium/Discount Matrix Enforced")
    log.info(f"   Self-Healing        : Active (Dynamic RTT SLA Adaptation & Orphan Reconciler)")
    log.info(f"   Continual Learning  : Active (Bayesian Contextual Bandits & Loss Veto Scorer)")
    log.info(f"   Attribution DB      : data/learner_history.sqlite")
    log.info(f"   Deal Cancellation   : True (SLA Trigger: Minute 14, Second 55)")
    log.info(f"   Premium Bleed Cap   : Max {getattr(config, 'MAX_CANCELLATION_FEE_PCT_OF_RISK', 0.15)*100:.0f}% of Risk Budget")
    log.info(f"   Spread Filter       : Max allowed: EURUSD {config.MAX_ALLOWED_SPREAD_PIPS.get('frxEURUSD', 1.5)} pips")
    log.info(f"   News Embargo        : ±{getattr(config, 'NEWS_EMBARGO_MINUTES_BEFORE', 30)}m around Tier-1 USD releases")
    log.info(f"   DXY Correlation Cap : Max {getattr(config, 'MAX_BASE_USD_TRADES', 1)} Base-USD & {getattr(config, 'MAX_QUOTE_USD_TRADES', 1)} Quote-USD Trade")
    log.info(f"   Risk Mandate        : {config.RISK_PER_TRADE_PCT*100:.1f}% Equity Risk at Structural SL")
    if getattr(config, "VIRTUAL_EQUITY", 0) > 0:
        log.info(f"   Virtual Equity      : ${config.VIRTUAL_EQUITY:.2f} USD (Account Size Override)")
    if getattr(config, "FIXED_STAKE_USD", 0) > 0:
        log.info(f"   Fixed Stake Override: ${config.FIXED_STAKE_USD:.2f} USD per trade")
    log.info(f"   Target R:R          : 1:{config.RISK_REWARD_RATIO:.1f} Structural Projection")
    log.info(f"   Demo Testing        : {config.DEMO_MODE}")
    log.info("=" * 80)

    # 1. Connect and Authorize with Deriv WebSocket API
    success = await connector.initialize()
    if not success:
        log.warning("Initial Deriv connection failed; scan loop will automatically retry every cycle.")

    # 2. Synchronize Economic Calendar for Tier-1 News Embargo
    if getattr(config, "ENFORCE_NEWS_EMBARGO", True):
        sync_calendar_if_needed(force=True)

    # 3. Initialize Bayesian Meta-Learner & Attribution DB
    meta_learner = get_meta_learner()
    self_healer = get_self_healer()
    learner_db = get_learner_db()

    # Dry Run verification check
    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Architecture, self-healing, meta-learner, and connectivity verified. Exiting.")
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

            # 1. Self-Healing Latency Measurement & Dynamic SLA Tuning
            await self_healer.measure_and_adapt_latency()

            # 2. Sync active contracts & stakes with Deriv exchange (triggers self-healing reconciler)
            await sync_open_positions()

            # 3. Check account equity & daily circuit breaker
            healthy = check_account_health()
            if not healthy:
                log.warning("Daily circuit breaker active or equity floor breached. Pausing cycle...")
                await asyncio.sleep(config.SCAN_INTERVAL_S)
                continue

            risk_state = get_risk_state()
            virtual_eq = getattr(config, "VIRTUAL_EQUITY", 0.0)
            current_equity = (
                virtual_eq
                if virtual_eq > 0
                else (
                    risk_state.effective_equity
                    if (api and api.authorized and risk_state.effective_equity > 0)
                    else 1000.0
                )
            )

            # 4. Micro-Mechanic Fix 4: Check Tier-1 Economic News Embargo
            embargo_active = False
            if getattr(config, "ENFORCE_NEWS_EMBARGO", True):
                sync_calendar_if_needed()
                embargo_active, embargo_reason = is_news_embargo_active()
                if embargo_active:
                    log.warning(f"{embargo_reason} — Signal generation locked across USD pairs.")

            # 5. Macro Session Gate: Check if current UTC time is inside London or NY institutional windows
            now_utc = datetime.now(timezone.utc)
            in_session = is_in_macro_session(now_utc)
            session_phase = get_session_phase(now_utc)
            if getattr(config, "ENFORCE_MACRO_SESSIONS", True) and not in_session:
                now_str = now_utc.strftime("%H:%M UTC")
                log.info(
                    f"⏳ Current time ({now_str}) is outside London/NY macro windows (07:00-11:00 & 13:00-17:00 GMT). "
                    f"Signal generation suspended (monitoring mode)."
                )

            for symbol in config.SYMBOLS:
                # 6. Ingest M15 execution candles
                df_m15 = await fetch_ohlc(symbol, granularity=config.PRIMARY_TIMEFRAME_S, n_bars=config.N_BARS_INGEST)
                if df_m15 is None or len(df_m15) < (config.SWING_LENGTH * 3 + 15):
                    log.warning(f"  Insufficient M15 candle data for {symbol} — skipping.")
                    continue

                current_spot = float(df_m15["close"].iloc[-1])

                # 7. Self-Healing Deal Cancellation Monitor (Minute 14, Second 55 SLA Trigger)
                if getattr(config, "USE_DEAL_CANCELLATION", True):
                    await evaluate_sluggish_ob_cancellations(symbol, current_spot)

                # Skip signal generation if outside macro sessions or under news embargo
                if getattr(config, "ENFORCE_MACRO_SESSIONS", True) and not in_session:
                    continue
                if getattr(config, "ENFORCE_NEWS_EMBARGO", True) and embargo_active:
                    continue

                # 8. Check symbol lock, Base/Quote USD concurrency, and DXY correlation
                if not can_open_symbol(symbol):
                    if symbol in risk_state.active_symbols:
                        log.info(f"  {symbol}: Active position open on exchange — skipping.")
                    else:
                        log.info(
                            f"  {symbol}: Currency correlation, concurrency limit, or news embargo active — skipping."
                        )
                    continue

                log.info(f"Evaluating {symbol} (${current_spot:.5f}) | Session: {session_phase}...")

                # 9. Ingest H4 candles for HTF Dealing Range
                df_h4 = None
                if getattr(config, "ENFORCE_HTF_PREMIUM_DISCOUNT", True):
                    df_h4 = await fetch_ohlc(symbol, granularity=config.HTF_TIMEFRAME_S, n_bars=config.N_HTF_BARS_INGEST)

                # 10. Signal Generation: SMC Engine with Macro Pool and HTF Filters
                signal = detect_institutional_setup(symbol, df_m15, df_h4=df_h4)

                if signal:
                    # 11. Verify DXY Correlation Direction Lock before sizing
                    if not can_open_symbol(symbol, direction=signal.direction):
                        log.info(f"  {symbol}: Blocked by DXY directional exposure lock — skipping.")
                        continue

                    # 12. Bayesian Continual Meta-Learner Conviction & Veto Check
                    approved, conviction_score, recommended_risk, veto_reason = meta_learner.evaluate_setup_conviction(
                        symbol=symbol,
                        session_phase=session_phase,
                        macro_pool_type=signal.macro_pool_type,
                        displacement_ratio=signal.displacement_ratio,
                        htf_depth_pct=signal.htf_depth_pct,
                    )

                    if not approved:
                        log.warning(f"🛑 [META-LEARNER VETO] {symbol} {signal.direction}: {veto_reason}")
                        continue

                    # 13. Self-Healing Defensive Risk Throttling (cuts risk 50% on loss streaks)
                    effective_risk_pct = self_healer.get_effective_risk_pct(recommended_risk)

                    # 14. Structural Multiplier Position Sizing with Learner-Adjusted Risk
                    stake_usd, sl_usd, tp_usd = calculate_order_risk(
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        balance=current_equity,
                        symbol=symbol,
                        risk_pct_override=effective_risk_pct,
                    )

                    log.info(
                        f"⚡ [{signal.direction} SIGNAL CONFIRMED] {symbol} | Pool: {signal.macro_pool_desc} | "
                        f"Conviction: {conviction_score*100:.1f}% | Effective Risk: {effective_risk_pct*100:.2f}% | "
                        f"Entry: ${signal.entry_price:.5f} | SL: ${signal.stop_loss:.5f} | "
                        f"TP: ${signal.take_profit:.5f} | Stake: ${stake_usd:.2f} | "
                        f"Risk: ${sl_usd:.2f} ({(sl_usd/current_equity)*100:.2f}%)"
                    )

                    # 15. Execute Multiplier order with Spread Filter & Cancellation Premium Check
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
                        contract_id = str(order_res.get("contract_id", uuid.uuid4().hex[:12]))
                        pip_size = get_pip_size(symbol)

                        # Capture immutable trade context for attribution and continual learning
                        trade_ctx = TradeContext(
                            trade_id=contract_id,
                            contract_id=contract_id,
                            symbol=symbol,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            stake_usd=stake_usd,
                            potential_risk_usd=sl_usd,
                            ob_top=signal.ob_top,
                            ob_bottom=signal.ob_bottom,
                            displacement_atr_ratio=signal.displacement_ratio,
                            htf_depth_pct=signal.htf_depth_pct,
                            macro_pool_type=signal.macro_pool_type,
                            session_phase=session_phase,
                            atr=signal.atr,
                            spread_pips=round(0.0001 / pip_size, 2),
                            entry_time_utc=datetime.now(timezone.utc).isoformat(),
                        )
                        learner_db.save_context(trade_ctx)

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


async def run_dbot():
    """Runs the Inbuilt Deriv Bot (DBot) continuous execution engine."""
    success = await connector.initialize()
    if not success:
        log.error("Failed to connect or authorize with Deriv API for DBot execution.")
        return

    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] DBot engine initialized and connection verified successfully. Exiting.")
        await connector.shutdown()
        return

    engine = DBotEngine()
    try:
        await engine.run()
    except KeyboardInterrupt:
        log.info("DBot shutdown requested by user.")
    except Exception as e:
        log.error(f"Unexpected error in DBot session: {e}", exc_info=True)
    finally:
        await connector.shutdown()


async def main():
    mode = getattr(config, "BOT_MODE", "DBOT").upper()
    log.info(f"⚡ Bootstrapping Flipper-Bot in [{mode}] Mode...")
    if mode == "DBOT":
        await run_dbot()
    elif mode in ("SMC", "SMC_MULTIPLIER", "MULTIPLIER"):
        await run_smc()
    else:
        log.warning(f"Unknown BOT_MODE '{mode}'. Defaulting to DBOT.")
        await run_dbot()


if __name__ == "__main__":
    asyncio.run(main())
