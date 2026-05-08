"""
main.py — Async Orchestrator (Deriv WebSocket edition)
Small Account Trading Bot | $10 → $100 | SMC + Full Kelly Compounding

Architecture change vs MT5 version:
  - No pending limit orders. Instead, the SMC engine identifies the OB
    price zone and a live tick subscriber watches for price to enter it.
  - When price touches the OB proximal edge, a MULTUP/MULTDOWN contract
    is bought at market with dollar-amount SL and TP embedded.

Run:
    .venv/bin/python main.py
    (or activate venv first: source .venv/bin/activate && python main.py)
"""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import config
from modules import connector
from modules.connector import get_api
from modules.data_feed import fetch_ohlc
from modules.smc_engine import detect_signal, TradeSignal
from modules.risk_manager import calculate_stake, get_balance, CircuitBreakerTripped
from modules.order_executor import (
    place_multiplier_contract,
    has_open_contract,
)
from modules.logger import setup_logging, log_trade, log_session_event

log = logging.getLogger(__name__)

# ─── Bot State ────────────────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()

@dataclass
class ActiveZone:
    """Holds the current unexecuted SMC signal waiting for price to enter."""
    signal:             TradeSignal
    stake:              float
    stop_loss_amount:   float
    take_profit_amount: float
    balance_snapshot:   float

_active_zone: Optional[ActiveZone] = None
_in_trade: bool = False


# ─── Tick Handler ─────────────────────────────────────────────────────────────

async def on_tick(tick: dict):
    """
    Called on every live tick from the Deriv subscription.
    Checks if price has entered the active OB zone and enters if so.
    """
    global _active_zone, _in_trade

    if _active_zone is None or _in_trade:
        return

    bid = float(tick.get("bid", 0))
    ask = float(tick.get("ask", 0))
    sig = _active_zone.signal

    zone_triggered = (
        (sig.direction == "BUY"  and ask <= sig.entry_price) or
        (sig.direction == "SELL" and bid >= sig.entry_price)
    )

    if not zone_triggered:
        return

    log.info(
        f"🎯 OB Zone HIT | {sig.direction} | Entry: {sig.entry_price:.5f} | "
        f"Tick bid/ask: {bid:.5f}/{ask:.5f}"
    )

    _in_trade = True
    zone = _active_zone
    _active_zone = None

    result = await place_multiplier_contract(
        signal=zone.signal,
        stake=zone.stake,
        stop_loss_amount=zone.stop_loss_amount,
        take_profit_amount=zone.take_profit_amount,
    )

    log_trade(
        signal=zone.signal,
        stake=zone.stake,
        stop_loss_amount=zone.stop_loss_amount,
        take_profit_amount=zone.take_profit_amount,
        balance_before=zone.balance_snapshot,
        order_result=result,
        result="PLACED" if result else "REJECTED",
    )

    if not result:
        _in_trade = False  # Allow retry on next scan if trade failed


# ─── Scan Cycle ───────────────────────────────────────────────────────────────

async def run_scan_cycle() -> bool:
    """
    One full scan: fetch OHLC → SMC analysis → update active zone.
    Returns False to stop the bot, True to continue.
    """
    global _active_zone, _in_trade

    symbol = config.SYMBOL

    # ── Balance & target check ────────────────────────────────────────────────
    try:
        balance = await get_balance()
    except Exception:
        balance = get_api().balance

    if balance >= config.ACCOUNT_TARGET_USD:
        log.info(f"🎯 TARGET REACHED! Balance ${balance:.2f}. Bot stopping.")
        log_session_event("TARGET_REACHED", f"Final balance: ${balance:.2f}")
        return False

    log.info(
        f"── Scan | Balance: ${balance:.2f} | "
        f"Progress: {balance / config.ACCOUNT_TARGET_USD * 100:.1f}% ──"
    )

    # ── Early open contract check (State Management) ─────────────────────────
    # Always check actual broker state before proceeding
    still_open = await has_open_contract(symbol)
    if still_open:
        if not _in_trade:
            log.info("Active contract detected on start/scan — syncing state.")
            _in_trade = True
        log.info("Open contract active — holding.")
        return True
    else:
        if _in_trade:
            log.info("Contract closed (SL/TP hit). Ready for next setup.")
            _in_trade = False
            _active_zone = None  # Clear any old zone if trade closed

    # ── Kelly sizing ──────────────────────────────────────────────────────────
    try:
        sizing = calculate_stake(balance)
    except CircuitBreakerTripped as e:
        log.critical(f"🛑 CIRCUIT BREAKER: {e}")
        log_session_event("CIRCUIT_BREAKER", str(e))
        log.info("Pausing for 5 minutes...")
        await asyncio.sleep(300)
        return True

    if sizing is None:
        return True

    stake, sl_amount, tp_amount = sizing

    # ── Fetch OHLC ────────────────────────────────────────────────────────────
    primary_df    = await fetch_ohlc(symbol, config.PRIMARY_TF_S)
    confluence_df = await fetch_ohlc(symbol, config.CONFLUENCE_TF_S)

    if primary_df is None or confluence_df is None:
        log.warning("OHLC fetch failed — skipping cycle.")
        return True

    # ── Current price (from connector's last cached tick) ─────────────────────
    api = get_api()
    # Use last known balance as price proxy — we'll get real price from ticks
    # Fetch a fresh tick via a one-shot ticks_history (last candle close)
    last_close = float(primary_df["close"].iloc[-1])
    bid = last_close
    ask = last_close

    # ── SMC Signal ────────────────────────────────────────────────────────────
    signal = await detect_signal(
        symbol=symbol,
        primary_ohlc=primary_df,
        confluence_ohlc=confluence_df,
        current_bid=bid,
        current_ask=ask,
    )

    if signal is None:
        log.info(f"No confluence setup found for {symbol}.")
        _active_zone = None
        return True

    # ── Arm the active zone (tick handler will trigger entry) ─────────────────
    # If we already have a zone armed for this direction, don't log unnecessarily
    if _active_zone and _active_zone.signal.direction == signal.direction:
        log.debug(f"Zone already armed for {signal.direction} {symbol}")
    else:
        _active_zone = ActiveZone(
            signal=signal,
            stake=stake,
            stop_loss_amount=sl_amount,
            take_profit_amount=tp_amount,
            balance_snapshot=balance,
        )
        log.info(
            f"Zone armed | {signal.direction} {symbol} | "
            f"Entry: {signal.entry_price:.5f} | "
            f"OB [{signal.ob_bottom:.5f}–{signal.ob_top:.5f}]"
        )
    return True


# ─── Health Monitor ───────────────────────────────────────────────────────────

async def health_monitor():
    while not _shutdown_event.is_set():
        await asyncio.sleep(config.HEALTH_CHECK_S)
        api = get_api()
        if not api or not await api.is_connected():
            log.warning("⚠️  Deriv API disconnected! Attempting reconnection...")
            if await api.connect():
                log.info("✅ Reconnected to Deriv. Re-subscribing to ticks...")
                # Re-subscribe to ticks after reconnection
                await api.subscribe_ticks(config.SYMBOL, on_tick)
            else:
                log.error("❌ Reconnection failed. Will retry next health check.")


# ─── Signal Handlers ──────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    log.warning(f"OS signal {sig} received — shutting down...")
    _shutdown_event.set()


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    setup_logging()

    log.info("=" * 60)
    mode_str = "🧪 DEMO / VIRTUAL ACCOUNT" if config.DEMO_MODE else "🔴 LIVE ACCOUNT"
    log.info(f"  Mode      : {mode_str}")
    log.info(f"  Symbol    : {config.SYMBOL}")
    log.info(f"  Target    : ${config.ACCOUNT_TARGET_USD:.2f}")
    log.info(f"  Kelly     : {config.KELLY_FRACTION * 100:.0f}% of balance per trade")
    log.info(f"  Max Stake : ${config.MAX_STAKE_USD:.2f} per trade")
    log.info(f"  Multiplier: {config.MULTIPLIER}x")
    log.info(f"  CB Floor  : ${config.CIRCUIT_BREAKER_USD:.2f}")
    log.info(f"  TF        : {config.PRIMARY_TF_S}s primary / {config.CONFLUENCE_TF_S}s confluence")
    log.info("=" * 60)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    # ── Connect to Deriv ─────────────────────────────────────────────────────
    if not await connector.initialize():
        log.critical("Cannot connect to Deriv API. Check credentials. Exiting.")
        sys.exit(1)

    log_session_event("BOT_START", f"Symbol: {config.SYMBOL}")

    # ── Subscribe to live ticks ───────────────────────────────────────────────
    sub_id = await get_api().subscribe_ticks(config.SYMBOL, on_tick)

    # ── Start health monitor ──────────────────────────────────────────────────
    monitor_task = asyncio.create_task(health_monitor())

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while not _shutdown_event.is_set():
            should_continue = await run_scan_cycle()
            if not should_continue:
                break

            # Wait for next cycle (interruptible by shutdown)
            try:
                await asyncio.wait_for(
                    asyncio.shield(_shutdown_event.wait()),
                    timeout=config.SCAN_INTERVAL_S,
                )
                break  # shutdown was signalled
            except asyncio.TimeoutError:
                pass  # normal — next scan

    except Exception as e:
        log.critical(f"Unhandled error in main loop: {e}", exc_info=True)
        log_session_event("ERROR", str(e))

    finally:
        _shutdown_event.set()
        monitor_task.cancel()
        try:
            await get_api().unsubscribe(sub_id)
        except Exception:
            pass
        log_session_event("BOT_STOP")
        await connector.shutdown()
        log.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
