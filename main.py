"""
main.py — Async Orchestrator (Multi-Asset Deriv WebSocket edition)
Small Account Trading Bot | $10 → $100 | SMC + Full Kelly Compounding
"""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Optional, Dict

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

# Dicts to track state per symbol
_active_zones: Dict[str, Optional[ActiveZone]] = {s: None for s in config.SYMBOLS}
_in_trades:    Dict[str, bool] = {s: False for s in config.SYMBOLS}


# ─── Tick Handler ─────────────────────────────────────────────────────────────

async def on_tick(tick: dict):
    """
    Called on every live tick from the Deriv subscription.
    Checks if price has entered the active OB zone for that specific symbol.
    """
    global _active_zones, _in_trades

    symbol = tick.get("symbol")
    if not symbol or symbol not in _active_zones:
        return

    if _active_zones[symbol] is None or _in_trades[symbol]:
        return

    bid = float(tick.get("bid", 0))
    ask = float(tick.get("ask", 0))
    sig = _active_zones[symbol].signal

    zone_triggered = (
        (sig.direction == "BUY"  and ask <= sig.entry_price) or
        (sig.direction == "SELL" and bid >= sig.entry_price)
    )

    if not zone_triggered:
        return

    log.info(
        f"🎯 OB Zone HIT | {symbol} | {sig.direction} | Entry: {sig.entry_price:.5f} | "
        f"Tick bid/ask: {bid:.5f}/{ask:.5f}"
    )

    _in_trades[symbol] = True
    zone = _active_zones[symbol]
    _active_zones[symbol] = None

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
        _in_trades[symbol] = False  # Allow retry on next scan if trade failed


# ─── Scan Cycle ───────────────────────────────────────────────────────────────

async def run_scan_cycle() -> bool:
    """
    One full scan across all symbols.
    Returns False to stop the bot, True to continue.
    """
    global _active_zones, _in_trades

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

    # ── Kelly sizing (computed once per cycle) ─────────────────────────────
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

    # ── Loop through all symbols ──────────────────────────────────────────────
    for symbol in config.SYMBOLS:
        # ── Early open contract check (State Management) ─────────────────────────
        still_open = await has_open_contract(symbol)
        if still_open:
            if not _in_trades[symbol]:
                log.info(f"Active contract detected for {symbol} — syncing state.")
                _in_trades[symbol] = True
            log.debug(f"{symbol} open contract active — holding.")
            continue
        else:
            if _in_trades[symbol]:
                log.info(f"Contract for {symbol} closed. Ready for next setup.")
                _in_trades[symbol] = False
                _active_zones[symbol] = None

        # ── Fetch OHLC ────────────────────────────────────────────────────────────
        primary_df    = await fetch_ohlc(symbol, config.PRIMARY_TF_S)
        confluence_df = await fetch_ohlc(symbol, config.CONFLUENCE_TF_S)

        if primary_df is None or confluence_df is None:
            log.warning(f"OHLC fetch failed for {symbol} — skipping.")
            continue

        # ── Current price (from last candle close) ─────────────────────────────
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
            log.debug(f"No confluence setup found for {symbol}.")
            _active_zones[symbol] = None
            continue

        # ── Arm the active zone (tick handler will trigger entry) ─────────────────
        if _active_zones[symbol] and _active_zones[symbol].signal.direction == signal.direction:
            log.debug(f"Zone already armed for {signal.direction} {symbol}")
        else:
            _active_zones[symbol] = ActiveZone(
                signal=signal,
                stake=stake,
                stop_loss_amount=sl_amount,
                take_profit_amount=tp_amount,
                balance_snapshot=balance,
            )
            log.info(
                f"Zone armed | {symbol} {signal.direction} | "
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
                for symbol in config.SYMBOLS:
                    await api.subscribe_ticks(symbol, on_tick)
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
    log.info(f"  Symbols   : {', '.join(config.SYMBOLS)}")
    log.info(f"  Target    : ${config.ACCOUNT_TARGET_USD:.2f}")
    log.info(f"  Kelly     : {config.KELLY_FRACTION * 100:.0f}% of balance per trade")
    log.info(f"  Max Stake : ${config.MAX_STAKE_USD:.2f} per trade")
    log.info(f"  Multiplier: {config.MULTIPLIER}x")
    log.info(f"  TF        : {config.PRIMARY_TF_S}s primary / {config.CONFLUENCE_TF_S}s confluence")
    log.info("=" * 60)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    # ── Connect to Deriv ─────────────────────────────────────────────────────
    if not await connector.initialize():
        log.critical("Cannot connect to Deriv API. Check credentials. Exiting.")
        sys.exit(1)

    log_session_event("BOT_START", f"Symbols: {config.SYMBOLS}")

    # ── Subscribe to live ticks ───────────────────────────────────────────────
    sub_ids = []
    for symbol in config.SYMBOLS:
        sid = await get_api().subscribe_ticks(symbol, on_tick)
        if sid: sub_ids.append(sid)

    # ── Start health monitor ──────────────────────────────────────────────────
    monitor_task = asyncio.create_task(health_monitor())

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while not _shutdown_event.is_set():
            should_continue = await run_scan_cycle()
            if not should_continue:
                break

            try:
                await asyncio.wait_for(
                    asyncio.shield(_shutdown_event.wait()),
                    timeout=config.SCAN_INTERVAL_S,
                )
                break
            except asyncio.TimeoutError:
                pass

    except Exception as e:
        log.critical(f"Unhandled error in main loop: {e}", exc_info=True)
        log_session_event("ERROR", str(e))

    finally:
        _shutdown_event.set()
        monitor_task.cancel()
        for sid in sub_ids:
            try: await get_api().unsubscribe(sid)
            except Exception: pass
        log_session_event("BOT_STOP")
        await connector.shutdown()
        log.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
