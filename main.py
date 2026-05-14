"""
main.py — Async Orchestrator (Multi-Asset Deriv WebSocket edition)
Small Account Trading Bot | $10 → $100 | SMC + Full Kelly Compounding

v2.0 — Overhauled with:
  - Phantom contract detection fix
  - Per-symbol dedup locks
  - Zone TTL (4h expiry)
  - Dynamic multiplier discovery
  - Startup portfolio sync
  - Disconnect-safe contract state management
  - 2 concurrent trades allowed across symbols
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
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
    get_last_contract_result,
)
from modules.logger import setup_logging, log_trade, log_trade_outcome, log_session_event

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
    created_at:         float = 0.0   # time.time() when zone was armed

# Dicts to track state per symbol
_active_zones: Dict[str, Optional[ActiveZone]] = {}
_in_trades:    Dict[str, bool] = {}
_last_trade_time: Dict[str, float] = {}
_tick_locks:   Dict[str, asyncio.Lock] = {}

PID_FILE = "bot.pid"


def _init_symbol_state():
    """Initialize per-symbol state dicts. Called after SYMBOLS is finalized."""
    global _active_zones, _in_trades, _last_trade_time, _tick_locks
    _active_zones    = {s: None for s in config.SYMBOLS}
    _in_trades       = {s: False for s in config.SYMBOLS}
    _last_trade_time = {s: 0.0 for s in config.SYMBOLS}
    _tick_locks      = {s: asyncio.Lock() for s in config.SYMBOLS}


def _count_open_trades() -> int:
    """Return the number of symbols currently in an active trade."""
    return sum(1 for v in _in_trades.values() if v)


# ─── Dynamic Multiplier Discovery ────────────────────────────────────────────

async def discover_multipliers():
    """
    Query the Deriv `contracts_for` API to discover valid multiplier values
    for each symbol. Updates config.SYMBOL_MULTIPLIERS in place.
    Removes symbols from config.SYMBOLS if no valid multiplier is found.
    """
    api = get_api()
    symbols_to_remove = []
    
    for symbol in config.SYMBOLS:
        try:
            resp = await api.send({
                "contracts_for": symbol,
                "currency": "USD",
                "product_type": "basic",
            })
            
            available = resp.get("contracts_for", {}).get("available", [])
            
            # Find multiplier contract types
            valid_multipliers = []
            for contract in available:
                if contract.get("contract_category") == "multiplier":
                    mults = contract.get("multiplier_range", [])
                    if mults:
                        valid_multipliers = [int(m) for m in mults]
                        break
            
            if valid_multipliers:
                # Pick a moderate multiplier (prefer 50x for R_75, 100x for R_100)
                preferred = config.SYMBOL_MULTIPLIERS.get(symbol, 50)
                if preferred in valid_multipliers:
                    chosen = preferred
                else:
                    # Pick the closest available
                    chosen = min(valid_multipliers, key=lambda x: abs(x - preferred))
                
                config.SYMBOL_MULTIPLIERS[symbol] = chosen
                log.info(f"✅ {symbol}: Valid multipliers {valid_multipliers} → using {chosen}x")
            else:
                log.warning(f"⚠️  {symbol}: No multiplier contracts available — disabling symbol.")
                symbols_to_remove.append(symbol)
                
        except Exception as e:
            log.warning(f"⚠️  {symbol}: Multiplier discovery failed ({e}) — using default {config.SYMBOL_MULTIPLIERS.get(symbol, '?')}x")
    
    # Remove symbols with no valid multipliers
    for sym in symbols_to_remove:
        config.SYMBOLS.remove(sym)
        config.SYMBOL_MULTIPLIERS.pop(sym, None)
    
    if not config.SYMBOLS:
        log.critical("No tradable symbols remaining after multiplier discovery!")
        return False
    
    return True


# ─── Startup Portfolio Sync ───────────────────────────────────────────────────

async def sync_portfolio_state():
    """
    Called once at startup to sync _in_trades with actual Deriv portfolio.
    Prevents opening duplicate positions after a restart.
    """
    for symbol in config.SYMBOLS:
        for attempt in range(1, 4):
            status = await has_open_contract(symbol)
            if status is True:
                _in_trades[symbol] = True
                _last_trade_time[symbol] = time.time()
                log.info(f"📋 Startup sync: Active contract found for {symbol} — marking in-trade.")
                break
            elif status is False:
                log.info(f"📋 Startup sync: No active contract for {symbol} — ready to trade.")
                break
            else:
                log.warning(f"📋 Startup sync: Could not determine state for {symbol} (attempt {attempt}/3).")
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)
                else:
                    log.critical(f"❌ Startup sync failed for {symbol}. Bot may open duplicate trades!")
                    # Safety: assume in-trade if we can't be sure
                    _in_trades[symbol] = True


# ─── Tick Handler ─────────────────────────────────────────────────────────────

async def on_tick(tick: dict):
    """
    Called on every live tick from the Deriv subscription.
    Checks if price has entered the active OB zone for that specific symbol.
    Uses per-symbol locks to prevent duplicate entries.
    """
    global _active_zones, _in_trades

    symbol = tick.get("symbol")
    if not symbol or symbol not in _active_zones:
        return

    # Quick pre-check without lock
    if _active_zones[symbol] is None or _in_trades[symbol]:
        return
    
    # Check concurrent trade limit
    if _count_open_trades() >= config.MAX_CONCURRENT_TRADES:
        return

    # Per-symbol lock — if already locked, another tick is being processed
    if _tick_locks[symbol].locked():
        return
    
    async with _tick_locks[symbol]:
        # Re-check after acquiring lock (state may have changed)
        if _active_zones[symbol] is None or _in_trades[symbol]:
            return
        if _count_open_trades() >= config.MAX_CONCURRENT_TRADES:
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

        # Set trade flag BEFORE executing — atomic dedup
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
        else:
            _last_trade_time[symbol] = time.time()


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

    open_count = _count_open_trades()
    log.info(
        f"── Scan | Balance: ${balance:.2f} | "
        f"Progress: {balance / config.ACCOUNT_TARGET_USD * 100:.1f}% | "
        f"Open trades: {open_count}/{config.MAX_CONCURRENT_TRADES} ──"
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
        # ── Open contract check (with disconnect safety) ──────────────────────
        # We ALWAYS check this first, even if we are at max trades, 
        # so we can detect when a trade has closed.
        still_open = await has_open_contract(symbol)
        
        if still_open is None:
            # Can't determine state — don't change anything, skip this symbol
            log.warning(f"Cannot determine contract state for {symbol} — skipping this cycle.")
            continue
        
        if still_open:
            if not _in_trades[symbol]:
                log.info(f"Active contract detected for {symbol} — syncing state.")
                _in_trades[symbol] = True
                _last_trade_time[symbol] = time.time()
            log.debug(f"{symbol} open contract active — holding.")
            continue
        else:
            # Portfolio says no open contract
            if _in_trades[symbol]:
                time_since_last = time.time() - _last_trade_time[symbol]
                if time_since_last < config.TRADE_COOLDOWN_S:
                    log.debug(
                        f"Portfolio empty for {symbol} but in cooldown "
                        f"({time_since_last:.1f}s < {config.TRADE_COOLDOWN_S}s) — waiting."
                    )
                    continue
                
                log.info(f"📊 Contract for {symbol} closed. Fetching result...")
                
                # Fetch result from profit table
                outcome = await get_last_contract_result(symbol)
                if outcome:
                    pnl = float(outcome.get("profit", 0))
                    res_str = "WIN" if pnl > 0 else "LOSS"
                    log.info(f"💰 Outcome for {symbol}: {res_str} | PnL: ${pnl:.2f}")
                    
                    log_trade_outcome(
                        symbol=symbol,
                        contract_id=str(outcome.get("contract_id", "")),
                        pnl_usd=pnl,
                        balance_after=balance,
                        result=res_str,
                        notes=f"Entry: ${outcome.get('buy_price')} | Sell: ${outcome.get('sell_price')}"
                    )
                
                _in_trades[symbol] = False
                _active_zones[symbol] = None

        # ── Check concurrent trade limit ──────────────────────────────────────
        # Only check this AFTER we've verified existing trades.
        if _count_open_trades() >= config.MAX_CONCURRENT_TRADES:
            log.debug(f"Max concurrent trades ({config.MAX_CONCURRENT_TRADES}) reached — skipping scan for {symbol}.")
            continue

        # ── Zone TTL check — expire stale zones ──────────────────────────────
        if _active_zones[symbol] is not None:
            zone_age = time.time() - _active_zones[symbol].created_at
            if zone_age > config.ZONE_TTL_S:
                log.info(
                    f"⏰ Zone expired for {symbol} after {zone_age/3600:.1f}h "
                    f"(TTL: {config.ZONE_TTL_S/3600:.0f}h) — clearing."
                )
                _active_zones[symbol] = None

        # ── Fetch OHLC ────────────────────────────────────────────────────────
        primary_df    = await fetch_ohlc(symbol, config.PRIMARY_TF_S)
        confluence_df = await fetch_ohlc(symbol, config.CONFLUENCE_TF_S)

        if primary_df is None or confluence_df is None:
            log.warning(f"OHLC fetch failed for {symbol} — skipping.")
            continue

        # ── Current price (from last candle close) ─────────────────────────────
        last_close = float(primary_df["close"].iloc[-1])
        bid = last_close
        ask = last_close

        # ── SMC Signal ────────────────────────────────────────────────────────
        signal = await detect_signal(
            symbol=symbol,
            primary_ohlc=primary_df,
            confluence_ohlc=confluence_df,
            current_bid=bid,
            current_ask=ask,
        )

        if signal is None:
            log.debug(f"No confluence setup found for {symbol}.")
            # Don't clear existing zone — it may still be valid within TTL
            continue

        # ── Zone freshness check — don't re-arm identical zones ──────────────
        if _active_zones[symbol] is not None:
            existing = _active_zones[symbol].signal
            price_diff = abs(existing.entry_price - signal.entry_price) / signal.entry_price
            if (existing.direction == signal.direction and price_diff < 0.001):
                log.debug(f"Zone unchanged for {symbol} ({signal.direction} @ {signal.entry_price:.2f}) — holding.")
                continue
            else:
                log.info(
                    f"🔄 New zone detected for {symbol}: "
                    f"{existing.direction}@{existing.entry_price:.2f} → "
                    f"{signal.direction}@{signal.entry_price:.2f} — re-arming."
                )

        # ── Arm the active zone (tick handler will trigger entry) ─────────────
        _active_zones[symbol] = ActiveZone(
            signal=signal,
            stake=stake,
            stop_loss_amount=sl_amount,
            take_profit_amount=tp_amount,
            balance_snapshot=balance,
            created_at=time.time(),
        )
        log.info(
            f"🎯 Zone armed | {symbol} {signal.direction} | "
            f"Entry: {signal.entry_price:.5f} | "
            f"OB [{signal.ob_bottom:.5f}–{signal.ob_top:.5f}] | "
            f"Conf: {signal.confidence:.0%}"
        )
            
    return True


# ─── Health Monitor ───────────────────────────────────────────────────────────

async def health_monitor():
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(config.HEALTH_CHECK_S)
            api = get_api()
            if not api or not await api.is_connected():
                log.warning("⚠️  Deriv API disconnected! Attempting reconnection...")
                if await api.connect():
                    log.info("✅ Reconnected to Deriv. Re-subscribing to ticks...")
                    for symbol in config.SYMBOLS:
                        try:
                            await api.subscribe_ticks(symbol, on_tick)
                        except Exception as e:
                            log.error(f"Failed to re-subscribe to {symbol}: {e}")
                else:
                    log.error("❌ Reconnection failed. Will retry next health check.")
        except Exception as e:
            log.error(f"Health monitor encountered an error: {e}", exc_info=True)
            await asyncio.sleep(10)  # Brief pause before retrying loop


# ─── Signal Handlers ──────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    log.warning(f"OS signal {sig} received — shutting down...")
    _shutdown_event.set()


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    # ── PID Lock ─────────────────────────────────────────────────────────────
    import os
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Check if process is actually running
            os.kill(old_pid, 0)
            print(f"❌ ERROR: Bot is already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (OSError, ValueError):
            # Process not running or stale pid file
            os.remove(PID_FILE)

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    setup_logging()

    log.info("=" * 60)
    mode_str = "🧪 DEMO / VIRTUAL ACCOUNT" if config.DEMO_MODE else "🔴 LIVE ACCOUNT"
    log.info(f"  Mode      : {mode_str}")
    log.info(f"  Symbols   : {', '.join(config.SYMBOLS)}")
    log.info(f"  Target    : ${config.ACCOUNT_TARGET_USD:.2f}")
    log.info(f"  Kelly     : {config.KELLY_FRACTION * 100:.0f}% of balance per trade")
    log.info(f"  Max Stake : ${config.MAX_STAKE_USD:.2f} per trade")
    log.info(f"  Max Concurrent: {config.MAX_CONCURRENT_TRADES} trades")
    log.info(f"  Zone TTL  : {config.ZONE_TTL_S/3600:.0f} hours")
    log.info(f"  AI Veto   : <{config.AI_VETO_THRESHOLD:.0%} win probability")
    m_str = ", ".join([f"{s}: {m}x" for s, m in config.SYMBOL_MULTIPLIERS.items()])
    log.info(f"  Multipliers: {m_str}")
    log.info(f"  TF        : {config.PRIMARY_TF_S}s primary / {config.CONFLUENCE_TF_S}s confluence")
    log.info("=" * 60)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    # ── Connect to Deriv ─────────────────────────────────────────────────────
    if not await connector.initialize():
        log.critical("Cannot connect to Deriv API. Check credentials. Exiting.")
        sys.exit(1)

    # ── Dynamic multiplier discovery ──────────────────────────────────────────
    log.info("🔍 Discovering valid multipliers for each symbol...")
    if not await discover_multipliers():
        log.critical("No tradable symbols after multiplier discovery. Exiting.")
        sys.exit(1)
    
    # ── Initialize per-symbol state (after SYMBOLS may have been pruned) ──────
    _init_symbol_state()

    log_session_event("BOT_START", f"Symbols: {config.SYMBOLS}")

    # ── Sync portfolio state ──────────────────────────────────────────────────
    await sync_portfolio_state()

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
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        log.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
