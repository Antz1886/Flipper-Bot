"""
main.py — Async Orchestrator (Deriv Tick-Momentum edition)
Small Account Trading Bot | $10 → $100 | Tick Auto-Correlation
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import logging
import signal
import sys
import time
from typing import Dict, List

import config
from modules import connector
from modules.connector import get_api
from modules.risk_manager import calculate_stake, can_open_position, CircuitBreakerTripped
from modules.order_executor import (
    place_tick_contract,
    place_multiplier_contract,
    has_open_contract,
    get_last_contract_result,
)
from modules.smc_engine import TradeSignal
from modules.logger import setup_logging, log_trade_outcome, log_session_event

log = logging.getLogger(__name__)

# ─── Bot State ────────────────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()
_portfolio_synced = False

_in_trades: Dict[str, bool] = {}
_last_trade_time: Dict[str, float] = {}
_tick_history: Dict[str, List[float]] = {}
_tick_locks: Dict[str, asyncio.Lock] = {}
_balance: float = 0.0
_initial_balance: float = None

PID_FILE = "bot.pid"
DRY_RUN = False


def _init_symbol_state():
    """Initialize state dictionaries for each configured symbol."""
    global _in_trades, _last_trade_time, _tick_history, _tick_locks
    _in_trades       = {s: False for s in config.SYMBOLS}
    _last_trade_time = {s: 0.0 for s in config.SYMBOLS}
    _tick_history    = {s: [] for s in config.SYMBOLS}
    _tick_locks      = {s: asyncio.Lock() for s in config.SYMBOLS}


async def sync_portfolio_state():
    """Sync bot trade state with the active Deriv portfolio and update balance."""
    global _portfolio_synced, _balance, _initial_balance
    _portfolio_synced = False

    log.info("📋 Syncing portfolio state with Deriv...")
    
    # 1. Fetch current account balance
    try:
        api = get_api()
        resp = await api.send({"balance": 1})
        _balance = float(resp["balance"]["balance"])
        log.info(f"Balance synchronized: USD {_balance:.2f}")
    except Exception as e:
        log.error(f"Failed to fetch balance during sync: {e}")
        _balance = get_api().balance

    if _initial_balance is None and _balance > 0:
        _initial_balance = _balance
        log.info(f"Daily starting balance set to: USD {_initial_balance:.2f}")

    # 2. Check for active open contracts
    for symbol in config.SYMBOLS:
        synced_symbol = False
        for attempt in range(1, 4):
            status = await has_open_contract(symbol)
            if status is True:
                _in_trades[symbol] = True
                _last_trade_time[symbol] = time.time()
                log.info(f"📋 Portfolio sync: Active contract found for {symbol} — marking in-trade.")
                synced_symbol = True
                break
            elif status is False:
                _in_trades[symbol] = False
                log.info(f"📋 Portfolio sync: No active contract for {symbol} — ready to trade.")
                synced_symbol = True
                break
            else:
                log.warning(f"📋 Portfolio sync: Could not determine state for {symbol} (attempt {attempt}/3).")
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)

        if not synced_symbol:
            log.critical(f"❌ Portfolio sync failed for {symbol}. Defaulting to in-trade status for safety.")
            _in_trades[symbol] = True

    _portfolio_synced = True


# ─── Tick Handler ─────────────────────────────────────────────────────────────

async def on_tick(tick: dict):
    """
    Called on every live tick from the Deriv subscription.
    Analyzes auto-correlation and triggers Rise/Fall trades on tick momentum.
    """
    global _tick_history, _in_trades, _last_trade_time, _balance

    if not _portfolio_synced:
        return

    symbol = tick.get("symbol")
    if not symbol or symbol not in config.SYMBOLS:
        return

    bid = float(tick.get("bid", 0))
    ask = float(tick.get("ask", 0))
    midpoint = (bid + ask) / 2.0

    # 1. Maintain tick history queue
    history = _tick_history[symbol]
    history.append(midpoint)
    if len(history) > 4:
        history.pop(0)

    if len(history) < 4:
        return

    # 2. Active trade & cooldown checks
    if _in_trades[symbol]:
        return

    now = time.time()
    if now - _last_trade_time[symbol] < config.TRADE_COOLDOWN_S:
        return

    if _tick_locks[symbol].locked():
        return

    # 3. Process signal under lock
    async with _tick_locks[symbol]:
        if _in_trades[symbol] or (now - _last_trade_time[symbol] < config.TRADE_COOLDOWN_S):
            return

        t1, t2, t3, t4 = history
        direction = None

        # Bullish acceleration: consecutive increases + velocity of last tick is higher
        if (t4 > t3 > t2 > t1) and ((t4 - t3) > (t3 - t2)):
            direction = "CALL"
        # Bearish acceleration: consecutive decreases + velocity of last tick is higher
        elif (t4 < t3 < t2 < t1) and ((t3 - t4) > (t2 - t3)):
            direction = "PUT"

        if not direction:
            return

        # 4. Sizing and risk check
        try:
            stake = calculate_stake(_balance)
            if stake is None:
                return  # Target reached
            
            # Count currently active trades
            current_open = sum(1 for v in _in_trades.values() if v)
            if not can_open_position(current_open, _balance):
                return
        except CircuitBreakerTripped as e:
            log.critical(f"🛑 CIRCUIT BREAKER TRIPPED: {e}")
            log_session_event("CIRCUIT_BREAKER", str(e))
            _shutdown_event.set()
            return

        # 5. Place Contract
        log.info(
            f"⚡ Tick Momentum Detected! | {symbol} {direction} | "
            f"Ticks: {t1:.4f} → {t2:.4f} → {t3:.4f} → {t4:.4f} | "
            f"Velocity: {abs(t4-t3):.4f} > {abs(t3-t2):.4f}"
        )

        if DRY_RUN:
            log.info(f"🧪 [DRY RUN] Would execute: {symbol} {direction} | Stake: ${stake:.2f}")
            _last_trade_time[symbol] = time.time()
            return

        _in_trades[symbol] = True
        _last_trade_time[symbol] = now

        if config.USE_MULTIPLIERS:
            sig = TradeSignal(
                symbol=symbol,
                direction="BUY" if direction == "CALL" else "SELL",
                entry_price=midpoint,
                stop_loss=0.0,
                take_profit=0.0,
                signal_type="LIMIT",
                ob_top=0.0,
                ob_bottom=0.0,
                fvg_top=0.0,
                fvg_bottom=0.0,
                confidence=1.0,
            )
            result = await place_multiplier_contract(
                signal=sig,
                stake=stake,
                stop_loss_amount=config.MULTIPLIER_SL_USD,
                take_profit_amount=config.MULTIPLIER_TP_USD,
            )
        else:
            result = await place_tick_contract(
                symbol=symbol,
                contract_type=direction,
                stake=stake,
                duration=config.TICK_DURATION,
            )

        if not result:
            log.warning(f"❌ Contract placement failed for {symbol}. Releasing lock.")
            _in_trades[symbol] = False
        else:
            _last_trade_time[symbol] = time.time()


# ─── Transaction Stream Handler ───────────────────────────────────────────────

async def on_transaction_event(tx: dict):
    """
    Callback for live transaction events from Deriv stream.
    Resets trading flags and updates balance when trades settle.
    """
    global _in_trades, _last_trade_time, _balance, _initial_balance

    action = tx.get("action")
    symbol = tx.get("symbol") or tx.get("underlying_symbol")
    contract_id = tx.get("contract_id")
    amount = float(tx.get("amount", 0) or 0)

    if not symbol or symbol not in config.SYMBOLS:
        return

    if action == "buy":
        log.info(f"🔔 Transaction Event: Trade Opened | {symbol} | ID: {contract_id}")
        _in_trades[symbol] = True
        _last_trade_time[symbol] = time.time()

    elif action == "sell":
        log.info(f"🔔 Transaction Event: Trade Settled | {symbol} | ID: {contract_id} | Payout: ${amount:.2f}")
        
        pnl = 0.0
        if config.USE_MULTIPLIERS:
            # Multipliers can settle with different SL/TP bounds, so we query the profit table
            outcome = await get_last_contract_result(symbol)
            if outcome and str(outcome.get("contract_id")) == str(contract_id):
                pnl = float(outcome.get("profit", 0))
            else:
                pnl = amount - config.TRADE_STAKE
        else:
            pnl = amount - config.TRADE_STAKE
            
        res_str = "WIN" if pnl > 0 else "LOSS"

        # Update balance
        try:
            api = get_api()
            resp = await api.send({"balance": 1})
            _balance = float(resp["balance"]["balance"])
        except Exception:
            _balance += pnl  # fallback addition

        log.info(f"💰 Outcome for {symbol}: {res_str} | PnL: ${pnl:.2f} | New Balance: ${_balance:.2f}")

        log_trade_outcome(
            symbol=symbol,
            contract_id=str(contract_id),
            pnl_usd=pnl,
            balance_after=_balance,
            result=res_str,
            notes=f"Payout: ${amount:.2f}"
        )

        _in_trades[symbol] = False
        _last_trade_time[symbol] = time.time()

        # Check daily limits
        if _initial_balance is not None:
            net_daily_pnl = _balance - _initial_balance
            log.info(f"📊 Daily PnL status: ${net_daily_pnl:.2f} (Target: +${config.DAILY_PROFIT_TARGET_USD:.2f} | Stop: -${config.DAILY_STOP_LOSS_USD:.2f})")
            if net_daily_pnl >= config.DAILY_PROFIT_TARGET_USD:
                log.info(f"🎉 Daily Profit Target reached! (+${net_daily_pnl:.2f} >= +${config.DAILY_PROFIT_TARGET_USD:.2f})")
                log.info("Stopping trading to lock in profits.")
                _shutdown_event.set()
            elif net_daily_pnl <= -config.DAILY_STOP_LOSS_USD:
                log.info(f"🛑 Daily Stop Loss hit! (-${-net_daily_pnl:.2f} <= -${config.DAILY_STOP_LOSS_USD:.2f})")
                log.info("Stopping trading to protect capital.")
                _shutdown_event.set()


# ─── Health Monitor ───────────────────────────────────────────────────────────

async def health_monitor():
    """Monitors connection health and reconnects automatically on dropout."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(config.HEALTH_CHECK_S)
            api = get_api()
            if not api or not await api.is_connected():
                log.warning("⚠️  Deriv API disconnected! Starting reconnection loop...")
                
                delay = 2.0
                reconnected = False
                while not _shutdown_event.is_set():
                    log.info(f"Attempting to reconnect in {delay:.1f}s...")
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(_shutdown_event.wait()),
                            timeout=delay
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        pass
                    
                    if await api.connect():
                        reconnected = True
                        break
                    delay = min(delay * 2, 60.0)
                
                if reconnected and not _shutdown_event.is_set():
                    log.info("✅ Reconnected to Deriv. Syncing portfolio state...")
                    await sync_portfolio_state()
                    
                    log.info("Re-subscribing to live feeds...")
                    for symbol in config.SYMBOLS:
                        try:
                            await api.subscribe_ticks(symbol, on_tick)
                        except Exception as e:
                            log.error(f"Failed to subscribe to ticks for {symbol}: {e}")
                    try:
                        tx_sub_id = await api.subscribe_transactions(on_transaction_event)
                        if tx_sub_id:
                            log.info(f"Re-subscribed to transactions | sub_id: {tx_sub_id}")
                    except Exception as e:
                        log.error(f"Failed to subscribe to transaction events: {e}")
                elif not reconnected:
                     log.error("❌ Reconnection loop stopped.")
        except Exception as e:
            log.error(f"Health monitor encountered an error: {e}", exc_info=True)
            await asyncio.sleep(10)


# ─── Scan Cycle (Heartbeat) ───────────────────────────────────────────────────

async def run_scan_cycle() -> bool:
    """
    Logs the heartbeat state and checks if targets or circuit breakers are hit.
    """
    global _balance
    
    try:
        api = get_api()
        resp = await api.send({"balance": 1})
        _balance = float(resp["balance"]["balance"])
    except Exception:
        pass

    if _balance >= config.ACCOUNT_TARGET_USD:
        log.info(f"🎯 TARGET REACHED! Balance ${_balance:.2f}. Bot stopping.")
        log_session_event("TARGET_REACHED", f"Final balance: ${_balance:.2f}")
        return False

    if _balance < config.CIRCUIT_BREAKER_USD:
        log.critical(f"🛑 CIRCUIT BREAKER TRIPPED! Balance ${_balance:.2f} < ${config.CIRCUIT_BREAKER_USD:.2f}")
        log_session_event("CIRCUIT_BREAKER", f"Final balance: ${_balance:.2f}")
        return False

    open_count = sum(1 for v in _in_trades.values() if v)
    progress = (_balance / config.ACCOUNT_TARGET_USD) * 100.0
    log.info(
        f"── Scan Heartbeat | Balance: ${_balance:.2f} | "
        f"Progress: {progress:.1f}% | Open trades: {open_count}/{config.MAX_CONCURRENT_TRADES} ──"
    )
    return True


# ─── Signal Handlers ──────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    log.warning(f"OS signal {sig} received — shutting down...")
    _shutdown_event.set()


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    global DRY_RUN

    # Check for dry run mode
    if "--dry-run" in sys.argv:
        DRY_RUN = True

    # PID Lock
    import os
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            if old_pid != os.getpid():
                if sys.platform == "win32":
                    import ctypes
                    exists = ctypes.windll.kernel32.GetProcessVersion(old_pid) > 0
                    if exists:
                        print(f"❌ ERROR: Bot is already running (PID {old_pid}). Exiting.")
                        sys.exit(1)
                    else:
                        raise OSError("Process not running")
                else:
                    os.kill(old_pid, 0)
                    print(f"❌ ERROR: Bot is already running (PID {old_pid}). Exiting.")
                    sys.exit(1)
        except (OSError, ValueError, SystemError):
            try:
                os.remove(PID_FILE)
            except OSError:
                pass

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    setup_logging()

    log.info("=" * 60)
    mode_str = "🧪 DEMO / VIRTUAL ACCOUNT" if config.DEMO_MODE else "🔴 LIVE ACCOUNT"
    if DRY_RUN:
        mode_str += " (DRY RUN MODE)"
    log.info(f"  Mode           : {mode_str}")
    log.info(f"  Symbols        : {', '.join(config.SYMBOLS)}")
    log.info(f"  Target         : ${config.ACCOUNT_TARGET_USD:.2f}")
    log.info(f"  Circuit Breaker: ${config.CIRCUIT_BREAKER_USD:.2f}")
    log.info(f"  Fixed Stake    : ${config.TRADE_STAKE:.2f}")
    log.info(f"  Tick Duration  : {config.TICK_DURATION} ticks")
    log.info(f"  Max Concurrent : {config.MAX_CONCURRENT_TRADES} trades")
    log.info("=" * 60)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    # ── Connect to Deriv ─────────────────────────────────────────────────────
    if not await connector.initialize():
        log.critical("Cannot connect to Deriv API. Check credentials. Exiting.")
        sys.exit(1)

    _init_symbol_state()
    log_session_event("BOT_START", f"Symbols: {config.SYMBOLS} (Dry Run: {DRY_RUN})")

    # ── Sync portfolio state ──────────────────────────────────────────────────
    await sync_portfolio_state()

    # ── Subscribe to live ticks ───────────────────────────────────────────────
    sub_ids = []
    for symbol in config.SYMBOLS:
        sid = await get_api().subscribe_ticks(symbol, on_tick)
        if sid:
            sub_ids.append(sid)
            log.info(f"Subscribed to ticks for {symbol}")

    # ── Subscribe to live transaction events ──────────────────────────────────
    try:
        tx_sub_id = await get_api().subscribe_transactions(on_transaction_event)
        if tx_sub_id:
            sub_ids.append(tx_sub_id)
            log.info(f"Subscribed to transactions")
    except Exception as e:
        log.error(f"Failed to subscribe to transaction events: {e}")

    # ── Start health monitor ──────────────────────────────────────────────────
    monitor_task = asyncio.create_task(health_monitor())

    # ── Main heartbeat loop ───────────────────────────────────────────────────
    try:
        while not _shutdown_event.is_set():
            should_continue = await run_scan_cycle()
            if not should_continue:
                break

            try:
                await asyncio.wait_for(
                    asyncio.shield(_shutdown_event.wait()),
                    timeout=config.HEALTH_CHECK_S,
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
            try:
                await get_api().unsubscribe(sid)
            except Exception:
                pass
        log_session_event("BOT_STOP")
        await connector.shutdown()
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        log.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
