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
    get_open_contracts,
    get_last_contract_result,
    update_contract_limit_order,
    get_contract_details,
)
from modules.logger import setup_logging, log_trade, log_trade_outcome, log_session_event

log = logging.getLogger(__name__)

# ─── Bot State ────────────────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()
_portfolio_synced = False

@dataclass
class ActiveZone:
    """Holds the current unexecuted SMC signal waiting for price to enter."""
    signal:             TradeSignal
    stake:              float
    stop_loss_amount:   float
    take_profit_amount: float
    balance_snapshot:   float
    created_at:         float = 0.0   # time.time() when zone was armed
    tapped:             bool = False  # True when price enters OB zone
    extreme_price:      float = 0.0   # Tracks lowest ask (BUY) or highest bid (SELL) inside OB

@dataclass
class ActiveTrade:
    """Holds details of a running multiplier contract."""
    contract_id:        int
    entry_price:        float
    direction:          str
    stop_loss_amount:   float
    take_profit_amount: float
    stake:              float
    is_break_even:      bool = False

# Dicts to track state per symbol
_active_zones: Dict[str, Optional[ActiveZone]] = {}
_active_trades: Dict[str, Optional[ActiveTrade]] = {}
_in_trades:    Dict[str, bool] = {}
_last_trade_time: Dict[str, float] = {}
_tick_locks:   Dict[str, asyncio.Lock] = {}
_scan_lock:    asyncio.Lock = None  # Will be initialized in main() or defined here

PID_FILE = "bot.pid"


def _init_symbol_state():
    """Initialize per-symbol state dicts. Called after SYMBOLS is finalized."""
    global _active_zones, _active_trades, _in_trades, _last_trade_time, _tick_locks, _scan_lock
    _active_zones    = {s: None for s in config.SYMBOLS}
    _active_trades   = {s: None for s in config.SYMBOLS}
    _in_trades       = {s: False for s in config.SYMBOLS}
    _last_trade_time = {s: 0.0 for s in config.SYMBOLS}
    _tick_locks      = {s: asyncio.Lock() for s in config.SYMBOLS}
    _scan_lock       = asyncio.Lock()


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
    Called once at startup (and upon reconnection) to sync _in_trades with actual Deriv portfolio.
    Prevents opening duplicate positions after a restart.
    """
    global _portfolio_synced
    _portfolio_synced = False

    log.info("📋 Syncing portfolio state with Deriv...")
    for symbol in config.SYMBOLS:
        synced_symbol = False
        for attempt in range(1, 4):
            status = await has_open_contract(symbol)
            if status is True:
                _in_trades[symbol] = True
                _last_trade_time[symbol] = time.time()
                log.info(f"📋 Portfolio sync: Active contract found for {symbol} — marking in-trade.")
                
                # Fetch details to populate _active_trades
                contracts = await get_open_contracts(symbol)
                if contracts:
                    cid = contracts[0].get("contract_id")
                    details = await get_contract_details(cid)
                    if details:
                        entry = float(details.get("entry_spot", 0) or 0)
                        stake = float(details.get("buy_price", 0) or 0)
                        limit_order = details.get("limit_order", {})
                        sl_amt = float(limit_order.get("stop_loss", {}).get("order_amount", 0) or 0)
                        tp_amt = float(limit_order.get("take_profit", {}).get("order_amount", 0) or 0)
                        
                        _active_trades[symbol] = ActiveTrade(
                            contract_id=cid,
                            entry_price=entry,
                            direction="BUY" if "MULTUP" in details.get("contract_type", "") else "SELL",
                            stop_loss_amount=sl_amt,
                            take_profit_amount=tp_amt,
                            stake=stake,
                            is_break_even=False
                        )
                        log.info(f"📋 Portfolio sync: Loaded trade details for {symbol} (ID: {cid}, Entry: {entry})")
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
            log.critical(f"❌ Portfolio sync failed for {symbol}. Bot may open duplicate trades!")
            # Safety: assume in-trade if we can't be sure
            _in_trades[symbol] = True

    _portfolio_synced = True


# ─── Tick Handler ─────────────────────────────────────────────────────────────

async def on_tick(tick: dict):
    """
    Called on every live tick from the Deriv subscription.
    Checks if price has entered the active OB zone for that specific symbol.
    Uses per-symbol locks to prevent duplicate entries.
    Also handles Break-Even Stop Loss protection checks.
    """
    global _active_zones, _active_trades, _in_trades

    if not _portfolio_synced:
        return

    symbol = tick.get("symbol")
    if not symbol:
        return

    # ─── Break-Even Stop Loss Protection Check ──────────────────────────────────
    trade = _active_trades.get(symbol)
    if trade and not trade.is_break_even:
        bid = float(tick.get("bid", 0))
        ask = float(tick.get("ask", 0))
        current_price = ask if trade.direction == "BUY" else bid
        
        # Calculate price change percentage
        if trade.direction == "BUY":
            change_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            change_pct = (trade.entry_price - current_price) / trade.entry_price
            
        multiplier = config.SYMBOL_MULTIPLIERS.get(symbol, 50)
        unrealized_profit = trade.stake * multiplier * change_pct
        
        # Target for break-even: if profit reaches 50% of take_profit_amount
        trigger_profit = trade.take_profit_amount * 0.5
        if unrealized_profit >= trigger_profit:
            log.info(f"🛡️ Break-Even Triggered for {symbol} | Profit: ${unrealized_profit:.2f} >= Trigger: ${trigger_profit:.2f}")
            
            # Send contract update to set stop loss to 0 (break-even)
            async def run_update(cid, sym):
                success, err_code = await update_contract_limit_order(cid, stop_loss=0.0)
                if success:
                    t = _active_trades.get(sym)
                    if t and t.contract_id == cid:
                        t.is_break_even = True
                        log.info(f"🛡️ Break-Even Stop successfully set for contract {cid} on {sym}")
                elif err_code in ("ContractAlreadySold", "InvalidContractProposal"):
                    log.warning(f"🛡️ Contract {cid} on {sym} already closed ({err_code}) — clearing active trade state.")
                    _active_trades[sym] = None
                    _in_trades[sym] = False
            asyncio.create_task(run_update(trade.contract_id, symbol))

    if symbol not in _active_zones:
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
        zone = _active_zones[symbol]
        sig = zone.signal

        zone_triggered = False

        if config.USE_REJECTION_FILTER:
            if not zone.tapped:
                # Check if price has entered the OB zone
                is_inside = False
                if sig.direction == "BUY" and ask <= sig.ob_top and ask >= sig.ob_bottom:
                    is_inside = True
                    zone.extreme_price = ask
                elif sig.direction == "SELL" and bid >= sig.ob_bottom and bid <= sig.ob_top:
                    is_inside = True
                    zone.extreme_price = bid

                if is_inside:
                    zone.tapped = True
                    log.info(
                        f"🎯 OB Zone Tapped | {symbol} {sig.direction} | "
                        f"OB range: [{sig.ob_bottom:.5f}–{sig.ob_top:.5f}] | "
                        f"Entered at: {ask if sig.direction == 'BUY' else bid:.5f} | Rejection tracking active."
                    )
                return  # Wait for subsequent ticks to confirm rejection
            else:
                # Zone is already tapped, check for invalidation (price went past SL)
                invalidated = False
                if sig.direction == "BUY" and ask < sig.stop_loss:
                    invalidated = True
                elif sig.direction == "SELL" and bid > sig.stop_loss:
                    invalidated = True

                if invalidated:
                    log.info(
                        f"❌ OB Zone Invalidated | {symbol} {sig.direction} | "
                        f"Price went past SL ({sig.stop_loss:.5f}) | "
                        f"Tick bid/ask: {bid:.5f}/{ask:.5f} | Clearing zone."
                    )
                    _active_zones[symbol] = None
                    return

                # Update extreme price
                if sig.direction == "BUY":
                    zone.extreme_price = min(zone.extreme_price, ask)
                else:
                    zone.extreme_price = max(zone.extreme_price, bid)

                # Calculate rejection bounce distance
                ob_height = abs(sig.ob_top - sig.ob_bottom)
                spread = ask - bid
                rejection_trigger = max(ob_height * config.REJECTION_TRIGGER_PCT, spread * 1.5)

                if sig.direction == "BUY" and ask >= zone.extreme_price + rejection_trigger:
                    log.info(
                        f"⚡ OB Zone Rejection Confirmed! | {symbol} BUY | "
                        f"Lowest Ask: {zone.extreme_price:.5f} | "
                        f"Current Ask: {ask:.5f} >= Trigger: {zone.extreme_price + rejection_trigger:.5f} | "
                        f"Executing Trade!"
                    )
                    zone_triggered = True
                elif sig.direction == "SELL" and bid <= zone.extreme_price - rejection_trigger:
                    log.info(
                        f"⚡ OB Zone Rejection Confirmed! | {symbol} SELL | "
                        f"Highest Bid: {zone.extreme_price:.5f} | "
                        f"Current Bid: {bid:.5f} <= Trigger: {zone.extreme_price - rejection_trigger:.5f} | "
                        f"Executing Trade!"
                    )
                    zone_triggered = True
        else:
            # Old limit-style entry
            zone_triggered = (
                (sig.direction == "BUY"  and ask <= sig.entry_price) or
                (sig.direction == "SELL" and bid >= sig.entry_price)
            )

        if not zone_triggered:
            return

        log.info(
            f"🎯 OB Zone Entry Triggered | {symbol} | {sig.direction} | Entry: {sig.entry_price:.5f} | "
            f"Tick bid/ask: {bid:.5f}/{ask:.5f}"
        )

        # Set trade flag BEFORE executing — atomic dedup
        _in_trades[symbol] = True
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
            # Populate active trade details
            buy_resp = result.get("buy", {})
            contract_id = buy_resp.get("contract_id")
            _active_trades[symbol] = ActiveTrade(
                contract_id=contract_id,
                entry_price=sig.entry_price,
                direction=sig.direction,
                stop_loss_amount=zone.stop_loss_amount,
                take_profit_amount=zone.take_profit_amount,
                stake=zone.stake,
                is_break_even=False
            )


# ─── Scan Cycle ───────────────────────────────────────────────────────────────

async def run_scan_cycle() -> bool:
    """
    One full scan across all symbols (wrapped in _scan_lock).
    """
    if _scan_lock.locked():
        log.debug("Scan cycle already in progress — skipping concurrent execution.")
        return True
    async with _scan_lock:
        return await _run_scan_cycle_inner()


async def _run_scan_cycle_inner() -> bool:
    """
    Inner scan cycle logic.
    """
    global _active_zones, _in_trades

    if not _portfolio_synced:
        log.warning("Scan cycle skipped: portfolio not synced.")
        return True

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
                _active_trades[symbol] = None
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
        htf_df        = await fetch_ohlc(symbol, config.HTF_TF_S)

        if primary_df is None or confluence_df is None or htf_df is None:
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
            htf_ohlc=htf_df,
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

        # ── Calculate Position Size ───────────────────────────────────────────
        try:
            multiplier = config.SYMBOL_MULTIPLIERS.get(symbol, 50)
            sizing = calculate_stake(balance, signal, multiplier)
        except CircuitBreakerTripped as e:
            log.critical(f"🛑 CIRCUIT BREAKER: {e}")
            log_session_event("CIRCUIT_BREAKER", str(e))
            log.info("Pausing for 5 minutes...")
            await asyncio.sleep(300)
            return True
            
        if sizing is None:
            log.debug(f"Sizing rejected trade for {symbol} (balance low or leverage too high).")
            continue
            
        stake, sl_amount, tp_amount = sizing

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
                log.warning("⚠️  Deriv API disconnected! Starting reconnection loop...")
                
                # Start exponential backoff reconnection loop
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
                    log.info("✅ Reconnected to Deriv. Syncing portfolio state first...")
                    await sync_portfolio_state()
                    
                    log.info("Re-subscribing to ticks...")
                    for symbol in config.SYMBOLS:
                        try:
                            await api.subscribe_ticks(symbol, on_tick)
                        except Exception as e:
                            log.error(f"Failed to re-subscribe to {symbol}: {e}")
                    try:
                        tx_sub_id = await api.subscribe_transactions(on_transaction_event)
                        if tx_sub_id:
                            log.info(f"Re-subscribed to transactions | sub_id: {tx_sub_id}")
                    except Exception as e:
                        log.error(f"Failed to re-subscribe to transaction events: {e}")
                elif not reconnected:
                    log.error("❌ Reconnection loop stopped.")
        except Exception as e:
            log.error(f"Health monitor encountered an error: {e}", exc_info=True)
            await asyncio.sleep(10)  # Brief pause before retrying loop


# ─── Transaction Stream Handler ───────────────────────────────────────────────

async def on_transaction_event(tx: dict):
    """
    Callback for live transaction events from Deriv stream.
    Updates bot state instantly when trades open or settle.
    """
    global _active_trades, _in_trades, _active_zones
    
    action = tx.get("action")
    symbol = tx.get("symbol")
    contract_id = tx.get("contract_id")
    amount = float(tx.get("amount", 0) or 0)
    
    if not symbol or symbol not in config.SYMBOLS:
        return
        
    if action == "buy":
        log.info(f"🔔 Transaction Event: Trade Opened | {symbol} | ID: {contract_id}")
        _in_trades[symbol] = True
        _last_trade_time[symbol] = time.time()
        
        # Details will be populated by the tick handler result once placed,
        # but as a fallback, we can fetch it if needed.
        if _active_trades.get(symbol) is None:
            async def load_details(cid, sym):
                details = await get_contract_details(cid)
                if details:
                    entry = float(details.get("entry_spot", 0) or 0)
                    stake = float(details.get("buy_price", 0) or 0)
                    limit_order = details.get("limit_order", {})
                    sl_amt = float(limit_order.get("stop_loss", {}).get("order_amount", 0) or 0)
                    tp_amt = float(limit_order.get("take_profit", {}).get("order_amount", 0) or 0)
                    _active_trades[sym] = ActiveTrade(
                        contract_id=cid,
                        entry_price=entry,
                        direction="BUY" if "MULTUP" in details.get("contract_type", "") else "SELL",
                        stop_loss_amount=sl_amt,
                        take_profit_amount=tp_amt,
                        stake=stake,
                        is_break_even=False
                    )
                    log.info(f"🔔 Transaction Event: Sync'd details for contract {cid} on {sym}")
            asyncio.create_task(load_details(contract_id, symbol))

    elif action == "sell":
        log.info(f"🔔 Transaction Event: Trade Settled | {symbol} | ID: {contract_id} | Payout: ${amount:.2f}")
        _in_trades[symbol] = False
        _active_trades[symbol] = None
        _active_zones[symbol] = None
        
        # Trigger immediate scan cycle
        log.info("🔔 Transaction Event: Trade cleared — triggering immediate scan cycle.")
        asyncio.create_task(run_scan_cycle())


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
            # Check if process is actually running (and is not us)
            if old_pid != os.getpid():
                os.kill(old_pid, 0)
                print(f"❌ ERROR: Bot is already running (PID {old_pid}). Exiting.")
                sys.exit(1)
        except (OSError, ValueError):
            # Process not running or stale pid file
            try:
                os.remove(PID_FILE)
            except OSError:
                pass

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    setup_logging()

    log.info("=" * 60)
    mode_str = "🧪 DEMO / VIRTUAL ACCOUNT" if config.DEMO_MODE else "🔴 LIVE ACCOUNT"
    log.info(f"  Mode      : {mode_str}")
    log.info(f"  Symbols   : {', '.join(config.SYMBOLS)}")
    log.info(f"  Target    : ${config.ACCOUNT_TARGET_USD:.2f}")
    log.info(f"  Target Risk: {config.KELLY_FRACTION * 100:.0f}% of balance (Quant Sizing)")
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

    # ── Subscribe to live transaction events ──────────────────────────────────
    try:
        tx_sub_id = await get_api().subscribe_transactions(on_transaction_event)
        if tx_sub_id:
            sub_ids.append(tx_sub_id)
    except Exception as e:
        log.error(f"Failed to subscribe to transaction events: {e}")

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
