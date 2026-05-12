"""
modules/order_executor.py — Deriv Multiplier Contract Executor
Places MULTUP (bullish) or MULTDOWN (bearish) multiplier contracts
with dollar-amount stop_loss and take_profit embedded in the
initial buy request. No post-trade modification needed.
"""

import logging

from modules.connector import get_api
from modules.smc_engine import TradeSignal
from config import SYMBOL_MULTIPLIERS, MAGIC_COMMENT, MAX_RETRIES, RETRY_DELAY_S
import asyncio

log = logging.getLogger(__name__)

# Execution lock to prevent concurrent trade placements for the same session
_exec_lock = asyncio.Lock()


async def place_multiplier_contract(
    signal:             TradeSignal,
    stake:              float,
    stop_loss_amount:   float,
    take_profit_amount: float,
) -> dict | None:
    """
    Buy a Deriv multiplier contract with embedded SL and TP.

    Parameters
    ----------
    signal             : TradeSignal with direction ("BUY" or "SELL")
    stake              : Dollar amount to stake (Kelly-calculated)
    stop_loss_amount   : Max loss in USD before auto-close
    take_profit_amount : Target profit in USD for auto-close

    Returns
    -------
    Deriv buy response dict on success, None on failure.
    """
    api = get_api()

    contract_type = "MULTUP" if signal.direction == "BUY" else "MULTDOWN"

    multiplier = SYMBOL_MULTIPLIERS.get(signal.symbol)
    if multiplier is None:
        log.error(f"No valid multiplier for {signal.symbol} — skipping trade.")
        return None
    
    payload = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount":        stake,
            "contract_type": contract_type,
            "symbol":        signal.symbol,
            "multiplier":    multiplier,
            "basis":         "stake",
            "currency":      "USD",
            "limit_order": {
                "stop_loss":   stop_loss_amount,
                "take_profit": take_profit_amount,
            },
        },
    }

    async with _exec_lock:
        log.info(
            f"→ Executing MARKET BUY | {contract_type} | {signal.symbol} | "
            f"Stake: ${stake:.2f} | Multiplier: {multiplier}x | "
            f"SL: ${stop_loss_amount:.2f} | TP: ${take_profit_amount:.2f}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Double-check one last time before sending the buy payload
                open_status = await has_open_contract(signal.symbol)
                if open_status is True:
                    log.warning(f"Aborting execution: Active contract already detected for {signal.symbol}")
                    return None

                resp = await api.send(payload)
                
                # Check for API errors in the response
                if "error" in resp:
                    err = resp["error"]
                    log.warning(
                        f"Contract buy error (attempt {attempt}/{MAX_RETRIES}): "
                        f"[{err.get('code')}] {err.get('message')}"
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S)
                    continue
                
                buy_resp = resp.get("buy", {})
                contract_id  = buy_resp.get("contract_id")
                purchase_price = buy_resp.get("buy_price")

                log.info(
                    f"✅ Contract opened | ID: {contract_id} | "
                    f"Buy price: ${purchase_price}"
                )
                return resp

            except Exception as e:
                log.warning(
                    f"Contract buy failed (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S)

    log.error(f"❌ Failed to open contract after {MAX_RETRIES} attempts.")
    return None


async def get_open_contracts(symbol: str) -> list[dict] | None:
    """
    Return only genuinely OPEN multiplier contracts for the given symbol.
    
    Returns None on connection error (caller must treat as "unknown" state).
    Returns [] if connected and no open contracts exist.
    """
    api = get_api()
    try:
        resp = await api.send({"portfolio": 1})
        contracts = resp.get("portfolio", {}).get("contracts", [])
        
        open_contracts = []
        for c in contracts:
            if c.get("symbol") != symbol:
                continue
            
            # Filter: only keep genuinely open multiplier contracts
            contract_type = c.get("contract_type", "")
            if "MULT" not in contract_type:
                continue
            
            # If the contract has a sell_time, it's already closed/settled
            if c.get("sell_time"):
                log.debug(
                    f"Skipping settled contract {c.get('contract_id')} for {symbol} "
                    f"(sell_time: {c.get('sell_time')})"
                )
                continue
            
            # If the contract has an expiry_time in the past, skip it
            if c.get("expiry_time"):
                import time
                if c["expiry_time"] < time.time():
                    log.debug(
                        f"Skipping expired contract {c.get('contract_id')} for {symbol}"
                    )
                    continue
            
            open_contracts.append(c)
            log.debug(
                f"Found open contract for {symbol}: "
                f"ID={c.get('contract_id')} type={contract_type} "
                f"buy_price={c.get('buy_price')}"
            )
        
        return open_contracts
        
    except Exception as e:
        log.error(f"Failed to fetch portfolio: {e}")
        return None  # None = unknown state (don't change trade flags)


async def has_open_contract(symbol: str) -> bool | None:
    """
    True if any open contract exists for the symbol.
    Returns None if we can't reliably determine (e.g. disconnect).
    """
    contracts = await get_open_contracts(symbol)
    if contracts is None:
        return None  # Unknown — caller must not change state
    return len(contracts) > 0


async def close_contract(contract_id: int) -> bool:
    """Sell (close) an open contract at market price."""
    api = get_api()
    try:
        resp = await api.send({"sell": contract_id, "price": 0})
        sold = resp.get("sell", {})
        log.info(
            f"Contract {contract_id} closed | "
            f"Sold price: ${sold.get('sold_for', '?')}"
        )
        return True
    except Exception as e:
        log.error(f"Failed to close contract {contract_id}: {e}")
        return False


async def get_last_contract_result(symbol: str) -> dict | None:
    """
    Fetch the most recently closed contract result for a symbol from profit_table.
    Used to log outcome (PnL) after a trade closes.
    """
    api = get_api()
    try:
        # Fetch the last 10 entries to be safe
        resp = await api.send({
            "profit_table": 1,
            "description": 1,
            "limit": 10,
            "sort": "DESC"
        })
        transactions = resp.get("profit_table", {}).get("transactions", [])
        for t in transactions:
            if t.get("symbol") == symbol:
                return t
        return None
    except Exception as e:
        log.error(f"Failed to fetch profit table for {symbol}: {e}")
        return None
