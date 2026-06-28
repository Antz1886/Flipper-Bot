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


async def place_tick_contract(
    symbol: str,
    contract_type: str,  # "CALL" or "PUT"
    stake: float,
    duration: int = 5,
) -> dict | None:
    """
    Buy a Deriv Rise/Fall contract by requesting a proposal first, then buying it.
    """
    api = get_api()

    proposal_payload = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": duration,
        "duration_unit": "t",
        "underlying_symbol": symbol
    }

    async with _exec_lock:
        log.info(
            f"→ Requesting Proposal | {contract_type} | {symbol} | "
            f"Stake: ${stake:.2f} | Duration: {duration}t"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Double-check active contract status before execution
                open_status = await has_open_contract(symbol)
                if open_status is True:
                    log.warning(f"Aborting execution: Active contract already detected for {symbol}")
                    return None

                # 1. Fetch proposal
                prop_resp = await api.send(proposal_payload)
                if "error" in prop_resp:
                    err = prop_resp["error"]
                    log.warning(
                        f"Proposal request failed (attempt {attempt}/{MAX_RETRIES}): "
                        f"[{err.get('code')}] {err.get('message')}"
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S)
                    continue

                proposal = prop_resp.get("proposal", {})
                proposal_id = proposal.get("id")
                ask_price = proposal.get("ask_price")
                if not proposal_id:
                    log.warning(f"No proposal ID in response (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S)
                    continue

                # 2. Buy contract
                log.info(f"→ Purchasing contract using proposal {proposal_id}...")
                buy_payload = {
                    "buy": proposal_id,
                    "price": ask_price,
                    "subscribe": 1
                }
                resp = await api.send(buy_payload)

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
                    f"✅ Tick Contract opened | ID: {contract_id} | "
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
    
    proposal_payload = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "underlying_symbol": signal.symbol,
        "multiplier": multiplier,
        "limit_order": {
            "stop_loss": stop_loss_amount,
            "take_profit": take_profit_amount,
        }
    }

    async with _exec_lock:
        log.info(
            f"→ Requesting Proposal | {contract_type} | {signal.symbol} | "
            f"Stake: ${stake:.2f} | Multiplier: {multiplier}x | "
            f"SL: ${stop_loss_amount:.2f} | TP: ${take_profit_amount:.2f}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Double-check one last time before execution
                open_status = await has_open_contract(signal.symbol)
                if open_status is True:
                    log.warning(f"Aborting execution: Active contract already detected for {signal.symbol}")
                    return None

                # 1. Fetch proposal
                prop_resp = await api.send(proposal_payload)
                if "error" in prop_resp:
                    err = prop_resp["error"]
                    log.warning(
                        f"Proposal request failed (attempt {attempt}/{MAX_RETRIES}): "
                        f"[{err.get('code')}] {err.get('message')}"
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S)
                    continue
                
                proposal = prop_resp.get("proposal", {})
                proposal_id = proposal.get("id")
                ask_price = proposal.get("ask_price")
                if not proposal_id:
                    log.warning(f"No proposal ID in response (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S)
                    continue

                # 2. Buy contract
                log.info(f"→ Purchasing contract using proposal {proposal_id}...")
                buy_payload = {
                    "buy": proposal_id,
                    "price": ask_price,
                    "subscribe": 1
                }
                resp = await api.send(buy_payload)
                
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
            c_symbol = c.get("symbol") or c.get("underlying_symbol")
            if c_symbol != symbol:
                continue
            
            # Filter: only keep genuinely open contracts for our strategy
            contract_type = c.get("contract_type", "")
            if not any(x in contract_type for x in ("MULT", "CALL", "PUT")):
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
            t_symbol = t.get("symbol") or t.get("underlying_symbol")
            if t_symbol == symbol:
                return t
        return None
    except Exception as e:
        log.error(f"Failed to fetch profit table for {symbol}: {e}")
        return None


async def update_contract_limit_order(contract_id: int, stop_loss: float = None, take_profit: float = None) -> tuple[bool, str | None]:
    """Update stop loss and/or take profit for an open contract."""
    api = get_api()
    limit_order = {}
    if stop_loss is not None:
        limit_order["stop_loss"] = stop_loss
    if take_profit is not None:
        limit_order["take_profit"] = take_profit
        
    payload = {
        "contract_update": 1,
        "contract_id": contract_id,
        "limit_order": limit_order
    }
    
    try:
        resp = await api.send(payload)
        if "error" in resp:
            err = resp["error"]
            log.error(f"Failed to update contract {contract_id}: [{err.get('code')}] {err.get('message')}")
            return False, err.get("code")
        return True, None
    except Exception as e:
        log.error(f"Error updating contract {contract_id}: {e}")
        return False, "UnknownError"


async def get_contract_details(contract_id: int) -> dict | None:
    """Fetch complete details of an open contract using proposal_open_contract."""
    api = get_api()
    try:
        resp = await api.send({"proposal_open_contract": 1, "contract_id": contract_id})
        return resp.get("proposal_open_contract")
    except Exception as e:
        log.error(f"Failed to fetch details for contract {contract_id}: {e}")
        return None


