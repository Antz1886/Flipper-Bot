"""
modules/order_executor.py — Deriv Multiplier Contract Executor
Places MULTUP (bullish) or MULTDOWN (bearish) multiplier contracts
with dollar-amount stop_loss and take_profit embedded in the
initial buy request. No post-trade modification needed.
"""

import logging

from modules.connector import get_api
from modules.smc_engine import TradeSignal
from config import MULTIPLIER, MAGIC_COMMENT, MAX_RETRIES, RETRY_DELAY_S
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

    payload = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount":        stake,
            "contract_type": contract_type,
            "symbol":        signal.symbol,
            "multiplier":    MULTIPLIER,
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
            f"Stake: ${stake:.2f} | SL: ${stop_loss_amount:.2f} | "
            f"TP: ${take_profit_amount:.2f}"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Double-check one last time before sending the buy payload
                if await has_open_contract(signal.symbol):
                    log.warning(f"Aborting execution: Active contract already detected for {signal.symbol}")
                    return None

                resp = await api.send(payload)
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


async def get_open_contracts(symbol: str) -> list[dict]:
    """Return all open multiplier contracts for the given symbol."""
    api = get_api()
    try:
        resp = await api.send({"portfolio": 1})
        contracts = resp.get("portfolio", {}).get("contracts", [])
        return [c for c in contracts if c.get("symbol") == symbol]
    except Exception as e:
        log.error(f"Failed to fetch portfolio: {e}")
        return []


async def has_open_contract(symbol: str) -> bool:
    """True if any open contract exists for the symbol."""
    return len(await get_open_contracts(symbol)) > 0


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
