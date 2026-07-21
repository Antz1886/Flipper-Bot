"""
modules/order_executor.py — Deriv Multiplier Contract Executor
Places MULTUP (bullish) or MULTDOWN (bearish) multiplier contracts
with dollar-amount stop_loss and take_profit embedded in the initial buy request.
"""

import logging
import asyncio
import config
from modules.connector import get_api

log = logging.getLogger(__name__)

_exec_lock = asyncio.Lock()


async def place_multiplier_contract(
    symbol: str,
    direction: str,  # "LONG" or "SHORT"
    stake: float = config.TRADE_STAKE,
    stop_loss_usd: float = config.MULTIPLIER_SL_USD,
    take_profit_usd: float = config.MULTIPLIER_TP_USD,
) -> dict | None:
    """
    Buy a Deriv Multiplier contract (MULTUP for LONG, MULTDOWN for SHORT)
    with embedded stop_loss and take_profit.
    """
    api = get_api()
    if not api or not api.is_active:
        log.error("Order execution failed: Deriv API is not connected.")
        return None

    if not api.authorized:
        log.warning(f"⚠️ [{symbol} {direction} SIGNAL DETECTED] Order execution requires a valid DERIV_API_TOKEN in .env. Skipping execution.")
        return None

    contract_type = "MULTUP" if direction.upper() == "LONG" else "MULTDOWN"
    multiplier = config.SYMBOL_MULTIPLIERS.get(symbol, 100)

    proposal_payload = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": api.currency or "USD",
        "multiplier": multiplier,
        "underlying_symbol": symbol,
        "stop_loss": stop_loss_usd,
        "take_profit": take_profit_usd
    }

    async with _exec_lock:
        log.info(
            f"→ Requesting Multiplier Proposal | {contract_type} ({multiplier}x) | {symbol} | "
            f"Stake: ${stake:.2f} | SL: ${stop_loss_usd:.2f} | TP: ${take_profit_usd:.2f}"
        )

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                # 1. Fetch proposal
                prop_resp = await api.send(proposal_payload)
                if "error" in prop_resp:
                    err = prop_resp["error"]
                    log.warning(
                        f"Proposal request failed (attempt {attempt}/{config.MAX_RETRIES}): "
                        f"[{err.get('code')}] {err.get('message')}"
                    )
                    if attempt < config.MAX_RETRIES:
                        await asyncio.sleep(config.RETRY_DELAY_S)
                    continue

                proposal = prop_resp.get("proposal", {})
                proposal_id = proposal.get("id")
                ask_price = proposal.get("ask_price")

                if not proposal_id:
                    log.warning(f"No proposal ID returned for {symbol}.")
                    continue

                # 2. Buy contract using proposal_id
                buy_resp = await api.send({"buy": proposal_id, "price": ask_price})
                if "error" in buy_resp:
                    err = buy_resp["error"]
                    log.error(f"Buy execution failed for {symbol}: [{err.get('code')}] {err.get('message')}")
                    return None

                buy_info = buy_resp.get("buy", {})
                contract_id = buy_info.get("contract_id")
                purchase_price = buy_info.get("buy_price", ask_price)

                log.info(
                    f"✅ Deriv Multiplier {contract_type} Purchased! | {symbol} | "
                    f"Contract ID: {contract_id} | Price: ${purchase_price:.2f}"
                )

                return {
                    "contract_id": contract_id,
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "purchase_price": purchase_price,
                    "stake": stake,
                    "multiplier": multiplier,
                    "stop_loss": stop_loss_usd,
                    "take_profit": take_profit_usd
                }

            except Exception as e:
                log.error(f"Exception executing multiplier order for {symbol}: {e}")
                if attempt < config.MAX_RETRIES:
                    await asyncio.sleep(config.RETRY_DELAY_S)

        log.error(f"Failed to place multiplier contract for {symbol} after {config.MAX_RETRIES} attempts.")
        return None
