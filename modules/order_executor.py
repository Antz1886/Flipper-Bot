"""
modules/order_executor.py — Deriv Multiplier Contract Executor
Places MULTUP (bullish) or MULTDOWN (bearish) multiplier contracts
with dollar-amount stop_loss and take_profit mathematically calibrated
to the SMC structural stop-loss price.
"""

import logging
import asyncio
import config
from modules.connector import get_api
from modules.risk_manager import can_open_symbol, mark_symbol_active

log = logging.getLogger(__name__)

_exec_lock = asyncio.Lock()


async def place_multiplier_contract(
    symbol: str,
    direction: str,  # "LONG" or "SHORT"
    stake: float,
    stop_loss_usd: float,
    take_profit_usd: float,
) -> dict | None:
    """
    Buy a Deriv Multiplier contract (MULTUP for LONG, MULTDOWN for SHORT)
    with embedded stop_loss and take_profit based on structural sizing.
    Engages the symbol lock ONLY after confirmed exchange purchase.
    """
    api = get_api()
    if not api or not api.is_active:
        log.error("Order execution failed: Deriv API is not connected.")
        return None

    if not api.authorized:
        log.warning(
            f"⚠️ [{symbol} {direction} SIGNAL DETECTED] Order execution requires a valid DERIV_API_TOKEN in .env. "
            f"Stake: ${stake:.2f} | SL: ${stop_loss_usd:.2f} | TP: ${take_profit_usd:.2f}. Skipping execution."
        )
        return None

    contract_type = "MULTUP" if direction.upper() == "LONG" else "MULTDOWN"
    multiplier = config.SYMBOL_MULTIPLIERS.get(symbol, 100)

    # Deriv OTP WebSocket (/trading/v1/options/ws/...) uses "underlying_symbol",
    # whereas Deriv v3 WebSocket (/websockets/v3) uses "symbol".
    symbol_key = "underlying_symbol" if getattr(api, "using_otp_ws", False) else "symbol"

    proposal_payload = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": api.currency or "USD",
        "multiplier": multiplier,
        symbol_key: symbol,
        "limit_order": {
            "stop_loss": stop_loss_usd,
            "take_profit": take_profit_usd,
        },
    }

    async with _exec_lock:
        if not can_open_symbol(symbol):
            log.info(f"⏭️ Skipping {symbol}: position limit or symbol lock already active.")
            return None

        log.info(
            f"→ Routing Institutional Order | {contract_type} ({multiplier}x) | {symbol} | "
            f"Stake: ${stake:.2f} | Structural SL: ${stop_loss_usd:.2f} | Structural TP: ${take_profit_usd:.2f}"
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
                purchase_price = float(buy_info.get("buy_price", ask_price or stake))

                # Engage lock ONLY after confirmed fill
                mark_symbol_active(symbol, stake_usd=purchase_price)

                log.info(
                    f"✅ Institutional {contract_type} Executed! | {symbol} | "
                    f"Contract ID: {contract_id} | Stake Locked: ${purchase_price:.2f} | "
                    f"SL: ${stop_loss_usd:.2f} | TP: ${take_profit_usd:.2f}"
                )

                return {
                    "contract_id": contract_id,
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "purchase_price": purchase_price,
                    "stake": stake,
                    "multiplier": multiplier,
                    "stop_loss_usd": stop_loss_usd,
                    "take_profit_usd": take_profit_usd,
                }

            except Exception as e:
                log.warning(f"Order execution attempt {attempt}/{config.MAX_RETRIES} error for {symbol}: {e}")
                if attempt < config.MAX_RETRIES:
                    await asyncio.sleep(config.RETRY_DELAY_S)

    log.error(f"❌ Failed to execute {direction} multiplier order for {symbol} after {config.MAX_RETRIES} attempts.")
    return None
