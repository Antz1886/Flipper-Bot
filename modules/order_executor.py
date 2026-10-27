"""
modules/order_executor.py — Deriv Multiplier Contract Executor
Places MULTUP (bullish) or MULTDOWN (bearish) multiplier contracts with:
  1. Execution Spread Widening Filter: Discards orders if live bid-ask spread
     exceeds baseline threshold (e.g. > 1.5 pips on EURUSD).
  2. Deal Cancellation Premium Bleed Protection: Rejects execution if the broker's
     cancellation fee exceeds 15% of the trade's 1% risk budget.
  3. Strict SLA Expiry Tracking: Captures exact broker `date_expiry` timestamp
     for the sub-second cancellation dispatch monitor.
"""

import logging
import asyncio
import config
from modules.connector import get_api
from modules.risk_manager import can_open_symbol, mark_symbol_active

log = logging.getLogger(__name__)

_exec_lock = asyncio.Lock()


async def check_spread_safe(symbol: str) -> tuple[bool, float, float]:
    """
    Check if current live bid-ask spread is within baseline institutional tolerance.
    Returns: (is_safe, live_spread_pips, max_allowed_pips)
    """
    if not getattr(config, "ENFORCE_SPREAD_FILTER", True):
        return True, 0.0, 999.0

    api = get_api()
    if not api or not api.is_active:
        return True, 0.0, 999.0

    try:
        resp = await api.send({"ticks": symbol})
        tick = resp.get("tick", {})
        ask = tick.get("ask")
        bid = tick.get("bid")

        if ask is not None and bid is not None and ask > bid:
            pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
            spread_pips = round((ask - bid) / pip_size, 2)
            max_spreads = getattr(config, "MAX_ALLOWED_SPREAD_PIPS", {})
            max_allowed = max_spreads.get(symbol, max_spreads.get("DEFAULT", 2.0))

            if spread_pips > max_allowed:
                log.warning(
                    f"🛑 [SPREAD WIDENING FILTER] {symbol}: Live spread ({spread_pips:.2f} pips) "
                    f"exceeds {max_allowed:.1f} pip threshold. Discarding execution to prevent slippage bleed."
                )
                return False, spread_pips, max_allowed

            return True, spread_pips, max_allowed

    except Exception as e:
        log.warning(f"Could not verify live spread for {symbol}: {e}")

    return True, 0.0, 999.0


async def place_multiplier_contract(
    symbol: str,
    direction: str,  # "LONG" or "SHORT"
    stake: float,
    stop_loss_usd: float,
    take_profit_usd: float,
    ob_top: float = 0.0,
    ob_bottom: float = 0.0,
) -> dict | None:
    """
    Buy a Deriv Multiplier contract with spread gating, premium bleed protection,
    and Deriv Deal Cancellation SLA tracking.
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

    # Micro-Mechanic Fix 3: Execution Spread Widening Filter
    spread_ok, live_spread, max_spread = await check_spread_safe(symbol)
    if not spread_ok:
        return None

    contract_type = "MULTUP" if direction.upper() == "LONG" else "MULTDOWN"
    multiplier = config.SYMBOL_MULTIPLIERS.get(symbol, 100)

    symbol_key = "underlying_symbol" if getattr(api, "using_otp_ws", False) else "symbol"
    use_cancellation = getattr(config, "USE_DEAL_CANCELLATION", True)
    cancel_duration = getattr(config, "DEAL_CANCELLATION_DURATION", "15m")

    # In Deriv Multipliers API: If cancellation is used, limit_order must NOT be sent.
    # If cancellation is not used, limit_order contains stop_loss and take_profit.
    proposal_payload = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": api.currency or "USD",
        "multiplier": multiplier,
        symbol_key: symbol,
    }

    if use_cancellation:
        proposal_payload["cancellation"] = cancel_duration
    else:
        proposal_payload["limit_order"] = {
            "stop_loss": stop_loss_usd,
            "take_profit": take_profit_usd,
        }

    async with _exec_lock:
        if not can_open_symbol(symbol, direction=direction):
            log.info(f"⏭️ Skipping {symbol} {direction}: position limit, correlation trap, or lock active.")
            return None

        log.info(
            f"→ Routing Institutional Order | {contract_type} ({multiplier}x) | {symbol} | "
            f"Stake: ${stake:.2f} | Spread: {live_spread:.2f} pips | "
            f"{'Deal Cancellation (' + cancel_duration + ')' if use_cancellation else 'Structural SL: $' + str(stop_loss_usd)}"
        )

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                # 1. Query proposal with robust broker fallback for cancellation
                try:
                    prop_resp = await api.send(proposal_payload)
                except RuntimeError as re:
                    re_str = str(re).lower()
                    if ("cancellation" in re_str or "contractbuyvalidationerror" in re_str) and "cancellation" in proposal_payload:
                        log.warning(
                            f"Deal cancellation rejected by Deriv for {symbol} ({re}). "
                            f"Falling back to structural stop-loss (${stop_loss_usd:.2f}) & take-profit (${take_profit_usd:.2f})."
                        )
                        proposal_payload.pop("cancellation", None)
                        proposal_payload["limit_order"] = {
                            "stop_loss": stop_loss_usd,
                            "take_profit": take_profit_usd,
                        }
                        prop_resp = await api.send(proposal_payload)
                    else:
                        raise re

                # Fallback if broker rejects cancellation in dict response
                if isinstance(prop_resp, dict) and "error" in prop_resp and "cancellation" in proposal_payload:
                    err_msg = prop_resp["error"].get("message", "")
                    log.warning(
                        f"Deal cancellation unavailable for {symbol} ({err_msg}). "
                        f"Falling back to structural stop-loss & take-profit."
                    )
                    proposal_payload.pop("cancellation", None)
                    proposal_payload["limit_order"] = {
                        "stop_loss": stop_loss_usd,
                        "take_profit": take_profit_usd,
                    }
                    prop_resp = await api.send(proposal_payload)

                if isinstance(prop_resp, dict) and "error" in prop_resp:
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

                # Micro-Mechanic Fix 2: Deal Cancellation Premium Bleed Check
                cancellation_info = proposal.get("cancellation")
                cancellation_fee = 0.0
                date_expiry = None

                if cancellation_info and "cancellation" in proposal_payload:
                    cancellation_fee = float(cancellation_info.get("ask_price", 0.0))
                    if cancellation_fee <= 0.0:
                        cancellation_fee = max(0.0, float(ask_price or stake) - stake)
                    date_expiry = cancellation_info.get("date_expiry")

                    max_bleed_pct = getattr(config, "MAX_CANCELLATION_FEE_PCT_OF_RISK", 0.15)
                    min_allowed_fee = 0.35 if getattr(config, "FIXED_STAKE_USD", 0) > 0 else 0.0
                    max_allowed_fee = max(min_allowed_fee, round(stop_loss_usd * max_bleed_pct, 2))

                    if cancellation_fee > max_allowed_fee:
                        log.warning(
                            f"🛑 [CANCELLATION PREMIUM BLEED] {symbol}: Cancellation fee (${cancellation_fee:.2f}) "
                            f"exceeds {max_bleed_pct*100:.0f}% of risk budget (${max_allowed_fee:.2f} on ${stop_loss_usd:.2f} SL). "
                            f"Rejecting trade to prevent negative expectation accumulation."
                        )
                        return None

                # 2. Buy contract using proposal_id
                buy_resp = await api.send({"buy": proposal_id, "price": ask_price})
                if "error" in buy_resp:
                    err = buy_resp["error"]
                    log.error(f"Buy execution failed for {symbol}: [{err.get('code')}] {err.get('message')}")
                    return None

                buy_info = buy_resp.get("buy", {})
                contract_id = buy_info.get("contract_id")
                purchase_price = float(buy_info.get("buy_price", ask_price or stake))

                # Engage lock and register direction
                mark_symbol_active(symbol, stake_usd=purchase_price, direction=direction)

                has_cancellation = "cancellation" in proposal_payload and cancellation_info is not None
                log.info(
                    f"✅ Institutional {contract_type} Executed! | {symbol} | "
                    f"Contract ID: {contract_id} | Stake: ${purchase_price:.2f} | "
                    f"Deal Cancellation: {has_cancellation} "
                    f"{'(Fee: $' + str(round(cancellation_fee, 2)) + ')' if has_cancellation else ''}"
                )

                return {
                    "contract_id": contract_id,
                    "symbol": symbol,
                    "direction": direction,
                    "contract_type": contract_type,
                    "purchase_price": purchase_price,
                    "stake": stake,
                    "multiplier": multiplier,
                    "stop_loss_usd": stop_loss_usd,
                    "take_profit_usd": take_profit_usd,
                    "has_deal_cancellation": has_cancellation,
                    "cancellation_fee": cancellation_fee,
                    "date_expiry": date_expiry,
                    "ob_top": ob_top,
                    "ob_bottom": ob_bottom,
                }

            except Exception as e:
                log.warning(f"Order execution attempt {attempt}/{config.MAX_RETRIES} error for {symbol}: {e}")
                if attempt < config.MAX_RETRIES:
                    await asyncio.sleep(config.RETRY_DELAY_S)

    log.error(f"❌ Failed to execute {direction} multiplier order for {symbol} after {config.MAX_RETRIES} attempts.")
    return None


async def cancel_multiplier_contract(contract_id: int | str) -> dict | None:
    """
    Cancel an active Deriv Multiplier contract under Deal Cancellation.
    Refunds initial stake minus cancellation fee.
    """
    api = get_api()
    if not api or not api.is_active or not api.authorized:
        log.error("Cannot cancel contract: Deriv API not connected/authorized.")
        return None
    try:
        resp = await api.send({"cancel": contract_id})
        return resp
    except Exception as e:
        log.error(f"Failed to cancel contract {contract_id}: {e}")
        return None
