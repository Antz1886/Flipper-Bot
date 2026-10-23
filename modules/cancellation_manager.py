"""
modules/cancellation_manager.py — Deriv Multiplier Deal Cancellation Monitor
Implements the Sub-Second SLA Timing Architecture (Fix 1):
  - Solves the M15 vs 15m Cancellation Race Condition.
  - Do NOT wait for the candle to close at 15:00:00 (which arrives at broker at 15:00:01 and rejects).
  - Evaluates live displacement at Minute 14, Second 55 (895.0s / 5s strict SLA before expiration).
  - If live spot has failed to displace away from OB, dispatches {"cancel": contract_id} immediately,
    guaranteeing server arrival before the 15m window shuts.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional
from modules.risk_manager import mark_symbol_closed
from modules.order_executor import cancel_multiplier_contract
import config

log = logging.getLogger(__name__)


@dataclass
class ActiveCancellationContract:
    contract_id: int | str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: float
    expiry_time: float
    ob_top: float
    ob_bottom: float
    cancellation_fee: float = 0.0
    evaluated: bool = False


_tracked_contracts: dict[str, ActiveCancellationContract] = {}


def register_cancellation_contract(
    contract_id: int | str,
    symbol: str,
    direction: str,
    ob_top: float,
    ob_bottom: float,
    date_expiry: Optional[float] = None,
    cancellation_fee: float = 0.0,
):
    """
    Register an active contract for sub-second SLA Deal Cancellation monitoring.
    Schedules evaluation at minute 14, second 55 (5.0s SLA before window shuts).
    """
    now_ts = time.time()
    # If broker provided date_expiry unix timestamp, use it; otherwise default to 15m (900s)
    exp_ts = float(date_expiry) if date_expiry and float(date_expiry) > now_ts else (now_ts + 900.0)

    _tracked_contracts[str(contract_id)] = ActiveCancellationContract(
        contract_id=contract_id,
        symbol=symbol,
        direction=direction.upper(),
        entry_time=now_ts,
        expiry_time=exp_ts,
        ob_top=ob_top,
        ob_bottom=ob_bottom,
        cancellation_fee=cancellation_fee,
    )
    sla_seconds = exp_ts - now_ts - getattr(config, "DEAL_CANCELLATION_SLA_BUFFER_S", 5.0)
    log.info(
        f"🛡️ [DEAL CANCELLATION SLA REGISTERED] {symbol} (Contract: {contract_id}) | "
        f"OB: [${ob_bottom:.5f} - ${ob_top:.5f}] | Fee: ${cancellation_fee:.2f} | "
        f"Strict SLA dispatch scheduled in {sla_seconds:.1f}s (Minute 14, Second 55)."
    )


def get_tracked_cancellation_contracts() -> dict[str, ActiveCancellationContract]:
    return _tracked_contracts


def clear_cancellation_contract(contract_id: int | str):
    _tracked_contracts.pop(str(contract_id), None)


async def evaluate_sluggish_ob_cancellations(
    symbol: str,
    current_spot: float,
) -> bool:
    """
    Evaluate live price against OB extreme at Minute 14, Second 55 (Fix 1).
    Does NOT wait for candle close. Dispatches {"cancel": contract_id} at 895.0s.
    """
    if current_spot <= 0:
        return False

    now_ts = time.time()
    sla_buffer = getattr(config, "DEAL_CANCELLATION_SLA_BUFFER_S", 5.0)
    trigger_delay = getattr(config, "DEAL_CANCELLATION_TRIGGER_S", 895.0)

    matching_c_ids = [cid for cid, c in list(_tracked_contracts.items()) if c.symbol == symbol]
    if not matching_c_ids:
        return False

    for cid in matching_c_ids:
        contract = _tracked_contracts[cid]
        if contract.evaluated:
            continue

        eval_deadline = min(contract.entry_time + trigger_delay, contract.expiry_time - sla_buffer)

        # Trigger ONLY when we reach minute 14, second 55 (895s / 5s before expiry)
        if now_ts < eval_deadline:
            remaining_s = eval_deadline - now_ts
            log.debug(f"  {symbol} contract {contract.contract_id}: {remaining_s:.1f}s remaining until SLA evaluation.")
            continue

        contract.evaluated = True
        sluggish = False
        reason = ""

        if contract.direction == "LONG":
            # For Long: price must have aggressively displaced ABOVE ob_top.
            # If current spot <= ob_top at 14m 55s mark, reaction is sluggish!
            if current_spot <= contract.ob_top:
                sluggish = True
                reason = (
                    f"At 14m 55s SLA mark, price (${current_spot:.5f}) has failed to displace "
                    f"above OB Top (${contract.ob_top:.5f})"
                )
        elif contract.direction == "SHORT":
            # For Short: price must have aggressively displaced BELOW ob_bottom.
            # If current spot >= ob_bottom at 14m 55s mark, reaction is sluggish!
            if current_spot >= contract.ob_bottom:
                sluggish = True
                reason = (
                    f"At 14m 55s SLA mark, price (${current_spot:.5f}) has failed to displace "
                    f"below OB Bottom (${contract.ob_bottom:.5f})"
                )

        if sluggish:
            log.warning(
                f"⏱️ [SLA TRIGGER: 14m 55s DISPATCH] {contract.symbol} | {reason}. "
                f"Dispatching API cancellation 5s before broker expiry!"
            )
            res = await cancel_multiplier_contract(contract.contract_id)
            if res and "error" not in res:
                log.info(
                    f"✅ [DEAL CANCELLATION SECURED] {contract.symbol} contract {contract.contract_id} cancelled! "
                    f"Recovered stake minus ${contract.cancellation_fee:.2f} fee. Full 1% SL stop-out averted."
                )
                mark_symbol_closed(contract.symbol)
                _tracked_contracts.pop(cid, None)
                return True
            else:
                err_msg = res.get("error", {}).get("message") if res else "No response"
                log.warning(f"Deal cancellation response for {contract.symbol}: {err_msg}")
                _tracked_contracts.pop(cid, None)
        else:
            log.info(
                f"🚀 [OB DISPLACEMENT CONFIRMED] {contract.symbol} (Contract: {contract.contract_id}) "
                f"successfully displaced away at 14m 55s (${current_spot:.5f}). "
                f"Deal Cancellation allowed to safely expire into profit."
            )
            _tracked_contracts.pop(cid, None)

    return False
