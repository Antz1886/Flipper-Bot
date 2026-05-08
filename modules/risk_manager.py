"""
modules/risk_manager.py — Kelly Criterion Stake Calculator (Deriv edition)
With Deriv multiplier contracts, risk is controlled via dollar stake amount
(not pip-based lot sizing). Kelly sizing is dramatically simpler here:

  stake_amount = balance × KELLY_FRACTION
  stop_loss_amount  = stake_amount          (max we can lose = full stake)
  take_profit_amount = stake_amount × R:R   (target profit)

Circuit-breaker halts trading if balance < CIRCUIT_BREAKER_USD.
"""

import logging

from modules.connector import get_api
from config import (
    KELLY_FRACTION,
    CIRCUIT_BREAKER_USD,
    MIN_STAKE_USD,
    MAX_STAKE_USD,
    RISK_REWARD_RATIO,
    ACCOUNT_TARGET_USD,
    DEMO_MODE,
)

log = logging.getLogger(__name__)


class CircuitBreakerTripped(Exception):
    pass


def calculate_stake(balance: float) -> tuple[float, float, float] | None:
    """
    Calculate stake, stop_loss amount, and take_profit amount.

    Parameters
    ----------
    balance : Current account balance in USD

    Returns
    -------
    (stake, stop_loss_amount, take_profit_amount)  — all in USD
    None if balance is too low to trade or circuit-breaker trips.
    """
    if balance < CIRCUIT_BREAKER_USD:
        raise CircuitBreakerTripped(
            f"Balance ${balance:.2f} < circuit-breaker ${CIRCUIT_BREAKER_USD:.2f}"
        )

    if balance >= ACCOUNT_TARGET_USD:
        return None  # Target reached — caller handles shutdown

    stake = round(balance * KELLY_FRACTION, 2)

    # Apply per-trade cap (important for demo — prevents $2,200 stakes on $10k balance)
    if stake > MAX_STAKE_USD:
        log.debug(f"Kelly stake ${stake:.2f} capped to MAX_STAKE_USD ${MAX_STAKE_USD:.2f}")
        stake = MAX_STAKE_USD

    if stake < MIN_STAKE_USD:
        log.warning(
            f"Kelly stake ${stake:.2f} is below Deriv minimum ${MIN_STAKE_USD:.2f}. "
            "Trade skipped."
        )
        return None

    stop_loss_amount   = round(stake, 2)
    take_profit_amount = round(stake * RISK_REWARD_RATIO, 2)
    mode_tag           = "[DEMO]" if DEMO_MODE else "[LIVE]"

    log.info(
        f"{mode_tag} Kelly Sizing | Balance: ${balance:.2f} | "
        f"Stake: ${stake:.2f} | SL: ${stop_loss_amount:.2f} | "
        f"TP: ${take_profit_amount:.2f} | R:R 1:{RISK_REWARD_RATIO}"
    )
    return stake, stop_loss_amount, take_profit_amount


async def get_balance() -> float:
    """Fetch current account balance from the Deriv API."""
    api = get_api()
    try:
        resp = await api.send({"balance": 1})
        bal = float(resp["balance"]["balance"])
        log.debug(f"Live balance: ${bal:.4f}")
        return bal
    except Exception as e:
        log.error(f"Failed to fetch balance: {e}")
        # Fall back to cached value on connector
        return api.balance
