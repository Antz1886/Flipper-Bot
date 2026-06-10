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
from modules.smc_engine import TradeSignal
from config import (
    KELLY_FRACTION,
    CIRCUIT_BREAKER_USD,
    MIN_STAKE_USD,
    MAX_STAKE_USD,
    RISK_REWARD_RATIO,
    ACCOUNT_TARGET_USD,
    DEMO_MODE,
    MAX_STAKE_PCT_OF_BALANCE,
)

log = logging.getLogger(__name__)


class CircuitBreakerTripped(Exception):
    pass


def calculate_stake(balance: float, signal: TradeSignal, multiplier: int) -> tuple[float, float, float] | None:
    """
    Calculate required margin stake, stop_loss amount, and take_profit amount
    such that the trade is executed with a hardcoded $1.00 stake (Account Flip mode)
    while maintaining absolute Stop Loss and Take Profit levels based on SMC engine's
    Order Block invalidation level.

    Parameters
    ----------
    balance    : Current account balance in USD
    signal     : The generated TradeSignal containing entry, SL, TP levels
    multiplier : The chosen multiplier for the contract

    Returns
    -------
    (stake, stop_loss_amount, take_profit_amount)  — all in USD
    None if balance is too low, or if the leverage is too high for the SL zone.
    """
    if balance < CIRCUIT_BREAKER_USD:
        raise CircuitBreakerTripped(
            f"Balance ${balance:.2f} < circuit-breaker ${CIRCUIT_BREAKER_USD:.2f}"
        )

    if balance >= ACCOUNT_TARGET_USD:
        return None  # Target reached — caller handles shutdown

    # 1. Growth Mode Compounding: Threshold sizing gate
    if balance < 5.00:
        stake = 1.00
        strategy_mode = "[ACCOUNT FLIP]"
    else:
        stake = balance * 0.20
        strategy_mode = "[GROWTH]"

    # 2. Structural Distances
    sl_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
    tp_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price

    # 3. Contract Loss Percentage at the SMC Stop Loss level
    contract_loss_pct_at_sl = sl_pct * multiplier
    
    # 4. Leverage Validation
    # If the SMC SL requires losing >95% of the contract value, the contract
    # will likely hit the 100% stop-out due to spread or micro-fluctuations
    # BEFORE it actually hits the structural SMC Stop Loss price.
    if contract_loss_pct_at_sl > 0.95:
        log.warning(
            f"Leverage Too High: SMC SL implies a {contract_loss_pct_at_sl:.1%} loss of stake. "
            f"Trade would margin call prematurely. Rejecting setup."
        )
        return None

    if contract_loss_pct_at_sl == 0:
        return None  # Prevent division by zero

    # 5. Calculate Stop Loss and Take Profit dollar amounts based on the calculated stake
    stop_loss_amount = stake * contract_loss_pct_at_sl
    # Ensure stop loss amount is at least 0.01 USD
    stop_loss_amount = max(0.01, stop_loss_amount)

    # Force absolute Take Profit (TP) to be exactly 3.0 * Stop Loss amount (1:3 R:R)
    take_profit_amount = 3.0 * stop_loss_amount
    take_profit_amount = max(0.01, take_profit_amount)

    stake = round(stake, 2)
    stop_loss_amount = round(stop_loss_amount, 2)
    take_profit_amount = round(take_profit_amount, 2)

    mode_tag = "[DEMO]" if DEMO_MODE else "[LIVE]"

    log.info(
        f"{mode_tag} {strategy_mode} Override Sizing | "
        f"Stake: ${stake:.2f} | SL_Amt: ${stop_loss_amount:.2f} | "
        f"TP_Amt: ${take_profit_amount:.2f} | "
        f"SL dist: {sl_pct*100:.2f}% | Risking {contract_loss_pct_at_sl*100:.1f}% of stake"
    )
    return stake, stop_loss_amount, take_profit_amount


# Alias for compatibility with Growth Mode requests
calculate_sizing = calculate_stake



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
