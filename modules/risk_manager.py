"""Portfolio‑level risk manager.
Provides stake calculation (Kelly) and portfolio exposure checks.
"""

import logging
from typing import Tuple

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
    # Portfolio‑level limits added to config
    MAX_TOTAL_RISK,
    MAX_OPEN_POSITIONS,
    MAX_LONG_EXPOSURE,
    MAX_SHORT_EXPOSURE,
)

log = logging.getLogger(__name__)

class CircuitBreakerTripped(Exception):
    pass

def calculate_stake(balance: float, signal: TradeSignal, multiplier: int) -> Tuple[float, float, float] | None:
    """Calculate stake, stop‑loss amount and take‑profit amount respecting portfolio limits.
    Returns ``None`` if balance is below circuit‑breaker or target reached.
    """
    if balance < CIRCUIT_BREAKER_USD:
        raise CircuitBreakerTripped(
            f"Balance ${balance:.2f} < circuit‑breaker ${CIRCUIT_BREAKER_USD:.2f}"
        )
    if balance >= ACCOUNT_TARGET_USD:
        return None  # target reached – caller should stop trading

    # Kelly‑fraction sizing, bounded by min/max stake
    stake = max(MIN_STAKE_USD, balance * KELLY_FRACTION)
    stake = min(stake, MAX_STAKE_USD, balance * MAX_STAKE_PCT_OF_BALANCE)

    # Ensure portfolio‑level risk does not exceed limits
    if stake / balance > MAX_TOTAL_RISK:
        stake = balance * MAX_TOTAL_RISK
        log.info("Stake reduced to respect MAX_TOTAL_RISK limit")

    # Compute SL/TP based on R:R ratio
    sl_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
    tp_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price
    contract_loss_pct_at_sl = sl_pct * multiplier
    if contract_loss_pct_at_sl > 0.95:
        log.warning("Leverage too high – rejecting trade")
        return None
    if contract_loss_pct_at_sl == 0:
        return None

    stop_loss_amount = max(0.01, stake * contract_loss_pct_at_sl)
    take_profit_amount = max(0.01, stake * RISK_REWARD_RATIO * contract_loss_pct_at_sl)

    stake = round(stake, 2)
    stop_loss_amount = round(stop_loss_amount, 2)
    take_profit_amount = round(take_profit_amount, 2)

    mode_tag = "[DEMO]" if DEMO_MODE else "[LIVE]"
    log.info(
        f"{mode_tag} Stake ${stake:.2f} | SL ${stop_loss_amount:.2f} | TP ${take_profit_amount:.2f}"
    )
    return stake, stop_loss_amount, take_profit_amount

# Portfolio‑level checks -------------------------------------------------------

def can_open_position(current_open: int, direction: str, balance: float, stake: float) -> bool:
    """Determine whether a new position respects portfolio caps.
    * ``current_open`` – number of currently open contracts.
    * ``direction`` – "BUY" or "SELL".
    * ``balance`` – account equity.
    * ``stake`` – dollar stake for the potential new trade.
    """
    if current_open >= MAX_OPEN_POSITIONS:
        log.info("Maximum open positions reached; rejecting new trade")
        return False
    exposure = stake / balance
    if direction == "BUY" and exposure > MAX_LONG_EXPOSURE:
        log.info("Long exposure limit exceeded")
        return False
    if direction == "SELL" and exposure > MAX_SHORT_EXPOSURE:
        log.info("Short exposure limit exceeded")
        return False
    if exposure > MAX_TOTAL_RISK:
        log.info("Total risk limit exceeded")
        return False
    return True

# Compatibility alias
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

