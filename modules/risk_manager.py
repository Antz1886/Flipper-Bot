"""Portfolio‑level risk manager (Tick-Momentum edition).
Provides stake verification and circuit-breaker checks.
"""

import logging
from config import (
    CIRCUIT_BREAKER_USD,
    ACCOUNT_TARGET_USD,
    TRADE_STAKE,
    MAX_CONCURRENT_TRADES,
)

log = logging.getLogger(__name__)

class CircuitBreakerTripped(Exception):
    pass

def calculate_stake(balance: float) -> float | None:
    """Check circuit breaker and target limits, returning the fixed trade stake.
    Returns ``None`` if target is reached.
    """
    if balance < CIRCUIT_BREAKER_USD:
        raise CircuitBreakerTripped(
            f"Balance ${balance:.2f} < circuit‑breaker ${CIRCUIT_BREAKER_USD:.2f}"
        )
    if balance >= ACCOUNT_TARGET_USD:
        log.info(f"Target ${ACCOUNT_TARGET_USD:.2f} reached or exceeded. Halting trading.")
        return None

    return TRADE_STAKE

def can_open_position(current_open: int, balance: float) -> bool:
    """Determine whether a new position can be opened.
    * ``current_open`` – number of currently open contracts.
    * ``balance`` – account balance.
    """
    if current_open >= MAX_CONCURRENT_TRADES:
        log.debug("Maximum open positions reached; rejecting new trade")
        return False
    if balance < CIRCUIT_BREAKER_USD:
        log.warning("Balance below circuit breaker; rejecting new trade")
        return False
    return True

# Compatibility alias
calculate_sizing = calculate_stake
