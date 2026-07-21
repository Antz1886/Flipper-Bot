"""
modules/risk_manager.py — Risk & Balance Manager for Deriv
Validates wallet balance against circuit breaker thresholds.
"""

import logging
import config
from modules.connector import get_api

log = logging.getLogger(__name__)


async def check_account_health() -> bool:
    """
    Check if the account balance is above the circuit breaker threshold.
    Returns True if healthy, False if circuit breaker triggered.
    """
    api = get_api()
    if not api or not api.is_active:
        log.error("Risk check failed: Deriv API is disconnected.")
        return False

    balance = api.balance
    if not api.authorized:
        log.warning("⚠️ DERIV_API_TOKEN is missing in .env. Bot is running in READ-ONLY Market Scanner Mode.")
        return True

    log.info(f"Deriv Wallet Balance: {api.currency} {balance:.2f}")

    if balance < config.CIRCUIT_BREAKER_USD:
        log.warning(
            f"🛑 CIRCUIT BREAKER: Balance {api.currency} {balance:.2f} "
            f"is below minimum threshold {config.CIRCUIT_BREAKER_USD:.2f}. Halting trading."
        )
        return False

    return True
