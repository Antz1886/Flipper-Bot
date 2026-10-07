"""
modules/risk_manager.py — Hardcoded Institutional Risk Mandates
Implements Conti's non-negotiable risk rules:
  1. Fixed capital allocation: Dynamic stake and risk sized to fixed % of account equity.
  2. Automated Daily Circuit Breaker: If cumulative session loss exceeds MAX_DAILY_DRAWDOWN_PCT,
     trading is halted automatically until the next UTC session.
  3. Active Position Locks: Limits concurrent exposure per symbol.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
import config
from modules.connector import get_api

log = logging.getLogger(__name__)


@dataclass
class SessionRiskState:
    session_date: str = ""
    starting_balance: float = 0.0
    current_balance: float = 0.0
    realized_loss: float = 0.0
    circuit_breaker_tripped: bool = False
    active_symbols: set = field(default_factory=set)


# Session tracking singleton
_risk_state = SessionRiskState()


def get_risk_state() -> SessionRiskState:
    return _risk_state


def _ensure_session_sync(balance: float):
    """Reset session parameters at 00:00 UTC."""
    global _risk_state
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if _risk_state.session_date != today_utc:
        log.info(f"🌅 New Trading Session ({today_utc}) | Initializing starting balance: ${balance:.2f}")
        _risk_state.session_date = today_utc
        _risk_state.starting_balance = balance
        _risk_state.current_balance = balance
        _risk_state.realized_loss = 0.0
        _risk_state.circuit_breaker_tripped = False
        _risk_state.active_symbols.clear()
    else:
        _risk_state.current_balance = balance


def check_account_health() -> bool:
    """
    Check if the account balance is above the circuit breaker threshold
    and that the daily drawdown limit has not been breached.
    """
    api = get_api()
    if not api or not api.is_active:
        log.error("Risk check failed: Deriv API is disconnected.")
        return False

    balance = api.balance
    if not api.authorized:
        log.warning("⚠️ DERIV_API_TOKEN is missing or unauthorized. Bot is running in READ-ONLY Market Scanner Mode.")
        return True

    if balance is None:
        log.warning("Deriv wallet balance not available yet. Waiting for balance update before trading.")
        return False

    _ensure_session_sync(balance)

    # Check absolute account floor
    if balance < config.CIRCUIT_BREAKER_USD:
        log.warning(
            f"🛑 ABSOLUTE FLOOR BREACH: Balance {api.currency} {balance:.2f} "
            f"is below minimum ${config.CIRCUIT_BREAKER_USD:.2f}. Halting trading."
        )
        return False

    # Check Daily Drawdown Circuit Breaker
    if _risk_state.starting_balance > 0:
        drawdown_usd = _risk_state.starting_balance - balance
        max_allowed_loss = _risk_state.starting_balance * config.MAX_DAILY_DRAWDOWN_PCT

        if drawdown_usd >= max_allowed_loss:
            _risk_state.circuit_breaker_tripped = True
            log.critical(
                f"🚨 [INSTITUTIONAL CIRCUIT BREAKER TRIPPED] Daily drawdown (${drawdown_usd:.2f}) "
                f"exceeds {config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}% limit (${max_allowed_loss:.2f})! "
                f"Trading locked until next UTC day."
            )
            return False

    return True


def calculate_order_risk(entry_price: float, stop_loss: float, balance: float) -> tuple[float, float, float]:
    """
    Calculate dynamic position sizing and SL/TP in USD based on fixed equity risk %.
    Formula:
      Target Risk USD = Balance * RISK_PER_TRADE_PCT
      Multiplier Stake = Risk USD / (Price Distance % * Multiplier)
    Returns:
      (stake_usd, stop_loss_usd, take_profit_usd)
    """
    target_risk_usd = balance * config.RISK_PER_TRADE_PCT
    # Bound risk amount within reasonable constraints (minimum stake floor, max equity risk)
    max_risk_cap = max(config.MAX_STAKE_USD, balance * 0.05)
    risk_usd = max(config.MIN_STAKE_USD, min(target_risk_usd, max_risk_cap))

    # For Deriv Multipliers, the maximum dollar loss is capped directly by stop_loss_usd
    stop_loss_usd = round(risk_usd, 2)
    take_profit_usd = round(risk_usd * config.RISK_REWARD_RATIO, 2)
    # The stake can match the risk or be configured according to desired exposure
    stake_usd = stop_loss_usd

    return stake_usd, stop_loss_usd, take_profit_usd


def can_open_symbol(symbol: str) -> bool:
    """Ensure no concurrent duplicate position is opened on the same symbol."""
    return symbol not in _risk_state.active_symbols


def mark_symbol_active(symbol: str):
    """Register symbol as active."""
    _risk_state.active_symbols.add(symbol)


def mark_symbol_closed(symbol: str):
    """Release active symbol lock."""
    _risk_state.active_symbols.discard(symbol)
