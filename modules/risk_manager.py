"""
modules/risk_manager.py — Quantitative Institutional Risk & Portfolio Manager
Implements Conti's non-negotiable risk rules with true Deriv Multiplier math:
  1. Structural Position Sizing: Links stake_usd and stop_loss_usd directly to
     the structural stop distance (|entry - stop_loss| / entry) and symbol multiplier.
  2. Cash + Open Stake Equity Accounting: Prevents active margin stakes from
     falsely tripping the 5% daily drawdown circuit breaker.
  3. Exchange-Synced Position Locks: Queries Deriv's live `portfolio` endpoint so
     closed trades unlock automatically and open trades persist across UTC midnight.
  4. Correlation Exposure Cap: Limits total concurrent USD positions to MAX_CONCURRENT_TRADES.
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
    open_stakes_usd: float = 0.0
    realized_loss: float = 0.0
    circuit_breaker_tripped: bool = False
    active_symbols: set = field(default_factory=set)

    @property
    def effective_equity(self) -> float:
        """Total account equity = cash balance + stakes currently locked in open contracts."""
        return self.current_balance + self.open_stakes_usd


# Session tracking singleton
_risk_state = SessionRiskState()


def get_risk_state() -> SessionRiskState:
    return _risk_state


def _ensure_session_sync(cash_balance: float):
    """Synchronize or roll over the daily UTC session without losing open exchange locks."""
    global _risk_state
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    effective_eq = cash_balance + _risk_state.open_stakes_usd

    if _risk_state.session_date != today_utc:
        log.info(
            f"🌅 New Trading Session ({today_utc}) | Starting Equity: ${effective_eq:.2f} "
            f"(Cash: ${cash_balance:.2f}, Active Stakes: ${_risk_state.open_stakes_usd:.2f})"
        )
        _risk_state.session_date = today_utc
        _risk_state.starting_balance = effective_eq
        _risk_state.current_balance = cash_balance
        _risk_state.realized_loss = 0.0
        _risk_state.circuit_breaker_tripped = False
    else:
        _risk_state.current_balance = cash_balance


async def sync_open_positions() -> bool:
    """
    Query Deriv's live `portfolio` API to synchronize active multiplier contracts.
    Automatically unlocks symbols whose contracts hit TP/SL and tracks active stakes.
    """
    api = get_api()
    if not api or not api.is_active or not api.authorized:
        return True

    try:
        resp = await api.send({"portfolio": 1})
        if "error" in resp:
            log.warning(f"Portfolio sync warning: {resp['error'].get('message')}")
            return False

        contracts = resp.get("portfolio", {}).get("contracts", [])
        live_symbols = set()
        total_open_stake = 0.0

        for c in contracts:
            sym = c.get("symbol") or c.get("underlying")
            if sym:
                live_symbols.add(sym)
            total_open_stake += float(c.get("buy_price", 0.0))

        # Log any symbols that closed since last scan
        closed_syms = _risk_state.active_symbols - live_symbols
        for sym in closed_syms:
            log.info(f"🔓 Position closed on exchange for {sym} — symbol lock released.")

        _risk_state.active_symbols = live_symbols
        _risk_state.open_stakes_usd = round(total_open_stake, 2)
        return True

    except Exception as e:
        log.warning(f"Failed to sync open portfolio positions: {e}")
        return False


def check_account_health() -> bool:
    """
    Validate account equity against the absolute floor and the 5% daily drawdown circuit breaker.
    Uses effective_equity (cash balance + active open stakes) so open margin does not
    falsely trip the drawdown breaker.
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
    equity = _risk_state.effective_equity

    log.info(
        f"Deriv Account | Equity: {api.currency} {equity:.2f} "
        f"(Cash: {balance:.2f} | Open Stakes: {_risk_state.open_stakes_usd:.2f} | "
        f"Active Trades: {len(_risk_state.active_symbols)}/{config.MAX_CONCURRENT_TRADES})"
    )

    # 1. Check absolute account floor
    if equity < config.CIRCUIT_BREAKER_USD:
        log.warning(
            f"🛑 ABSOLUTE FLOOR BREACH: Equity {api.currency} {equity:.2f} "
            f"is below minimum ${config.CIRCUIT_BREAKER_USD:.2f}. Halting trading."
        )
        return False

    # 2. Check Daily Drawdown Circuit Breaker against effective equity
    if _risk_state.starting_balance > 0:
        drawdown_usd = _risk_state.starting_balance - equity
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


def calculate_order_risk(
    entry_price: float,
    stop_loss: float,
    balance: float,
    symbol: str = "frxEURUSD",
) -> tuple[float, float, float]:
    """
    Calculate dynamic Deriv Multiplier stake_usd, stop_loss_usd, and take_profit_usd
    mathematically linked to the structural stop-loss price distance:
      - Fractional stop distance: delta = |entry_price - stop_loss| / entry_price
      - Multiplier leverage: M = SYMBOL_MULTIPLIERS[symbol]
      - Contract loss at structural SL: Loss_USD = Stake * M * delta
      - Therefore, Required Stake: Stake = Target_Risk_USD / (M * delta)
    Returns:
      (stake_usd, stop_loss_usd, take_profit_usd)
    """
    multiplier = float(config.SYMBOL_MULTIPLIERS.get(symbol, 100))
    target_risk_usd = balance * config.RISK_PER_TRADE_PCT
    max_risk_cap = max(config.MAX_STAKE_USD, balance * 0.05)
    risk_usd = max(config.MIN_STAKE_USD, min(target_risk_usd, max_risk_cap))

    if entry_price > 0 and stop_loss > 0 and entry_price != stop_loss:
        price_dist_pct = abs(entry_price - stop_loss) / entry_price
        leverage_factor = multiplier * price_dist_pct  # Fraction of stake lost at structural SL

        if leverage_factor > 0:
            # Exact stake required so that hitting `stop_loss` price loses `risk_usd`
            raw_stake = risk_usd / leverage_factor
            max_margin_usd = max(config.MAX_STAKE_USD, balance * config.MAX_MARGIN_PER_TRADE_PCT)
            stake_usd = round(max(config.MIN_STAKE_USD, min(raw_stake, max_margin_usd)), 2)

            # Dollar stop loss corresponding to the structural stop price on `stake_usd`
            structural_sl_usd = stake_usd * leverage_factor
            stop_loss_usd = round(max(config.MIN_STAKE_USD, min(stake_usd, risk_usd, structural_sl_usd)), 2)
            take_profit_usd = round(stop_loss_usd * config.RISK_REWARD_RATIO, 2)
            return stake_usd, stop_loss_usd, take_profit_usd

    # Fallback if zero price distance passed
    stop_loss_usd = round(risk_usd, 2)
    take_profit_usd = round(stop_loss_usd * config.RISK_REWARD_RATIO, 2)
    stake_usd = stop_loss_usd
    return stake_usd, stop_loss_usd, take_profit_usd


def can_open_symbol(symbol: str) -> bool:
    """
    Ensure:
      1. No duplicate position is open on `symbol`.
      2. Total concurrent positions across correlated USD pairs < MAX_CONCURRENT_TRADES.
    """
    if symbol in _risk_state.active_symbols:
        return False
    if len(_risk_state.active_symbols) >= config.MAX_CONCURRENT_TRADES:
        return False
    return True


def mark_symbol_active(symbol: str, stake_usd: float = 0.0):
    """Register symbol as active AFTER confirmed order execution."""
    _risk_state.active_symbols.add(symbol)
    if stake_usd > 0:
        _risk_state.open_stakes_usd = round(_risk_state.open_stakes_usd + stake_usd, 2)


def mark_symbol_closed(symbol: str):
    """Release active symbol lock."""
    _risk_state.active_symbols.discard(symbol)
