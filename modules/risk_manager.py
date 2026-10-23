"""
modules/risk_manager.py — Quantitative Institutional Risk & Portfolio Manager
Implements Conti's non-negotiable risk rules with true Deriv Multiplier math:
  1. Structural Position Sizing: Links stake_usd and stop_loss_usd directly to
     the structural stop distance (|entry - stop_loss| / entry) and symbol multiplier.
  2. Cash + Open Stake Equity Accounting: Prevents active margin stakes from
     falsely tripping the 5% daily drawdown circuit breaker.
  3. Exchange-Synced Position Locks: Queries Deriv's live `portfolio` endpoint
     (parsing `underlying_symbol`, `symbol`, and `shortcode`) + enforces a 15m
     per-symbol cooldown so duplicate orders can never stack on the same candle.
  4. Base / Quote USD Concurrency & DXY Correlation Direction Lock:
     Limits base-USD pairs (max 1) and quote-USD pairs (max 1) to eliminate the
     DXY correlation trap (e.g. going Long EURUSD and Long GBPUSD simultaneously).
"""

import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
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
    active_directions: dict[str, str] = field(default_factory=dict)
    last_entry_time: dict[str, float] = field(default_factory=dict)

    @property
    def effective_equity(self) -> float:
        """Total account equity = cash balance + stakes currently locked in open contracts."""
        return self.current_balance + self.open_stakes_usd


# Session tracking singleton
_risk_state = SessionRiskState()


def get_risk_state() -> SessionRiskState:
    return _risk_state


def get_dxy_exposure(symbol: str, direction: str) -> str:
    """
    Determine net US Dollar (DXY) directional exposure of a trade:
      - Long Quote-USD (EURUSD, GBPUSD, AUDUSD) = DXY_SHORT (Selling USD)
      - Short Quote-USD = DXY_LONG (Buying USD)
      - Long Base-USD (USDJPY, USDCAD) = DXY_LONG (Buying USD)
      - Short Base-USD = DXY_SHORT (Selling USD)
    """
    quote_pairs = set(getattr(config, "QUOTE_USD_PAIRS", ["frxEURUSD", "frxGBPUSD", "frxAUDUSD"]))
    d = direction.upper()
    if symbol in quote_pairs:
        return "DXY_SHORT" if d == "LONG" else "DXY_LONG"
    else:
        return "DXY_LONG" if d == "LONG" else "DXY_SHORT"


def _extract_contract_symbol(contract: dict) -> str | None:
    """
    Extract normalized symbol name from Deriv portfolio contract object across
    both v3 WebSocket (`symbol` / `underlying`) and OTP WebSocket (`underlying_symbol` / `shortcode`).
    """
    for key in ("underlying_symbol", "symbol", "underlying"):
        val = contract.get(key)
        if val and isinstance(val, str):
            return val

    shortcode = (contract.get("shortcode") or "").upper()
    if shortcode:
        for configured_sym in config.SYMBOLS:
            if configured_sym.upper() in shortcode:
                return configured_sym
    return None


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
    Automatically unlocks symbols whose contracts hit TP/SL and tracks active stakes & directions.
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
        live_directions = {}
        total_open_stake = 0.0

        for c in contracts:
            sym = _extract_contract_symbol(c)
            if sym:
                live_symbols.add(sym)
                c_type = (c.get("contract_type") or "").upper()
                c_dir = "LONG" if "UP" in c_type or "CALL" in c_type else "SHORT"
                live_directions[sym] = c_dir
            total_open_stake += float(c.get("buy_price", 0.0))

        # Preserve any symbol entered within the last 60 seconds against portfolio settlement lag
        now_ts = time.time()
        for sym, entry_ts in list(_risk_state.last_entry_time.items()):
            if now_ts - entry_ts < 60.0 and sym in _risk_state.active_symbols:
                live_symbols.add(sym)
                if sym in _risk_state.active_directions:
                    live_directions[sym] = _risk_state.active_directions[sym]

        # Log any symbols that closed since last scan
        closed_syms = _risk_state.active_symbols - live_symbols
        for sym in closed_syms:
            log.info(f"🔓 Position closed on exchange for {sym} — symbol lock released.")
            _risk_state.active_directions.pop(sym, None)

        # Self-Healing Reconciliation & Continual Learning
        try:
            from modules.self_healer import get_self_healer
            get_self_healer().reconcile_exchange_portfolio(contracts)

            if closed_syms:
                prof_resp = await api.send({"profit_table": 1, "limit": 5})
                transactions = prof_resp.get("profit_table", {}).get("transactions", [])
                for t in transactions:
                    t_sym = _extract_contract_symbol(t)
                    if t_sym in closed_syms:
                        sell_price = float(t.get("sell_price", 0.0))
                        buy_price = float(t.get("buy_price", 0.0))
                        pnl = sell_price - buy_price
                        r_mult = round(pnl / buy_price * 2.0, 2) if buy_price > 0 else 0.0
                        exit_reason = "WIN_TP" if pnl > 0 else "LOSS_SL"

                        get_self_healer().record_trade_result(r_mult, exit_reason)
                        from modules.meta_learner import get_meta_learner
                        from modules.trade_learner import get_session_phase, get_learner_db, TradePostMortem
                        get_meta_learner().learn_from_trade(
                            symbol=t_sym,
                            session_phase=get_session_phase(),
                            macro_pool_type="SWING",
                            pnl_usd=pnl,
                            r_multiple=r_mult,
                            exit_reason=exit_reason,
                        )
                        get_learner_db().save_outcome(TradePostMortem(
                            trade_id=str(t.get("contract_id", "")),
                            exit_price=float(t.get("sell_spot", 0.0)),
                            pnl_usd=pnl,
                            r_multiple=r_mult,
                            exit_reason=exit_reason,
                            loss_root_cause="ORGANIC_STOP_OUT" if pnl <= 0 else None,
                            holding_time_seconds=float(t.get("sell_time", 0)) - float(t.get("purchase_time", 0)),
                            exit_time_utc=datetime.now(timezone.utc).isoformat(),
                        ))
        except Exception as heal_err:
            log.debug(f"Self-healing reconciliation check skipped: {heal_err}")

        _risk_state.active_symbols = live_symbols
        _risk_state.active_directions = live_directions
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
    risk_pct_override: Optional[float] = None,
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
    risk_pct = risk_pct_override if risk_pct_override is not None else getattr(config, "RISK_PER_TRADE_PCT", 0.01)
    target_risk_usd = balance * risk_pct
    max_risk_cap = min(config.MAX_STAKE_USD, balance * 0.05)
    risk_usd = max(config.MIN_STAKE_USD, min(target_risk_usd, max_risk_cap))

    if entry_price > 0 and stop_loss > 0 and entry_price != stop_loss:
        price_dist_pct = abs(entry_price - stop_loss) / entry_price
        leverage_factor = multiplier * price_dist_pct

        if leverage_factor > 0:
            raw_stake = risk_usd / leverage_factor
            max_margin_usd = max(
                config.MIN_STAKE_USD,
                min(config.MAX_STAKE_USD, balance * config.MAX_MARGIN_PER_TRADE_PCT),
            )
            stake_usd = round(max(config.MIN_STAKE_USD, min(raw_stake, max_margin_usd)), 2)
            structural_sl_usd = stake_usd * leverage_factor
            stop_loss_usd = round(max(config.MIN_STAKE_USD, min(stake_usd, risk_usd, structural_sl_usd)), 2)
            take_profit_usd = round(stop_loss_usd * config.RISK_REWARD_RATIO, 2)
            return stake_usd, stop_loss_usd, take_profit_usd

    stop_loss_usd = round(risk_usd, 2)
    take_profit_usd = round(stop_loss_usd * config.RISK_REWARD_RATIO, 2)
    stake_usd = stop_loss_usd
    return stake_usd, stop_loss_usd, take_profit_usd


def can_open_symbol(symbol: str, direction: Optional[str] = None) -> bool:
    """
    Ensure:
      1. No duplicate position is open on `symbol`.
      2. Per-symbol M15 cooldown (`SYMBOL_COOLDOWN_S`) has elapsed since last entry.
      3. Total concurrent positions < MAX_CONCURRENT_TRADES.
      4. Concurrency limits on Base-USD (max 1) and Quote-USD (max 1) pairs.
      5. DXY Correlation Direction Lock: Disallows stacking identical DXY directional exposure.
      6. Tier-1 Macroeconomic Data Embargo: Disallows opening USD positions during high-impact news window.
    """
    # 6. Tier-1 Economic News Embargo Check
    if getattr(config, "ENFORCE_NEWS_EMBARGO", True):
        from modules.news_embargo import is_news_embargo_active
        embargo_active, embargo_reason = is_news_embargo_active()
        if embargo_active:
            log.warning(f"🚫 {embargo_reason} — blocking entry on {symbol}.")
            return False

    if symbol in _risk_state.active_symbols:
        return False
    if len(_risk_state.active_symbols) >= getattr(config, "MAX_CONCURRENT_TRADES", 2):
        return False

    cooldown_s = getattr(config, "SYMBOL_COOLDOWN_S", 0)
    last_ts = _risk_state.last_entry_time.get(symbol, 0.0)
    if cooldown_s > 0 and last_ts > 0 and (time.time() - last_ts) < cooldown_s:
        return False

    # 4. Enforce Base-USD (max 1) and Quote-USD (max 1) Concurrency
    base_pairs = set(getattr(config, "BASE_USD_PAIRS", ["frxUSDJPY", "frxUSDCAD"]))
    quote_pairs = set(getattr(config, "QUOTE_USD_PAIRS", ["frxEURUSD", "frxGBPUSD", "frxAUDUSD"]))
    max_base = getattr(config, "MAX_BASE_USD_TRADES", 1)
    max_quote = getattr(config, "MAX_QUOTE_USD_TRADES", 1)

    active_base = [s for s in _risk_state.active_symbols if s in base_pairs]
    active_quote = [s for s in _risk_state.active_symbols if s in quote_pairs]

    if symbol in base_pairs and len(active_base) >= max_base:
        log.info(f"⏭️ Skipping {symbol}: Base-USD pair concurrency limit ({len(active_base)}/{max_base}) reached.")
        return False

    if symbol in quote_pairs and len(active_quote) >= max_quote:
        log.info(f"⏭️ Skipping {symbol}: Quote-USD pair concurrency limit ({len(active_quote)}/{max_quote}) reached.")
        return False

    # 5. DXY Correlation Direction Lock
    if direction and getattr(config, "ENFORCE_DXY_DIRECTION_LOCK", True):
        new_dxy = get_dxy_exposure(symbol, direction)
        for act_sym, act_dir in _risk_state.active_directions.items():
            existing_dxy = get_dxy_exposure(act_sym, act_dir)
            if new_dxy == existing_dxy:
                log.info(
                    f"⏭️ Skipping {symbol} {direction}: DXY correlation trap! "
                    f"Already holding {existing_dxy} via {act_sym} ({act_dir}). "
                    f"Preventing stacked {new_dxy} exposure."
                )
                return False

    return True


def mark_symbol_active(symbol: str, stake_usd: float = 0.0, direction: str = "LONG"):
    """Register symbol and direction as active AFTER confirmed order execution."""
    _risk_state.active_symbols.add(symbol)
    _risk_state.active_directions[symbol] = direction.upper()
    _risk_state.last_entry_time[symbol] = time.time()
    if stake_usd > 0:
        _risk_state.open_stakes_usd = round(_risk_state.open_stakes_usd + stake_usd, 2)


def mark_symbol_closed(symbol: str):
    """Release active symbol lock and clear cooldown for immediate re-evaluation if needed."""
    _risk_state.active_symbols.discard(symbol)
    _risk_state.active_directions.pop(symbol, None)
    _risk_state.last_entry_time.pop(symbol, None)
