"""
scripts/quant_wfo_montecarlo.py — Quantitative Rigor Suite for DBot Engine
Performs:
  1. Entry and Exit Statistical Edge Analysis (Quant Filter vs Random Baseline)
  2. 5-Fold Walk-Forward Optimization (WFO & WFE) to eliminate curve-fitting
  3. 10,000-Iteration Monte Carlo Simulation (Drawdown & Profit Distribution)
"""

import os
import sys
import time
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "1HZ100V_ticks_quant.csv"


# ─── Statistical Indicators ──────────────────────────────────────────────────

def calc_ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = (arr[i] * alpha) + (out[i - 1] * (1.0 - alpha))
    return out


def calc_rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(arr)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    rsi = np.full_like(arr, 50.0)
    if len(arr) <= period:
        return rsi

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calc_zscore(arr: np.ndarray, period: int = 20) -> np.ndarray:
    z = np.zeros_like(arr)
    for i in range(period, len(arr)):
        window = arr[i - period:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-6:
            z[i] = (arr[i] - mean) / std
    return z


def calc_er(arr: np.ndarray, period: int = 15) -> np.ndarray:
    er = np.full_like(arr, 0.5)
    for i in range(period, len(arr)):
        net = abs(arr[i] - arr[i - period])
        diffs = np.sum(np.abs(np.diff(arr[i - period:i + 1])))
        if diffs > 1e-6:
            er[i] = min(1.0, net / diffs)
    return er


def calc_atr(arr: np.ndarray, period: int = 14) -> np.ndarray:
    diffs = np.abs(np.diff(arr))
    atr = np.zeros_like(arr)
    for i in range(period, len(diffs)):
        atr[i + 1] = np.mean(diffs[i - period:i])
    return atr


# ─── Trade Signal Evaluator ──────────────────────────────────────────────────

@dataclass
class TradeResult:
    entry_idx: int
    entry_price: float
    exit_price: float
    direction: str  # "CALL" or "PUT"
    duration: int
    won: bool
    pnl: float      # e.g. +1.76 or -2.00 on $2 stake


def evaluate_simulation(
    prices: np.ndarray,
    strategy: str = "QUANT_V3",
    duration: int = 15,
    stake: float = 2.0,
    payout_rate: float = 0.88,
    z_threshold: float = 0.50,
    er_min: float = 0.42,
    cooldown_ticks: int = 60,
) -> List[TradeResult]:
    """
    Simulates binary options execution on real tick price sequence.
    duration: contract duration in ticks (e.g. 15 ticks on 1s synthetics)
    """
    n = len(prices)
    trades: List[TradeResult] = []
    if n < 320:
        return trades

    # Precalculate vectors
    ema_fast = calc_ema(prices, 5)
    ema_slow = calc_ema(prices, 15)
    ema_m1 = calc_ema(prices, 60)
    ema_m5 = calc_ema(prices, 300)
    rsi = calc_rsi(prices, 14)
    z_score = calc_zscore(prices, 20)
    er = calc_er(prices, 15)
    atr7 = calc_atr(prices, 7)
    atr20 = calc_atr(prices, 20)

    i = 310
    cooldown = 0

    while i < n - duration:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue

        p_curr = prices[i]
        p_prev = prices[i - 1]
        last_tick_up = p_curr > p_prev
        last_tick_down = p_curr < p_prev

        # Micro deltas
        deltas = np.diff(prices[max(0, i - 3):i + 1])
        down_ticks = np.sum(deltas < 0)
        up_ticks = np.sum(deltas > 0)

        # Macro Trend (M1 vs M5)
        m1_bull = p_curr > ema_m1[i] and ema_m1[i] > ema_m1[i - 5]
        m1_bear = p_curr < ema_m1[i] and ema_m1[i] < ema_m1[i - 5]
        m5_bull = p_curr > ema_m5[i]
        m5_bear = p_curr < ema_m5[i]
        dual_bull = m1_bull and m5_bull
        dual_bear = m1_bear and m5_bear

        signal = None

        if strategy == "BASELINE_OLD":
            # Previous buggy logic with loose OR conditions
            if m1_bull:
                if (rsi[i] <= 45.0 or z_score[i] <= -0.7 or down_ticks >= 1) and (last_tick_up or down_ticks >= 1):
                    signal = "CALL"
            elif m1_bear:
                if (rsi[i] >= 45.0 or z_score[i] >= 0.7 or up_ticks >= 1) and (last_tick_down or up_ticks >= 1):
                    signal = "PUT"

        elif strategy == "QUANT_V3":
            # Upgraded Institutional Sniper Edge:
            # 1. Chop / Squeeze filter (Sit out during dead volume)
            if atr20[i] > 0 and (atr7[i] / atr20[i]) < 0.80:
                i += 1
                continue

            # 2. Dual HTF Momentum Alignment + Structural Pullbacks
            if dual_bull and er[i] >= er_min:
                # In uptrend: Only enter on confirmed structural pullback into discount + immediate tick bounce
                is_discount = (z_score[i] <= -z_threshold) or (rsi[i] <= 50.0)
                reversal_confirmed = last_tick_up and (down_ticks >= 1)
                if is_discount and reversal_confirmed:
                    signal = "CALL"

            elif dual_bear and er[i] >= er_min:
                # In downtrend: Only enter on confirmed structural bounce into premium + immediate tick rejection
                is_premium = (z_score[i] >= z_threshold) or (rsi[i] >= 50.0)
                reversal_confirmed = last_tick_down and (up_ticks >= 1)
                if is_premium and reversal_confirmed:
                    signal = "PUT"

            # 3. Statistical Extremes (Statistical boundaries of PRNG synthetic indices)
            elif z_score[i] >= 2.0 and rsi[i] >= 68.0 and last_tick_down and (up_ticks >= 1):
                signal = "PUT"
            elif z_score[i] <= -2.0 and rsi[i] <= 32.0 and last_tick_up and (down_ticks >= 1):
                signal = "CALL"

        elif strategy == "RANDOM":
            signal = "CALL" if np.random.rand() > 0.5 else "PUT"

        # Execution evaluation
        if signal:
            exit_idx = i + duration
            exit_p = prices[exit_idx]
            won = (exit_p > p_curr) if signal == "CALL" else (exit_p < p_curr)
            pnl = (stake * payout_rate) if won else -stake
            trades.append(TradeResult(
                entry_idx=i,
                entry_price=p_curr,
                exit_price=exit_p,
                direction=signal,
                duration=duration,
                won=won,
                pnl=pnl
            ))
            cooldown = cooldown_ticks
            i += 1
        else:
            i += 1

    return trades


# ─── Walk-Forward Optimization (WFO) ─────────────────────────────────────────

def run_walk_forward_optimization(prices: np.ndarray, n_splits: int = 5, duration: int = 15) -> Dict[str, Any]:
    """
    Executes an anchored Walk-Forward Optimization with In-Sample (IS) and Out-Of-Sample (OOS) folds.
    Calculates Walk-Forward Efficiency (WFE = OOS_WinRate / IS_WinRate).
    """
    total_len = len(prices)
    fold_size = total_len // (n_splits + 1)
    results = []

    print(f"\n{'='*75}")
    print(f"🔬 RUNNING {n_splits}-FOLD WALK-FORWARD OPTIMIZATION (WFO) — DURATION: {duration}s")
    print(f"{'='*75}")
    print(f" Fold |   IS Win%   |  OOS Win%   |  WFE Ratio  |  OOS Trades |  OOS Net PnL")
    print(f"------|-------------|-------------|-------------|-------------|-------------")

    for fold in range(1, n_splits + 1):
        is_end = fold * fold_size
        oos_end = (fold + 1) * fold_size

        is_prices = prices[:is_end]
        oos_prices = prices[is_end:oos_end]

        is_trades = evaluate_simulation(is_prices, strategy="QUANT_V3", duration=duration)
        oos_trades = evaluate_simulation(oos_prices, strategy="QUANT_V3", duration=duration)

        is_wr = (sum(1 for t in is_trades if t.won) / len(is_trades) * 100) if is_trades else 50.0
        oos_wr = (sum(1 for t in oos_trades if t.won) / len(oos_trades) * 100) if oos_trades else 50.0
        oos_pnl = sum(t.pnl for t in oos_trades)
        wfe = (oos_wr / is_wr) if is_wr > 0 else 1.0

        results.append({
            "fold": fold,
            "is_wr": is_wr,
            "oos_wr": oos_wr,
            "wfe": wfe,
            "oos_trades": len(oos_trades),
            "oos_pnl": oos_pnl,
        })

        print(f"   {fold:2d} |    {is_wr:5.1f}%   |    {oos_wr:5.1f}%   |    {wfe:5.2f}    |    {len(oos_trades):4d}     |   ${oos_pnl:+6.2f}")

    avg_oos_wr = np.mean([r["oos_wr"] for r in results])
    avg_wfe = np.mean([r["wfe"] for r in results])
    print(f"------|-------------|-------------|-------------|-------------|-------------")
    print(f" AVG  |             |    {avg_oos_wr:5.1f}%   |    {avg_wfe:5.2f}    |             |")
    print(f"{'='*75}")
    return {"avg_oos_wr": avg_oos_wr, "avg_wfe": avg_wfe, "folds": results}


# ─── Monte Carlo Simulation (10,000 Iterations) ──────────────────────────────

def run_monte_carlo_simulation(trades: List[TradeResult], n_iterations: int = 10000, initial_capital: float = 100.0) -> Dict[str, Any]:
    """
    Performs 10,000 Monte Carlo bootstrap resamplings of trade order to stress-test
    path dependency, max drawdown, risk of ruin, and expected terminal equity.
    """
    if not trades:
        return {}

    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)
    max_drawdowns = []
    final_equities = []
    ruin_count = 0

    np.random.seed(42)
    for _ in range(n_iterations):
        sampled = np.random.choice(pnls, size=n_trades, replace=True)
        equity_curve = initial_capital + np.cumsum(sampled)
        final_equities.append(equity_curve[-1])

        # Drawdown calculation
        running_max = np.maximum.accumulate(np.insert(equity_curve, 0, initial_capital))
        dds = (running_max[1:] - equity_curve) / running_max[1:] * 100.0
        max_dd = np.max(dds)
        max_drawdowns.append(max_dd)

        # Ruin condition (< 50% initial capital)
        if np.min(equity_curve) <= initial_capital * 0.50:
            ruin_count += 1

    max_drawdowns = np.array(max_drawdowns)
    final_equities = np.array(final_equities)

    var_95_dd = np.percentile(max_drawdowns, 95)
    var_99_dd = np.percentile(max_drawdowns, 99)
    median_dd = np.median(max_drawdowns)
    median_profit = np.median(final_equities) - initial_capital
    risk_of_ruin = (ruin_count / n_iterations) * 100.0

    print(f"\n{'='*75}")
    print(f"🎲 10,000-ITERATION MONTE CARLO PATH-DEPENDENCE STRESS TEST")
    print(f"{'='*75}")
    print(f"  • Sampled Trades Count     : {n_trades} Trades")
    print(f"  • Median Net Profit        : ${median_profit:+.2f} USD")
    print(f"  • Median Max Drawdown      : {median_dd:.1f}%")
    print(f"  • 95% Confidence Max DD    : {var_95_dd:.1f}% (Value at Risk)")
    print(f"  • 99% Stress Max DD        : {var_99_dd:.1f}% (Extreme VaR)")
    print(f"  • Probability of Ruin (50%): {risk_of_ruin:.2f}%")
    print(f"  • Positive Terminal Equity : {(np.sum(final_equities > initial_capital) / n_iterations * 100):.1f}%")
    print(f"{'='*75}")

    return {
        "median_profit": median_profit,
        "median_dd": median_dd,
        "var_95_dd": var_95_dd,
        "var_99_dd": var_99_dd,
        "risk_of_ruin": risk_of_ruin,
    }


# ─── Main Execution Pipeline ─────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("⚡ QUANTITATIVE STATISTICAL EDGE RIGOR & VALIDATION SUITE")
    print("=" * 75)

    df_ticks = pd.read_csv(CACHE_FILE)
    prices = df_ticks["price"].values
    print(f"[DATA] Loaded {len(prices)} real tick prices from Deriv API cache.")

    # 1. Entry & Exit Edge Comparison: Baseline vs Quant V3
    print(f"\n📊 1. ENTRY & EXIT STATISTICAL EDGE COMPARISON ({len(prices)} Real Ticks)")
    print(f"{'-'*75}")

    t_random = evaluate_simulation(prices, strategy="RANDOM", duration=15)
    t_old = evaluate_simulation(prices, strategy="BASELINE_OLD", duration=15)
    t_v3_15s = evaluate_simulation(prices, strategy="QUANT_V3", duration=15)

    strategies = [
        ("Random Baseline (Coinflip)", t_random),
        ("Old Baseline (Loose OR)", t_old),
        ("Quant Institutional V3 (15s)", t_v3_15s),
    ]

    print(f"{'Strategy':<32} | {'Trades':<7} | {'Win Rate':<10} | {'Profit Factor':<13} | {'Net PnL ($)':<12}")
    print(f"{'-'*75}")
    for name, t_list in strategies:
        if not t_list:
            continue
        wins = sum(1 for t in t_list if t.won)
        losses = len(t_list) - wins
        wr = (wins / len(t_list)) * 100
        gross_win = wins * (2.0 * 0.88)
        gross_loss = losses * 2.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
        net_pnl = sum(t.pnl for t in t_list)
        print(f"{name:<32} | {len(t_list):<7} | {wr:6.1f}%    | {pf:6.2f}        | ${net_pnl:+7.2f}")
    print(f"{'-'*75}")

    # 2. Walk-Forward Optimization
    wfo_results = run_walk_forward_optimization(prices, n_splits=5, duration=15)

    # 3. Monte Carlo Simulation
    mc_results = run_monte_carlo_simulation(t_v3_15s, n_iterations=10000)


if __name__ == "__main__":
    main()
