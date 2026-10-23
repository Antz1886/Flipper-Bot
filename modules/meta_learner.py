"""
modules/meta_learner.py — Bayesian Continual Learning & Meta-Conviction Engine
Implements Marcos López de Prado's Meta-Labeling concept via Bayesian Contextual Bandits
(Thompson Sampling / Beta-Binomial updating with exponential recency decay).
Continually learns from settled trades (both wins and losses) to:
  1. Dynamically score setups by Bayesian posterior expectancy.
  2. Veto setups whose context signatures exhibit negative empirical expectancy (E[R] < 0).
  3. Scale position sizing adaptively (0.50% - 1.00% risk) based on conviction.
"""

import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any

import config
from modules.trade_learner import get_learner_db

log = logging.getLogger(__name__)

LEARNER_STATE_FILE = Path("data/meta_learner_state.json")

PRIOR_ALPHA = 2.0  # Weakly informative prior (50% initial expectation)
PRIOR_BETA = 2.0
RECENCY_DECAY = 0.96  # Exponential decay per trade batch to prioritize recent market regimes
MIN_TRADES_FOR_VETO = 3
VETO_WINRATE_THRESHOLD = 0.33  # E[R] breakeven with 1:2.0 R:R is 33.3%


@dataclass
class BayesianClusterState:
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    trades_count: int = 0
    total_pnl_usd: float = 0.0

    @property
    def expected_winrate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def expected_r_multiple(self) -> float:
        rr = getattr(config, "RISK_REWARD_RATIO", 2.0)
        p = self.expected_winrate
        return (p * rr) - ((1.0 - p) * 1.0)


class MetaLearner:
    def __init__(self, state_file: Path = LEARNER_STATE_FILE):
        self.state_file = state_file
        self.clusters: Dict[str, BayesianClusterState] = {}
        self.load_state()

    def _cluster_key(self, symbol: str, session_phase: str, macro_pool_type: str) -> str:
        return f"{symbol}:{session_phase}:{macro_pool_type}".upper()

    def load_state(self):
        """Load Bayesian cluster counts from disk or initialize from DB history."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.clusters[k] = BayesianClusterState(
                        alpha=v.get("alpha", PRIOR_ALPHA),
                        beta=v.get("beta", PRIOR_BETA),
                        trades_count=v.get("trades_count", 0),
                        total_pnl_usd=v.get("total_pnl_usd", 0.0),
                    )
                log.info(f"Loaded {len(self.clusters)} Bayesian contextual learning clusters from disk.")
                return
            except Exception as e:
                log.warning(f"Could not parse meta_learner_state.json: {e}")

        # Initialize from database history if state file absent
        self._replay_history_from_db()

    def save_state(self):
        """Persist Bayesian cluster states to disk."""
        os.makedirs(self.state_file.parent, exist_ok=True)
        data = {
            k: {
                "alpha": round(v.alpha, 4),
                "beta": round(v.beta, 4),
                "trades_count": v.trades_count,
                "total_pnl_usd": round(v.total_pnl_usd, 2),
                "expected_winrate": round(v.expected_winrate, 4),
                "expected_r": round(v.expected_r_multiple, 4),
            }
            for k, v in self.clusters.items()
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.debug(f"Persisted {len(self.clusters)} Bayesian learning clusters.")

    def _replay_history_from_db(self):
        """Replay historical trades from SQLite to initialize cluster distributions."""
        db = get_learner_db()
        trades = db.get_recent_trades(limit=200)
        # Replay oldest to newest
        for t in reversed(trades):
            self.learn_from_trade(
                symbol=t["symbol"],
                session_phase=t["session_phase"],
                macro_pool_type=t["macro_pool_type"],
                pnl_usd=t["pnl_usd"],
                r_multiple=t["r_multiple"],
                exit_reason=t["exit_reason"],
                save=False,
            )
        self.save_state()

    def learn_from_trade(
        self,
        symbol: str,
        session_phase: str,
        macro_pool_type: str,
        pnl_usd: float,
        r_multiple: float,
        exit_reason: str,
        save: bool = True,
    ):
        """
        Bayesian Online Updating:
        Incorporates trade outcome into cluster posterior.
        - Take Profit (Win): Adds +2.0 to alpha
        - Deal Cancellation (Loss Salvaged): Adds +0.35 to beta (minor penalty)
        - Structural Stop Loss (Full Loss): Adds +1.50 to beta
        """
        cluster_key = self._cluster_key(symbol, session_phase, macro_pool_type)
        if cluster_key not in self.clusters:
            self.clusters[cluster_key] = BayesianClusterState()

        st = self.clusters[cluster_key]

        # Apply recency decay across cluster before update
        st.alpha = (st.alpha - PRIOR_ALPHA) * RECENCY_DECAY + PRIOR_ALPHA
        st.beta = (st.beta - PRIOR_BETA) * RECENCY_DECAY + PRIOR_BETA

        if r_multiple > 0 or exit_reason == "WIN_TP":
            st.alpha += min(max(r_multiple, 1.0), 2.5)
        elif exit_reason == "LOSS_DEAL_CANCELLATION":
            st.beta += 0.35  # Salvaged exit, small penalty
        else:
            st.beta += 1.50  # Full structural stop loss

        st.trades_count += 1
        st.total_pnl_usd += pnl_usd

        # Also update symbol-wide marginal cluster
        sym_key = f"{symbol}:ALL:ALL".upper()
        if sym_key not in self.clusters:
            self.clusters[sym_key] = BayesianClusterState()
        sym_st = self.clusters[sym_key]
        if r_multiple > 0:
            sym_st.alpha += 1.0
        else:
            sym_st.beta += 1.0
        sym_st.trades_count += 1
        sym_st.total_pnl_usd += pnl_usd

        log.info(
            f"📈 [CONTINUAL LEARNING UPDATE] Cluster {cluster_key} updated: "
            f"Posterior WinRate = {st.expected_winrate*100:.1f}% | "
            f"Expected R = {st.expected_r_multiple:+.2f}R | "
            f"Trades: {st.trades_count}"
        )

        if save:
            self.save_state()

    def evaluate_setup_conviction(
        self,
        symbol: str,
        session_phase: str,
        macro_pool_type: str,
        displacement_ratio: float = 1.0,
        htf_depth_pct: float = 0.50,
    ) -> Tuple[bool, float, float, str]:
        """
        Evaluate an SMC setup against the Bayesian learning model.
        Returns:
          (approved: bool, conviction_score: float, recommended_risk_pct: float, reason: str)
        """
        cluster_key = self._cluster_key(symbol, session_phase, macro_pool_type)
        cluster = self.clusters.get(cluster_key)
        sym_cluster = self.clusters.get(f"{symbol}:ALL:ALL".upper())

        # Blend cluster with symbol marginal for Bayesian shrinkage
        if cluster and cluster.trades_count >= 1:
            win_rate = cluster.expected_winrate
            trades = cluster.trades_count
        elif sym_cluster and sym_cluster.trades_count >= 1:
            win_rate = sym_cluster.expected_winrate
            trades = sym_cluster.trades_count
        else:
            win_rate = 0.50  # Prior expectation
            trades = 0

        # Adjust score for structural conviction factors
        disp_boost = min(0.10, max(-0.05, (displacement_ratio - 1.0) * 0.15))
        # Deep discount (< 35%) or deep premium (> 65%) gives structural boost
        depth_boost = 0.05 if (htf_depth_pct < 0.35 or htf_depth_pct > 0.65) else 0.0

        conviction_score = min(0.95, max(0.10, win_rate + disp_boost + depth_boost))

        # Check Veto Rule
        if trades >= MIN_TRADES_FOR_VETO and win_rate < VETO_WINRATE_THRESHOLD:
            reason = (
                f"Negative empirical expectancy in {cluster_key}: "
                f"Posterior win rate {win_rate*100:.1f}% < {VETO_WINRATE_THRESHOLD*100:.0f}% over {trades} trades. "
                f"Vetoing to prevent repeat loss."
            )
            return False, conviction_score, 0.0, reason

        # Sizing recommendations based on conviction score
        base_risk = getattr(config, "RISK_PER_TRADE_PCT", 0.01)
        if conviction_score >= 0.60:
            rec_risk = base_risk           # Full 1.0% risk
            conv_level = "HIGH CONVICTION"
        elif conviction_score >= 0.45:
            rec_risk = base_risk * 0.75    # 0.75% risk
            conv_level = "MODERATE CONVICTION"
        else:
            rec_risk = base_risk * 0.50    # 0.50% defensive risk
            conv_level = "DEFENSIVE CONVICTION"

        reason = (
            f"Authorized [{conv_level}] | Score: {conviction_score*100:.1f}% | "
            f"Expected WinRate: {win_rate*100:.1f}% ({trades} prior trades in cluster) | "
            f"Risk: {rec_risk*100:.2f}%"
        )
        return True, conviction_score, rec_risk, reason


_meta_learner: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner
