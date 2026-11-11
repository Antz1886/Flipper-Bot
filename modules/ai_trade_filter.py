"""
modules/ai_trade_filter.py — Multi-Agent AI Pre-Trade Gatekeeper
Integrates TradingAgents & Google Gemini with Flipper-Bot:
- Adversarial trade vetting (checks for chop traps, exhaustion, and trend fights).
- High-performance caching (TTL-based) to ensure zero latency lag on tick execution.
- Graceful degradation: falls back seamlessly if API limits or network timeouts occur.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from tradingagents.llm_clients.base_client import normalize_content
from tradingagents.llm_clients.factory import create_llm_client

import config

log = logging.getLogger("ai_trade_filter")

_SYSTEM_PROMPT = """You are an Institutional Quantitative Risk & Trade Gatekeeper for high-volatility financial markets and synthetic indices.
Your sole job is to protect trading capital by filtering out low-probability trades, false breakouts, and counter-trend traps.

Evaluation Rules:
1. VETO if the market is in low-efficiency consolidation or chop (Kaufman Efficiency Ratio < 0.25, flat RSI around 45-55, near zero Z-score).
2. VETO counter-trend mean reversion if the higher-timeframe trend is strong (ER > 0.60) unless a confirmed structural reversal is proven.
3. APPROVE when the proposed direction aligns with higher-timeframe trend momentum, pullback to key levels is complete, and micro-tick reversal confirms entry timing.
4. APPROVE statistical extreme mean-reversion ONLY when Z-score >= 2.0 or <= -2.0 with confirmed tick reversal exhaustion.

You MUST respond strictly with valid JSON conforming to this schema:
{
  "approved": boolean,
  "confidence": float between 0.0 and 1.0,
  "reason": "Concise 1-2 sentence explanation of your decision"
}
Do NOT include markdown formatting or extra text outside the JSON block.
"""


class AITradeFilter:
    """Pre-trade gatekeeper leveraging TradingAgents cognitive models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_ttl_s: float = 45.0,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.cache_ttl_s = cache_ttl_s
        self.model_name = model_name or getattr(config, "AI_FILTER_MODEL", "gemini-2.5-flash")
        self._llm = None
        self._cache: Dict[str, Tuple[float, bool, float, str]] = {}  # key -> (timestamp, approved, confidence, reason)

    def _get_llm(self):
        """Lazy-initialize TradingAgents LLM client."""
        if self._llm is None:
            try:
                client = create_llm_client("google", self.model_name)
                self._llm = client.get_llm()
            except Exception as e:
                log.error(f"Failed to initialize TradingAgents LLM client ({self.model_name}): {e}")
                self._llm = None
        return self._llm

    def _clean_json(self, raw_text: str) -> str:
        """Strip markdown code fence blocks if present."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    async def validate_trade(
        self,
        symbol: str,
        proposed_direction: str,
        conviction: float,
        context: Dict[str, Any],
        stake: float = 2.0,
    ) -> Tuple[bool, float, str]:
        """
        Validate whether a trade should be executed.
        Returns:
            Tuple[bool, float, str]: (approved, ai_confidence, reason)
        """
        if not self.enabled:
            return True, conviction, "AI Gatekeeper disabled in configuration."

        # Check Cache to prevent repetitive LLM roundtrips on identical conditions
        cache_key = f"{symbol}:{proposed_direction.upper()}:{context.get('htf_trend', 'NEUTRAL')}"
        now = time.time()
        if cache_key in self._cache:
            ts, cached_app, cached_conf, cached_reason = self._cache[cache_key]
            if (now - ts) < self.cache_ttl_s:
                log.debug(f"AI Filter Cache Hit [{cache_key}]: {cached_app} ({cached_reason})")
                return cached_app, cached_conf, f"[CACHED] {cached_reason}"

        llm = self._get_llm()
        if llm is None:
            return True, conviction, "AI Gatekeeper bypass: LLM client unavailable."

        # Format Trade Details
        user_prompt = (
            f"Asset: {symbol}\n"
            f"Proposed Direction: {proposed_direction.upper()}\n"
            f"Technical Conviction Score: {conviction*100:.1f}%\n"
            f"Proposed Stake: ${stake:.2f} USD\n"
            f"Higher Timeframe Trend: {context.get('htf_trend', 'UNKNOWN')}\n"
            f"Dual HTF Aligned: {context.get('dual_htf_aligned', False)}\n"
            f"Kaufman Efficiency Ratio (ER): {context.get('er', 'N/A')}\n"
            f"Rolling Z-Score: {context.get('z_score', 'N/A')}\n"
            f"Tick RSI: {context.get('tick_rsi', 'N/A')}\n"
            f"Micro-Tick Action: {context.get('micro_action', 'N/A')}\n"
            f"Technical Rationale: {context.get('rationale', 'N/A')}\n\n"
            "Evaluate this trade setup and return your JSON decision."
        )

        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            # Enforce 6.0s timeout to protect tick-level execution timing
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=6.0)
            response = normalize_content(response)
            raw_content = response.content if hasattr(response, "content") else str(response)
            cleaned = self._clean_json(raw_content)
            data = json.loads(cleaned)

            approved = bool(data.get("approved", True))
            ai_conf = float(data.get("confidence", conviction))
            reason = str(data.get("reason", "No reason provided."))

            # Update cache
            self._cache[cache_key] = (now, approved, ai_conf, reason)
            return approved, ai_conf, reason

        except Exception as e:
            log.warning(f"AI Gatekeeper evaluation error ({e}). Bypassing to quantitative fallback.")
            return True, conviction, f"AI Gatekeeper bypass: {e}"


# Singleton instance
_filter_instance: Optional[AITradeFilter] = None


def get_ai_trade_filter() -> AITradeFilter:
    """Retrieve singleton instance of AITradeFilter."""
    global _filter_instance
    if _filter_instance is None:
        enabled = getattr(config, "AI_TRADE_FILTER_ENABLED", True)
        model = getattr(config, "AI_FILTER_MODEL", "gemini-2.5-flash")
        ttl = float(getattr(config, "AI_FILTER_CACHE_TTL_S", 45.0))
        _filter_instance = AITradeFilter(model_name=model, cache_ttl_s=ttl, enabled=enabled)
    return _filter_instance
