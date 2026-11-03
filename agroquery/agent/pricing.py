"""Model prices in USD per million tokens.

Anthropic first-party API list prices as of November 2026
(https://platform.claude.com/docs/en/about-claude/pricing). Haiku 5.5 prices
apply to prompts up to 100K tokens, which covers every prompt in this study.
Local models served by Ollama cost $0 in API fees.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_AS_OF = "2026-11"


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
}

FREE = Price(0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    p = PRICES.get(model, FREE)
    return (input_tokens * p.input_per_mtok
            + output_tokens * p.output_per_mtok
            + cache_read_tokens * p.cache_read_per_mtok
            + cache_write_tokens * p.cache_write_per_mtok) / 1e6
