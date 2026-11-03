"""One interface over two LLM backends (Anthropic Claude API and local Ollama),
with token, cost and latency accounting, an on-disk response cache and a hard
spending cap.

The cache key includes a `sample` index, so repeated runs of the benchmark
(`--repeat K`) draw fresh samples while re-running analysis on the same runs
costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

from agroquery.agent.pricing import cost_usd
from agroquery.config import CACHE_DIR, MAX_USD, OLLAMA_HOST

MAX_OUTPUT_TOKENS = 4096


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    stop_reason: str = ""
    cached: bool = False


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "anthropic" | "ollama"
    model: str

    @property
    def label(self) -> str:
        return self.model


class LLMClient:
    """Routes calls to the right backend and keeps the books."""

    def __init__(self, max_usd: float = MAX_USD, use_cache: bool = True):
        self.max_usd = max_usd
        self.use_cache = use_cache
        self.spent_usd = 0.0  # fresh (non-cached) spend in this process
        self._anthropic = None
        self._ollama = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -- public ---------------------------------------------------------------
    def complete(self, spec: ModelSpec, system: str, messages: list[dict],
                 sample: int = 0) -> LLMResult:
        key = self._cache_key(spec, system, messages, sample)
        path = CACHE_DIR / f"{key}.json"
        if self.use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(**{**data, "cached": True})
        if self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"Spent ${self.spent_usd:.4f} >= MAX_USD=${self.max_usd}")

        if spec.provider == "anthropic":
            result = self._call_anthropic(spec.model, system, messages)
        elif spec.provider == "ollama":
            result = self._call_ollama(spec.model, system, messages)
        else:
            raise ValueError(f"unknown provider {spec.provider!r}")

        self.spent_usd += result.cost_usd
        if self.use_cache:
            path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        return result

    # -- backends -------------------------------------------------------------
    def _call_anthropic(self, model: str, system: str, messages: list[dict]) -> LLMResult:
        import anthropic

        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(max_retries=5)
        # Thinking is switched off so that every strategy is compared on the
        # tokens it actually needs to write SQL. Claude Sonnet 5.5 rejects
        # {"type": "disabled"}; "between_tools" is its no-thinking mode for a
        # request without tools.
        thinking = {"type": "between_tools"} if model.startswith("claude-sonnet-5-5") \
            else {"type": "disabled"}
        t0 = time.perf_counter()
        response = self._anthropic.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=messages,
            thinking=thinking,
            output_config={"effort": "medium"},
        )
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return LLMResult(
            text=text,
            model=model,
            input_tokens=u.input_tokens + cache_read + cache_write,
            output_tokens=u.output_tokens,
            cost_usd=cost_usd(model, u.input_tokens, u.output_tokens, cache_read, cache_write),
            latency_s=latency,
            stop_reason=response.stop_reason or "",
        )

    def _call_ollama(self, model: str, system: str, messages: list[dict]) -> LLMResult:
        import ollama

        if self._ollama is None:
            self._ollama = ollama.Client(host=OLLAMA_HOST)
        t0 = time.perf_counter()
        response = self._ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
        )
        latency = time.perf_counter() - t0
        return LLMResult(
            text=response.message.content or "",
            model=model,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            cost_usd=0.0,
            latency_s=latency,
            stop_reason=response.done_reason or "",
        )

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _cache_key(spec: ModelSpec, system: str, messages: list[dict], sample: int) -> str:
        payload = json.dumps(
            {"p": spec.provider, "m": spec.model, "s": system, "msgs": messages,
             "sample": sample, "max_tokens": MAX_OUTPUT_TOKENS},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
