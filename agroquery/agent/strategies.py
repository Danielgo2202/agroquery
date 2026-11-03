"""The four agent strategies compared in the study.

S1  Baseline         full schema in the prompt, one call to the large model
S2  Compact schema   the small model picks relevant tables/columns from a short
                     catalog; the large model writes SQL with only those
S3  Self-correcting  S1, then execute; on error or empty result, retry with the
                     feedback (at most 3 attempts in total)
S4  Model routing    S3 with the small model; escalate to S3 with the large
                     model only if the small model never produced a query that
                     runs and returns rows

Every strategy returns a StrategyResult with the final SQL, its execution
result and every LLM call made, so tokens, cost and latency add up exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agroquery.agent.llm import LLMClient, LLMResult, ModelSpec
from agroquery.agent.schema import (
    SchemaInfo,
    render_compact_schema,
    render_full_schema,
    render_table_catalog,
)
from agroquery.agent.sql_exec import ExecResult, SQLExecutor

MAX_ATTEMPTS = 3

SQL_SYSTEM = """You are an expert PostgreSQL analyst for an agricultural monitoring platform.
Translate the user's question into one PostgreSQL SELECT query over the database below.

Rules:
- Return exactly the columns the question asks for, and nothing else.
- Use only tables and columns that exist in the schema.
- Add LIMIT only when the question asks for a top-k or a single best item.
- Do not round numbers unless the question asks for it.
- Answer with the query in a single ```sql code block and no explanation.

Database schema:
{schema}"""

SELECT_SYSTEM = """You select the parts of a database needed to answer a question.
Given the catalog below, answer with JSON only, mapping each needed table to the
list of its columns the query will use (for filtering, joining, grouping or output):
{{"tables": {{"table_name": ["column", ...]}}}}
Include every table needed for joins. Do not write SQL.

Catalog:
{catalog}"""

ERROR_FEEDBACK = """The query failed with this error:
{error}
Fix the query. Answer with the corrected query in a single ```sql code block."""

EMPTY_FEEDBACK = """The query ran but returned no rows. Check filters, string values, date ranges
and joins against the schema, and fix the query. Answer with the corrected query in a
single ```sql code block."""

SQL_BLOCK = re.compile(r"```(?:sql|postgresql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """The SQL inside the last ```sql block, or the bare text if there is none."""
    blocks = SQL_BLOCK.findall(text)
    sql = blocks[-1] if blocks else text
    return sql.strip().rstrip(";").strip()


def parse_selection(text: str) -> dict[str, list[str]] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    tables = data.get("tables", data) if isinstance(data, dict) else None
    if not isinstance(tables, dict):
        return None
    return {str(t): [str(c) for c in cols] if isinstance(cols, list) else []
            for t, cols in tables.items()}


@dataclass
class StrategyResult:
    sql: str
    execution: ExecResult
    calls: list[LLMResult] = field(default_factory=list)
    attempts: int = 1
    escalated: bool = False
    notes: dict = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def llm_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)


@dataclass
class Agent:
    """Holds what the strategies share: models, schema renderings, LLM and DB."""

    llm: LLMClient
    executor: SQLExecutor
    schema: SchemaInfo
    small: ModelSpec
    large: ModelSpec

    def __post_init__(self):
        self.full_schema = render_full_schema(self.schema)
        self.catalog = render_table_catalog(self.schema)

    # -- building blocks -------------------------------------------------------
    def _generate(self, model: ModelSpec, schema_text: str, question: str,
                  sample: int) -> tuple[str, LLMResult, list[dict]]:
        messages = [{"role": "user", "content": question}]
        call = self.llm.complete(model, SQL_SYSTEM.format(schema=schema_text), messages, sample)
        return extract_sql(call.text), call, messages

    def _self_correct(self, model: ModelSpec, question: str, sample: int) -> StrategyResult:
        system = SQL_SYSTEM.format(schema=self.full_schema)
        messages = [{"role": "user", "content": question}]
        calls: list[LLMResult] = []
        sql, result = "", ExecResult(error="no attempt")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            call = self.llm.complete(model, system, messages, sample)
            calls.append(call)
            sql = extract_sql(call.text)
            result = self.executor.run(sql)
            if result.ok and not result.empty:
                break
            if attempt < MAX_ATTEMPTS:
                feedback = (ERROR_FEEDBACK.format(error=result.error) if not result.ok
                            else EMPTY_FEEDBACK)
                messages = messages + [
                    {"role": "assistant", "content": call.text},
                    {"role": "user", "content": feedback},
                ]
        return StrategyResult(sql, result, calls, attempts=len(calls))

    # -- strategies ------------------------------------------------------------
    def s1_baseline(self, question: str, sample: int = 0) -> StrategyResult:
        sql, call, _ = self._generate(self.large, self.full_schema, question, sample)
        return StrategyResult(sql, self.executor.run(sql), [call])

    def s2_compact_schema(self, question: str, sample: int = 0) -> StrategyResult:
        pick = self.llm.complete(
            self.small,
            SELECT_SYSTEM.format(catalog=self.catalog),
            [{"role": "user", "content": question}],
            sample,
        )
        selection = parse_selection(pick.text)
        valid = {t: c for t, c in (selection or {}).items() if t in self.schema.tables}
        if valid:
            schema_text = render_compact_schema(self.schema, valid)
        else:  # unusable selection: fall back to the full schema
            schema_text = self.full_schema
        sql, call, _ = self._generate(self.large, schema_text, question, sample)
        return StrategyResult(
            sql, self.executor.run(sql), [pick, call],
            notes={"selected_tables": sorted(valid), "selection_fallback": not valid},
        )

    def s3_self_correcting(self, question: str, sample: int = 0) -> StrategyResult:
        return self._self_correct(self.large, question, sample)

    def s4_model_routing(self, question: str, sample: int = 0) -> StrategyResult:
        small = self._self_correct(self.small, question, sample)
        if small.execution.ok and not small.execution.empty:
            small.notes["answered_by"] = self.small.label
            return small
        large = self._self_correct(self.large, question, sample)
        return StrategyResult(
            large.sql, large.execution, small.calls + large.calls,
            attempts=small.attempts + large.attempts, escalated=True,
            notes={"answered_by": self.large.label},
        )

    def run(self, strategy: str, question: str, sample: int = 0) -> StrategyResult:
        return STRATEGIES[strategy](self, question, sample)


STRATEGIES = {
    "S1": Agent.s1_baseline,
    "S2": Agent.s2_compact_schema,
    "S3": Agent.s3_self_correcting,
    "S4": Agent.s4_model_routing,
}
