"""Strategy logic with a scripted fake LLM and a fake executor (no API, no DB)."""

from agroquery.agent.llm import LLMResult, ModelSpec
from agroquery.agent.schema import Column, SchemaInfo, Table, render_compact_schema
from agroquery.agent.sql_exec import ExecResult
from agroquery.agent.strategies import Agent, extract_sql, parse_selection

SMALL = ModelSpec("fake", "small")
LARGE = ModelSpec("fake", "large")


def schema() -> SchemaInfo:
    farms = Table("farms", "Farms.", [
        Column("farm_id", "integer", primary_key=True),
        Column("name", "text", "Farm name."),
        Column("region_id", "integer", references="regions(region_id)"),
        Column("joined_date", "date"),
    ])
    regions = Table("regions", "Regions.", [
        Column("region_id", "integer", primary_key=True),
        Column("name", "text", values=["Meta", "Huila"]),
    ])
    return SchemaInfo({"farms": farms, "regions": regions}, glossary="Glossary.")


class FakeLLM:
    """Returns scripted replies per model, recording every call."""

    def __init__(self, replies: dict[str, list[str]]):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls = []

    def complete(self, spec, system, messages, sample=0):
        self.calls.append((spec.model, system, messages))
        text = self.replies[spec.model].pop(0)
        return LLMResult(text, spec.model, input_tokens=100, output_tokens=10,
                         cost_usd=0.001 if spec.model == "large" else 0.0001, latency_s=0.5)


class FakeExecutor:
    """'SELECT ok' returns a row, 'SELECT empty' returns none, anything else errors."""

    def run(self, sql):
        if sql == "SELECT ok":
            return ExecResult(["x"], [(1,)])
        if sql == "SELECT empty":
            return ExecResult(["x"], [])
        return ExecResult(error="SyntaxError: bad query")


def agent(replies):
    return Agent(FakeLLM(replies), FakeExecutor(), schema(), SMALL, LARGE)


def block(sql):
    return f"```sql\n{sql};\n```"


def test_extract_sql_takes_last_block_and_strips_semicolon():
    assert extract_sql("x\n```sql\nSELECT 1;\n```\n```sql\nSELECT 2;\n```") == "SELECT 2"
    assert extract_sql("SELECT 3;") == "SELECT 3"


def test_parse_selection_handles_fences_and_garbage():
    text = 'Sure:\n```json\n{"tables": {"farms": ["name"], "regions": ["name"]}}\n```'
    assert parse_selection(text) == {"farms": ["name"], "regions": ["name"]}
    assert parse_selection("no json here") is None


def test_compact_schema_keeps_keys_and_drops_other_columns():
    text = render_compact_schema(schema(), {"farms": ["name"]})
    assert "farm_id" in text and "region_id" in text and "name" in text
    assert "joined_date" not in text
    assert "CREATE TABLE regions" not in text


def test_s1_single_call_to_large_model():
    a = agent({"large": [block("SELECT ok")]})
    r = a.s1_baseline("q")
    assert r.execution.ok and len(r.calls) == 1 and r.calls[0].model == "large"


def test_s2_uses_small_model_for_selection_then_compact_schema():
    a = agent({"small": ['{"tables": {"farms": ["name"]}}'], "large": [block("SELECT ok")]})
    r = a.s2_compact_schema("q")
    assert [c.model for c in r.calls] == ["small", "large"]
    sql_prompt = a.llm.calls[1][1]
    assert "CREATE TABLE farms" in sql_prompt and "CREATE TABLE regions" not in sql_prompt
    assert r.notes["selected_tables"] == ["farms"]


def test_s2_falls_back_to_full_schema_on_bad_selection():
    a = agent({"small": ["I cannot"], "large": [block("SELECT ok")]})
    r = a.s2_compact_schema("q")
    assert r.notes["selection_fallback"]
    assert "CREATE TABLE regions" in a.llm.calls[1][1]


def test_s3_retries_with_error_then_empty_feedback():
    a = agent({"large": [block("SELEC bad"), block("SELECT empty"), block("SELECT ok")]})
    r = a.s3_self_correcting("q")
    assert r.attempts == 3 and r.execution.ok and not r.execution.empty
    last_messages = a.llm.calls[-1][2]
    assert "SyntaxError" in last_messages[2]["content"]
    assert "returned no rows" in last_messages[4]["content"]


def test_s3_stops_after_max_attempts():
    a = agent({"large": [block("bad")] * 3})
    r = a.s3_self_correcting("q")
    assert r.attempts == 3 and not r.execution.ok


def test_s4_no_escalation_when_small_model_succeeds():
    a = agent({"small": [block("SELECT ok")]})
    r = a.s4_model_routing("q")
    assert not r.escalated and [c.model for c in r.calls] == ["small"]


def test_s4_escalates_after_small_model_fails():
    a = agent({"small": [block("bad")] * 3, "large": [block("SELECT ok")]})
    r = a.s4_model_routing("q")
    assert r.escalated and r.attempts == 4
    assert [c.model for c in r.calls] == ["small"] * 3 + ["large"]
    assert abs(r.cost_usd - (3 * 0.0001 + 0.001)) < 1e-12
