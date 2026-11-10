"""AgroQuery command line.

    agroquery up                     start PostgreSQL (docker compose)
    agroquery data [--refresh]       download weather, generate and load the database
    agroquery validate               check the gold SQL of the benchmark
    agroquery eval [options]         run the benchmark (see `agroquery eval -h`)
    agroquery analyze [--run NAME]   results table + figures
    agroquery ask "question" [-s S3] answer one question
    agroquery models                 list models of the OpenAI-compatible API in .env
    agroquery all [eval options]     everything above, in order
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from agroquery.config import ROOT


def cmd_up() -> int:
    return subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=ROOT).returncode


def cmd_data(refresh: bool = False) -> int:
    from agroquery.data import fetch_weather, load

    fetch_weather.main(force=refresh)
    load.main()
    return 0


def cmd_ask(argv: list[str]) -> int:
    from agroquery.agent.llm import LLMClient, ModelSpec
    from agroquery.agent.schema import introspect
    from agroquery.agent.sql_exec import SQLExecutor
    from agroquery.agent.strategies import STRATEGIES, Agent
    from agroquery.eval.run_eval import PRESETS

    p = argparse.ArgumentParser(prog="agroquery ask")
    p.add_argument("question")
    p.add_argument("-s", "--strategy", default="S3", choices=sorted(STRATEGIES))
    p.add_argument("--provider", choices=sorted(PRESETS), default=default_provider())
    args = p.parse_args(argv)

    small, large = (ModelSpec(args.provider, m) for m in PRESETS[args.provider])
    executor = SQLExecutor()
    agent = Agent(LLMClient(use_cache=False), executor, introspect(), small, large)
    res = agent.run(args.strategy, args.question)
    print(f"\n-- {args.strategy} via {res.calls[-1].model}, {len(res.calls)} call(s), "
          f"{res.input_tokens}+{res.output_tokens} tokens, ${res.cost_usd:.5f}, "
          f"{res.llm_latency_s:.1f}s\n{res.sql}\n")
    ex = res.execution
    if not ex.ok:
        print(f"ERROR: {ex.error}")
        return 1
    print(" | ".join(ex.columns))
    for row in ex.rows[:20]:
        print(" | ".join(str(v) for v in row))
    if len(ex.rows) > 20:
        print(f"... {len(ex.rows) - 20} more rows")
    executor.close()
    return 0


def default_provider() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("COMPAT_API_KEY"):
        return "compat"
    return "ollama"


def main(argv: list[str] | None = None) -> int:
    from agroquery.db import DatabaseUnavailable

    try:
        return _main(argv)
    except DatabaseUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "up":
        return cmd_up()
    if cmd == "data":
        return cmd_data(refresh="--refresh" in rest)
    if cmd == "validate":
        from agroquery.eval import validate

        return validate.main()
    if cmd == "eval":
        from agroquery.eval import run_eval

        if not any(a.startswith("--provider") for a in rest):
            rest = ["--provider", default_provider(), *rest]
        return run_eval.main(rest)
    if cmd == "analyze":
        from agroquery.eval import analyze

        return analyze.main(rest)
    if cmd == "ask":
        return cmd_ask(rest)
    if cmd == "models":
        from agroquery.agent.llm import LLMClient

        for model_id in LLMClient().list_compat_models():
            print(model_id)
        return 0
    if cmd == "all":
        for step in (cmd_up, cmd_data):
            if step() != 0:
                return 1
        if main(["validate"]) != 0:
            return 1
        if main(["eval", *rest]) not in (0,):
            return 1
        return main(["analyze"])
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
