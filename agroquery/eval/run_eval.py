"""Run strategies x questions x repeats and record accuracy, tokens, cost, latency.

Output (results/<run>/):
  results.csv   one row per (question, strategy, repeat)
  traces.jsonl  final SQL, per-call tokens and errors, for inspection
  meta.json     models, prices date, settings and totals
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime

from agroquery.agent.llm import BudgetExceeded, LLMClient, ModelSpec
from agroquery.agent.pricing import PRICES_AS_OF
from agroquery.agent.schema import introspect
from agroquery.agent.sql_exec import SQLExecutor
from agroquery.agent.strategies import STRATEGIES, Agent
from agroquery.config import MAX_USD, RESULTS_DIR, ROOT
from agroquery.eval.benchmark import load_questions
from agroquery.eval.compare import results_match

PRESETS = {
    "anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5"),
    "ollama": ("qwen2.5-coder:3b", "qwen2.5-coder:3b"),
}

FIELDS = [
    "run", "provider", "strategy", "question_id", "difficulty", "repeat", "correct",
    "exec_ok", "empty_result", "attempts", "escalated", "n_calls", "input_tokens",
    "output_tokens", "cost_usd", "llm_latency_s", "sql_latency_s", "total_latency_s",
    "cached_calls", "models", "error",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=sorted(PRESETS), default="ollama")
    p.add_argument("--small", help="override the small model id")
    p.add_argument("--large", help="override the large model id")
    p.add_argument("--strategies", default="S1,S2,S3,S4")
    p.add_argument("--questions", help="comma-separated question ids (default: all)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--repeat", type=int, default=1, help="independent samples per question")
    p.add_argument("--run", help="run name (default: <provider>_<timestamp>)")
    p.add_argument("--max-usd", type=float, default=MAX_USD)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    small_id, large_id = PRESETS[args.provider]
    small = ModelSpec(args.provider, args.small or small_id)
    large = ModelSpec(args.provider, args.large or large_id)
    strategies = [s.strip().upper() for s in args.strategies.split(",")]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies: {sorted(unknown)}")

    questions = load_questions()
    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",")}
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    run = args.run or f"{args.provider}_{datetime.now():%Y%m%d-%H%M}"
    out_dir = RESULTS_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(max_usd=args.max_usd, use_cache=not args.no_cache)
    executor = SQLExecutor()
    agent = Agent(llm, executor, introspect(), small, large)
    gold = {q.id: executor.run(q.gold_sql) for q in questions}
    bad_gold = [qid for qid, r in gold.items() if not r.ok or r.empty]
    if bad_gold:
        raise SystemExit(f"gold queries failing, run `agroquery validate`: {bad_gold}")

    meta = {
        "run": run, "provider": args.provider, "small_model": small.model,
        "large_model": large.model, "strategies": strategies, "n_questions": len(questions),
        "repeat": args.repeat, "prices_as_of": PRICES_AS_OF, "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    total = len(questions) * len(strategies) * args.repeat
    done, correct_count, t_start = 0, 0, time.perf_counter()
    stopped_early = None

    with (open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fcsv,
          open(out_dir / "traces.jsonl", "w", encoding="utf-8") as ftr):
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        writer.writeheader()
        try:
            for rep in range(args.repeat):
                for q in questions:
                    for strat in strategies:
                        res = agent.run(strat, q.question, sample=rep)
                        ex = res.execution
                        correct = ex.ok and results_match(gold[q.id].rows, ex.rows,
                                                          q.order_matters)
                        row = {
                            "run": run, "provider": args.provider, "strategy": strat,
                            "question_id": q.id, "difficulty": q.difficulty, "repeat": rep,
                            "correct": int(correct), "exec_ok": int(ex.ok),
                            "empty_result": int(ex.empty), "attempts": res.attempts,
                            "escalated": int(res.escalated), "n_calls": len(res.calls),
                            "input_tokens": res.input_tokens,
                            "output_tokens": res.output_tokens,
                            "cost_usd": round(res.cost_usd, 8),
                            "llm_latency_s": round(res.llm_latency_s, 3),
                            "sql_latency_s": round(ex.elapsed_s, 3),
                            "total_latency_s": round(res.llm_latency_s + ex.elapsed_s, 3),
                            "cached_calls": sum(c.cached for c in res.calls),
                            "models": "+".join(c.model for c in res.calls),
                            "error": ex.error or "",
                        }
                        writer.writerow(row)
                        fcsv.flush()
                        ftr.write(json.dumps({
                            **{k: row[k] for k in ("strategy", "question_id", "repeat",
                                                   "correct")},
                            "question": q.question, "sql": res.sql, "gold_sql": q.gold_sql,
                            "error": ex.error, "rows_returned": len(ex.rows),
                            "rows_expected": len(gold[q.id].rows), "notes": res.notes,
                            "calls": [{k: v for k, v in asdict(c).items() if k != "text"}
                                      for c in res.calls],
                        }, default=str) + "\n")
                        done += 1
                        correct_count += correct
                        print(f"[{done:4d}/{total}] {q.id} {strat} rep{rep} "
                              f"{'OK ' if correct else 'BAD'} tokens={res.input_tokens}"
                              f"+{res.output_tokens} ${res.cost_usd:.5f} "
                              f"{res.llm_latency_s:.1f}s  (spent ${llm.spent_usd:.4f})")
        except BudgetExceeded as exc:
            stopped_early = str(exc)
            print(f"\nSTOPPED: {exc}")
        except KeyboardInterrupt:
            stopped_early = "interrupted by user"
            print("\nSTOPPED: interrupted")
        finally:
            executor.close()

    meta.update({
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "completed_items": done, "planned_items": total,
        "fresh_spend_usd": round(llm.spent_usd, 6),
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "stopped_early": stopped_early,
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    acc = correct_count / done if done else 0.0
    print(f"\nRun {run}: {done}/{total} items, accuracy {acc:.1%}, "
          f"fresh spend ${llm.spent_usd:.4f} -> {out_dir}")
    return 0 if stopped_early is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
