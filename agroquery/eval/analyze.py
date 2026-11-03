"""Summarize a run: markdown results table + two figures.

    agroquery analyze                 # latest run in results/
    agroquery analyze --run <name>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agroquery.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from agroquery.eval.benchmark import DIFFICULTIES  # noqa: E402

STRATEGY_NAMES = {
    "S1": "Baseline",
    "S2": "Compact schema",
    "S3": "Self-correcting",
    "S4": "Model routing",
}
# Categorical slots 1-4 of the validated reference palette (light mode),
# assigned to strategies in fixed order so colors match across figures.
COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def latest_run() -> Path:
    runs = [p.parent for p in RESULTS_DIR.glob("*/results.csv")]
    if not runs:
        raise SystemExit("no runs in results/ - run `agroquery eval` first")
    return max(runs, key=lambda p: (p / "results.csv").stat().st_mtime)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("strategy"):
        k, n = int(g.correct.sum()), len(g)
        lo, hi = wilson(k, n)
        row = {
            "strategy": s,
            "name": STRATEGY_NAMES.get(s, s),
            "n": n,
            "accuracy": k / n,
            "ci_low": lo,
            "ci_high": hi,
            **{f"acc_{d}": g[g.difficulty == d].correct.mean() for d in DIFFICULTIES},
            "avg_input_tokens": g.input_tokens.mean(),
            "avg_output_tokens": g.output_tokens.mean(),
            "avg_cost_usd": g.cost_usd.mean(),
            "total_cost_usd": g.cost_usd.sum(),
            "cost_per_correct_usd": g.cost_usd.sum() / k if k else float("nan"),
            "latency_p50_s": g.total_latency_s.median(),
            "latency_p95_s": g.total_latency_s.quantile(0.95),
            "avg_attempts": g.attempts.mean(),
            "escalation_rate": g.escalated.mean(),
            "exec_error_rate": 1 - g.exec_ok.mean(),
        }
        if "repeat" in g and g["repeat"].nunique() > 1:
            per_rep = g.groupby("repeat").correct.mean()
            row["acc_std_across_repeats"] = per_rep.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy").sort_index()


def markdown_table(summary: pd.DataFrame) -> str:
    show_esc = "S4" in summary.index
    head = ("| Strategy | Accuracy (95% CI) | Easy | Medium | Hard | Avg input tokens "
            "| Avg output tokens | Avg cost / query | Cost / correct answer | Latency p50 "
            + ("| Escalated " if show_esc else "") + "|")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for s, r in summary.iterrows():
        esc = (f"| {r.escalation_rate:.0%} " if s == "S4" else "| - ") if show_esc else ""
        lines.append(
            f"| {s} {r['name']} | {r.accuracy:.1%} ({r.ci_low:.0%}-{r.ci_high:.0%}) "
            f"| {r.acc_easy:.0%} | {r.acc_medium:.0%} | {r.acc_hard:.0%} "
            f"| {r.avg_input_tokens:,.0f} | {r.avg_output_tokens:,.0f} "
            f"| ${r.avg_cost_usd:.5f} | ${r.cost_per_correct_usd:.5f} "
            f"| {r.latency_p50_s:.1f} s {esc}|"
        )
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy_vs_cost(summary: pd.DataFrame, out: Path, title_suffix: str) -> None:
    use_cost = summary.avg_cost_usd.sum() > 0
    x = summary.avg_cost_usd if use_cost else summary.avg_input_tokens + summary.avg_output_tokens
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160, facecolor=SURFACE)
    _style(ax)
    for s, r in summary.iterrows():
        xv = x[s]
        ax.errorbar(xv, r.accuracy * 100,
                    yerr=[[100 * (r.accuracy - r.ci_low)], [100 * (r.ci_high - r.accuracy)]],
                    color=GRID, elinewidth=2, capsize=0, zorder=1)
        ax.scatter(xv, r.accuracy * 100, s=90, color=COLORS.get(s, INK_2),
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=f"{s} {r['name']}")
        ax.annotate(f"{s} {r['name']}", (xv, r.accuracy * 100), xytext=(8, 6),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Average cost per question (USD)" if use_cost
                  else "Average tokens per question (input + output)", color=INK_2)
    ax.set_ylabel("Execution accuracy (%)", color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0, right=x.max() * 1.35)
    ax.set_title(f"Accuracy vs. cost per strategy{title_suffix}", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_tokens_by_difficulty(df: pd.DataFrame, out: Path, title_suffix: str) -> None:
    tok = (df.assign(tokens=df.input_tokens + df.output_tokens)
             .groupby(["difficulty", "strategy"]).tokens.mean().unstack("strategy")
             .reindex(DIFFICULTIES))
    strategies = list(tok.columns)
    n = len(strategies)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    for i, s in enumerate(strategies):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(DIFFICULTIES))]
        ax.bar(xs, tok[s], width=width, color=COLORS.get(s, INK_2), edgecolor=SURFACE,
               linewidth=1.5, label=f"{s} {STRATEGY_NAMES.get(s, s)}", zorder=2)
    ax.set_xticks(range(len(DIFFICULTIES)), [d.capitalize() for d in DIFFICULTIES])
    ax.set_ylabel("Average tokens per question", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(f"Tokens per question by difficulty{title_suffix}", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncols=2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", help="run name under results/ (default: latest)")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = p.parse_args(argv)

    run_dir = RESULTS_DIR / args.run if args.run else latest_run()
    df = pd.read_csv(run_dir / "results.csv")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summary = summarize(df)

    models = (f" ({meta['small_model']} / {meta['large_model']})"
              if meta.get("small_model") else "")
    table = markdown_table(summary)
    header = (f"Run `{run_dir.name}`{models}: {df.question_id.nunique()} questions x "
              f"{df['repeat'].nunique()} repeat(s), {len(df)} evaluations, "
              f"total cost ${df.cost_usd.sum():.4f}.")
    (run_dir / "summary.md").write_text(header + "\n\n" + table + "\n", encoding="utf-8")
    summary.to_csv(run_dir / "summary.csv")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_cost(summary, args.figures_dir / "accuracy_vs_cost.png", models)
    plot_tokens_by_difficulty(df, args.figures_dir / "tokens_by_difficulty.png", models)
    print(header + "\n\n" + table)
    print(f"\nWrote {run_dir / 'summary.md'} and figures in {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
