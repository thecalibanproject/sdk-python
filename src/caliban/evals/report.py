"""Render an :class:`EvalReport` as JSON and Markdown."""

from __future__ import annotations

from pathlib import Path

from .runner import EvalReport

__all__ = ["to_json", "to_markdown", "write_reports"]


def to_json(report: EvalReport) -> str:
    return report.model_dump_json(indent=2)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f} ms"


def _usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.6f}"


def _cell(text: str, limit: int = 160) -> str:
    one = " ".join(text.split())
    if len(one) > limit:
        one = one[: limit - 1] + "…"
    return one.replace("|", "\\|").replace("`", "'")


def to_markdown(report: EvalReport) -> str:
    s = report.summary
    k = report.k
    target = f"model `{report.target.model}`"
    if report.target.node:
        target += f", node `{report.target.node}`"
    lines = [
        f"# Eval report: {report.dataset}",
        "",
        f"- Target: {target}",
        f"- k = {k}, trials per case = {report.n}",
        f"- Started: {report.started_at.isoformat()} ({report.duration_s:.1f}s)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| pass@1 | {_pct(s.pass_at_1)} |",
        f"| pass^{k} | {_pct(s.pass_hat_k)} |",
        f"| Cost per success | {_usd(s.cost_per_success_usd)} |",
        f"| Total cost | {_usd(s.cost_total_usd)} |",
        f"| Latency p50 | {_ms(s.latency_p50_ms)} |",
        f"| Latency p95 | {_ms(s.latency_p95_ms)} |",
        f"| Cases / trials / successes | {s.cases} / {s.trials} / {s.successes} |",
        f"| Errors | {s.errors} |",
        f"| Cache hits | {s.cache_hits} |",
        "",
    ]
    if k > 1:
        lines.insert(
            lines.index(f"| pass^{k} | {_pct(s.pass_hat_k)} |") + 1,
            f"| pass@{k} | {_pct(s.pass_at_k)} |",
        )
    if report.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report.warnings] + [""]
    lines += [
        "## Cases",
        "",
        f"| Case | Passed | pass@1 | pass^{k} | p50 |",
        "|---|---|---|---|---|",
    ]
    for c in report.cases:
        lines.append(
            f"| `{c.id}` | {c.successes}/{c.n} | {_pct(c.pass_at_1)} | "
            f"{_pct(c.pass_hat_k)} | {_ms(c.latency_p50_ms)} |"
        )
    lines.append("")
    failures = report.failures()
    lines += ["## Failures", ""]
    if not failures:
        lines.append("None.")
    for case_id, t in failures:
        why = t.error if t.error and not t.failures else "; ".join(t.failures) or "failed"
        line = f"- `{case_id}` trial {t.index}: {_cell(why)}"
        if t.output is not None:
            line += f"; output: `{_cell(t.output, 120)}`"
        if t.request_id:
            line += f" (request {t.request_id})"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def write_reports(report: EvalReport, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(to_json(report) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
