"""Run a golden-set dataset k times per case against a model or node."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .._client import Caliban
from ..errors import CalibanError
from ..types import CalibanOptions, CompletionUsage
from .checkers import Judge, run_check
from .dataset import Case, Dataset
from .metrics import mean, pass_at_k, pass_hat_k, percentile

__all__ = [
    "CaseResult",
    "EvalReport",
    "EvalSummary",
    "Pricing",
    "Target",
    "Trial",
    "run_eval",
]

MAX_OUTPUT_CHARS = 4000


@dataclass(frozen=True)
class Pricing:
    """Fallback pricing (USD per million tokens) when the server sends no cost header."""

    in_per_mtok: float
    out_per_mtok: float

    def cost(self, usage: CompletionUsage | None) -> float | None:
        if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
            return None
        return (
            usage.prompt_tokens * self.in_per_mtok + usage.completion_tokens * self.out_per_mtok
        ) / 1_000_000


class Target(BaseModel):
    model: str
    node: str | None = None


class Trial(BaseModel):
    index: int
    passed: bool
    output: str | None = None
    failures: list[str] = Field(default_factory=list)
    error: str | None = None
    """Request failed (transport/API error) or a check could not be evaluated."""
    latency_ms: float | None = None
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None
    routed_model: str | None = None
    cache: str | None = None


class CaseResult(BaseModel):
    id: str
    tags: list[str] = Field(default_factory=list)
    n: int
    successes: int
    pass_at_1: float
    pass_hat_k: float
    pass_at_k: float
    latency_p50_ms: float | None = None
    trials: list[Trial]


class EvalSummary(BaseModel):
    cases: int
    trials: int
    successes: int
    errors: int
    pass_at_1: float
    pass_hat_k: float
    pass_at_k: float
    cost_total_usd: float | None = None
    cost_per_success_usd: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cache_hits: int = 0


class EvalReport(BaseModel):
    dataset: str
    target: Target
    k: int
    n: int
    started_at: datetime
    duration_s: float
    summary: EvalSummary
    cases: list[CaseResult]
    warnings: list[str] = Field(default_factory=list)

    def failures(self) -> list[tuple[str, Trial]]:
        return [(c.id, t) for c in self.cases for t in c.trials if not t.passed]


def _merge_options(*opts: CalibanOptions | None, node: str | None) -> CalibanOptions:
    merged: dict[str, Any] = {"cache": "off"}  # avoid cache hits making trials identical
    for o in opts:
        if o is not None:
            merged.update(o.model_dump(exclude_unset=True, exclude_none=True))
    if node is not None:
        merged["node"] = node
    return CalibanOptions(**merged)


def _run_trial(
    client: Caliban,
    dataset: Dataset,
    case: Case,
    index: int,
    target: Target,
    judge: Judge | None,
    pricing: Pricing | None,
) -> Trial:
    d = dataset.defaults
    node = case.node or target.node
    opts = _merge_options(d.caliban, case.caliban, node=node)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=target.model,
            messages=case.build_messages(d.system),
            caliban=opts,
            temperature=d.temperature,
            max_tokens=d.max_tokens,
        )
    except CalibanError as exc:
        return Trial(index=index, passed=False, error=f"{type(exc).__name__}: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    output = resp.text
    failures: list[str] = []
    eval_error: str | None = None
    for check in case.all_checks():
        result = run_check(check, output, case, judge)
        if not result.passed:
            failures.append(result.reason or check.type)
            if result.error and eval_error is None:
                eval_error = result.reason

    meta = resp.caliban
    cost = meta.cost_usd
    if cost is None and pricing is not None:
        cost = pricing.cost(resp.usage)
    return Trial(
        index=index,
        passed=not failures,
        output=output[:MAX_OUTPUT_CHARS],
        failures=failures,
        error=eval_error,
        latency_ms=latency_ms,
        cost_usd=cost,
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
        completion_tokens=resp.usage.completion_tokens if resp.usage else None,
        request_id=meta.request_id,
        routed_model=meta.routed_model,
        cache=meta.cache,
    )


def resolve_target(dataset: Dataset, model: str | None, node: str | None) -> Target:
    node = node or dataset.defaults.node
    model = model or dataset.defaults.model or ("caliban/auto" if node else None)
    if model is None:
        raise CalibanError("no target: pass --model/--node or set defaults.model in the dataset")
    return Target(model=model, node=node)


def run_eval(
    dataset: Dataset,
    client: Caliban,
    *,
    model: str | None = None,
    node: str | None = None,
    k: int = 3,
    trials: int | None = None,
    concurrency: int = 1,
    judge: Judge | None = None,
    pricing: Pricing | None = None,
    on_trial: Callable[[str, Trial], None] | None = None,
) -> EvalReport:
    """Run every case ``trials`` times (default ``k``) and compute metrics.

    ``pass^k`` uses the unbiased estimator ``C(c,k)/C(n,k)``, so ``trials`` may
    exceed ``k`` for a lower-variance estimate.
    """
    n = trials if trials is not None else k
    if k < 1 or n < k:
        raise CalibanError(f"need 1 <= k <= trials (got k={k}, trials={n})")
    if concurrency < 1:
        raise CalibanError("concurrency must be >= 1")
    target = resolve_target(dataset, model, node)
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    jobs = [(case, i) for case in dataset.cases for i in range(n)]
    results: dict[tuple[str, int], Trial] = {}

    def work(job: tuple[Case, int]) -> None:
        case, i = job
        trial = _run_trial(client, dataset, case, i, target, judge, pricing)
        results[(case.id, i)] = trial
        if on_trial is not None:
            on_trial(case.id, trial)

    if concurrency == 1:
        for job in jobs:
            work(job)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(work, jobs))

    case_results: list[CaseResult] = []
    for case in dataset.cases:
        ts = [results[(case.id, i)] for i in range(n)]
        c = sum(t.passed for t in ts)
        lat = [t.latency_ms for t in ts if t.latency_ms is not None]
        case_results.append(
            CaseResult(
                id=case.id,
                tags=case.tags,
                n=n,
                successes=c,
                pass_at_1=c / n,
                pass_hat_k=pass_hat_k(n, c, k),
                pass_at_k=pass_at_k(n, c, k),
                latency_p50_ms=percentile(lat, 50),
                trials=ts,
            )
        )

    all_trials = [t for cr in case_results for t in cr.trials]
    successes = sum(t.passed for t in all_trials)
    latencies = [t.latency_ms for t in all_trials if t.latency_ms is not None]
    costs = [t.cost_usd for t in all_trials if t.cost_usd is not None]
    cost_total = sum(costs) if costs else None
    cache_hits = sum(1 for t in all_trials if t.cache == "hit")

    warnings: list[str] = []
    if cache_hits:
        warnings.append(
            f"{cache_hits} trial(s) were cache hits; repeated trials are not independent."
        )
    if costs and len(costs) < len(all_trials):
        warnings.append(f"cost unknown for {len(all_trials) - len(costs)} trial(s); counted as $0.")
    if not costs:
        warnings.append("no cost data (server sent no x-caliban-cost-usd; pass pricing=...).")

    summary = EvalSummary(
        cases=len(case_results),
        trials=len(all_trials),
        successes=successes,
        errors=sum(1 for t in all_trials if t.error),
        pass_at_1=mean([cr.pass_at_1 for cr in case_results]) or 0.0,
        pass_hat_k=mean([cr.pass_hat_k for cr in case_results]) or 0.0,
        pass_at_k=mean([cr.pass_at_k for cr in case_results]) or 0.0,
        cost_total_usd=cost_total,
        cost_per_success_usd=(
            cost_total / successes if cost_total is not None and successes else None
        ),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        cache_hits=cache_hits,
    )
    return EvalReport(
        dataset=dataset.name,
        target=target,
        k=k,
        n=n,
        started_at=started,
        duration_s=time.perf_counter() - t0,
        summary=summary,
        cases=case_results,
        warnings=warnings,
    )
