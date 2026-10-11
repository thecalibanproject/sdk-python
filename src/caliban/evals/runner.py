"""Run a golden-set dataset k times per case against a model or node.

A model target sends each trial as a chat completion. A node target starts each trial as a
node run (``POST /v1/nodes/{name}/runs``), the same kind of run core's server-side evals
create: the case input is the run input, a run that does not succeed fails the trial (its
cost still counts), cost is what the run spent on model calls as metered, and latency is the
run's execution time.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .. import _base
from .._base import parse_response
from .._client import Caliban
from ..errors import CalibanError
from ..types import CalibanOptions, CompletionUsage
from .checkers import Judge, run_check
from .dataset import Case, Dataset, compact_json
from .metrics import mean, pass_at_k, pass_hat_k, percentile

__all__ = [
    "CaseResult",
    "EvalReport",
    "EvalSummary",
    "Pricing",
    "Target",
    "Trial",
    "parse_node_ref",
    "run_eval",
]

MAX_OUTPUT_CHARS = 4000
DEFAULT_NODE_RUN_TIMEOUT_S = 600.0
"""How long a node trial may take (from its start, human steps included) before it is
cancelled and counted as an error."""
NODE_POLL_S = 1.0
TERMINAL_RUN_STATUSES = frozenset({"succeeded", "failed", "budget_exhausted", "cancelled"})
_NODE_REF = re.compile(r"^(?:node/)?([a-z0-9][a-z0-9_-]{0,63})(?:@v([1-9][0-9]{0,8}))?$")


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
    """What the eval ran against: a ``model``, or a ``node`` (its live version unless
    ``version`` pins a published one). Exactly one of ``model`` and ``node`` is set."""

    model: str | None = None
    node: str | None = None
    version: int | None = None


def parse_node_ref(ref: str) -> tuple[str, int | None]:
    """``"triage"``, ``"triage@v3"`` or ``"node/triage@v3"`` to ``(name, version)``."""
    m = _NODE_REF.match(ref.strip())
    if m is None:
        raise CalibanError(f"invalid node {ref!r}: expected a node name, optionally @v<N>")
    return m.group(1), int(m.group(2)) if m.group(2) else None


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
    """Model targets: the model that answered. Node targets: ``node/<name>@v<N>``."""
    cache: str | None = None
    run_id: str | None = None
    """Node targets: the run of this trial."""
    run_status: str | None = None
    """Node targets: how the run ended (``succeeded``, ``failed``, ``budget_exhausted``,
    ``cancelled``)."""
    input_labels: list[str] | None = None
    """Node targets: taint labels the run input carried from the start (``caller`` when it
    held untrusted caller data)."""


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


def _merge_options(*opts: CalibanOptions | None) -> CalibanOptions:
    merged: dict[str, Any] = {"cache": "off"}  # avoid cache hits making trials identical
    for o in opts:
        if o is not None:
            merged.update(o.model_dump(exclude_unset=True, exclude_none=True))
    return CalibanOptions(**merged)


def _score(case: Case, output: str, judge: Judge | None) -> tuple[list[str], str | None]:
    failures: list[str] = []
    eval_error: str | None = None
    for check in case.all_checks():
        result = run_check(check, output, case, judge)
        if not result.passed:
            failures.append(result.reason or check.type)
            if result.error and eval_error is None:
                eval_error = result.reason
    return failures, eval_error


def _run_trial(
    client: Caliban,
    dataset: Dataset,
    case: Case,
    index: int,
    target: Target,
    judge: Judge | None,
    pricing: Pricing | None,
    cfg: _NodeTrialConfig,
) -> Trial:
    if case.node is not None or target.node is not None:
        name, version = (
            parse_node_ref(case.node) if case.node else (target.node or "", target.version)
        )
        return _run_node_trial(client, dataset, case, index, name, version, judge, pricing, cfg)
    assert target.model is not None
    d = dataset.defaults
    opts = _merge_options(d.caliban, case.caliban)
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
    failures, eval_error = _score(case, output, judge)

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


# ───────────────────────────── node trials ─────────────────────────────


class _RunAwaiting(BaseModel):
    model_config = ConfigDict(extra="allow")
    step: str | None = None
    question: str | None = None


class _NodeRun(BaseModel):
    """The parts of ``components.schemas.NodeRun`` a trial needs."""

    model_config = ConfigDict(extra="allow")
    id: str
    node: str | None = None
    version: int | None = None
    status: str
    output: Any = None
    error: str | None = None
    stop_reason: str | None = None
    awaiting: _RunAwaiting | None = None
    budget: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = None
    usage: CompletionUsage | None = None
    input_labels: list[str] | None = None


_RUN = TypeAdapter(_NodeRun)


@dataclass(frozen=True)
class _NodeTrialConfig:
    eval_id: str
    timeout_s: float


class _Unanswered(Exception):
    def __init__(self, run: _NodeRun, step: str) -> None:
        self.run = run
        self.step = step


class _RunTimeout(Exception):
    def __init__(self, run: _NodeRun) -> None:
        self.run = run


def _idempotency_key(cfg: _NodeTrialConfig, case: Case, index: int) -> str:
    # Visible ASCII, at most 255 characters, whatever the case id.
    digest = hashlib.sha256(case.id.encode("utf-8")).hexdigest()[:16]
    return f"caliban-eval:{cfg.eval_id}:{digest}:{index}"


def _run_path(run: _NodeRun, suffix: str = "") -> str:
    return f"runs/{quote(run.id, safe='')}{suffix}"


def _follow(client: Caliban, run: _NodeRun, case: Case, deadline: float) -> _NodeRun:
    """Wait until the run ends, answering its human steps from ``case.answers``."""
    http = client._http
    while run.status not in TERMINAL_RUN_STATUSES:
        if time.monotonic() >= deadline:
            raise _RunTimeout(run)
        if run.status == "input_required":
            step = (run.awaiting.step if run.awaiting else None) or ""
            answer = case.answer_for(step)
            if answer is None and step.endswith("@approve"):
                answer = {"approve": False}  # a tainted write nobody approved is denied
            if answer is None:
                raise _Unanswered(run, step)
            body = {"answer": answer, "step": step}
            run = parse_response(http.request("POST", _run_path(run, "/input"), json=body), _RUN)
            continue
        _base._sleep(NODE_POLL_S)
        run = parse_response(http.request("GET", _run_path(run)), _RUN)
    return run


def _cancel(client: Caliban, run: _NodeRun) -> _NodeRun:
    try:
        resp = client._http.request("POST", _run_path(run, "/cancel"))
        return parse_response(resp, _RUN)
    except CalibanError:
        return run


def _run_node_trial(
    client: Caliban,
    dataset: Dataset,
    case: Case,
    index: int,
    name: str,
    version: int | None,
    judge: Judge | None,
    pricing: Pricing | None,
    cfg: _NodeTrialConfig,
) -> Trial:
    body: dict[str, Any]
    if case.input is not None:
        body = {"input": case.input}
    else:
        body = {"chat": {"messages": case.build_messages(dataset.defaults.system)}}
    if version is not None:
        body["version"] = version
    untrusted = case.untrusted if case.untrusted is not None else dataset.defaults.untrusted
    if untrusted is not None:
        body["untrusted"] = untrusted
    # The run API deduplicates on the key, so a retried request never starts a second run.
    headers = {"Idempotency-Key": _idempotency_key(cfg, case, index)}
    t0 = time.perf_counter()
    failures: list[str] = []
    error: str | None = None
    try:
        resp = client._http.request(
            "POST", f"nodes/{quote(name, safe='')}/runs", json=body, headers=headers
        )
        request_id = resp.headers.get("x-caliban-request-id")
        run = parse_response(resp, _RUN)
        try:
            run = _follow(client, run, case, time.monotonic() + cfg.timeout_s)
        except _Unanswered as exc:
            run = _cancel(client, exc.run)
            failures.append(
                f"the run asked for input at step {exc.step!r} and the case has no answer"
            )
        except _RunTimeout as exc:
            run = _cancel(client, exc.run)
            error = f"the run did not end within {cfg.timeout_s:g}s; cancelled"
    except CalibanError as exc:
        return Trial(index=index, passed=False, error=f"{type(exc).__name__}: {exc}")
    wall_ms = (time.perf_counter() - t0) * 1000.0

    output = run.output if isinstance(run.output, str) else None
    if run.output is not None and output is None:
        output = compact_json(run.output)
    if run.status == "succeeded" and not failures and error is None:
        failures, error = _score(case, output or "", judge)
    elif not failures and error is None:
        why = f"run {run.status}"
        detail = run.error or run.stop_reason
        failures.append(f"{why}: {detail}" if detail else why)

    cost = run.cost_usd
    if cost is None and isinstance(run.budget.get("usd"), int | float):
        cost = float(run.budget["usd"])
    if cost is None and pricing is not None:
        cost = pricing.cost(run.usage)
    exec_ms = run.budget.get("wall_clock_ms_used")
    return Trial(
        index=index,
        passed=not failures and error is None,
        output=output[:MAX_OUTPUT_CHARS] if output is not None else None,
        failures=failures,
        error=error,
        latency_ms=float(exec_ms) if isinstance(exec_ms, int | float) else wall_ms,
        cost_usd=cost,
        prompt_tokens=run.usage.prompt_tokens if run.usage else None,
        completion_tokens=run.usage.completion_tokens if run.usage else None,
        request_id=request_id,
        routed_model=f"node/{run.node or name}@v{run.version}" if run.version else None,
        run_id=run.id,
        run_status=run.status,
        input_labels=run.input_labels,
    )


def resolve_target(dataset: Dataset, model: str | None, node: str | None) -> Target:
    """Explicit ``model``/``node`` first, then the dataset defaults (a default node wins
    over a default model)."""
    if model and node:
        raise CalibanError("pass either a model or a node, not both")
    d = dataset.defaults
    if node or (not model and d.node):
        name, version = parse_node_ref(node or d.node or "")
        return Target(node=name, version=version)
    model = model or d.model
    if model is None:
        raise CalibanError("no target: pass --model/--node or set defaults.model in the dataset")
    if model.startswith("node/"):
        name, version = parse_node_ref(model)
        return Target(node=name, version=version)
    return Target(model=model)


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
    node_run_timeout_s: float = DEFAULT_NODE_RUN_TIMEOUT_S,
) -> EvalReport:
    """Run every case ``trials`` times (default ``k``) and compute metrics.

    ``pass^k`` uses the unbiased estimator ``C(c,k)/C(n,k)``, so ``trials`` may
    exceed ``k`` for a lower-variance estimate.

    ``node`` (``name`` or ``name@vN``; also ``model="node/<name>"``) runs each trial as a
    node run through ``POST /v1/nodes/{name}/runs``. ``node_run_timeout_s`` bounds one node
    trial; a run still going then is cancelled and the trial counted as an error.
    """
    n = trials if trials is not None else k
    if k < 1 or n < k:
        raise CalibanError(f"need 1 <= k <= trials (got k={k}, trials={n})")
    if concurrency < 1:
        raise CalibanError("concurrency must be >= 1")
    target = resolve_target(dataset, model, node)
    node_cfg = _NodeTrialConfig(eval_id=uuid.uuid4().hex, timeout_s=node_run_timeout_s)
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    jobs = [(case, i) for case in dataset.cases for i in range(n)]
    results: dict[tuple[str, int], Trial] = {}

    def work(job: tuple[Case, int]) -> None:
        case, i = job
        trial = _run_trial(client, dataset, case, i, target, judge, pricing, node_cfg)
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
        warnings.append("no cost data (the server sent no cost; pass pricing=...).")
    d = dataset.defaults
    runs_nodes = target.node is not None or any(c.node for c in dataset.cases)
    if not runs_nodes and (
        d.untrusted is not None or any(c.untrusted is not None for c in dataset.cases)
    ):
        warnings.append("untrusted applies to node targets only; a model target ignores it.")
    if runs_nodes and (
        d.temperature is not None
        or d.max_tokens is not None
        or d.caliban is not None
        or any(c.caliban is not None for c in dataset.cases)
    ):
        warnings.append(
            "temperature, max_tokens and caliban options apply to model targets only; "
            "a node run uses its own spec."
        )

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
