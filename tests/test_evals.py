from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pytest

from caliban import Caliban, CalibanError
from caliban.evals import (
    Case,
    CheckResult,
    ContainsCheck,
    Dataset,
    DatasetError,
    ExactCheck,
    JsonSchemaCheck,
    LLMJudgeCheck,
    Pricing,
    RegexCheck,
    Target,
    cli,
    extract_json,
    load_dataset,
    pass_at_k,
    pass_hat_k,
    percentile,
    run_check,
    run_eval,
    to_markdown,
    validate_json_schema,
)
from caliban.evals.runner import parse_node_ref, resolve_target

from .conftest import completion_body, error_body, make_client

GOLDEN = Path(__file__).parent / "fixtures" / "golden.yaml"

# ───────────────────────────── metrics math ─────────────────────────────


@pytest.mark.parametrize(
    ("n", "c", "k", "hat", "at"),
    [
        (3, 3, 3, 1.0, 1.0),
        (3, 2, 3, 0.0, 1.0),
        (3, 0, 3, 0.0, 0.0),
        (1, 1, 1, 1.0, 1.0),
        (5, 4, 3, 0.4, 1.0),  # C(4,3)/C(5,3) = 4/10
        (5, 1, 3, 0.0, 0.6),  # 1 - C(4,3)/C(5,3)
        (10, 7, 2, 21 / 45, 1 - 3 / 45),
        (4, 2, 1, 0.5, 0.5),  # k=1 reduces to pass@1
    ],
)
def test_pass_k_known_values(n: int, c: int, k: int, hat: float, at: float) -> None:
    assert math.isclose(pass_hat_k(n, c, k), hat)
    assert math.isclose(pass_at_k(n, c, k), at)


def test_pass_k_matches_brute_force_enumeration() -> None:
    for n in range(1, 7):
        for c in range(n + 1):
            outcomes = [True] * c + [False] * (n - c)
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(outcomes, k))
                all_pass = sum(all(s) for s in subsets) / len(subsets)
                any_pass = sum(any(s) for s in subsets) / len(subsets)
                assert math.isclose(pass_hat_k(n, c, k), all_pass), (n, c, k)
                assert math.isclose(pass_at_k(n, c, k), any_pass), (n, c, k)


def test_pass_hat_k_is_monotone_non_increasing_in_k() -> None:
    vals = [pass_hat_k(8, 6, k) for k in range(1, 9)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == 6 / 8


@pytest.mark.parametrize(("n", "c", "k"), [(3, 1, 4), (0, 0, 1), (3, 4, 1), (3, -1, 1), (3, 1, 0)])
def test_pass_k_rejects_bad_input(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_hat_k(n, c, k)


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isclose(percentile(list(range(1, 101)), 95) or 0, 95.05)
    assert percentile([10, 1, 5], 0) == 1 and percentile([10, 1, 5], 100) == 10
    with pytest.raises(ValueError):
        percentile([1.0], 101)


# ───────────────────────────── checkers ─────────────────────────────

CASE = Case(id="c", input="q", expected="x")


def ok(check: Any, output: str) -> bool:
    return run_check(check, output, CASE).passed


def test_exact() -> None:
    assert ok(ExactCheck(value="Paris"), "  Paris\n")
    assert not ok(ExactCheck(value="Paris"), "paris")
    assert ok(ExactCheck(value="Paris", case_sensitive=False), "PARIS")
    assert not ok(ExactCheck(value="Paris", strip=False), "Paris ")


def test_contains() -> None:
    assert ok(ContainsCheck(value=["a", "b"]), "xaxbx")
    res = run_check(ContainsCheck(value=["a", "z"]), "abc", CASE)
    assert not res.passed and "'z'" in (res.reason or "")
    assert ok(ContainsCheck(value="ABC", case_sensitive=False), "xabcx")


def test_regex() -> None:
    assert ok(RegexCheck(pattern=r"\b42\b"), "It is 42.")
    assert not ok(RegexCheck(pattern=r"\b42\b"), "420")
    assert ok(RegexCheck(pattern="^yes$", flags=["IGNORECASE", "MULTILINE"]), "x\nYES\ny")
    with pytest.raises(ValueError, match="invalid regex"):
        RegexCheck(pattern="(")


def test_json_schema_check() -> None:
    schema = {
        "type": "object",
        "required": ["total", "items"],
        "additionalProperties": False,
        "properties": {
            "total": {"type": "number", "minimum": 0},
            "currency": {"enum": ["USD", "EUR"]},
            "items": {"type": "array", "minItems": 1, "items": {"type": "string", "maxLength": 3}},
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
        },
    }
    chk = JsonSchemaCheck.model_validate({"type": "json_schema", "schema": schema})
    assert ok(chk, '```json\n{"total": 1.5, "items": ["a"], "count": 2.0}\n```')
    assert not ok(chk, "{}")
    assert not ok(chk, '{"total": -1, "items": ["a"]}')
    assert not ok(chk, '{"total": 1, "items": []}')
    assert not ok(chk, '{"total": 1, "items": ["toolong"]}')
    assert not ok(chk, '{"total": 1, "items": ["a"], "extra": 1}')
    assert not ok(chk, '{"total": 1, "items": ["a"], "currency": "GBP"}')
    assert not ok(chk, '{"total": 1, "items": ["a"], "flag": 1}')
    assert not ok(chk, '{"total": 1, "items": ["a"], "count": true}')
    assert not ok(chk, '{"total": "1", "items": ["a"]}')
    res = run_check(chk, "not json", CASE)
    assert not res.passed and "not JSON" in (res.reason or "")


def test_json_schema_combinators() -> None:
    assert validate_json_schema(3, {"anyOf": [{"type": "string"}, {"type": "integer"}]}) == []
    assert validate_json_schema(3.5, {"anyOf": [{"type": "string"}, {"type": "integer"}]})
    assert validate_json_schema(3, {"oneOf": [{"type": "number"}, {"type": "integer"}]})
    assert validate_json_schema("a", {"not": {"type": "string"}})
    assert validate_json_schema([1, 1], {"uniqueItems": True})
    assert validate_json_schema(7, {"multipleOf": 2})
    assert validate_json_schema(0.3, {"multipleOf": 0.1}) == []
    assert validate_json_schema("ab", {"pattern": "^a"}) == []
    assert validate_json_schema(None, {"type": ["string", "null"]}) == []
    assert validate_json_schema(1, {"const": True})


def test_extract_json() -> None:
    assert extract_json(" ```\n[1, 2]\n``` ") == [1, 2]
    assert extract_json('{"a": 1}') == {"a": 1}


def test_llm_judge_stub_and_pluggable_judge() -> None:
    chk = LLMJudgeCheck(rubric="Is it polite?")
    res = run_check(chk, "hi", CASE)
    assert not res.passed and res.error and "no judge" in (res.reason or "")

    def judge(check: LLMJudgeCheck, output: str, case: Case) -> CheckResult:
        return CheckResult("please" in output, "impolite")

    assert run_check(chk, "please", CASE, judge).passed
    assert not run_check(chk, "now", CASE, judge).passed


# ───────────────────────────── dataset ─────────────────────────────


def test_load_dataset_fixture() -> None:
    ds = load_dataset(GOLDEN)
    assert ds.name == "demo-golden"
    assert [c.id for c in ds.cases] == ["capital", "math", "json"]
    assert [type(c) for c in ds.cases[2].all_checks()] == [JsonSchemaCheck, ContainsCheck]
    msgs = ds.cases[0].build_messages(ds.defaults.system)
    assert msgs[0] == {"role": "system", "content": "Answer tersely."}


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("cases: []", "no cases"),
        ("cases: [{id: a, input: q}]", "at least one"),
        ("cases: [{id: a, expected: x}]", "exactly one"),
        ("cases: [{id: a, input: q, messages: [{role: user}], expected: x}]", "exactly one"),
        ("cases: [{id: a, input: q, expected: x}, {id: a, input: r, expected: y}]", "duplicate"),
        ("cases: [{id: a, input: q, check: {type: fuzzy}}]", "fuzzy"),
        ("cases: [{id: a, input: q, expected: x, typo: 1}]", "typo"),
        ("- just a list", "mapping"),
        ("cases: [", "cannot read"),
    ],
)
def test_dataset_errors(tmp_path: Path, body: str, match: str) -> None:
    p = tmp_path / "ds.yaml"
    p.write_text(body)
    with pytest.raises(DatasetError, match=match):
        load_dataset(p)


# ───────────────────────────── runner ─────────────────────────────

SCRIPT: dict[str, list[str]] = {
    "What is the capital of France?": ["Paris", "Paris", "paris"],  # 2/3
    "What is 6 times 7?": ["42", "It's 42.", "42"],  # 3/3
    'Return {"ok": true} as JSON.': ['```json\n{"ok": true}\n```', '{"ok": false}', "nope"],  # 1/3
}


class FakeServer:
    def __init__(self, script: dict[str, list[str]], cost: str | None = "0.001") -> None:
        self.script = script
        self.cost = cost
        self.counts: dict[str, int] = defaultdict(int)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        q = body["messages"][-1]["content"]
        if q not in self.script:
            return httpx.Response(400, json=error_body("unknown prompt", "invalid_request_error"))
        outputs = self.script[q]
        i = self.counts[q]
        self.counts[q] += 1
        headers = {"x-caliban-request-id": f"req-{i}", "x-caliban-cache": "miss"}
        if self.cost is not None:
            headers["x-caliban-cost-usd"] = self.cost
        return httpx.Response(200, json=completion_body(outputs[i % len(outputs)]), headers=headers)


def test_run_eval_metrics() -> None:
    server = FakeServer(SCRIPT)
    report = run_eval(load_dataset(GOLDEN), make_client(server), k=3)

    by_id = {c.id: c for c in report.cases}
    assert (by_id["capital"].successes, by_id["math"].successes, by_id["json"].successes) == (
        2,
        3,
        1,
    )
    assert by_id["capital"].pass_hat_k == 0.0 and by_id["capital"].pass_at_k == 1.0
    assert by_id["math"].pass_hat_k == 1.0

    s = report.summary
    assert math.isclose(s.pass_at_1, (2 / 3 + 1 + 1 / 3) / 3)
    assert math.isclose(s.pass_hat_k, 1 / 3)
    assert math.isclose(s.pass_at_k, 1.0)
    assert (s.cases, s.trials, s.successes, s.errors) == (3, 9, 6, 0)
    assert math.isclose(s.cost_total_usd or 0, 0.009)
    assert math.isclose(s.cost_per_success_usd or 0, 0.0015)
    assert s.latency_p50_ms is not None and s.latency_p95_ms is not None
    assert s.latency_p50_ms <= s.latency_p95_ms
    assert report.warnings == []

    fails = report.failures()
    assert [(cid, t.index) for cid, t in fails] == [("capital", 2), ("json", 1), ("json", 2)]
    assert fails[0][1].failures == ["exact: expected 'Paris'"]
    assert fails[0][1].request_id == "req-2"

    first = server.bodies[0]
    assert first["model"] == "caliban/auto"
    assert first["caliban"] == {"pii": "mask", "cache": "off"}
    assert first["temperature"] == 0
    assert first["messages"][0]["role"] == "system"

    md = to_markdown(report)
    assert "pass^3 | 33.3%" in md and "`capital` trial 2" in md
    json.loads(report.model_dump_json())


def test_run_eval_trials_greater_than_k_with_pricing_fallback() -> None:
    server = FakeServer(SCRIPT, cost=None)
    report = run_eval(
        load_dataset(GOLDEN),
        make_client(server),
        k=2,
        trials=3,
        pricing=Pricing(in_per_mtok=1.0, out_per_mtok=2.0),
    )
    capital = next(c for c in report.cases if c.id == "capital")
    assert math.isclose(capital.pass_hat_k, 1 / 3)  # C(2,2)/C(3,2)
    # usage = 10 prompt + 5 completion tokens per call -> 20e-6 USD
    assert math.isclose(report.summary.cost_total_usd or 0, 9 * 20e-6)


# ───────────────────────────── node targets ─────────────────────────────


def run_body(
    status: str = "succeeded",
    output: Any = "ok",
    *,
    run_id: str = "run_1",
    usd: float = 0.002,
    exec_ms: int = 120,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": run_id,
        "object": "node.run",
        "node": "geo-node",
        "version": 2,
        "hash": "sha256:abc",
        "status": status,
        "output": output,
        "steps": [],
        "budget": {"steps_used": 1, "usd": usd, "wall_clock_ms_used": exec_ms},
        "cost_usd": usd,
        "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37},
        "created_at": "2026-10-11T00:00:00Z",
    }
    body.update(extra)
    return body


class NodeServer:
    """Answers node runs from ``SCRIPT`` (keyed by the run input); refuses chat completions."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = script
        self.counts: dict[str, int] = defaultdict(int)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.path == "/v1/nodes/geo-node/runs", request.url.path
        body = json.loads(request.content)
        q = body["input"] if "input" in body else body["chat"]["messages"][-1]["content"]
        outputs = self.script[q]
        i = self.counts[q]
        self.counts[q] += 1
        run = run_body(output=outputs[i % len(outputs)], run_id=f"run_{len(self.requests)}")
        return httpx.Response(200, json=run, headers={"x-caliban-request-id": "req-n"})


def test_node_target_runs_the_node_not_the_model() -> None:
    """Regression: node evals used to send a chat completion with caliban.node, which core
    parses and ignores, so they measured the plain model."""
    server = NodeServer(SCRIPT)
    report = run_eval(load_dataset(GOLDEN), make_client(server), node="geo-node@v2", k=3)

    assert len(server.requests) == 9
    assert all(r.url.path == "/v1/nodes/geo-node/runs" for r in server.requests)
    first = json.loads(server.requests[0].content)
    assert first == {"input": "What is the capital of France?", "version": 2}
    assert "model" not in first and "caliban" not in first
    msgs = json.loads(server.requests[-1].content)["chat"]["messages"]
    assert msgs[0] == {"role": "system", "content": "Answer tersely."}
    keys = {r.headers["Idempotency-Key"] for r in server.requests}
    assert len(keys) == 9 and all(k.startswith("caliban-eval:") for k in keys)

    assert report.target.model_dump() == {"model": None, "node": "geo-node", "version": 2}
    s = report.summary
    assert (s.trials, s.successes) == (9, 6)
    assert math.isclose(s.cost_total_usd or 0, 9 * 0.002)
    assert math.isclose(s.cost_per_success_usd or 0, 9 * 0.002 / 6)
    assert s.latency_p50_ms == 120
    t = report.cases[0].trials[0]
    assert (t.run_id, t.run_status, t.routed_model) == ("run_1", "succeeded", "node/geo-node@v2")
    assert (t.prompt_tokens, t.completion_tokens, t.cache) == (30, 7, None)
    assert any("model targets only" in w for w in report.warnings)
    assert "node `geo-node` v2" in to_markdown(report)


def test_node_target_from_dataset_defaults_and_model_prefix(tmp_path: Path) -> None:
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: n\ndefaults: {model: caliban/auto, node: geo-node}\n"
        "cases: [{id: s, input: {city: Paris}, expected: '{\"ok\":true}'}]\n"
    )
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/nodes/geo-node/runs"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=run_body(output={"ok": True}))

    ds = load_dataset(p)
    report = run_eval(ds, make_client(handler), k=1)  # the default node wins over the model
    assert bodies == [{"input": {"city": "Paris"}}]  # structured input, sent as is
    assert report.summary.successes == 1  # non-string output is checked as compact JSON
    assert report.target.node == "geo-node" and report.target.version is None

    run_eval(ds, make_client(handler), model="node/geo-node@v4", k=1)
    assert bodies[-1] == {"input": {"city": "Paris"}, "version": 4}


def test_node_trial_outcomes_failed_runs_cost_counts() -> None:
    ds = Dataset.model_validate(
        {
            "name": "o",
            "cases": [
                {"id": "fail", "input": "f", "expected": "x"},
                {"id": "budget", "input": "b", "expected": "x"},
                {"id": "slow", "input": "s", "expected": "done"},
            ],
        }
    )
    polls: dict[str, int] = defaultdict(int)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/nodes/geo-node/runs":
            q = json.loads(request.content)["input"]
            if q == "f":
                return httpx.Response(200, json=run_body("failed", None, error="tool down"))
            if q == "b":
                body = run_body("budget_exhausted", "x", stop_reason="budget.usd", usd=0.01)
                return httpx.Response(200, json=body)
            return httpx.Response(202, json=run_body("running", None, run_id="run_s"))
        assert path == "/v1/runs/run_s" and request.method == "GET"
        polls[path] += 1
        status = "running" if polls[path] < 3 else "succeeded"
        return httpx.Response(200, json=run_body(status, "done", run_id="run_s"))

    report = run_eval(ds, make_client(handler), node="geo-node", k=1)
    fail, budget, slow = (c.trials[0] for c in report.cases)
    assert not fail.passed and fail.failures == ["run failed: tool down"] and fail.error is None
    assert not budget.passed and budget.failures == ["run budget_exhausted: budget.usd"]
    assert budget.output == "x"  # partial output kept, checks not run
    assert slow.passed and polls["/v1/runs/run_s"] == 3
    s = report.summary
    # Failures are paid for too: cost per success divides every trial's cost.
    assert math.isclose(s.cost_total_usd or 0, 0.002 + 0.01 + 0.002)
    assert math.isclose(s.cost_per_success_usd or 0, 0.014)


def test_node_trial_human_steps_answers_denials_and_cancels() -> None:
    ds = Dataset.model_validate(
        {
            "name": "h",
            "cases": [
                {
                    "id": "answered",
                    "input": "a",
                    "answers": {"clarify": "two days"},
                    "expected": "triaged",
                },
                {"id": "approval", "input": "w", "expected": "not sent"},
                {"id": "unanswered", "input": "q", "expected": "x"},
            ],
        }
    )
    seen: list[tuple[str, str, Any]] = []
    waiting = {"a": "clarify#0", "w": "send#0@approve", "q": "ask#0"}
    final = {"a": "triaged", "w": "not sent"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/v1/nodes/geo-node/runs":
            q = body["input"]
            return httpx.Response(
                202,
                json=run_body(
                    "input_required",
                    None,
                    run_id=f"run_{q}",
                    awaiting={"step": waiting[q], "question": "?"},
                ),
            )
        run_id = path.split("/")[3]
        q = run_id.removeprefix("run_")
        if path.endswith("/input"):
            return httpx.Response(202, json=run_body("succeeded", final[q], run_id=run_id))
        assert path.endswith("/cancel")
        return httpx.Response(200, json=run_body("cancelled", None, run_id=run_id))

    report = run_eval(ds, make_client(handler), node="geo-node", k=1)
    answered, approval, unanswered = (c.trials[0] for c in report.cases)
    assert answered.passed and approval.passed
    assert ("POST", "/v1/runs/run_a/input", {"answer": "two days", "step": "clarify#0"}) in seen
    assert (
        "POST",
        "/v1/runs/run_w/input",
        {"answer": {"approve": False}, "step": "send#0@approve"},
    ) in seen
    assert ("POST", "/v1/runs/run_q/cancel", None) in seen
    assert not unanswered.passed and unanswered.run_status == "cancelled"
    assert "no answer" in unanswered.failures[0]


def one_case(name: str) -> Dataset:
    return Dataset.model_validate(
        {"name": name, "cases": [{"id": "c", "input": "x", "expected": "y"}]}
    )


def test_node_trial_timeout_cancels_and_errors() -> None:
    cancelled: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/cancel"):
            cancelled.append(request.url.path)
            return httpx.Response(202, json=run_body("running", None, cancel_requested=True))
        return httpx.Response(202, json=run_body("running", None))

    ds = one_case("t")
    report = run_eval(ds, make_client(handler), node="geo-node", k=1, node_run_timeout_s=0)
    t = report.cases[0].trials[0]
    assert cancelled == ["/v1/runs/run_1/cancel"]
    assert not t.passed and t.error and "did not end within 0s" in t.error
    assert report.summary.errors == 1


def test_node_trial_request_errors_are_trial_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=error_body("input does not match", "invalid_request_error"))

    ds = one_case("e")
    report = run_eval(ds, make_client(handler), node="geo-node", k=1)
    assert (report.cases[0].trials[0].error or "").startswith("BadRequestError")


def test_resolve_target_rules() -> None:
    ds = load_dataset(GOLDEN)
    assert resolve_target(ds, None, None).model == "caliban/auto"
    assert resolve_target(ds, None, "node/geo@v3") == Target(node="geo", version=3)
    with pytest.raises(CalibanError, match="not both"):
        resolve_target(ds, "m", "geo")
    for bad in ("Geo", "geo@3", "geo@v0", "a/b"):
        with pytest.raises(CalibanError, match="invalid node"):
            parse_node_ref(bad)


def test_case_answer_lookup_follows_core() -> None:
    c = Case(
        id="a",
        input="x",
        expected="y",
        answers={"clarify": 1, "write@approve": 2, "plan#1": 3, "deep": 4},
    )
    assert c.answer_for("clarify#0") == 1
    assert c.answer_for("write#0@approve") == 2
    assert c.answer_for("plan#1") == 3  # an exact step id first
    assert c.answer_for("parent#0>deep#2/1") == 4
    assert c.answer_for("write#0") is None and c.answer_for("other#0") is None


def test_model_target_sends_structured_input_as_compact_json() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion_body("y"))

    ds = Dataset.model_validate(
        {"name": "m", "cases": [{"id": "c", "input": {"a": "é", "b": [1]}, "expected": "y"}]}
    )
    run_eval(ds, make_client(handler), model="m", k=1)
    assert bodies[0]["messages"] == [{"role": "user", "content": '{"a":"é","b":[1]}'}]


def test_run_eval_counts_api_errors_and_warns_on_cache(tmp_path: Path) -> None:
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: e\ndefaults: {model: m, caliban: {cache: exact}}\n"
        "cases: [{id: unknown, input: '???', expected: x}, {id: ok, input: hi, expected: yo}]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["caliban"]["cache"] == "exact"  # dataset overrides the eval default
        if body["messages"][-1]["content"] == "???":
            return httpx.Response(400, json=error_body("bad", "invalid_request_error"))
        return httpx.Response(200, json=completion_body("yo"), headers={"x-caliban-cache": "hit"})

    report = run_eval(load_dataset(p), make_client(handler), k=2, concurrency=4)
    s = report.summary
    assert s.errors == 2 and s.successes == 2 and s.cache_hits == 2
    assert s.cost_per_success_usd is None
    assert any("cache hits" in w for w in report.warnings)
    err = report.cases[0].trials[0].error or ""
    assert err.startswith("BadRequestError")


def test_run_eval_argument_validation() -> None:
    ds = load_dataset(GOLDEN)
    client = make_client(FakeServer(SCRIPT))
    with pytest.raises(CalibanError, match="k <= trials"):
        run_eval(ds, client, k=3, trials=2)
    ds.defaults.model = None
    with pytest.raises(CalibanError, match="no target"):
        run_eval(ds, client, k=1)


# ───────────────────────────── CLI ─────────────────────────────


def test_cli_validate(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(GOLDEN)]) == 0
    assert "3 cases" in capsys.readouterr().out


def test_cli_run_writes_reports_and_fail_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeServer(SCRIPT)
    built: list[Any] = []

    def fake_make_client(args: Any) -> Caliban:
        built.append(args)
        return make_client(server)

    monkeypatch.setattr(cli, "_make_client", fake_make_client)
    argv = ["run", str(GOLDEN), "--model", "small", "--k", "3", "--out-dir", str(tmp_path), "-q"]
    argv += ["--base-url", "http://gw/v1"]
    assert cli.main([*argv, "--fail-under", "0.3"]) == 0
    assert cli.main([*argv, "--fail-under", "0.5"]) == 1
    assert built[0].base_url == "http://gw/v1"

    data = json.loads((tmp_path / "demo-golden.eval.json").read_text())
    assert data["target"]["model"] == "small"
    assert math.isclose(data["summary"]["pass_hat_k"], 1 / 3)
    md = (tmp_path / "demo-golden.eval.md").read_text()
    assert md.startswith("# Eval report: demo-golden")
    assert "pass^3=0.333" in capsys.readouterr().out


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(tmp_path / "missing.yaml")]) == 2
    assert "error" in capsys.readouterr().err
    assert cli.main(["run", str(GOLDEN), "--price-in", "1"]) == 2


def test_cli_all_errors_exit_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(cli, "_make_client", lambda args: make_client(refuse, max_retries=0))
    assert cli.main(["run", str(GOLDEN), "--k", "1", "--out-dir", str(tmp_path), "-q"]) == 2
    md = (tmp_path / "demo-golden.eval.md").read_text()
    assert "| Errors | 3 |" in md
    assert md.count("| pass@1 | 0.0% |") == 1


def test_cli_node_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = NodeServer(SCRIPT)
    monkeypatch.setattr(cli, "_make_client", lambda args: make_client(server))
    argv = ["run", str(GOLDEN), "--node", "geo-node@v2", "--k", "1", "--out-dir", str(tmp_path)]
    assert cli.main([*argv, "-q"]) == 0
    assert server.requests and all(r.url.path.endswith("/runs") for r in server.requests)
    data = json.loads((tmp_path / "demo-golden.eval.json").read_text())
    assert data["target"] == {"model": None, "node": "geo-node", "version": 2}
    with pytest.raises(SystemExit):
        cli.main([*argv, "--model", "m"])  # --model and --node are exclusive
    assert "not allowed with" in capsys.readouterr().err


def test_node_trial_untrusted_marks_and_input_labels(tmp_path: Path) -> None:
    p = tmp_path / "ds.yaml"
    p.write_text(
        "name: u\ndefaults: {node: geo-node, untrusted: true}\n"
        "cases:\n"
        "  - {id: all, input: {note: x}, expected: ok}\n"
        "  - {id: field, input: {note: x}, untrusted: ['/note'], expected: ok}\n"
        "  - {id: none, input: {note: x}, untrusted: false, expected: ok}\n"
    )
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        labels = [] if body["untrusted"] is False else ["caller"]
        return httpx.Response(200, json=run_body(input_labels=labels))

    report = run_eval(load_dataset(p), make_client(handler), k=1)
    assert [b["untrusted"] for b in bodies] == [True, ["/note"], False]
    assert [c.trials[0].input_labels for c in report.cases] == [["caller"], ["caller"], []]

    model_ds = load_dataset(p)
    model_ds.defaults.node = None
    model_ds.defaults.model = "m"
    ok = completion_body("ok")
    chat = run_eval(model_ds, make_client(lambda r: httpx.Response(200, json=ok)), k=1)
    assert any("node targets only" in w for w in chat.warnings)
