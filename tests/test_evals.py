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
    DatasetError,
    ExactCheck,
    JsonSchemaCheck,
    LLMJudgeCheck,
    Pricing,
    RegexCheck,
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


def test_run_eval_trials_greater_than_k_and_node_target() -> None:
    server = FakeServer(SCRIPT, cost=None)
    ds = load_dataset(GOLDEN)
    report = run_eval(
        ds,
        make_client(server),
        node="geo-node",
        k=2,
        trials=3,
        pricing=Pricing(in_per_mtok=1.0, out_per_mtok=2.0),
    )
    assert report.target.node == "geo-node" and report.target.model == "caliban/auto"
    assert all(b["caliban"]["node"] == "geo-node" for b in server.bodies)
    capital = next(c for c in report.cases if c.id == "capital")
    assert math.isclose(capital.pass_hat_k, 1 / 3)  # C(2,2)/C(3,2)
    # usage = 10 prompt + 5 completion tokens per call -> 20e-6 USD
    assert math.isclose(report.summary.cost_total_usd or 0, 9 * 20e-6)


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
