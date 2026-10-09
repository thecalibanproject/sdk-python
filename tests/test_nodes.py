from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from caliban.nodes import NodeSpec, NodeValidationError, json_schema, lint, load_node, validate

FIXTURES = Path(__file__).parent / "fixtures"
CORE_SCHEMA = Path(__file__).resolve().parents[2] / "core" / "schemas" / "node.schema.json"

MINIMAL: dict[str, Any] = {
    "kind": "agent",
    "prompt": {"system": "x"},
    "model_policy": {},
    "budgets": {"steps": 1, "tokens": 1, "wall_clock_s": 1},
}


def with_(path: str, value: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = copy.deepcopy(base or MINIMAL)
    *parents, leaf = path.split(".")
    cur = spec
    for p in parents:
        cur = cur.setdefault(p, {})
    cur[leaf] = value
    return spec


def issues_of(spec: dict[str, Any]) -> list[str]:
    with pytest.raises(NodeValidationError) as ei:
        validate(spec)
    return ei.value.issues


def test_load_full_fixture_roundtrip() -> None:
    spec = load_node(FIXTURES / "invoice-triage.node.yaml")
    assert spec.kind == "workflow"
    assert spec.graph is not None and spec.graph.edges is not None
    assert spec.graph.edges[0].from_ == "classify"
    d = spec.to_dict()
    assert d["graph"]["edges"][0]["from"] == "classify"
    assert validate(d).to_dict() == d
    assert NodeSpec.model_validate(json.loads(json.dumps(d))) == spec
    assert lint(spec) == []


def test_minimal_spec_gets_budget_defaults() -> None:
    spec = validate(MINIMAL)
    assert (spec.budgets.depth, spec.budgets.fanout) == (3, 8)


@pytest.mark.parametrize("missing", ["kind", "prompt", "model_policy", "budgets"])
def test_required_top_level(missing: str) -> None:
    spec = copy.deepcopy(MINIMAL)
    del spec[missing]
    assert any(i.startswith(missing) for i in issues_of(spec))


@pytest.mark.parametrize("missing", ["steps", "tokens", "wall_clock_s"])
def test_required_budgets(missing: str) -> None:
    spec = copy.deepcopy(MINIMAL)
    del spec["budgets"][missing]
    assert any(f"budgets.{missing}" in i for i in issues_of(spec))


def test_unknown_top_level_key_rejected() -> None:
    issues = issues_of({**MINIMAL, "node": "invoice-triage@v7"})
    assert issues == ["node: Extra inputs are not permitted"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("kind", "crew"),
        ("prompt.system", 42),
        ("budgets.steps", 0),
        ("budgets.steps", 201),
        ("budgets.depth", 11),
        ("budgets.fanout", 65),
        ("budgets.tokens", "200k"),
        ("budgets.tokens", 1.5),
        ("budgets.tokens", True),
        ("budgets.wall_clock_s", 0),
        ("model_policy.max_cost_usd", -0.01),
        ("model_policy.max_cost_usd", "0.05"),
        ("model_policy.escalate_on", ["timeout"]),
        ("model_policy.min_trust_tier", "t9"),
        ("tools", [{"ref": "http://x", "effect": "read"}]),
        ("tools", [{"ref": "mcp://x/y", "effect": "delete"}]),
        ("tools", [{"ref": "mcp://x/y"}]),
        ("datasources.scopes", ["erp.invoices:admin"]),
        ("datasources.scopes", ["ERP.invoices:read"]),
        ("datasources.scopes", ["invoices:read"]),
        ("memory.short_term", "forever"),
        ("guardrails.injection_mode", "pray"),
        ("guardrails.pii", "anonymize"),
        ("exposure.http", "yes"),
    ],
)
def test_schema_violations(path: str, value: Any) -> None:
    assert issues_of(with_(path, value))


def test_valid_scope_patterns() -> None:
    validate(with_("datasources.scopes", ["erp.invoices:read", "docs.*:write", "a_1.b_2:read"]))


def test_graph_only_for_workflow() -> None:
    spec = with_("graph", {"vertices": [{"id": "a", "type": "llm"}]})
    assert "graph: only allowed when kind=workflow" in issues_of(spec)


def test_workflow_needs_vertices() -> None:
    assert "graph: kind=workflow requires graph.vertices" in issues_of(
        {**MINIMAL, "kind": "workflow"}
    )


def _workflow(vertices: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {**MINIMAL, "kind": "workflow", "graph": {"vertices": vertices, "edges": edges}}


def test_graph_semantics() -> None:
    issues = issues_of(
        _workflow(
            [{"id": "a", "type": "llm"}, {"id": "a", "type": "tool"}],
            [{"from": "a", "to": "zzz"}],
        )
    )
    assert "graph.vertices: duplicate id 'a'" in issues
    assert "graph.edges[0]: unknown vertex 'zzz'" in issues


def test_unbounded_loop_rejected() -> None:
    v = [{"id": "a", "type": "llm"}, {"id": "b", "type": "verify"}, {"id": "c", "type": "tool"}]
    e = [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}, {"from": "b", "to": "c"}]
    assert issues_of(_workflow(v, e)) == [
        "graph: loop through [a, b] has no vertex with max_iterations"
    ]
    v[1]["max_iterations"] = 3
    validate(_workflow(v, e))


def test_self_loop_needs_bound() -> None:
    v = [{"id": "a", "type": "llm"}]
    assert issues_of(_workflow(v, [{"from": "a", "to": "a"}]))
    v[0]["max_iterations"] = 2
    validate(_workflow(v, [{"from": "a", "to": "a"}]))


def test_dag_is_fine() -> None:
    v = [{"id": x, "type": "llm"} for x in "abcd"]
    e = [
        {"from": "a", "to": "b"},
        {"from": "a", "to": "c"},
        {"from": "b", "to": "d"},
        {"from": "c", "to": "d"},
    ]
    validate(_workflow(v, e))


def test_lint_warnings() -> None:
    spec = validate(
        {
            **MINIMAL,
            "prompt": {"system": "x", "sytem_typo": 1},
            "tools": [
                {"ref": "mcp://erp/approve", "effect": "write"},
                {"ref": "node://risk", "effect": "read"},
            ],
        }
    )
    warnings = lint(spec)
    assert any("prompt.sytem_typo" in w for w in warnings)
    assert any("tools[0].ref" in w and "sha256" in w for w in warnings)
    assert any("tools[1].ref" in w and "@vN" in w for w in warnings)
    assert any("injection_mode" in w for w in warnings)


def test_load_node_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: agent\nprompt: [\n")
    with pytest.raises(NodeValidationError, match="YAML"):
        load_node(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n")
    with pytest.raises(NodeValidationError, match="mapping"):
        load_node(lst)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("kind: agent\n")
    with pytest.raises(NodeValidationError) as ei:
        load_node(invalid)
    assert ei.value.source == str(invalid)
    assert "invalid.yaml" in str(ei.value)


def test_nodespec_matches_vendored_schema() -> None:
    schema = json_schema()
    assert schema["additionalProperties"] is False
    fields = NodeSpec.model_fields
    assert sorted(fields) == sorted(schema["properties"])
    assert sorted(n for n, f in fields.items() if f.is_required()) == sorted(schema["required"])


@pytest.mark.skipif(not CORE_SCHEMA.exists(), reason="core repo not checked out alongside")
def test_vendored_schema_in_sync_with_core() -> None:
    core = json.loads(CORE_SCHEMA.read_text())
    assert core == json_schema(), "re-copy core/schemas/node.schema.json into src/caliban/nodes/"
