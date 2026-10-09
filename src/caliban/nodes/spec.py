"""Pydantic mirror of ``core/schemas/node.schema.json`` (Caliban Node spec v0).

The top level forbids unknown keys (``additionalProperties: false`` in the
schema). Nested objects allow extra keys, as the schema does; :func:`lint`
reports them as warnings so typos are still visible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, Strict
from pydantic import ValidationError as PydanticValidationError

from ..errors import CalibanError

__all__ = [
    "Budgets",
    "DatasourceScopes",
    "Edge",
    "EvalConfig",
    "Exposure",
    "Graph",
    "Guardrails",
    "LongTermMemory",
    "Memory",
    "ModelPolicy",
    "NodeSpec",
    "NodeValidationError",
    "Prompt",
    "ToolRef",
    "Vertex",
    "json_schema",
    "lint",
    "load_node",
    "validate",
]

TrustTier = Literal["t0_sovereign", "t1_attested", "t2_contracted", "t3_public"]
StrictInt = Annotated[int, Strict()]
StrictBool = Annotated[bool, Strict()]
StrictFloat = Annotated[float, Strict()]
Scope = Annotated[str, Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_*]+:(read|write)$")]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Prompt(_Open):
    system: str
    output_schema: dict[str, Any] | None = None


class ModelPolicy(_Open):
    intent_classes: list[str] | None = None
    candidates: list[str] | None = None
    """Model ids or tiers (``tier:small``, ``tier:frontier``)."""
    escalate_on: list[Literal["schema_violation", "low_confidence", "tool_error"]] | None = None
    max_cost_usd: Annotated[StrictFloat, Field(ge=0)] | None = None
    min_trust_tier: TrustTier | None = None


class ToolRef(_Open):
    ref: Annotated[str, Field(pattern=r"^(mcp|node)://")]
    """``mcp://server/tool#sha256:<hash>`` or ``node://name@vN``."""
    effect: Literal["read", "write"]
    requires: list[str] | None = None


class DatasourceScopes(_Open):
    scopes: list[Scope] | None = None


class LongTermMemory(_Open):
    scope: Literal["end_user", "tenant"] | None = None
    ttl_days: StrictInt | None = None


class Memory(_Open):
    short_term: Literal["none", "run"] | None = None
    long_term: LongTermMemory | None = None


class Guardrails(_Open):
    pii: Literal["off", "mask", "reversible"] | None = None
    injection_mode: Literal["none", "plan_then_execute", "camel"] | None = None
    egress: list[str] | None = None


class Budgets(_Open):
    steps: Annotated[StrictInt, Field(ge=1, le=200)]
    depth: Annotated[StrictInt, Field(ge=1, le=10)] = 3
    fanout: Annotated[StrictInt, Field(ge=1, le=64)] = 8
    tokens: Annotated[StrictInt, Field(ge=1)]
    wall_clock_s: Annotated[StrictInt, Field(ge=1)]


class Exposure(_Open):
    http: StrictBool | None = None
    mcp_tool: StrictBool | None = None
    a2a_agent: StrictBool | None = None


VertexType = Literal["llm", "tool", "router", "map", "reduce", "verify", "human", "subnode", "code"]


class Vertex(_Open):
    id: str
    type: VertexType
    config: dict[str, Any] | None = None
    max_iterations: Annotated[StrictInt, Field(ge=1)] | None = None


class Edge(_Open):
    from_: str = Field(alias="from")
    to: str
    when: str | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Graph(_Open):
    vertices: list[Vertex] | None = None
    edges: list[Edge] | None = None


class EvalConfig(_Open):
    suite: str | None = None
    metrics: list[str] | None = None


class NodeSpec(BaseModel):
    """A node: a versioned, declarative AI sub-app."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["agent", "workflow"]
    description: str | None = None
    prompt: Prompt
    model_policy: ModelPolicy
    tools: list[ToolRef] | None = None
    datasources: DatasourceScopes | None = None
    memory: Memory | None = None
    guardrails: Guardrails | None = None
    budgets: Budgets
    exposure: Exposure | None = None
    graph: Graph | None = None
    eval: EvalConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (aliases applied, ``None`` fields dropped)."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


class NodeValidationError(CalibanError):
    """A node spec failed schema or semantic validation."""

    def __init__(self, issues: list[str], source: str | None = None) -> None:
        self.issues = issues
        self.source = source
        where = f" in {source}" if source else ""
        super().__init__(f"Invalid node spec{where}:\n" + "\n".join(f"  - {i}" for i in issues))


def _loc(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "<root>"


def _cyclic_components(vertices: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    """Strongly connected components that contain a cycle (Kosaraju, iterative)."""
    adj: dict[str, list[str]] = {v: [] for v in vertices}
    radj: dict[str, list[str]] = {v: [] for v in vertices}
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)

    order: list[str] = []
    seen: set[str] = set()
    for root in vertices:
        if root in seen:
            continue
        seen.add(root)
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, i = stack.pop()
            if i < len(adj[node]):
                stack.append((node, i + 1))
                nxt = adj[node][i]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, 0))
            else:
                order.append(node)

    comps: list[set[str]] = []
    assigned: set[str] = set()
    for root in reversed(order):
        if root in assigned:
            continue
        comp = {root}
        assigned.add(root)
        todo = [root]
        while todo:
            n = todo.pop()
            for p in radj[n]:
                if p not in assigned:
                    assigned.add(p)
                    comp.add(p)
                    todo.append(p)
        if len(comp) > 1 or any(b == root for a, b in edges if a == root):
            comps.append(comp)
    return comps


def semantic_issues(spec: NodeSpec) -> list[str]:
    """Rules the JSON Schema states in prose but cannot express."""
    issues: list[str] = []
    if spec.graph is not None and spec.kind != "workflow":
        issues.append("graph: only allowed when kind=workflow")
    if spec.kind == "workflow" and (spec.graph is None or not spec.graph.vertices):
        issues.append("graph: kind=workflow requires graph.vertices")
    if spec.graph is not None:
        verts = spec.graph.vertices or []
        ids = [v.id for v in verts]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        for d in dupes:
            issues.append(f"graph.vertices: duplicate id {d!r}")
        known = set(ids)
        edges: list[tuple[str, str]] = []
        for n, e in enumerate(spec.graph.edges or []):
            bad = [x for x in (e.from_, e.to) if x not in known]
            for x in bad:
                issues.append(f"graph.edges[{n}]: unknown vertex {x!r}")
            if not bad:
                edges.append((e.from_, e.to))
        bounded = {v.id for v in verts if v.max_iterations is not None}
        for comp in _cyclic_components(list(dict.fromkeys(ids)), edges):
            if not comp & bounded:
                names = ", ".join(sorted(comp))
                issues.append(f"graph: loop through [{names}] has no vertex with max_iterations")
    return issues


def validate(spec: NodeSpec | Mapping[str, Any], *, source: str | None = None) -> NodeSpec:
    """Validate a spec (schema + semantic rules). Raises :class:`NodeValidationError`."""
    if isinstance(spec, NodeSpec):
        model = spec
    else:
        if not isinstance(spec, Mapping):
            raise NodeValidationError(["<root>: expected a mapping"], source)
        try:
            model = NodeSpec.model_validate(dict(spec))
        except PydanticValidationError as exc:
            issues = [f"{_loc(err['loc'])}: {err['msg']}" for err in exc.errors()]
            raise NodeValidationError(issues, source) from exc
    issues = semantic_issues(model)
    if issues:
        raise NodeValidationError(issues, source)
    return model


def lint(spec: NodeSpec) -> list[str]:
    """Non-fatal warnings: unknown nested keys, unpinned tools, risky combos."""
    warnings: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, BaseModel):
            for key in obj.model_extra or {}:
                warnings.append(f"{path}.{key}: unknown key (allowed by schema, likely a typo)")
            for name in type(obj).model_fields:
                walk(getattr(obj, name), f"{path}.{name}" if path else name)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]")

    walk(spec, "")
    for i, tool in enumerate(spec.tools or []):
        if tool.ref.startswith("mcp://") and "#sha256:" not in tool.ref:
            warnings.append(f"tools[{i}].ref: MCP tool is not pinned (#sha256:<hash>)")
        if tool.ref.startswith("node://") and "@v" not in tool.ref:
            warnings.append(f"tools[{i}].ref: node tool is not pinned (@vN)")
    has_write = any(t.effect == "write" for t in spec.tools or [])
    mode = spec.guardrails.injection_mode if spec.guardrails else None
    if has_write and mode in (None, "none"):
        warnings.append("guardrails.injection_mode: write-capable tools with no injection defence")
    return warnings


def load_node(path: str | Path, *, check: bool = True) -> NodeSpec:
    """Load a node spec from a YAML (or JSON) file and validate it."""
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise NodeValidationError([f"YAML parse error: {exc}"], str(p)) from exc
    if not isinstance(data, Mapping):
        raise NodeValidationError(["<root>: expected a mapping"], str(p))
    if check:
        return validate(data, source=str(p))
    return NodeSpec.model_validate(dict(data))


def json_schema() -> dict[str, Any]:
    """The vendored copy of ``core/schemas/node.schema.json``."""
    text = resources.files("caliban.nodes").joinpath("node.schema.json").read_text("utf-8")
    result: dict[str, Any] = json.loads(text)
    return result
