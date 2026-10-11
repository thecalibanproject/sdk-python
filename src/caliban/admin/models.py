"""Control-plane models, hand-written to match ``core/api/openapi.yaml`` (v0.1.0).

Keep in sync with the contract. Response models allow extra fields so that a
newer control plane does not break an older SDK; request models forbid them so
typos are caught client-side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import CalibanError
from ..evals.dataset import Case
from ..types import CacheTier, ModelCapabilities, ModelKind, RouteStage, TrustTier

__all__ = [
    "ApiKeyCreate",
    "ApiKeyCreated",
    "ApiKeyInfo",
    "CacheTier",
    "Datasource",
    "DatasourceCreate",
    "DatasourceKind",
    "DatasourceStatus",
    "Eval",
    "EvalList",
    "EvalPolicy",
    "EvalPolicyInfo",
    "EvalStart",
    "EvalStatus",
    "EvalSuite",
    "EvalTrial",
    "EvalTrialStatus",
    "GateCheck",
    "GateProblem",
    "GoldenBar",
    "GoldenCase",
    "GoldenSet",
    "GoldenSetPut",
    "Health",
    "InjectionBar",
    "IntrospectJob",
    "Model",
    "ModelCapabilities",
    "ModelCreate",
    "ModelDiscovery",
    "ModelKind",
    "Node",
    "NodeCreate",
    "Ontology",
    "OntologyElement",
    "OntologyElementKind",
    "OntologyElementStatus",
    "OntologyProvenance",
    "OntologyReview",
    "PiiMode",
    "PiiSurrogateScope",
    "ProviderHealth",
    "ProviderKey",
    "ProviderKeyCreate",
    "ProviderKind",
    "RouteStage",
    "SemanticCacheSetting",
    "SharedProvider",
    "SharedProviderCreate",
    "Tenant",
    "TenantCreate",
    "TenantStatus",
    "TenantUpdate",
    "TrustTier",
    "UsageEvent",
    "UsageReport",
    "UsageTotals",
]

PiiMode = Literal["off", "mask", "reversible"]
PiiSurrogateScope = Literal["tenant", "session"]
"""``tenant`` (default): one surrogate per value within the tenant, so pseudonymised requests
can hit the exact cache but become linkable. ``session``: fresh surrogates per request."""
SemanticCacheSetting = Literal["off", "on"]
TenantStatus = Literal["active", "deleted"]
ProviderKind = Literal[
    "openai", "anthropic", "openai_compatible", "azure_openai", "bedrock", "vertex"
]
DatasourceKind = Literal[
    "mongodb",
    "postgres",
    "mysql",
    "sqlserver",
    "snowflake",
    "bigquery",
    "clickhouse",
    "elasticsearch",
    "rest_openapi",
    "s3_parquet",
    "mcp",
]
DatasourceStatus = Literal["pending", "connected", "error", "introspecting"]
OntologyElementKind = Literal[
    "entity",
    "attribute",
    "relation",
    "metric",
    "dimension",
    "glossary_term",
    "policy",
    "verified_query",
]
OntologyElementStatus = Literal["proposed", "approved", "rejected", "stale", "deprecated"]
OntologyProvenance = Literal["introspect", "profile", "query_log", "llm", "human"]
EvalSuite = Literal["golden", "injection"]
EvalStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
EvalTrialStatus = Literal["running", "passed", "failed", "error"]


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class Health(_Response):
    status: Literal["ok", "degraded"]
    version: str
    mode: Literal["router", "control-plane", "standalone"]
    config_version: str | None = None


class Tenant(_Response):
    id: str
    name: str
    region: str | None = None
    pii_default: PiiMode | None = None
    pii_surrogate_scope: PiiSurrogateScope | None = None
    """Surrogate consistency for reversible PII. ``None`` from older servers."""
    semantic_cache: SemanticCacheSetting | None = None
    """Whether the tenant may use the semantic cache. ``None`` from older servers."""
    auto_cache_hit_fraction: float | None = None
    """Share of the flat ``caliban/auto`` price billed for this tenant's cache hits, 0..1.
    ``None``: the deployment's ``[routing] auto_cache_hit_fraction`` (0.2 unless configured)
    applies, or an older server."""
    created_at: datetime
    status: TenantStatus | None = None
    """``"deleted"`` only appears with ``include_deleted=True``. ``None`` from older servers."""
    deleted_at: datetime | None = None


class TenantCreate(_Request):
    name: str
    region: str | None = None
    pii_default: PiiMode | None = None
    pii_surrogate_scope: PiiSurrogateScope | None = None
    semantic_cache: SemanticCacheSetting | None = None
    auto_cache_hit_fraction: float | None = Field(default=None, ge=0, le=1)


class TenantUpdate(_Request):
    """Body of ``PATCH /api/v1/tenants/{tenantId}``. Unset fields keep their value.

    ``auto_cache_hit_fraction`` is sent only when set; clearing it (an explicit ``null``) goes
    through ``tenants.update(..., auto_cache_hit_fraction="default")``."""

    pii_default: PiiMode | None = None
    pii_surrogate_scope: PiiSurrogateScope | None = None
    semantic_cache: SemanticCacheSetting | None = None
    auto_cache_hit_fraction: float | None = Field(default=None, ge=0, le=1)


class ApiKeyInfo(_Response):
    id: str
    name: str
    prefix: str
    created_at: datetime
    revoked_at: datetime | None = None
    """Set once the key is revoked (listed only with ``include_revoked=True``)."""


class ApiKeyCreated(ApiKeyInfo):
    key: str = Field(repr=False)
    """Plaintext key. Returned exactly once; store it now."""


class ApiKeyCreate(_Request):
    name: str | None = None


class ProviderKey(_Response):
    id: str
    tenant_id: str
    kind: ProviderKind
    label: str
    base_url: str | None = None
    trust_tier: TrustTier
    last4: str | None = None
    created_at: datetime


class ProviderKeyCreate(_Request):
    kind: ProviderKind
    label: str
    trust_tier: TrustTier
    provider_id: str | None = None
    """Provider id referenced by the model catalogue; defaults to the slugified label."""
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)


class Model(_Response):
    """A model in the catalogue (``components.schemas.Model``)."""

    id: str
    provider_id: str
    provider_kind: ProviderKind | None = None
    upstream_model: str
    kind: ModelKind
    family: str | None = None
    capabilities: ModelCapabilities
    trust_tier: TrustTier
    licence: str | None = None
    context_window: int | None = None
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None


class ModelCreate(_Request):
    """Register a model (``components.schemas.ModelCreate``); also the shape of discovery
    suggestions."""

    id: str
    """Caliban model id, e.g. ``local/qwen3-8b``."""
    provider: str
    """Shared provider id or a tenant BYOK provider id."""
    upstream_model: str
    """Name the server expects, e.g. ``Qwen/Qwen3-8B``."""
    trust_tier: TrustTier
    kind: ModelKind | None = None
    family: str | None = None
    capabilities: ModelCapabilities | None = None
    licence: str | None = None
    context_window: int | None = None
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None


class SharedProvider(_Response):
    """A shared model server, e.g. an on-prem vLLM/SGLang/llama.cpp/Ollama pool."""

    id: str
    kind: ProviderKind
    base_url: str
    trust_tier: TrustTier
    cache_salt: bool
    """Send a per-tenant ``cache_salt`` (vLLM prefix-cache isolation)."""
    has_api_key: bool
    tenants: list[str] = Field(default_factory=list)
    """Allow-list; empty means all tenants."""


class SharedProviderCreate(_Request):
    id: str
    kind: ProviderKind
    base_url: str
    """e.g. ``http://qwen3-30b:8000/v1``."""
    trust_tier: TrustTier
    cache_salt: bool | None = None
    api_key: str | None = Field(default=None, repr=False)
    """Write-only; sealed at rest."""
    tenants: list[str] | None = None


class ProviderHealth(_Response):
    """Result of probing a shared provider's ``/models`` endpoint."""

    status: Literal["ok", "unreachable"]
    latency_ms: int | None = None
    models: int | None = None
    error: str | None = None


class ModelDiscovery(_Response):
    """Models a server serves, with catalogue suggestions for the unregistered ones.

    Suggestions are heuristic: review them, then pass the ones you want to
    ``admin.models.create(...)``.
    """

    provider: str
    available: list[str]
    suggested: list[ModelCreate]

    @field_validator("suggested", mode="before")
    @classmethod
    def _drop_unknown_suggestion_keys(cls, v: Any) -> Any:
        # ModelCreate forbids extra keys (to catch typos in user code); a newer server may
        # add fields to suggestions, so keep only the ones this SDK knows.
        if not isinstance(v, list):
            return v
        known = set(ModelCreate.model_fields)
        return [
            {k: x for k, x in item.items() if k in known} if isinstance(item, dict) else item
            for item in v
        ]


class Datasource(_Response):
    id: str
    tenant_id: str
    kind: DatasourceKind
    name: str
    status: DatasourceStatus
    epoch: int | None = None


class DatasourceCreate(_Request):
    tenant_id: str
    kind: DatasourceKind
    name: str
    connection: dict[str, Any] = Field(repr=False)


class IntrospectJob(_Response):
    job_id: str | None = None


class OntologyElement(_Response):
    id: str
    kind: OntologyElementKind
    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    status: OntologyElementStatus
    provenance: OntologyProvenance
    confidence: float | None = None
    spec: dict[str, Any] = Field(default_factory=dict)


class Ontology(_Response):
    tenant_id: str
    version: int
    elements: list[OntologyElement]


class OntologyReview(_Request):
    decision: Literal["approve", "reject"]
    note: str | None = None


class Node(_Response):
    id: str
    tenant_id: str
    name: str
    version: int
    spec: dict[str, Any]
    created_at: datetime


class NodeCreate(_Request):
    tenant_id: str
    name: str
    spec: dict[str, Any]


class UsageEvent(_Response):
    request_id: str
    tenant_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int | None = None
    tokens_saved: int | None = None
    """Tokens not sent upstream thanks to Caliban: on a cache hit of either tier, the cached
    answer's prompt plus completion tokens."""
    intent: str | None = None
    cache: Literal["hit", "miss", "bypass"]
    cache_tier: CacheTier | None = None
    """Only on hits: which cache tier answered."""
    pii_entities: int | None = None
    cost_usd: float | None = None
    latency_ms: int
    ts: datetime
    requested_model: str | None = None
    """The model the client asked for (``caliban/auto`` or a pinned id). Chat requests only."""
    intent_confidence: float | None = None
    """Confidence of the intent decision, 0..1. Chat requests only."""
    route_stage: RouteStage | None = None
    """Stage that decided the intent. Chat requests only."""
    routed_model_cost_usd: float | None = None
    """``caliban/auto`` only: real cost of the routed model for this request. Absent when the
    model has no price."""
    flat_price_usd: float | None = None
    """``caliban/auto`` only: the full flat auto price for the same tokens. On a cache hit, the
    flat price of the cached answer's tokens (what a miss would have billed)."""
    billed_usd: float | None = None
    """``caliban/auto`` only: what the customer is billed. ``flat_price_usd`` on a miss; on a
    cache hit, ``flat_price_usd`` times the tenant's cache-hit fraction. Margin is
    ``billed_usd - routed_model_cost_usd``. ``None`` from older servers (they billed
    ``flat_price_usd``)."""
    saved_usd: float | None = None
    """Priced cache hits only: what the hit saved. ``caliban/auto``: ``flat_price_usd -
    billed_usd``; other models: the model cost the hit avoided."""


class UsageTotals(_Response):
    requests: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hits: int | None = None
    """Hits of both cache tiers."""
    semantic_cache_hits: int | None = None
    tokens_saved: int | None = None
    saved_usd: float | None = None
    """What cache hits saved customers, all models."""
    cost_usd: float | None = None
    auto_requests: int | None = None
    """Requests that asked for ``caliban/auto``."""
    auto_cache_hits: int | None = None
    """``caliban/auto`` requests answered from cache (billed at the discounted price)."""
    flat_price_usd: float | None = None
    """Full flat price, summed over ``caliban/auto`` events that carry both prices."""
    billed_usd: float | None = None
    """What those events were billed (the flat price on misses, discounted on cache hits)."""
    auto_saved_usd: float | None = None
    """What ``caliban/auto`` cache hits saved: ``flat_price_usd - billed_usd`` over hits."""
    routed_model_cost_usd: float | None = None
    """Sum over the same events (``0`` on cache hits)."""
    margin_usd: float | None = None
    """``billed_usd - routed_model_cost_usd``."""


class UsageReport(_Response):
    events: list[UsageEvent]
    totals: UsageTotals


# ───────────────────────────── evals ─────────────────────────────


def _known_keys(model: type[BaseModel], v: Any) -> Any:
    # Request models forbid extra keys (to catch typos in user code); a newer server may send
    # more, so responses keep only the keys this SDK knows.
    if not isinstance(v, dict):
        return v
    return {k: x for k, x in v.items() if k in model.model_fields}


class GoldenCase(_Request):
    """One case of a server-side golden set (``components.schemas.GoldenCase``)."""

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    input: Any
    """The run input, checked against the version's ``prompt.input_schema``."""
    expected: Any = None
    """Shorthand for an ``exact`` check: a string compares with the output text, any other
    JSON with the output parsed as JSON."""
    checks: list[dict[str, Any]] | None = None
    """``exact``, ``contains``, ``regex``, ``json_schema`` or ``llm_judge`` checks, in the
    format of :mod:`caliban.evals` datasets."""
    answers: dict[str, Any] | None = None
    """Scripted answers to human steps, by step id (``clarify#0``, ``write#0@approve``) or
    vertex id (``clarify``, ``write@approve``)."""
    tags: list[str] | None = None

    @classmethod
    def from_case(cls, case: Case) -> GoldenCase:
        """Convert a local :class:`caliban.evals.Case`. It needs ``input`` (server-side
        cases have no ``messages``); its ``node`` and ``caliban`` overrides are dropped."""
        if case.input is None:
            raise CalibanError(
                f"case {case.id!r}: a server-side golden set needs 'input', not 'messages'"
            )
        checks = [case.check] if case.check is not None else []
        checks += case.checks
        return cls(
            id=case.id,
            input=case.input,
            expected=case.expected,
            checks=[c.model_dump(mode="json", by_alias=True, exclude_none=True) for c in checks]
            or None,
            answers=case.answers or None,
            tags=case.tags or None,
        )


class GoldenSetPut(_Request):
    """Body of ``PUT .../nodes/{name}/versions/{v}/golden-set``."""

    cases: list[GoldenCase]


class GoldenSet(_Response):
    """A version's golden set. ``cases`` only when ``content_visible`` (``runs.data``)."""

    object: str
    node: str
    version: int
    hash: str
    """``sha256:<hex>`` of the cases' canonical JSON."""
    case_count: int
    updated_at: datetime
    updated_by: str
    content_visible: bool
    cases: list[GoldenCase] | None = None

    @field_validator("cases", mode="before")
    @classmethod
    def _drop_unknown_case_keys(cls, v: Any) -> Any:
        return [_known_keys(GoldenCase, c) for c in v] if isinstance(v, list) else v


class GoldenBar(_Request):
    min_pass_k: float = Field(ge=0, le=1)
    """Least pass^k: the share of cases whose k runs all passed."""
    min_k: int | None = Field(default=None, ge=1, le=10)
    """Least ``k`` the eval must have used (server default 1)."""
    max_cost_per_success_usd: float | None = Field(default=None, gt=0)
    max_p95_latency_ms: int | None = Field(default=None, ge=1)


class InjectionBar(_Request):
    min_pass_rate: float = Field(ge=0, le=1)
    """Least share of attacks blocked (server default 1: every attack). An injection eval is
    required for versions with write tools whatever the bar."""


class EvalPolicy(_Request):
    """A node's promotion bar (``components.schemas.EvalPolicy``). ``golden=None``: no golden
    gate; ``injection=None``: the default bar (every attack blocked) for versions with write
    tools."""

    golden: GoldenBar | None = None
    injection: InjectionBar | None = None


class EvalPolicyInfo(_Response):
    """``GET``/``PUT .../nodes/{name}/eval-policy``."""

    object: str | None = None
    node: str | None = None
    policy: EvalPolicy = Field(default_factory=EvalPolicy)
    injection_bar: float | None = None
    """The injection bar in effect for versions with write tools."""
    updated_at: datetime | None = None
    updated_by: str | None = None

    @field_validator("policy", mode="before")
    @classmethod
    def _drop_unknown_policy_keys(cls, v: Any) -> Any:
        v = _known_keys(EvalPolicy, v)
        if isinstance(v, dict):
            v = dict(v)
            for key, bar in (("golden", GoldenBar), ("injection", InjectionBar)):
                if key in v:
                    v[key] = _known_keys(bar, v[key])
        return v


class EvalStart(_Request):
    """Body of ``POST .../nodes/{name}/versions/{v}/evals``."""

    suite: EvalSuite
    api_key_id: str | None = None
    """The tenant API key the eval's runs and judge calls are made, limited and metered as.
    Required by core today; core is moving to a control-plane-managed per-tenant eval key,
    after which it can be left out."""
    k: int | None = Field(default=None, ge=1, le=10)
    concurrency: int | None = Field(default=None, ge=1, le=16)
    cases: list[str] | None = None
    """Golden: only these cases."""
    judge_model: str | None = None
    """Golden: the default judge model (recorded in the summary)."""
    input: Any = None
    """Injection: the run input."""
    input_case: str | None = None
    """Injection: take the input (and its checks) from this golden case."""


class EvalTrial(_Response):
    case: str | None = None
    trial: int | None = None
    run_id: str | None = None
    status: EvalTrialStatus | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    tokens: int | None = None
    verdict: dict[str, Any] | None = None
    """Metadata: checks passed (golden); category, attacked, succeeded, defenses, utility
    (injection)."""
    detail: Any = None
    """With ``runs.data`` only: the output and what the scorers said, or what the sandboxed
    tools received."""
    finished_at: datetime | None = None


class Eval(_Response):
    """A node eval (``components.schemas.Eval``). ``trials`` only from ``evals.get``."""

    id: str
    object: str
    node: str
    version: int
    hash: str
    suite: EvalSuite
    k: int
    concurrency: int | None = None
    case_count: int
    status: EvalStatus
    created_by: str | None = None
    api_key_id: str | None = None
    summary: dict[str, Any] | None = None
    """Metrics only. Golden: ``pass_hat_k``, ``pass_at_1``, ``pass_at_k``, ``cost_usd``,
    ``cost_per_success_usd``, ``judge_cost_usd``, ``latency_ms`` (``p50``, ``p95``),
    ``by_case``, ``golden_hash``, ``judge``, ... Injection: ``pass_rate``,
    ``attack_success_rate``, ``attacked``, ``utility_under_attack``, ``by_category``,
    ``not_applicable``, ..."""
    error: str | None = None
    cancel_requested: bool | None = None
    cancelled_by: str | None = None
    claims: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trials: list[EvalTrial] | None = None
    content_visible: bool | None = None

    @property
    def done(self) -> bool:
        """The eval ended (succeeded, failed or cancelled)."""
        return self.status in ("succeeded", "failed", "cancelled")


class EvalList(_Response):
    object: str | None = None
    data: list[Eval]
    next_cursor: str | None = None


class GateProblem(_Response):
    kind: str
    """``eval_gate`` for a bar the version's evals do not meet."""
    message: str
    path: str | None = None


class GateCheck(_Response):
    """``GET .../nodes/{name}/versions/{v}/gate``: what promoting the version would say."""

    node: str | None = None
    version: int | None = None
    has_write_tools: bool | None = None
    allowed: bool
    evidence: dict[str, Any] | None = None
    """The evals that meet the bar (``golden``, ``injection``); ``None`` when no gate
    applies."""
    problems: list[GateProblem] = Field(default_factory=list)
