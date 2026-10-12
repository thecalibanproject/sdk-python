"""Control-plane models, hand-written to match ``core/api/openapi.yaml`` (v0.1.0).

Keep in sync with the contract. Response models allow extra fields so that a
newer control plane does not break an older SDK; request models forbid them so
typos are caught client-side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

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
    "EvalKey",
    "EvalKeyRotation",
    "EvalList",
    "EvalPolicy",
    "EvalPolicyView",
    "EvalStart",
    "EvalStatus",
    "EvalSuite",
    "EvalSummary",
    "EvalTrial",
    "EvalTrialStatus",
    "GateCheck",
    "GateEvidence",
    "GoldenBar",
    "GoldenCase",
    "GoldenCaseSummary",
    "GoldenEvidence",
    "GoldenSet",
    "GoldenSetPut",
    "GoldenSummary",
    "GoldenVerdict",
    "Health",
    "InjectionBar",
    "InjectionCaseResult",
    "InjectionCategory",
    "InjectionCategoryName",
    "InjectionDefense",
    "InjectionEvidence",
    "InjectionReadOnly",
    "InjectionSummary",
    "InjectionVerdict",
    "IntentExemplars",
    "IntentSource",
    "IntrospectJob",
    "IntrospectionJob",
    "IntrospectionStatus",
    "JudgeInfo",
    "LatencyPercentiles",
    "Model",
    "ModelCapabilities",
    "ModelCreate",
    "ModelDiscovery",
    "ModelKind",
    "Node",
    "NodeCreate",
    "NotApplicable",
    "Ontology",
    "OntologyBatchReview",
    "OntologyBreakingChange",
    "OntologyDecision",
    "OntologyDecisionItem",
    "OntologyElement",
    "OntologyElementKind",
    "OntologyElementProposal",
    "OntologyElementStatus",
    "OntologyNodeVersion",
    "OntologyProposalResult",
    "OntologyProvenance",
    "OntologyReview",
    "OntologyReviewResult",
    "PiiMode",
    "PiiPattern",
    "PiiSurrogateScope",
    "PiiTerm",
    "Problem",
    "ProblemKind",
    "ProviderHealth",
    "ProviderKey",
    "ProviderKeyCreate",
    "ProviderKind",
    "RagCollection",
    "RagCollectionCreate",
    "RagCollectionSettings",
    "RagDocument",
    "RagDocumentIngest",
    "RagDocumentList",
    "RagDocumentStatus",
    "RagDocumentVersions",
    "RagFinding",
    "RagModel",
    "RagModels",
    "RagTrust",
    "RouteStage",
    "RunStatus",
    "SemanticCacheSetting",
    "SharedProvider",
    "SharedProviderCreate",
    "Tenant",
    "TenantCreate",
    "TenantIntent",
    "TenantIntentSet",
    "TenantIntents",
    "TenantStatus",
    "TenantUpdate",
    "TrialError",
    "TrustTier",
    "UsageByPurpose",
    "UsageEvent",
    "UsagePurpose",
    "UsageReport",
    "UsageTotals",
    "VerdictCheck",
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
OntologyDecision = Literal["approve", "reject", "deprecate"]
IntentSource = Literal["config", "api"]
RagTrust = Literal["trusted", "untrusted"]
RagDocumentStatus = Literal[
    "pending", "indexing", "ready", "quarantined", "failed", "superseded", "deleted"
]
EvalSuite = Literal["golden", "injection"]
InjectionReadOnly = Literal["required", "optional"]
EvalStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
EvalTrialStatus = Literal["running", "passed", "failed", "error"]
RunStatus = Literal[
    "pending",
    "running",
    "sleeping",
    "input_required",
    "succeeded",
    "failed",
    "budget_exhausted",
    "cancelled",
]
InjectionCategoryName = Literal[
    "tainted_write",
    "exfiltration",
    "pii_leak",
    "tool_poisoning",
    "replay",
    "subnode",
    "caller_data",
    "answer_integrity",
]
InjectionDefense = Literal[
    "approval_required", "pii_guard", "declared_tools_only", "manifest_scan", "answer_guard"
]
UsagePurpose = Literal["rag.ingest", "rag.search"]
"""Work Caliban did on the tenant's behalf: ``rag.ingest`` (embedding a collection's
documents) or ``rag.search`` (a node's retrieval query, its embedding and reranking)."""
ProblemKind = Literal[
    "spec",
    "tool",
    "datasource_scope",
    "budget",
    "node_ref",
    "cycle",
    "eval_gate",
    "golden_set",
    "allowlist",
]


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


PiiLabel = Annotated[str, Field(pattern=r"^[A-Za-z0-9_]{1,40}$")]


class PiiPattern(_Request):
    """One of the tenant's own L0 PII patterns (``components.schemas.PiiPattern``)."""

    model_config = ConfigDict(extra="ignore")

    label: PiiLabel
    """Case-insensitive: stored as sent, used upper-cased (``phone`` is ``PHONE``).
    ``NATIONAL_ID``, ``PHONE``, ``EMAIL``, ``PERSON``, ``ORG``, ``LOCATION``, ``IBAN`` or
    ``CARD`` (and the aliases ``ID``, ``NATIONALID``, ``NAME``, ``ORGANIZATION``,
    ``ADDRESS``) use the built-in surrogates; other labels get shape-preserving ones and are
    masked as ``[LABEL]``, upper-cased."""
    pattern: str = Field(max_length=500)
    """Rust ``regex`` syntax (linear time: no backreferences or lookaround); must not match
    the empty string."""


class PiiTerm(_Request):
    """One term of the tenant's PII dictionary: whole-word, case-insensitive."""

    model_config = ConfigDict(extra="ignore")

    term: str = Field(min_length=2, max_length=200)
    label: PiiLabel
    """Case-insensitive, as for :attr:`PiiPattern.label`."""


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
    pii_on_sovereign: bool | None = None
    """The PII mode also applies to sovereign and attested models. ``None`` from older
    servers."""
    pii_patterns: list[PiiPattern] | None = None
    """Only for callers with ``tenant.write`` on the tenant (``None`` for others, and from
    older servers): read :attr:`pii_patterns_count` to see whether the tenant has any."""
    pii_dictionary: list[PiiTerm] | None = None
    """Only for callers with ``tenant.write`` (the dictionary may hold staff or customer
    names); others get :attr:`pii_dictionary_count`."""
    pii_patterns_count: int | None = None
    """How many patterns the tenant has, for every caller of ``tenants.get`` and
    ``tenants.list``. ``None`` from older servers (and on create and update results)."""
    pii_dictionary_count: int | None = None
    """How many dictionary terms the tenant has, for every caller of ``tenants.get`` and
    ``tenants.list``. ``None`` from older servers (and on create and update results)."""
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
    pii_on_sovereign: bool | None = None
    pii_patterns: list[PiiPattern] | None = Field(default=None, max_length=50)
    pii_dictionary: list[PiiTerm] | None = Field(default=None, max_length=10000)
    semantic_cache: SemanticCacheSetting | None = None
    auto_cache_hit_fraction: float | None = Field(default=None, ge=0, le=1)


class TenantUpdate(_Request):
    """Body of ``PATCH /api/v1/tenants/{tenantId}``. Unset fields keep their value.

    ``auto_cache_hit_fraction`` is sent only when set; clearing it (an explicit ``null``) goes
    through ``tenants.update(..., auto_cache_hit_fraction="default")``."""

    pii_default: PiiMode | None = None
    pii_surrogate_scope: PiiSurrogateScope | None = None
    pii_on_sovereign: bool | None = None
    pii_patterns: list[PiiPattern] | None = Field(default=None, max_length=50)
    """Replaced as a whole; ``[]`` clears them."""
    pii_dictionary: list[PiiTerm] | None = Field(default=None, max_length=10000)
    """Replaced as a whole; ``[]`` clears it. Audited as a count of terms, never the terms."""
    semantic_cache: SemanticCacheSetting | None = None
    auto_cache_hit_fraction: float | None = Field(default=None, ge=0, le=1)


class ApiKeyInfo(_Response):
    id: str
    name: str
    prefix: str
    created_at: datetime
    revoked_at: datetime | None = None
    """Set once the key is revoked (listed only with ``include_revoked=True``)."""
    purpose: Literal["eval"] | None = None
    """``"eval"``: the tenant's eval key, managed by the control plane (it cannot be revoked
    through ``api_keys.revoke``; rotate it with ``evals.rotate_key``)."""


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


IntrospectionStatus = Literal["running", "succeeded", "failed"]


class IntrospectionJob(_Response):
    """An ontology bootstrap job of a datasource (``components.schemas.IntrospectionJob``)."""

    id: str | None = None
    datasource_id: str | None = None
    tenant_id: str | None = None
    status: IntrospectionStatus | None = None
    objects: int | None = None
    """Collections the snapshot found."""
    proposed: int | None = None
    skipped: int | None = None
    """Proposals for elements already reviewed, or unchanged."""
    version: int | None = None
    """The ontology version the proposals were written in."""
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IntrospectJob(_Response):
    """``POST /api/v1/datasources/{id}/introspect`` (``202``)."""

    job_id: str | None = None
    job: IntrospectionJob | None = None
    """The job as it started. ``None`` from older servers."""


class OntologyElementProposal(_Request):
    """An element for ``ontology.propose`` (``components.schemas.OntologyElementProposal``).

    ``status`` may be left out: whatever it says is ignored, because the server writes every
    proposal as ``proposed``."""

    id: str = Field(min_length=1, max_length=200)
    kind: OntologyElementKind
    name: str
    description: str | None = None
    synonyms: list[str] | None = None
    status: OntologyElementStatus | None = None
    provenance: OntologyProvenance
    confidence: float | None = None
    spec: dict[str, Any] | None = None


class OntologyElement(_Response):
    """A stored element (``components.schemas.OntologyElement``): a proposal that always has a
    ``status``, plus ``changed_in``."""

    id: str
    kind: OntologyElementKind
    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    status: OntologyElementStatus
    provenance: OntologyProvenance
    confidence: float | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    changed_in: int | None = None
    """The ontology version its spec or status last changed in. ``None`` from older servers
    or on elements written before it was tracked."""


class Ontology(_Response):
    tenant_id: str
    version: int
    elements: list[OntologyElement]


class OntologyReview(_Request):
    """Body of ``POST /api/v1/ontology/elements/{id}/review``: ``tenant_id`` is required
    (element ids are per tenant)."""

    tenant_id: str
    decision: OntologyDecision
    note: str | None = None
    acknowledge_breaking: bool | None = None


class OntologyDecisionItem(_Request):
    id: str
    decision: OntologyDecision


class OntologyBatchReview(_Request):
    """Body of ``POST /api/v1/tenants/{t}/ontology/review``: one new ontology version."""

    decisions: list[OntologyDecisionItem] = Field(min_length=1, max_length=1000)
    note: str | None = None
    acknowledge_breaking: bool | None = None


class OntologyNodeVersion(_Response):
    """A published node version, structured (``OntologyBreaking.versions``)."""

    node: str
    version: int


class OntologyBreakingChange(_Response):
    """An approved element withdrawn while published node versions' fixed queries touch it
    (``components.schemas.OntologyBreaking``)."""

    element: str
    change: str
    """For example ``approved -> proposed``."""
    nodes: list[str] = Field(default_factory=list)
    """``name@vN`` of the published node versions."""
    versions: list[OntologyNodeVersion] = Field(default_factory=list)
    """The same node versions, structured. Empty from older servers."""


class OntologyReviewResult(_Response):
    """Result of a batch review: the new ``version``, the reviewed elements and the breaking
    changes acknowledged."""

    tenant_id: str
    version: int
    elements: list[OntologyElement] = Field(default_factory=list)
    breaking: list[OntologyBreakingChange] = Field(default_factory=list)


class OntologyProposalResult(_Response):
    """Result of ``POST /api/v1/tenants/{t}/ontology/proposals`` (``201``)."""

    tenant_id: str
    version: int
    """The tenant's ontology version after the call (unchanged when nothing was written)."""
    proposed: int
    skipped: int
    """Elements left as they are: unchanged in spec, name and description, whatever status
    they were sent with (so a re-posted reviewed element keeps its review decision)."""
    breaking: list[OntologyBreakingChange] = Field(default_factory=list)
    """Approved elements this call withdrew that published node versions query (empty unless
    ``acknowledge_breaking``)."""


class TenantIntent(_Response):
    """One ``caliban/auto`` intent as routers use it."""

    intent: str
    exemplars: list[str] = Field(default_factory=list)
    source: IntentSource | None = None
    """``config``: the config file; ``api``: set through the admin API."""
    node: str | None = None
    """The node the intent hands to (``node/<name>``), if any."""


class TenantIntentSet(_Response):
    """Result of ``intents.put``: the intent as stored (``components.schemas.TenantIntentSet``)."""

    tenant_id: str
    intent: str
    exemplars: list[str] = Field(default_factory=list)
    """As stored: trimmed, duplicates dropped."""
    source: Literal["api"] = "api"


class TenantIntents(_Response):
    """``GET /api/v1/tenants/{t}/intents``."""

    tenant_id: str
    intents: list[TenantIntent] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    """Config-file intents an empty list removes for the tenant."""


class IntentExemplars(_Request):
    exemplars: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(max_length=200)


# ───────────────────────────── document collections (RAG) ─────────────────────────────


class RagModel(_Response):
    id: str
    provider: str
    trust_tier: TrustTier


class RagModels(_Response):
    """The embedding and rerank models the tenant can reach."""

    embedding: list[RagModel]
    rerank: list[RagModel]


class RagCollectionSettings(BaseModel):
    """Closed on the server (``additionalProperties: false``); responses keep extra keys."""

    model_config = ConfigDict(extra="allow")

    chunk_tokens: int | None = Field(default=None, ge=64, le=2048)
    default_trust: RagTrust | None = None
    nav_min_chunks: int | None = Field(default=None, ge=4)
    vet_deadline_secs: int | None = Field(default=None, ge=60, le=604800)
    """An untrusted document whose checks have not passed this long after ingest (or after
    its release) fails and stays invisible."""


class RagCollection(_Response):
    id: str
    tenant_id: str
    name: str
    """What node scopes name: ``rag.<name>:read``."""
    description: str | None = None
    embedding_model: str
    rerank_model: str | None = None
    settings: RagCollectionSettings
    created_at: datetime
    created_by: str
    document_counts: dict[str, int] | None = None
    """In the listing only: documents per status, counting each document's newest version;
    deleted documents are not counted."""
    used_by: list[str] | None = None
    """From ``collections.get`` only: published versions (``name@vN``) that search it."""


class RagCollectionCreate(_Request):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,63}$")
    embedding_model: str
    description: str | None = Field(default=None, max_length=1000)
    rerank_model: str | None = None
    settings: RagCollectionSettings | None = None


class RagFinding(_Response):
    kind: str
    """``instruction``, ``exfiltration``, ``hidden_unicode``, ``url`` or ``near_duplicate``."""
    chunk: int
    """The chunk (0-based) it was found in."""
    heading_path: list[str] | None = None
    """The headings above that chunk (pseudonymised, never an excerpt of what triggered the
    finding). ``None`` from older servers."""


class RagDocument(_Response):
    """One version of a document (``components.schemas.RagDocument``). Never the content."""

    tenant_id: str
    collection_id: str
    doc_id: str
    version: int
    content_type: str
    source: str | None = None
    trust: RagTrust
    bytes: int
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    status: RagDocumentStatus
    chunks: int
    tokens: int
    pii_entities: int
    findings: list[RagFinding] = Field(default_factory=list)
    findings_acknowledged: bool
    error: str | None = None
    attempts: int
    created_at: datetime
    created_by: str
    indexed_at: datetime | None = None
    released_by: str | None = None
    released_at: datetime | None = None
    vet_deadline_at: datetime | None = None
    """Untrusted documents only: when its checks must have passed, or it fails."""
    live_version: int | None = None
    """In the document listing: the version being served (``None``: none is ready)."""
    unchanged: bool | None = None
    """On ingest: ``True`` when the same content was already there."""


class RagDocumentIngest(_Request):
    doc_id: str = Field(min_length=1, max_length=200)
    content: str = Field(repr=False)
    title: str | None = Field(default=None, max_length=500)
    content_type: str | None = None
    source: str | None = Field(default=None, max_length=2000)
    trust: RagTrust | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class RagDocumentList(_Response):
    documents: list[RagDocument]
    next_cursor: str | None = None


class RagDocumentVersions(_Response):
    doc_id: str
    live_version: int | None = None
    versions: list[RagDocument]


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
    purpose: UsagePurpose | None = None
    """Work Caliban did on the tenant's behalf (``rag.ingest``, ``rag.search``). ``None`` for
    client calls and node model calls."""


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


class UsageByPurpose(_Response):
    purpose: str
    totals: UsageTotals


class UsageReport(_Response):
    events: list[UsageEvent]
    totals: UsageTotals
    by_purpose: list[UsageByPurpose] | None = None
    """The totals per ``purpose``, so retrieval spend is separable from the tenant's own
    calls. ``None`` from older servers."""


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
    read_only: InjectionReadOnly | None = None
    """Read-only versions (read tools that bring data in, no write tool): ``required``
    (server default) gates them on an injection eval too; ``optional`` lets them promote
    without one. A finished injection eval is checked against the bar either way."""


class EvalPolicy(_Request):
    """A node's promotion bar (``components.schemas.EvalPolicy``). ``golden=None``: no golden
    gate; ``injection=None``: the default bar (every attack blocked) for versions with write
    tools."""

    golden: GoldenBar | None = None
    injection: InjectionBar | None = None


class EvalPolicyView(_Response):
    """``GET``/``PUT .../nodes/{name}/eval-policy`` (``components.schemas.EvalPolicyView``)."""

    object: str
    node: str
    policy: EvalPolicy
    injection_bar: float
    """The injection bar in effect for versions with write tools and read-only versions
    (default 1)."""
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
    """Optional: a customer API key to run the eval as (it must be active and allowed to run
    the node). Absent: the tenant's eval key, which the control plane creates and manages."""
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


class LatencyPercentiles(_Response):
    p50: int | None = None
    """Nearest rank, ms."""
    p95: int | None = None


class GoldenCaseSummary(_Response):
    case: str
    trials: int
    passed: int
    all_passed: bool


class JudgeInfo(_Response):
    """The judge configuration an eval ran with, so scores compare under the same judge."""

    default_model: str | None = None
    prompt_version: int | None = None
    prompt_sha256: str | None = None
    temperature: float | None = None


class GoldenSummary(_Response):
    """Metrics of a finished golden eval (``components.schemas.GoldenSummary``)."""

    suite: Literal["golden"]
    k: int
    cases: int
    trials: int
    passed_trials: int
    errors: int
    pass_hat_k: float
    """pass^k: the share of cases whose k runs all passed."""
    pass_at_1: float
    pass_at_k: float
    cost_usd: float
    cost_per_success_usd: float | None = None
    """``None`` without a success."""
    judge_cost_usd: float
    tokens: int
    latency_ms: LatencyPercentiles
    by_case: list[GoldenCaseSummary]
    golden_hash: str | None = None
    judge: JudgeInfo | None = None


class InjectionCategory(_Response):
    trials: int
    blocked: int
    pass_rate: float


class NotApplicable(_Response):
    case: str
    reason: str


class InjectionCaseResult(_Response):
    case: str
    trial: int
    blocked: bool
    attacked: bool | None = None
    defenses: list[str] | None = None
    run_id: str | None = None


class InjectionSummary(_Response):
    """Metrics of a finished injection eval (``components.schemas.InjectionSummary``)."""

    suite: Literal["injection"]
    suite_version: int
    trials: int
    blocked: int
    pass_rate: float
    """Share of attacks blocked."""
    attack_success_rate: float
    attacks_succeeded: int
    errors: int
    attacked: int
    """Trials where the model followed the injection."""
    utility_under_attack: float
    """Share of runs that still did their job."""
    by_category: dict[str, InjectionCategory]
    not_applicable: list[NotApplicable]
    cost_usd: float
    latency_ms: LatencyPercentiles
    cases: list[InjectionCaseResult]


EvalSummary = Annotated[GoldenSummary | InjectionSummary, Field(discriminator="suite")]


class VerdictCheck(_Response):
    type: Literal["exact", "contains", "regex", "json_schema", "llm_judge"]
    passed: bool
    score: float | None = None
    """The judge's score (``llm_judge``)."""


class GoldenVerdict(_Response):
    run_status: RunStatus
    checks: list[VerdictCheck]
    judge_usd: float | None = None


class InjectionVerdict(_Response):
    case: str
    category: InjectionCategoryName
    succeeded: bool
    """The attack reached its goal."""
    attacked: bool
    """The model followed the injection."""
    defenses: list[InjectionDefense]
    tool_calls: int
    """Sandboxed calls of the run."""
    utility: bool | None = None
    """The run still did its job (absent for the manifest scan)."""
    run_status: RunStatus | None = None


class TrialError(_Response):
    """The verdict of a trial whose run could not be created."""

    error: str


class EvalTrial(_Response):
    case: str | None = None
    trial: int | None = None
    run_id: str | None = None
    status: EvalTrialStatus | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    tokens: int | None = None
    verdict: TrialError | GoldenVerdict | InjectionVerdict | None = None
    """Metadata, never content; the shape follows the eval's suite."""
    detail: Any = None
    """With ``runs.data`` only: the output and what the scorers said, or what the sandboxed
    tools received. ``None`` once run retention purged it (``Eval.content_purged``)."""
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
    summary: EvalSummary | None = None
    """Metrics only, never content; ``None`` until the eval ends. A :class:`GoldenSummary` or
    an :class:`InjectionSummary`, following ``suite``."""
    error: str | None = None
    cancel_requested: bool | None = None
    cancelled_by: str | None = None
    claims: int | None = None
    content_purged: bool | None = None
    """Run retention cleared the eval's sealed content (trial ``detail``, the plan); the
    summary and verdicts stay."""
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


class Problem(_Response):
    """A reason a node version is refused (``components.schemas.Problem``)."""

    kind: ProblemKind
    """``eval_gate`` for a bar the version's evals do not meet; ``allowlist`` only in
    warnings."""
    message: str
    path: str | None = None
    """Where in the spec; for ``eval_gate``, the subnode (``node://<child>@v<N>``) whose own
    gate failed."""


class GoldenEvidence(_Response):
    eval: str
    k: int
    pass_hat_k: float


class InjectionEvidence(_Response):
    eval: str
    pass_rate: float
    bar: float


class GateEvidence(_Response):
    """The evals that met the bar (``components.schemas.GateEvidence``)."""

    golden: GoldenEvidence | None = None
    injection: InjectionEvidence | None = None
    subnodes: dict[str, GateEvidence | None] | None = None
    """Per pinned subnode version (``name@vN``): its own evidence, or ``None`` when no gate
    applies to it."""


class GateCheck(_Response):
    """``GET .../nodes/{name}/versions/{v}/gate``: what promoting the version would say."""

    node: str | None = None
    version: int | None = None
    has_write_tools: bool | None = None
    has_untrusted_reads: bool | None = None
    """Read tools that bring data in (MCP, ``datasource_query``, ``rag_search``): the version
    is gated on an injection eval unless the policy makes it optional."""
    allowed: bool
    evidence: GateEvidence | None = None
    """The evals that meet the bar; ``None`` when no gate applies."""
    problems: list[Problem] = Field(default_factory=list)
    """Every reason it cannot be promoted."""


class EvalKey(_Response):
    """``GET /api/v1/tenants/{tenantId}/eval-key``."""

    tenant_id: str
    key: ApiKeyInfo | None = None
    """``None`` before the tenant's first eval that names no key."""


class EvalKeyRotation(_Response):
    """``POST /api/v1/tenants/{tenantId}/eval-key/rotate``."""

    tenant_id: str
    key: ApiKeyInfo
    revoked: str | None = None
    """The id of the key it replaced."""
