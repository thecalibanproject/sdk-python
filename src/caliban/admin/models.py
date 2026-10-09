"""Control-plane models, hand-written to match ``core/api/openapi.yaml`` (v0.1.0).

Keep in sync with the contract. Response models allow extra fields so that a
newer control plane does not break an older SDK; request models forbid them so
typos are caught client-side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..types import ModelCapabilities, ModelKind, TrustTier

__all__ = [
    "ApiKeyCreate",
    "ApiKeyCreated",
    "ApiKeyInfo",
    "Datasource",
    "DatasourceCreate",
    "DatasourceKind",
    "DatasourceStatus",
    "Health",
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
    "ProviderHealth",
    "ProviderKey",
    "ProviderKeyCreate",
    "ProviderKind",
    "SharedProvider",
    "SharedProviderCreate",
    "Tenant",
    "TenantCreate",
    "TenantStatus",
    "TrustTier",
    "UsageEvent",
    "UsageReport",
    "UsageTotals",
]

PiiMode = Literal["off", "mask", "reversible"]
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
    created_at: datetime
    status: TenantStatus | None = None
    """``"deleted"`` only appears with ``include_deleted=True``. ``None`` from older servers."""
    deleted_at: datetime | None = None


class TenantCreate(_Request):
    name: str
    region: str | None = None
    pii_default: PiiMode | None = None


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
    """Tokens not sent upstream thanks to Caliban (e.g. cache hits)."""
    intent: str | None = None
    cache: Literal["hit", "miss", "bypass"]
    pii_entities: int | None = None
    cost_usd: float | None = None
    latency_ms: int
    ts: datetime


class UsageTotals(_Response):
    requests: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hits: int | None = None
    tokens_saved: int | None = None
    cost_usd: float | None = None


class UsageReport(_Response):
    events: list[UsageEvent]
    totals: UsageTotals
