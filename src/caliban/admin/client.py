"""Control-plane (admin API, port 8081) client."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter

from .._base import (
    DEFAULT_MAX_RETRIES,
    SyncHTTP,
    TimeoutTypes,
    parse_response,
)
from ..errors import CalibanError
from ..nodes import NodeSpec, validate
from .models import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyInfo,
    Datasource,
    DatasourceCreate,
    DatasourceKind,
    Health,
    IntrospectJob,
    Model,
    ModelCapabilities,
    ModelCreate,
    ModelDiscovery,
    ModelKind,
    Node,
    NodeCreate,
    Ontology,
    OntologyElement,
    OntologyReview,
    PiiMode,
    PiiSurrogateScope,
    ProviderHealth,
    ProviderKey,
    ProviderKeyCreate,
    ProviderKind,
    SemanticCacheSetting,
    SharedProvider,
    SharedProviderCreate,
    Tenant,
    TenantCreate,
    TenantUpdate,
    TrustTier,
    UsageReport,
)

__all__ = ["DEFAULT_ADMIN_URL", "CalibanAdmin"]

DEFAULT_ADMIN_URL = "http://localhost:8081"
API_PREFIX = "/api/v1"

M = TypeVar("M", bound=BaseModel)


def _p(segment: str) -> str:
    return quote(segment, safe="")


def model_id_path(model_id: str) -> str:
    """Percent-encode a model id for a URL path, keeping its ``/`` separators.

    The contract sends ids such as ``local/qwen3-8b`` unescaped
    (``DELETE /api/v1/models/local/qwen3-8b``). Each segment is quoted; empty, ``.``
    and ``..`` segments are rejected because URL normalisation would change the path.
    """
    segments = model_id.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise CalibanError(
            f"Invalid model id {model_id!r}: segments must be non-empty and not '.' or '..'"
        )
    return "/".join(_p(seg) for seg in segments)


class _Resource:
    def __init__(self, admin: CalibanAdmin) -> None:
        self._admin = admin

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> httpx.Response:
        return self._admin._http.request("GET", path, params=params)

    def _post(self, path: str, body: Any = None) -> httpx.Response:
        # Not idempotent: retried on 429/503 only, never 502 (the server may have acted).
        return self._admin._http.request("POST", path, json=body)

    def _patch(self, path: str, body: Any) -> httpx.Response:
        # Not idempotent by default (each call is audited): retried on 429/503 only.
        return self._admin._http.request("PATCH", path, json=body)

    def _delete(self, path: str) -> httpx.Response:
        return self._admin._http.request("DELETE", path)

    @staticmethod
    def _one(model: type[M], resp: httpx.Response) -> M:
        return parse_response(resp, TypeAdapter(model))

    @staticmethod
    def _many(model: type[M], resp: httpx.Response) -> list[M]:
        return parse_response(resp, TypeAdapter(list[model]))  # type: ignore[valid-type]


class Tenants(_Resource):
    def list(self, *, include_deleted: bool | None = None) -> list[Tenant]:
        """Active tenants; ``include_deleted=True`` also returns tombstones."""
        return self._many(Tenant, self._get("tenants", {"include_deleted": include_deleted}))

    def get(self, tenant_id: str) -> Tenant:
        """Raises :class:`~caliban.NotFoundError` for an unknown or deleted tenant."""
        return self._one(Tenant, self._get(f"tenants/{_p(tenant_id)}"))

    def delete(self, tenant_id: str) -> None:
        """Permanently delete a tenant.

        In one audited transaction the server turns it into a tombstone, revokes its API
        keys, destroys its BYOK credentials, removes its routes and soft-deletes its
        datasources and nodes. The id cannot be reused. Raises
        :class:`~caliban.NotFoundError` if the tenant is unknown or already deleted.
        """
        self._delete(f"tenants/{_p(tenant_id)}")

    def create(
        self,
        *,
        name: str,
        region: str | None = None,
        pii_default: PiiMode | None = None,
        pii_surrogate_scope: PiiSurrogateScope | None = None,
        semantic_cache: SemanticCacheSetting | None = None,
        auto_cache_hit_fraction: float | None = None,
    ) -> Tenant:
        """Create a tenant. ``auto_cache_hit_fraction`` (0..1) overrides the share of the flat
        ``caliban/auto`` price billed for its cache hits; omit it for the deployment value."""
        body = TenantCreate(
            name=name,
            region=region,
            pii_default=pii_default,
            pii_surrogate_scope=pii_surrogate_scope,
            semantic_cache=semantic_cache,
            auto_cache_hit_fraction=auto_cache_hit_fraction,
        ).to_body()
        return self._one(Tenant, self._post("tenants", body))

    def update(
        self,
        tenant_id: str,
        *,
        pii_default: PiiMode | None = None,
        pii_surrogate_scope: PiiSurrogateScope | None = None,
        semantic_cache: SemanticCacheSetting | None = None,
        auto_cache_hit_fraction: float | Literal["default"] | None = None,
    ) -> Tenant:
        """Change a tenant's PII, cache and cache-hit billing settings (``PATCH``). Omitted
        fields keep their value.

        ``pii_surrogate_scope="session"`` gives every request fresh PII surrogates (requests
        cannot be linked through them, and requests carrying PII bypass the caches);
        ``"tenant"`` (the default) keeps one surrogate per value within the tenant.
        ``semantic_cache="on"`` lets eligible requests use the semantic cache (the deployment
        must enable it too). ``auto_cache_hit_fraction`` (0..1) is the share of the flat
        ``caliban/auto`` price billed when a cache tier answers (no model is called);
        ``"default"`` clears the override so the deployment value applies. Audited as
        ``tenant.update``; routers apply the change with their next snapshot. Raises
        :class:`~caliban.NotFoundError` for an unknown or deleted tenant.
        """
        clear = auto_cache_hit_fraction == "default"
        fraction = None if isinstance(auto_cache_hit_fraction, str) else auto_cache_hit_fraction
        body = TenantUpdate(
            pii_default=pii_default,
            pii_surrogate_scope=pii_surrogate_scope,
            semantic_cache=semantic_cache,
            auto_cache_hit_fraction=fraction,
        ).to_body()
        if clear:
            body["auto_cache_hit_fraction"] = None
        return self._one(Tenant, self._patch(f"tenants/{_p(tenant_id)}", body))


class ApiKeys(_Resource):
    """Tenant Caliban API keys (``cal_...``). Only metadata is listable."""

    def list(self, tenant_id: str, *, include_revoked: bool | None = None) -> list[ApiKeyInfo]:
        """Active keys; ``include_revoked=True`` also returns revoked ones."""
        return self._many(
            ApiKeyInfo,
            self._get(f"tenants/{_p(tenant_id)}/api-keys", {"include_revoked": include_revoked}),
        )

    def create(self, tenant_id: str, *, name: str | None = None) -> ApiKeyCreated:
        """Mint a key. ``result.key`` holds the plaintext and is shown exactly once."""
        body = ApiKeyCreate(name=name).to_body()
        return self._one(ApiKeyCreated, self._post(f"tenants/{_p(tenant_id)}/api-keys", body))

    def revoke(self, tenant_id: str, key_id: str) -> None:
        """Revoke a key (permanent).

        A standalone deployment rejects it on the next request; a split-mode router
        rejects it after its next snapshot poll (10 s by default). Raises
        :class:`~caliban.NotFoundError` if the key is unknown, belongs to another tenant
        or is already revoked.
        """
        self._delete(f"tenants/{_p(tenant_id)}/api-keys/{_p(key_id)}")


class ProviderKeys(_Resource):
    """BYOK provider credentials and local endpoints. Secrets are write-only."""

    def list(self, tenant_id: str) -> list[ProviderKey]:
        return self._many(ProviderKey, self._get(f"tenants/{_p(tenant_id)}/provider-keys"))

    def create(
        self,
        tenant_id: str,
        *,
        kind: ProviderKind,
        label: str,
        trust_tier: TrustTier,
        base_url: str | None = None,
        api_key: str | None = None,
        provider_id: str | None = None,
    ) -> ProviderKey:
        body = ProviderKeyCreate(
            kind=kind,
            label=label,
            trust_tier=trust_tier,
            base_url=base_url,
            api_key=api_key,
            provider_id=provider_id,
        ).to_body()
        return self._one(ProviderKey, self._post(f"tenants/{_p(tenant_id)}/provider-keys", body))

    def delete(self, tenant_id: str, key_id: str) -> None:
        """Delete a credential; the secret is crypto-shredded server-side."""
        self._delete(f"tenants/{_p(tenant_id)}/provider-keys/{_p(key_id)}")


class Models(_Resource):
    """The model catalogue."""

    def list(self) -> list[Model]:
        return self._many(Model, self._get("models"))

    def create(
        self,
        spec: ModelCreate | Mapping[str, Any] | None = None,
        /,
        *,
        id: str | None = None,
        provider: str | None = None,
        upstream_model: str | None = None,
        trust_tier: TrustTier | None = None,
        kind: ModelKind | None = None,
        family: str | None = None,
        capabilities: ModelCapabilities | Mapping[str, Any] | None = None,
        licence: str | None = None,
        context_window: int | None = None,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
    ) -> Model:
        """Register a model, e.g. an open model served on-prem.

        Pass the fields as keywords, or a :class:`ModelCreate` (typically a reviewed
        suggestion from :meth:`Providers.discover`) with optional keyword overrides::

            for s in admin.providers.discover("gpu-pool").suggested:
                admin.models.create(s, context_window=32768)

        Raises :class:`~caliban.ConflictError` (409) if the id already exists.
        """
        fields: dict[str, Any] = {
            "id": id,
            "provider": provider,
            "upstream_model": upstream_model,
            "trust_tier": trust_tier,
            "kind": kind,
            "family": family,
            "capabilities": capabilities,
            "licence": licence,
            "context_window": context_window,
            "price_in_per_mtok": price_in_per_mtok,
            "price_out_per_mtok": price_out_per_mtok,
        }
        overrides = {k: v for k, v in fields.items() if v is not None}
        if spec is None:
            base: dict[str, Any] = {}
        elif isinstance(spec, ModelCreate):
            base = spec.model_dump(exclude_none=True)
        else:
            base = dict(spec)
        body = ModelCreate.model_validate({**base, **overrides}).to_body()
        return self._one(Model, self._post("models", body))

    def delete(self, model_id: str) -> None:
        """Remove a model (409 :class:`~caliban.ConflictError` if a route still uses it).

        Ids such as ``local/qwen3-8b`` are sent with the slash unescaped.
        """
        self._delete(f"models/{model_id_path(model_id)}")


class Providers(_Resource):
    """Shared model servers (on-prem pools such as vLLM, SGLang, llama.cpp, Ollama)."""

    def list(self) -> list[SharedProvider]:
        return self._many(SharedProvider, self._get("providers"))

    def create(
        self,
        *,
        id: str,
        kind: ProviderKind,
        base_url: str,
        trust_tier: TrustTier,
        cache_salt: bool | None = None,
        api_key: str | None = None,
        tenants: Sequence[str] | None = None,
    ) -> SharedProvider:
        """Register a shared model server. ``api_key`` is write-only and sealed at rest;
        ``tenants`` is an allow-list (omit for all tenants)."""
        body = SharedProviderCreate(
            id=id,
            kind=kind,
            base_url=base_url,
            trust_tier=trust_tier,
            cache_salt=cache_salt,
            api_key=api_key,
            tenants=None if tenants is None else [*tenants],
        ).to_body()
        return self._one(SharedProvider, self._post("providers", body))

    def delete(self, provider_id: str) -> None:
        """Remove a shared model server (409 if models still use it)."""
        self._delete(f"providers/{_p(provider_id)}")

    def health(self, provider_id: str) -> ProviderHealth:
        """Probe the server's ``/models`` endpoint."""
        return self._one(ProviderHealth, self._get(f"providers/{_p(provider_id)}/health"))

    def discover(self, provider_id: str) -> ModelDiscovery:
        """List the models the server serves and suggest catalogue entries for new ones.

        Suggestions are heuristic: review them, then register the ones you want with
        ``admin.models.create(suggestion)``.
        """
        return self._one(ModelDiscovery, self._post(f"providers/{_p(provider_id)}/discover"))


class Datasources(_Resource):
    def list(self, *, tenant_id: str | None = None) -> list[Datasource]:
        return self._many(Datasource, self._get("datasources", {"tenant_id": tenant_id}))

    def create(
        self,
        *,
        tenant_id: str,
        kind: DatasourceKind,
        name: str,
        connection: Mapping[str, Any],
    ) -> Datasource:
        body = DatasourceCreate(
            tenant_id=tenant_id, kind=kind, name=name, connection=dict(connection)
        ).to_body()
        return self._one(Datasource, self._post("datasources", body))

    def delete(self, tenant_id: str, datasource_id: str) -> None:
        """Permanently delete a datasource of ``tenant_id``.

        Its stored connection settings are wiped and its name can be used again. Raises
        :class:`~caliban.NotFoundError` if it is unknown, belongs to another tenant or is
        already deleted.
        """
        self._delete(f"tenants/{_p(tenant_id)}/datasources/{_p(datasource_id)}")

    def introspect(self, datasource_id: str) -> IntrospectJob:
        """Start an ontology bootstrap job; proposed elements land with status=proposed."""
        return self._one(IntrospectJob, self._post(f"datasources/{_p(datasource_id)}/introspect"))


class OntologyResource(_Resource):
    def get(self, *, tenant_id: str | None = None) -> Ontology:
        return self._one(Ontology, self._get("ontology", {"tenant_id": tenant_id}))

    def review(
        self,
        element_id: str,
        *,
        decision: Literal["approve", "reject"],
        note: str | None = None,
    ) -> OntologyElement:
        body = OntologyReview(decision=decision, note=note).to_body()
        return self._one(
            OntologyElement, self._post(f"ontology/elements/{_p(element_id)}/review", body)
        )

    def approve(self, element_id: str, *, note: str | None = None) -> OntologyElement:
        return self.review(element_id, decision="approve", note=note)

    def reject(self, element_id: str, *, note: str | None = None) -> OntologyElement:
        return self.review(element_id, decision="reject", note=note)


class Nodes(_Resource):
    def list(self, *, tenant_id: str | None = None) -> list[Node]:
        return self._many(Node, self._get("nodes", {"tenant_id": tenant_id}))

    def create(
        self,
        *,
        tenant_id: str,
        name: str,
        spec: NodeSpec | Mapping[str, Any],
        validate_spec: bool = True,
    ) -> Node:
        """Create a new node version.

        ``spec`` is validated locally against the node schema first (disable with
        ``validate_spec=False`` to let the server be the only judge).
        """
        if isinstance(spec, NodeSpec):
            spec_body = spec.to_dict()
        elif validate_spec:
            spec_body = validate(spec).to_dict()
        else:
            spec_body = dict(spec)
        body = NodeCreate(tenant_id=tenant_id, name=name, spec=spec_body).to_body()
        return self._one(Node, self._post("nodes", body))

    def delete(self, tenant_id: str, node_id: str) -> None:
        """Permanently delete one node version of ``tenant_id``; its version number is not
        reused. Raises :class:`~caliban.NotFoundError` if it is unknown, belongs to another
        tenant or is already deleted.
        """
        self._delete(f"tenants/{_p(tenant_id)}/nodes/{_p(node_id)}")


class Usage(_Resource):
    def get(self, *, tenant_id: str | None = None, limit: int = 100) -> UsageReport:
        if not 1 <= limit <= 1000:
            raise CalibanError("limit must be between 1 and 1000")
        return self._one(UsageReport, self._get("usage", {"tenant_id": tenant_id, "limit": limit}))


class CalibanAdmin:
    """Synchronous control-plane client.

    Args:
        token: Admin bearer token. Defaults to ``$CALIBAN_ADMIN_TOKEN``.
        base_url: Control-plane origin, e.g. ``http://localhost:8081`` (a trailing
            ``/api/v1`` is accepted). Defaults to ``$CALIBAN_ADMIN_URL``.

    Retries: GET/DELETE retry on 429/502/503; POST retries only on 429/503, unless an
    ``Idempotency-Key`` header is set (for example via ``default_headers``).
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutTypes = 30.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        tok = token if token is not None else os.environ.get("CALIBAN_ADMIN_TOKEN")
        if not tok:
            raise CalibanError("No admin token. Pass token=... or set CALIBAN_ADMIN_TOKEN.")
        origin = (base_url or os.environ.get("CALIBAN_ADMIN_URL") or DEFAULT_ADMIN_URL).rstrip("/")
        if origin.endswith(API_PREFIX):
            origin = origin[: -len(API_PREFIX)]
        self._http = SyncHTTP(
            base_url=origin + API_PREFIX,
            token=tok,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.tenants = Tenants(self)
        self.api_keys = ApiKeys(self)
        self.provider_keys = ProviderKeys(self)
        self.models = Models(self)
        self.providers = Providers(self)
        self.datasources = Datasources(self)
        self.ontology = OntologyResource(self)
        self.nodes = Nodes(self)
        self.usage = Usage(self)

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> Health:
        return parse_response(self._http.request("GET", "health"), TypeAdapter(Health))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> CalibanAdmin:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
