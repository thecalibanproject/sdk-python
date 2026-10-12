"""Control-plane (admin API, port 8081) client."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter

from .. import _base
from .._base import (
    DEFAULT_MAX_RETRIES,
    SyncHTTP,
    TimeoutTypes,
    parse_response,
)
from ..errors import CalibanError
from ..evals.dataset import Dataset
from ..nodes import NodeSpec, validate
from .models import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyInfo,
    Datasource,
    DatasourceCreate,
    DatasourceKind,
    Eval,
    EvalKey,
    EvalKeyRotation,
    EvalList,
    EvalPolicy,
    EvalPolicyView,
    EvalStart,
    EvalStatus,
    EvalSuite,
    GateCheck,
    GoldenBar,
    GoldenCase,
    GoldenSet,
    GoldenSetPut,
    Health,
    InjectionBar,
    IntentExemplars,
    IntrospectionJob,
    IntrospectJob,
    Model,
    ModelCapabilities,
    ModelCreate,
    ModelDiscovery,
    ModelKind,
    Node,
    NodeCreate,
    Ontology,
    OntologyBatchReview,
    OntologyDecision,
    OntologyDecisionItem,
    OntologyElement,
    OntologyElementProposal,
    OntologyProposalResult,
    OntologyReview,
    OntologyReviewResult,
    PiiMode,
    PiiPattern,
    PiiSurrogateScope,
    PiiTerm,
    ProviderHealth,
    ProviderKey,
    ProviderKeyCreate,
    ProviderKind,
    RagCollection,
    RagCollectionCreate,
    RagCollectionSettings,
    RagDocument,
    RagDocumentIngest,
    RagDocumentList,
    RagDocumentStatus,
    RagDocumentVersions,
    RagModels,
    RagTrust,
    SemanticCacheSetting,
    SharedProvider,
    SharedProviderCreate,
    Tenant,
    TenantCreate,
    TenantIntent,
    TenantIntents,
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

    def _put(self, path: str, body: Any) -> httpx.Response:
        # PUT replaces the resource, so it is idempotent: retried on 429/502/503.
        return self._admin._http.request("PUT", path, json=body)

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
        pii_on_sovereign: bool | None = None,
        pii_patterns: Sequence[PiiPattern | Mapping[str, Any]] | None = None,
        pii_dictionary: Sequence[PiiTerm | Mapping[str, Any]] | None = None,
        semantic_cache: SemanticCacheSetting | None = None,
        auto_cache_hit_fraction: float | None = None,
    ) -> Tenant:
        """Create a tenant. ``auto_cache_hit_fraction`` (0..1) overrides the share of the flat
        ``caliban/auto`` price billed for its cache hits; omit it for the deployment value.
        The PII arguments are those of :meth:`update`."""
        body = TenantCreate.model_validate(
            {
                "name": name,
                "region": region,
                "pii_default": pii_default,
                "pii_surrogate_scope": pii_surrogate_scope,
                "pii_on_sovereign": pii_on_sovereign,
                "pii_patterns": _pii_list(pii_patterns),
                "pii_dictionary": _pii_list(pii_dictionary),
                "semantic_cache": semantic_cache,
                "auto_cache_hit_fraction": auto_cache_hit_fraction,
            }
        ).to_body()
        return self._one(Tenant, self._post("tenants", body))

    def update(
        self,
        tenant_id: str,
        *,
        pii_default: PiiMode | None = None,
        pii_surrogate_scope: PiiSurrogateScope | None = None,
        pii_on_sovereign: bool | None = None,
        pii_patterns: Sequence[PiiPattern | Mapping[str, Any]] | None = None,
        pii_dictionary: Sequence[PiiTerm | Mapping[str, Any]] | None = None,
        semantic_cache: SemanticCacheSetting | None = None,
        auto_cache_hit_fraction: float | Literal["default"] | None = None,
    ) -> Tenant:
        """Change a tenant's PII, cache and cache-hit billing settings (``PATCH``). Omitted
        fields keep their value.

        ``pii_surrogate_scope="session"`` gives every request fresh PII surrogates (requests
        cannot be linked through them, and requests carrying PII bypass the caches);
        ``"tenant"`` (the default) keeps one surrogate per value within the tenant.
        ``pii_on_sovereign=True`` applies the PII mode to sovereign and attested models too
        (off by default: they receive the text as is). ``pii_patterns`` (at most 50
        ``{label, pattern}``, Rust regex syntax) and ``pii_dictionary`` (at most 10 000
        whole-word ``{term, label}``) are the tenant's own detection rules, each replaced as a
        whole; ``[]`` clears them.
        ``semantic_cache="on"`` lets eligible requests use the semantic cache (the deployment
        must enable it too). ``auto_cache_hit_fraction`` (0..1) is the share of the flat
        ``caliban/auto`` price billed when a cache tier answers (no model is called);
        ``"default"`` clears the override so the deployment value applies. Audited as
        ``tenant.update``; routers apply the change with their next snapshot. Raises
        :class:`~caliban.NotFoundError` for an unknown or deleted tenant.
        """
        clear = auto_cache_hit_fraction == "default"
        fraction = None if isinstance(auto_cache_hit_fraction, str) else auto_cache_hit_fraction
        body = TenantUpdate.model_validate(
            {
                "pii_default": pii_default,
                "pii_surrogate_scope": pii_surrogate_scope,
                "pii_on_sovereign": pii_on_sovereign,
                "pii_patterns": _pii_list(pii_patterns),
                "pii_dictionary": _pii_list(pii_dictionary),
                "semantic_cache": semantic_cache,
                "auto_cache_hit_fraction": fraction,
            }
        ).to_body()
        if clear:
            body["auto_cache_hit_fraction"] = None
        return self._one(Tenant, self._patch(f"tenants/{_p(tenant_id)}", body))


def _pii_list(items: Sequence[BaseModel | Mapping[str, Any]] | None) -> list[Any] | None:
    # An empty list is sent (it clears the server's list); None leaves it unchanged.
    if items is None:
        return None
    return [i.model_dump() if isinstance(i, BaseModel) else dict(i) for i in items]


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
        or is already revoked, and :class:`~caliban.ConflictError` for the tenant's eval key
        (``purpose="eval"``; rotate it with ``admin.evals.rotate_key``).
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
        """Start the ontology bootstrap (``202``).

        The control plane connects with the datasource's credentials, checks the principal is
        read only, samples every collection and writes proposals (``status="proposed"``) in one
        ontology version; elements a human already reviewed, and unchanged ones, are skipped.
        MongoDB only. Follow the job with :meth:`introspection`.
        """
        return self._one(IntrospectJob, self._post(f"datasources/{_p(datasource_id)}/introspect"))

    def introspection(self, datasource_id: str) -> IntrospectionJob:
        """The newest introspection job of the datasource (answered by the control plane that
        ran it). Raises :class:`~caliban.NotFoundError` when there is none."""
        return self._one(
            IntrospectionJob, self._get(f"datasources/{_p(datasource_id)}/introspection")
        )


class OntologyResource(_Resource):
    def get(self, tenant_id: str) -> Ontology:
        """The tenant's current ontology (Caliban Semantic Model).

        ``tenant_id`` is required: element ids are per tenant, and the server answers ``400``
        without it.
        """
        if not tenant_id:
            raise CalibanError("ontology.get needs a tenant_id")
        return self._one(Ontology, self._get("ontology", {"tenant_id": tenant_id}))

    def review(
        self,
        element_id: str,
        *,
        tenant_id: str,
        decision: OntologyDecision,
        note: str | None = None,
        acknowledge_breaking: bool | None = None,
    ) -> OntologyElement:
        """Approve, reject or deprecate one element of ``tenant_id`` (element ids are per
        tenant).

        Withdrawing an approved element that a published node version's fixed query touches
        raises :class:`~caliban.OntologyBreakingChangeError` (a ``ConflictError`` whose
        ``breaking`` lists them) unless ``acknowledge_breaking=True``.
        """
        body = OntologyReview(
            tenant_id=tenant_id,
            decision=decision,
            note=note,
            acknowledge_breaking=acknowledge_breaking,
        ).to_body()
        return self._one(
            OntologyElement, self._post(f"ontology/elements/{_p(element_id)}/review", body)
        )

    def approve(
        self, element_id: str, *, tenant_id: str, note: str | None = None
    ) -> OntologyElement:
        return self.review(element_id, tenant_id=tenant_id, decision="approve", note=note)

    def reject(
        self, element_id: str, *, tenant_id: str, note: str | None = None
    ) -> OntologyElement:
        return self.review(element_id, tenant_id=tenant_id, decision="reject", note=note)

    def deprecate(
        self,
        element_id: str,
        *,
        tenant_id: str,
        note: str | None = None,
        acknowledge_breaking: bool | None = None,
    ) -> OntologyElement:
        return self.review(
            element_id,
            tenant_id=tenant_id,
            decision="deprecate",
            note=note,
            acknowledge_breaking=acknowledge_breaking,
        )

    def review_batch(
        self,
        tenant_id: str,
        decisions: Mapping[str, OntologyDecision] | Sequence[OntologyDecisionItem],
        *,
        note: str | None = None,
        acknowledge_breaking: bool | None = None,
    ) -> OntologyReviewResult:
        """Review many of the tenant's elements as one new ontology version.

        ``decisions`` maps element ids to ``approve``, ``reject`` or ``deprecate`` (or is a list
        of :class:`OntologyDecisionItem`). A breaking change raises
        :class:`~caliban.OntologyBreakingChangeError` unless ``acknowledge_breaking=True``; the
        result lists the ``breaking`` changes acknowledged.
        """
        items = (
            [OntologyDecisionItem(id=k, decision=v) for k, v in decisions.items()]
            if isinstance(decisions, Mapping)
            else list(decisions)
        )
        body = OntologyBatchReview(
            decisions=items, note=note, acknowledge_breaking=acknowledge_breaking
        ).to_body()
        return self._one(
            OntologyReviewResult, self._post(f"tenants/{_p(tenant_id)}/ontology/review", body)
        )

    def propose(
        self,
        tenant_id: str,
        elements: Sequence[OntologyElementProposal | OntologyElement | Mapping[str, Any]],
        *,
        acknowledge_breaking: bool | None = None,
    ) -> OntologyProposalResult:
        """Propose elements made outside Caliban (an exported model, a curated file).

        They are written as ``proposed`` in one version for a human to review; ``status`` may
        be left out, and whatever it says is ignored. An element whose spec, name and
        description match the tenant's is skipped, so posting the same model again writes no
        version and keeps every review decision. A changed element goes back to ``proposed``:
        over an approved element that withdraws it until it is approved again, which raises
        :class:`~caliban.OntologyBreakingChangeError` when published node versions query it,
        unless ``acknowledge_breaking=True``. 1 to 5000 elements; the result always lists
        ``breaking`` (empty unless acknowledged).
        """
        if not 1 <= len(elements) <= 5000:
            raise CalibanError("propose takes 1 to 5000 elements")
        body: dict[str, Any] = {"elements": [_proposal(e) for e in elements]}
        if acknowledge_breaking is not None:
            body["acknowledge_breaking"] = acknowledge_breaking
        return self._one(
            OntologyProposalResult,
            self._post(f"tenants/{_p(tenant_id)}/ontology/proposals", body),
        )


def _proposal(
    element: OntologyElementProposal | OntologyElement | Mapping[str, Any],
) -> dict[str, Any]:
    # Sent as given: the server ignores `status` and writes every proposal as `proposed`.
    if isinstance(element, OntologyElementProposal):
        return element.to_body()
    if isinstance(element, OntologyElement):
        return element.model_dump(mode="json", exclude_none=True)
    return dict(element)


class Intents(_Resource):
    """The tenant's ``caliban/auto`` intents (kNN exemplars).

    Set here, they override the config file intent by intent; routers rebuild their index
    with the next snapshot, without a restart.
    """

    def list(self, tenant_id: str) -> TenantIntents:
        """Every intent as routers use it, with its ``source`` and node (``routes.read``)."""
        return self._one(TenantIntents, self._get(f"tenants/{_p(tenant_id)}/intents"))

    def set(self, tenant_id: str, intent: str, exemplars: Sequence[str]) -> TenantIntent:
        """Set the tenant's exemplars of an intent (``PUT``, ``routes.write``).

        At most 100 intents per tenant and 200 exemplars per intent, each 1 to 2000
        characters (duplicates dropped). An empty list removes a config-file intent for the
        tenant. Intent names are 1 to 64 characters of ``[a-z0-9_.-]``; ``pinned`` is
        reserved. Audited as ``tenant.intent.set`` with counts, never the texts.
        """
        body = IntentExemplars(exemplars=list(exemplars)).to_body()
        resp = self._put(f"tenants/{_p(tenant_id)}/intents/{_p(intent)}", body)
        return self._one(TenantIntent, resp)

    def delete(self, tenant_id: str, intent: str) -> None:
        """Drop the tenant's own exemplars of an intent; the config file's apply again.
        Raises :class:`~caliban.NotFoundError` when it has none."""
        self._delete(f"tenants/{_p(tenant_id)}/intents/{_p(intent)}")


class RagCollections(_Resource):
    def _base(self, tenant_id: str) -> str:
        return f"tenants/{_p(tenant_id)}/rag/collections"

    def list(self, tenant_id: str) -> list[RagCollection]:
        """The tenant's collections, each with ``document_counts`` per status."""
        return self._many(RagCollection, self._get(self._base(tenant_id)))

    def create(
        self,
        tenant_id: str,
        *,
        name: str,
        embedding_model: str,
        description: str | None = None,
        rerank_model: str | None = None,
        settings: RagCollectionSettings | Mapping[str, Any] | None = None,
    ) -> RagCollection:
        """Create a collection; ``embedding_model`` is fixed for its life."""
        body = RagCollectionCreate.model_validate(
            {
                "name": name,
                "embedding_model": embedding_model,
                "description": description,
                "rerank_model": rerank_model,
                "settings": settings,
            }
        ).to_body()
        return self._one(RagCollection, self._post(self._base(tenant_id), body))

    def get(self, tenant_id: str, collection: str) -> RagCollection:
        """A collection with ``used_by``: the published node versions that search it."""
        return self._one(RagCollection, self._get(f"{self._base(tenant_id)}/{_p(collection)}"))

    def delete(self, tenant_id: str, collection: str) -> None:
        """Delete a collection and its documents. Raises :class:`~caliban.ConflictError`
        while a published node version names it in its scopes."""
        self._delete(f"{self._base(tenant_id)}/{_p(collection)}")


class RagDocuments(_Resource):
    def _base(self, tenant_id: str, collection: str) -> str:
        return f"tenants/{_p(tenant_id)}/rag/collections/{_p(collection)}/documents"

    def list(
        self,
        tenant_id: str,
        collection: str,
        *,
        status: RagDocumentStatus | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> RagDocumentList:
        """One page: each document's newest version with its ``live_version``. Never the
        content; follow ``next_cursor``."""
        params = {"status": status, "cursor": cursor, "limit": limit}
        return self._one(RagDocumentList, self._get(self._base(tenant_id, collection), params))

    def ingest(
        self,
        tenant_id: str,
        collection: str,
        *,
        doc_id: str,
        content: str,
        title: str | None = None,
        content_type: str | None = None,
        source: str | None = None,
        trust: RagTrust | None = None,
    ) -> RagDocument:
        """Ingest a version of ``doc_id``: ``pending`` and invisible until indexed and its
        checks passed (an untrusted one fails at ``vet_deadline_at`` otherwise). The same
        content again returns the newest version with ``unchanged=True``."""
        body = RagDocumentIngest(
            doc_id=doc_id,
            content=content,
            title=title,
            content_type=content_type,
            source=source,
            trust=trust,
        ).to_body()
        return self._one(RagDocument, self._post(self._base(tenant_id, collection), body))

    def get(self, tenant_id: str, collection: str, doc_id: str) -> RagDocumentVersions:
        """Every version, newest first, with its ``findings`` and ``vet_deadline_at``."""
        path = f"{self._base(tenant_id, collection)}/{_p(doc_id)}"
        return self._one(RagDocumentVersions, self._get(path))

    def delete(self, tenant_id: str, collection: str, doc_id: str) -> None:
        """Delete every version; it stops being served at once."""
        self._delete(f"{self._base(tenant_id, collection)}/{_p(doc_id)}")

    def release(self, tenant_id: str, collection: str, doc_id: str, *, version: int) -> RagDocument:
        """Release a quarantined version (``rag.approve``), acknowledging its findings. It
        goes back to ``pending`` and is indexed again; its text stays labelled ``retrieved``
        in node runs."""
        path = f"{self._base(tenant_id, collection)}/{_p(doc_id)}/release"
        return self._one(
            RagDocument, self._post(path, {"version": version, "acknowledge_findings": True})
        )


class Rag(_Resource):
    """Document collections nodes search with ``builtin://rag_search`` (scope
    ``rag.<name>:read``). Needs a document registry on the control plane, else 503."""

    def __init__(self, admin: CalibanAdmin) -> None:
        super().__init__(admin)
        self.collections = RagCollections(admin)
        self.documents = RagDocuments(admin)

    def models(self, tenant_id: str) -> RagModels:
        """The embedding and rerank models the tenant can reach (``rag.read``)."""
        return self._one(RagModels, self._get(f"tenants/{_p(tenant_id)}/rag/models"))


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


class Evals(_Resource):
    """Server-side node evals: golden sets, the promotion bar, eval runs and the gate.

    These are the evals core counts when a version is promoted. Evals run on the workers as
    ordinary node runs, through the full request pipeline, and are audited. For a quick check
    of a model or a node from your machine or CI, the local harness (:mod:`caliban.evals`,
    ``caliban-eval``) runs the same golden-set format client-side, but its results never
    count toward the promotion bar.
    """

    @staticmethod
    def _node(tenant_id: str, node: str) -> str:
        return f"tenants/{_p(tenant_id)}/nodes/{_p(node)}"

    def _version(self, tenant_id: str, node: str, version: int) -> str:
        return f"{self._node(tenant_id, node)}/versions/{int(version)}"

    # golden sets

    def get_golden_set(self, tenant_id: str, node: str, version: int) -> GoldenSet:
        """The golden set of a version. ``cases`` is included only for a caller with
        ``runs.data`` (``content_visible``). Raises :class:`~caliban.NotFoundError` when the
        version has none."""
        return self._one(
            GoldenSet, self._get(f"{self._version(tenant_id, node, version)}/golden-set")
        )

    def put_golden_set(
        self,
        tenant_id: str,
        node: str,
        version: int,
        cases: Dataset | Sequence[GoldenCase | Mapping[str, Any]],
    ) -> GoldenSet:
        """Create or replace the golden set of a version (at most 200 cases).

        ``cases`` may be a local :class:`caliban.evals.Dataset`: its cases are converted with
        :meth:`GoldenCase.from_case` (each needs ``input``). The server checks every input
        against the version's input schema (422, problems of kind ``golden_set``).
        """
        if isinstance(cases, Dataset):
            items = [GoldenCase.from_case(c) for c in cases.cases]
        else:
            items = [
                c if isinstance(c, GoldenCase) else GoldenCase.model_validate(c) for c in cases
            ]
        body = GoldenSetPut(cases=items).to_body()
        path = f"{self._version(tenant_id, node, version)}/golden-set"
        return self._one(GoldenSet, self._put(path, body))

    def delete_golden_set(self, tenant_id: str, node: str, version: int) -> None:
        self._delete(f"{self._version(tenant_id, node, version)}/golden-set")

    # the promotion bar

    def get_policy(self, tenant_id: str, node: str) -> EvalPolicyView:
        """The node's promotion bar, and the injection bar in effect for versions with write
        tools."""
        return self._one(EvalPolicyView, self._get(f"{self._node(tenant_id, node)}/eval-policy"))

    def set_policy(
        self,
        tenant_id: str,
        node: str,
        *,
        golden: GoldenBar | Mapping[str, Any] | None = None,
        injection: InjectionBar | Mapping[str, Any] | None = None,
    ) -> EvalPolicyView:
        """Replace the node's promotion bar (``PUT``; needs ``nodes.publish``; audited).

        ``golden`` gates promotion on the version's latest golden eval; leaving it out removes
        the golden gate. ``injection`` lowers the injection bar (default: every attack
        blocked); an injection eval stays required for versions with write tools. Calling it
        with neither clears the bar.
        """
        body = EvalPolicy.model_validate({"golden": golden, "injection": injection}).to_body()
        resp = self._put(f"{self._node(tenant_id, node)}/eval-policy", body)
        return self._one(EvalPolicyView, resp)

    # evals

    def start(
        self,
        tenant_id: str,
        node: str,
        version: int,
        *,
        suite: EvalSuite,
        api_key_id: str | None = None,
        k: int | None = None,
        concurrency: int | None = None,
        cases: Sequence[str] | None = None,
        judge_model: str | None = None,
        input: Any = None,
        input_case: str | None = None,
    ) -> Eval:
        """Start an eval of a published version (``202``; a draft is a 409).

        ``suite="golden"`` runs every case of the version's golden set (or ``cases``) ``k``
        times (default 3); ``suite="injection"`` runs the built-in injection suite against
        sandboxed tools (``k`` default 1) with ``input``, or the input of ``input_case`` (else
        the golden set's first case).

        The eval's runs and judge calls are made, limited and metered as the tenant's eval
        key, which the control plane creates on the first eval that names no key and manages
        itself (see :meth:`get_key`). ``api_key_id`` runs the eval as a customer API key
        instead (it must be active and allowed to run the node); ``None`` is not sent.
        """
        body = EvalStart(
            suite=suite,
            api_key_id=api_key_id,
            k=k,
            concurrency=concurrency,
            cases=None if cases is None else [*cases],
            judge_model=judge_model,
            input=input,
            input_case=input_case,
        ).to_body()
        return self._one(Eval, self._post(f"{self._version(tenant_id, node, version)}/evals", body))

    def list(
        self,
        tenant_id: str,
        *,
        node: str | None = None,
        version: int | None = None,
        suite: EvalSuite | None = None,
        status: EvalStatus | Sequence[EvalStatus] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> EvalList:
        """The tenant's evals, newest first (metrics, never content). Page with ``cursor``
        (``next_cursor`` of the previous page)."""
        if limit is not None and not 1 <= limit <= 200:
            raise CalibanError("limit must be between 1 and 200")
        params = {
            "node": node,
            "version": version,
            "suite": suite,
            "status": status if status is None or isinstance(status, str) else ",".join(status),
            "limit": limit,
            "cursor": cursor,
        }
        return self._one(EvalList, self._get(f"tenants/{_p(tenant_id)}/evals", params))

    def get(self, tenant_id: str, eval_id: str) -> Eval:
        """An eval and its trials (each trial's ``detail`` only with ``runs.data``)."""
        return self._one(Eval, self._get(f"tenants/{_p(tenant_id)}/evals/{_p(eval_id)}"))

    def cancel(self, tenant_id: str, eval_id: str) -> Eval:
        """Cancel an eval. A pending one ends at once; a running one stops, with its runs in
        flight, at its worker's next check (``cancel_requested``). Raises
        :class:`~caliban.ConflictError` if it already ended otherwise."""
        return self._one(Eval, self._post(f"tenants/{_p(tenant_id)}/evals/{_p(eval_id)}/cancel"))

    def wait(
        self,
        tenant_id: str,
        eval_id: str,
        *,
        poll_s: float = 5.0,
        timeout_s: float | None = None,
    ) -> Eval:
        """Poll :meth:`get` until the eval ends. Raises :class:`~caliban.CalibanError` after
        ``timeout_s`` (the eval keeps running; :meth:`cancel` stops it)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            e = self.get(tenant_id, eval_id)
            if e.done:
                return e
            if deadline is not None and time.monotonic() >= deadline:
                raise CalibanError(f"eval {eval_id} still {e.status} after {timeout_s:g}s")
            _base._sleep(poll_s)

    # the eval key

    def get_key(self, tenant_id: str) -> EvalKey:
        """The tenant's eval key (metadata only; its plaintext never leaves the control
        plane). ``key`` is ``None`` before the first eval that names no key. Needs
        ``api_keys.read``."""
        return self._one(EvalKey, self._get(f"tenants/{_p(tenant_id)}/eval-key"))

    def rotate_key(self, tenant_id: str) -> EvalKeyRotation:
        """Revoke the eval key and create a new one (``api_keys.write``; audited). Evals in
        flight lose their key, so rotate between evals."""
        return self._one(EvalKeyRotation, self._post(f"tenants/{_p(tenant_id)}/eval-key/rotate"))

    def gate(self, tenant_id: str, node: str, version: int) -> GateCheck:
        """What promoting the version would say: ``allowed``, the evals that meet the bar
        (``evidence``, with each pinned subnode's under ``subnodes``) or the ``problems``
        (a ``path`` names a subnode whose own gate failed)."""
        return self._one(GateCheck, self._get(f"{self._version(tenant_id, node, version)}/gate"))


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
        self.intents = Intents(self)
        self.rag = Rag(self)
        self.nodes = Nodes(self)
        self.evals = Evals(self)
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
