from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import httpx
import pytest
import respx

from caliban import (
    APIConnectionError,
    APIResponseValidationError,
    CalibanAdmin,
    CalibanError,
    NotFoundError,
    UpstreamError,
)
from caliban.admin import models as m
from caliban.nodes import NodeValidationError

BASE = "http://cp.test:8081/api/v1"
TS = "2026-10-02T10:00:00Z"

TENANT = {"id": "t1", "name": "Acme", "region": None, "pii_default": "reversible", "created_at": TS}
NODE_SPEC = {
    "kind": "agent",
    "prompt": {"system": "Be terse."},
    "model_policy": {"candidates": ["tier:small"]},
    "budgets": {"steps": 5, "tokens": 1000, "wall_clock_s": 30},
}


@pytest.fixture
def admin() -> Iterator[CalibanAdmin]:
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        client = CalibanAdmin(token="adm", base_url="http://cp.test:8081/api/v1/")
        client.router = router  # type: ignore[attr-defined]
        yield client


def router(admin: CalibanAdmin) -> respx.MockRouter:
    return admin.router  # type: ignore[attr-defined,no-any-return]


def test_base_url_and_auth(admin: CalibanAdmin) -> None:
    route = (
        router(admin)
        .get("/health")
        .respond(200, json={"status": "ok", "version": "0.1.0", "mode": "standalone"})
    )
    assert admin.base_url == BASE
    assert admin.health().mode == "standalone"
    assert route.calls.last.request.headers["authorization"] == "Bearer adm"


def test_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBAN_ADMIN_TOKEN", raising=False)
    with pytest.raises(CalibanError, match="admin token"):
        CalibanAdmin()


def test_tenants(admin: CalibanAdmin) -> None:
    r = router(admin)
    r.get("/tenants").respond(200, json=[TENANT])
    r.get("/tenants/t1").respond(200, json=TENANT)
    create = r.post("/tenants").respond(201, json=TENANT)

    tenants = admin.tenants.list()
    assert tenants[0].name == "Acme" and tenants[0].created_at.year == 2026
    assert admin.tenants.get("t1").id == "t1"
    admin.tenants.create(name="Acme", pii_default="mask")
    assert json.loads(create.calls.last.request.content) == {"name": "Acme", "pii_default": "mask"}


def test_api_keys(admin: CalibanAdmin) -> None:
    info = {"id": "k1", "name": "ci", "prefix": "cal_abcd", "created_at": TS}
    r = router(admin)
    r.get("/tenants/t1/api-keys").respond(200, json=[info])
    r.post("/tenants/t1/api-keys").respond(201, json={**info, "key": "cal_secret"})
    assert admin.api_keys.list("t1")[0].prefix == "cal_abcd"
    created = admin.api_keys.create("t1", name="ci")
    assert created.key == "cal_secret"
    assert "cal_secret" not in repr(created)


def test_provider_keys(admin: CalibanAdmin) -> None:
    pk = {
        "id": "p1",
        "tenant_id": "t1",
        "kind": "openai_compatible",
        "label": "vllm",
        "base_url": "http://vllm:8000/v1",
        "trust_tier": "t0_sovereign",
        "last4": None,
        "created_at": TS,
    }
    r = router(admin)
    r.get("/tenants/t1/provider-keys").respond(200, json=[pk])
    post = r.post("/tenants/t1/provider-keys").respond(201, json=pk)
    delete = r.delete("/tenants/t1/provider-keys/p1").respond(204)

    assert admin.provider_keys.list("t1")[0].trust_tier == "t0_sovereign"
    admin.provider_keys.create(
        "t1",
        kind="openai_compatible",
        label="vllm",
        trust_tier="t0_sovereign",
        base_url="http://vllm:8000/v1",
    )
    assert "api_key" not in json.loads(post.calls.last.request.content)
    admin.provider_keys.delete("t1", "p1")
    assert delete.called
    with pytest.raises(ValueError):
        m.ProviderKeyCreate(kind="nope", label="x", trust_tier="t3_public")  # type: ignore[arg-type]


def test_models_datasources_introspect(admin: CalibanAdmin) -> None:
    r = router(admin)
    r.get("/models").respond(
        200,
        json=[
            {
                "id": "gpt",
                "provider_id": "openai",
                "provider_kind": "openai",
                "upstream_model": "gpt-x",
                "kind": "chat",
                "capabilities": {"tools": True},
                "trust_tier": "t3_public",
                "price_in_per_mtok": 1.5,
            }
        ],
    )
    ds = {"id": "d1", "tenant_id": "t1", "kind": "postgres", "name": "dw", "status": "pending"}
    lst = r.get("/datasources").respond(200, json=[ds])
    post = r.post("/datasources").respond(201, json=ds)
    r.post("/datasources/d1/introspect").respond(202, json={"job_id": "j1"})

    assert admin.models.list()[0].price_in_per_mtok == 1.5
    admin.datasources.list(tenant_id="t1")
    assert lst.calls.last.request.url.params["tenant_id"] == "t1"
    admin.datasources.list()
    assert "tenant_id" not in lst.calls.last.request.url.params
    admin.datasources.create(
        tenant_id="t1", kind="postgres", name="dw", connection={"dsn": "postgres://x"}
    )
    assert json.loads(post.calls.last.request.content)["connection"] == {"dsn": "postgres://x"}
    assert admin.datasources.introspect("d1").job_id == "j1"


def test_ontology(admin: CalibanAdmin) -> None:
    el = {
        "id": "e1",
        "kind": "metric",
        "name": "revenue",
        "status": "proposed",
        "provenance": "introspect",
        "confidence": 0.8,
    }
    r = router(admin)
    r.get("/ontology").respond(200, json={"tenant_id": "t1", "version": 4, "elements": [el]})
    review = r.post("/ontology/elements/e1/review").respond(200, json={**el, "status": "approved"})

    onto = admin.ontology.get(tenant_id="t1")
    assert onto.version == 4 and onto.elements[0].synonyms == []
    assert admin.ontology.approve("e1", note="lgtm").status == "approved"
    assert json.loads(review.calls.last.request.content) == {"decision": "approve", "note": "lgtm"}
    admin.ontology.review("e1", decision="reject")
    assert json.loads(review.calls.last.request.content) == {"decision": "reject"}


def test_nodes_create_validates_locally(admin: CalibanAdmin) -> None:
    node = {
        "id": "n1",
        "tenant_id": "t1",
        "name": "triage",
        "version": 1,
        "spec": NODE_SPEC,
        "created_at": TS,
    }
    r = router(admin)
    r.get("/nodes").respond(200, json=[node])
    post = r.post("/nodes").respond(201, json=node)

    assert admin.nodes.list(tenant_id="t1")[0].version == 1
    created = admin.nodes.create(tenant_id="t1", name="triage", spec=NODE_SPEC)
    assert created.id == "n1"
    sent = json.loads(post.calls.last.request.content)
    assert sent["spec"]["budgets"]["depth"] == 3  # defaults made explicit

    bad = {**NODE_SPEC, "budgets": {"steps": 0, "tokens": 1, "wall_clock_s": 1}}
    with pytest.raises(NodeValidationError):
        admin.nodes.create(tenant_id="t1", name="triage", spec=bad)
    assert post.call_count == 1
    admin.nodes.create(tenant_id="t1", name="triage", spec=bad, validate_spec=False)
    assert post.call_count == 2


def test_usage(admin: CalibanAdmin) -> None:
    ev = {
        "request_id": "r",
        "tenant_id": "t1",
        "model": "m",
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "cache": "hit",
        "latency_ms": 5,
        "ts": TS,
        "tokens_saved": 7,
        "intent": "extraction",
    }
    route = (
        router(admin)
        .get("/usage")
        .respond(200, json={"events": [ev], "totals": {"requests": 1, "cost_usd": 0.01}})
    )
    rep = admin.usage.get(tenant_id="t1", limit=10)
    assert rep.events[0].tokens_saved == 7 and rep.totals.cost_usd == 0.01
    assert route.calls.last.request.url.params["limit"] == "10"
    with pytest.raises(CalibanError):
        admin.usage.get(limit=5000)


def test_deletes_return_none_on_204(admin: CalibanAdmin) -> None:
    r = router(admin)
    routes = [
        r.delete("/tenants/t1").respond(204),
        r.delete("/tenants/t1/api-keys/k1").respond(204),
        r.delete("/tenants/t1/datasources/ds1").respond(204),
        r.delete("/tenants/t1/nodes/n1").respond(204),
    ]
    calls: list[Callable[[], object]] = [
        lambda: admin.tenants.delete("t1"),
        lambda: admin.api_keys.revoke("t1", "k1"),
        lambda: admin.datasources.delete("t1", "ds1"),
        lambda: admin.nodes.delete("t1", "n1"),
    ]
    assert [call() for call in calls] == [None, None, None, None]
    for route in routes:
        assert route.call_count == 1
        req = route.calls.last.request
        assert req.headers["authorization"] == "Bearer adm"
        assert req.content == b""


DELETE_CALLS: list[tuple[str, Callable[[CalibanAdmin], None]]] = [
    ("/tenants/gone", lambda a: a.tenants.delete("gone")),
    ("/tenants/t1/api-keys/gone", lambda a: a.api_keys.revoke("t1", "gone")),
    ("/tenants/t1/datasources/gone", lambda a: a.datasources.delete("t1", "gone")),
    ("/tenants/t1/nodes/gone", lambda a: a.nodes.delete("t1", "gone")),
]


@pytest.mark.parametrize(("path", "call"), DELETE_CALLS, ids=[p for p, _ in DELETE_CALLS])
def test_deletes_raise_not_found_on_404(
    admin: CalibanAdmin, path: str, call: Callable[[CalibanAdmin], None]
) -> None:
    route = (
        router(admin)
        .delete(path)
        .respond(
            404,
            json={"error": {"message": "not found", "type": "not_found", "code": None}},
            headers={"x-caliban-request-id": "r404"},
        )
    )
    with pytest.raises(NotFoundError) as exc_info:
        call(admin)
    assert exc_info.value.status_code == 404
    assert route.call_count == 1


def test_delete_paths_are_escaped() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    client = CalibanAdmin(
        token="adm",
        base_url="http://cp.test:8081",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.nodes.delete("t/1", "n/1")
    client.api_keys.revoke("t1", "../k")
    assert [req.method for req in seen] == ["DELETE", "DELETE"]
    assert seen[0].url.raw_path == b"/api/v1/tenants/t%2F1/nodes/n%2F1"
    assert seen[1].url.raw_path == b"/api/v1/tenants/t1/api-keys/..%2Fk"


def test_revoke_retried_on_502(admin: CalibanAdmin) -> None:
    route = (
        router(admin)
        .delete("/tenants/t1/api-keys/k1")
        .mock(side_effect=[httpx.Response(502), httpx.Response(204)])
    )
    admin.api_keys.revoke("t1", "k1")
    assert route.call_count == 2


def test_list_include_deleted_and_include_revoked(admin: CalibanAdmin) -> None:
    deleted = {**TENANT, "id": "t2", "status": "deleted", "deleted_at": "2026-10-08T12:00:00Z"}
    key = {"id": "k1", "name": "ci", "prefix": "cal_abcd", "created_at": TS}
    revoked = {**key, "revoked_at": "2026-10-08T12:00:00Z"}
    r = router(admin)
    tenants = r.get("/tenants").mock(
        side_effect=[
            httpx.Response(200, json=[TENANT]),
            httpx.Response(200, json=[{**TENANT, "status": "active", "deleted_at": None}, deleted]),
            httpx.Response(200, json=[]),
        ]
    )
    keys = r.get("/tenants/t1/api-keys").mock(
        side_effect=[httpx.Response(200, json=[key]), httpx.Response(200, json=[revoked])]
    )

    old = admin.tenants.list()
    assert old[0].status is None and old[0].deleted_at is None  # older server: fields absent
    listed = admin.tenants.list(include_deleted=True)
    assert [t.status for t in listed] == ["active", "deleted"]
    assert listed[1].deleted_at is not None and listed[1].deleted_at.day == 8
    admin.tenants.list(include_deleted=False)
    assert [c.request.url.query for c in tenants.calls] == [
        b"",
        b"include_deleted=true",
        b"include_deleted=false",
    ]

    assert admin.api_keys.list("t1")[0].revoked_at is None
    got = admin.api_keys.list("t1", include_revoked=True)
    assert got[0].revoked_at is not None and got[0].revoked_at.year == 2026
    assert [c.request.url.query for c in keys.calls] == [b"", b"include_revoked=true"]


def test_admin_errors_and_retry_policy(admin: CalibanAdmin) -> None:
    r = router(admin)
    r.get("/tenants/missing").respond(404, json={"error": {"message": "nope", "type": "not_found"}})
    with pytest.raises(NotFoundError):
        admin.tenants.get("missing")

    # GET is retried on 502...
    get = r.get("/models").mock(side_effect=[httpx.Response(502), httpx.Response(200, json=[])])
    assert admin.models.list() == []
    assert get.call_count == 2
    # ...but a non-idempotent POST is not.
    post = r.post("/tenants").respond(
        502, json={"error": {"message": "u", "type": "upstream_error"}}
    )
    with pytest.raises(UpstreamError):
        admin.tenants.create(name="x")
    assert post.call_count == 1


def test_admin_post_retried_on_429_and_503(admin: CalibanAdmin, no_sleep: list[float]) -> None:
    post = (
        router(admin)
        .post("/tenants")
        .mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(503),
                httpx.Response(201, json=TENANT),
            ]
        )
    )
    assert admin.tenants.create(name="Acme").id == "t1"
    assert post.call_count == 3
    assert no_sleep[0] == 1.0


def test_admin_delete_retried_on_502(admin: CalibanAdmin) -> None:
    route = (
        router(admin)
        .delete("/providers/p1")
        .mock(side_effect=[httpx.Response(502), httpx.Response(204)])
    )
    admin.providers.delete("p1")
    assert route.call_count == 2


def test_admin_post_with_idempotency_key_retried_on_502() -> None:
    with respx.mock(base_url=BASE) as r:
        post = r.post("/tenants").mock(
            side_effect=[httpx.Response(502), httpx.Response(201, json=TENANT)]
        )
        admin = CalibanAdmin(
            token="adm", base_url=BASE, default_headers={"Idempotency-Key": "tenant-acme"}
        )
        assert admin.tenants.create(name="Acme").id == "t1"
        assert post.call_count == 2
        assert post.calls.last.request.headers["idempotency-key"] == "tenant-acme"


def test_admin_post_not_retried_after_send_failure(admin: CalibanAdmin) -> None:
    post = router(admin).post("/tenants").mock(side_effect=httpx.ReadError("reset"))
    with pytest.raises(APIConnectionError):
        admin.tenants.create(name="x")
    assert post.call_count == 1


def test_path_segments_are_escaped() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TENANT)

    client = CalibanAdmin(
        token="adm",
        base_url="http://cp.test:8081",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.tenants.get("t/../1")
    assert seen[0].url.raw_path == b"/api/v1/tenants/t%2F..%2F1"


def test_contract_mismatch_raises_typed_error(admin: CalibanAdmin) -> None:
    router(admin).get("/tenants").respond(200, json=[{"id": "t1"}])
    with pytest.raises(APIResponseValidationError, match="does not match the contract"):
        admin.tenants.list()


def test_tenant_update_patches_only_given_fields(admin: CalibanAdmin) -> None:
    updated = {**TENANT, "pii_surrogate_scope": "session", "semantic_cache": "on"}
    route = router(admin).patch("/tenants/t1").respond(200, json=updated)
    t = admin.tenants.update("t1", pii_surrogate_scope="session", semantic_cache="on")
    req = route.calls.last.request
    assert req.method == "PATCH"
    assert req.headers["authorization"] == "Bearer adm"
    assert req.headers["content-type"] == "application/json"
    assert json.loads(req.content) == {"pii_surrogate_scope": "session", "semantic_cache": "on"}
    assert (t.pii_surrogate_scope, t.semantic_cache) == ("session", "on")

    admin.tenants.update("t1", pii_default="mask")
    assert json.loads(route.calls.last.request.content) == {"pii_default": "mask"}


def test_tenant_update_escapes_path_and_rejects_bad_values() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TENANT)

    client = CalibanAdmin(
        token="adm",
        base_url="http://cp.test:8081",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.tenants.update("t/1", semantic_cache="off")
    assert seen[0].url.raw_path == b"/api/v1/tenants/t%2F1"
    with pytest.raises(ValueError, match="semantic_cache"):
        client.tenants.update("t1", semantic_cache="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extra"):
        m.TenantUpdate(name="x")  # type: ignore[call-arg]
    assert len(seen) == 1


def test_tenant_cache_hit_fraction_set_clear_and_validated(admin: CalibanAdmin) -> None:
    r = router(admin)
    route = r.patch("/tenants/t1").mock(
        side_effect=[
            httpx.Response(200, json={**TENANT, "auto_cache_hit_fraction": 0.15}),
            httpx.Response(200, json={**TENANT, "auto_cache_hit_fraction": None}),
        ]
    )
    t = admin.tenants.update("t1", auto_cache_hit_fraction=0.15)
    assert json.loads(route.calls[0].request.content) == {"auto_cache_hit_fraction": 0.15}
    assert t.auto_cache_hit_fraction == 0.15
    # "default" sends an explicit null: the override is cleared, the deployment value applies.
    t = admin.tenants.update("t1", auto_cache_hit_fraction="default")
    assert json.loads(route.calls[1].request.content) == {"auto_cache_hit_fraction": None}
    assert t.auto_cache_hit_fraction is None
    # Out of range never reaches the server.
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="auto_cache_hit_fraction"):
            admin.tenants.update("t1", auto_cache_hit_fraction=bad)
    assert route.call_count == 2
    create = r.post("/tenants").respond(201, json={**TENANT, "auto_cache_hit_fraction": 0.1})
    created = admin.tenants.create(name="Acme", auto_cache_hit_fraction=0.1)
    assert json.loads(create.calls.last.request.content) == {
        "name": "Acme",
        "auto_cache_hit_fraction": 0.1,
    }
    assert created.auto_cache_hit_fraction == 0.1


def test_tenant_update_not_found(admin: CalibanAdmin) -> None:
    route = (
        router(admin)
        .patch("/tenants/gone")
        .respond(404, json={"error": {"message": "nope", "type": "not_found", "code": None}})
    )
    with pytest.raises(NotFoundError):
        admin.tenants.update("gone", semantic_cache="on")
    assert route.call_count == 1


def test_tenant_update_retry_policy(admin: CalibanAdmin) -> None:
    # PATCH is treated like POST: retried on 429/503, never on 502 (each call is audited).
    retried = (
        router(admin)
        .patch("/tenants/t1")
        .mock(side_effect=[httpx.Response(503), httpx.Response(200, json=TENANT)])
    )
    admin.tenants.update("t1", semantic_cache="off")
    assert retried.call_count == 2
    assert retried.calls[0].request.content == retried.calls[1].request.content
    bad = (
        router(admin)
        .patch("/tenants/t2")
        .mock(
            side_effect=[
                httpx.Response(502, json={"error": {"message": "u", "type": "upstream_error"}}),
                httpx.Response(200, json=TENANT),
            ]
        )
    )
    with pytest.raises(UpstreamError):
        admin.tenants.update("t2", semantic_cache="off")
    assert bad.call_count == 1


def test_tenant_settings_on_create_and_older_servers(admin: CalibanAdmin) -> None:
    r = router(admin)
    create = r.post("/tenants").respond(
        201, json={**TENANT, "pii_surrogate_scope": "tenant", "semantic_cache": "off"}
    )
    r.get("/tenants").respond(200, json=[TENANT])
    t = admin.tenants.create(name="Acme", pii_surrogate_scope="tenant", semantic_cache="off")
    assert json.loads(create.calls.last.request.content) == {
        "name": "Acme",
        "pii_surrogate_scope": "tenant",
        "semantic_cache": "off",
    }
    assert (t.pii_surrogate_scope, t.semantic_cache) == ("tenant", "off")
    older = admin.tenants.list()[0]
    assert older.pii_surrogate_scope is None and older.semantic_cache is None
    assert older.auto_cache_hit_fraction is None


USAGE_BASE = {
    "request_id": "r1",
    "tenant_id": "t1",
    "model": "local/qwen3-8b",
    "prompt_tokens": 12,
    "completion_tokens": 7,
    "cache": "miss",
    "latency_ms": 40,
    "ts": TS,
}
USAGE_AUTO_HIT = {
    **USAGE_BASE,
    "request_id": "r2",
    "cache": "hit",
    "cache_tier": "semantic",
    "tokens_saved": 19,
    "intent": "translate",
    "requested_model": "caliban/auto",
    "intent_confidence": 0.912,
    "route_stage": "knn",
    "routed_model_cost_usd": 0,
    "flat_price_usd": 0.00026,
    "billed_usd": 0.000052,
    "saved_usd": 0.000208,
}
USAGE_TOTALS = {
    "requests": 2,
    "prompt_tokens": 24,
    "completion_tokens": 14,
    "cache_hits": 1,
    "semantic_cache_hits": 1,
    "tokens_saved": 19,
    "saved_usd": 0.000208,
    "cost_usd": 0.000026,
    "auto_requests": 1,
    "auto_cache_hits": 1,
    "flat_price_usd": 0.00026,
    "billed_usd": 0.000052,
    "auto_saved_usd": 0.000208,
    "routed_model_cost_usd": 0,
    "margin_usd": 0.000052,
}


def test_usage_new_event_fields_and_totals(admin: CalibanAdmin) -> None:
    router(admin).get("/usage").respond(
        200, json={"events": [USAGE_BASE, USAGE_AUTO_HIT], "totals": USAGE_TOTALS}
    )
    rep = admin.usage.get(tenant_id="t1")
    plain, auto = rep.events
    assert plain.cache == "miss"
    assert plain.cache_tier is None and plain.requested_model is None
    assert plain.intent_confidence is None and plain.route_stage is None
    assert plain.routed_model_cost_usd is None and plain.flat_price_usd is None
    assert (auto.cache, auto.cache_tier, auto.tokens_saved) == ("hit", "semantic", 19)
    assert (auto.requested_model, auto.intent_confidence, auto.route_stage) == (
        "caliban/auto",
        0.912,
        "knn",
    )
    assert (auto.routed_model_cost_usd, auto.flat_price_usd) == (0, 0.00026)
    # A caliban/auto cache hit: billed a share of the flat price, the rest is the saving.
    assert (auto.billed_usd, auto.saved_usd) == (0.000052, 0.000208)
    assert plain.billed_usd is None and plain.saved_usd is None
    totals = rep.totals
    assert totals.model_dump(exclude_none=True) == USAGE_TOTALS
    assert totals.semantic_cache_hits == 1 and totals.auto_requests == 1
    assert totals.flat_price_usd is not None and totals.routed_model_cost_usd is not None
    assert totals.billed_usd is not None and totals.auto_saved_usd is not None
    assert totals.auto_cache_hits == 1
    assert totals.margin_usd == pytest.approx(totals.billed_usd - totals.routed_model_cost_usd)
    assert totals.auto_saved_usd == pytest.approx(totals.flat_price_usd - totals.billed_usd)


def test_usage_from_older_server_without_new_fields(admin: CalibanAdmin) -> None:
    router(admin).get("/usage").respond(
        200, json={"events": [USAGE_BASE], "totals": {"requests": 1, "cache_hits": 0}}
    )
    rep = admin.usage.get()
    assert rep.events[0].cache_tier is None
    totals = rep.totals
    assert (totals.requests, totals.cache_hits) == (1, 0)
    assert totals.semantic_cache_hits is None and totals.auto_requests is None
    assert totals.flat_price_usd is None and totals.margin_usd is None
    assert totals.billed_usd is None and totals.auto_cache_hits is None


def test_usage_cache_enum_unchanged(admin: CalibanAdmin) -> None:
    # `cache` stays hit/miss/bypass; the tier lives in `cache_tier`.
    router(admin).get("/usage").respond(
        200, json={"events": [{**USAGE_BASE, "cache": "semantic"}], "totals": {}}
    )
    with pytest.raises(APIResponseValidationError):
        admin.usage.get()
