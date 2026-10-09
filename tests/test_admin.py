from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import respx

from caliban import (
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
