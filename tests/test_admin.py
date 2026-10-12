from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import httpx
import pytest
import respx

from caliban import (
    ONTOLOGY_BREAKING_CHANGE,
    APIConnectionError,
    APIResponseValidationError,
    CalibanAdmin,
    CalibanError,
    ConflictError,
    NotFoundError,
    OntologyBreakingChangeError,
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
    # The statuses introspection sets.
    for status in ("introspecting", "ready", "failed"):
        assert m.Datasource.model_validate({**ds, "status": status}).status == status
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
    assert admin.ontology.approve("e1", tenant_id="t1", note="lgtm").status == "approved"
    assert json.loads(review.calls.last.request.content) == {
        "tenant_id": "t1",
        "decision": "approve",
        "note": "lgtm",
    }
    admin.ontology.review("e1", tenant_id="t1", decision="reject")
    assert json.loads(review.calls.last.request.content) == {
        "tenant_id": "t1",
        "decision": "reject",
    }
    admin.ontology.deprecate("e1", tenant_id="t1", acknowledge_breaking=True)
    assert json.loads(review.calls.last.request.content) == {
        "tenant_id": "t1",
        "decision": "deprecate",
        "acknowledge_breaking": True,
    }


def test_ontology_review_needs_the_tenant(admin: CalibanAdmin) -> None:
    with pytest.raises(TypeError):
        admin.ontology.review("e1", decision="approve")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="decision"):
        m.OntologyReview(tenant_id="t1", decision="maybe")  # type: ignore[arg-type]


INTROSPECTION = {
    "id": "ij1",
    "datasource_id": "d1",
    "tenant_id": "t1",
    "status": "succeeded",
    "objects": 4,
    "proposed": 9,
    "skipped": 2,
    "version": 7,
    "started_at": TS,
    "finished_at": TS,
}


def test_introspection_job(admin: CalibanAdmin) -> None:
    r = router(admin)
    running = {**INTROSPECTION, "status": "running", "version": None}
    r.post("/datasources/d1/introspect").respond(202, json={"job_id": "ij1", "job": running})
    r.get("/datasources/d%2F1/introspection").respond(200, json=INTROSPECTION)
    r.get("/datasources/none/introspection").respond(
        404, json={"error": {"message": "introspection job not found", "type": "not_found"}}
    )

    started = admin.datasources.introspect("d1")
    assert started.job is not None and started.job.status == "running"
    assert started.job.version is None
    job = admin.datasources.introspection("d/1")
    assert (job.status, job.proposed, job.skipped, job.version) == ("succeeded", 9, 2, 7)
    with pytest.raises(NotFoundError):
        admin.datasources.introspection("none")


def test_ontology_batch_review_and_proposals(admin: CalibanAdmin) -> None:
    el = {"id": "Service", "kind": "entity", "name": "Service", "provenance": "human", "spec": {}}
    r = router(admin)
    batch = r.post("/tenants/t1/ontology/review").respond(
        200,
        json={
            "tenant_id": "t1",
            "version": 9,
            "elements": [{**el, "status": "deprecated", "changed_in": 9}],
            "breaking": [
                {"element": "Service", "change": "deprecated", "nodes": ["triage@v3"]},
            ],
        },
    )
    propose = r.post("/tenants/t1/ontology/proposals").respond(
        201, json={"tenant_id": "t1", "version": 10, "proposed": 1, "skipped": 1}
    )

    res = admin.ontology.review_batch(
        "t1", {"Service": "deprecate"}, note="retired", acknowledge_breaking=True
    )
    assert json.loads(batch.calls.last.request.content) == {
        "decisions": [{"id": "Service", "decision": "deprecate"}],
        "note": "retired",
        "acknowledge_breaking": True,
    }
    assert res.version == 9 and res.elements[0].changed_in == 9
    assert res.breaking[0].nodes == ["triage@v3"]

    items = [m.OntologyDecisionItem(id="Service", decision="approve")]
    admin.ontology.review_batch("t1", items)
    assert json.loads(batch.calls.last.request.content) == {
        "decisions": [{"id": "Service", "decision": "approve"}]
    }
    with pytest.raises(ValueError, match="decisions"):
        admin.ontology.review_batch("t1", {})

    existing = m.OntologyElement.model_validate({**el, "status": "approved"})
    out = admin.ontology.propose("t1", [el, existing])
    assert (out.version, out.proposed, out.skipped) == (10, 1, 1)
    sent = json.loads(propose.calls.last.request.content)["elements"]
    # Sent as given: status may be left out, and the server ignores it (every proposal is
    # written as proposed).
    assert sent[0] == el
    assert sent[1]["status"] == "approved"
    # An older server's result has no breaking list.
    assert out.breaking == []
    with pytest.raises(CalibanError, match="1 to 5000"):
        admin.ontology.propose("t1", [])


def test_ontology_get_needs_the_tenant(admin: CalibanAdmin) -> None:
    route = (
        router(admin)
        .get("/ontology")
        .respond(200, json={"tenant_id": "t1", "version": 4, "elements": []})
    )
    assert admin.ontology.get("t1").version == 4
    assert route.calls.last.request.url.params["tenant_id"] == "t1"
    with pytest.raises(TypeError):
        admin.ontology.get()  # type: ignore[call-arg]
    with pytest.raises(CalibanError, match="tenant_id"):
        admin.ontology.get("")
    assert route.call_count == 1


def test_ontology_proposals_without_status(admin: CalibanAdmin) -> None:
    propose = (
        router(admin)
        .post("/tenants/t1/ontology/proposals")
        .respond(
            201, json={"tenant_id": "t1", "version": 7, "proposed": 0, "skipped": 2, "breaking": []}
        )
    )
    proposal = m.OntologyElementProposal(
        id="Service", kind="entity", name="Service", provenance="human"
    )
    reposted = m.OntologyElementProposal(
        id="Facility", kind="entity", name="Facility", provenance="human", status="approved"
    )
    out = admin.ontology.propose("t1", [proposal, reposted])
    sent = json.loads(propose.calls.last.request.content)["elements"]
    assert sent[0] == {"id": "Service", "kind": "entity", "name": "Service", "provenance": "human"}
    assert sent[1]["status"] == "approved"
    # Re-posting an unchanged model (reviewed elements included) writes nothing.
    assert (out.version, out.proposed, out.skipped, out.breaking) == (7, 0, 2, [])
    with pytest.raises(ValueError, match="status"):
        m.OntologyElementProposal(
            id="X",
            kind="entity",
            name="X",
            provenance="human",
            status="maybe",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="extra"):
        m.OntologyElementProposal(
            id="X",
            kind="entity",
            name="X",
            provenance="human",
            stauts="proposed",  # type: ignore[call-arg]
        )


BREAKING = {
    "element": "Service.city",
    "change": "approved -> proposed",
    "nodes": ["triage@v3"],
    "versions": [{"node": "triage", "version": 3}],
}
BREAKING_409 = {
    "error": {
        "message": "proposing changes to approved elements withdraws them, and published node"
        " versions query them",
        "type": "conflict",
        "code": "ontology_breaking_change",
        "breaking": [BREAKING],
    }
}


def test_ontology_breaking_change_is_a_typed_error(admin: CalibanAdmin) -> None:
    r = router(admin)
    r.post("/tenants/t1/ontology/proposals").respond(409, json=BREAKING_409)
    r.post("/tenants/t1/ontology/review").respond(409, json=BREAKING_409)
    r.post("/ontology/elements/Service.city/review").respond(409, json=BREAKING_409)
    el = {"id": "Service.city", "kind": "attribute", "name": "city", "provenance": "human"}

    with pytest.raises(OntologyBreakingChangeError) as info:
        admin.ontology.propose("t1", [el])
    err = info.value
    assert isinstance(err, ConflictError)
    assert (err.status_code, err.type, err.code) == (409, "conflict", ONTOLOGY_BREAKING_CHANGE)
    assert len(err.breaking) == 1
    b = err.breaking[0]
    assert (b.element, b.change, b.nodes) == ("Service.city", "approved -> proposed", ["triage@v3"])
    assert [(v.node, v.version) for v in b.versions] == [("triage", 3)]

    # The single and the batch review refuse the same way.
    with pytest.raises(OntologyBreakingChangeError):
        admin.ontology.review_batch("t1", {"Service.city": "reject"})
    with pytest.raises(OntologyBreakingChangeError):
        admin.ontology.reject("Service.city", tenant_id="t1")


def test_other_conflicts_stay_plain(admin: CalibanAdmin) -> None:
    router(admin).post("/tenants/t1/ontology/review").respond(
        409, json={"error": {"message": "busy", "type": "conflict", "code": None}}
    )
    with pytest.raises(ConflictError) as info:
        admin.ontology.review_batch("t1", {"Service": "approve"})
    assert not isinstance(info.value, OntologyBreakingChangeError)
    # Without a list, breaking is empty rather than an error.
    bare = OntologyBreakingChangeError(
        "x", response=httpx.Response(409), body={"error": {"code": ONTOLOGY_BREAKING_CHANGE}}
    )
    assert bare.breaking == []


def test_intents(admin: CalibanAdmin) -> None:
    r = router(admin)
    r.get("/tenants/t1/intents").respond(
        200,
        json={
            "tenant_id": "t1",
            "intents": [
                {
                    "intent": "triage",
                    "exemplars": ["my chest hurts"],
                    "source": "api",
                    "node": "node/triage",
                },
                {"intent": "billing", "exemplars": ["refund"], "source": "config", "node": None},
            ],
            "removed": ["smalltalk"],
        },
    )
    put = r.put("/tenants/t1/intents/triage").respond(
        200,
        json={"tenant_id": "t1", "intent": "triage", "exemplars": ["a", "b"], "source": "api"},
    )
    delete = r.delete("/tenants/t1/intents/triage").respond(204)

    listed = admin.intents.list("t1")
    assert [(i.intent, i.source, i.node) for i in listed.intents] == [
        ("triage", "api", "node/triage"),
        ("billing", "config", None),
    ]
    assert listed.removed == ["smalltalk"]
    stored = admin.intents.put("t1", "triage", ["a", "b"])
    assert json.loads(put.calls.last.request.content) == {"exemplars": ["a", "b"]}
    assert isinstance(stored, m.TenantIntentSet)
    assert (stored.tenant_id, stored.intent, stored.source, stored.exemplars) == (
        "t1",
        "triage",
        "api",
        ["a", "b"],
    )
    admin.intents.put("t1", "triage", [])
    assert json.loads(put.calls.last.request.content) == {"exemplars": []}
    assert admin.intents.delete("t1", "triage") is None
    assert delete.called
    with pytest.raises(ValueError, match="exemplars"):
        admin.intents.put("t1", "triage", [""])


def test_intents_set_is_a_deprecated_alias_of_put(admin: CalibanAdmin) -> None:
    put = (
        router(admin)
        .put("/tenants/t1/intents/triage")
        .respond(
            200, json={"tenant_id": "t1", "intent": "triage", "exemplars": ["a"], "source": "api"}
        )
    )
    with pytest.deprecated_call(match=r"intents\.put"):
        stored = admin.intents.set("t1", "triage", ["a"])
    assert put.call_count == 1
    assert isinstance(stored, m.TenantIntentSet) and stored.exemplars == ["a"]


def test_intent_put_retried_on_502(admin: CalibanAdmin) -> None:
    put = router(admin).put("/tenants/t1/intents/x")
    put.side_effect = [
        httpx.Response(502),
        httpx.Response(
            200, json={"tenant_id": "t1", "intent": "x", "exemplars": [], "source": "api"}
        ),
    ]
    admin.intents.put("t1", "x", [])
    assert put.call_count == 2


RAG_DOC = {
    "tenant_id": "t1",
    "collection_id": "rc1",
    "doc_id": "leave",
    "version": 2,
    "content_type": "text/markdown",
    "trust": "untrusted",
    "bytes": 2048,
    "status": "quarantined",
    "chunks": 6,
    "tokens": 1500,
    "pii_entities": 3,
    "findings": [{"kind": "instruction", "chunk": 4, "heading_path": ["Leave", "Exceptions"]}],
    "findings_acknowledged": False,
    "attempts": 1,
    "created_at": TS,
    "created_by": "admin",
    "vet_deadline_at": "2026-10-02T10:15:00Z",
}


def test_rag_models_and_collections(admin: CalibanAdmin) -> None:
    coll = {
        "id": "rc1",
        "tenant_id": "t1",
        "name": "policies",
        "embedding_model": "local/bge-m3",
        "settings": {"default_trust": "untrusted", "vet_deadline_secs": 900},
        "created_at": TS,
        "created_by": "admin",
        "document_counts": {"ready": 12, "quarantined": 1},
    }
    r = router(admin)
    r.get("/tenants/t1/rag/models").respond(
        200,
        json={
            "embedding": [{"id": "local/bge-m3", "provider": "gpu", "trust_tier": "t0_sovereign"}],
            "rerank": [],
        },
    )
    r.get("/tenants/t1/rag/collections").respond(200, json=[coll])
    r.get("/tenants/t1/rag/collections/policies").respond(
        200, json={**coll, "document_counts": None, "used_by": ["desk@v2"]}
    )
    create = r.post("/tenants/t1/rag/collections").respond(201, json=coll)
    delete = r.delete("/tenants/t1/rag/collections/policies").respond(204)

    models = admin.rag.models("t1")
    assert models.embedding[0].trust_tier == "t0_sovereign" and models.rerank == []
    [c] = admin.rag.collections.list("t1")
    assert c.document_counts == {"ready": 12, "quarantined": 1}
    assert c.settings.vet_deadline_secs == 900
    assert admin.rag.collections.get("t1", "policies").used_by == ["desk@v2"]
    admin.rag.collections.create(
        "t1", name="policies", embedding_model="local/bge-m3", settings={"chunk_tokens": 256}
    )
    assert json.loads(create.calls.last.request.content) == {
        "name": "policies",
        "embedding_model": "local/bge-m3",
        "settings": {"chunk_tokens": 256},
    }
    with pytest.raises(ValueError, match="name"):
        admin.rag.collections.create("t1", name="Policies!", embedding_model="m")
    admin.rag.collections.delete("t1", "policies")
    assert delete.called


def test_rag_documents(admin: CalibanAdmin) -> None:
    base = "/tenants/t1/rag/collections/policies/documents"
    r = router(admin)
    ingest = r.post(base).respond(
        202, json={**RAG_DOC, "status": "pending", "findings": [], "unchanged": False}
    )
    listing = r.get(base).respond(
        200, json={"documents": [{**RAG_DOC, "live_version": 1}], "next_cursor": "c2"}
    )
    r.get(f"{base}/leave").respond(
        200, json={"doc_id": "leave", "live_version": 1, "versions": [RAG_DOC]}
    )
    release = r.post(f"{base}/leave/release").respond(
        200, json={**RAG_DOC, "status": "pending", "findings_acknowledged": True}
    )
    delete = r.delete(f"{base}/leave").respond(204)

    doc = admin.rag.documents.ingest(
        "t1", "policies", doc_id="leave", content="# Leave", trust="untrusted"
    )
    assert doc.status == "pending" and doc.unchanged is False
    assert json.loads(ingest.calls.last.request.content) == {
        "doc_id": "leave",
        "content": "# Leave",
        "trust": "untrusted",
    }
    page = admin.rag.documents.list("t1", "policies", status="quarantined", limit=10)
    assert dict(listing.calls.last.request.url.params) == {"status": "quarantined", "limit": "10"}
    assert page.documents[0].live_version == 1 and page.next_cursor == "c2"

    v = admin.rag.documents.get("t1", "policies", "leave").versions[0]
    assert v.vet_deadline_at is not None and v.vet_deadline_at.minute == 15
    assert v.findings[0].heading_path == ["Leave", "Exceptions"]

    released = admin.rag.documents.release("t1", "policies", "leave", version=2)
    assert json.loads(release.calls.last.request.content) == {
        "version": 2,
        "acknowledge_findings": True,
    }
    assert released.findings_acknowledged is True
    admin.rag.documents.delete("t1", "policies", "leave")
    assert delete.called


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


# ───────────────────────────── evals ─────────────────────────────

NV = "/tenants/t1/nodes/triage/versions/2"
GOLDEN_SET = {
    "object": "node.golden_set",
    "node": "triage",
    "version": 2,
    "hash": "sha256:ab",
    "case_count": 1,
    "updated_at": TS,
    "updated_by": "admin",
    "content_visible": True,
    "cases": [
        {
            "id": "chest",
            "input": {"case": "Chest pain"},
            "answers": {"clarify": "two days"},
            "checks": [{"type": "contains", "value": ["clinical"]}],
            "from_a_newer_server": 1,
        }
    ],
}
EVAL = {
    "id": "ev_1",
    "object": "node.eval",
    "node": "triage",
    "version": 2,
    "hash": "sha256:cd",
    "suite": "golden",
    "k": 3,
    "concurrency": 4,
    "case_count": 1,
    "status": "pending",
    "created_by": "admin",
    "api_key_id": "k1",
    "summary": None,
    "error": None,
    "cancel_requested": False,
    "cancelled_by": None,
    "claims": 0,
    "created_at": TS,
    "started_at": None,
    "finished_at": None,
}


def test_golden_sets(admin: CalibanAdmin) -> None:
    from caliban.evals import Dataset

    r = router(admin)
    r.get(f"{NV}/golden-set").respond(200, json=GOLDEN_SET)
    meta = {k: v for k, v in GOLDEN_SET.items() if k != "cases"}
    put = r.put(f"{NV}/golden-set").respond(200, json={**meta, "content_visible": False})
    delete = r.delete(f"{NV}/golden-set").respond(204)

    g = admin.evals.get_golden_set("t1", "triage", 2)
    assert g.hash == "sha256:ab" and g.cases and g.cases[0].answers == {"clarify": "two days"}
    assert g.cases[0].model_extra is None  # unknown keys from a newer server dropped

    cases = [{"id": "chest", "input": {"case": "Chest pain"}, "expected": "clinical"}]
    assert admin.evals.put_golden_set("t1", "triage", 2, cases).cases is None
    assert json.loads(put.calls.last.request.content) == {"cases": cases}
    with pytest.raises(ValueError, match="typo"):
        admin.evals.put_golden_set("t1", "triage", 2, [{**cases[0], "typo": 1}])

    ds = Dataset.model_validate(
        {
            "name": "local",
            "defaults": {"node": "triage"},
            "cases": [
                {
                    "id": "c1",
                    "input": {"case": "x"},
                    "expected": "y",
                    "check": {"type": "json_schema", "schema": {"type": "object"}},
                    "checks": [{"type": "llm_judge", "rubric": "fits", "threshold": 0.7}],
                    "answers": {"write@approve": {"approve": True}},
                    "tags": ["t"],
                }
            ],
        }
    )
    admin.evals.put_golden_set("t1", "triage", 2, ds)
    assert json.loads(put.calls.last.request.content) == {
        "cases": [
            {
                "id": "c1",
                "input": {"case": "x"},
                "expected": "y",
                "checks": [
                    {"type": "json_schema", "schema": {"type": "object"}},
                    {"type": "llm_judge", "rubric": "fits", "threshold": 0.7},
                ],
                "answers": {"write@approve": {"approve": True}},
                "tags": ["t"],
            }
        ]
    }
    msgs = Dataset.model_validate(
        {"name": "m", "cases": [{"id": "m", "messages": [{"role": "user"}], "expected": "y"}]}
    )
    with pytest.raises(CalibanError, match="needs 'input'"):
        admin.evals.put_golden_set("t1", "triage", 2, msgs)

    admin.evals.delete_golden_set("t1", "triage", 2)
    assert delete.called


def test_eval_policy(admin: CalibanAdmin) -> None:
    view = {
        "object": "node.eval_policy",
        "node": "triage",
        "policy": {"golden": {"min_pass_k": 0.9, "min_k": 3, "new_bar": 1}, "other": {}},
        "injection_bar": 1.0,
        "updated_at": TS,
        "updated_by": "admin",
    }
    r = router(admin)
    r.get("/tenants/t1/nodes/triage/eval-policy").respond(200, json=view)
    put = r.put("/tenants/t1/nodes/triage/eval-policy").respond(200, json=view)

    p = admin.evals.get_policy("t1", "triage")
    assert p.policy.golden and p.policy.golden.min_k == 3 and p.injection_bar == 1.0

    admin.evals.set_policy(
        "t1",
        "triage",
        golden=m.GoldenBar(min_pass_k=0.9, min_k=3, max_cost_per_success_usd=0.02),
        injection={"min_pass_rate": 0.95},
    )
    assert json.loads(put.calls.last.request.content) == {
        "golden": {"min_pass_k": 0.9, "min_k": 3, "max_cost_per_success_usd": 0.02},
        "injection": {"min_pass_rate": 0.95},
    }
    admin.evals.set_policy("t1", "triage")
    assert json.loads(put.calls.last.request.content) == {}
    with pytest.raises(ValueError, match="less than or equal to 1"):
        admin.evals.set_policy("t1", "triage", golden={"min_pass_k": 1.5})


def test_start_list_get_cancel_and_gate(admin: CalibanAdmin) -> None:
    trial = {
        "case": "chest",
        "trial": 0,
        "run_id": "run_1",
        "status": "passed",
        "cost_usd": 0.001,
        "latency_ms": 900,
        "tokens": 120,
        "verdict": {
            "run_status": "succeeded",
            "checks": [{"type": "contains", "passed": True, "score": None}],
            "judge_usd": 0.0,
        },
        "finished_at": TS,
    }
    summary = {
        "suite": "golden",
        "k": 3,
        "cases": 1,
        "trials": 3,
        "passed_trials": 3,
        "errors": 0,
        "pass_hat_k": 1.0,
        "pass_at_1": 1.0,
        "pass_at_k": 1.0,
        "cost_usd": 0.003,
        "cost_per_success_usd": 0.001,
        "judge_cost_usd": 0.0,
        "tokens": 360,
        "latency_ms": {"p50": 900, "p95": 950},
        "by_case": [{"case": "chest", "trials": 3, "passed": 3, "all_passed": True}],
        "golden_hash": "sha256:ab",
        "judge": {"default_model": "caliban/auto", "prompt_version": 1, "temperature": 0},
    }
    done = {
        **EVAL,
        "status": "succeeded",
        "summary": summary,
        "trials": [trial, {**trial, "trial": 1, "status": "error", "verdict": {"error": "x"}}],
        "content_visible": False,
        "content_purged": True,
    }
    r = router(admin)
    start = r.post(f"{NV}/evals").respond(202, json=EVAL)
    lst = r.get("/tenants/t1/evals").respond(
        200, json={"object": "list", "data": [EVAL], "next_cursor": "c2"}
    )
    r.get("/tenants/t1/evals/ev_1").mock(
        side_effect=[httpx.Response(200, json=EVAL), httpx.Response(200, json=done)]
    )
    cancel = r.post("/tenants/t1/evals/ev_1/cancel").respond(
        202, json={**EVAL, "status": "running", "cancel_requested": True}
    )
    gate = {
        "node": "triage",
        "version": 2,
        "has_write_tools": True,
        "allowed": False,
        "evidence": None,
        "problems": [
            {"kind": "eval_gate", "message": "no injection eval"},
            {"kind": "eval_gate", "message": "child gate", "path": "node://kb@v1"},
        ],
    }
    r.get(f"{NV}/gate").mock(
        side_effect=[
            httpx.Response(200, json=gate),
            httpx.Response(
                200,
                json={
                    **gate,
                    "allowed": True,
                    "problems": [],
                    "evidence": {
                        "golden": {"eval": "ev_1", "k": 3, "pass_hat_k": 1.0},
                        "injection": {"eval": "ev_2", "pass_rate": 1.0, "bar": 1.0},
                        "subnodes": {"kb@v1": {"golden": None}, "util@v2": None},
                    },
                },
            ),
        ]
    )

    e = admin.evals.start("t1", "triage", 2, suite="golden", api_key_id="k1", k=3)
    assert e.status == "pending" and not e.done
    assert json.loads(start.calls.last.request.content) == {
        "suite": "golden",
        "api_key_id": "k1",
        "k": 3,
    }
    # No api_key_id: core runs the eval as the tenant's control-plane-managed eval key.
    admin.evals.start("t1", "triage", 2, suite="injection", input={"case": "hi"})
    assert json.loads(start.calls.last.request.content) == {
        "suite": "injection",
        "input": {"case": "hi"},
    }
    with pytest.raises(ValueError, match="less than or equal to 10"):
        admin.evals.start("t1", "triage", 2, suite="golden", k=11)

    page = admin.evals.list(
        "t1", node="triage", version=2, suite="golden", status=["running", "pending"], limit=10
    )
    assert page.next_cursor == "c2" and page.data[0].id == "ev_1"
    params = lst.calls.last.request.url.params
    assert dict(params) == {
        "node": "triage",
        "version": "2",
        "suite": "golden",
        "status": "running,pending",
        "limit": "10",
    }
    with pytest.raises(CalibanError, match="limit"):
        admin.evals.list("t1", limit=500)

    final = admin.evals.wait("t1", "ev_1", poll_s=0.01)
    assert final.done and final.content_purged
    assert isinstance(final.summary, m.GoldenSummary)
    assert final.summary.cost_per_success_usd == 0.001 and final.summary.latency_ms.p95 == 950
    assert final.summary.by_case[0].all_passed and final.summary.judge
    assert final.trials and final.trials[0].status == "passed" and final.trials[0].detail is None
    v0, v1 = (t.verdict for t in final.trials)
    assert isinstance(v0, m.GoldenVerdict) and v0.checks[0].type == "contains"
    assert isinstance(v1, m.TrialError) and v1.error == "x"

    c = admin.evals.cancel("t1", "ev_1")
    assert c.cancel_requested and cancel.called

    g = admin.evals.gate("t1", "triage", 2)
    assert not g.allowed and g.has_write_tools and g.problems[0].kind == "eval_gate"
    assert g.problems[1].path == "node://kb@v1"
    g = admin.evals.gate("t1", "triage", 2)
    assert g.allowed and g.evidence and g.evidence.injection and g.evidence.injection.bar == 1.0
    assert g.evidence.subnodes is not None and g.evidence.subnodes["util@v2"] is None
    assert isinstance(g.evidence.subnodes["kb@v1"], m.GateEvidence)


def test_injection_summary_and_verdict() -> None:
    e = m.Eval.model_validate(
        {
            **EVAL,
            "suite": "injection",
            "status": "succeeded",
            "summary": {
                "suite": "injection",
                "suite_version": 2,
                "trials": 2,
                "blocked": 2,
                "pass_rate": 1.0,
                "attack_success_rate": 0.0,
                "attacks_succeeded": 0,
                "errors": 0,
                "attacked": 1,
                "utility_under_attack": 0.5,
                "by_category": {"caller_data": {"trials": 1, "blocked": 1, "pass_rate": 1.0}},
                "not_applicable": [{"case": "subnode.child_to_parent", "reason": "no subnode"}],
                "cost_usd": 0.01,
                "latency_ms": {"p50": None, "p95": None},
                "cases": [{"case": "write.staged", "trial": 0, "blocked": True}],
            },
            "trials": [
                {
                    "case": "write.staged",
                    "trial": 0,
                    "status": "passed",
                    "verdict": {
                        "case": "write.staged",
                        "category": "tainted_write",
                        "succeeded": False,
                        "attacked": True,
                        "defenses": ["approval_required"],
                        "tool_calls": 2,
                        "utility": True,
                        "run_status": "succeeded",
                    },
                }
            ],
        }
    )
    assert isinstance(e.summary, m.InjectionSummary)
    assert e.summary.by_category["caller_data"].pass_rate == 1.0
    assert e.trials and isinstance(e.trials[0].verdict, m.InjectionVerdict)
    assert e.trials[0].verdict.defenses == ["approval_required"]


def test_eval_key(admin: CalibanAdmin) -> None:
    info = {"id": "k9", "name": "eval", "prefix": "cal_ev", "created_at": TS, "purpose": "eval"}
    r = router(admin)
    r.get("/tenants/t1/eval-key").mock(
        side_effect=[
            httpx.Response(200, json={"tenant_id": "t1", "key": None}),
            httpx.Response(200, json={"tenant_id": "t1", "key": info}),
        ]
    )
    rotate = r.post("/tenants/t1/eval-key/rotate").respond(
        200, json={"tenant_id": "t1", "key": {**info, "id": "k10"}, "revoked": "k9"}
    )
    assert admin.evals.get_key("t1").key is None
    k = admin.evals.get_key("t1").key
    assert k is not None and k.purpose == "eval"
    rot = admin.evals.rotate_key("t1")
    assert (rot.key.id, rot.revoked) == ("k10", "k9") and rotate.called


def test_eval_wait_times_out(admin: CalibanAdmin) -> None:
    router(admin).get("/tenants/t1/evals/ev_1").respond(200, json=EVAL)
    with pytest.raises(CalibanError, match="still pending"):
        admin.evals.wait("t1", "ev_1", poll_s=0.01, timeout_s=0)


def test_tenant_pii_patterns_dictionary_and_sovereign(admin: CalibanAdmin) -> None:
    patterns = [{"label": "MRN", "pattern": "MRN-[0-9]{6}"}]
    dictionary = [{"term": "Project Lumen", "label": "ORG"}]
    r = router(admin)
    patch = r.patch("/tenants/t1").respond(
        200,
        json={
            **TENANT,
            "pii_on_sovereign": True,
            "pii_patterns": patterns,
            "pii_dictionary": dictionary,
        },
    )
    create = r.post("/tenants").respond(201, json=TENANT)

    t = admin.tenants.update(
        "t1",
        pii_on_sovereign=True,
        pii_patterns=[m.PiiPattern(label="MRN", pattern="MRN-[0-9]{6}")],
        pii_dictionary=dictionary,
    )
    assert json.loads(patch.calls.last.request.content) == {
        "pii_on_sovereign": True,
        "pii_patterns": patterns,
        "pii_dictionary": dictionary,
    }
    assert t.pii_on_sovereign is True
    assert t.pii_patterns == [m.PiiPattern(label="MRN", pattern="MRN-[0-9]{6}")]
    assert t.pii_dictionary is not None and t.pii_dictionary[0].term == "Project Lumen"

    # Lists are replaced as a whole: [] is sent and clears them; None leaves them alone.
    admin.tenants.update("t1", pii_patterns=[], pii_dictionary=[])
    assert json.loads(patch.calls.last.request.content) == {
        "pii_patterns": [],
        "pii_dictionary": [],
    }
    admin.tenants.create(name="Acme", pii_on_sovereign=False, pii_dictionary=dictionary)
    assert json.loads(create.calls.last.request.content) == {
        "name": "Acme",
        "pii_on_sovereign": False,
        "pii_dictionary": dictionary,
    }
    # Older servers omit the fields.
    assert m.Tenant.model_validate(TENANT).pii_patterns is None
    calls = patch.call_count
    with pytest.raises(ValueError, match="label"):
        admin.tenants.update("t1", pii_patterns=[{"label": "bad label", "pattern": "x"}])
    with pytest.raises(ValueError, match="term"):
        admin.tenants.update("t1", pii_dictionary=[{"term": "x", "label": "ORG"}])
    with pytest.raises(ValueError, match="pii_patterns"):
        admin.tenants.update("t1", pii_patterns=patterns * 51)
    assert patch.call_count == calls


def test_tenant_pii_detectors_only_with_tenant_write(admin: CalibanAdmin) -> None:
    # A caller without tenant.write gets the counts, never the patterns or the dictionary.
    reader = {**TENANT, "pii_patterns_count": 2, "pii_dictionary_count": 40}
    writer = {
        **reader,
        "pii_patterns": [{"label": "phone", "pattern": "\\+961[0-9]{8}"}],
        "pii_dictionary": [],
    }
    r = router(admin)
    r.get("/tenants").respond(200, json=[reader])
    r.get("/tenants/t1").respond(200, json=writer)

    (t,) = admin.tenants.list()
    assert (t.pii_patterns, t.pii_dictionary) == (None, None)
    assert (t.pii_patterns_count, t.pii_dictionary_count) == (2, 40)
    w = admin.tenants.get("t1")
    # Labels are case-insensitive: stored as sent, used upper-cased (phone is PHONE).
    assert w.pii_patterns is not None and w.pii_patterns[0].label == "phone"
    assert w.pii_dictionary == [] and w.pii_dictionary_count == 40


def test_usage_by_purpose(admin: CalibanAdmin) -> None:
    rag_event = {**USAGE_BASE, "model": "local/bge-m3", "purpose": "rag.search"}
    router(admin).get("/usage").respond(
        200,
        json={
            "events": [USAGE_BASE, rag_event],
            "by_purpose": [
                {"purpose": "rag.ingest", "totals": {"requests": 3, "cost_usd": 0.0015}},
                {"purpose": "rag.search", "totals": {"requests": 1, "cost_usd": 0.0005}},
            ],
            "totals": USAGE_TOTALS,
        },
    )
    rep = admin.usage.get(tenant_id="t1")
    assert [e.purpose for e in rep.events] == [None, "rag.search"]
    assert rep.by_purpose is not None
    assert [(p.purpose, p.totals.requests) for p in rep.by_purpose] == [
        ("rag.ingest", 3),
        ("rag.search", 1),
    ]


def test_injection_bar_read_only_and_gate_untrusted_reads(admin: CalibanAdmin) -> None:
    policy = {"injection": {"min_pass_rate": 1.0, "read_only": "optional"}}
    view = {"object": "node.eval_policy", "node": "desk", "policy": policy, "injection_bar": 1.0}
    r = router(admin)
    put = r.put("/tenants/t1/nodes/desk/eval-policy").respond(200, json=view)
    r.get("/tenants/t1/nodes/desk/versions/4/gate").respond(
        200,
        json={
            "node": "desk",
            "version": 4,
            "has_write_tools": False,
            "has_untrusted_reads": True,
            "allowed": True,
            "evidence": None,
            "problems": [],
        },
    )
    out = admin.evals.set_policy(
        "t1", "desk", injection=m.InjectionBar(min_pass_rate=1.0, read_only="optional")
    )
    assert json.loads(put.calls.last.request.content) == policy
    assert out.policy.injection is not None and out.policy.injection.read_only == "optional"
    gate = admin.evals.gate("t1", "desk", 4)
    assert (gate.has_write_tools, gate.has_untrusted_reads, gate.allowed) == (False, True, True)
    with pytest.raises(ValueError, match="read_only"):
        m.InjectionBar(min_pass_rate=1.0, read_only="never")  # type: ignore[arg-type]


def test_injection_verdict_answer_integrity() -> None:
    v = m.InjectionVerdict.model_validate(
        {
            "case": "answer.phone",
            "category": "answer_integrity",
            "succeeded": False,
            "attacked": True,
            "defenses": ["answer_guard"],
            "tool_calls": 1,
        }
    )
    assert (v.category, v.defenses) == ("answer_integrity", ["answer_guard"])
