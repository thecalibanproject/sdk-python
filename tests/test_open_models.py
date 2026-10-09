"""Open models on-prem: embeddings, reasoning, cost header, model catalogue and providers."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import caliban
from caliban import CalibanAdmin, CalibanError, CalibanOptions, ConflictError, UpstreamError
from caliban.admin import models as m
from caliban.admin.client import model_id_path
from caliban.evals import Pricing, load_dataset, run_eval

from .conftest import completion_body, make_async_client, make_client, sse_body

EMBEDDINGS = {
    "object": "list",
    "model": "local/bge-m3",
    "data": [
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    ],
    "usage": {"prompt_tokens": 4, "total_tokens": 4},
}
EMB_HEADERS = {
    "x-caliban-request-id": "req-e",
    "x-caliban-routed-model": "local/bge-m3",
    "x-caliban-pii-entities": "2",
}


# ───────────────────────────── embeddings ─────────────────────────────


def test_embeddings_batch_body_headers_and_vectors() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EMBEDDINGS, headers=EMB_HEADERS)

    res = make_client(handler).embeddings.create(
        model="local/bge-m3", input=("a", "b"), dimensions=2, extra_body={"truncate": 512}
    )
    req = seen[0]
    assert req.url == "http://caliban.test/v1/embeddings"
    assert req.method == "POST"
    assert req.headers["authorization"] == "Bearer cal_test"
    assert json.loads(req.content) == {
        "model": "local/bge-m3",
        "input": ["a", "b"],
        "dimensions": 2,
        "truncate": 512,
    }
    assert res.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert res.usage is not None and res.usage.prompt_tokens == 4
    assert res.caliban.request_id == "req-e"
    assert res.caliban.routed_model == "local/bge-m3"
    assert res.caliban.pii_entities == 2


def test_embeddings_single_string_and_bad_input() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=EMBEDDINGS)

    client = make_client(handler)
    client.embeddings.create(model="e", input="one")
    assert json.loads(calls[0].content) == {"model": "e", "input": "one"}
    with pytest.raises(CalibanError, match="string"):
        client.embeddings.create(model="e", input=[1, 2])  # type: ignore[list-item]
    assert len(calls) == 1


def test_embeddings_async_and_errors(no_sleep: list[float]) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        if json.loads(request.content)["model"] == "chat-model":
            return httpx.Response(
                400, json={"error": {"message": "not an embedding model", "type": "x"}}
            )
        return httpx.Response(200, json=EMBEDDINGS, headers=EMB_HEADERS)

    async def run() -> caliban.CreateEmbeddingResponse:
        async with make_async_client(handler) as client:
            res = await client.embeddings.create(model="local/bge-m3", input=["a", "b"])
            with pytest.raises(caliban.BadRequestError):
                await client.embeddings.create(model="chat-model", input="x")
            return res

    res = asyncio.run(run())
    assert res.vectors[0] == [0.1, 0.2]
    assert res.caliban.pii_entities == 2
    assert calls[0].url.path == "/v1/embeddings"
    assert len(no_sleep) == 1  # 503 retried once


# ───────────────────────────── rerank ─────────────────────────────

RERANK_DOCS = ["Paris is in France.", "Bananas are yellow.", "The Eiffel Tower is in Paris."]
RERANK = {
    "model": "local/qwen3-reranker",
    # Sorted by relevance_score desc, so indices are not in input order.
    "results": [
        {"index": 2, "relevance_score": 0.97},
        {"index": 0, "relevance_score": 0.81},
        {"index": 1, "relevance_score": 0.02},
    ],
    "usage": {"total_tokens": 42},
}
RERANK_HEADERS = {
    "x-caliban-request-id": "req-r",
    "x-caliban-routed-model": "local/qwen3-reranker",
}


def test_rerank_body_order_and_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RERANK, headers=RERANK_HEADERS)

    res = make_client(handler).rerank.create(
        model="local/qwen3-reranker",
        query="Where is the Eiffel Tower?",
        documents=tuple(RERANK_DOCS),
    )
    req = seen[0]
    assert req.url == "http://caliban.test/v1/rerank"
    assert req.method == "POST"
    assert req.headers["authorization"] == "Bearer cal_test"
    assert json.loads(req.content) == {
        "model": "local/qwen3-reranker",
        "query": "Where is the Eiffel Tower?",
        "documents": RERANK_DOCS,
    }
    assert [r.index for r in res.results] == [2, 0, 1]
    assert [r.relevance_score for r in res.results] == [0.97, 0.81, 0.02]
    assert res.results[0].document is None
    assert res.usage is not None and res.usage.total_tokens == 42
    assert res.caliban.request_id == "req-r"
    assert res.caliban.routed_model == "local/qwen3-reranker"


def test_rerank_top_n_and_return_documents() -> None:
    seen: list[dict[str, Any]] = []
    body = {
        "model": "local/bge-reranker",
        "results": [
            {"index": 2, "relevance_score": 0.97, "document": {"text": RERANK_DOCS[2]}},
            {"index": 0, "relevance_score": 0.81, "document": {"text": RERANK_DOCS[0]}},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    res = make_client(handler).rerank.create(
        model="local/bge-reranker",
        query="Eiffel",
        documents=RERANK_DOCS,
        top_n=2,
        return_documents=True,
        extra_body={"truncate": True},
    )
    assert seen[0] == {
        "model": "local/bge-reranker",
        "query": "Eiffel",
        "documents": RERANK_DOCS,
        "top_n": 2,
        "return_documents": True,
        "truncate": True,
    }
    assert len(res.results) == 2
    assert [r.document.text for r in res.results if r.document] == [
        RERANK_DOCS[2],
        RERANK_DOCS[0],
    ]
    assert res.usage is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"documents": []}, "at least one"),
        ({"documents": ["a", 1]}, "sequence of strings"),
        ({"documents": "a"}, "sequence of strings"),
        ({"query": None}, "query"),
        ({"top_n": 0}, "top_n"),
        ({"top_n": True}, "top_n"),
        ({"return_documents": "yes"}, "return_documents"),
    ],
)
def test_rerank_rejects_invalid_input_before_sending(kwargs: dict[str, Any], match: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=RERANK)

    params: dict[str, Any] = {"model": "r", "query": "q", "documents": ["a"], **kwargs}
    with pytest.raises(CalibanError, match=match):
        make_client(handler).rerank.create(**params)

    async def run() -> None:
        async with make_async_client(handler) as client:
            with pytest.raises(CalibanError, match=match):
                await client.rerank.create(**params)

    asyncio.run(run())
    assert calls == []


def test_rerank_async_retries_and_errors(no_sleep: list[float]) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        if json.loads(request.content)["model"] == "chat-model":
            return httpx.Response(
                400, json={"error": {"message": "not a rerank model", "type": "x"}}
            )
        return httpx.Response(200, json=RERANK, headers=RERANK_HEADERS)

    async def run() -> caliban.RerankResponse:
        async with make_async_client(handler) as client:
            res = await client.rerank.create(model="r", query="q", documents=RERANK_DOCS)
            with pytest.raises(caliban.BadRequestError):
                await client.rerank.create(model="chat-model", query="q", documents=["a"])
            return res

    res = asyncio.run(run())
    assert [r.index for r in res.results] == [2, 0, 1]
    assert res.caliban.request_id == "req-r"
    assert calls[0].url.path == "/v1/rerank"
    assert len(calls) == 3
    assert len(no_sleep) == 1  # 503 retried once


def test_rerank_malformed_response_raises_validation_error() -> None:
    client = make_client(lambda r: httpx.Response(200, json={"model": "r", "results": [{}]}))
    with pytest.raises(caliban.APIResponseValidationError):
        client.rerank.create(model="r", query="q", documents=["a"])


# ───────────────────────────── reasoning ─────────────────────────────


def test_reasoning_option_is_sent_and_validated() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_body())

    client = make_client(handler)
    client.chat.completions.create(
        model="local/qwen3-8b", messages=[], caliban={"reasoning": "off"}
    )
    client.chat.completions.create(
        model="local/qwen3-8b", messages=[], caliban=CalibanOptions(reasoning="high", pii="mask")
    )
    assert seen[0]["caliban"] == {"reasoning": "off"}
    assert seen[1]["caliban"] == {"pii": "mask", "reasoning": "high"}
    with pytest.raises(ValueError, match="reasoning"):
        client.chat.completions.create(model="m", messages=[], caliban={"reasoning": "max"})
    assert len(seen) == 2


def test_reasoning_content_on_completion() -> None:
    body = completion_body("4")
    body["choices"][0]["message"]["reasoning_content"] = "2+2=4"
    resp = make_client(lambda r: httpx.Response(200, json=body)).chat.completions.create(
        model="m", messages=[]
    )
    assert resp.text == "4"
    assert resp.reasoning_text == "2+2=4"
    assert resp.choices[0].message.reasoning_content == "2+2=4"

    alt = completion_body("ok")
    alt["choices"][0]["message"]["reasoning"] = "hmm"
    resp2 = make_client(lambda r: httpx.Response(200, json=alt)).chat.completions.create(
        model="m", messages=[]
    )
    assert resp2.reasoning_text == "hmm"
    plain = make_client(lambda r: httpx.Response(200, json=completion_body()))
    assert plain.chat.completions.create(model="m", messages=[]).reasoning_text is None


def _delta_chunk(delta: dict[str, Any], index: int = 0) -> str:
    return json.dumps(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "model": "m",
            "choices": [{"index": index, "delta": delta, "finish_reason": None}],
        }
    )


REASONING_STREAM = sse_body(
    _delta_chunk({"role": "assistant"}),
    _delta_chunk({"reasoning_content": "Let me "}),
    _delta_chunk({"reasoning_content": "think."}),
    _delta_chunk({"content": "other"}, index=1),
    _delta_chunk({"content": "Answer"}),
    _delta_chunk({"content": " 42"}),
)


def test_stream_separates_reasoning_from_content() -> None:
    client = make_client(lambda r: httpx.Response(200, content=REASONING_STREAM))
    with client.chat.completions.create(model="m", messages=[], stream=True) as stream:
        parts = list(stream.iter_parts())
    assert parts == [
        caliban.TextPart("reasoning", "Let me "),
        caliban.TextPart("reasoning", "think."),
        caliban.TextPart("content", "Answer"),
        caliban.TextPart("content", " 42"),
    ]
    collected = client.chat.completions.create(model="m", messages=[], stream=True).collect()
    assert collected == caliban.StreamText(reasoning="Let me think.", content="Answer 42")
    stream = client.chat.completions.create(model="m", messages=[], stream=True)
    assert "".join(stream.iter_text()) == "Answer 42"


def test_stream_reasoning_alias_and_async() -> None:
    body = sse_body(_delta_chunk({"reasoning": "hmm"}), _delta_chunk({"content": "ok"}))

    async def run() -> caliban.StreamText:
        client = make_async_client(lambda r: httpx.Response(200, content=body))
        stream = await client.chat.completions.create(model="m", messages=[], stream=True)
        out = await stream.collect()
        await client.aclose()
        return out

    assert asyncio.run(run()) == caliban.StreamText("hmm", "ok")


# ───────────────────────────── cost header ─────────────────────────────


@pytest.mark.parametrize(
    ("header", "expected"),
    [("0.00012500", 0.000125), ("1e-6", 1e-6), (None, None), ("n/a", None), ("nan", None)],
)
def test_cost_usd_header(header: str | None, expected: float | None) -> None:
    headers = {"x-caliban-cost-usd": header} if header is not None else {}
    client = make_client(lambda r: httpx.Response(200, json=completion_body(), headers=headers))
    cost = client.chat.completions.create(model="m", messages=[]).caliban.cost_usd
    if expected is None:
        assert cost is None
    else:
        assert cost is not None and math.isclose(cost, expected)


def test_eval_cost_prefers_header_over_pricing() -> None:
    golden = Path(__file__).parent / "fixtures" / "golden.yaml"
    ds = load_dataset(golden)
    client = make_client(
        lambda r: httpx.Response(
            200, json=completion_body("x"), headers={"x-caliban-cost-usd": "0.25000000"}
        )
    )
    report = run_eval(ds, client, k=1, pricing=Pricing(in_per_mtok=1e6, out_per_mtok=1e6))
    costs = [t.cost_usd for c in report.cases for t in c.trials]
    assert costs and all(c == 0.25 for c in costs)
    assert report.summary.cost_total_usd is not None
    assert math.isclose(report.summary.cost_total_usd, 0.25 * len(costs))


# ───────────────────────────── /v1/models ─────────────────────────────


def test_models_list_caliban_info() -> None:
    data = {
        "object": "list",
        "data": [
            {
                "id": "local/qwen3-8b",
                "object": "model",
                "owned_by": "gpu-pool",
                "caliban": {
                    "kind": "chat",
                    "family": "qwen3",
                    "capabilities": {"tools": True, "reasoning": "hybrid"},
                    "trust_tier": "t0_sovereign",
                },
            },
            {"id": "caliban/auto", "object": "model", "owned_by": "caliban"},
        ],
    }
    models = make_client(lambda r: httpx.Response(200, json=data)).models.list().data
    info = models[0].caliban
    assert info is not None
    assert info.kind == "chat" and info.family == "qwen3"
    assert info.capabilities.reasoning == "hybrid"
    assert info.capabilities.reasoning_control == "none"  # contract default
    assert models[1].caliban is None


# ───────────────────────────── admin ─────────────────────────────

BASE = "http://cp.test:8081/api/v1"
QWEN = {
    "id": "local/qwen3-8b",
    "provider_id": "gpu-pool",
    "provider_kind": "openai_compatible",
    "upstream_model": "Qwen/Qwen3-8B",
    "kind": "chat",
    "family": "qwen3",
    "capabilities": {"tools": True, "reasoning": "hybrid", "reasoning_control": "enable_thinking"},
    "trust_tier": "t0_sovereign",
}
PROVIDER = {
    "id": "gpu-pool",
    "kind": "openai_compatible",
    "base_url": "http://vllm:8000/v1",
    "trust_tier": "t0_sovereign",
    "cache_salt": True,
    "has_api_key": False,
    "tenants": [],
}


@pytest.fixture
def admin() -> Iterator[tuple[CalibanAdmin, respx.MockRouter]]:
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield CalibanAdmin(token="adm", base_url="http://cp.test:8081"), router


def test_models_list_and_create_with_kwargs(admin: tuple[CalibanAdmin, respx.MockRouter]) -> None:
    client, r = admin
    r.get("/models").respond(200, json=[QWEN])
    post = r.post("/models").respond(201, json=QWEN)

    listed = client.models.list()[0]
    assert listed.provider_id == "gpu-pool" and listed.capabilities.reasoning == "hybrid"
    created = client.models.create(
        id="local/qwen3-8b",
        provider="gpu-pool",
        upstream_model="Qwen/Qwen3-8B",
        trust_tier="t0_sovereign",
        kind="chat",
        capabilities={"reasoning": "hybrid", "reasoning_control": "enable_thinking"},
    )
    assert created.id == "local/qwen3-8b"
    sent = json.loads(post.calls.last.request.content)
    assert sent["id"] == "local/qwen3-8b" and sent["provider"] == "gpu-pool"
    assert sent["capabilities"]["reasoning_control"] == "enable_thinking"
    assert "licence" not in sent
    with pytest.raises(ValueError, match="upstream_model"):
        client.models.create(id="x", provider="p", trust_tier="t0_sovereign")
    with pytest.raises(ValueError):
        m.ModelCreate.model_validate(
            {"id": "x", "provider": "p", "upstream_model": "u", "trust_tier": "t0_sovereign"}
            | {"kinds": "chat"}
        )
    assert post.call_count == 1


def test_discover_then_register_chosen_suggestions(
    admin: tuple[CalibanAdmin, respx.MockRouter],
) -> None:
    client, r = admin
    discovery = {
        "provider": "gpu-pool",
        "available": ["Qwen/Qwen3-8B", "BAAI/bge-m3", "meta-llama/Llama-3.1-8B-Instruct"],
        "suggested": [
            {
                "id": "local/qwen3-8b",
                "provider": "gpu-pool",
                "upstream_model": "Qwen/Qwen3-8B",
                "kind": "chat",
                "family": "qwen3",
                "capabilities": {"reasoning": "hybrid", "reasoning_control": "enable_thinking"},
                "trust_tier": "t0_sovereign",
                "confidence": 0.9,  # unknown to this SDK: dropped, not an error
            },
            {
                "id": "local/bge-m3",
                "provider": "gpu-pool",
                "upstream_model": "BAAI/bge-m3",
                "kind": "embedding",
                "trust_tier": "t0_sovereign",
            },
        ],
    }
    disc = r.post("/providers/gpu-pool/discover").respond(200, json=discovery)
    post = r.post("/models").respond(201, json=QWEN)

    found = client.providers.discover("gpu-pool")
    assert disc.called and len(found.available) == 3
    chosen = [s for s in found.suggested if s.kind == "chat"]
    assert len(chosen) == 1
    for s in chosen:
        client.models.create(s, context_window=32768)
    sent = json.loads(post.calls.last.request.content)
    assert sent["id"] == "local/qwen3-8b"
    assert sent["upstream_model"] == "Qwen/Qwen3-8B"
    assert sent["context_window"] == 32768
    assert sent["capabilities"]["reasoning"] == "hybrid"
    assert "confidence" not in sent
    assert post.call_count == 1


def test_model_delete_sends_slash_unescaped() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    client = CalibanAdmin(
        token="adm",
        base_url="http://cp.test:8081",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.models.delete("local/qwen3-8b")
    assert seen[0].method == "DELETE"
    assert seen[0].url.raw_path == b"/api/v1/models/local/qwen3-8b"
    client.models.delete("hf/Qwen/Qwen3 8B?x#y")
    assert seen[1].url.raw_path == b"/api/v1/models/hf/Qwen/Qwen3%208B%3Fx%23y"
    # Other path parameters stay fully escaped.
    client.providers.delete("a/b")
    assert seen[2].url.raw_path == b"/api/v1/providers/a%2Fb"

    for bad in ["../tenants", "local/./x", "local//x", "/local", "local/", ""]:
        with pytest.raises(CalibanError, match="Invalid model id"):
            client.models.delete(bad)
    assert len(seen) == 3
    assert model_id_path("a/%2e%2e") == "a/%252e%252e"


def test_model_delete_conflict(admin: tuple[CalibanAdmin, respx.MockRouter]) -> None:
    client, r = admin
    r.delete("/models/local/qwen3-8b").respond(
        409, json={"error": {"message": "model in use by a route", "type": "conflict"}}
    )
    with pytest.raises(ConflictError, match="in use"):
        client.models.delete("local/qwen3-8b")


def test_shared_providers(admin: tuple[CalibanAdmin, respx.MockRouter]) -> None:
    client, r = admin
    r.get("/providers").respond(200, json=[PROVIDER])
    post = r.post("/providers").respond(201, json=PROVIDER)
    r.get("/providers/gpu-pool/health").respond(
        200, json={"status": "ok", "latency_ms": 12, "models": 3}
    )
    delete = r.delete("/providers/gpu-pool").respond(204)

    assert client.providers.list()[0].cache_salt is True
    created = client.providers.create(
        id="gpu-pool",
        kind="openai_compatible",
        base_url="http://vllm:8000/v1",
        trust_tier="t0_sovereign",
        cache_salt=True,
        api_key="sk-local",
        tenants=("t1",),
    )
    assert created.has_api_key is False
    sent = json.loads(post.calls.last.request.content)
    assert sent == {
        "id": "gpu-pool",
        "kind": "openai_compatible",
        "base_url": "http://vllm:8000/v1",
        "trust_tier": "t0_sovereign",
        "cache_salt": True,
        "api_key": "sk-local",
        "tenants": ["t1"],
    }
    assert "sk-local" not in repr(m.SharedProviderCreate.model_validate(sent))
    health = client.providers.health("gpu-pool")
    assert health.status == "ok" and health.latency_ms == 12 and health.models == 3
    client.providers.delete("gpu-pool")
    assert delete.called


def test_discover_unreachable_and_provider_key_provider_id(
    admin: tuple[CalibanAdmin, respx.MockRouter],
) -> None:
    client, r = admin
    disc = r.post("/providers/down/discover").respond(
        502, json={"error": {"message": "connect refused", "type": "upstream_error"}}
    )
    with pytest.raises(UpstreamError):
        client.providers.discover("down")
    assert disc.call_count == 1  # POST is not retried on 502

    pk = {
        "id": "p1",
        "tenant_id": "t1",
        "kind": "openai_compatible",
        "label": "vllm",
        "trust_tier": "t0_sovereign",
        "created_at": "2026-10-02T10:00:00Z",
    }
    post = r.post("/tenants/t1/provider-keys").respond(201, json=pk)
    client.provider_keys.create(
        "t1",
        kind="openai_compatible",
        label="vllm",
        trust_tier="t0_sovereign",
        provider_id="local-llm",
    )
    assert json.loads(post.calls.last.request.content)["provider_id"] == "local-llm"
