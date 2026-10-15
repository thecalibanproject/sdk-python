"""Parsing of the ``x-caliban-*`` response headers (cache tier, intent, node answers)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import caliban
from caliban import CalibanResponseMeta, IntentDecision

from .conftest import completion_body, make_async_client, make_client, sse_body


def test_parses_knn_decision() -> None:
    d = IntentDecision.parse("translate;confidence=0.912;stage=knn")
    assert d == IntentDecision(intent="translate", confidence=0.912, stage="knn")
    assert d is not None and d.knn_reason is None


def test_parses_pinned_and_keyword_fallback() -> None:
    assert IntentDecision.parse("pinned;confidence=1.000;stage=rules") == IntentDecision(
        intent="pinned", confidence=1.0, stage="rules"
    )
    d = IntentDecision.parse("summarize;confidence=0.640;stage=keyword;knn=timeout")
    assert d == IntentDecision(
        intent="summarize", confidence=0.64, stage="keyword", knn_reason="timeout"
    )


def test_tolerates_whitespace_case_order_unknown_fields_and_trailing_separator() -> None:
    d = IntentDecision.parse(
        "  code ; STAGE=KNN ; flavour=x; bare ; confidence=0 ; knn=abstain_margin;"
    )
    assert d == IntentDecision(
        intent="code", confidence=0.0, stage="knn", knn_reason="abstain_margin"
    )


def test_first_occurrence_wins_and_empty_knn_reason_dropped() -> None:
    d = IntentDecision.parse("qa;confidence=0.5;confidence=0.9;stage=knn;stage=rules;knn=")
    assert d == IntentDecision(intent="qa", confidence=0.5, stage="knn")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "translate",
        "translate;confidence=0.9",
        "translate;stage=knn",
        ";confidence=0.9;stage=knn",
        "confidence=0.9;stage=knn",
        "translate;confidence=high;stage=knn",
        "translate;confidence=-0.1;stage=knn",
        "translate;confidence=1.5;stage=knn",
        "translate;confidence=nan;stage=knn",
        "translate;confidence=inf;stage=knn",
        "translate;confidence=1e-3;stage=knn",
        "translate;confidence=;stage=knn",
        "translate;confidence=0.9;stage=llm",
        ";;;=;==",
    ],
)
def test_malformed_or_partial_intent_gives_none(value: str | None) -> None:
    assert IntentDecision.parse(value) is None


def test_intent_decision_is_frozen() -> None:
    d = IntentDecision(intent="qa", confidence=0.5, stage="knn")
    with pytest.raises(ValueError, match="frozen"):
        d.intent = "x"  # type: ignore[misc]


def test_meta_with_good_headers() -> None:
    meta = CalibanResponseMeta.from_headers(
        {
            "X-Caliban-Cache": "hit",
            "X-Caliban-Cache-Tier": "Semantic",
            "X-Caliban-Intent": "translate;confidence=0.912;stage=knn",
        }
    )
    assert meta.cache == "hit"
    assert meta.cache_tier == "semantic"
    assert meta.intent == IntentDecision(intent="translate", confidence=0.912, stage="knn")
    assert meta.headers["x-caliban-intent"] == "translate;confidence=0.912;stage=knn"


def test_meta_without_new_headers() -> None:
    meta = CalibanResponseMeta.from_headers({"x-caliban-cache": "miss"})
    assert meta.cache == "miss"
    assert meta.cache_tier is None and meta.intent is None


@pytest.mark.parametrize("tier", ["", "hit", "fuzzy", "semantic;q=1"])
def test_meta_with_malformed_headers(tier: str) -> None:
    meta = CalibanResponseMeta.from_headers(
        {
            "x-caliban-request-id": "r1",
            "x-caliban-cache": "hit",
            "x-caliban-cache-tier": tier,
            "x-caliban-intent": "translate;confidence=2;stage=knn",
        }
    )
    assert (meta.request_id, meta.cache) == ("r1", "hit")
    assert meta.cache_tier is None and meta.intent is None


HEADERS = {
    "x-caliban-cache": "hit",
    "x-caliban-cache-tier": "exact",
    "x-caliban-intent": "summarize;confidence=0.640;stage=keyword;knn=embed_error",
}
EXPECTED = IntentDecision(
    intent="summarize", confidence=0.64, stage="keyword", knn_reason="embed_error"
)


def test_exposed_on_completions_and_streams() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if b'"stream":true' in request.content.replace(b" ", b""):
            return httpx.Response(200, content=sse_body(), headers=HEADERS)
        return httpx.Response(200, json=completion_body(), headers=HEADERS)

    client = make_client(handler)
    resp = client.chat.completions.create(model="caliban/auto", messages=[])
    assert resp.caliban.cache_tier == "exact"
    assert resp.caliban.intent == EXPECTED
    with client.chat.completions.create(model="caliban/auto", messages=[], stream=True) as s:
        assert s.caliban.cache_tier == "exact"
        assert s.caliban.intent == EXPECTED


def test_exposed_on_async_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body(), headers=HEADERS)

    async def run() -> caliban.ChatCompletion:
        async with make_async_client(handler) as client:
            return await client.chat.completions.create(model="caliban/auto", messages=[])

    resp = asyncio.run(run())
    assert resp.caliban.cache_tier == "exact"
    assert resp.caliban.intent == EXPECTED


NODE_ANSWER = {
    "x-caliban-request-id": "req_7",
    "x-caliban-routed-model": "node/triage@v3",
    "x-caliban-cache": "bypass",
    "x-caliban-pii-entities": "2",
    "x-caliban-cost-usd": "0.0011",
    "x-caliban-billed-usd": "0.004",
    "x-caliban-route": "node/triage@v3",
    "x-caliban-run-status": "succeeded",
    "Caliban-Run-Id": "run_9",
}


def test_meta_on_a_node_answer() -> None:
    meta = CalibanResponseMeta.from_headers(NODE_ANSWER)
    assert (meta.request_id, meta.routed_model, meta.cache) == ("req_7", "node/triage@v3", "bypass")
    assert (meta.pii_entities, meta.cost_usd, meta.billed_usd) == (2, 0.0011, 0.004)
    assert (meta.route, meta.route_fallback) == ("node/triage@v3", None)
    assert (meta.run_id, meta.run_status) == ("run_9", "succeeded")
    # Caliban-Run-Id is not an x-caliban-* header, so it is not in `headers`.
    assert "caliban-run-id" not in meta.headers


def test_meta_caliban_auto_fallback_and_malformed_node_headers() -> None:
    meta = CalibanResponseMeta.from_headers(
        {"x-caliban-route": "model:local/qwen3-8b", "x-caliban-route-fallback": "low_confidence"}
    )
    assert (meta.route, meta.route_fallback, meta.run_id) == (
        "model:local/qwen3-8b",
        "low_confidence",
        None,
    )
    bad = CalibanResponseMeta.from_headers(
        {"x-caliban-billed-usd": "free", "x-caliban-route": "  ", "caliban-run-id": ""}
    )
    assert (bad.billed_usd, bad.route, bad.run_id, bad.run_status) == (None, None, None, None)
    assert CalibanResponseMeta.from_headers({"x-caliban-billed-usd": "-1"}).billed_usd is None


def test_node_answer_meta_on_completions() -> None:
    client = make_client(lambda r: httpx.Response(200, json=completion_body(), headers=NODE_ANSWER))
    resp = client.chat.completions.create(
        model="node/triage", messages=[{"role": "user", "content": "hi"}]
    )
    assert (resp.caliban.run_id, resp.caliban.billed_usd) == ("run_9", 0.004)
    assert resp.caliban.pii_entities == 2


def test_meta_from_an_anthropic_messages_response() -> None:
    # /v1/messages sends the billing, route and run headers chat does; from_headers reads them
    # off any response (here as the Anthropic SDK's raw response would carry them).
    headers = httpx.Headers(
        {
            "x-caliban-request-id": "req_m1",
            "x-caliban-cost-usd": "0.0021",
            "x-caliban-billed-usd": "0.01",
            "x-caliban-route": "node/triage@v3",
            "Caliban-Run-Id": "run_m1",
            "x-caliban-run-status": "succeeded",
            "content-type": "application/json",
        }
    )
    meta = CalibanResponseMeta.from_headers(headers)
    assert (meta.request_id, meta.cost_usd, meta.billed_usd) == ("req_m1", 0.0021, 0.01)
    assert (meta.route, meta.run_id, meta.run_status) == ("node/triage@v3", "run_m1", "succeeded")


SERVER_TIMING = (
    'auth;dur=0.210, route;dur=4.810;desc="knn", embed;dur=4.020, pii;dur=1.130, '
    'cache;dur=0.040;desc="miss", upstream;dur=182.400, upstream_total;dur=611.900, '
    "total;dur=618.700"
)


def test_server_timing_parses_every_stage_in_order() -> None:
    t = caliban.ServerTimingEntry.parse(SERVER_TIMING)
    assert t is not None
    assert [e.name for e in t] == [
        "auth", "route", "embed", "pii", "cache", "upstream", "upstream_total", "total",
    ]  # fmt: skip
    assert (t[1].dur, t[1].desc) == (4.81, "knn")
    assert (t[4].dur, t[4].desc) == (0.04, "miss")
    assert (t[-1].name, t[-1].dur, t[-1].desc) == ("total", 618.7, None)


def test_server_timing_node_without_dur_quotes_and_bad_values() -> None:
    parse = caliban.ServerTimingEntry.parse
    node = parse('node;desc="run_1", total;dur=3')
    assert node is not None and [(e.name, e.dur, e.desc) for e in node] == [
        ("node", None, "run_1"),
        ("total", 3.0, None),
    ]
    quoted = parse('x;desc="a, b; c"')
    assert quoted is not None and quoted[0].desc == "a, b; c"
    assert parse(None) is None and parse("  ") is None and parse(", ;dur=1") is None
    bad = parse("auth;dur=abc, pii;dur=-1")
    assert bad is not None and [(e.name, e.dur) for e in bad] == [("auth", None), ("pii", None)]


def test_server_timing_on_response_meta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body(), headers={"Server-Timing": SERVER_TIMING})

    resp = make_client(handler).chat.completions.create(
        model="caliban/auto", messages=[{"role": "user", "content": "hi"}]
    )
    timing = resp.caliban.server_timing
    assert timing is not None and timing[5].name == "upstream" and timing[5].dur == 182.4
    assert CalibanResponseMeta.from_headers({}).server_timing is None
