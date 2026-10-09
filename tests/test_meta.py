"""Parsing of the ``x-caliban-cache-tier`` and ``x-caliban-intent`` response headers."""

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
