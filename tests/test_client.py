from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import httpx
import pytest

import caliban
from caliban import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Caliban,
    CalibanError,
    CalibanOptions,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    PolicyViolationError,
    RateLimitError,
    ServiceUnavailableError,
    StreamError,
    UpstreamError,
)

from .conftest import (
    chunk,
    completion_body,
    error_body,
    make_async_client,
    make_client,
    sse_body,
)

HEADERS = {
    "x-caliban-request-id": "req-123",
    "x-caliban-routed-model": "llama-3.1-8b",
    "x-caliban-cache": "hit",
    "x-caliban-pii-entities": "3",
    "x-caliban-something-new": "v",
}


def test_create_sends_body_auth_and_parses_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_body("hi"), headers=HEADERS)

    client = make_client(handler)
    resp = client.chat.completions.create(
        model="caliban/auto",
        messages=[{"role": "user", "content": "hello"}, caliban.ChatMessage(role="user")],
        caliban=CalibanOptions(pii="reversible", cache="exact", datasources=["sales_dw"], zdr=True),
        temperature=0,
        top_p=0.5,
        extra_body={"seed": 1},
    )
    req = seen[0]
    assert req.url == "http://caliban.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer cal_test"
    assert req.headers["user-agent"].startswith("caliban-sdk-python/")
    body = json.loads(req.content)
    assert body["caliban"] == {
        "pii": "reversible",
        "cache": "exact",
        "datasources": ["sales_dw"],
        "zdr": True,
    }
    assert body["temperature"] == 0
    assert body["top_p"] == 0.5
    assert body["seed"] == 1
    assert "stream" not in body
    assert body["messages"][1] == {"role": "user"}

    assert resp.text == "hi"
    assert resp.usage is not None and resp.usage.total_tokens == 15
    assert resp.caliban.request_id == "req-123"
    assert resp.caliban.routed_model == "llama-3.1-8b"
    assert resp.caliban.cache == "hit"
    assert resp.caliban.pii_entities == 3
    assert resp.caliban.headers["x-caliban-something-new"] == "v"


def test_caliban_options_from_dict_and_typo_rejected() -> None:
    client = make_client(lambda r: httpx.Response(200, json=completion_body()))
    client.chat.completions.create(model="m", messages=[], caliban={"node": "n", "trace_id": "t"})
    with pytest.raises(ValueError, match="piii"):
        client.chat.completions.create(model="m", messages=[], caliban={"piii": "off"})
    with pytest.raises(ValueError):
        CalibanOptions(max_cost_usd=-1)


def test_unknown_response_fields_preserved() -> None:
    client = make_client(
        lambda r: httpx.Response(200, json=completion_body(system_fingerprint="fp", caliban=1))
    )
    resp = client.chat.completions.create(model="m", messages=[])
    assert resp.model_extra == {"system_fingerprint": "fp", "caliban": 1}
    assert isinstance(resp.caliban, caliban.CalibanResponseMeta)


def test_models_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "a", "object": "model", "owned_by": "t"}]},
        )

    assert [m.id for m in make_client(handler).models.list().data] == ["a"]


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBAN_API_KEY", raising=False)
    with pytest.raises(CalibanError, match="API key"):
        Caliban()


def test_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBAN_API_KEY", "cal_env")
    monkeypatch.setenv("CALIBAN_BASE_URL", "http://gw:8080/v1/")
    c = Caliban()
    assert c.base_url == "http://gw:8080/v1"
    c.close()


# ───────────────────────────── errors ─────────────────────────────


@pytest.mark.parametrize(
    ("status", "body", "exc", "etype"),
    [
        (400, error_body("bad", "invalid_request_error"), BadRequestError, "invalid_request_error"),
        (401, error_body("nope", "authentication_error"), AuthenticationError, None),
        (403, error_body("denied", "forbidden"), PermissionDeniedError, None),
        (403, error_body("pii", "policy_violation", "pii_t3"), PolicyViolationError, None),
        (400, error_body("cost", "policy_violation"), PolicyViolationError, None),
        (404, error_body("missing", "not_found"), NotFoundError, None),
        (429, error_body("slow", "rate_limited"), RateLimitError, None),
        (502, error_body("upstream", "upstream_error"), UpstreamError, None),
        (503, error_body("down", "unavailable"), ServiceUnavailableError, None),
        (500, error_body("boom", "server_error"), InternalServerError, None),
    ],
)
def test_error_mapping(
    status: int, body: dict[str, object], exc: type[Exception], etype: str | None
) -> None:
    client = make_client(
        lambda r: httpx.Response(status, json=body, headers={"x-caliban-request-id": "r1"}),
        max_retries=0,
    )
    with pytest.raises(exc) as ei:
        client.chat.completions.create(model="m", messages=[])
    err = ei.value
    assert isinstance(err, caliban.APIStatusError)
    assert err.status_code == status
    assert err.message == body["error"]["message"]  # type: ignore[index]
    assert err.type == body["error"]["type"]  # type: ignore[index]
    assert err.request_id == "r1"
    if etype:
        assert err.type == etype


def test_policy_violation_code_and_str() -> None:
    client = make_client(
        lambda r: httpx.Response(403, json=error_body("PII to t3", "policy_violation", "pii_t3"))
    )
    with pytest.raises(PolicyViolationError) as ei:
        client.chat.completions.create(model="m", messages=[])
    assert ei.value.code == "pii_t3"
    assert "403" in str(ei.value) and "policy_violation" in str(ei.value)


def test_non_json_error_body() -> None:
    client = make_client(lambda r: httpx.Response(500, text="<html>oops</html>"))
    with pytest.raises(InternalServerError, match="oops"):
        client.chat.completions.create(model="m", messages=[])


# ───────────────────────────── retries ─────────────────────────────


def _sequence(*responses: httpx.Response) -> tuple[list[httpx.Request], object]:
    calls: list[httpx.Request] = []
    it: Iterator[httpx.Response] = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return next(it)

    return calls, handler


@pytest.mark.parametrize("status", [429, 502, 503])
def test_retries_retryable_status_then_succeeds(status: int, no_sleep: list[float]) -> None:
    calls, handler = _sequence(
        httpx.Response(status, json=error_body("x", "y")),
        httpx.Response(200, json=completion_body("ok")),
    )
    client = make_client(handler)  # type: ignore[arg-type]
    assert client.chat.completions.create(model="m", messages=[]).text == "ok"
    assert len(calls) == 2
    assert len(no_sleep) == 1 and 0 < no_sleep[0] <= 0.5


def test_retry_after_header_honoured(no_sleep: list[float]) -> None:
    _, handler = _sequence(
        httpx.Response(429, json=error_body("x", "rate_limited"), headers={"retry-after": "2"}),
        httpx.Response(200, json=completion_body()),
    )
    make_client(handler).chat.completions.create(model="m", messages=[])  # type: ignore[arg-type]
    assert no_sleep == [2.0]


def test_retries_exhausted(no_sleep: list[float]) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json=error_body("slow down", "rate_limited"))

    with pytest.raises(RateLimitError, match="slow down"):
        make_client(handler, max_retries=3).chat.completions.create(model="m", messages=[])
    assert len(calls) == 4
    assert no_sleep == sorted(no_sleep)  # exponential backoff grows


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_non_retryable_status_not_retried(status: int) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=error_body("x", "y"))

    with pytest.raises(caliban.APIStatusError):
        make_client(handler).chat.completions.create(model="m", messages=[])
    assert len(calls) == 1


def test_connect_error_retried_then_raised() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(APIConnectionError, match="refused"):
        make_client(handler, max_retries=2).chat.completions.create(model="m", messages=[])
    assert len(calls) == 3


def test_read_timeout_not_retried_for_post() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(APITimeoutError):
        make_client(handler).chat.completions.create(model="m", messages=[])
    assert len(calls) == 1


def test_read_timeout_retried_for_idempotent_get() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"object": "list", "data": []})

    assert make_client(handler).models.list().data == []
    assert len(calls) == 2


def test_stream_retried_before_first_byte() -> None:
    calls, handler = _sequence(
        httpx.Response(503, json=error_body("warming", "unavailable")),
        httpx.Response(200, content=sse_body(chunk("ok"))),
    )
    client = make_client(handler)  # type: ignore[arg-type]
    out = list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert len(calls) == 2
    assert out[0].choices[0].delta.content == "ok"


def test_stream_never_retried_after_it_begins() -> None:
    calls: list[httpx.Request] = []

    def body() -> Iterator[bytes]:
        yield sse_body(chunk("partial"), done=False)
        raise httpx.ReadError("connection reset")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=body())

    stream = make_client(handler, max_retries=5).chat.completions.create(
        model="m", messages=[], stream=True
    )
    assert next(stream).choices[0].delta.content == "partial"
    with pytest.raises(APIConnectionError, match="interrupted"):
        next(stream)
    assert len(calls) == 1


def test_stream_mid_error_not_retried() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            content=sse_body(chunk("a"), done=False)
            + b'data: {"error": {"message": "x", "type": "rate_limited"}}\n\n',
        )

    with pytest.raises(StreamError):
        list(make_client(handler).chat.completions.create(model="m", messages=[], stream=True))
    assert len(calls) == 1


# ───────────────────────────── timeouts ─────────────────────────────


def test_timeouts_default_and_per_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_body())

    client = make_client(handler, timeout=12.0)
    client.chat.completions.create(model="m", messages=[])
    client.chat.completions.create(model="m", messages=[], timeout=httpx.Timeout(3.0, connect=1.0))
    assert seen[0].extensions["timeout"]["read"] == 12.0
    assert seen[1].extensions["timeout"] == {"connect": 1.0, "read": 3.0, "write": 3.0, "pool": 3.0}


def test_timeout_maps_to_api_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("t/o", request=request)

    with pytest.raises(APITimeoutError):
        make_client(handler, max_retries=0).chat.completions.create(model="m", messages=[])


# ───────────────────────────── async ─────────────────────────────


def test_async_create_retry_and_headers(no_sleep: list[float]) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(502, json=error_body("x", "upstream_error"))
        return httpx.Response(200, json=completion_body("async"), headers=HEADERS)

    async def run() -> caliban.ChatCompletion:
        async with make_async_client(handler) as client:
            return await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "q"}], caliban={"node": "n"}
            )

    resp = asyncio.run(run())
    assert resp.text == "async"
    assert resp.caliban.routed_model == "llama-3.1-8b"
    assert len(calls) == 2 and len(no_sleep) == 1
    assert json.loads(calls[1].content)["caliban"] == {"node": "n"}


def test_async_errors_and_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "x", "owned_by": "t"}]}
            )
        return httpx.Response(401, json=error_body("bad key", "authentication_error"))

    async def run() -> list[str]:
        client = make_async_client(handler)
        with pytest.raises(AuthenticationError):
            await client.chat.completions.create(model="m", messages=[])
        ids = [m.id for m in (await client.models.list()).data]
        await client.aclose()
        return ids

    assert asyncio.run(run()) == ["x"]


def test_contract_mismatch_raises_typed_error() -> None:
    client = make_client(lambda r: httpx.Response(200, json={"id": "x"}))
    with pytest.raises(caliban.APIResponseValidationError):
        client.chat.completions.create(model="m", messages=[])
    bad_chunk = make_client(lambda r: httpx.Response(200, content=sse_body('{"choices": []}')))
    with pytest.raises(caliban.APIResponseValidationError):
        list(bad_chunk.chat.completions.create(model="m", messages=[], stream=True))
