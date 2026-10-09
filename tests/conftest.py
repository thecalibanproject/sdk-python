from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

import caliban._base as base
from caliban import AsyncCaliban, Caliban

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of sleeping."""
    delays: list[float] = []

    async def fake_async_sleep(s: float) -> None:
        delays.append(s)

    monkeypatch.setattr(base, "_sleep", delays.append)
    monkeypatch.setattr(base, "_async_sleep", fake_async_sleep)
    return delays


def make_client(handler: Handler, **kw: Any) -> Caliban:
    return Caliban(
        api_key="cal_test",
        base_url="http://caliban.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


def make_async_client(handler: Handler, **kw: Any) -> AsyncCaliban:
    return AsyncCaliban(
        api_key="cal_test",
        base_url="http://caliban.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kw,
    )


def completion_body(text: str = "hello", model: str = "m-1", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    body.update(extra)
    return body


def chunk(content: str | None = None, *, role: str | None = None, finish: str | None = None) -> str:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "m-1",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def sse_body(*datas: str, done: bool = True) -> bytes:
    out = "".join(f"data: {d}\n\n" for d in datas)
    if done:
        out += "data: [DONE]\n\n"
    return out.encode()


def chunked(data: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def error_body(message: str, type_: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}
