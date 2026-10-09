from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from caliban import SSEDecoder, StreamError
from caliban._streaming import ServerSentEvent

from .conftest import chunk, chunked, make_async_client, make_client, sse_body


def decode(data: bytes, size: int | None = None) -> list[ServerSentEvent]:
    dec = SSEDecoder()
    events: list[ServerSentEvent] = []
    for part in chunked(data, size or max(len(data), 1)):
        events.extend(dec.feed(part))
    events.extend(dec.flush())
    return events


STREAM = (
    b": keep-alive\n\n"
    b"event: message\r\n"
    b"id: 7\r\n"
    b"data: first\r\n"
    b"data:second\r\n"
    b"\r\n"
    b"data: caf\xc3\xa9 \xe2\x82\xac\r"  # CR-only line ending, multi-byte UTF-8
    b"\r"
    b"retry: 1500\n"
    b"data\n"  # field with no colon -> empty value
    b"\n"
    b"event: ping\n\n"  # no data -> dropped
)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 64, None])
def test_decoder_handles_every_split(size: int | None) -> None:
    events = decode(STREAM, size)
    assert [e.data for e in events] == ["first\nsecond", "café €", ""]
    assert events[0].event == "message"
    assert events[0].id == "7"
    assert events[1].id == "7"  # last event id persists
    assert events[2].retry == 1500


def test_crlf_split_across_chunks_is_one_line_break() -> None:
    dec = SSEDecoder()
    assert dec.feed(b"data: a\r") == []
    # The \n completing \r\n must not be read as a second (blank) line.
    assert dec.feed(b"\ndata: b\r\n\r\n") == [ServerSentEvent(data="a\nb")]


def test_bom_and_comments_ignored() -> None:
    events = decode(b"\xef\xbb\xbf: hello\n:another\ndata: x\n\n")
    assert [e.data for e in events] == ["x"]


def test_only_one_leading_space_stripped() -> None:
    assert decode(b"data:   padded\n\n")[0].data == "  padded"


def test_unterminated_final_event_is_flushed() -> None:
    assert [e.data for e in decode(b"data: tail")] == ["tail"]


def test_utf8_split_inside_codepoint() -> None:
    raw = "data: 日本語\n\n".encode()
    assert decode(raw, 1)[0].data == "日本語"


def _stream_handler(body: bytes, size: int, headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", **(headers or {})},
            content=chunked(body, size),
        )

    return handler


@pytest.mark.parametrize("size", [1, 4, 17, 4096])
def test_stream_yields_chunks_and_stops_at_done(size: int) -> None:
    body = sse_body(chunk(role="assistant"), chunk("Hel"), chunk("lo"), chunk(finish="stop"))
    body += b"data: " + chunk("AFTER-DONE").encode() + b"\n\n"
    client = make_client(
        _stream_handler(body, size, {"x-caliban-request-id": "req-9", "x-caliban-cache": "miss"})
    )
    with client.chat.completions.create(model="m", messages=[], stream=True) as stream:
        assert stream.caliban.request_id == "req-9"
        assert stream.caliban.cache == "miss"
        chunks = list(stream)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert stream.response.is_closed


def test_iter_text() -> None:
    body = sse_body(chunk(role="assistant"), chunk("a"), chunk(""), chunk("b"))
    client = make_client(_stream_handler(body, 3))
    stream = client.chat.completions.create(model="m", messages=[], stream=True)
    assert list(stream.iter_text()) == ["a", "b"]


def test_stream_sends_stream_flag_and_accept_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse_body(chunk("x")))

    client = make_client(handler)
    list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert b'"stream":true' in seen[0].content
    assert seen[0].headers["accept"] == "text/event-stream"


def test_unknown_events_and_empty_data_skipped() -> None:
    body = b'event: caliban.meta\ndata: {"x":1}\n\ndata: \n\n' + sse_body(chunk("ok"))
    client = make_client(_stream_handler(body, 5))
    out = list(client.chat.completions.create(model="m", messages=[], stream=True))
    assert len(out) == 1


def test_error_event_mid_stream_raises_stream_error() -> None:
    body = (
        sse_body(chunk("partial"), done=False)
        + b'data: {"error": {"message": "upstream died", '
        + b'"type": "upstream_error", "code": "x1"}}\n\n'
    )
    client = make_client(_stream_handler(body, 8, {"x-caliban-request-id": "req-1"}))
    stream = client.chat.completions.create(model="m", messages=[], stream=True)
    assert next(stream).choices[0].delta.content == "partial"
    with pytest.raises(StreamError) as ei:
        next(stream)
    assert ei.value.type == "upstream_error"
    assert ei.value.code == "x1"
    assert ei.value.request_id == "req-1"


def test_named_error_event() -> None:
    body = b'event: error\ndata: {"error": {"message": "budget", "type": "policy_violation"}}\n\n'
    client = make_client(_stream_handler(body, 1000))
    with pytest.raises(StreamError, match="budget"):
        list(client.chat.completions.create(model="m", messages=[], stream=True))


def test_malformed_json_raises_stream_error() -> None:
    client = make_client(_stream_handler(b"data: {not json\n\n", 1000))
    with pytest.raises(StreamError, match="Malformed"):
        list(client.chat.completions.create(model="m", messages=[], stream=True))


def test_async_stream() -> None:
    body = sse_body(chunk("Hel"), chunk("lo"))

    async def agen() -> AsyncIterator[bytes]:
        for part in chunked(body, 3):
            yield part

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-caliban-routed-model": "small"}, content=agen())

    async def run() -> tuple[str, str | None]:
        async with make_async_client(handler) as client:
            stream = await client.chat.completions.create(model="m", messages=[], stream=True)
            async with stream:
                text = "".join([t async for t in stream.iter_text()])
            return text, stream.caliban.routed_model

    assert asyncio.run(run()) == ("Hello", "small")
