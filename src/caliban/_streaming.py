"""Server-Sent Events parsing and chat-completion streams.

:class:`SSEDecoder` is an incremental decoder following the WHATWG
event-stream rules:

* lines end with ``\\r\\n``, ``\\r`` or ``\\n``, including a ``\\r\\n`` pair split
  across two network chunks;
* multi-byte UTF-8 sequences may be split across chunks;
* a leading BOM is ignored;
* lines starting with ``:`` are comments (keep-alives) and are ignored;
* ``field: value`` strips exactly one leading space from the value; a line with
  no colon is a field with an empty value;
* multiple ``data:`` lines are joined with ``\\n``;
* a blank line dispatches the event; an event with no data is dropped.

Unlike a browser, a final event that is not followed by a blank line is still
dispatched at end of stream (lenient towards proxies that trim the trailer).
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, NamedTuple

import httpx
from pydantic import ValidationError

from .errors import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    StreamError,
    parse_error_body,
)
from .types import CalibanResponseMeta, ChatCompletionChunk

__all__ = [
    "AsyncChatCompletionStream",
    "ChatCompletionStream",
    "SSEDecoder",
    "ServerSentEvent",
    "StreamText",
    "TextPart",
]


class TextPart(NamedTuple):
    """A piece of streamed text: model reasoning ("thinking") or answer content."""

    kind: Literal["reasoning", "content"]
    text: str


class StreamText(NamedTuple):
    """Reasoning and answer text of a fully consumed stream (choice 0)."""

    reasoning: str
    """Concatenated ``delta.reasoning_content`` (or ``delta.reasoning``); empty if none."""
    content: str
    """Concatenated ``delta.content``: the answer."""


def _parts(chunk: ChatCompletionChunk) -> Iterator[TextPart]:
    for choice in chunk.choices:
        if choice.index != 0:
            continue
        reasoning = choice.delta.reasoning_text
        if reasoning:
            yield TextPart("reasoning", reasoning)
        if choice.delta.content:
            yield TextPart("content", choice.delta.content)


_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True)
class ServerSentEvent:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None

    def json(self) -> Any:
        return json.loads(self.data)


class SSEDecoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._pending_cr = False
        self._started = False
        self._data: list[str] = []
        self._event: str | None = None
        self._last_id: str | None = None
        self._retry: int | None = None

    def feed(self, chunk: bytes) -> list[ServerSentEvent]:
        return self._feed_text(self._utf8.decode(chunk))

    def flush(self) -> list[ServerSentEvent]:
        events = self._feed_text(self._utf8.decode(b"", final=True))
        if self._buf:
            line, self._buf = self._buf, ""
            self._process_line(line, events)
        if self._data:
            self._dispatch(events)
        return events

    def _feed_text(self, text: str) -> list[ServerSentEvent]:
        events: list[ServerSentEvent] = []
        if not text:
            return events
        if not self._started:
            self._started = True
            if text.startswith("﻿"):
                text = text[1:]
        if self._pending_cr:
            self._pending_cr = False
            if text.startswith("\n"):
                text = text[1:]
        buf = self._buf + text
        pos = 0
        for m in _LINE_END.finditer(buf):
            if m.group() == "\r" and m.end() == len(buf):
                # Might be the first half of a \r\n split across chunks.
                self._pending_cr = True
            self._process_line(buf[pos : m.start()], events)
            pos = m.end()
        self._buf = buf[pos:]
        return events

    def _process_line(self, line: str, events: list[ServerSentEvent]) -> None:
        if line == "":
            self._dispatch(events)
            return
        if line.startswith(":"):
            return
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id":
            if "\0" not in value:
                self._last_id = value
        elif field == "retry" and value.isdigit():
            self._retry = int(value)
        # Unknown fields are ignored per spec.

    def _dispatch(self, events: list[ServerSentEvent]) -> None:
        if not self._data:
            self._event = None
            return
        events.append(
            ServerSentEvent(
                data="\n".join(self._data),
                event=self._event or None,
                id=self._last_id,
                retry=self._retry,
            )
        )
        self._data = []
        self._event = None


_DONE = object()


def _event_to_chunk(sse: ServerSentEvent, request_id: str | None) -> ChatCompletionChunk | object:
    """Return a chunk, ``_DONE``, or ``None`` (skip). Raises :class:`StreamError`."""
    data = sse.data.strip()
    if data == "[DONE]":
        return _DONE
    if sse.event not in (None, "message", "error"):
        return None  # unknown/custom event types (e.g. pings) are skipped
    if not data:
        return None
    try:
        obj = json.loads(data)
    except ValueError as exc:
        if sse.event == "error":
            raise StreamError(data, request_id=request_id, body=data) from exc
        raise StreamError(
            f"Malformed JSON in stream event: {data[:200]!r}",
            type="invalid_stream",
            request_id=request_id,
            body=data,
        ) from exc
    if sse.event == "error" or (isinstance(obj, dict) and "error" in obj and "choices" not in obj):
        message, err_type, code = parse_error_body(obj)
        raise StreamError(
            message or "Stream error",
            type=err_type,
            code=code,
            request_id=request_id,
            body=obj,
        )
    try:
        return ChatCompletionChunk.model_validate(obj)
    except ValidationError as exc:
        raise APIResponseValidationError(
            f"Stream chunk does not match the contract: {exc.errors()[0]['msg']}",
            type="invalid_response",
            request_id=request_id,
            body=obj,
        ) from exc


class ChatCompletionStream:
    """Iterator over :class:`ChatCompletionChunk` from an SSE response.

    Use as a context manager (or exhaust it) so the connection is released.
    """

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self._meta = CalibanResponseMeta.from_headers(response.headers)
        self._iterator = self._iter_chunks()

    @property
    def caliban(self) -> CalibanResponseMeta:
        return self._meta

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        return next(self._iterator)

    def __enter__(self) -> ChatCompletionStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.response.close()

    def iter_text(self) -> Iterator[str]:
        """Yield only the non-empty content deltas of choice 0 (reasoning is skipped)."""
        for chunk in self:
            for choice in chunk.choices:
                if choice.index == 0 and choice.delta.content:
                    yield choice.delta.content

    def iter_parts(self) -> Iterator[TextPart]:
        """Yield choice 0's non-empty deltas as :class:`TextPart`, tagged ``reasoning``
        (``delta.reasoning_content``, or ``delta.reasoning`` on some servers) or ``content``."""
        for chunk in self:
            yield from _parts(chunk)

    def collect(self) -> StreamText:
        """Consume the stream; return reasoning and answer text separately."""
        reasoning: list[str] = []
        content: list[str] = []
        for part in self.iter_parts():
            (reasoning if part.kind == "reasoning" else content).append(part.text)
        return StreamText("".join(reasoning), "".join(content))

    def _iter_chunks(self) -> Iterator[ChatCompletionChunk]:
        decoder = SSEDecoder()
        rid = self._meta.request_id
        try:
            for raw in self.response.iter_bytes():
                for sse in decoder.feed(raw):
                    item = _event_to_chunk(sse, rid)
                    if item is _DONE:
                        return
                    if isinstance(item, ChatCompletionChunk):
                        yield item
            for sse in decoder.flush():
                item = _event_to_chunk(sse, rid)
                if item is _DONE:
                    return
                if isinstance(item, ChatCompletionChunk):
                    yield item
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                f"Stream timed out: {exc}", request=self.response.request
            ) from exc
        except httpx.TransportError as exc:
            raise APIConnectionError(
                f"Stream interrupted: {exc}", request=self.response.request
            ) from exc
        finally:
            self.response.close()


class AsyncChatCompletionStream:
    """Async iterator over :class:`ChatCompletionChunk` from an SSE response."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self._meta = CalibanResponseMeta.from_headers(response.headers)
        self._iterator = self._iter_chunks()

    @property
    def caliban(self) -> CalibanResponseMeta:
        return self._meta

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        return await self._iterator.__anext__()

    async def __aenter__(self) -> AsyncChatCompletionStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.response.aclose()

    async def iter_text(self) -> AsyncIterator[str]:
        async for chunk in self:
            for choice in chunk.choices:
                if choice.index == 0 and choice.delta.content:
                    yield choice.delta.content

    async def iter_parts(self) -> AsyncIterator[TextPart]:
        """Async version of :meth:`ChatCompletionStream.iter_parts`."""
        async for chunk in self:
            for part in _parts(chunk):
                yield part

    async def collect(self) -> StreamText:
        """Consume the stream; return reasoning and answer text separately."""
        reasoning: list[str] = []
        content: list[str] = []
        async for part in self.iter_parts():
            (reasoning if part.kind == "reasoning" else content).append(part.text)
        return StreamText("".join(reasoning), "".join(content))

    async def _iter_chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        decoder = SSEDecoder()
        rid = self._meta.request_id
        try:
            async for raw in self.response.aiter_bytes():
                for sse in decoder.feed(raw):
                    item = _event_to_chunk(sse, rid)
                    if item is _DONE:
                        return
                    if isinstance(item, ChatCompletionChunk):
                        yield item
            for sse in decoder.flush():
                item = _event_to_chunk(sse, rid)
                if item is _DONE:
                    return
                if isinstance(item, ChatCompletionChunk):
                    yield item
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                f"Stream timed out: {exc}", request=self.response.request
            ) from exc
        except httpx.TransportError as exc:
            raise APIConnectionError(
                f"Stream interrupted: {exc}", request=self.response.request
            ) from exc
        finally:
            await self.response.aclose()
