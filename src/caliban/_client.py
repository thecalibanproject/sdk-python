"""Data-plane clients: :class:`Caliban` and :class:`AsyncCaliban`."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, overload

import httpx
from pydantic import BaseModel, TypeAdapter

from ._base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_STATUSES,
    AsyncHTTP,
    SyncHTTP,
    TimeoutTypes,
    parse_response,
)
from ._streaming import AsyncChatCompletionStream, ChatCompletionStream
from .errors import CalibanError
from .types import (
    CalibanOptions,
    CalibanResponseMeta,
    ChatCompletion,
    ChatMessage,
    CreateEmbeddingResponse,
    ModelList,
    RerankResponse,
)

__all__ = ["DEFAULT_BASE_URL", "AsyncCaliban", "Caliban"]

DEFAULT_BASE_URL = "http://localhost:8080/v1"

MessageParam = ChatMessage | Mapping[str, Any]
CalibanParam = CalibanOptions | Mapping[str, Any]


def _resolve(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    key = api_key if api_key is not None else os.environ.get("CALIBAN_API_KEY")
    if not key:
        raise CalibanError(
            "No API key. Pass api_key=... or set CALIBAN_API_KEY (a tenant key, cal_...)."
        )
    url = base_url or os.environ.get("CALIBAN_BASE_URL") or DEFAULT_BASE_URL
    return key, url


def _message_to_dict(m: MessageParam) -> dict[str, Any]:
    if isinstance(m, BaseModel):
        return m.model_dump(mode="json", exclude_none=True)
    return dict(m)


def build_chat_body(
    *,
    model: str,
    messages: Sequence[MessageParam],
    stream: bool,
    caliban: CalibanParam | None,
    temperature: float | None,
    max_tokens: int | None,
    tools: Sequence[Mapping[str, Any]] | None,
    extra_body: Mapping[str, Any] | None,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": [_message_to_dict(m) for m in messages]}
    if stream:
        body["stream"] = True
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if tools is not None:
        body["tools"] = [dict(t) for t in tools]
    body.update({k: v for k, v in params.items() if v is not None})
    if caliban is not None:
        opts = caliban if isinstance(caliban, CalibanOptions) else CalibanOptions(**caliban)
        ext = opts.to_body()
        if ext:
            body["caliban"] = ext
    if extra_body:
        body.update(extra_body)
    return body


def build_embeddings_body(
    *,
    model: str,
    input: str | Sequence[str],
    dimensions: int | None,
    encoding_format: Literal["float"] | None,
    user: str | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(input, str):
        inp: str | list[str] = input
    elif isinstance(input, Sequence) and all(isinstance(t, str) for t in input):
        inp = list(input)
    else:
        raise CalibanError("embeddings input must be a string or a sequence of strings")
    body: dict[str, Any] = {"model": model, "input": inp}
    if dimensions is not None:
        body["dimensions"] = dimensions
    if encoding_format is not None:
        body["encoding_format"] = encoding_format
    if user is not None:
        body["user"] = user
    if extra_body:
        body.update(extra_body)
    return body


def build_rerank_body(
    *,
    model: str,
    query: str,
    documents: Sequence[str],
    top_n: int | None,
    return_documents: bool | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(query, str):
        raise CalibanError("rerank query must be a string")
    if (
        isinstance(documents, str)
        or not isinstance(documents, Sequence)
        or not all(isinstance(d, str) for d in documents)
    ):
        raise CalibanError("rerank documents must be a sequence of strings")
    if len(documents) == 0:
        raise CalibanError("rerank documents must contain at least one document")
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
        raise CalibanError(f"rerank top_n must be an integer >= 1 (got {top_n!r})")
    if return_documents is not None and not isinstance(return_documents, bool):
        raise CalibanError("rerank return_documents must be a bool")
    body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
    if top_n is not None:
        body["top_n"] = top_n
    if return_documents is not None:
        body["return_documents"] = return_documents
    if extra_body:
        body.update(extra_body)
    return body


_COMPLETION = TypeAdapter(ChatCompletion)
_MODEL_LIST = TypeAdapter(ModelList)
_EMBEDDINGS = TypeAdapter(CreateEmbeddingResponse)
_RERANK = TypeAdapter(RerankResponse)


def _completion(resp: httpx.Response) -> ChatCompletion:
    out = parse_response(resp, _COMPLETION)
    out._caliban = CalibanResponseMeta.from_headers(resp.headers)
    return out


def _embeddings(resp: httpx.Response) -> CreateEmbeddingResponse:
    out = parse_response(resp, _EMBEDDINGS)
    out._caliban = CalibanResponseMeta.from_headers(resp.headers)
    return out


def _rerank(resp: httpx.Response) -> RerankResponse:
    out = parse_response(resp, _RERANK)
    out._caliban = CalibanResponseMeta.from_headers(resp.headers)
    return out


# ───────────────────────────── sync ─────────────────────────────


class Completions:
    def __init__(self, http: SyncHTTP) -> None:
        self._http = http

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[False] = False,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[True],
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletionStream: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion | ChatCompletionStream: ...

    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool = False,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion | ChatCompletionStream:
        """Create a chat completion. ``stream=True`` returns a chunk iterator.

        Any extra keyword (``top_p``, ``response_format``, ``seed``...) is passed
        through in the body, as the contract allows additional properties.
        """
        body = build_chat_body(
            model=model,
            messages=messages,
            stream=stream,
            caliban=caliban,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body,
            params=params,
        )
        resp = self._http.request(
            "POST",
            "chat/completions",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            stream=stream,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        if stream:
            return ChatCompletionStream(resp)
        return _completion(resp)


class Chat:
    def __init__(self, http: SyncHTTP) -> None:
        self.completions = Completions(http)


class Models:
    def __init__(self, http: SyncHTTP) -> None:
        self._http = http

    def list(self, *, timeout: TimeoutTypes = None) -> ModelList:
        """``GET /v1/models``. Items carry a ``caliban`` object (kind, family, capabilities,
        trust tier), except the virtual ``caliban/auto`` entry."""
        resp = self._http.request("GET", "models", timeout=timeout)
        return parse_response(resp, _MODEL_LIST)


class Embeddings:
    def __init__(self, http: SyncHTTP) -> None:
        self._http = http

    def create(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None = None,
        encoding_format: Literal["float"] | None = None,
        user: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> CreateEmbeddingResponse:
        """``POST /v1/embeddings`` (OpenAI shape). ``input`` is one text or a list of texts.

        Texts sent outside the trust boundary are PII-masked. Header metadata is on
        ``.caliban``; ``.vectors`` gives the embeddings in input order.
        """
        body = build_embeddings_body(
            model=model,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
            extra_body=extra_body,
        )
        resp = self._http.request(
            "POST",
            "embeddings",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        return _embeddings(resp)


class Rerank:
    def __init__(self, http: SyncHTTP) -> None:
        self._http = http

    def create(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
        return_documents: bool | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> RerankResponse:
        """``POST /v1/rerank``: score ``documents`` against ``query`` with a rerank model.

        ``results`` keep the server's order (highest ``relevance_score`` first) and each
        ``index`` points into ``documents``. ``return_documents=True`` echoes the original
        text in ``results[i].document.text``. Header metadata is on ``.caliban``.
        """
        body = build_rerank_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            extra_body=extra_body,
        )
        resp = self._http.request(
            "POST",
            "rerank",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        return _rerank(resp)


class Caliban:
    """Synchronous data-plane client (OpenAI-compatible ``/v1``).

    Args:
        api_key: Tenant key (``cal_...``). Defaults to ``$CALIBAN_API_KEY``.
        base_url: Including ``/v1``. Defaults to ``$CALIBAN_BASE_URL`` or
            ``http://localhost:8080/v1``.
        timeout: Seconds or an ``httpx.Timeout``; for streams it bounds the gap
            between chunks. Default 600s total, 10s connect.
        max_retries: Retries on 429/502/503 and connect failures (default 2).
        http_client: Bring your own ``httpx.Client`` (custom CA, proxies, mTLS).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutTypes = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        key, url = _resolve(api_key, base_url)
        self._http = SyncHTTP(
            base_url=url,
            token=key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.chat = Chat(self._http)
        self.models = Models(self._http)
        self.embeddings = Embeddings(self._http)
        self.rerank = Rerank(self._http)

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Caliban:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ───────────────────────────── async ─────────────────────────────


class AsyncCompletions:
    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[False] = False,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[True],
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> AsyncChatCompletionStream: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion | AsyncChatCompletionStream: ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool = False,
        caliban: CalibanParam | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        **params: Any,
    ) -> ChatCompletion | AsyncChatCompletionStream:
        body = build_chat_body(
            model=model,
            messages=messages,
            stream=stream,
            caliban=caliban,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body,
            params=params,
        )
        resp = await self._http.request(
            "POST",
            "chat/completions",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            stream=stream,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        if stream:
            return AsyncChatCompletionStream(resp)
        return _completion(resp)


class AsyncChat:
    def __init__(self, http: AsyncHTTP) -> None:
        self.completions = AsyncCompletions(http)


class AsyncModels:
    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http

    async def list(self, *, timeout: TimeoutTypes = None) -> ModelList:
        resp = await self._http.request("GET", "models", timeout=timeout)
        return parse_response(resp, _MODEL_LIST)


class AsyncEmbeddings:
    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http

    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None = None,
        encoding_format: Literal["float"] | None = None,
        user: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> CreateEmbeddingResponse:
        """Async ``POST /v1/embeddings``. See :meth:`Embeddings.create`."""
        body = build_embeddings_body(
            model=model,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
            extra_body=extra_body,
        )
        resp = await self._http.request(
            "POST",
            "embeddings",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        return _embeddings(resp)


class AsyncRerank:
    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http

    async def create(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
        return_documents: bool | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> RerankResponse:
        """Async ``POST /v1/rerank``. See :meth:`Rerank.create`."""
        body = build_rerank_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            extra_body=extra_body,
        )
        resp = await self._http.request(
            "POST",
            "rerank",
            json=body,
            headers=extra_headers,
            timeout=timeout,
            retry_statuses=DEFAULT_RETRY_STATUSES,
            idempotent=False,
        )
        return _rerank(resp)


class AsyncCaliban:
    """Asynchronous data-plane client. Same arguments as :class:`Caliban`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: TimeoutTypes = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        key, url = _resolve(api_key, base_url)
        self._http = AsyncHTTP(
            base_url=url,
            token=key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.chat = AsyncChat(self._http)
        self.models = AsyncModels(self._http)
        self.embeddings = AsyncEmbeddings(self._http)
        self.rerank = AsyncRerank(self._http)

    @property
    def base_url(self) -> str:
        return self._http.base_url

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncCaliban:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
