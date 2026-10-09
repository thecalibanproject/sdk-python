"""Shared HTTP plumbing: auth headers, timeouts, retry with backoff, error mapping.

Retry policy
------------
* Retries happen only *before* a response body is handed to the caller. A stream
  that has started yielding chunks is never retried.
* Retryable statuses default to 429, 502 and 503. ``Retry-After`` (seconds or an
  HTTP date) is honoured, capped at :data:`MAX_RETRY_AFTER`.
* Connection failures where the request provably never reached the server
  (``ConnectError``/``ConnectTimeout``) are always retryable. Other transport
  errors (read timeouts, resets) are retried only for idempotent requests.
"""

from __future__ import annotations

import asyncio
import email.utils
import platform
import random
import time
from collections.abc import Mapping
from typing import Any, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from ._version import __version__
from .errors import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    CalibanError,
    make_status_error,
)

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503})
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0
MAX_RETRY_AFTER = 60.0

TimeoutTypes = float | httpx.Timeout | None
T = TypeVar("T")


def _sleep(seconds: float) -> None:  # patched in tests
    time.sleep(seconds)


async def _async_sleep(seconds: float) -> None:  # patched in tests
    await asyncio.sleep(seconds)


def user_agent() -> str:
    return f"caliban-sdk-python/{__version__} python/{platform.python_version()}"


def to_timeout(value: TimeoutTypes) -> httpx.Timeout:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, httpx.Timeout):
        return value
    return httpx.Timeout(float(value))


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        secs = float(value)
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        secs = dt.timestamp() - time.time()
    if secs < 0:
        return 0.0
    return min(secs, MAX_RETRY_AFTER)


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """Delay before retry number ``attempt`` (0-based): exponential with jitter."""
    ra = parse_retry_after(retry_after)
    if ra is not None:
        return ra
    base = min(INITIAL_BACKOFF * float(1 << min(attempt, 30)), MAX_BACKOFF)
    return base * (1.0 - 0.25 * random.random())


def parse_response(resp: httpx.Response, adapter: TypeAdapter[T]) -> T:
    """Validate a JSON body, raising :class:`APIResponseValidationError` on mismatch."""
    try:
        return adapter.validate_json(resp.content)
    except ValidationError as exc:
        raise APIResponseValidationError(
            f"Response from {resp.request.method} {resp.request.url.path} does not match "
            f"the contract: {exc.error_count()} error(s): {exc.errors()[0]['msg']}",
            type="invalid_response",
            request_id=resp.headers.get("x-caliban-request-id"),
            body=resp.text,
        ) from exc


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


class _HTTPBase:
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout: TimeoutTypes,
        max_retries: int,
        default_headers: Mapping[str, str] | None,
    ) -> None:
        if max_retries < 0:
            raise CalibanError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.timeout = to_timeout(timeout)
        self.max_retries = max_retries
        headers = {"User-Agent": user_agent(), "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers.update(default_headers or {})
        self.headers = headers

    def _build(
        self,
        client: httpx.Client | httpx.AsyncClient,
        method: str,
        path: str,
        *,
        json: Any,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout: TimeoutTypes,
        stream: bool,
    ) -> httpx.Request:
        h = dict(self.headers)
        if stream:
            h["Accept"] = "text/event-stream"
        h.update(headers or {})
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        return client.build_request(
            method,
            join_url(self.base_url, path),
            json=json,
            params=clean_params or None,
            headers=h,
            timeout=to_timeout(timeout) if timeout is not None else self.timeout,
        )

    @staticmethod
    def _retryable_transport_error(exc: httpx.TransportError, idempotent: bool) -> bool:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return True
        return idempotent

    @staticmethod
    def _wrap_transport_error(exc: httpx.TransportError, request: httpx.Request) -> CalibanError:
        if isinstance(exc, httpx.TimeoutException):
            return APITimeoutError(f"Request timed out: {exc}", request=request)
        return APIConnectionError(f"Connection error: {exc}", request=request)


class SyncHTTP(_HTTPBase):
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout: TimeoutTypes = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._owns_client = http_client is None
        self.client = http_client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        stream: bool = False,
        retry_statuses: frozenset[int] = DEFAULT_RETRY_STATUSES,
        idempotent: bool | None = None,
    ) -> httpx.Response:
        """Send with retries. Returns a 2xx response (open, unread if ``stream``)."""
        if idempotent is None:
            idempotent = method.upper() in {"GET", "HEAD", "OPTIONS", "DELETE", "PUT"}
        attempt = 0
        while True:
            req = self._build(
                self.client,
                method,
                path,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            try:
                resp = self.client.send(req, stream=True)
            except httpx.TransportError as exc:
                if attempt < self.max_retries and self._retryable_transport_error(exc, idempotent):
                    _sleep(backoff_delay(attempt))
                    attempt += 1
                    continue
                raise self._wrap_transport_error(exc, req) from exc

            if resp.is_success:
                if not stream:
                    try:
                        resp.read()
                    except httpx.TransportError as exc:
                        raise self._wrap_transport_error(exc, req) from exc
                    finally:
                        resp.close()
                return resp

            try:
                body = resp.read()
            except httpx.TransportError:
                body = b""
            finally:
                resp.close()
            if resp.status_code in retry_statuses and attempt < self.max_retries:
                _sleep(backoff_delay(attempt, resp.headers.get("retry-after")))
                attempt += 1
                continue
            raise make_status_error(resp, body)


class AsyncHTTP(_HTTPBase):
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout: TimeoutTypes = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            token=token,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._owns_client = http_client is None
        self.client = http_client or httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
        stream: bool = False,
        retry_statuses: frozenset[int] = DEFAULT_RETRY_STATUSES,
        idempotent: bool | None = None,
    ) -> httpx.Response:
        if idempotent is None:
            idempotent = method.upper() in {"GET", "HEAD", "OPTIONS", "DELETE", "PUT"}
        attempt = 0
        while True:
            req = self._build(
                self.client,
                method,
                path,
                json=json,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            try:
                resp = await self.client.send(req, stream=True)
            except httpx.TransportError as exc:
                if attempt < self.max_retries and self._retryable_transport_error(exc, idempotent):
                    await _async_sleep(backoff_delay(attempt))
                    attempt += 1
                    continue
                raise self._wrap_transport_error(exc, req) from exc

            if resp.is_success:
                if not stream:
                    try:
                        await resp.aread()
                    except httpx.TransportError as exc:
                        raise self._wrap_transport_error(exc, req) from exc
                    finally:
                        await resp.aclose()
                return resp

            try:
                body = await resp.aread()
            except httpx.TransportError:
                body = b""
            finally:
                await resp.aclose()
            if resp.status_code in retry_statuses and attempt < self.max_retries:
                await _async_sleep(backoff_delay(attempt, resp.headers.get("retry-after")))
                attempt += 1
                continue
            raise make_status_error(resp, body)
