"""Node runs on the data plane (``POST /v1/nodes/{name}/runs``, ``/v1/runs``)."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter

from . import _base
from ._base import AsyncHTTP, SyncHTTP, TimeoutTypes, parse_response
from .errors import CalibanError
from .types import (
    CalibanResponseMeta,
    ChatMessage,
    NodeRun,
    NodeRunList,
    RunStatus,
    Untrusted,
)

__all__ = ["AsyncRuns", "Runs", "build_run_body"]

_RUN = TypeAdapter(NodeRun)
_RUN_LIST = TypeAdapter(NodeRunList)
_UNTRUSTED: TypeAdapter[Untrusted] = TypeAdapter(Untrusted)


def _seg(value: str, what: str) -> str:
    if not value:
        raise CalibanError(f"{what} must not be empty")
    return quote(value, safe="")


def build_run_body(
    *,
    input: Any,
    messages: Sequence[ChatMessage | Mapping[str, Any]] | None,
    version: int | None,
    async_: bool | None,
    wait_s: float | None,
    budget: Mapping[str, Any] | None,
    untrusted: Untrusted | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The body of ``POST /v1/nodes/{name}/runs``: exactly one of ``input`` and ``messages``.
    ``untrusted`` (``True``, or up to 64 JSON pointers) is checked before sending."""
    if (input is None) == (messages is None):
        raise CalibanError("pass exactly one of input= or messages=")
    body: dict[str, Any] = {}
    if messages is not None:
        body["chat"] = {
            "messages": [
                m.model_dump(mode="json", exclude_none=True)
                if isinstance(m, ChatMessage)
                else dict(m)
                for m in messages
            ]
        }
    else:
        body["input"] = input
    if version is not None:
        body["version"] = version
    if async_ is not None:
        body["async"] = async_
    if wait_s is not None:
        if wait_s < 0:
            raise CalibanError("wait_s must be >= 0")
        body["wait_s"] = wait_s
    if budget is not None:
        body["budget"] = dict(budget)
    if untrusted is not None:
        body["untrusted"] = _UNTRUSTED.validate_python(untrusted)
    if extra_body:
        body.update(extra_body)
    return body


def _headers(
    extra_headers: Mapping[str, str] | None, idempotency_key: str | None
) -> dict[str, str] | None:
    if idempotency_key is None:
        return dict(extra_headers) if extra_headers else None
    return {**(extra_headers or {}), "Idempotency-Key": idempotency_key}


def _list_params(
    node: str | None,
    version: int | None,
    status: RunStatus | Sequence[RunStatus] | None,
    created_after: datetime | str | None,
    created_before: datetime | str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    if limit is not None and not 1 <= limit <= 200:
        raise CalibanError("limit must be between 1 and 200")
    return {
        "node": node,
        "version": version,
        "status": status if status is None or isinstance(status, str) else ",".join(status),
        "created_after": (
            created_after.isoformat() if isinstance(created_after, datetime) else created_after
        ),
        "created_before": (
            created_before.isoformat() if isinstance(created_before, datetime) else created_before
        ),
        "limit": limit,
        "cursor": cursor,
    }


def _run(resp: httpx.Response) -> NodeRun:
    out = parse_response(resp, _RUN)
    out._caliban = CalibanResponseMeta.from_headers(resp.headers)
    return out


class Runs:
    """Node runs: start one, follow it, answer its human steps, cancel it.

    The API key must be allowed to run the node (its node allowlist). Streaming a run's events
    (``stream: true``, ``GET /v1/runs/{id}/events``) is not wrapped yet: poll :meth:`get` or
    use :meth:`wait`.
    """

    def __init__(self, http: SyncHTTP) -> None:
        self._http = http

    def create(
        self,
        node: str,
        *,
        input: Any = None,
        messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None,
        version: int | None = None,
        async_: bool | None = None,
        wait_s: float | None = None,
        budget: Mapping[str, Any] | None = None,
        untrusted: Untrusted | None = None,
        idempotency_key: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> NodeRun:
        """``POST /v1/nodes/{node}/runs``: run the node's live version (or ``version``).

        Pass ``input`` (checked against the node's ``prompt.input_schema``) or chat
        ``messages``. Sync by default: returns when the run ends or waits for a human, or
        after the server's wait (``wait_s``) with the run so far; check ``status`` (or
        ``done``). ``async_=True`` returns the pending run at once. ``budget`` can only lower
        the version's (``steps``, ``tokens``, ``usd``, ``wall_clock_s``). ``untrusted`` marks
        caller data (``True``, or JSON pointers such as ``["/email/body"]``) with the
        ``caller`` taint label. ``idempotency_key`` makes a retry return the run the first
        request created (and lets the SDK retry on 502). Header metadata is on ``.caliban``.
        """
        body = build_run_body(
            input=input,
            messages=messages,
            version=version,
            async_=async_,
            wait_s=wait_s,
            budget=budget,
            untrusted=untrusted,
            extra_body=extra_body,
        )
        resp = self._http.request(
            "POST",
            f"nodes/{_seg(node, 'node')}/runs",
            json=body,
            headers=_headers(extra_headers, idempotency_key),
            timeout=timeout,
        )
        return _run(resp)

    def get(self, run_id: str, *, timeout: TimeoutTypes = None) -> NodeRun:
        """``GET /v1/runs/{run_id}``: status, output, the awaited question, budget, usage
        and the step timeline (``steps``)."""
        return _run(self._http.request("GET", f"runs/{_seg(run_id, 'run_id')}", timeout=timeout))

    def list(
        self,
        *,
        node: str | None = None,
        version: int | None = None,
        status: RunStatus | Sequence[RunStatus] | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: TimeoutTypes = None,
    ) -> NodeRunList:
        """``GET /v1/runs``: one page of run summaries (no content, no steps), newest first,
        of the nodes the key may run. Page with ``cursor`` (``next_cursor``)."""
        params = _list_params(node, version, status, created_after, created_before, limit, cursor)
        resp = self._http.request("GET", "runs", params=params, timeout=timeout)
        return parse_response(resp, _RUN_LIST)

    def input(
        self, run_id: str, answer: Any, *, step: str | None = None, timeout: TimeoutTypes = None
    ) -> NodeRun:
        """``POST /v1/runs/{run_id}/input``: answer the human step the run waits for
        (``step`` defaults to ``awaiting.step``) and resume it. For a ``<step>@approve`` step
        answer ``{"approve": True}`` or ``{"approve": False}``. Raises
        :class:`~caliban.ConflictError` (``run_not_awaiting_input``) otherwise."""
        body: dict[str, Any] = {"answer": answer}
        if step is not None:
            body["step"] = step
        path = f"runs/{_seg(run_id, 'run_id')}/input"
        return _run(self._http.request("POST", path, json=body, timeout=timeout))

    def cancel(self, run_id: str, *, timeout: TimeoutTypes = None) -> NodeRun:
        """``POST /v1/runs/{run_id}/cancel``: a waiting run is cancelled at once; a running
        one stops at its next step (``cancel_requested``). Raises
        :class:`~caliban.ConflictError` (``run_already_ended``) if it ended otherwise."""
        path = f"runs/{_seg(run_id, 'run_id')}/cancel"
        return _run(self._http.request("POST", path, timeout=timeout))

    def wait(
        self,
        run_id: str,
        *,
        poll_s: float = 1.0,
        timeout_s: float | None = None,
        until_input: bool = True,
    ) -> NodeRun:
        """Poll :meth:`get` until the run ends, or (``until_input``, the default) waits for
        a human (``input_required``). Raises :class:`~caliban.CalibanError` after
        ``timeout_s``; the run keeps going (:meth:`cancel` stops it)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            run = self.get(run_id)
            if run.done or (until_input and run.status == "input_required"):
                return run
            if deadline is not None and time.monotonic() >= deadline:
                raise CalibanError(f"run {run_id} still {run.status} after {timeout_s:g}s")
            _base._sleep(poll_s)


class AsyncRuns:
    """Asynchronous :class:`Runs`."""

    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http

    async def create(
        self,
        node: str,
        *,
        input: Any = None,
        messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None,
        version: int | None = None,
        async_: bool | None = None,
        wait_s: float | None = None,
        budget: Mapping[str, Any] | None = None,
        untrusted: Untrusted | None = None,
        idempotency_key: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: TimeoutTypes = None,
    ) -> NodeRun:
        """See :meth:`Runs.create`."""
        body = build_run_body(
            input=input,
            messages=messages,
            version=version,
            async_=async_,
            wait_s=wait_s,
            budget=budget,
            untrusted=untrusted,
            extra_body=extra_body,
        )
        resp = await self._http.request(
            "POST",
            f"nodes/{_seg(node, 'node')}/runs",
            json=body,
            headers=_headers(extra_headers, idempotency_key),
            timeout=timeout,
        )
        return _run(resp)

    async def get(self, run_id: str, *, timeout: TimeoutTypes = None) -> NodeRun:
        """See :meth:`Runs.get`."""
        path = f"runs/{_seg(run_id, 'run_id')}"
        return _run(await self._http.request("GET", path, timeout=timeout))

    async def list(
        self,
        *,
        node: str | None = None,
        version: int | None = None,
        status: RunStatus | Sequence[RunStatus] | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: TimeoutTypes = None,
    ) -> NodeRunList:
        """See :meth:`Runs.list`."""
        params = _list_params(node, version, status, created_after, created_before, limit, cursor)
        resp = await self._http.request("GET", "runs", params=params, timeout=timeout)
        return parse_response(resp, _RUN_LIST)

    async def input(
        self, run_id: str, answer: Any, *, step: str | None = None, timeout: TimeoutTypes = None
    ) -> NodeRun:
        """See :meth:`Runs.input`."""
        body: dict[str, Any] = {"answer": answer}
        if step is not None:
            body["step"] = step
        path = f"runs/{_seg(run_id, 'run_id')}/input"
        return _run(await self._http.request("POST", path, json=body, timeout=timeout))

    async def cancel(self, run_id: str, *, timeout: TimeoutTypes = None) -> NodeRun:
        """See :meth:`Runs.cancel`."""
        path = f"runs/{_seg(run_id, 'run_id')}/cancel"
        return _run(await self._http.request("POST", path, timeout=timeout))

    async def wait(
        self,
        run_id: str,
        *,
        poll_s: float = 1.0,
        timeout_s: float | None = None,
        until_input: bool = True,
    ) -> NodeRun:
        """See :meth:`Runs.wait`."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            run = await self.get(run_id)
            if run.done or (until_input and run.status == "input_required"):
                return run
            if deadline is not None and time.monotonic() >= deadline:
                raise CalibanError(f"run {run_id} still {run.status} after {timeout_s:g}s")
            await _base._async_sleep(poll_s)
