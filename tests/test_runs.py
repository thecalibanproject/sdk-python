"""Node runs on the data plane: ``client.runs``."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from caliban import CalibanError, ConflictError, NodeRun, UpstreamError

from .conftest import make_async_client, make_client

TS = "2026-10-15T10:00:00Z"
RUN = {
    "id": "run_1",
    "object": "node.run",
    "node": "triage",
    "version": 3,
    "hash": "sha256:ab",
    "status": "input_required",
    "awaiting": {"step": "write#0@approve", "question": "Allow erp/approve?"},
    "budget": {"steps_limit": 8, "steps_used": 2, "usd": 0.0004},
    "cost_usd": 0.0004,
    "usage": {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120},
    "input_labels": ["caller"],
    "steps": [
        {"id": "classify#0", "vertex": "classify", "kind": "llm", "status": "completed",
         "tokens": 120, "usd": 0.0004, "labels": [], "started_at": TS, "duration_ms": 420},
    ],
    "created_at": TS,
}  # fmt: skip
DONE = {**RUN, "status": "succeeded", "awaiting": None, "output": {"urgency": "high"}}


def test_create_sync_run_with_input_untrusted_and_idempotency_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            202, json=RUN, headers={"x-caliban-request-id": "req_1", "Caliban-Run-Id": "run_1"}
        )

    run = make_client(handler).runs.create(
        "triage",
        input={"email": {"body": "refund me"}},
        version=3,
        wait_s=5,
        budget={"steps": 4},
        untrusted=["/email/body"],
        idempotency_key="order-42",
    )
    req = seen[0]
    assert (req.method, req.url.path) == ("POST", "/v1/nodes/triage/runs")
    assert req.headers["Idempotency-Key"] == "order-42"
    assert json.loads(req.content) == {
        "input": {"email": {"body": "refund me"}},
        "version": 3,
        "wait_s": 5,
        "budget": {"steps": 4},
        "untrusted": ["/email/body"],
    }
    assert isinstance(run, NodeRun) and not run.done
    assert run.awaiting is not None and run.awaiting.step == "write#0@approve"
    assert run.steps[0].duration_ms == 420 and run.budget.steps_used == 2
    assert run.caliban.request_id == "req_1" and run.caliban.run_id == "run_1"


def test_create_with_chat_messages_async_flag_and_checks() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(202, json={**RUN, "status": "pending", "awaiting": None})

    client = make_client(handler)
    client.runs.create("triage", messages=[{"role": "user", "content": "hi"}], async_=True)
    assert seen[0] == {"chat": {"messages": [{"role": "user", "content": "hi"}]}, "async": True}
    with pytest.raises(CalibanError, match="exactly one"):
        client.runs.create("triage")
    with pytest.raises(CalibanError, match="exactly one"):
        client.runs.create("triage", input={}, messages=[])
    with pytest.raises(ValueError, match="pattern"):
        client.runs.create("triage", input={}, untrusted=["email"])
    with pytest.raises(CalibanError, match="node"):
        client.runs.create("", input={})
    assert len(seen) == 1


def test_get_list_input_cancel() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/v1/runs":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "run_1", "object": "node.run", "node": "triage",
                              "version": 3, "status": "succeeded", "cost_usd": 0.001,
                              "invoker": "api_key:ab", "created_at": TS}],
                    "next_cursor": None,
                },
            )  # fmt: skip
        if path.endswith("/input"):
            return httpx.Response(202, json={**RUN, "status": "running", "awaiting": None})
        if path.endswith("/cancel"):
            return httpx.Response(409, json={"error": {"message": "run ended",
                                                       "type": "conflict",
                                                       "code": "run_already_ended"}})  # fmt: skip
        return httpx.Response(200, json=DONE)

    runs = make_client(handler).runs
    assert runs.get("run/1").output == {"urgency": "high"}
    assert seen[-1].url.raw_path == b"/v1/runs/run%2F1"
    page = runs.list(node="triage", status=["succeeded", "failed"], limit=10)
    assert page.data[0].status == "succeeded" and page.next_cursor is None
    assert dict(seen[-1].url.params) == {
        "node": "triage",
        "status": "succeeded,failed",
        "limit": "10",
    }
    resumed = runs.input("run_1", {"approve": True}, step="write#0@approve")
    assert resumed.status == "running"
    assert json.loads(seen[-1].content) == {"answer": {"approve": True}, "step": "write#0@approve"}
    with pytest.raises(ConflictError, match="run ended"):
        runs.cancel("run_1")
    with pytest.raises(CalibanError, match="limit"):
        runs.list(limit=500)


def test_create_not_retried_on_502_without_a_key() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(502, json={"error": {"message": "bad gateway", "type": "x"}})

    client = make_client(handler)
    with pytest.raises(UpstreamError):
        client.runs.create("triage", input={})
    assert len(calls) == 1
    with pytest.raises(UpstreamError):
        client.runs.create("triage", input={}, idempotency_key="k1")
    assert len(calls) == 4  # with the key: retried like a GET (2 retries)


def test_wait_until_done_or_input(no_sleep: list[float]) -> None:
    replies = iter([{**RUN, "status": "running", "awaiting": None}, RUN, DONE])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(replies))

    runs = make_client(handler).runs
    assert runs.wait("run_1", poll_s=0.5).status == "input_required"
    assert no_sleep == [0.5]
    assert runs.wait("run_1", until_input=False).status == "succeeded"


def test_async_runs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=DONE)
        return httpx.Response(200, json={**RUN, "status": "running", "awaiting": None})

    async def go() -> None:
        client = make_async_client(handler)
        run = await client.runs.create("triage", input={"q": "x"})
        assert run.done and run.output == {"urgency": "high"}
        assert (await client.runs.get("run_1")).status == "running"
        assert (await client.runs.cancel("run_1")).status == "succeeded"
        with pytest.raises(CalibanError, match="still running"):
            await client.runs.wait("run_1", timeout_s=0)
        await client.aclose()

    asyncio.run(go())
