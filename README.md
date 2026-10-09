<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/thecalibanproject/website/main/public/brand/logo-white.svg">
  <img alt="Caliban" src="https://raw.githubusercontent.com/thecalibanproject/website/main/public/brand/logo.svg" width="200">
</picture>

# Caliban Python SDK

Typed Python client for the Caliban AI gateway: OpenAI-compatible inference with Caliban extensions, the admin API, node authoring and an eval harness.

[Docs](https://github.com/thecalibanproject/docs) · [Core](https://github.com/thecalibanproject/core) · [TypeScript SDK](https://github.com/thecalibanproject/sdk-typescript)

## What it is

[Caliban](https://github.com/thecalibanproject/core) is a sovereign AI gateway. It gives a company a single OpenAI- and Anthropic-compatible endpoint and handles:

- intent classification and routing (`caliban/auto` picks the model);
- PII screening and pseudonymisation before anything reaches an outside provider;
- an ontology layer that compiles typed queries (CQIR) to MongoDB pipelines, or to SQL over a CDC replica;
- exact and semantic caching;
- agents ("nodes");
- open-weight models (Qwen and others) served on your own hardware.

Caliban is BYOK only: you bring your own provider keys or local model endpoints, and upstream keys are never pooled. It can run fully on-prem with zero egress. The core is a single Rust binary.

This package (distribution `caliban-sdk`, import name `caliban`) covers four surfaces:

| Surface | Where | In this SDK |
|---|---|---|
| Data plane (OpenAI-compatible) | `http://localhost:8080/v1` | `Caliban`, `AsyncCaliban` |
| Control plane (admin API) | `http://localhost:8081/api/v1` | `CalibanAdmin` |
| Node authoring | `core/schemas/node.schema.json` | `caliban.nodes` |
| Evals | golden sets, pass^k | `caliban.evals`, the `caliban-eval` CLI |

It needs Python 3.11 or newer. Runtime dependencies are `httpx`, `pydantic>=2` and `pyyaml`. The SDK sends no telemetry and makes no network calls except to the URLs you configure. The models are hand-written to match the [core API contract](https://github.com/thecalibanproject/core/blob/main/api/openapi.yaml) (v0.1.0) and ship with type information (`py.typed`).

**Status:** Caliban is in active development with design partners. The SDK is at version 0.1.0 and its API may still change.

## Install

`caliban-sdk` is not published to PyPI yet. Install it from GitHub:

```bash
pip install git+https://github.com/thecalibanproject/sdk-python
```

This also installs the `caliban-eval` command. For development, see [Development and tests](#development-and-tests).

## Quick start

You need a running Caliban gateway and a tenant API key (`cal_...`).

```python
from caliban import Caliban

client = Caliban(api_key="cal_...", base_url="http://localhost:8080/v1")
# or set CALIBAN_API_KEY / CALIBAN_BASE_URL and call Caliban()

resp = client.chat.completions.create(
    model="caliban/auto",
    messages=[{"role": "user", "content": "Which vendors are overdue?"}],
    caliban={"pii": "reversible", "cache": "semantic", "datasources": ["sales"]},
)
print(resp.text)
print(resp.caliban.routed_model, resp.caliban.cache, resp.caliban.pii_entities)
```

## Core concepts

### The `caliban` request extension

Every data-plane request body can carry a `caliban` object. Clients that do not know about it simply leave it out, so the gateway stays wire-compatible with OpenAI. In this SDK it is the `caliban=` argument, which takes a `CalibanOptions` or a plain dict.

| Field | Values | Meaning |
|---|---|---|
| `pii` | `off`, `mask`, `reversible` | How PII is handled before text leaves your trust boundary. `mask` replaces entities with placeholders such as `[EMAIL]`; `reversible` pseudonymises them and restores the originals in the response. |
| `cache` | `off`, `exact`, `semantic` | Response cache mode. |
| `datasources` | `list[str]` | Datasources (by name) the request may query through the ontology layer. |
| `node` | `str` | Run the request through a named node (agent). |
| `max_cost_usd` | `float`, `>= 0` | Cost ceiling for the request. |
| `reasoning` | `off`, `low`, `medium`, `high` | Reasoning effort, translated for the model family (see [Reasoning models](#reasoning-models)). |
| `zdr` | `bool` | Zero data retention. |
| `trace_id` | `str` | Your own correlation id. |

```python
from caliban import CalibanOptions

resp = client.chat.completions.create(
    model="caliban/auto",
    messages=[{"role": "user", "content": "Which vendors are overdue?"}],
    caliban=CalibanOptions(
        pii="reversible",
        cache="semantic",
        datasources=["erp"],
        node="invoice-triage",
        max_cost_usd=0.05,
        reasoning="low",
        zdr=True,
        trace_id="abc123",
    ),
    temperature=0,
    response_format={"type": "json_object"},  # other OpenAI params pass straight through
)
```

Unset fields are dropped, so the gateway's defaults apply (in the API contract: `pii="reversible"`, `cache="exact"`, `zdr=False`). Unknown keys and a negative `max_cost_usd` raise a pydantic `ValidationError` before anything is sent, so typos surface instead of silently dropping a PII or ZDR policy. To send fields this SDK does not know about yet, use `extra_body=`.

### Response metadata

The gateway reports what it did in `x-caliban-*` response headers. Completions, embeddings and rerank results expose them as `.caliban`, a `CalibanResponseMeta`:

| Header | Attribute | Notes |
|---|---|---|
| `x-caliban-request-id` | `request_id` | Quote it in bug reports. |
| `x-caliban-routed-model` | `routed_model` | The model the router actually used (useful with `caliban/auto`). |
| `x-caliban-cache` | `cache` | `hit`, `miss` or `bypass`. |
| `x-caliban-pii-entities` | `pii_entities` | Number of PII entities detected and protected. |
| `x-caliban-cost-usd` | `cost_usd` | Non-streaming responses only; `None` when the model has no price. |

`headers` holds every `x-caliban-*` header, lower-cased. Response models allow extra fields, so provider-specific fields survive (see `resp.model_extra`).

### Reasoning models

`caliban.reasoning` is translated for the model family: Qwen3's `enable_thinking` switch, or `reasoning_effort` for models that take one. `off` disables thinking on hybrid models. The thinking text comes back in `reasoning_content`, separate from the answer in `content`; Caliban also moves inline `<think>…</think>` blocks there.

```python
msgs = [{"role": "user", "content": "Is 3599 prime?"}]

resp = client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, caliban={"reasoning": "high"}
)
print(resp.reasoning_text)  # message.reasoning_content, or .reasoning on some servers
print(resp.text)            # the answer
```

See [Streaming](#streaming) for separating reasoning from content in a stream.

## API reference highlights

### `Caliban` and `AsyncCaliban` (data plane)

```python
Caliban(*, api_key=None, base_url=None, timeout=None, max_retries=2,
        default_headers=None, http_client=None)
```

| Argument | Default | Notes |
|---|---|---|
| `api_key` | `$CALIBAN_API_KEY` | Tenant key (`cal_...`). Required. |
| `base_url` | `$CALIBAN_BASE_URL` or `http://localhost:8080/v1` | Includes `/v1`. |
| `timeout` | 600 s total, 10 s connect | A float or an `httpx.Timeout`. |
| `max_retries` | `2` | See [Retries](#retries). |
| `default_headers` | none | Sent on every request. |
| `http_client` | a new `httpx.Client` | Bring your own for a custom CA, a proxy or mTLS. |

`AsyncCaliban` takes the same arguments (with an `httpx.AsyncClient`). Use either as a context manager, or call `close()` / `aclose()`.

| Method | Endpoint | Returns |
|---|---|---|
| `chat.completions.create(...)` | `POST /v1/chat/completions` | `ChatCompletion`, or a `ChatCompletionStream` when `stream=True` |
| `embeddings.create(...)` | `POST /v1/embeddings` | `CreateEmbeddingResponse` |
| `rerank.create(...)` | `POST /v1/rerank` | `RerankResponse` |
| `models.list()` | `GET /v1/models` | `ModelList` |

`chat.completions.create` takes `model`, `messages`, `stream`, `caliban`, `temperature`, `max_tokens` and `tools`. Any other keyword argument (`top_p`, `response_format`, `seed`, ...) is passed through in the body. Every method also accepts `timeout=`, and the `create` methods accept `extra_body=` and `extra_headers=`.

`ChatCompletion` adds two helpers: `.text` (content of the first choice) and `.reasoning_text`.

**Models.** Items from `models.list()` carry `.caliban` (`kind`, `family`, `capabilities`, `trust_tier`). It is `None` on the virtual `caliban/auto` entry.

**Embeddings.**

```python
emb = client.embeddings.create(
    model="local/bge-m3",                         # a model registered with kind="embedding"
    input=["first passage", "second passage"],    # a string or a list of strings
)
vectors = emb.vectors  # list[list[float]] in input order
print(emb.caliban.routed_model, emb.caliban.pii_entities)
```

Inputs sent to a provider outside the trust boundary are PII-masked (`[EMAIL]`, `[PERSON]`, and so on), not pseudonymised, because vectors cannot be rehydrated. `dimensions`, `encoding_format="float"` and `user` are also accepted.

**Rerank.** Score candidate passages against a query with a model registered as `kind="rerank"`, for example Qwen3-Reranker on vLLM or bge-reranker on TEI.

```python
docs = ["Paris is in France.", "Bananas are yellow.", "The Eiffel Tower is in Paris."]
res = client.rerank.create(
    model="local/qwen3-reranker",
    query="Where is the Eiffel Tower?",
    documents=docs,          # at least one string
    top_n=2,                 # optional: keep only the best N
    return_documents=True,   # optional: echo the text in results[i].document.text
)
for r in res.results:  # best first
    print(r.relevance_score, docs[r.index])
```

`results` keep the server's order (highest `relevance_score` first) and `index` points into `documents`. The SDK raises `CalibanError` before sending for a non-string query, an empty or non-string `documents` list, or an invalid `top_n`. Text sent outside the trust boundary is PII-masked; returned documents are always your originals.

The async client has the same methods:

```python
from caliban import AsyncCaliban

async with AsyncCaliban() as client:
    emb = await client.embeddings.create(model="local/bge-m3", input="one passage")
    res = await client.rerank.create(model="local/qwen3-reranker", query="q", documents=docs)
```

### `CalibanAdmin` (control plane)

```python
from caliban import CalibanAdmin
from caliban.nodes import load_node

admin = CalibanAdmin(token="...", base_url="http://localhost:8081")
# or set CALIBAN_ADMIN_TOKEN / CALIBAN_ADMIN_URL; a trailing /api/v1 is accepted
admin.health()

t = admin.tenants.create(name="Acme", pii_default="reversible")
key = admin.api_keys.create(t.id, name="ci")  # key.key is the plaintext, shown once

# BYOK: a provider key, or a keyless on-prem endpoint (vLLM, Ollama, ...)
admin.provider_keys.create(
    t.id,
    kind="openai_compatible",
    label="vllm",
    base_url="http://vllm:8000/v1",
    trust_tier="t0_sovereign",
)
admin.provider_keys.delete(t.id, "pk_123")  # the secret is crypto-shredded server-side

ds = admin.datasources.create(
    tenant_id=t.id,
    kind="mongodb",
    name="sales",
    connection={"uri": "mongodb://mongo:27017", "database": "sales"},
)
admin.datasources.introspect(ds.id)  # starts an introspection job
onto = admin.ontology.get(tenant_id=t.id)
admin.ontology.approve(onto.elements[0].id, note="looks right")  # or .reject / .review(decision=...)

admin.nodes.create(tenant_id=t.id, name="invoice-triage", spec=load_node("triage.yaml"))
admin.nodes.list(tenant_id=t.id)
print(admin.usage.get(tenant_id=t.id, limit=100).totals)
```

| Resource | Methods |
|---|---|
| `tenants` | `list`, `get`, `create` |
| `api_keys` | `list`, `create` |
| `provider_keys` | `list`, `create`, `delete` |
| `models` | `list`, `create`, `delete` |
| `providers` | `list`, `create`, `delete`, `health`, `discover` |
| `datasources` | `list`, `create`, `introspect` |
| `ontology` | `get`, `review`, `approve`, `reject` |
| `nodes` | `list`, `create` (validates `spec` locally first; `validate_spec=False` skips that) |
| `usage` | `get` (`limit` between 1 and 1000) |

The constructor takes `token`, `base_url`, `timeout` (default 30 s), `max_retries`, `default_headers` and `http_client`. GET and DELETE requests retry on 429, 502 and 503. POST requests retry only on 429 and 503, because a 502 may mean the server already acted. The response models live in `caliban.admin.models`.

### Open models on-prem

Caliban can route to open models you serve yourself (vLLM, SGLang, llama.cpp, Ollama, TEI) through any OpenAI-compatible endpoint. Traffic to a `t0_sovereign` server stays inside your network.

```python
admin = CalibanAdmin()

# 1. A shared model server, usable by all tenants (or pass tenants=[...] as an allow-list).
admin.providers.create(
    id="gpu-pool",
    kind="openai_compatible",
    base_url="http://vllm.internal:8000/v1",
    trust_tier="t0_sovereign",
    cache_salt=True,  # per-tenant vLLM prefix-cache isolation
)
print(admin.providers.health("gpu-pool"))  # status='ok' latency_ms=... models=...

# 2. Ask the server what it serves. Caliban suggests catalogue entries (kind, family,
#    reasoning capabilities) for unregistered models. Suggestions are heuristic: review them.
found = admin.providers.discover("gpu-pool")
print(found.available)  # e.g. ['Qwen/Qwen3-8B', 'BAAI/bge-m3']

# 3. Register the ones you want. Keyword arguments override fields of the suggestion.
for s in found.suggested:
    if s.upstream_model.startswith("Qwen/") or s.kind == "embedding":
        admin.models.create(s, context_window=32768)

# Or register a model by hand:
admin.models.create(
    id="local/qwen3-30b",
    provider="gpu-pool",
    upstream_model="Qwen/Qwen3-30B-A3B",
    trust_tier="t0_sovereign",
    kind="chat",
    family="qwen3",
    capabilities={"tools": True, "reasoning": "hybrid", "reasoning_control": "enable_thinking"},
)

# Model ids contain '/'. The SDK keeps the slash unescaped, as the API expects.
admin.models.delete("local/old-model")  # ConflictError (409) if a route still uses it
```

`admin.providers.delete(id)` returns 409 while models still use the server. Background on model choice, serving engines and hardware tiers is in [research note 09](https://github.com/thecalibanproject/docs/blob/main/research/09-open-models-on-prem.md).

### Nodes

`caliban.nodes.NodeSpec` is a pydantic mirror of `core/schemas/node.schema.json`. A copy of the schema ships in the package (`caliban/nodes/node.schema.json`, also returned by `json_schema()`).

```python
from caliban.nodes import NodeValidationError, lint, load_node, validate

spec = load_node("invoice-triage.node.yaml")  # YAML or JSON; raises NodeValidationError listing every issue
validate({"kind": "agent", ...})               # validate a dict or a NodeSpec
for warning in lint(spec):                     # non-fatal checks
    print(warning)
print(spec.to_yaml())
```

`validate` enforces the JSON Schema rules (required fields, enums, budget bounds, tool-ref and scope patterns, strict integers, no unknown top-level keys) and the rules the schema states only in prose:

- `graph` is allowed only for `kind: workflow`, and a workflow needs vertices;
- vertex ids are unique and edges reference existing vertices;
- every loop contains a vertex with `max_iterations`.

`lint` warns about unknown nested keys, MCP tools not pinned with `#sha256:`, node tools not pinned with `@vN`, and write tools without an injection defence. See [`tests/fixtures/invoice-triage.node.yaml`](tests/fixtures/invoice-triage.node.yaml) for a full example. Once created, a node is called from the data plane with `caliban={"node": "invoice-triage"}`. The node design is described in [research note 04](https://github.com/thecalibanproject/docs/blob/main/research/04-agent-orchestration.md).

### Evals (golden sets, pass^k)

`caliban.evals` implements the evaluation loop from [research note 04, section 7](https://github.com/thecalibanproject/docs/blob/main/research/04-agent-orchestration.md#7-evaluation-loop). Each case runs `n` times (default `k`). The report gives pass@1, **pass^k**, pass@k, cost per success, p50/p95 latency and the failures for each case.

```yaml
# golden.yaml
name: invoice-triage-golden
defaults:
  model: caliban/auto          # or pass --model / --node
  node: invoice-triage
  system: Answer tersely.
  temperature: 0
  caliban: { pii: mask }       # merged over the eval default cache: off
cases:
  - id: total
    input: "Total of invoice 42?"           # or messages: [{role, content}, ...]
    expected: "1250.00"                     # shorthand for an exact check
  - id: decision
    messages: [{ role: user, content: "Triage invoice 42. Reply in JSON." }]
    checks:                                 # every check must pass
      - type: json_schema
        schema: { type: object, required: [decision], properties: { decision: { enum: [approve, escalate] } } }
      - { type: contains, value: [approve], case_sensitive: false }
      - { type: regex, pattern: "approve", flags: [IGNORECASE] }
  - id: tone
    input: "Write the vendor a reminder."
    check: { type: llm_judge, rubric: "Polite and under 80 words" }   # needs a judge, see below
```

```bash
caliban-eval validate golden.yaml
caliban-eval run golden.yaml --model caliban/auto --k 3 --base-url http://localhost:8080/v1 --api-key cal_...
caliban-eval run golden.yaml --node invoice-triage --k 3 --trials 8 --concurrency 4 \
    --price-in 0.15 --price-out 0.60 --fail-under 0.9 --out-dir eval-reports
```

`run` writes `eval-reports/<name>.eval.json` and `<name>.eval.md` and prints the Markdown. It exits with `0` on success, `1` when pass^k is below `--fail-under`, and `2` on a usage or dataset error or when every trial errored.

For a case with `n` trials and `c` passes:

| Metric | Formula | Meaning |
|---|---|---|
| pass@1 | `c/n` | Average accuracy. |
| **pass^k** | `C(c,k) / C(n,k)` | Probability that k random trials **all** pass (the reliability metric from τ-bench). |
| pass@k | `1 - C(n-c,k) / C(n,k)` | Probability that at least one of k trials passes. |

Both estimators are unbiased when `n >= k`, so use `--trials` above `k` to reduce variance. Dataset-level values are the mean over cases.

- **Cost per success** is total cost divided by passing trials. Cost comes from `x-caliban-cost-usd` when the gateway sends it, otherwise from usage tokens and `--price-in`/`--price-out` (USD per 1M tokens, given together).
- **Latency p50/p95** uses linear interpolation over all trials that got a response.
- Evals default to `caliban.cache: "off"` so repeated trials are independent. If a trial is still a cache hit, the report warns about it.
- `llm_judge` is a hook, not a built-in grader. Pass `judge=callable` to `run_eval(...)` to grade with a model of your choice. Without a judge, the check is reported as an error, not a pass.

Programmatic use:

```python
from caliban import Caliban
from caliban.evals import load_dataset, run_eval, to_markdown

report = run_eval(load_dataset("golden.yaml"), Caliban(), model="caliban/auto", k=3, trials=5)
print(report.summary.pass_hat_k, to_markdown(report))
```

## Streaming

With `stream=True`, `create()` returns a `ChatCompletionStream`. Header metadata is available as `stream.caliban` before the first chunk. Use the stream as a context manager (or exhaust it) so the connection is released.

```python
with client.chat.completions.create(model="caliban/auto", messages=msgs, stream=True) as stream:
    print("routed to", stream.caliban.routed_model)
    for chunk in stream:  # ChatCompletionChunk
        print(chunk.choices[0].delta.content or "", end="")
```

Instead of iterating chunks you can use:

- `stream.iter_text()`: only the answer text deltas of choice 0;
- `stream.iter_parts()`: `TextPart(kind, text)` items, with `kind` either `"reasoning"` or `"content"`;
- `stream.collect()`: a `StreamText(reasoning, content)` with both strings.

```python
import sys

with client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, stream=True, caliban={"reasoning": "medium"}
) as stream:
    for part in stream.iter_parts():
        print(part.text, end="", file=sys.stderr if part.kind == "reasoning" else sys.stdout)
```

Reasoning deltas are read from `delta.reasoning_content`, falling back to `delta.reasoning`.

Async streams work the same way, and `AsyncChatCompletionStream` has the same `iter_text()`, `iter_parts()` and `collect()` methods:

```python
from caliban import AsyncCaliban

async with AsyncCaliban() as client:
    stream = await client.chat.completions.create(model="caliban/auto", messages=msgs, stream=True)
    async with stream:
        async for chunk in stream:
            ...
```

The SSE parser (`caliban.SSEDecoder`) follows the WHATWG rules: CRLF, CR or LF line endings (including a CRLF pair split across network chunks), UTF-8 characters split across chunks, comment and keep-alive lines, multi-line `data:` fields and a leading BOM. `[DONE]` ends the stream and unknown event types are skipped. An `event: error`, or an `{"error": …}` payload, raises `StreamError`.

## Errors, retries and timeouts

Every error derives from `caliban.CalibanError`. HTTP errors are parsed from the contract's error body (`{"error": {"message", "type", "code"}}`) and expose `.status_code`, `.message`, `.type`, `.code`, `.request_id` and `.body`.

```python
from caliban import PolicyViolationError, RateLimitError, APIStatusError

try:
    client.chat.completions.create(model="caliban/auto", messages=msgs, caliban={"pii": "off"})
except PolicyViolationError as e:
    print("blocked by policy:", e.message, e.code, e.request_id)
except RateLimitError:
    ...
except APIStatusError as e:
    print(e.status_code, e.type, e.message)
```

| Exception | When |
|---|---|
| `BadRequestError` | 400 |
| `AuthenticationError` | 401, or `type=authentication_error` |
| `PermissionDeniedError` | 403 |
| `PolicyViolationError` | `type=policy_violation`, at any status (PII, trust tier, budget) |
| `NotFoundError`, `ConflictError`, `UnprocessableEntityError` | 404, 409, 422 |
| `RateLimitError` | 429, or `type=rate_limited` |
| `UpstreamError` | 502, or `type=upstream_error` (a subclass of `InternalServerError`) |
| `ServiceUnavailableError`, `InternalServerError` | 503, other 5xx |
| `StreamError` | An error event, or malformed JSON, after a stream started |
| `APIResponseValidationError` | A 2xx body (or stream chunk) that does not match the contract |
| `APIConnectionError`, `APITimeoutError` | No HTTP response, or the connection dropped or timed out mid-stream |

All status errors derive from `APIStatusError`, which derives from `APIError`. Client-side problems such as a missing API key or invalid rerank input raise `CalibanError` directly.

### Retries

| Setting | Behaviour |
|---|---|
| `max_retries` | 2. Exponential backoff with jitter (0.5 s, 1 s, ... up to 8 s). `Retry-After` is honoured, capped at 60 s. |
| Retried statuses | 429, 502, 503 on the data plane. (The admin client skips 502 for POST, see above.) |
| Transport errors | Connect failures are always retried. Read timeouts and resets are retried only for idempotent requests (GET, DELETE), never for `POST /chat/completions`. |
| Streams | Retried only before the first byte reaches you. Once iteration has started, a failure raises and is never replayed. |

Data-plane POSTs, including chat completions, are retried on 502. The gateway does not support idempotency keys yet, so if a duplicate upstream call would matter, pass `max_retries=0` and handle `UpstreamError` yourself.

### Timeouts

The default is 600 s total and 10 s to connect on the data plane, and 30 s on the admin client. Pass a float or an `httpx.Timeout` to the constructor, or override per call with `create(..., timeout=)`. For streams the read timeout bounds the gap between chunks.

## Using the stock OpenAI and Anthropic SDKs

You do not need this package to use Caliban. `openai` and `anthropic` are not dependencies of `caliban-sdk`; this section is documentation only.

**OpenAI SDK.** Point `base_url` at the gateway and put the Caliban options in `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="cal_...")  # tenant key

resp = client.chat.completions.create(
    model="caliban/auto",  # the router decides; or a tenant alias / registry id
    messages=[{"role": "user", "content": "Summarise Q3 sales"}],
    extra_body={"caliban": {"pii": "reversible", "cache": "exact", "datasources": ["sales"]}},
)

# Caliban metadata is in the response headers:
raw = client.chat.completions.with_raw_response.create(
    model="caliban/auto", messages=[{"role": "user", "content": "hi"}]
)
print(raw.headers.get("x-caliban-routed-model"), raw.headers.get("x-caliban-cache"))
completion = raw.parse()
```

Streaming with `stream=True` works as usual.

**Anthropic SDK.** The gateway also serves the Anthropic Messages API (`POST /v1/messages`, plus an approximate `POST /v1/messages/count_tokens`) through the same pipeline. Point `base_url` at the gateway root, without `/v1`. The SDK sends the tenant key as `x-api-key`, which Caliban accepts.

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="cal_...")

raw = client.messages.with_raw_response.create(
    model="caliban/auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarise Q3 sales"}],
    extra_body={"caliban": {"pii": "reversible"}},
)
print(raw.headers.get("x-caliban-routed-model"))
message = raw.parse()
```

When the routed model is an Anthropic model the request is passed through natively (`cache_control` breakpoints are kept and `anthropic-beta` is forwarded); otherwise the request and response, including stream events, are translated. `thinking` maps to `caliban.reasoning`. Errors use the Anthropic error shape.

## Configuration

| Env var | Used by | Default |
|---|---|---|
| `CALIBAN_API_KEY` | `Caliban`, `AsyncCaliban`, `caliban-eval` | none (required) |
| `CALIBAN_BASE_URL` | `Caliban`, `AsyncCaliban`, `caliban-eval` | `http://localhost:8080/v1` |
| `CALIBAN_ADMIN_TOKEN` | `CalibanAdmin` | none (required) |
| `CALIBAN_ADMIN_URL` | `CalibanAdmin` | `http://localhost:8081` |

## Development and tests

```bash
git clone https://github.com/thecalibanproject/sdk-python.git
cd sdk-python
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy src          # strict
.venv/bin/pytest
```

The tests use `httpx.MockTransport` and `respx`, so no network access is needed.

Keeping the contract in sync with [core](https://github.com/thecalibanproject/core):

- When `core/api/openapi.yaml` changes, update `src/caliban/types.py` and `src/caliban/admin/models.py`.
- When `core/schemas/node.schema.json` changes, copy it to `src/caliban/nodes/` and update `nodes/spec.py`. With `core` checked out next to this repo (as `../core`), `tests/test_nodes.py` fails if the copies drift; without it, that test is skipped.

## Licence

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 Elie Sfeir.

This SDK is open source. The Caliban gateway and the other Caliban repositories are proprietary and source-available under their own terms.
