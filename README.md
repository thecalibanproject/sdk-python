# caliban-sdk (Python)

Python SDK for **Caliban**, the on-prem-capable, BYOK AI gateway.

| Surface | Port | Paths | In this SDK |
|---|---|---|---|
| Data plane (OpenAI-compatible) | `8080` | `/v1/*` | `Caliban`, `AsyncCaliban` |
| Control plane (admin) | `8081` | `/api/v1/*` | `CalibanAdmin` |
| Node authoring | — | `core/schemas/node.schema.json` | `caliban.nodes` |
| Evals | — | golden sets, pass^k | `caliban.evals`, `caliban-eval` CLI |

The contract is `core/api/openapi.yaml` (v0.1.0). The models in this repo are hand-written to match it.

- Package: `caliban-sdk`
- Import name: `caliban`
- Python: 3.11 or newer
- Runtime dependencies: `httpx`, `pydantic>=2` and `pyyaml`. The SDK sends no telemetry and makes no network calls except to the URLs you configure.

> **Licence:** [Apache-2.0](./LICENSE). Copyright 2026 Elie Sfeir.

## Install (development)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

---

## 1. Use the official `openai` package (no Caliban SDK required)

Caliban's data plane speaks the OpenAI API, so any OpenAI client works once you point `base_url` at it. The Caliban-specific options go in `extra_body`. Clients that don't know about them simply leave them out.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="cal_...")  # tenant key

resp = client.chat.completions.create(
    model="caliban/auto",  # router decides; or a tenant alias / registry id
    messages=[{"role": "user", "content": "Summarise Q3 sales"}],
    extra_body={"caliban": {"pii": "reversible", "cache": "exact", "datasources": ["sales_dw"]}},
)

# Caliban metadata arrives in response headers:
raw = client.chat.completions.with_raw_response.create(
    model="caliban/auto", messages=[{"role": "user", "content": "hi"}]
)
print(raw.headers.get("x-caliban-routed-model"), raw.headers.get("x-caliban-cache"))
completion = raw.parse()
```

Streaming with `stream=True` works as normal. This section is documentation only: `openai` is not a dependency of this package.

## 2. Native client

The native client adds typed Caliban options, parsed response metadata, typed errors, and retries that never replay a stream once it has started.

```python
from caliban import Caliban, CalibanOptions

client = Caliban(api_key="cal_...", base_url="http://localhost:8080/v1")
# or set CALIBAN_API_KEY / CALIBAN_BASE_URL and call Caliban()

resp = client.chat.completions.create(
    model="caliban/auto",
    messages=[{"role": "user", "content": "Which vendors are overdue?"}],
    caliban=CalibanOptions(
        pii="reversible",  # off | mask | reversible
        cache="semantic",  # off | exact | semantic
        datasources=["erp"],
        node="invoice-triage",
        max_cost_usd=0.05,
        reasoning="low",  # off | low | medium | high (reasoning models only)
        zdr=True,
        trace_id="abc123",
    ),
    temperature=0,
    response_format={"type": "json_object"},  # other OpenAI params pass straight through
)
print(resp.text)
print(
    resp.caliban.request_id,
    resp.caliban.routed_model,
    resp.caliban.cache,
    resp.caliban.pii_entities,
    resp.caliban.cost_usd,  # x-caliban-cost-usd; None for streams or unpriced models
)
print(client.models.list().data)  # items carry .caliban (kind, family, capabilities, trust_tier)

emb = client.embeddings.create(model="local/bge-m3", input=["a", "b"])  # POST /v1/embeddings
print(emb.vectors, emb.caliban.request_id)
```

- `caliban=` also accepts a plain dict. Unknown keys are rejected so typos surface; use `extra_body=` for fields this SDK doesn't know about yet.
- `resp.caliban` is a `CalibanResponseMeta` built from the `x-caliban-*` headers. `headers` holds all of them verbatim.
- The response models allow extra fields, so provider-specific fields survive (see `resp.model_extra`).

### Timeouts and retries

| Setting | Default |
|---|---|
| `timeout` | 600 s total, 10 s connect. Accepts a float or an `httpx.Timeout`, and can be overridden per call with `create(..., timeout=)`. For streams it bounds the gap between chunks. |
| `max_retries` | 2. Exponential backoff with jitter (0.5 s, 1 s, … up to 8 s). `Retry-After` is honoured, capped at 60 s. |
| Retried statuses | 429, 502, 503 |
| Transport errors | Connect failures are always retried. Read timeouts and resets are retried only for idempotent requests (GET/DELETE), never for `POST /chat/completions`. |
| Streams | Retried only **before** the first byte reaches you. Once iteration has started, a failure raises and is never replayed. |

If you need a custom CA, a proxy or mTLS, pass your own `http_client=httpx.Client(verify=..., proxy=...)`.

### Errors

Every error derives from `caliban.CalibanError`. HTTP errors are parsed from the contract's `Error` body (`{"error": {"message", "type", "code"}}`) and expose `.status_code`, `.message`, `.type`, `.code`, `.request_id` and `.body`.

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
| `StreamError` | An error event received mid-stream |
| `APIResponseValidationError` | A 2xx body that doesn't match the contract |
| `APIConnectionError`, `APITimeoutError` | No HTTP response |

## 3. Streaming

```python
with client.chat.completions.create(model="caliban/auto", messages=msgs, stream=True) as stream:
    print("routed to", stream.caliban.routed_model)  # headers are available before the first chunk
    for chunk in stream:  # ChatCompletionChunk
        print(chunk.choices[0].delta.content or "", end="")

# or yield only the answer text deltas:
for text in client.chat.completions.create(model="m", messages=msgs, stream=True).iter_text():
    ...
```

For reasoning models, `iter_parts()` and `collect()` separate thinking from the answer (see
[Open models on-prem](#5-open-models-on-prem-qwen-etc)).

Async usage:

```python
from caliban import AsyncCaliban

async with AsyncCaliban() as client:
    stream = await client.chat.completions.create(model="caliban/auto", messages=msgs, stream=True)
    async with stream:
        async for chunk in stream:
            ...
```

The SSE parser follows the WHATWG rules:

- lines can end in CRLF, CR or LF, including a CRLF pair split across network chunks
- UTF-8 characters can be split across chunks
- comment and keep-alive lines (`:`) are ignored
- multi-line `data:` fields are joined
- a BOM at the start is ignored
- `[DONE]` ends the stream
- unknown event types are skipped
- `event: error`, or an `{"error": …}` payload, raises `StreamError`

## 4. Admin (control plane)

```python
from caliban import CalibanAdmin

admin = CalibanAdmin(
    token="...", base_url="http://localhost:8081"
)  # or CALIBAN_ADMIN_TOKEN / CALIBAN_ADMIN_URL
admin.health()

t = admin.tenants.create(name="Acme", pii_default="reversible")
key = admin.api_keys.create(t.id, name="ci")  # key.key is the plaintext and is shown ONCE
admin.provider_keys.create(
    t.id,
    kind="openai_compatible",
    label="vllm",
    base_url="http://vllm:8000/v1",
    trust_tier="t0_sovereign",
)
admin.provider_keys.delete(t.id, "pk_123")  # the secret is crypto-shredded server-side
admin.models.list()  # catalogue; also models.create / models.delete, providers.* (below)

ds = admin.datasources.create(tenant_id=t.id, kind="postgres", name="dw", connection={"dsn": "..."})
admin.datasources.introspect(ds.id)  # proposes ontology elements
onto = admin.ontology.get(tenant_id=t.id)
admin.ontology.approve(
    onto.elements[0].id, note="looks right"
)  # or .reject / .review(decision=...)

admin.nodes.create(tenant_id=t.id, name="invoice-triage", spec=load_node("triage.yaml"))
admin.nodes.list(tenant_id=t.id)
admin.usage.get(tenant_id=t.id, limit=100).totals
```

- Models live in `caliban/admin/models.py`.
- GET and DELETE requests retry on 429/502/503. POST requests retry only on 429/503, because a 502 may mean the server already acted.

## 5. Open models on-prem (Qwen etc.)

Caliban can route to open models you serve yourself (vLLM, SGLang, llama.cpp, Ollama, TEI…) through any OpenAI-compatible endpoint. Traffic to a `t0_sovereign` server stays inside your network.

### Register an on-prem server and its models

```python
from caliban import CalibanAdmin

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
#    reasoning capabilities...) for models that are not registered yet. The suggestions
#    are heuristic, so review them before saving.
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

# Model ids contain '/'. The SDK sends them unescaped, as the API expects.
admin.models.delete("local/old-model")  # ConflictError (409) if a route still uses it
```

`admin.providers` also has `list()` and `delete(id)`. A delete returns 409 while models still use the server.

### Reasoning toggle

`caliban.reasoning` (`off | low | medium | high`) is translated for the model family: Qwen3's `enable_thinking` switch, or `reasoning_effort` for models that take one. Thinking comes back in `reasoning_content`, separate from the answer in `content`. Caliban also moves inline `<think>…</think>` blocks into that field.

```python
import sys

from caliban import Caliban

client = Caliban()
msgs = [{"role": "user", "content": "Is 3599 prime?"}]

quick = client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, caliban={"reasoning": "off"}
)

resp = client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, caliban={"reasoning": "high"}
)
print(resp.reasoning_text)  # message.reasoning_content (or .reasoning)
print(resp.text)  # the answer

# Streaming: iter_parts() tags each delta, and collect() returns both strings.
with client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, stream=True, caliban={"reasoning": "medium"}
) as stream:
    for part in stream.iter_parts():  # TextPart(kind, text)
        print(part.text, end="", file=sys.stderr if part.kind == "reasoning" else sys.stdout)

out = client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, stream=True, caliban={"reasoning": "low"}
).collect()
print(out.reasoning, out.content)
```

Streams read `delta.reasoning_content`, and fall back to `delta.reasoning`, which some servers use. `iter_text()` yields only the answer. The async stream has the same `iter_parts()` and `collect()` methods.

### Embeddings

```python
emb = client.embeddings.create(
    model="local/bge-m3",  # a model registered with kind="embedding"
    input=["first passage", "second passage"],  # a string or a list of strings
)
vectors = emb.vectors  # list[list[float]] in input order
print(emb.caliban.routed_model, emb.caliban.pii_entities)

# async (from caliban import AsyncCaliban)
async with AsyncCaliban() as aclient:
    emb = await aclient.embeddings.create(model="local/bge-m3", input="one passage")
```

Inputs sent to a provider outside the trust boundary are PII-masked (`[EMAIL]`, `[PERSON]`…), not pseudonymised, because vectors cannot be rehydrated. Pass server-specific fields with `extra_body=`.

### Rerank

Score candidate passages against a query with a rerank model, e.g. Qwen3-Reranker served by vLLM or bge-reranker served by TEI, registered with `kind="rerank"`.

```python
docs = ["Paris is in France.", "Bananas are yellow.", "The Eiffel Tower is in Paris."]
res = client.rerank.create(
    model="local/qwen3-reranker",  # or e.g. "local/bge-reranker-v2-m3" on TEI
    query="Where is the Eiffel Tower?",
    documents=docs,  # at least one string
    top_n=2,  # optional: keep only the best N
    return_documents=True,  # optional: echo the text in results[i].document.text
)
for r in res.results:  # best first
    print(r.relevance_score, docs[r.index])
print(res.caliban.request_id, res.caliban.routed_model)

# async
async with AsyncCaliban() as aclient:
    res = await aclient.rerank.create(model="local/qwen3-reranker", query="q", documents=docs)
```

`results` keep the server's order (highest `relevance_score` first), and `index` points into `documents`. The SDK raises `CalibanError` before sending for an empty `documents` list, a non-string document or an invalid `top_n`. Text sent outside the trust boundary is PII-masked, and returned documents are always your originals.

## 6. Nodes

`NodeSpec` is a pydantic mirror of `core/schemas/node.schema.json`. A vendored copy of the schema ships at `caliban/nodes/node.schema.json`, and a test fails if it drifts from `../core`.

```python
from caliban.nodes import load_node, validate, lint, NodeValidationError

spec = load_node("invoice-triage.node.yaml")      # YAML or JSON; raises NodeValidationError with every issue
validate({"kind": "agent", ...})                  # validate a dict or a NodeSpec
for warning in lint(spec):                        # non-fatal checks
    print(warning)
print(spec.to_yaml())
```

`validate` enforces:

- the JSON Schema rules: required fields, enums, budget bounds, tool-ref and scope patterns, strict integers, and no unknown top-level keys
- the rules the schema states only in prose:
  - `graph` is allowed only for `kind: workflow`, and a workflow needs vertices
  - vertex ids are unique
  - edges reference existing vertices
  - **every loop contains a vertex with `max_iterations`**

`lint` warns about:

- unknown nested keys
- MCP tools not pinned with `#sha256:`
- node tools not pinned with `@vN`
- write tools without an injection defence

See `tests/fixtures/invoice-triage.node.yaml` for a full example.

## 7. Evals (golden sets, pass^k)

This implements §7 "Evaluation loop" of `docs/research/04-agent-orchestration.md`. Each case runs `n` times (default `k`). The harness reports pass@1, **pass^k**, pass@k, cost per success, p50/p95 latency and the failures for each case.

```yaml
# golden.yaml
name: invoice-triage-golden
defaults:
  model: caliban/auto          # or pass --model / --node
  node: invoice-triage
  system: Answer tersely.
  temperature: 0
  caliban: { pii: mask }       # merged with the eval default cache: off
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
    check: { type: llm_judge, rubric: "Polite and under 80 words" }   # stub, see below
```

```bash
caliban-eval validate golden.yaml
caliban-eval run golden.yaml --model caliban/auto --k 3 --base-url http://localhost:8080/v1 --api-key cal_...
caliban-eval run golden.yaml --node invoice-triage --k 3 --trials 8 --concurrency 4 \
    --price-in 0.15 --price-out 0.60 --fail-under 0.9 --out-dir eval-reports
```

The run writes `eval-reports/<name>.eval.json` and `<name>.eval.md` and prints the Markdown. Exit codes:

- `0`: ok
- `1`: pass^k is below `--fail-under`
- `2`: a usage or dataset error, or every trial errored

### Metrics

For each case with `n` trials and `c` passes:

| Metric | Formula | Meaning |
|---|---|---|
| pass@1 | `c/n` | Average accuracy |
| **pass^k** | `C(c,k) / C(n,k)` | Probability that k random trials **all** pass. This is the reliability metric from τ-bench. |
| pass@k | `1 − C(n−c,k) / C(n,k)` | Probability that at least one of k trials passes |

Both estimators are unbiased when `n ≥ k`, so use `--trials` above `k` to reduce variance. Dataset-level values are the mean over cases.

How the other metrics are computed:

- **Cost per success** = total cost ÷ number of passing trials. Cost comes from the `x-caliban-cost-usd` header (`resp.caliban.cost_usd`), which the gateway sends on non-streaming responses for priced models. Otherwise it is computed from usage tokens and `--price-in/--price-out` (USD per 1M tokens).
- **Latency p50/p95** uses linear interpolation over all trials that got a response.

How trials are run:

- The eval default is `caliban.cache: "off"`, so repeated trials are independent. If any trial is still a cache hit, the report includes a warning.
- `llm_judge` is a **stub**. Pass `judge=callable` to `run_eval(...)` to grade with a model of your choice. Without a judge, the check is reported as an error, not a pass.

Programmatic use:

```python
from caliban import Caliban
from caliban.evals import load_dataset, run_eval, to_markdown

report = run_eval(load_dataset("golden.yaml"), Caliban(), model="caliban/auto", k=3, trials=5)
print(report.summary.pass_hat_k, to_markdown(report))
```

---

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy src          # strict
.venv/bin/pytest
```

The tests use `httpx.MockTransport` and `respx`, so no network access is needed.

**Keeping the contract in sync:**

- When `core/api/openapi.yaml` changes, update `src/caliban/types.py` and `src/caliban/admin/models.py`.
- When `core/schemas/node.schema.json` changes, re-copy it to `src/caliban/nodes/` and update `nodes/spec.py`. `tests/test_nodes.py` checks for drift.
