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
| `pii_floor` | `off`, `mask`, `reversible` | A minimum protection applied to every model class, sovereign and attested models included. It only strengthens: the request gets the stricter of this and `pii` (or the tenant's mode), ordered `off` < `reversible` < `mask`. A node's `guardrails.pii` sets it on the node's calls. |
| `cache` | `off`, `exact`, `semantic` | Response cache mode. `off`: no cache. `exact`: exact cache only. `semantic`: exact, then the semantic cache if the tenant has it on, whatever the temperature. Unset: exact, plus semantic up to the gateway's temperature limit if the tenant has it on (see [Tenant settings](#tenant-settings)). |
| `datasources` | `list[str]` | Datasources (by name) the request may query through the ontology layer. |
| `node` | `str` | Reserved: core parses it but does not act on it. To run a node, use `model="node/<name>"` (see [Nodes](#nodes)). |
| `max_cost_usd` | `float`, `>= 0` | Cost ceiling for the request. |
| `reasoning` | `off`, `low`, `medium`, `high` | Reasoning effort, translated for the model family (see [Reasoning models](#reasoning-models)). |
| `zdr` | `bool` | Zero data retention. |
| `trace_id` | `str` | Your own correlation id. |
| `untrusted` | `bool`, or up to 64 JSON pointers (`"/email/body"`, `""` for all) | With `model="node/<name>"` (or `caliban/auto` handing the request to a node): the node input the caller marks as untrusted data. The run input then carries the `caller` taint label, so a write its data reaches needs an `allow_tainted` entry or a human approval. |

```python
from caliban import CalibanOptions

resp = client.chat.completions.create(
    model="caliban/auto",
    messages=[{"role": "user", "content": "Which vendors are overdue?"}],
    caliban=CalibanOptions(
        pii="reversible",
        cache="semantic",
        datasources=["erp"],
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
| `x-caliban-cache` | `cache` | `hit`, `miss` or `bypass`. `hit` covers both cache tiers. |
| `x-caliban-cache-tier` | `cache_tier` | `"exact"` or `"semantic"`, sent on hits only. `None` otherwise. |
| `x-caliban-intent` | `intent` | The routing decision, parsed into an `IntentDecision` (`intent`, `confidence`, `stage`, `knn_reason`). `None` when absent or malformed. |
| `x-caliban-pii-entities` | `pii_entities` | Number of PII entities detected and protected. |
| `x-caliban-cost-usd` | `cost_usd` | Non-streaming responses only; `None` when the model has no price. For a node answer, the routed models' real cost (the run's `model_cost_usd`). |
| `x-caliban-billed-usd` | `billed_usd` | Non-streaming node answers: what the run is billed (its `cost_usd`). |
| `x-caliban-route` | `route` | `node/<name>@v<N>` when a node answered, `model:<id>` when `caliban/auto` answered with a model for a tenant that maps intents to nodes. |
| `x-caliban-route-fallback` | `route_fallback` | Why `caliban/auto` answered with a model rather than a node (`low_confidence`, `no_node_for_intent`, ...). |
| `Caliban-Run-Id` | `run_id` | The node run that answered. |
| `x-caliban-run-status` | `run_status` | Its status, for example `succeeded` or `input_required`. |

A node answer (`model="node/<name>"`, or `caliban/auto` handing the request to a node) carries the headers of every chat response: `request_id`, `routed_model` (`node/<name>@v<N>`), `cache="bypass"`, and on non-streaming answers `pii_entities` (the entities protected across the run's model calls), `cost_usd` and `billed_usd`.

`headers` holds every `x-caliban-*` header, lower-cased. Response models allow extra fields, so provider-specific fields survive (see `resp.model_extra`).

The gateway sends `x-caliban-intent` as `<intent>;confidence=<0..1>;stage=<rules|knn|keyword>`, plus `;knn=<reason>` when kNN routing was on but did not decide (`timeout`, `embed_error`, `unavailable`, `no_text`, `abstain_oos`, `abstain_confidence`, `abstain_margin`, `abstain_empty`). A request for a named model reports `pinned;confidence=1.000;stage=rules`.

```python
resp = client.chat.completions.create(model="caliban/auto", messages=msgs)
if resp.caliban.cache == "hit":
    print("served from the", resp.caliban.cache_tier, "cache")
route = resp.caliban.intent  # IntentDecision(intent='translate', confidence=0.912, stage='knn')
if route and route.knn_reason:
    print("kNN fell back to", route.stage, "because of", route.knn_reason)
```

Parsing never raises. Unknown `key=value` fields are ignored, so a newer gateway can add fields. A value with no intent name, a missing or out-of-range `confidence`, or a missing or unknown `stage` gives `intent=None`; an unknown cache tier gives `cache_tier=None`. `IntentDecision.parse(value)` works on a raw header string.

### Reasoning models

`caliban.reasoning` is translated for the model family: Qwen3's `enable_thinking` switch, or `reasoning_effort` for models that take one. `off` disables thinking on hybrid models. The thinking text comes back in `reasoning_content`, separate from the answer in `content`; Caliban also moves inline `<think>…</think>` blocks there.

```python
msgs = [{"role": "user", "content": "Is 3599 prime?"}]

resp = client.chat.completions.create(
    model="local/qwen3-8b", messages=msgs, caliban={"reasoning": "high"}
)
print(resp.reasoning_text)  # message.reasoning_content, or .reasoning on some servers
print(resp.text)  # the answer
```

See [Streaming](#streaming) for separating reasoning from content in a stream.

## API reference highlights

### `Caliban` and `AsyncCaliban` (data plane)

```python
# All arguments are keyword-only.
Caliban(
    api_key=None,
    base_url=None,
    timeout=None,
    max_retries=2,
    default_headers=None,
    http_client=None,
)
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
    model="local/bge-m3",  # a model registered with kind="embedding"
    input=["first passage", "second passage"],  # a string or a list of strings
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
    documents=docs,  # at least one string
    top_n=2,  # optional: keep only the best N
    return_documents=True,  # optional: echo the text in results[i].document.text
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
admin.datasources.introspect(ds.id)  # starts the ontology bootstrap: job_id and job
job = admin.datasources.introspection(ds.id)  # the newest job: status, proposed, skipped, version
onto = admin.ontology.get(t.id)
# Approve an element (or .reject / .deprecate / .review(decision=...)); ids are per tenant.
admin.ontology.approve(onto.elements[0].id, tenant_id=t.id, note="looks right")

admin.nodes.create(tenant_id=t.id, name="invoice-triage", spec=load_node("triage.yaml"))
admin.nodes.list(tenant_id=t.id)
print(admin.usage.get(tenant_id=t.id, limit=100).totals)
```

| Resource | Methods |
|---|---|
| `tenants` | `list` (`include_deleted=`), `get`, `create`, `update`, `delete` |
| `api_keys` | `list` (`include_revoked=`), `create`, `revoke` |
| `provider_keys` | `list`, `create`, `delete` |
| `models` | `list`, `create`, `delete` |
| `providers` | `list`, `create`, `delete`, `health`, `discover` |
| `datasources` | `list`, `create`, `delete`, `introspect`, `introspection` |
| `ontology` | `get`, `review`, `approve`, `reject`, `deprecate`, `review_batch`, `propose` |
| `intents` | `list`, `put` (`set` is a deprecated alias), `delete` (the tenant's `caliban/auto` exemplars) |
| `rag` | `models`, `collections.list`, `.create`, `.get`, `.delete`, `documents.list`, `.ingest`, `.get`, `.delete`, `.release` |
| `nodes` | `list`, `create` (validates `spec` locally first; `validate_spec=False` skips that), `delete` |
| `evals` | `get_golden_set`, `put_golden_set`, `delete_golden_set`, `get_policy`, `set_policy`, `start`, `list`, `get`, `cancel`, `wait`, `gate`, `get_key`, `rotate_key` (see [Server-side evals](#server-side-evals-and-the-promotion-gate)) |
| `usage` | `get` (`limit` between 1 and 1000) |

The constructor takes `token`, `base_url`, `timeout` (default 30 s), `max_retries`, `default_headers` and `http_client`. GET and DELETE requests retry on 429, 502 and 503. POST and PATCH requests retry only on 429 and 503, because a 502 may mean the server already acted (see [Retries](#retries)). The response models live in `caliban.admin.models`.

#### Tenant settings

`tenants.update(tenant_id, ...)` sends `PATCH /api/v1/tenants/{tenantId}` and returns the updated `Tenant`. Arguments you leave out keep their value. The change is audited as `tenant.update`, and split-mode routers apply it with their next snapshot. An unknown or deleted tenant raises `NotFoundError`. `tenants.create(...)` takes the same keyword arguments.

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `pii_default` | `"off"`, `"mask"`, `"reversible"` | `"reversible"` | PII mode for requests that do not set `caliban.pii`. |
| `pii_surrogate_scope` | `"tenant"`, `"session"` | `"tenant"` | How reversible PII surrogates are chosen. |
| `pii_on_sovereign` | `bool` | `False` | Also apply the tenant's PII mode to models inside the trust boundary (`t0_sovereign`, `t1_attested`). Off, those models receive the text as is. External models always get the mode, and `caliban.pii_floor` (a node's `guardrails.pii`) applies to every model class. |
| `pii_patterns` | list of `PiiPattern` or mappings (at most 50) | `[]` | The tenant's own L0 detection patterns: `{label, pattern}`. `NATIONAL_ID`, `PHONE`, `EMAIL`, `PERSON`, `ORG`, `LOCATION`, `IBAN` and `CARD` use the built-in surrogates; other labels get shape-preserving ones. `pattern` uses the Rust `regex` syntax (linear time: no backreferences or lookaround) and must not match the empty string. Replaced as a whole; `[]` clears them. |
| `pii_dictionary` | list of `PiiTerm` or mappings (at most 10 000) | `[]` | Whole-word, case-insensitive terms to protect: `{term, label}` (customer names, internal codes). Replaced as a whole; `[]` clears it. Audited as a count of terms, never the terms. |
| `semantic_cache` | `"off"`, `"on"` | `"off"` | Whether the tenant's eligible requests may use the semantic cache. |
| `auto_cache_hit_fraction` | `0` to `1`, or `"default"` | deployment value | Share of the flat `caliban/auto` price billed for the tenant's cache hits. `"default"` clears the override, so the deployment's `[routing] auto_cache_hit_fraction` (0.2 unless configured) applies. |

```python
admin.tenants.update(t.id, semantic_cache="on")
t = admin.tenants.update(t.id, pii_surrogate_scope="session")
print(t.pii_surrogate_scope, t.semantic_cache)  # session on
```

- **`pii_surrogate_scope="tenant"`** (the default): a given value always gets the same surrogate within the tenant (a keyed HMAC per tenant, derived from `CALIBAN_KEK`). That lets pseudonymised requests hit the exact cache. The trade-off is linkability: anyone who can see the pseudonymised traffic (an upstream provider, for example) can tell that two requests or sessions of the tenant mention the same person, even without learning who it is. Surrogates never cross tenants.
- **`pii_surrogate_scope="session"`**: every request gets fresh surrogates, so requests cannot be linked through them. Requests that carry PII then bypass the exact and semantic caches.
- **`pii_patterns` and `pii_dictionary`** are the deterministic first tier of PII detection, compiled per tenant (see [research note 05](https://github.com/thecalibanproject/docs/blob/main/research/05-anonymization-and-privacy.md)): `admin.tenants.update(t.id, pii_patterns=[{"label": "MRN", "pattern": "MRN-[0-9]{6}"}], pii_dictionary=[{"term": "Project Lumen", "label": "ORG"}])`. Labels are checked (`[A-Za-z0-9_]{1,40}`) before sending; the regex is checked by the server.
- **`auto_cache_hit_fraction`**: a cache hit (exact or semantic) calls no model, so a `caliban/auto` request answered from cache is billed this share of the flat price. Values outside 0 to 1 raise `ValueError` before any request. On `Tenant`, `None` means the deployment value applies.
- **`semantic_cache="on"`**: an eligible request may be answered with the response to an earlier, semantically similar request of the same tenant (same model, system prompt, history and parameters). Entries never cross tenants. The deployment must also enable it (`[cache.semantic] enabled`). Hits report `x-caliban-cache: hit` with `x-caliban-cache-tier: semantic`.

These fields are `None` on tenants from an older server.

#### Usage fields

`usage.get()` returns a `UsageReport` with `events` (`UsageEvent`) and `totals` (`UsageTotals`). Beyond the request basics (`model`, tokens, `cache`, `cost_usd`, `latency_ms`, `ts`), an event can carry these optional fields. The server omits them when they do not apply, so they are `None` then, and on events from an older server.

| Field | When present | Meaning |
|---|---|---|
| `cache_tier` | Cache hits | `"exact"` or `"semantic"`. `cache` stays `"hit"`, `"miss"` or `"bypass"`. |
| `tokens_saved` | Cache hits | Tokens not sent upstream: the cached answer's prompt plus completion tokens. |
| `requested_model` | Chat requests | The model the client asked for: `caliban/auto` or a pinned id. |
| `intent_confidence` | Chat requests | Confidence of the intent decision, 0..1. |
| `route_stage` | Chat requests | `"rules"`, `"knn"` or `"keyword"`. |
| `routed_model_cost_usd` | `caliban/auto`, priced model | Real cost of the routed model for this request. |
| `flat_price_usd` | `caliban/auto` | The full flat auto price for the same tokens. On a cache hit, the flat price of the cached answer's tokens (what a miss would have billed). |
| `billed_usd` | `caliban/auto` | What the customer is billed: `flat_price_usd` on a miss, `flat_price_usd` times the tenant's cache-hit fraction on a hit. |
| `saved_usd` | Priced cache hits | What the hit saved: `flat_price_usd - billed_usd` for `caliban/auto`, the avoided model cost for other models. |
| `purpose` | Work Caliban did for the tenant | `"rag.ingest"` (embedding a collection's documents) or `"rag.search"` (a node's retrieval query, its embedding and reranking). `None` on client calls and node model calls. |

`by_purpose` gives the same totals per `purpose` (`UsageByPurpose`), so retrieval spend is separable from the tenant's own calls; it is `None` from an older server.

`totals` adds `semantic_cache_hits` (`cache_hits` counts both tiers), `saved_usd` (what cache hits saved, all models), `auto_requests` (requests for `caliban/auto`) and `auto_cache_hits`, and, summed over the `caliban/auto` events that carry both prices, `flat_price_usd`, `billed_usd`, `auto_saved_usd`, `routed_model_cost_usd` and `margin_usd` (`billed_usd - routed_model_cost_usd`).

```python
totals = admin.usage.get(tenant_id=t.id).totals
print(f"{totals.semantic_cache_hits or 0} of {totals.cache_hits or 0} hits were semantic")
print(f"auto margin: ${totals.margin_usd or 0:.4f} over {totals.auto_requests or 0} requests")
print(f"{totals.auto_cache_hits or 0} auto cache hits saved ${totals.auto_saved_usd or 0:.4f}")
```

#### Ontology bootstrap and review

`datasources.introspect(datasource_id)` starts the ontology bootstrap and returns an `IntrospectJob` (`job_id`, and the `job` as it started): the control plane connects with the datasource's credentials, checks that the principal is read only, samples every collection and writes its proposals (`status="proposed"`) in one ontology version. Elements a human already reviewed, and unchanged ones, are skipped. It is MongoDB only for now. `datasources.introspection(datasource_id)` returns the newest `IntrospectionJob` (`running`, `succeeded` or `failed`, with `objects`, `proposed`, `skipped` and the ontology `version`); the control plane that ran it answers, and a datasource without a job raises `NotFoundError`.

```python
from caliban import OntologyBreakingChangeError

admin.ontology.propose(
    t.id, curated_elements
)  # made outside Caliban: version, proposed, skipped, breaking
res = admin.ontology.review_batch(t.id, {"Service": "approve", "Legacy": "deprecate"}, note="ok")
print(res.version, res.breaking)  # one new ontology version
try:
    admin.ontology.review("Customer", tenant_id=t.id, decision="reject")
except OntologyBreakingChangeError as e:
    for b in e.breaking:
        print(b.element, b.change, b.nodes)  # Customer approved -> rejected ['triage@v3']
```

- Element ids are per tenant (two tenants may both have `Service`), so `ontology.get(tenant_id)`, `review`, `approve`, `reject` and `deprecate` require `tenant_id`; the server answers `400` without it.
- **Proposals.** `propose` takes `OntologyElementProposal`s, `OntologyElement`s or mappings (1 to 5000). `status` may be left out, and whatever it says is ignored: every element is written as `proposed`. An element whose spec, name and description match the tenant's is skipped (`skipped`), whatever status it was sent with, so posting the same exported model again writes no version and keeps every review decision (synonyms, confidence and provenance are not compared). A changed element goes back to `proposed`. The result always has `breaking` (empty unless you acknowledged some). `BadRequestError` for an empty, too long or repeated id; `NotFoundError` for an unknown tenant.
- **Breaking changes.** Withdrawing an approved element that a published node version's fixed CQIR query touches (a decision, or a proposal that changes it) is refused with a `409` unless you pass `acknowledge_breaking=True` (those versions then fail on that query until a version written on the new ontology is published). The SDK raises it as `OntologyBreakingChangeError`, a `ConflictError` with `code == "ontology_breaking_change"`, whose `breaking` lists each element, the `change` (`approved -> proposed`), the node versions as `name@vN` (`nodes`) and the same versions as `OntologyNodeVersion(node, version)` (`versions`). `review_batch` and `propose` return the `breaking` changes they acknowledged.
- Stored elements (`OntologyElement`) always have a `status`, and carry `changed_in`, the ontology version their spec or status last changed in.

#### Intents for `caliban/auto`

A tenant's `caliban/auto` intents (the kNN exemplars routers classify with) come from the config file and from the admin API, which overrides the file intent by intent and needs no restart: routers rebuild their index with the next snapshot.

```python
admin.intents.put(
    t.id, "triage", ["my chest hurts", "I feel dizzy"]
)  # TenantIntentSet, source "api"
listed = admin.intents.list(t.id)  # each intent with source ("config" or "api") and node
admin.intents.delete(t.id, "triage")  # the config file's exemplars apply again
```

At most 100 intents per tenant and 200 exemplars per intent, each 1 to 2000 characters (duplicates are dropped; the SDK checks lengths before sending). Intent names are 1 to 64 characters of `[a-z0-9_.-]`, and `pinned` is reserved. An empty list removes a config-file intent for the tenant (listed in `removed`). Hand an intent to a node with the tenant's `node_routes`. Changes are audited as `tenant.intent.set` and `tenant.intent.delete`, with counts and never the texts. `put` is a `PUT`, retried like a GET; `pinned` and invalid names raise `BadRequestError`, an unknown tenant `NotFoundError`. `intents.set` is a deprecated alias of `put` (it warns with `DeprecationWarning`): the method follows the HTTP verb, as in `put_golden_set` and the TypeScript SDK's `intents.put`.

#### Document collections (RAG)

Nodes search a collection with the `builtin://rag_search` tool, naming it in their scopes as `rag.<name>:read`. The control plane needs a document registry for these calls (else `ServiceUnavailableError`).

```python
models = admin.rag.models(t.id)  # embedding and rerank models the tenant can reach
admin.rag.collections.create(t.id, name="policies", embedding_model=models.embedding[0].id)
admin.rag.documents.ingest(t.id, "policies", doc_id="leave", content=markdown, trust="untrusted")

for c in admin.rag.collections.list(t.id):
    print(c.name, c.document_counts)  # {"ready": 12, "quarantined": 1}
v = admin.rag.documents.get(t.id, "policies", "leave").versions[0]
if v.status == "quarantined":
    print([(f.kind, f.heading_path) for f in v.findings])
    admin.rag.documents.release(t.id, "policies", "leave", version=v.version)
```

- `rag.models` lists the catalogue models of kind `embedding` and `rerank` that one of the tenant's providers, or a shared provider that serves it, can reach, each with its `trust_tier`.
- `document_counts` (on the collection listing only) counts documents per status, using each document's newest version; deleted documents are not counted. `collections.get` adds `used_by`, the published node versions that search the collection.
- A version is `pending` and invisible until a worker has indexed it and its checks passed. An untrusted version whose checks have not passed by `vet_deadline_at` (the collection's `vet_deadline_secs` after ingest, or after its release) fails and stays invisible.
- A finding's `heading_path` gives the headings above the chunk it was found in, so a reviewer can find the place in the source. It is pseudonymised like the indexed text and is never an excerpt of what triggered the finding.
- `release` acknowledges the version's findings (that is the decision it records). Released text is still labelled `retrieved` in node runs: a tainted write it inspires needs an allowlist entry or a human approval.

**Deleting and revoking.** `tenants.delete(tenant_id)`, `api_keys.revoke(tenant_id, key_id)`, `datasources.delete(tenant_id, datasource_id)` and `nodes.delete(tenant_id, node_id)` return `None` on success (204). An unknown id, an id that belongs to another tenant, or one that is already deleted raises `NotFoundError`, so a repeated delete raises too.

```python
# Revoke a leaked key, then retire the whole tenant.
admin.api_keys.revoke(t.id, key.id)
# Revokes keys, wipes BYOK credentials, removes routes, soft-deletes datasources and nodes.
admin.tenants.delete(t.id)

# Deleted tenants and revoked keys are hidden unless you ask for them.
tombstones = [x for x in admin.tenants.list(include_deleted=True) if x.status == "deleted"]
revoked = [k for k in admin.api_keys.list(other_id, include_revoked=True) if k.revoked_at]
```

- Deletes are permanent. The server keeps the rows for audit, but nothing can be restored, and secrets they held (BYOK credentials, datasource connection settings) are destroyed.
- A deleted tenant's id cannot be reused, and every tenant-scoped call for it raises `NotFoundError`.
- A standalone deployment rejects a revoked key on the next request. A split-mode router stops accepting it after its next snapshot poll (`CALIBAN_SNAPSHOT_POLL_SECS`, 10 s by default).
- `Tenant` has `status` (`"active"` or `"deleted"`) and `deleted_at`; `ApiKeyInfo` has `revoked_at`. They are optional and `None` when an older server omits them.

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

# YAML or JSON; raises NodeValidationError listing every issue.
spec = load_node("invoice-triage.node.yaml")
validate(spec)  # takes a NodeSpec or a plain dict
for warning in lint(spec):  # non-fatal checks
    print(warning)
print(spec.to_yaml())
```

`validate` enforces the JSON Schema rules (required fields, enums, budget bounds, tool-ref and scope patterns, strict integers, no unknown top-level keys or `guards` keys) and the rules the schema states only in prose:

- `graph` is allowed only for `kind: workflow`, and a workflow needs vertices;
- `agent` (the tool loop settings, `max_turns`) is allowed only for `kind: agent`;
- vertex ids are unique, and edges and `graph.entry` reference existing vertices;
- every loop contains a vertex with `max_iterations`;
- a vertex's `config.reasoning` is `off`, `low`, `medium` or `high`, and its `config.structured_output` is `json_schema` or `prompt`.

Tool refs are `mcp://server/tool#sha256:<hash>`, `node://name@vN`, `builtin://datasource_query` or `builtin://rag_search` (a document collection, scoped as `rag.<collection>:read`). `budgets.usd` caps what one run may spend on model calls; `guards` sets the loop guard (`max_repeats`, default 3) and transient tool retries (`tool_retries`, default 2).

**Reasoning, constrained decoding and guardrails.** `model_policy.reasoning` (`off`, `low`, `medium`, `high`) is the default reasoning effort of the node's model calls; a model vertex's `config.reasoning` (`llm`, `router`, `verify`, `reduce` with mode `llm`; `MODEL_VERTICES`) overrides it, and the gateway maps it per model family. `config.structured_output` is `json_schema` (the default: the vertex's `output_schema`, or a router's labels, go as `response_format` so vLLM, SGLang and OpenAI constrain decoding) or `prompt` (the schema in the prompt only, for servers without guided decoding); the answer is validated either way. `guardrails.answer` (`redact`, the default, `refuse` or `off`) is the answer guard on the run's final output when it carries untrusted labels: links, contacts and references that only untrusted free text supplied, and echoes of injected instructions, are removed or fail the run. `guardrails.pii` is a floor on the PII protection of the node's model calls on every model class, sovereign models included, and of untrusted tool arguments and results; it only strengthens the tenant's mode.

```yaml
model_policy: { reasoning: low }
guardrails: { answer: refuse, pii: mask }
graph:
  vertices:
    - { id: route, type: router, config: { routes: { urgent: Needs care now, routine: Can wait }, reasoning: "off" } }
    - { id: advise, type: llm, config: { output_schema: { type: object }, reasoning: high, structured_output: json_schema } }
  edges: [{ from: route, to: advise }]
```

`lint` warns about unknown nested keys, MCP tools not pinned with `#sha256:`, node tools not pinned with `@vN`, unknown or write-effect `builtin://` tools, `builtin://datasource_query` without `datasources.scopes`, `builtin://rag_search` without a `rag.<collection>:read` scope, `reasoning` or `structured_output` on a vertex that calls no model, `allow_tainted` on a built-in or node read tool (it applies to writes and to MCP reads that reach the network), `code` vertices (core refuses them at publish), and write tools without an injection defence. See [`tests/fixtures/invoice-triage.node.yaml`](tests/fixtures/invoice-triage.node.yaml) for a full example. Once published, a node is called from the data plane with `model="node/invoice-triage"` (the live version) or `model="node/invoice-triage@v3"` on chat completions, or through the run API (`POST /v1/nodes/{name}/runs`). The `caliban.node` field does not run a node. To mark input fields as untrusted caller data whoever calls, put `"x-caliban-untrusted": true` on them in `prompt.input_schema`; a caller marks more with `caliban.untrusted` (or `untrusted` on the run API). The node design is described in [research note 04](https://github.com/thecalibanproject/docs/blob/main/research/04-agent-orchestration.md).

### Evals (golden sets, pass^k)

`caliban.evals` implements the evaluation loop from [research note 04, section 7](https://github.com/thecalibanproject/docs/blob/main/research/04-agent-orchestration.md#7-evaluation-loop). Each case runs `n` times (default `k`). The report gives pass@1, **pass^k**, pass@k, cost per success, p50/p95 latency and the failures for each case.

```yaml
# golden.yaml
name: invoice-triage-golden
defaults:
  model: caliban/auto          # a model target, or pass --model
  # node: invoice-triage       # or a node target (name or name@vN), or pass --node
  system: Answer tersely.
  temperature: 0               # model targets only, like max_tokens and caliban
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
caliban-eval run node-golden.yaml --node invoice-triage@v3 --k 3 --trials 8 --concurrency 4 \
    --fail-under 0.9 --out-dir eval-reports
```

`--model` and `--node` are exclusive; on the command line they override the dataset defaults, and a default `node` wins over a default `model`.

`run` writes `eval-reports/<name>.eval.json` and `<name>.eval.md` and prints the Markdown. It exits with `0` on success, `1` when pass^k is below `--fail-under`, and `2` on a usage or dataset error or when every trial errored.

For a case with `n` trials and `c` passes:

| Metric | Formula | Meaning |
|---|---|---|
| pass@1 | `c/n` | Average accuracy. |
| **pass^k** | `C(c,k) / C(n,k)` | Probability that k random trials **all** pass (the reliability metric from τ-bench). |
| pass@k | `1 - C(n-c,k) / C(n,k)` | Probability that at least one of k trials passes. |

Both estimators are unbiased when `n >= k`, so use `--trials` above `k` to reduce variance. Dataset-level values are the mean over cases.

- **Cost per success** is total cost divided by passing trials, so failed trials are paid for too. Cost comes from `x-caliban-cost-usd` (models) or the run's metered `cost_usd` (nodes) when the gateway sends it, otherwise from usage tokens and `--price-in`/`--price-out` (USD per 1M tokens, given together).
- **Latency p50/p95** uses linear interpolation over all trials that got a response. For nodes it is the run's execution time (`budget.wall_clock_ms_used`), as in core's evals.
- Evals default to `caliban.cache: "off"` so repeated trials are independent. If a trial is still a cache hit, the report warns about it.
- `llm_judge` is a hook, not a built-in grader. Pass `judge=callable` to `run_eval(...)` to grade with a model of your choice. Without a judge, the check is reported as an error, not a pass.

#### Node targets

A node target (`--node name`, `--node name@vN`, `node:` in the dataset defaults or per case, or `model: node/<name>`) starts every trial as a node run with `POST /v1/nodes/{name}/runs`, the kind of run core's server-side evals create. The run API is used rather than the `model: "node/<name>"` chat shortcut because it matches how core meters and scores an eval: a run that fails or exhausts its budget still returns its cost (the chat shortcut turns it into a `502` with no cost, which would make cost per success look better than it is), the input is sent as is, and each trial carries an `Idempotency-Key`, so a retried request never starts a second run.

- `input` is the run input as is (any JSON, as in core golden sets); `messages` go through the node's chat mapping.
- A trial passes when the run succeeds and every check passes. A run that fails, exhausts its budget or is cancelled fails the trial, and its cost counts.
- `answers` script human steps, by step id (`clarify#0`, `write#0@approve`) or vertex id (`clarify`, `write@approve`). A question without an answer cancels the run and fails the trial; an approval without one is denied.
- `untrusted` (in the defaults or per case: `true`, or JSON pointers into the input) is sent on the run to mark untrusted caller data; each trial reports the run's `input_labels` (`caller` when marked data was present).
- Each trial reports `cost_usd` (what the run is billed), `model_cost_usd` (the real cost of the models its calls were routed to; they differ only for runs billed as `caliban/auto`), `pii_entities` (entities the gateway protected in the run's model calls) and `answer_check` (`passed`, `redacted` or `refused`: what the node's answer guard did when the output carried untrusted labels; `None` when it did not apply).
- A run still going after `--run-timeout` seconds (default 600) is cancelled and counted as an error.
- `temperature`, `max_tokens` and `caliban` do not apply: the node's spec decides. The report warns when they are set.

```yaml
# node-golden.yaml
name: invoice-triage-golden
defaults: { node: invoice-triage }
cases:
  - id: chest
    input: { case: "Chest pain since Tuesday" }
    answers: { clarify: "two days, nobody in danger" }
    checks:
      - { type: json_schema, schema: { type: object, required: [urgency] } }
```

Programmatic use:

```python
from caliban import Caliban
from caliban.evals import load_dataset, run_eval, to_markdown

report = run_eval(load_dataset("golden.yaml"), Caliban(), model="caliban/auto", k=3, trials=5)
print(report.summary.pass_hat_k, to_markdown(report))
```

### Server-side evals and the promotion gate

Core runs node evals itself and uses them to gate promotion: a version with write tools, or a version of a node with a golden bar, is promoted only when its latest finished evals meet the node's promotion bar (see core `docs/evals.md`). `CalibanAdmin.evals` drives that API.

| | Local (`caliban-eval`, `run_eval`) | Server-side (`admin.evals`) |
|---|---|---|
| Runs on | your machine or CI, as a data-plane API key | the workers, as ordinary node runs |
| Targets | any model, or a node | a published node version |
| Golden set | a YAML file you keep | stored per node version, sealed at rest, audited |
| Suites | golden sets | golden sets and the injection suite (sandboxed tools) |
| LLM judge | your `judge=` callable | built in, called through the gateway |
| Counts for promotion | no | yes |

Use the local harness while you write a node or compare models and routing policies: it is quick, needs no control-plane access, and a set can target a plain model. Use server-side evals for anything that decides a promotion: only they count toward the gate, they run the injection suite, and they are audited. The two use the same golden-set format, so a local set (with `input`, not `messages`) can be uploaded as is.

```python
from caliban import CalibanAdmin
from caliban.evals import load_dataset

admin = CalibanAdmin()
admin.evals.put_golden_set("acme", "triage", 3, load_dataset("node-golden.yaml"))
admin.evals.set_policy(
    "acme",
    "triage",
    golden={"min_pass_k": 0.9, "min_k": 3, "max_cost_per_success_usd": 0.02},
    injection={"min_pass_rate": 1.0},
)

# The version is published with {"promote": false}, evaluated, then promoted.
for suite in ("golden", "injection"):
    e = admin.evals.start("acme", "triage", 3, suite=suite)
    e = admin.evals.wait("acme", e.id, timeout_s=1800)
    print(suite, e.status, e.summary)  # a GoldenSummary or an InjectionSummary

gate = admin.evals.gate("acme", "triage", 3)
print(gate.allowed, gate.evidence, [p.message for p in gate.problems])
```

- Evals run as the tenant's eval key: an API key the control plane creates on the first eval that names no key and manages itself (its plaintext never leaves the control plane; the data plane refuses it). The eval's runs and judge calls are limited and metered as that key. `get_key` shows it (`purpose="eval"`), `rotate_key` replaces it (rotate between evals: evals in flight lose their key), and `api_keys.revoke` refuses it with a `409`. Pass `api_key_id` to `start` to run an eval as a customer API key instead.
- `summary` is typed by suite (`GoldenSummary`: pass^k, cost per success, latency, per case; `InjectionSummary`: pass rate, attack success rate, utility under attack, per category), and so is each trial's `verdict` (`GoldenVerdict`, `InjectionVerdict`, or `TrialError` when its run could not be created). `content_purged` says run retention cleared the trials' `detail`; summaries and verdicts stay.
- `gate` returns typed `evidence` (with each pinned subnode's own evidence under `subnodes`) and `problems`; a problem's `path` names a subnode whose own gate failed.
- `get_golden_set` returns the cases only to a caller with `runs.data` (`content_visible`); `evals.get` returns the trials, with their `detail` under the same permission.
- `set_policy` replaces the whole bar: leaving `golden` out removes the golden gate, and an injection eval stays required for versions with write tools whatever `injection` says.
- A read-only version (read tools that bring data in: MCP, `builtin://datasource_query`, `builtin://rag_search`; no write tool) needs an injection eval too by default, because what it reads can steer its answer (the suite's `answer_integrity` cases, which the `answer_guard` defense blocks). `injection={"min_pass_rate": 1.0, "read_only": "optional"}` lets such versions promote without one; a finished injection eval is still checked against the bar. `gate` reports `has_untrusted_reads` next to `has_write_tools`.
- `list` pages with `cursor`/`next_cursor`; `status` takes one status or a list.

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
| `OntologyBreakingChangeError` | 409 with `code="ontology_breaking_change"` (a subclass of `ConflictError`): an ontology decision or proposal would withdraw approved elements that published node versions query; `breaking` lists them |
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
| Retried statuses | GET, HEAD, OPTIONS, PUT and DELETE: 429, 502, 503. POST (chat completions, embeddings, rerank, admin creates) and PATCH (`tenants.update`): 429 and 503 only. Both clients follow the same rules. |
| Transport errors | Connect failures (`ConnectError`, `ConnectTimeout`) are always retried, since the request was never sent. Read timeouts and resets are retried only for idempotent requests, never for a POST. |
| Streams | Retried only before the first byte reaches you. Once iteration has started, a failure raises and is never replayed. |

A POST is never retried on 500, 502 or 504, because the gateway may already have run the request and billed it upstream.

To make a POST retryable like a GET (502, read timeouts and resets included), send an `Idempotency-Key` header, for example `create(..., extra_headers={"Idempotency-Key": key})`. The admin client has no per-call headers, so there it can only come from `default_headers`. The SDK never sends the header on its own; use a fresh key per logical request and the same key for its retries. The gateway deduplicates on it for chat completions, messages, embeddings and rerank: the first request runs (to completion, even if the connection drops); a retry after it finished gets the stored response for 24 h (header `idempotent-replayed: true`, nothing charged again; streams are replayed as one SSE body); a retry while it is still running gets `409` with code `idempotency_key_in_use` and `Retry-After: 1`, which the SDK does not retry, so catch it and retry later; the same key with a different body gets `422` (`idempotency_key_reused`). A failed request frees its key. The admin API does not deduplicate: there a retried POST after a 502 can run twice.

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
