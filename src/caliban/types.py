"""Data-plane request/response types (OpenAI-compatible, plus Caliban extensions).

Response models use ``extra="allow"`` so fields added by upstream providers or
future Caliban versions survive a round-trip.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

__all__ = [
    "CacheMode",
    "CacheStatus",
    "CacheTier",
    "CalibanOptions",
    "CalibanResponseMeta",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatMessage",
    "Choice",
    "ChoiceDelta",
    "ChunkChoice",
    "CompletionUsage",
    "CreateEmbeddingResponse",
    "Embedding",
    "EmbeddingUsage",
    "IntentDecision",
    "Model",
    "ModelCalibanInfo",
    "ModelCapabilities",
    "ModelKind",
    "ModelList",
    "PiiMode",
    "ReasoningEffort",
    "RerankDocument",
    "RerankResponse",
    "RerankResult",
    "RerankUsage",
    "Role",
    "RouteStage",
    "TrustTier",
]

PiiMode = Literal["off", "mask", "reversible"]
CacheMode = Literal["off", "exact", "semantic"]
CacheStatus = Literal["hit", "miss", "bypass"]
CacheTier = Literal["exact", "semantic"]
"""Which cache tier answered a hit (``x-caliban-cache-tier``, ``UsageEvent.cache_tier``)."""
RouteStage = Literal["rules", "knn", "keyword"]
"""Routing stage that decided the intent: ``rules`` (pinned model), ``knn`` or ``keyword``."""
Role = Literal["system", "developer", "user", "assistant", "tool"]
ReasoningEffort = Literal["off", "low", "medium", "high"]
"""``caliban.reasoning``. Mapped per model family (Qwen3 ``enable_thinking``,
``reasoning_effort``); ``off`` disables thinking on hybrid models."""
TrustTier = Literal["t0_sovereign", "t1_attested", "t2_contracted", "t3_public"]
ModelKind = Literal["chat", "embedding", "rerank"]


class CalibanOptions(BaseModel):
    """The ``caliban`` request extension (``components.schemas.CalibanExtension``).

    Unknown keys are rejected to catch typos; use ``extra_body`` on ``create`` to
    send fields this SDK version does not know about yet.
    """

    model_config = ConfigDict(extra="forbid")

    pii: PiiMode | None = None
    cache: CacheMode | None = None
    datasources: list[str] | None = None
    node: str | None = None
    max_cost_usd: float | None = Field(default=None, ge=0)
    reasoning: ReasoningEffort | None = None
    """How hard a reasoning model should think (``off|low|medium|high``)."""
    zdr: bool | None = None
    trace_id: str | None = None

    def to_body(self) -> dict[str, Any]:
        """Serialize, dropping unset fields so server-side defaults apply."""
        return self.model_dump(mode="json", exclude_none=True)


class _ReasoningFields(BaseModel):
    reasoning_content: str | None = None
    """Thinking text from a reasoning model (vLLM/SGLang/DeepSeek naming; Caliban also moves
    inline ``<think>`` blocks here)."""
    reasoning: str | None = None
    """Thinking text under the name some newer servers use."""

    @property
    def reasoning_text(self) -> str | None:
        """``reasoning_content``, falling back to ``reasoning``; ``None`` if neither is set."""
        if self.reasoning_content is not None:
            return self.reasoning_content
        return self.reasoning


class ChatMessage(_ReasoningFields):
    """A chat message. Assistant messages from reasoning models may carry
    ``reasoning_content`` (or ``reasoning``) next to ``content``.

    TODO(contract): the reasoning fields are not yet in ``ChatMessage`` in openapi.yaml.
    """

    model_config = ConfigDict(extra="allow")

    role: Role | str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class CompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


_CACHE_TIERS: frozenset[str] = frozenset(("exact", "semantic"))
_ROUTE_STAGES: frozenset[str] = frozenset(("rules", "knn", "keyword"))
_DECIMAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


class IntentDecision(BaseModel):
    """The routing decision from ``x-caliban-intent``.

    The gateway sends ``<intent>;confidence=<0..1>;stage=<rules|knn|keyword>``, plus
    ``;knn=<reason>`` when kNN routing was on but did not decide.
    """

    model_config = ConfigDict(frozen=True)

    intent: str
    """Intent name, e.g. ``translate``; ``pinned`` when the client named a model."""
    confidence: float
    """Confidence of the decision, 0..1."""
    stage: RouteStage
    knn_reason: str | None = None
    """Why kNN did not decide although it was on (``timeout``, ``embed_error``,
    ``unavailable``, ``no_text``, ``abstain_oos``, ``abstain_confidence``,
    ``abstain_margin``, ``abstain_empty``, or a newer reason). ``None`` otherwise."""

    @classmethod
    def parse(cls, value: str | None) -> IntentDecision | None:
        """Parse an ``x-caliban-intent`` value. Never raises.

        Unknown ``key=value`` fields are ignored. Returns ``None`` for an absent value or a
        malformed one: no intent name, a missing or out-of-range ``confidence``, or a
        missing or unknown ``stage``.
        """
        if value is None:
            return None
        head, *rest = value.strip().split(";")
        intent = head.strip()
        if not intent or "=" in intent:
            return None
        fields: dict[str, str] = {}
        for part in rest:
            key, eq, val = part.partition("=")
            key = key.strip().lower()
            if eq and key and key not in fields:
                fields[key] = val.strip()
        conf_raw = fields.get("confidence", "")
        if not _DECIMAL.fullmatch(conf_raw):
            return None
        confidence = float(conf_raw)
        if not 0.0 <= confidence <= 1.0:
            return None
        stage = fields.get("stage", "").lower()
        if stage not in _ROUTE_STAGES:
            return None
        return cls(
            intent=intent,
            confidence=confidence,
            stage=cast(RouteStage, stage),
            knn_reason=fields.get("knn") or None,
        )


def _cache_tier(value: str | None) -> CacheTier | None:
    v = (value or "").strip().lower()
    return cast(CacheTier, v) if v in _CACHE_TIERS else None


class CalibanResponseMeta(BaseModel):
    """Metadata Caliban returns in ``x-caliban-*`` response headers."""

    request_id: str | None = None
    routed_model: str | None = None
    cache: CacheStatus | str | None = None
    """``x-caliban-cache``: ``hit`` (either cache tier), ``miss`` or ``bypass``."""
    cache_tier: CacheTier | None = None
    """``x-caliban-cache-tier``: ``exact`` or ``semantic``, sent on hits only. ``None`` when
    absent or not one of those values."""
    intent: IntentDecision | None = None
    """``x-caliban-intent``, parsed. ``None`` when absent or malformed."""
    pii_entities: int | None = None
    cost_usd: float | None = None
    """``x-caliban-cost-usd``: cost of the request in USD. Sent on non-streaming responses
    when the model has a price; ``None`` otherwise (streams report cost in usage events)."""
    headers: dict[str, str] = Field(default_factory=dict)
    """All ``x-caliban-*`` headers, lower-cased, verbatim."""

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> CalibanResponseMeta:
        cal = {k.lower(): v for k, v in headers.items() if k.lower().startswith("x-caliban-")}
        return cls(
            request_id=cal.get("x-caliban-request-id"),
            routed_model=cal.get("x-caliban-routed-model"),
            cache=cal.get("x-caliban-cache"),
            cache_tier=_cache_tier(cal.get("x-caliban-cache-tier")),
            intent=IntentDecision.parse(cal.get("x-caliban-intent")),
            pii_entities=_to_int(cal.get("x-caliban-pii-entities")),
            cost_usd=_to_float(cal.get("x-caliban-cost-usd")),
            headers=cal,
        )


def _to_int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _to_float(v: str | None) -> float | None:
    try:
        f = float(v) if v is not None else None
    except ValueError:
        return None
    return f if f is not None and math.isfinite(f) and f >= 0 else None


class _WithMeta(BaseModel):
    _caliban: CalibanResponseMeta = PrivateAttr(default_factory=CalibanResponseMeta)

    @property
    def caliban(self) -> CalibanResponseMeta:
        """Response metadata parsed from ``x-caliban-*`` headers."""
        return self._caliban


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletion(_WithMeta):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int | None = None
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None

    @property
    def text(self) -> str:
        """Content of the first choice as a string (empty if none)."""
        if not self.choices:
            return ""
        content = self.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))

    @property
    def reasoning_text(self) -> str | None:
        """Reasoning of the first choice (``reasoning_content`` or ``reasoning``), if any."""
        if not self.choices:
            return None
        return self.choices[0].message.reasoning_text


class ChoiceDelta(_ReasoningFields):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: ChoiceDelta = Field(default_factory=ChoiceDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion.chunk"
    created: int | None = None
    model: str | None = None
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: CompletionUsage | None = None


class ModelCapabilities(BaseModel):
    """What a model supports (``components.schemas.ModelCapabilities``).

    Extra keys are kept so newer capabilities survive a discover -> create round-trip.
    """

    model_config = ConfigDict(extra="allow")

    tools: bool = False
    vision: bool = False
    reasoning: Literal["none", "always", "hybrid"] = "none"
    reasoning_control: Literal["none", "enable_thinking", "reasoning_effort"] = "none"
    inline_think_tags: bool = False
    """Server returns ``<think>...</think>`` in content; Caliban moves it to reasoning_content."""


class ModelCalibanInfo(BaseModel):
    """Caliban details on a ``/v1/models`` item."""

    model_config = ConfigDict(extra="allow")

    kind: ModelKind
    family: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    trust_tier: TrustTier


class Model(BaseModel):
    """A ``/v1/models`` item.

    TODO(contract): ``caliban`` is returned by the gateway but not yet described in openapi.yaml.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "model"
    owned_by: str
    caliban: ModelCalibanInfo | None = None
    """``None`` for the virtual ``caliban/auto`` entry."""


class ModelList(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: str = "list"
    data: list[Model]


class Embedding(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    total_tokens: int | None = None


class CreateEmbeddingResponse(_WithMeta):
    """Response of ``POST /v1/embeddings``; header metadata on ``.caliban``."""

    model_config = ConfigDict(extra="allow")

    object: str = "list"
    model: str
    data: list[Embedding]
    usage: EmbeddingUsage | None = None

    @property
    def vectors(self) -> list[list[float]]:
        """The embeddings in input order (sorted by ``index``)."""
        return [d.embedding for d in sorted(self.data, key=lambda d: d.index)]


class RerankDocument(BaseModel):
    """Echoed document (``return_documents=True``); always the caller's original text."""

    model_config = ConfigDict(extra="allow")

    text: str | None = None


class RerankResult(BaseModel):
    """One scored document. ``index`` points into the request's ``documents``."""

    model_config = ConfigDict(extra="allow")

    index: int
    relevance_score: float
    document: RerankDocument | None = None


class RerankUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_tokens: int | None = None


class RerankResponse(_WithMeta):
    """Response of ``POST /v1/rerank``; header metadata on ``.caliban``.

    ``results`` keep the server's order: highest ``relevance_score`` first.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    results: list[RerankResult]
    usage: RerankUsage | None = None
