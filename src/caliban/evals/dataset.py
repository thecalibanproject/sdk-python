"""Golden-set dataset format.

.. code-block:: yaml

    name: invoice-triage-golden
    defaults:               # all optional
      model: caliban/auto
      node: invoice-triage
      system: "You are terse."
      temperature: 0
      max_tokens: 256
      caliban: { pii: mask } # CalibanOptions; cache defaults to "off" for evals
    cases:
      - id: total
        input: "Total of invoice 42?"          # or: messages: [{role, content}, ...]
        expected: "1250.00"                    # shorthand for an exact check
      - id: json-out
        messages: [{ role: user, content: "Give JSON" }]
        checks:                                # all must pass
          - { type: json_schema, schema: { type: object, required: [ok] } }
          - { type: regex, pattern: "ok", flags: [IGNORECASE] }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..errors import CalibanError
from ..types import CalibanOptions, ChatMessage

__all__ = [
    "AnyCheck",
    "Case",
    "CheckSpec",
    "ContainsCheck",
    "Dataset",
    "DatasetDefaults",
    "DatasetError",
    "ExactCheck",
    "JsonSchemaCheck",
    "LLMJudgeCheck",
    "RegexCheck",
    "load_dataset",
]


class DatasetError(CalibanError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExactCheck(_Strict):
    type: Literal["exact"] = "exact"
    value: str
    strip: bool = True
    case_sensitive: bool = True


class ContainsCheck(_Strict):
    """Passes if every value is a substring of the output."""

    type: Literal["contains"] = "contains"
    value: str | list[str]
    case_sensitive: bool = True

    @property
    def values(self) -> list[str]:
        return [self.value] if isinstance(self.value, str) else list(self.value)


RegexFlag = Literal["IGNORECASE", "MULTILINE", "DOTALL"]


class RegexCheck(_Strict):
    """Passes if ``re.search(pattern, output)`` matches."""

    type: Literal["regex"] = "regex"
    pattern: str
    flags: list[RegexFlag] = Field(default_factory=list)

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return v

    def compiled(self) -> re.Pattern[str]:
        flags = 0
        for f in self.flags:
            flags |= getattr(re, f)
        return re.compile(self.pattern, flags)


class JsonSchemaCheck(_Strict):
    """Parses the output as JSON (code fences tolerated) and validates it."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any] = Field(alias="schema")


class LLMJudgeCheck(_Strict):
    """Stub: graded by a pluggable judge callable (see ``runner.Judge``)."""

    type: Literal["llm_judge"] = "llm_judge"
    rubric: str
    model: str | None = None
    threshold: float = 0.5


AnyCheck = ExactCheck | ContainsCheck | RegexCheck | JsonSchemaCheck | LLMJudgeCheck
CheckSpec = Annotated[AnyCheck, Field(discriminator="type")]


class Case(_Strict):
    id: str
    input: str | None = None
    messages: list[ChatMessage] | None = None
    expected: str | None = None
    check: CheckSpec | None = None
    checks: list[CheckSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    node: str | None = None
    """Per-case node override."""
    caliban: CalibanOptions | None = None
    """Per-case extension override, merged over the dataset defaults."""

    @model_validator(mode="after")
    def _shape(self) -> Case:
        if (self.input is None) == (self.messages is None):
            raise ValueError("exactly one of 'input' or 'messages' is required")
        if self.messages is not None and not self.messages:
            raise ValueError("'messages' must not be empty")
        if not self.all_checks():
            raise ValueError("at least one of 'expected', 'check' or 'checks' is required")
        return self

    def all_checks(self) -> list[AnyCheck]:
        out: list[AnyCheck] = []
        if self.expected is not None:
            out.append(ExactCheck(value=self.expected))
        if self.check is not None:
            out.append(self.check)
        out.extend(self.checks)
        return out

    def build_messages(self, system: str | None) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.messages is not None:
            msgs = [m.model_dump(mode="json", exclude_none=True) for m in self.messages]
        else:
            msgs = [{"role": "user", "content": self.input}]
        if system and not any(m.get("role") in ("system", "developer") for m in msgs):
            msgs.insert(0, {"role": "system", "content": system})
        return msgs


class DatasetDefaults(_Strict):
    model: str | None = None
    node: str | None = None
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    caliban: CalibanOptions | None = None


class Dataset(_Strict):
    name: str
    description: str | None = None
    defaults: DatasetDefaults = Field(default_factory=DatasetDefaults)
    cases: list[Case]

    @model_validator(mode="after")
    def _unique_ids(self) -> Dataset:
        if not self.cases:
            raise ValueError("dataset has no cases")
        ids = [c.id for c in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        return self


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"cannot read dataset {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{p}: top level must be a mapping")
    raw.setdefault("name", p.stem)
    try:
        return Dataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"invalid dataset {p}:\n{exc}") from exc
