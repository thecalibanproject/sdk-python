"""Deterministic output checkers (plus the ``llm_judge`` stub hook)."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dataset import (
    AnyCheck,
    Case,
    ContainsCheck,
    ExactCheck,
    JsonSchemaCheck,
    LLMJudgeCheck,
    RegexCheck,
)

__all__ = ["CheckResult", "Judge", "extract_json", "run_check", "validate_json_schema"]


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str | None = None
    error: bool = False
    """True when the check could not be evaluated (e.g. no judge configured)."""


Judge = Callable[[LLMJudgeCheck, str, Case], CheckResult]
"""Grades ``output`` for ``case`` against the rubric. Supply one to enable llm_judge."""


def run_check(check: AnyCheck, output: str, case: Case, judge: Judge | None = None) -> CheckResult:
    if isinstance(check, ExactCheck):
        got, want = output, check.value
        if check.strip:
            got, want = got.strip(), want.strip()
        if not check.case_sensitive:
            got, want = got.casefold(), want.casefold()
        if got == want:
            return CheckResult(True)
        return CheckResult(False, f"exact: expected {check.value!r}")
    if isinstance(check, ContainsCheck):
        hay = output if check.case_sensitive else output.casefold()
        missing = [
            v for v in check.values if (v if check.case_sensitive else v.casefold()) not in hay
        ]
        if not missing:
            return CheckResult(True)
        return CheckResult(False, f"contains: missing {missing!r}")
    if isinstance(check, RegexCheck):
        if check.compiled().search(output):
            return CheckResult(True)
        return CheckResult(False, f"regex: no match for /{check.pattern}/")
    if isinstance(check, JsonSchemaCheck):
        try:
            doc = extract_json(output)
        except ValueError as exc:
            return CheckResult(False, f"json_schema: output is not JSON ({exc})")
        errors = validate_json_schema(doc, check.json_schema)
        if not errors:
            return CheckResult(True)
        return CheckResult(False, "json_schema: " + "; ".join(errors[:5]))
    if isinstance(check, LLMJudgeCheck):
        if judge is None:
            return CheckResult(
                False, "llm_judge: no judge configured (stub; pass judge=...)", error=True
            )
        return judge(check, output, case)
    raise TypeError(f"unknown check {check!r}")  # pragma: no cover


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating surrounding whitespace and a code fence."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


# ───────────── minimal JSON Schema validator (draft 2020-12 subset) ─────────────
# Supported: type, enum, const, properties, required, additionalProperties,
# items, minItems, maxItems, uniqueItems, minLength, maxLength, pattern,
# minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf,
# allOf, anyOf, oneOf, not. Unsupported keywords ($ref, format, ...) are ignored.


def _is_type(v: Any, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        if isinstance(v, bool):
            return False
        return isinstance(v, int) or (isinstance(v, float) and v.is_integer())
    if t == "number":
        return isinstance(v, int | float) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return bool(a == b)


def validate_json_schema(instance: Any, schema: Any, path: str = "$") -> list[str]:
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: not allowed"]
    if not isinstance(schema, dict):
        return []
    errs: list[str] = []

    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(instance, x) for x in types):
            return [f"{path}: expected type {t}"]
    if "enum" in schema and not any(_eq(instance, e) for e in schema["enum"]):
        errs.append(f"{path}: not in enum {schema['enum']}")
    if "const" in schema and not _eq(instance, schema["const"]):
        errs.append(f"{path}: expected const {schema['const']!r}")

    if isinstance(instance, dict):
        props: dict[str, Any] = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in instance:
                errs.append(f"{path}: missing required property {req!r}")
        addl = schema.get("additionalProperties", True)
        for key, val in instance.items():
            sub = f"{path}.{key}"
            if key in props:
                errs.extend(validate_json_schema(val, props[key], sub))
            elif addl is False:
                errs.append(f"{sub}: additional property not allowed")
            elif isinstance(addl, dict):
                errs.extend(validate_json_schema(val, addl, sub))

    if isinstance(instance, list):
        items = schema.get("items")
        if items is not None:
            for i, val in enumerate(instance):
                errs.extend(validate_json_schema(val, items, f"{path}[{i}]"))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(x, sort_keys=True) for x in instance]
            if len(seen) != len(set(seen)):
                errs.append(f"{path}: items are not unique")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errs.append(f"{path}: does not match /{schema['pattern']}/")

    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errs.append(f"{path}: >= exclusiveMaximum {schema['exclusiveMaximum']}")
        mult = schema.get("multipleOf")
        if mult:
            q = instance / mult
            if not math.isclose(q, round(q), abs_tol=1e-9):
                errs.append(f"{path}: not a multiple of {mult}")

    for sub in schema.get("allOf", []):
        errs.extend(validate_json_schema(instance, sub, path))
    if "anyOf" in schema and not any(
        not validate_json_schema(instance, s, path) for s in schema["anyOf"]
    ):
        errs.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        n = sum(1 for s in schema["oneOf"] if not validate_json_schema(instance, s, path))
        if n != 1:
            errs.append(f"{path}: matches {n} of oneOf (expected exactly 1)")
    if "not" in schema and not validate_json_schema(instance, schema["not"], path):
        errs.append(f"{path}: must not match 'not' schema")
    return errs
