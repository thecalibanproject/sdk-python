"""Golden-set eval harness: run cases k times, report pass@1, pass^k, cost, latency."""

from .checkers import CheckResult, Judge, extract_json, run_check, validate_json_schema
from .dataset import (
    Case,
    ContainsCheck,
    Dataset,
    DatasetDefaults,
    DatasetError,
    ExactCheck,
    JsonSchemaCheck,
    LLMJudgeCheck,
    RegexCheck,
    load_dataset,
)
from .metrics import pass_at_k, pass_hat_k, percentile
from .report import to_json, to_markdown, write_reports
from .runner import CaseResult, EvalReport, EvalSummary, Pricing, Target, Trial, run_eval

__all__ = [
    "Case",
    "CaseResult",
    "CheckResult",
    "ContainsCheck",
    "Dataset",
    "DatasetDefaults",
    "DatasetError",
    "EvalReport",
    "EvalSummary",
    "ExactCheck",
    "JsonSchemaCheck",
    "Judge",
    "LLMJudgeCheck",
    "Pricing",
    "RegexCheck",
    "Target",
    "Trial",
    "extract_json",
    "load_dataset",
    "pass_at_k",
    "pass_hat_k",
    "percentile",
    "run_check",
    "run_eval",
    "to_json",
    "to_markdown",
    "validate_json_schema",
    "write_reports",
]
