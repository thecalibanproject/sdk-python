"""``caliban-eval`` command line.

Examples::

    caliban-eval validate golden.yaml
    caliban-eval run golden.yaml --model caliban/auto --k 3 --base-url http://localhost:8080/v1
    caliban-eval run golden.yaml --node invoice-triage --k 3 --trials 5 --fail-under 0.9

Exit codes: 0 ok, 1 below ``--fail-under``, 2 usage/dataset error or every trial errored.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .._client import Caliban
from ..errors import CalibanError
from .dataset import load_dataset
from .report import to_markdown, write_reports
from .runner import Pricing, Trial, run_eval

__all__ = ["build_parser", "main"]


def _make_client(args: argparse.Namespace) -> Caliban:  # patched in tests
    return Caliban(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="caliban-eval", description="Caliban golden-set eval harness")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a dataset file")
    v.add_argument("dataset", type=Path)

    r = sub.add_parser("run", help="run a dataset against a model or node")
    r.add_argument("dataset", type=Path)
    r.add_argument("--model", help="model id, tenant alias or caliban/auto")
    r.add_argument(
        "--node", help="node name (sent as caliban.node; model defaults to caliban/auto)"
    )
    r.add_argument("--k", "-k", type=int, default=3, help="k for pass^k / pass@k (default 3)")
    r.add_argument("--trials", type=int, help="trials per case (default k; must be >= k)")
    r.add_argument("--base-url", help="data-plane URL incl. /v1 (default $CALIBAN_BASE_URL)")
    r.add_argument("--api-key", help="tenant key (default $CALIBAN_API_KEY)")
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    r.add_argument("--max-retries", type=int, default=2)
    r.add_argument("--price-in", type=float, help="USD per 1M prompt tokens (fallback cost)")
    r.add_argument("--price-out", type=float, help="USD per 1M completion tokens")
    r.add_argument("--out-dir", type=Path, default=Path("eval-reports"))
    r.add_argument("--json", dest="json_path", type=Path, help="JSON report path")
    r.add_argument("--md", dest="md_path", type=Path, help="Markdown report path")
    r.add_argument("--fail-under", type=float, help="exit 1 if pass^k is below this (0..1)")
    r.add_argument("--quiet", "-q", action="store_true")
    return p


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "dataset"


def _cmd_validate(args: argparse.Namespace) -> int:
    ds = load_dataset(args.dataset)
    n_checks = sum(len(c.all_checks()) for c in ds.cases)
    print(f"ok: {ds.name}: {len(ds.cases)} cases, {n_checks} checks")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    ds = load_dataset(args.dataset)
    if (args.price_in is None) != (args.price_out is None):
        raise CalibanError("--price-in and --price-out must be given together")
    pricing = Pricing(args.price_in, args.price_out) if args.price_in is not None else None

    def progress(case_id: str, t: Trial) -> None:
        if not args.quiet:
            mark = "." if t.passed else ("E" if t.error else "F")
            print(mark, end="", flush=True, file=sys.stderr)

    with _make_client(args) as client:
        report = run_eval(
            ds,
            client,
            model=args.model,
            node=args.node,
            k=args.k,
            trials=args.trials,
            concurrency=args.concurrency,
            pricing=pricing,
            on_trial=progress,
        )
    if not args.quiet:
        print(file=sys.stderr)

    stem = _slug(ds.name)
    json_path = args.json_path or args.out_dir / f"{stem}.eval.json"
    md_path = args.md_path or args.out_dir / f"{stem}.eval.md"
    write_reports(report, json_path, md_path)

    if not args.quiet:
        print(to_markdown(report))
    s = report.summary
    print(
        f"pass@1={s.pass_at_1:.3f} pass^{report.k}={s.pass_hat_k:.3f} "
        f"errors={s.errors} -> {json_path}, {md_path}"
    )
    if s.trials and s.errors == s.trials:
        print("FAIL: every trial errored (is the gateway reachable?)", file=sys.stderr)
        return 2
    if args.fail_under is not None and s.pass_hat_k < args.fail_under:
        print(f"FAIL: pass^{report.k}={s.pass_hat_k:.3f} < {args.fail_under}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _cmd_validate(args)
        return _cmd_run(args)
    except CalibanError as exc:
        print(f"caliban-eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
