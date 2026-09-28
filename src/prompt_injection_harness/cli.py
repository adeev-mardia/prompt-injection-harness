"""Command-line interface for prompt-injection-harness.

    injection-harness benchmark [--pipeline naive|sanitizing]
    injection-harness detect <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .carriers import CARRIER_DOCUMENTS
from .detector import is_likely_injection, likely_categories
from .generator import build_corpus
from .payloads import PAYLOAD_LIBRARY
from .pipelines import NaiveConcatPipeline, SanitizingPipeline
from .scorer import score_pipeline

PIPELINES = {
    "naive": NaiveConcatPipeline,
    "sanitizing": SanitizingPipeline,
}


def _print_asr_table(result) -> None:
    print(f"\nPipeline: {result.pipeline_name}")
    print(f"Overall ASR: {result.overall_asr:.1%} "
          f"({result.total_successful_attacks}/{result.total_poisoned})")
    print(f"False positive rate on clean docs: {result.false_positive_rate_on_clean:.1%}")

    print("\nASR by attack category:")
    for cat, asr in sorted(result.asr_by_category.items()):
        print(f"  {cat:24s} {asr:6.1%}")

    print("\nASR by injection position:")
    for pos, asr in sorted(result.asr_by_position.items()):
        print(f"  {pos:24s} {asr:6.1%}")

    print("\nASR by stealth level:")
    for s, asr in sorted(result.asr_by_stealth.items()):
        print(f"  {s:24s} {asr:6.1%}")


def cmd_benchmark(args: argparse.Namespace) -> int:
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=args.seed)

    if args.pipeline == "all":
        names = list(PIPELINES.keys())
    else:
        names = [args.pipeline]

    for name in names:
        pipeline = PIPELINES[name]()
        result = score_pipeline(pipeline, corpus)
        _print_asr_table(result)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    flagged = is_likely_injection(text)
    print(f"file: {path}")
    print(f"likely_injection: {flagged}")
    if flagged:
        print(f"matched_signal_groups: {likely_categories(text)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="injection-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Run the full labeled corpus through a target pipeline and print an ASR table")
    bench.add_argument("--pipeline", choices=["naive", "sanitizing", "all"], default="all")
    bench.add_argument("--seed", type=int, default=1337)
    bench.set_defaults(func=cmd_benchmark)

    detect = sub.add_parser("detect", help="Run the heuristic injection detector on a document file")
    detect.add_argument("file", help="Path to a text file to scan")
    detect.set_defaults(func=cmd_detect)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
