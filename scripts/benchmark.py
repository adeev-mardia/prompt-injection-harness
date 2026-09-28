#!/usr/bin/env python3
"""Demonstrates Naive vs Sanitizing ASR, and detector precision/recall.

Run with:  python3 scripts/benchmark.py

Fully offline and deterministic. This is the script whose output was used
to populate the "Results" section of the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.detector import evaluate_detector
from prompt_injection_harness.generator import build_corpus
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY
from prompt_injection_harness.pipelines import NaiveConcatPipeline, SanitizingPipeline
from prompt_injection_harness.scorer import score_pipeline


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=1337)
    poisoned_n = sum(1 for d in corpus if d.poisoned)
    clean_n = sum(1 for d in corpus if not d.poisoned)

    _hr("Corpus")
    print(f"Carriers:          {len(CARRIER_DOCUMENTS)}")
    print(f"Payload templates: {len(PAYLOAD_LIBRARY)}")
    print(f"Total documents:   {len(corpus)}  ({poisoned_n} poisoned, {clean_n} clean)")

    results = {}
    for label, pipeline_cls in [("naive", NaiveConcatPipeline), ("sanitizing", SanitizingPipeline)]:
        _hr(f"Pipeline benchmark: {label}")
        pipeline = pipeline_cls()
        result = score_pipeline(pipeline, corpus)
        results[label] = result

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

    _hr("Defense-in-depth summary")
    naive_asr = results["naive"].overall_asr
    sanitizing_asr = results["sanitizing"].overall_asr
    if naive_asr > 0:
        reduction = 1 - (sanitizing_asr / naive_asr)
        print(f"NaiveConcatPipeline ASR:    {naive_asr:.1%}")
        print(f"SanitizingPipeline ASR:     {sanitizing_asr:.1%}")
        print(f"Relative ASR reduction:     {reduction:.1%}")

    _hr("Heuristic detector evaluation")
    metrics = evaluate_detector(corpus)
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1:        {metrics.f1:.3f}")
    print(f"TP={metrics.true_positives}  FP={metrics.false_positives}  "
          f"FN={metrics.false_negatives}  TN={metrics.true_negatives}")

    print("\nRecall by attack category:")
    for cat, recall in sorted(metrics.per_category_recall.items()):
        print(f"  {cat:24s} {recall:6.1%}")

    # Cross-check with sklearn.metrics if available (matches the
    # hand-rolled computation in detector.py exactly).
    try:
        from sklearn.metrics import precision_recall_fscore_support

        from prompt_injection_harness.detector import is_likely_injection

        y_true = [1 if d.poisoned else 0 for d in corpus]
        y_pred = [1 if is_likely_injection(d.text) else 0 for d in corpus]
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        _hr("sklearn.metrics cross-check")
        print(f"precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
