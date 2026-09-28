#!/usr/bin/env python3
"""Standalone detector precision/recall/F1 report using sklearn.metrics.

Run with:  python3 scripts/detector_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import classification_report, confusion_matrix

from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.detector import is_likely_injection
from prompt_injection_harness.generator import build_corpus
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY


def main() -> int:
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=1337)
    y_true = [1 if d.poisoned else 0 for d in corpus]
    y_pred = [1 if is_likely_injection(d.text) else 0 for d in corpus]

    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred, target_names=["clean", "poisoned"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
