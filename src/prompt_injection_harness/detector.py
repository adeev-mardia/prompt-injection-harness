"""Heuristic prompt-injection detector.

A regex/keyword-based classifier that flags a document as "likely contains
a prompt injection attempt" *before* it ever reaches the target pipeline.
This is the reusable defensive artifact of the project: a lightweight,
dependency-free pre-filter that a real RAG ingestion pipeline could run on
every retrieved chunk.

It is intentionally simple (no ML, no external calls) so it is fast,
auditable, and fully deterministic -- trading some recall (especially on
heavily-obfuscated or deeply-indirect payloads, by design the hardest
category) for zero false positives on ordinary prose.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .generator import GeneratedDocument

# ---------------------------------------------------------------------------
# Pattern library. Grouped by the same attack categories as payloads.py so
# a match can (optionally) be attributed to a likely category, though the
# primary output is a boolean flag.
# ---------------------------------------------------------------------------

_PATTERNS = {
    "direct_override": [
        re.compile(r"ignore\s+(all\s+)?(the\s+)?(previous|above|prior)\s+instructions", re.I),
        re.compile(r"disregard\s+(everything\s+|all\s+)?(above|previous|prior)", re.I),
        re.compile(r"\bnew\s+instructions?\s*:", re.I),
    ],
    "fake_system_message": [
        re.compile(r"\[\s*system\s*\]", re.I),
        re.compile(r"</?\s*(system|developer_message)\b", re.I),
        re.compile(r"\bdeveloper\s+(message|override)\b", re.I),
    ],
    "data_exfiltration": [
        re.compile(r"reveal\s+(your\s+)?(full\s+)?(system\s+)?prompt", re.I),
        re.compile(r"repeat\s+(the\s+)?(system\s+)?prompt\s+verbatim", re.I),
        re.compile(r"print\s+everything\s+above", re.I),
        re.compile(r"reveal\s+.*hidden\s+instructions", re.I),
    ],
    "jailbreak_roleplay": [
        re.compile(r"\byou\s+are\s+now\s+DAN\b", re.I),
        re.compile(r"\bDAN\b.*no\s+restrictions", re.I),
        re.compile(r"unrestricted\s+(assistant|ai)\s+with\s+no\s+(content\s+)?(policy|filters)", re.I),
        re.compile(r"no\s+content\s+policy", re.I),
    ],
    "obfuscation_hint": [
        # A long base64-looking run is suspicious in ordinary prose.
        re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})"),
    ],
    "indirect_ai_address": [
        re.compile(r"\bfor\s+any\s+(ai|assistant|llm|model|language model)\b", re.I),
    ],
    "canary_leak_hint": [
        # If a document itself instructs emitting a "CANARY-" style token,
        # that is a strong, category-agnostic signal of injected content in
        # this harness's synthetic corpus (real deployments would swap this
        # for a generic "asks the model to output a fixed marker" heuristic).
        re.compile(r"output\s+(the\s+)?(exact\s+)?(string\s+|text\s+|token\s+)?CANARY-", re.I),
    ],
}

# A conservative homoglyph check: a handful of Cyrillic letters that are
# visually near-identical to common Latin letters, appearing inside an
# otherwise-Latin word (a strong obfuscation signal, rare in genuine prose).
_HOMOGLYPH_CHARS = re.compile(r"[а-яА-Я]")


def _looks_like_base64(candidate: str) -> bool:
    try:
        padded = candidate + "=" * (-len(candidate) % 4)
        decoded = base64.b64decode(padded, validate=True)
        decoded.decode("utf-8")
        return True
    except Exception:
        return False


def is_likely_injection(text: str) -> bool:
    """Return True if ``text`` is heuristically likely to contain a
    prompt-injection attempt."""
    for patterns in _PATTERNS.values():
        for pattern in patterns:
            for match in pattern.finditer(text):
                if pattern in _PATTERNS["obfuscation_hint"]:
                    if _looks_like_base64(match.group(0)):
                        return True
                    continue
                return True
    if _HOMOGLYPH_CHARS.search(text):
        return True
    return False


def likely_categories(text: str) -> List[str]:
    """Return the list of pattern-group names that matched ``text``."""
    hits = []
    for name, patterns in _PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                if name == "obfuscation_hint" and not _looks_like_base64(match.group(0)):
                    continue
                hits.append(name)
                break
            else:
                continue
            break
    if _HOMOGLYPH_CHARS.search(text) and "obfuscation_hint" not in hits:
        hits.append("obfuscation_hint")
    return hits


@dataclass
class DetectorMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    per_category_recall: dict


def evaluate_detector(
    corpus: Sequence[GeneratedDocument],
    detector_fn=is_likely_injection,
) -> DetectorMetrics:
    """Evaluate the detector against a labeled corpus.

    Uses hand-rolled precision/recall/F1 (matching sklearn.metrics'
    definitions) so the library has no hard runtime dependency on
    scikit-learn; ``scripts/benchmark.py`` cross-checks these against
    ``sklearn.metrics`` when it is installed.
    """
    tp = fp = fn = tn = 0
    per_category_hits: dict = {}
    per_category_total: dict = {}

    for doc in corpus:
        predicted = detector_fn(doc.text)
        actual = doc.poisoned

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

        if actual and doc.attack_category is not None:
            cat = doc.attack_category.value if hasattr(doc.attack_category, "value") else str(doc.attack_category)
            per_category_total[cat] = per_category_total.get(cat, 0) + 1
            if predicted:
                per_category_hits[cat] = per_category_hits.get(cat, 0) + 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_category_recall = {
        cat: per_category_hits.get(cat, 0) / total
        for cat, total in per_category_total.items()
    }

    return DetectorMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        per_category_recall=per_category_recall,
    )
