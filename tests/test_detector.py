from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.detector import (
    evaluate_detector,
    is_likely_injection,
)
from prompt_injection_harness.generator import build_corpus
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY


def test_detector_flags_direct_override_payload():
    from prompt_injection_harness.payloads import AttackCategory

    payload = next(
        p for p in PAYLOAD_LIBRARY if p.category == AttackCategory.DIRECT_OVERRIDE
    )
    text = payload.render("DEADBEEF")
    assert is_likely_injection(text) is True


def test_detector_flags_jailbreak_payload():
    from prompt_injection_harness.payloads import AttackCategory

    payload = next(
        p for p in PAYLOAD_LIBRARY if p.category == AttackCategory.JAILBREAK_ROLEPLAY
    )
    text = payload.render("DEADBEEF")
    assert is_likely_injection(text) is True


def test_detector_does_not_flag_clean_carrier_docs():
    for carrier in CARRIER_DOCUMENTS:
        assert is_likely_injection(carrier.text) is False, carrier.name


def test_evaluate_detector_reports_reasonable_precision_recall():
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=5)
    metrics = evaluate_detector(corpus)

    # No false positives on our clean synthetic carrier docs.
    assert metrics.false_positives == 0
    assert metrics.precision == 1.0

    # Recall should be well above zero (most categories use recognizable
    # trigger phrasing) but not necessarily perfect -- the harness
    # deliberately includes a maximally-stealthy indirect payload that is
    # designed to evade keyword heuristics.
    assert 0.5 < metrics.recall <= 1.0
    assert metrics.f1 > 0.0

    # The bookkeeping must be internally consistent.
    assert metrics.true_positives + metrics.false_negatives == sum(
        1 for d in corpus if d.poisoned
    )
    assert metrics.true_negatives + metrics.false_positives == sum(
        1 for d in corpus if not d.poisoned
    )


def test_evaluate_detector_per_category_recall_keys_match_categories():
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=5)
    metrics = evaluate_detector(corpus)
    categories_in_corpus = {
        d.attack_category.value for d in corpus if d.poisoned
    }
    assert set(metrics.per_category_recall.keys()) == categories_in_corpus
    for recall in metrics.per_category_recall.values():
        assert 0.0 <= recall <= 1.0
