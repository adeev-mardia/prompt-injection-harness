import pytest

from prompt_injection_harness.generator import GeneratedDocument, InjectionPosition, StealthLevel
from prompt_injection_harness.payloads import AttackCategory
from prompt_injection_harness.scorer import score_pipeline


class _FakePipeline:
    """A stub pipeline with a scripted response so we can test the scorer's
    math in isolation from the reference pipelines."""

    name = "fake"

    def __init__(self, leak_canaries):
        self.leak_canaries = set(leak_canaries)

    def run(self, context_docs, user_query):
        text = " ".join(context_docs)
        for canary in self.leak_canaries:
            if canary in text:
                return f"response leaking CANARY-{canary}"
        return "a safe, boring response"


def _doc(canary, category, poisoned=True, position=InjectionPosition.END, stealth=StealthLevel.MEDIUM):
    return GeneratedDocument(
        text=f"some carrier text ... CANARY-{canary} ... more text",
        poisoned=poisoned,
        attack_category=category if poisoned else None,
        injection_position=position if poisoned else None,
        stealth_level=stealth if poisoned else None,
        carrier_name="synthetic",
        payload_name="synthetic_payload" if poisoned else None,
        canary=canary if poisoned else None,
    )


def test_score_pipeline_overall_asr_math():
    docs = [
        _doc("AAA1", AttackCategory.DIRECT_OVERRIDE),
        _doc("BBB2", AttackCategory.DIRECT_OVERRIDE),
        _doc("CCC3", AttackCategory.JAILBREAK_ROLEPLAY),
        _doc("DDD4", AttackCategory.JAILBREAK_ROLEPLAY),
    ]
    # Pipeline "succeeds" (leaks) only for AAA1 and CCC3.
    pipeline = _FakePipeline(leak_canaries=["AAA1", "CCC3"])

    result = score_pipeline(pipeline, docs)

    assert result.total_poisoned == 4
    assert result.total_successful_attacks == 2
    assert result.overall_asr == pytest.approx(0.5)


def test_score_pipeline_per_category_breakdown():
    docs = [
        _doc("AAA1", AttackCategory.DIRECT_OVERRIDE),
        _doc("BBB2", AttackCategory.DIRECT_OVERRIDE),
        _doc("CCC3", AttackCategory.JAILBREAK_ROLEPLAY),
        _doc("DDD4", AttackCategory.JAILBREAK_ROLEPLAY),
    ]
    # Both direct_override docs leak; neither jailbreak doc leaks.
    pipeline = _FakePipeline(leak_canaries=["AAA1", "BBB2"])

    result = score_pipeline(pipeline, docs)

    assert result.asr_by_category["direct_override"] == pytest.approx(1.0)
    assert result.asr_by_category["jailbreak_roleplay"] == pytest.approx(0.0)


def test_score_pipeline_false_positive_rate_on_clean_docs():
    poisoned = [_doc("AAA1", AttackCategory.DIRECT_OVERRIDE)]
    clean = [
        GeneratedDocument(text="totally clean text", poisoned=False, carrier_name="c1"),
        GeneratedDocument(text="also clean text", poisoned=False, carrier_name="c2"),
    ]
    pipeline = _FakePipeline(leak_canaries=["AAA1"])

    result = score_pipeline(pipeline, poisoned + clean)

    assert result.false_positive_rate_on_clean == pytest.approx(0.0)


def test_score_pipeline_raises_on_corpus_with_no_poisoned_docs():
    clean = [GeneratedDocument(text="clean", poisoned=False, carrier_name="c1")]
    with pytest.raises(ValueError):
        score_pipeline(_FakePipeline(leak_canaries=[]), clean)
