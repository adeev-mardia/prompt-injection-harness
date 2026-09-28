from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.generator import build_corpus
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY
from prompt_injection_harness.pipelines import NaiveConcatPipeline, SanitizingPipeline
from prompt_injection_harness.scorer import score_pipeline


def _corpus():
    return build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=1)


def test_naive_pipeline_is_measurably_vulnerable():
    result = score_pipeline(NaiveConcatPipeline(), _corpus())
    assert result.overall_asr > 0.0
    assert result.total_successful_attacks > 0


def test_naive_pipeline_leaks_canary_for_direct_override():
    from prompt_injection_harness.payloads import AttackCategory

    pipeline = NaiveConcatPipeline()
    payload = next(
        p for p in PAYLOAD_LIBRARY if p.category == AttackCategory.DIRECT_OVERRIDE
    )
    canary = "ABC12345"
    poisoned_doc = payload.render(canary)
    output = pipeline.run([poisoned_doc], "What does this document say?")
    assert f"CANARY-{canary}" in output


def test_naive_pipeline_does_not_leak_canary_for_clean_docs():
    pipeline = NaiveConcatPipeline()
    for carrier in CARRIER_DOCUMENTS:
        output = pipeline.run([carrier.text], "Summarize this.")
        assert "CANARY-" not in output


def test_sanitizing_pipeline_reduces_asr_versus_naive():
    corpus = _corpus()
    naive_result = score_pipeline(NaiveConcatPipeline(), corpus)
    sanitizing_result = score_pipeline(SanitizingPipeline(), corpus)

    assert sanitizing_result.overall_asr < naive_result.overall_asr
    assert sanitizing_result.overall_asr >= 0.0


def test_sanitizing_pipeline_no_false_positives_on_clean_docs():
    pipeline = SanitizingPipeline()
    for carrier in CARRIER_DOCUMENTS:
        output = pipeline.run([carrier.text], "Summarize this.")
        assert "CANARY-" not in output


def test_sanitizing_pipeline_wraps_documents_in_delimiters():
    pipeline = SanitizingPipeline()
    sanitized = pipeline._sanitize("hello world")
    assert sanitized.startswith("<<DOCUMENT")
    assert sanitized.rstrip().endswith("<<END DOCUMENT>>")
