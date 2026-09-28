from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.generator import build_corpus
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY
from prompt_injection_harness.pipelines import (
    NaiveConcatPipeline,
    OpenAICompatibleTargetPipeline,
    SanitizingPipeline,
)
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


def test_openai_compatible_pipeline_sends_real_request_and_parses_response(monkeypatch):
    """Exercises the *real* HTTP integration code path: only the transport
    (the actual network socket) is stubbed, so this verifies the pipeline
    builds a correct OpenAI-compatible request and correctly parses a
    real-shaped response -- it does not stub the pipeline's own logic."""
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "real model reply"}}]}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse()

    import requests

    monkeypatch.setattr(requests, "post", _fake_post)

    pipeline = OpenAICompatibleTargetPipeline(model="gpt-4o-mini", api_key="sk-test")
    output = pipeline.run(["some context"], "what is this?")

    assert output == "real model reply"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-4o-mini"
    assert "some context" in captured["json"]["messages"][1]["content"]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
