"""prompt-injection-harness

A deterministic, fully offline test harness for evaluating whether a
RAG/agent pipeline is vulnerable to prompt injection carried inside
retrieved documents (indirect / "second-order" prompt injection).

Public API surface is re-exported here for convenience:

    from prompt_injection_harness import (
        AttackCategory, PAYLOAD_LIBRARY,
        CARRIER_DOCUMENTS,
        InjectionPosition, StealthLevel, build_corpus, GeneratedDocument,
        TargetPipeline, NaiveConcatPipeline, SanitizingPipeline,
        is_likely_injection, evaluate_detector,
        score_pipeline, ScoreResult,
    )
"""

from .payloads import AttackCategory, PayloadTemplate, PAYLOAD_LIBRARY
from .carriers import CarrierDocument, CARRIER_DOCUMENTS
from .generator import (
    InjectionPosition,
    StealthLevel,
    GeneratedDocument,
    inject_payload,
    build_corpus,
)
from .pipelines import TargetPipeline, NaiveConcatPipeline, SanitizingPipeline
from .detector import is_likely_injection, evaluate_detector, DetectorMetrics
from .scorer import score_pipeline, ScoreResult

__version__ = "0.1.0"

__all__ = [
    "AttackCategory",
    "PayloadTemplate",
    "PAYLOAD_LIBRARY",
    "CarrierDocument",
    "CARRIER_DOCUMENTS",
    "InjectionPosition",
    "StealthLevel",
    "GeneratedDocument",
    "inject_payload",
    "build_corpus",
    "TargetPipeline",
    "NaiveConcatPipeline",
    "SanitizingPipeline",
    "is_likely_injection",
    "evaluate_detector",
    "DetectorMetrics",
    "score_pipeline",
    "ScoreResult",
]
