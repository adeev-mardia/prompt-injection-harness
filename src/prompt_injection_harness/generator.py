"""Builds a labeled synthetic corpus of poisoned / clean documents.

The generator takes clean carrier documents and injects payloads from the
template library at a controllable position (start/middle/end) and a
controllable "stealth" level (how much the injected text is disguised to
blend with the surrounding content), producing labeled
:class:`GeneratedDocument` records suitable for benchmarking a target
pipeline and evaluating a detector.
"""

from __future__ import annotations

import random
import secrets
import string
from enum import Enum
from typing import List, Optional, Sequence

from pydantic import BaseModel

from .carriers import CarrierDocument
from .payloads import AttackCategory, PayloadTemplate


class InjectionPosition(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


class StealthLevel(str, Enum):
    """How much the injected payload is disguised.

    LOW:    payload is wrapped in loud, obvious markers.
    MEDIUM: payload is inserted as a plain, unmarked paragraph.
    HIGH:   payload is disguised to blend with the carrier's own style
            (quoted as an aside, wrapped in an HTML-comment-like span, or
            phrased as a footnote) -- the hardest case for both the
            target pipeline's defenses and the standalone detector.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GeneratedDocument(BaseModel):
    text: str
    poisoned: bool
    attack_category: Optional[AttackCategory] = None
    injection_position: Optional[InjectionPosition] = None
    stealth_level: Optional[StealthLevel] = None
    carrier_name: str
    payload_name: Optional[str] = None
    canary: Optional[str] = None


def _make_canary(rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(8))


def _split_sentences(text: str) -> List[str]:
    # Simple, dependency-free sentence splitter sufficient for our
    # synthetic carrier documents.
    parts = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in ".\n" and len(current) > 1:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _wrap_for_stealth(rendered_payload: str, stealth: StealthLevel) -> str:
    if stealth == StealthLevel.LOW:
        return (
            "\n\n### SYSTEM OVERRIDE -- READ FIRST ###\n"
            f"{rendered_payload}\n"
            "### END OVERRIDE ###\n\n"
        )
    if stealth == StealthLevel.MEDIUM:
        return f"\n\n{rendered_payload}\n\n"
    # HIGH: blend in as a quiet aside / HTML-comment-like span, no loud markers
    return f" <!-- note: {rendered_payload} --> "


def inject_payload(
    carrier_text: str,
    payload: PayloadTemplate,
    canary: str,
    position: InjectionPosition = InjectionPosition.END,
    stealth: StealthLevel = StealthLevel.MEDIUM,
) -> str:
    """Insert a rendered payload into ``carrier_text`` at the given position.

    Returns the combined document text. Does not mutate the inputs.
    """
    rendered = payload.render(canary)
    chunk = _wrap_for_stealth(rendered, stealth)

    if position == InjectionPosition.START:
        return chunk.strip() + "\n\n" + carrier_text

    if position == InjectionPosition.END:
        return carrier_text + "\n\n" + chunk.strip()

    # MIDDLE: split into sentences and splice the payload in around the
    # midpoint, so it is surrounded by carrier text on both sides.
    sentences = _split_sentences(carrier_text)
    if len(sentences) < 2:
        return carrier_text + "\n\n" + chunk.strip()
    mid = len(sentences) // 2
    before = "".join(sentences[:mid])
    after = "".join(sentences[mid:])
    return before + chunk + after


def build_corpus(
    payload_library: Sequence[PayloadTemplate],
    carriers: Sequence[CarrierDocument],
    positions: Sequence[InjectionPosition] = tuple(InjectionPosition),
    stealth_levels: Sequence[StealthLevel] = tuple(StealthLevel),
    include_clean: bool = True,
    seed: int = 1337,
) -> List[GeneratedDocument]:
    """Build a labeled corpus of poisoned and (optionally) clean documents.

    For every combination of carrier x payload x position x stealth level,
    one poisoned :class:`GeneratedDocument` is produced with a unique
    canary token. When ``include_clean`` is True, every carrier document
    is also included once, unmodified, with ``poisoned=False``.

    The corpus is fully deterministic for a given ``seed``.
    """
    rng = random.Random(seed)
    docs: List[GeneratedDocument] = []

    if include_clean:
        for carrier in carriers:
            docs.append(
                GeneratedDocument(
                    text=carrier.text,
                    poisoned=False,
                    carrier_name=carrier.name,
                )
            )

    for carrier in carriers:
        for payload in payload_library:
            for position in positions:
                for stealth in stealth_levels:
                    canary = _make_canary(rng)
                    text = inject_payload(
                        carrier.text, payload, canary, position, stealth
                    )
                    docs.append(
                        GeneratedDocument(
                            text=text,
                            poisoned=True,
                            attack_category=payload.category,
                            injection_position=position,
                            stealth_level=stealth,
                            carrier_name=carrier.name,
                            payload_name=payload.name,
                            canary=canary,
                        )
                    )
    return docs


def secure_canary(nbytes: int = 6) -> str:
    """Cryptographically-random canary, for callers who need unpredictability
    beyond the deterministic seeded generator (e.g. one-off manual tests)."""
    return secrets.token_hex(nbytes).upper()
