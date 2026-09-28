import base64

import pytest

from prompt_injection_harness.carriers import CARRIER_DOCUMENTS
from prompt_injection_harness.generator import (
    InjectionPosition,
    StealthLevel,
    build_corpus,
    inject_payload,
)
from prompt_injection_harness.payloads import PAYLOAD_LIBRARY, AttackCategory


CARRIER = CARRIER_DOCUMENTS[0]
PAYLOAD = PAYLOAD_LIBRARY[0]


def _canary_present(text: str, canary: str) -> bool:
    """True if the canary appears either as plaintext or inside a
    base64-encoded span (the obfuscation payload encodes the whole
    instruction, including the canary, rather than embedding it as
    plaintext)."""
    needle = f"CANARY-{canary}"
    if needle in text:
        return True
    for token in text.replace("\n", " ").split():
        stripped = token.strip("()")
        try:
            padded = stripped + "=" * (-len(stripped) % 4)
            decoded = base64.b64decode(padded, validate=True).decode("utf-8")
        except Exception:
            continue
        if needle in decoded:
            return True
    return False


@pytest.mark.parametrize("position", list(InjectionPosition))
def test_inject_payload_at_each_position(position):
    canary = "TESTCAN1"
    combined = inject_payload(CARRIER.text, PAYLOAD, canary, position=position)

    # The canary (and therefore the payload) must always be present.
    assert f"CANARY-{canary}" in combined
    # Original carrier content must be preserved somewhere in the output.
    assert CARRIER.text.split(".")[0] in combined


def test_inject_payload_start_puts_payload_before_carrier_body():
    canary = "STARTCAN"
    combined = inject_payload(
        CARRIER.text, PAYLOAD, canary, position=InjectionPosition.START
    )
    payload_idx = combined.index(f"CANARY-{canary}")
    carrier_idx = combined.index(CARRIER.text[:30])
    assert payload_idx < carrier_idx


def test_inject_payload_end_puts_payload_after_carrier_body():
    canary = "ENDCAN01"
    combined = inject_payload(
        CARRIER.text, PAYLOAD, canary, position=InjectionPosition.END
    )
    payload_idx = combined.index(f"CANARY-{canary}")
    carrier_idx = combined.index(CARRIER.text[:30])
    assert carrier_idx < payload_idx


def test_inject_payload_middle_is_surrounded_by_carrier_text():
    canary = "MIDCAN01"
    combined = inject_payload(
        CARRIER.text, PAYLOAD, canary, position=InjectionPosition.MIDDLE
    )
    payload_idx = combined.index(f"CANARY-{canary}")
    # Some carrier text should appear both before and after the payload.
    assert payload_idx > 0
    assert payload_idx < len(combined) - 1
    before = combined[:payload_idx]
    after = combined[payload_idx:]
    assert len(before.strip()) > 0
    assert any(c.isalpha() for c in after)


def test_build_corpus_labels_are_correct():
    corpus = build_corpus(
        PAYLOAD_LIBRARY,
        CARRIER_DOCUMENTS,
        positions=[InjectionPosition.END],
        stealth_levels=[StealthLevel.MEDIUM],
        include_clean=True,
        seed=42,
    )

    clean_docs = [d for d in corpus if not d.poisoned]
    poisoned_docs = [d for d in corpus if d.poisoned]

    # One clean doc per carrier.
    assert len(clean_docs) == len(CARRIER_DOCUMENTS)
    for doc in clean_docs:
        assert doc.attack_category is None
        assert doc.canary is None
        assert doc.injection_position is None

    # One poisoned doc per carrier x payload (single position/stealth combo).
    assert len(poisoned_docs) == len(CARRIER_DOCUMENTS) * len(PAYLOAD_LIBRARY)
    for doc in poisoned_docs:
        assert doc.attack_category is not None
        assert doc.injection_position == InjectionPosition.END
        assert doc.stealth_level == StealthLevel.MEDIUM
        assert doc.canary is not None
        assert _canary_present(doc.text, doc.canary)


def test_build_corpus_canaries_are_unique():
    corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=7)
    canaries = [d.canary for d in corpus if d.poisoned]
    assert len(canaries) == len(set(canaries))


def test_build_corpus_is_deterministic_for_a_given_seed():
    corpus_a = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=99)
    corpus_b = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=99)
    assert [d.canary for d in corpus_a] == [d.canary for d in corpus_b]
    assert [d.text for d in corpus_a] == [d.text for d in corpus_b]


def test_all_attack_categories_represented_in_library():
    categories = {p.category for p in PAYLOAD_LIBRARY}
    assert categories == set(AttackCategory)
