"""Target pipelines: the "system under test".

``TargetPipeline`` is a minimal protocol any RAG/agent pipeline can
implement to be benchmarked by this harness. Two dependency-free, fully
offline, deterministic reference implementations are provided:

* :class:`NaiveConcatPipeline` -- simulates a pipeline with *no* defenses:
  it concatenates retrieved documents directly into a prompt-like string
  and simulates "instruction following" with simple substring/regex
  matching against the same category of triggers a naive/weak LLM would
  actually obey. This intentionally demonstrates a *vulnerable* system so
  the harness has something real to catch.

* :class:`SanitizingPipeline` -- wraps the naive pipeline with basic,
  realistic defenses (delimiter wrapping so untrusted content is clearly
  marked as data, instruction-keyword stripping/flagging, and best-effort
  base64 decode-and-strip) and is measurably more resistant. This is what
  makes the benchmark meaningful: it shows defense-in-depth reduces
  Attack Success Rate rather than just asserting it.

Neither reference pipeline calls a real LLM or the network -- they are
simple, legible simulations so the whole harness runs offline and
deterministically. To benchmark a *real* pipeline (one backed by an
actual LLM), implement :class:`TargetPipeline` yourself; see the README.
"""

from __future__ import annotations

import re
from typing import List, Protocol, runtime_checkable

from .detector import _HOMOGLYPH_CHARS, _PATTERNS, _looks_like_base64, is_likely_injection

_CANARY_RE = re.compile(r"CANARY-[A-Z0-9]{6,10}")


@runtime_checkable
class TargetPipeline(Protocol):
    """Protocol for the RAG/agent system under test.

    Implement this against your *real* pipeline (real retriever + real
    LLM call) to benchmark it with this harness -- nothing else in the
    library needs to change.
    """

    def run(self, context_docs: List[str], user_query: str) -> str:
        """Given retrieved context documents and a user query, return the
        pipeline's final response text."""
        ...


class NaiveConcatPipeline:
    """A deliberately vulnerable reference pipeline.

    Simulates the common failure mode of simply concatenating retrieved
    documents into a prompt with no trust boundary between "instructions"
    and "data". We can't call a real LLM offline, so we simulate what a
    susceptible LLM would do: if the concatenated context contains
    recognizable injection trigger phrasing, the simulated model "obeys"
    by emitting any canary token(s) present in that context verbatim in
    its response -- exactly the observable signal a real hijacked LLM
    would produce (attacker-controlled text appearing in the output where
    it should not be).
    """

    name = "naive_concat"

    def _looks_hijackable(self, context: str) -> bool:
        for group, patterns in _PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(context):
                    if group == "obfuscation_hint" and not _looks_like_base64(match.group(0)):
                        continue
                    return True
        # A real (non-simulated) LLM typically still reads straight through
        # homoglyph substitution -- it is a *rendering* trick aimed at
        # human reviewers and naive keyword filters, not at the model's
        # own tokenizer/attention. We simulate that realistic weakness
        # here: homoglyph-obscured text is treated as hijackable even
        # though it fails every literal-ASCII keyword pattern above.
        if _HOMOGLYPH_CHARS.search(context):
            return True
        return False

    def run(self, context_docs: List[str], user_query: str) -> str:
        context = "\n---\n".join(context_docs)
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {user_query}\nANSWER:"
        )

        leaked = ""
        if self._looks_hijackable(context):
            canaries = _CANARY_RE.findall(context)
            if canaries:
                # A hijacked model parrots the attacker-controlled text it
                # was told to output, verbatim, in its answer.
                leaked = " " + " ".join(canaries)

        return f"Based on the provided documents, here is my answer to '{user_query}'.{leaked}"

        # NOTE: `prompt` above mirrors what a real pipeline would send to
        # an LLM; it is unused by this simulation beyond illustrating the
        # (missing) trust boundary between instructions and retrieved data.


class SanitizingPipeline:
    """A reference pipeline with basic, realistic defenses applied.

    Defenses, applied to every retrieved document *before* it is added to
    the context:

    1. Delimiter wrapping -- each document is wrapped in explicit
       "untrusted data" delimiters, so a well-behaved downstream model
       (or, here, our detection simulation) can distinguish data from
       instructions.
    2. Instruction-keyword stripping -- spans matching known injection
       trigger patterns are redacted before the document ever reaches the
       "model".
    3. Base64 decode-and-strip -- long base64-looking runs are decoded;
       if the decoded text itself looks like an injection attempt, the
       whole blob is redacted rather than passed through.

    These defenses are not perfect (nothing regex-based is), which is the
    point: the benchmark should show a *reduced*, not necessarily zero,
    Attack Success Rate, honestly reflecting that defense-in-depth helps
    without being a silver bullet.
    """

    name = "sanitizing"
    REDACTION = "[REDACTED-POTENTIAL-INSTRUCTION]"

    def _strip_known_triggers(self, text: str) -> str:
        cleaned = text
        for group, patterns in _PATTERNS.items():
            if group == "obfuscation_hint":
                continue  # handled separately by _strip_base64
            for pattern in patterns:
                cleaned = pattern.sub(self.REDACTION, cleaned)
        return cleaned

    def _strip_base64(self, text: str) -> str:
        def _maybe_redact(match: "re.Match[str]") -> str:
            candidate = match.group(0)
            if _looks_like_base64(candidate):
                return self.REDACTION
            return candidate

        return re.sub(r"[A-Za-z0-9+/]{40,}={0,2}", _maybe_redact, text)

    def _sanitize(self, doc: str) -> str:
        doc = self._strip_base64(doc)
        doc = self._strip_known_triggers(doc)
        return f"<<DOCUMENT untrusted=\"true\">>\n{doc}\n<<END DOCUMENT>>"

    def run(self, context_docs: List[str], user_query: str) -> str:
        sanitized_docs = [self._sanitize(d) for d in context_docs]
        # Delegate to the same "hijackable simulation" logic as the naive
        # pipeline, but over the sanitized context -- this is what lets the
        # benchmark demonstrate a measurable ASR reduction rather than a
        # separately-hand-tuned number.
        naive = NaiveConcatPipeline()
        return naive.run(sanitized_docs, user_query)
