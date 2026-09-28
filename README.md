# prompt-injection-harness

A fully offline, deterministic test harness for evaluating whether a
RAG/agent pipeline is vulnerable to **prompt injection via poisoned
documents** -- one of the most concrete, actively-exploited LLM security
issues in production systems today.

## Why this matters

Any pipeline that retrieves untrusted text and hands it to an LLM (RAG
over a knowledge base, an agent that reads web pages or tickets, a
browsing/tool-using assistant) faces a trust-boundary problem: **the
model cannot reliably tell the difference between "data it was asked to
summarize" and "instructions it should obey"**. If an attacker can get
even a small amount of text into the retrieval corpus -- a product review,
a support ticket, a wiki edit, a web page the agent later fetches -- they
can attempt to hijack the model's behavior. This is *indirect* (or
"second-order") prompt injection, and it has been demonstrated
repeatedly against real RAG and browsing-agent systems: the attacker
never talks to the model directly, they just poison something it will
later read.

This harness treats that risk the way a security engineer would treat
any other input-validation problem: build a labeled corpus of known
attack patterns, run it through the system under test, measure how often
the attack actually succeeds (not just "does the text look suspicious"),
and separately measure how good a cheap pre-filter is at catching the
attempt before it ever reaches the model.

## How it works end to end

```
payloads.py (attack templates)         carriers.py (clean sample docs)
        \                                       /
         \                                     /
          v                                   v
              generator.py: build_corpus()
                          |
                          v
        labeled corpus: GeneratedDocument(text, poisoned,
                         attack_category, injection_position,
                         stealth_level, canary)
                    /              \
                   v                v
        detector.py               pipelines.py: TargetPipeline
   (pre-filter, runs BEFORE         (the RAG/agent system under
    a doc reaches the model)         test -- yours, or a reference)
                   \                /
                    v              v
                  scorer.py: score_pipeline()
                (Attack Success Rate, canary-based)
```

1. **`payloads.py`** is a categorized library of synthetic injection
   templates covering six distinct, real attack techniques (see below).
   Each template contains a `{canary}` placeholder.
2. **`carriers.py`** holds a handful of realistic clean documents
   (product reviews, support tickets, wiki articles) that stand in for
   "things a retriever might legitimately return."
3. **`generator.py`** builds a labeled corpus by injecting every payload
   into every carrier at every combination of **position**
   (start/middle/end) and **stealth level** (low/medium/high -- how much
   the payload is disguised to blend with the surrounding content), each
   with a unique, unpredictable canary token. It also includes the
   unmodified clean carriers as negative examples. The whole process is
   seeded and fully deterministic.
4. **`pipelines.py`** defines the `TargetPipeline` protocol (`.run(context_docs, user_query) -> str`)
   and two offline reference implementations: a deliberately vulnerable
   `NaiveConcatPipeline` and a defended `SanitizingPipeline`.
5. **`scorer.py`** runs the poisoned corpus through a pipeline and checks
   whether each document's unique canary token leaked into the output --
   unambiguous proof the injected instruction, not the user's actual
   query, drove that part of the response. It reports Attack Success
   Rate (ASR) overall and broken down by category/position/stealth.
6. **`detector.py`** is a standalone, reusable heuristic classifier that
   flags a document as "likely contains a prompt injection attempt"
   *before* it is ever retrieved into a prompt, evaluated with
   precision/recall/F1 against the same labeled corpus.

### Attack categories in the payload library

| Category | Real-world pattern |
|---|---|
| `direct_override` | "Ignore all previous instructions and..." |
| `fake_system_message` | Text pretending to be a system/developer message embedded in content |
| `data_exfiltration` | Asking the model to reveal its system prompt or repeat prior context verbatim |
| `jailbreak_roleplay` | "You are now DAN..." / "act as an unrestricted assistant..." |
| `obfuscation` | Base64-encoded instructions; Cyrillic-homoglyph-obscured text |
| `indirect_delayed` | Instructions buried deep in otherwise-benign content (e.g. a fake product review) |

Two templates per category, 12 total, each independently combined with
3 positions x 3 stealth levels x 6 carriers.

### Canary-based scoring, honestly

Since the reference pipelines don't call a real LLM (the whole harness
is offline/deterministic on purpose), `NaiveConcatPipeline` and
`SanitizingPipeline` *simulate* "did the model get hijacked" using the
same class of pattern-matching a real weak/naive LLM would visibly act
on (recognizable instruction phrasing, decoded base64, homoglyph text).
This is a deliberate simplification: it lets the whole benchmark run in
milliseconds with no API key and no network call, at the cost of not
capturing a real model's semantic understanding of a *fully* disguised
instruction with no recognizable trigger phrasing at all (see
`disguised_footnote_instruction` in `payloads.py`, which is deliberately
built to test that edge). Plug in a real pipeline (see below) to get a
real answer for your model.

## Reference pipelines

* **`NaiveConcatPipeline`** -- no trust boundary at all: retrieved
  documents are concatenated straight into a prompt-like string. If
  recognizable injection phrasing is present, it "leaks" any canary
  token found in context into its output, simulating a hijacked model
  parroting attacker-controlled text.
* **`SanitizingPipeline`** -- applies three concrete defenses before any
  document reaches the (simulated) model:
  1. **Delimiter wrapping** -- every document is wrapped in explicit
     `<<DOCUMENT untrusted="true">> ... <<END DOCUMENT>>` markers.
  2. **Instruction-keyword stripping** -- spans matching known injection
     trigger patterns (the same pattern library `detector.py` uses) are
     redacted before the document is added to context.
  3. **Base64 decode-and-strip** -- long base64-looking runs are decoded
     and checked; if the decoded text itself looks like an injection
     attempt, the whole blob is redacted.

## Results (from `scripts/benchmark.py`, run offline, seed=1337)

Corpus: 6 carriers x 12 payload templates x 3 positions x 3 stealth
levels = **648 poisoned documents**, plus 6 clean carrier documents.

### Attack Success Rate: Naive vs. Sanitizing

| Pipeline | Overall ASR | False positive rate (clean docs) |
|---|---|---|
| `NaiveConcatPipeline` | **83.3%** (540/648) | 0.0% |
| `SanitizingPipeline`  | **8.3%** (54/648)   | 0.0% |

That's a **90% relative reduction** in Attack Success Rate from basic,
regex-based defense-in-depth -- a real, measured effect, not an
aspirational claim. Position (start/middle/end) and stealth level made
no measurable difference to either pipeline's ASR in this corpus, since
both reference pipelines scan the *entire* concatenated context rather
than being position-sensitive -- a real LLM's susceptibility to position
is a good candidate extension (see Limitations).

ASR by attack category:

| Category | Naive | Sanitizing |
|---|---|---|
| `data_exfiltration` | 100.0% | 0.0% |
| `direct_override` | 100.0% | 0.0% |
| `fake_system_message` | 100.0% | 0.0% |
| `jailbreak_roleplay` | 100.0% | 0.0% |
| `obfuscation` | 50.0% | 50.0% |
| `indirect_delayed` | 50.0% | 0.0% |

The one category `SanitizingPipeline` does **not** meaningfully improve
on is `obfuscation` -- specifically the homoglyph-based template. The
sanitizer's keyword stripping is literal-ASCII regex; it has no
Unicode/homoglyph normalization step, so `Іgnоre prеviоus instructіons`
(Cyrillic lookalikes) sails straight through untouched. This is a real,
common gap in naive regex-based defenses, and exactly the kind of
concrete, falsifiable finding this harness is meant to surface -- not a
bug in the benchmark, but the benchmark doing its job. (A homoglyph
normalization pass, e.g. Unicode NFKC + a confusables table, would close
this gap; left as a documented extension rather than papering over the
number.)

### Heuristic detector: precision / recall / F1

Evaluated against the full 654-document labeled corpus:

| Metric | Value |
|---|---|
| Precision | **1.000** |
| Recall | **0.917** |
| F1 | **0.957** |
| True positives | 594 |
| False positives | 0 |
| False negatives | 54 |
| True negatives | 6 |

Recall by attack category:

| Category | Recall |
|---|---|
| `data_exfiltration` | 100.0% |
| `direct_override` | 100.0% |
| `fake_system_message` | 100.0% |
| `jailbreak_roleplay` | 100.0% |
| `obfuscation` | 100.0% |
| `indirect_delayed` | 50.0% |

The detector has **zero false positives** on the clean carrier
documents in this corpus and near-perfect recall on categories with
recognizable trigger phrasing. Its one soft spot is by design:
`disguised_footnote_instruction` in the `indirect_delayed` category was
deliberately written with *no* keyword-matchable trigger phrase at all
(it reads as an innocuous editorial footnote), specifically to
stress-test the limits of a keyword-based detector. That it evades
detection is the expected, honest result of a regex approach -- it is
also exactly the class of attack that most benefits from a second layer
of defense (e.g. the `SanitizingPipeline`'s delimiter wrapping, or a
model-based classifier) rather than relying on keyword matching alone.

Numbers cross-checked against `sklearn.metrics.precision_recall_fscore_support`
in `scripts/benchmark.py` (matches the hand-rolled computation in
`detector.py` exactly). Reproduce with:

```bash
python3 scripts/benchmark.py
python3 scripts/detector_eval.py   # confusion matrix + sklearn classification_report
```

## Plugging in your own pipeline

Nothing in this library needs to change to benchmark a *real* RAG/agent
system. Implement the `TargetPipeline` protocol:

```python
from prompt_injection_harness import TargetPipeline

class MyRealPipeline:
    def run(self, context_docs: list[str], user_query: str) -> str:
        # 1. build your real prompt from context_docs + user_query
        # 2. call your real retriever/LLM/agent
        # 3. return the final response text
        prompt = build_prompt(context_docs, user_query)
        return my_llm_client.generate(prompt)
```

Then run the same corpus and scorer against it:

```python
from prompt_injection_harness import (
    PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, build_corpus, score_pipeline,
)

corpus = build_corpus(PAYLOAD_LIBRARY, CARRIER_DOCUMENTS, seed=1337)
result = score_pipeline(MyRealPipeline(), corpus)
print(result.overall_asr, result.asr_by_category)
```

Because scoring is canary-based (did *this specific document's* unique
token leak into the output) rather than keyword-based on the output,
this works unmodified against a real LLM-backed pipeline -- you get a
genuine ASR number instead of the pattern-matching simulation the two
reference pipelines use offline.

## CLI

```bash
# Run the full corpus through a reference pipeline and print an ASR table
injection-harness benchmark --pipeline naive
injection-harness benchmark --pipeline sanitizing
injection-harness benchmark --pipeline all   # default

# Run the standalone heuristic detector on a single document
injection-harness detect suspicious_ticket.txt
```

## Installation

```bash
pip install -e ".[dev]"
```

## Running the tests

```bash
python3 -m pytest -v
```

30 tests, fully offline and deterministic: payload injection at each
position, corpus label correctness, canary uniqueness and
determinism-under-seed, `NaiveConcatPipeline`'s measured vulnerability,
`SanitizingPipeline`'s measured ASR reduction, detector precision/recall
including false-positive behavior on clean docs, and the scorer's
per-category ASR math.

## Project layout

```
src/prompt_injection_harness/
    __init__.py      public API re-exports
    payloads.py       categorized attack payload template library
    carriers.py       sample clean carrier documents
    generator.py      builds the labeled synthetic corpus
    pipelines.py      TargetPipeline protocol + Naive/Sanitizing reference pipelines
    detector.py       heuristic injection detector + evaluation
    scorer.py         canary-based Attack Success Rate scoring
    cli.py            `injection-harness` command-line interface
scripts/
    benchmark.py      Naive vs Sanitizing ASR + detector precision/recall report
    detector_eval.py  sklearn confusion matrix / classification report
tests/                pytest suite (30 tests)
```

## Limitations (read before treating any number here as gospel)

* The reference pipelines are **regex-based simulations of LLM
  behavior**, not real LLM calls -- necessary to keep the whole harness
  offline and deterministic, but it means their ASR numbers describe
  "does this pattern of text match a known trigger," not "would a real
  model actually comply." A real model may be more robust (better
  instruction-hierarchy training) or less robust (more willing to follow
  subtly-phrased instructions with no obvious keywords, like
  `disguised_footnote_instruction`) than these simulations suggest.
  Point the harness at a real `TargetPipeline` implementation for a real
  answer.
* The detector and `SanitizingPipeline` are deliberately simple
  (regex/keyword-based) so they are fast, auditable, and dependency-free
  -- they are a reusable first line of defense, not a complete solution.
  Both have a documented, honest gap on Unicode homoglyph obfuscation.
* The synthetic corpus is intentionally small and hand-authored so every
  result is inspectable; it is not a substitute for red-teaming against
  a live system with real, adaptive adversarial input.

## License

MIT -- see `LICENSE`.
