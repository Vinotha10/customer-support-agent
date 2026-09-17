"""
Deterministic escalation-trigger checks. These are NOT invented in the
abstract -- every category here was pulled directly from patterns that
actually showed up during golden-set hand-labeling (see
outputs/eval/day3_failure_analysis_notes.md and the labeler_notes column
in golden_set_labeled.csv). That traceability matters: when asked to
defend a specific escalation rule, the answer is "this exact pattern
appeared in golden-set row X," not "it seemed like a good idea."

These rules are checked BEFORE the LLM drafts a reply -- they're passed
into the drafting prompt as explicit signals the LLM must factor into its
escalate/reason output, rather than relying on the LLM to notice them
unprompted. Deterministic rules catch things reliably; LLM judgment
handles severity/nuance the rules can't express (e.g. "multiple symptoms
reported simultaneously").
"""
import re

# g0135: swollen battery is a fire/explosion hazard -- always escalate,
# independent of classifier confidence or intent.
SAFETY_PATTERNS = [
    r"\bswoll?en?\b", r"\bswelling\b", r"\bfire\b", r"\bexplod\w*\b",
    r"\bsmoke\b", r"\bburn(ing|t)?\b",
]

# g0054: explicit legal threats trigger escalation regardless of the
# underlying issue being a known/trivial bug.
LEGAL_PATTERNS = [
    r"\bsue\b", r"\blawsuit\b", r"\blawyer\b", r"\battorney\b",
    r"\blegal action\b",
]

# g0004, g0033, g0053, g0088, g0135, g0179, g0197: billing/refund/warranty
# cost disputes recurred often enough in labeling to warrant a standing
# rule -- these involve real money and were consistently escalated.
FINANCIAL_PATTERNS = [
    r"\brefund\b", r"\bcharged?\b", r"\bbill(ed|ing)?\b", r"\bdispute\b",
    r"\bmoney back\b", r"\bduplicate charge\b", r"\bpay(ing)? for (a |the )?repair\b",
]

# g0104, g0114, g0147, g0172, g0185, g0187, g0191, g0199: a recurring
# pattern where the customer states they ALREADY tried the standard fix
# (often the exact update/workaround that would otherwise be suggested)
# and it didn't work -- escalate rather than repeat the same suggestion.
ALREADY_TRIED_PATTERNS = [
    r"\balready (tried|did|updated|restarted|done)\b",
    r"\bstill (happening|broken|seeing|not working)\b",
    r"\bdidn'?t (help|work|fix)\b",
    r"\btried (that|this) (already|before)\b",
    r"\bdone (all|everything)\b",
]

# g0093 (Spanish), g0132 (French): non-English messages need routing to
# language-appropriate support rather than an English-only auto-reply.
# Heuristic only -- not a real language detector, but catches the two
# concrete cases found during labeling reasonably well: a low ratio of
# common English function words relative to message length.
_ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "my", "and", "to", "a", "of", "in", "it", "on",
    "for", "with", "this", "that", "i", "you", "your", "was", "have",
}


def _compile(patterns):
    return re.compile("|".join(patterns), re.IGNORECASE)


_safety_re = _compile(SAFETY_PATTERNS)
_legal_re = _compile(LEGAL_PATTERNS)
_financial_re = _compile(FINANCIAL_PATTERNS)
_already_tried_re = _compile(ALREADY_TRIED_PATTERNS)


def looks_non_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 5:
        return False  # too short to judge reliably, don't flag
    function_word_count = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    return (function_word_count / len(words)) < 0.08


def check_rules(customer_text: str, prior_context: str = None,
                 classifier_confidence: float = None, confidence_threshold: float = 0.40) -> dict:
    """
    Returns a dict of {rule_name: (triggered: bool, evidence: str|None)}.
    Checks both customer_text and prior_context where relevant, since
    several golden-set cases only revealed the real issue via prior context
    (see day3 notes / labeler_notes for g0055, g0171).
    """
    combined_text = customer_text or ""
    if prior_context:
        combined_text = f"{prior_context} {combined_text}"

    results = {}

    m = _safety_re.search(combined_text)
    results["safety_hazard"] = (bool(m), m.group(0) if m else None)

    m = _legal_re.search(combined_text)
    results["legal_threat"] = (bool(m), m.group(0) if m else None)

    m = _financial_re.search(combined_text)
    results["financial_dispute"] = (bool(m), m.group(0) if m else None)

    m = _already_tried_re.search(combined_text)
    results["remedy_already_exhausted"] = (bool(m), m.group(0) if m else None)

    non_eng = looks_non_english(customer_text or "")
    results["non_english"] = (non_eng, "low English function-word ratio" if non_eng else None)

    if classifier_confidence is not None:
        low_conf = classifier_confidence < confidence_threshold
        results["low_classifier_confidence"] = (
            low_conf,
            f"confidence {classifier_confidence:.2f} < threshold {confidence_threshold:.2f}" if low_conf else None,
        )
    else:
        results["low_classifier_confidence"] = (False, None)

    return results


def any_rule_triggered(rule_results: dict) -> bool:
    return any(triggered for triggered, _ in rule_results.values())


def summarize_triggered(rule_results: dict) -> list:
    return [f"{name}: {evidence}" for name, (triggered, evidence) in rule_results.items() if triggered]


if __name__ == "__main__":
    # quick manual smoke test
    examples = [
        ("my rMBP battery is swollen with just 100 cycles", None),
        ("gonna sue you for emotional distress over these question marks", None),
        ("charged me three times for one failed purchase, want a refund", None),
        ("I already restarted and updated to 11.1.1 and it's still happening", None),
        ("Desde que instale la ultima actualizacion mi iPhone es mas lento", None),
        ("my battery drains fast since the update", None),
    ]
    for text, prior in examples:
        result = check_rules(text, prior, classifier_confidence=0.6)
        triggered = summarize_triggered(result)
        print(f"{text[:60]!r:65s} -> {triggered if triggered else 'no rules triggered'}")
