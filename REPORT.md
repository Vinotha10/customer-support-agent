# Report — AI Support Agent for AppleSupport

## 1. Problem framing

**What "good" means for this brand.** AppleSupport's Twitter support
traffic is dominated by a small number of recognizable, previously-seen
issue types (a keyboard autocorrect bug, post-update battery drain,
connectivity glitches) alongside a long tail of vague or one-off
complaints. For this brand, "good" means three specific things, in
priority order:

1. **High recall on escalation for anything genuinely high-stakes** —
   safety language, legal threats, financial asks, data loss, or a
   customer who states they already tried the standard fix and it
   failed. Missing these is the failure mode that actually damages trust
   and, in the safety cases, could cause real harm. This matters more
   than overall reply quality.
2. **Correct, on-topic drafts for the clear, recurring issue types** —
   the keyboard bug, standard battery/performance complaints — where a
   grounded, accurate reply genuinely saves a human's time.
3. **Honesty about the taxonomy's edges** — when a message doesn't fit
   any of the defined intents, the system should say so (`other`) rather
   than force a confident wrong answer.

**What was deliberately not built**, given a 5-6 day scope:
- No multi-turn conversational memory beyond the immediately preceding
  tweet in a thread (`prior_context`) — real account history, order
  lookups, or CRM integration were out of scope for a text-only dataset.
- No non-English language handling. Spanish and French messages appear
  in the real data; the system currently drafts an English reply
  regardless. This is a stated scope boundary, not an oversight — see
  Failure Analysis.
- No deployment/UI layer — the focus was the ML/eval core the brief
  actually asks to be defended, not a demo frontend.
- **A dedicated rule-based pre-filter ahead of the LLM escalation
  call, layered before the drafting step, was deferred as "future
  work" rather than built first.** In hindsight, given the results
  below, this was the wrong prioritization — see Section 4.

## 2. Results vs. baselines

### Intent classification (n=196 in-taxonomy golden-set rows; 5 rows are `other`, reported separately)

| Tier | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always predicts majority class) | 0.107 | 0.024 |
| **TF-IDF + LogisticRegression** (trained on 101,116 cluster-derived labels) | **0.735** | **0.717** |
| Few-shot LLM (Gemini, zero training data) | 0.658 | 0.564 |

The trained classifier beat the zero-shot LLM. This is explainable, not
disappointing: the taxonomy itself was derived from clustering the same
training data, giving the trained model structural home-field advantage,
and domain slang (the keyboard bug is called "boxes" in the wild) is
something the trained model saw thousands of times versus one example in
the LLM's prompt. The LLM's one clear advantage: it correctly flagged
**5/5** true out-of-taxonomy messages as `other`, something the 8-class
trained classifier structurally cannot do.

### Escalation decision (n=200)

| Metric | Value |
|---|---|
| Precision | 0.933 |
| **Recall** | **0.149** |
| F1 | 0.257 |

Confusion matrix:

|  | gold = escalate | gold = auto-handle |
|---|---|---|
| **pred = escalate** | 14 | 1 |
| **pred = auto-handle** | 80 | 105 |

**This is the headline result of the project, and it's bad in a specific,
important way.** See Section 3.

### Reply quality (LLM-as-judge, n=200; 1-5 scale)

| Dimension | Mean score |
|---|---|
| Correctness | `[[INSERT: mean of judge_scores.csv 'correctness' column]]` |
| Tone | `[[INSERT: mean of judge_scores.csv 'tone' column]]` |
| Actionability | `[[INSERT: mean of judge_scores.csv 'actionability' column]]` |

**Judge-vs-human agreement** (n=25 hand-scored calibration rows, Spearman
correlation):

| Dimension | Spearman ρ | p-value | % within 1 point |
|---|---|---|---|
| Correctness | 0.406 | 0.044 | 76% |
| Tone | 0.815 | <0.001 | 88% |
| Actionability | 0.338 | 0.098 (not significant) | 64% |
| Overall | 0.614 | 0.001 | 64% |

**The judge should be trusted differently across dimensions.** It agrees
strongly with a human on tone (a surface-level, easy-to-assess quality)
but only weakly on correctness, and the actionability correlation is not
statistically significant at this sample size. Practically: the tone
score is reliable; the correctness and actionability scores should be
read as a rough signal, not a validated metric, until the calibration
sample is grown past 25.

## 3. Failure analysis: top 5 failure modes

### 1. Escalation recall crisis — the agent systematically fails to act on its own rubric
80 of 94 messages that genuinely needed escalation were auto-handled
instead. The rubric embedded in the agent's prompt explicitly lists
"customer states they already tried the standard fix and it did not
work" as an escalate trigger — and this exact pattern appears, unescalated,
across a large share of the false negatives (e.g. a customer stating
they already updated to the version meant to fix the keyboard bug and
still see the issue). A second, starker example: a message containing an
**explicit legal threat** ("gonna sue you") was auto-handled despite the
rubric stating legal threats should always escalate "regardless of
known-bug status."

**Hypothesis:** when a message matches a recognizable, previously-seen
issue pattern (the keyboard bug, a battery complaint), the model appears
to strongly default to "give the standard fix," and this pattern-match
suppresses attention to other escalation triggers present in the *same*
message. The combined draft+escalate prompt may be asking the model to
do two cognitively different things at once (empathetic drafting,
rubric-based auditing) and the audit gets less effort than the draft.

### 2. Non-English messages are drafted in English rather than routed
Both Spanish- and French-language messages in the golden set were
auto-handled with (presumably) an English-language reply rather than
flagged for language-appropriate routing, despite the escalation rubric
implicitly covering "not confident a standard reply would resolve this."

**Hypothesis:** the rubric doesn't explicitly name language mismatch as
a trigger — this is a rubric-completeness gap, not a comprehension
failure, since the model would very likely recognize the language if
asked directly.

### 3. Severity-based triggers under-weighted relative to keyword-based ones
Cases with extreme, quantified severity (battery life reported in
*minutes* rather than hours; a device reported completely dead/DOA) were
inconsistently escalated — some DOA cases were caught, others weren't.
Safety-adjacent but non-alarming phrasing (a swollen battery mentioned
without the word "danger" or "fire") is a plausible related risk not
directly tested here.

**Hypothesis:** the model responds reliably to explicit, named triggers
("refund," "hacked") but less reliably to triggers that require
quantitative reasoning about severity ("18 percentage points in under 5
minutes" implies urgency but isn't a keyword match).

### 4. A confound the project cannot currently resolve: model-mixing
Due to API quota exhaustion and model deprecations during the golden-set
run, the 200 replies were drafted by a mix of at least two different
underlying models (a Gemini Flash-tier model for roughly the first 150
rows, a small local Llama model for the remainder). **`agent_outputs.csv`
does not record which model handled which row.** This means the 0.149
recall figure cannot currently be broken down by model — it's possible
the smaller local model accounts for a disproportionate share of the
misses, or it's possible the failure is uniform across models. This is
an honest, structural gap in this evaluation, not a hidden one.

### 5. Retrieval grounds on some historical replies that were themselves poor
At least one case surfaced during golden-set labeling where the brand's
own historical reply was mismatched to the customer's actual complaint
(a lock-screen issue answered with information about a "Podcast app").
Retrieval has no quality filter for this — a bad historical exemplar is
currently retrievable and usable for grounding exactly as if it were a
good one.

## 4. What is misleading about my headline number?

Three things, all real and all worth stating plainly rather than
burying:

1. **93% escalation precision sounds like a trustworthy, cautious
   system. It is the opposite.** A system that almost never wrongly
   escalates but also misses 85% of what actually needed a human isn't
   "precise" in any sense that matters for a support agent — it's
   dangerously overconfident in its own ability to auto-handle. If this
   project reported *only* precision (a defensible-sounding 0.93), it
   would actively misrepresent the system's real behavior. Recall is the
   number that matters here, and it's 0.149.

2. **The 0.735 classifier accuracy has a hidden, compounding ceiling.**
   It's not "73.5% correct" in an absolute sense — it's 73.5% agreement
   with training labels that were themselves found to only agree with
   actual human judgment about 73% of the time (54/200 disagreements
   during golden-set labeling). The classifier's real reliability is
   better understood as bounded above by that 73% figure than as a clean
   standalone accuracy number.

3. **None of the golden-set metrics reflect what live traffic would look
   like.** The golden set is deliberately balanced (25 examples per
   intent); real AppleSupport traffic is ~49.5% one intent and as low as
   2% for others. An aggregate accuracy computed here would not predict
   aggregate accuracy on live traffic — only the per-intent breakdowns
   are transferable, and even those assume the *rate* of each intent
   doesn't change how well the classifier handles it, which wasn't
   separately tested.

## 5. What I'd do with one more week

1. **Log which model handled each row**, then re-run the escalation
   analysis broken out by model. This is the single highest-value next
   step — it would tell us whether the recall crisis is a rubric problem
   (fixable with better prompting) or a weak-model problem (fixable by
   requiring a stronger model for the escalation judgment specifically,
   even if a cheaper model still drafts the reply).
2. **Split escalation into a rule-based pre-filter plus the LLM call**,
   not one combined judgment. Hard-trigger keywords (legal threats,
   safety language, explicit financial amounts) should force escalation
   deterministically before the LLM ever gets a chance to pattern-match
   past them. Given the results, this should have been built first, not
   deferred.
3. **Grow the judge-calibration sample past 25**, particularly to get a
   statistically significant read on actionability agreement.
4. **Add explicit language-mismatch detection** as its own escalation
   trigger, separate from the general rubric.
5. **Re-run reply drafting with a single, consistent model** (now that
   `reply_agent.py` logs exemplar text and the API/quota situation is
   understood) to remove the model-mixing confound from Section 3.4.
