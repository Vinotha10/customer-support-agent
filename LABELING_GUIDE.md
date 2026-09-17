# Golden Set Labeling Guide

## Sampling method (fill this into the report's "how did you sample" section)

- Source: `data/processed/applesupport_pairs_labeled.parquet` (101,316 pairs
  that survived thread reconstruction and weren't excluded as
  follow-up/acknowledgment noise).
- **25 examples per intent, 8 intents = 200 total**, sampled uniformly at
  random within each intent (`random_state=42`, reproducible).
- This is deliberately **NOT proportional** to the real intent distribution
  (which ranges from 49.5% down to 2.0%). Proportional sampling at n=200
  would give the rarest intents (`connectivity_issue`, `keyboard_autocorrect_bug`)
  only 3-4 examples each -- not enough to compute a trustworthy per-class
  score. Equal sampling trades "realistic mix" for "every intent is
  actually measurable." The true distribution is reported separately
  (from `apply_intent_labels.py` output) so this tradeoff is visible, not
  hidden.
- Known limitation to state explicitly in the report: a model's aggregate
  accuracy on this golden set does NOT reflect what accuracy would look
  like on live traffic, because live traffic is ~49.5% one intent and the
  golden set is 12.5% each. Report per-intent metrics, not just an
  aggregate, and say why.

## How to label each row

Open `outputs/golden_set/golden_set_unlabeled.csv`. For every row, fill in:

### `gold_intent`
Your own judgment of the correct intent, chosen from the 8 in
`src/intent_taxonomy.py` (or write `other` if none fit, or `ambiguous` if
you genuinely can't tell).

**Ignore the `suggested_intent_from_clustering` column while you decide.**
It's shown for reference only, after you've already formed your own
opinion -- read `customer_text` (and `prior_context` if present) first,
decide, then glance at the suggestion. If you anchor on the clustering
output you're not actually testing whether the clustering was right, and
you'll defeat the point of having an independent gold label at all.

If you disagree with the suggested intent, that's useful signal, not an
error on your part -- note *why* in `labeler_notes` (e.g. "clustering
called this software_bug_after_update but it's actually about a physical
drop, not an update").

### `gold_escalate`
`True` or `False` -- would you, as a human reviewer, want this message
escalated to a person rather than auto-replied? Use your judgment, but
some starting criteria:
- **Escalate**: safety-adjacent language, explicit anger/threat of
  leaving, account security (hacked/locked out), anything mentioning
  money/refunds with a specific dollar amount, `ambiguous`-intent rows,
  or messages you personally couldn't confidently answer.
- **Auto-handle**: a clearly recognized, previously-seen issue with a
  known standard resolution (e.g. the keyboard autocorrect bug, a
  standard "try a restart" battery/performance issue).

### `escalate_reason`
One short phrase explaining the `gold_escalate` call -- this becomes the
ground truth your later escalation-logic component is graded against, so
be specific enough that someone else reading it would make the same call
("mentions iCloud was hacked" not just "seems bad").

### `labeler_notes`
Anything odd: non-English text, sarcasm that changes the meaning, the
message actually being multiple issues at once, text that's cut off, etc.
This is exactly the raw material for the failure-analysis section later --
don't skip it just because it feels like extra work now.

## When you're done

Save as `outputs/golden_set/golden_set_labeled.csv` (same folder, new
filename) -- don't overwrite the unlabeled version, you want both for the
record.

## A note on being your own labeler

The brief doesn't require a second annotator, but it does implicitly
expect you to be honest about what a single-annotator golden set can and
can't prove. Put this in the report: no inter-annotator agreement check
was done because there was only one labeler (you), so `gold_intent` and
`gold_escalate` encode your judgment specifically, not an objective
ground truth. This matters most for `ambiguous` cases and for
`gold_escalate` calls near the boundary -- a second labeler might
reasonably disagree on some of them.

## Status: complete

All 200 rows labeled (drafted with reasoning by Claude, then reviewed
and corrected by hand — see DECISION_LOG.md #7 for how this was split).
54/200 (27%) of drafted labels were corrected away from clustering's
suggestion during review. Saved as `outputs/golden_set/golden_set_labeled.csv`.
