# Day 3 Findings — Classifier Baselines & Failure Analysis

Captured for direct use in the report's "Results vs. baselines" and
"Top 5 failure modes" sections.

## Baseline comparison (all evaluated against the 196 in-taxonomy golden-set rows)

| Tier | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always predicts majority class `general_software_complaint`) | 0.107 | 0.024 |
| TF-IDF + LogisticRegression (trained on 101,116 cluster-derived labels) | 0.735 | 0.717 |
| Gemini few-shot (`gemini-3.5-flash-lite`, zero training data) | 0.658 | 0.564 |

**Headline finding: the trained baseline beat the zero-shot LLM.** Report
this plainly rather than treating it as a disappointing result — it's
explainable and legitimate:
1. The taxonomy itself was derived from clustering the training data, so
   the TF-IDF+LogReg model has a structural home-field advantage the LLM
   doesn't get.
2. Domain slang (e.g. the keyboard bug being called "boxes" or "the I
   thing" in the wild) is something the trained model saw thousands of
   times; the few-shot prompt only had one example sentence per intent.
3. The LLM's weakest classes were exactly the two vaguest/catch-all
   intents (`general_software_complaint` F1 0.32, `general_help_request`
   F1 0.38) — a single example can't teach the boundary of "everything
   that doesn't fit elsewhere."

**One place the LLM clearly wins:** on the 5 golden-set rows that were
genuinely outside the 8-intent taxonomy (Xcode bug report, phishing
report, a typo fragment, etc.), the LLM correctly said "other" for **5/5**.
The TF-IDF classifier structurally cannot do this at all -- it only knows
8 classes and must always pick one. This is worth its own sentence: a
production system needs this kind of fallback judgment, and only the LLM
tier currently provides it.

## Failure analysis: the LLM's 13 "other" misfires split into two very different buckets

Out of 196 in-taxonomy golden rows, the LLM predicted `other` (i.e.
"none of the 8 intents fit") for 13 rows where a human had assigned a
real intent. These are NOT all the same kind of error:

### Bucket A -- genuine misses (3 rows, cite as plain failure examples)
- *"Apple is about to pmtfo w/ these boxes... fix this shit"* -- gold
  `keyboard_autocorrect_bug`. "Boxes" is literally in the few-shot prompt
  for this intent. Should have been caught.
- *"Just dropped my phone in the dirt and that [word] shattered"* -- gold
  `screen_display_issue`. Unambiguous keyword match, missed anyway
  (possibly confused by profanity/emoji in the message).
- *"unable to download x update, why don't you tell me WHY"* -- gold
  `software_bug_after_update`. Clear update-failure language, missed.

### Bucket B -- defensible disagreements (10 rows) -- the more important finding
The other 10 misfires cluster almost entirely around messages that were
hand-labeled `general_help_request` or `general_software_complaint` --
the two vaguest, most catch-all intents in the taxonomy. Several are
cases where the LLM's "other" call is arguably MORE defensible than the
original hand label:
- A billing dispute (charged 3x for one failed purchase) -- the original
  labeling notes for this exact row already flagged it as not cleanly
  fitting any of the 8 intents and suggested a `billing_payment_dispute`
  category might be worth adding. The LLM independently reached the same
  conclusion.
- "Just DM'd you, but no reply" and "it seems to have resolved on its
  own... thank you" -- both already flagged during hand-labeling as
  follow-up/acknowledgment messages rather than genuine new intents,
  similar in nature to the clusters excluded from training entirely. The
  LLM treating them as not-a-real-intent is consistent with that same
  earlier judgment call, not a contradiction of it.
- A support-access complaint (can't call a US toll-free number from
  Brazil) and a repeated-activation-failure complaint (4 failed attempts
  across channels) -- genuinely structural/logistics problems that don't
  match any of the 8 categories well.

**Why this matters for the report:** this is independent evidence (a
second "labeler" -- the LLM -- disagreeing with the human labeler) that
`general_help_request` specifically has an unstable boundary: on this
sample, roughly half of what a human put there, an independent model
would not have. This is stronger and more specific than a generic
"single-annotator limitation" disclaimer -- it's a quantified, concrete
finding about exactly which category is shakiest and why.

## Open item: one golden-set row has NaN customer_text/gold_intent
Row 200 (0-indexed) in the predictions file came back completely empty.
Needs a quick check against `outputs/golden_set/golden_set_labeled.csv`
before finalizing golden-set counts in the report -- likely either a
blank row introduced during CSV editing, or a genuinely empty source
tweet that should have been excluded during sampling.
