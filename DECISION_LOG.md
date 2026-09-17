# Decision Log

15 non-obvious decisions made while building this, and why. (A much longer
working log was kept during development; this is the curated set most
worth defending. The full engineering story — including the API-provider
saga — is in REPORT.md.)

1. **Brand = AppleSupport**, not a smaller/quieter brand. Needed enough
   volume per discovered intent to retrieve grounded exemplars later, and
   wanted a support style with recognizable resolution patterns rather
   than highly bespoke/booking-specific replies (ruled out airlines).

2. **Pair unit is the brand reply, not the customer message, with strict
   parent validation.** A customer often tweets 2-3 times before a reply
   lands; anchoring on the brand's reply makes the grounding corpus
   unambiguous. A pair is only kept if its parent tweet actually exists
   and is genuinely inbound (customer) — dangling references and
   brand-to-brand cross-posts are dropped and the drop counts are
   printed, not silently swallowed.

3. **Intent taxonomy discovered via clustering (TF-IDF + KMeans), then
   manually named** — not hand-guessed up front, and not left as raw
   unlabeled clusters. TF-IDF (not sentence embeddings) was the default
   specifically so the whole pipeline stays dependency-light and
   reproducible offline in under 15 minutes.

4. **Thread reconstruction is unit-tested against a hand-built fixture
   with a known-correct answer**, not just eyeballed on real data. The
   failure modes here (off-by-one hops, wrong inbound direction) are easy
   to introduce and easy to miss by eye, and would silently corrupt every
   downstream stage.

5. **One cluster (~47% of the sample) was kept as an explicit, named
   catch-all (`general_software_complaint`) rather than force-split**
   into fake specificity once manual review found no clean further
   split. Named honestly as a limitation rather than hidden.

6. **`keyboard_autocorrect_bug` kept as its own narrow intent** despite
   being a small cluster, because it's one specific, recognizable known
   bug calling for a different reply strategy (point to the known fix)
   than a generic post-update bug report. Intent granularity was driven
   by "does this need a different reply/escalation policy," not cluster
   size.

7. **The golden set was AI-drafted, then human-reviewed line by line, not
   AI-final.** The brief requires this judgment to be the candidate's
   own. All 200 rows were drafted with reasoning shown per row, then
   corrected by hand. 54/200 (27%) of the drafted labels disagreed with
   clustering's own suggestion — kept as a headline finding about
   training-label noise, not smoothed over.

8. **Classifiers train on noisy cluster-derived labels but are evaluated
   ONLY against the hand-labeled golden set**, never against the cluster
   labels themselves (which would just measure agreement with
   clustering, not correctness). This gives the simple classifier a real
   documented ceiling (~73%, the cluster/human agreement rate) rather
   than an unexplained gap discovered at eval time.

9. **Golden-set rows are excluded from the classifier training pool by
   `customer_tweet_id`** before fitting, since they were originally
   sampled from that same labeled pool — otherwise the classifier would
   be partially evaluated on data it already saw.

10. **Golden-set rows outside the 8-intent taxonomy (`other`) are
    reported as separate "out-of-taxonomy coverage,"** not folded into
    accuracy/F1 and not silently dropped. A classifier limited to 8
    classes can't be scored fairly against a class it never had, but
    hiding those rows would overstate real-world coverage.

11. **The few-shot LLM baseline uses zero training data by design** —
    intent definitions and a couple of hand-written examples, nothing
    from the golden set. This sidesteps the noisy-label ceiling that
    bounds the trained baseline, so the fact it still scored lower
    (0.658 vs. 0.735 accuracy) is a genuine, reportable result, not a
    methodology flaw.

12. **Reply drafting and escalation are decided in ONE combined LLM
    call**, using an escalation rubric copied directly from the same
    criteria used to hand-label `gold_escalate`. One call controls API
    cost and keeps the escalation judgment consistent with how the
    ground truth itself was produced, rather than inventing a second,
    different standard at evaluation time.

13. **Retrieval for grounding is per-intent, using the TRAINED classifier's
    prediction** (the one that beat the LLM baseline), not the LLM's own
    guess — used operationally, not just reported as a result. Retrieval
    has no quality filter for historical replies that were themselves
    poor exemplars (confirmed at least one case exists during labeling)
    — a documented gap, not a silent one.

14. **The LLM client is provider-agnostic and falls back automatically**
    across models and even across providers (Anthropic, Gemini, a local
    Ollama model). This was forced by real project constraints — paid
    credits weren't an option, and Gemini's free tier hit quota limits
    and model deprecations repeatedly — but it also means the final
    200-row run was completed by more than one underlying model, which
    is reported plainly rather than hidden.

15. **Escalation errors are interpreted asymmetrically, not as one
    blended rate.** A false negative (agent auto-handles something that
    needed a human) is a safety/trust failure; a false positive (agent
    escalates something a human would have auto-handled) is a
    cost/efficiency issue. The final result — 93% precision but only 15%
    recall — would look fine as a single F1 (0.257 already signals
    trouble) but is only actually interpretable once split this way: the
    system is dangerously under-escalating, not over-cautious.
