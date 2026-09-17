"""
Samples the golden evaluation set from the labeled pairs.

Deliberately uses EQUAL sampling per intent (not proportional to the real
class distribution) -- see rationale in README/decision log. The real
distribution is ~49.5% general_software_complaint down to ~2.0%
connectivity_issue; proportional sampling at n=200 would give the smallest
classes only 3-4 examples, nowhere near enough to compute a meaningful
per-class F1. Equal sampling means every intent gets a fair, statistically
usable evaluation slice. The TRUE class distribution is still recorded
separately (from apply_intent_labels.py's printed output) and reported
alongside golden-set results so the mismatch is visible, not hidden.

Output: outputs/golden_set/golden_set_unlabeled.csv
    -- open this, fill in gold_intent / gold_escalate / escalate_reason /
       notes by hand, save as golden_set_labeled.csv (see LABELING_GUIDE.md)
"""
import pandas as pd

import config
import intent_taxonomy as tax

N_PER_INTENT = 25          # 8 intents x 25 = 200, top of the 150-250 target range
RANDOM_SEED = config.RANDOM_SEED

LABELED_PATH = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_labeled.parquet"
OUT_DIR = config.ROOT / "outputs" / "golden_set"


def main():
    df = pd.read_parquet(LABELED_PATH)
    print(f"[golden_set] loaded {len(df)} labeled pairs across {df['intent'].nunique()} intents")

    samples = []
    for intent in tax.INTENTS:
        pool = df[df["intent"] == intent]
        n = min(N_PER_INTENT, len(pool))
        if n < N_PER_INTENT:
            print(f"[golden_set] warning: only {n} available for '{intent}' "
                  f"(wanted {N_PER_INTENT})")
        samples.append(pool.sample(n=n, random_state=RANDOM_SEED))

    golden = pd.concat(samples, ignore_index=True)
    golden = golden.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle order
    golden.insert(0, "example_id", [f"g{i:04d}" for i in range(len(golden))])

    # columns a human labeler needs to see, plus empty columns to fill in
    out = golden[[
        "example_id", "customer_tweet_id", "customer_text", "prior_context",
        "brand_text", "followup_text", "intent",  # intent here = clustering's SUGGESTED label, not gold
    ]].rename(columns={"intent": "suggested_intent_from_clustering"})

    for col in ["gold_intent", "gold_escalate", "escalate_reason", "labeler_notes"]:
        out[col] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_set_unlabeled.csv"
    out.to_csv(out_path, index=False)

    print(f"[golden_set] wrote {len(out)} examples to {out_path}")
    print(f"[golden_set] per-intent counts: {golden['intent'].value_counts().to_dict()}")
    print("[golden_set] next: open the CSV, fill gold_intent / gold_escalate / "
          "escalate_reason / labeler_notes by hand for every row, save as "
          "golden_set_labeled.csv in the same folder. See LABELING_GUIDE.md.")


if __name__ == "__main__":
    main()
