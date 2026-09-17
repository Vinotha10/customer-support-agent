"""
Samples a subset of judge_scores.csv for YOU to independently score by
hand, using the exact same rubric the LLM judge used. This is the
mandatory "evidence of how well your judge agrees with a human" piece of
the brief -- without it, the LLM-as-judge rubric is just an unverified
claim, not evidence.

40 rows is enough to compute a meaningful correlation without being an
unreasonable amount of manual work -- stratified across the score range
the judge gave (not just random), so you're checking the judge across
its full range of opinions, not only its easy/confident middle-of-the-
road calls.

Output: outputs/eval/judge_calibration_unlabeled.csv
    -- open this, fill in human_correctness / human_tone /
       human_actionability for all 40 rows WITHOUT looking at the
       judge's own scores (those columns are deliberately not included
       in this file, to avoid anchoring your independent judgment on the
       model's), save as judge_calibration_labeled.csv in the same folder.
"""
import numpy as np
import pandas as pd

import config

N_SAMPLES = 40
RANDOM_SEED = config.RANDOM_SEED


def main():
    judged = pd.read_csv(config.EVAL_DIR / "judge_scores.csv")
    judged["avg_score"] = judged[["correctness", "tone", "actionability"]].mean(axis=1)

    # stratify across score bins so the sample isn't just the judge's
    # easy/confident middle -- include some rows it scored low and high too
    bins = pd.cut(judged["avg_score"], bins=[0, 2, 3.5, 5], labels=["low", "mid", "high"])
    judged["_bin"] = bins

    per_bin = max(1, N_SAMPLES // judged["_bin"].nunique())
    samples = []
    for b, group in judged.groupby("_bin", observed=True):
        n = min(per_bin, len(group))
        samples.append(group.sample(n=n, random_state=RANDOM_SEED))
    sample = pd.concat(samples).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    out = sample[["example_id", "customer_text", "reply"]].copy()
    out["human_correctness"] = ""
    out["human_tone"] = ""
    out["human_actionability"] = ""

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "judge_calibration_unlabeled.csv"
    out.to_csv(out_path, index=False)
    print(f"[calibration] wrote {len(out)} rows to {out_path}")
    print("[calibration] score each row 1-5 on correctness/tone/actionability "
          "(same definitions as llm_judge.py's rubric), WITHOUT peeking at "
          "judge_scores.csv for these rows first")
    print("[calibration] save as judge_calibration_labeled.csv in the same folder when done")


if __name__ == "__main__":
    main()
