"""
Computes how well the LLM judge agrees with your independent hand-scores
-- the mandatory evidence piece for the LLM-as-judge rubric in the brief.

Reports, per dimension (correctness/tone/actionability) and overall:
  - Spearman correlation (rank agreement -- robust to the two of you
    using the 1-5 scale slightly differently in absolute terms)
  - % of rows where judge and human are within 1 point of each other
    (an intuitive, easy-to-state number for the report)
  - mean absolute difference

Run AFTER filling in outputs/eval/judge_calibration_labeled.csv by hand
(see build_judge_calibration_set.py).

Output: outputs/eval/judge_agreement.txt
"""
import pandas as pd
from scipy.stats import spearmanr

import config

DIMENSIONS = ["correctness", "tone", "actionability"]


def main():
    human = pd.read_csv(config.EVAL_DIR / "judge_calibration_labeled.csv")
    judge = pd.read_csv(config.EVAL_DIR / "judge_scores.csv")

    missing = human[[f"human_{d}" for d in DIMENSIONS]].isna().any(axis=1)
    if missing.any():
        print(f"[agreement] WARNING: {missing.sum()} row(s) have an unfilled human score, dropping: "
              f"{human.loc[missing, 'example_id'].tolist()}")
        human = human[~missing]

    merged = human.merge(judge, on="example_id", suffixes=("_human_file", "_judge_file"))
    if len(merged) < len(human):
        print(f"[agreement] WARNING: only {len(merged)}/{len(human)} calibration rows matched "
              f"an example_id in judge_scores.csv -- check for id mismatches")

    lines = []
    for dim in DIMENSIONS:
        human_col = f"human_{dim}"
        judge_col = dim
        h = merged[human_col].astype(float)
        j = merged[judge_col].astype(float)

        rho, pval = spearmanr(h, j)
        rho_str = f"{rho:.3f} (p={pval:.3f})" if not pd.isna(rho) else "undefined (one of your score columns has zero variance -- can't compute a correlation from constant values)"
        within_1 = (abs(h - j) <= 1).mean()
        mae = abs(h - j).mean()

        lines.append(f"{dim}:")
        lines.append(f"  Spearman rho: {rho_str}")
        lines.append(f"  % within 1 point: {within_1:.1%}")
        lines.append(f"  mean absolute difference: {mae:.2f}")
        lines.append("")

    overall_h = merged[[f"human_{d}" for d in DIMENSIONS]].mean(axis=1)
    overall_j = merged[DIMENSIONS].mean(axis=1)
    rho, pval = spearmanr(overall_h, overall_j)
    lines.append(f"overall (mean of all 3 dimensions):")
    lines.append(f"  Spearman rho: {rho:.3f} (p={pval:.3f})")
    lines.append(f"  % within 1 point: {(abs(overall_h - overall_j) <= 1).mean():.1%}")
    lines.append(f"  n = {len(merged)}")

    for line in lines:
        print(line)

    out_path = config.EVAL_DIR / "judge_agreement.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[agreement] wrote {out_path}")


if __name__ == "__main__":
    main()
