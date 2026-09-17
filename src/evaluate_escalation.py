"""
Evaluates the agent's escalate/auto-handle decisions against gold_escalate
from the hand-labeled golden set. No LLM call needed -- this is pure
comparison of two boolean columns already sitting in agent_outputs.csv.

Precision/recall here matter in DIFFERENT directions than a normal
classifier eval:
  - A false NEGATIVE (agent said auto-handle, human said escalate) is the
    dangerous direction -- a real problem gets an automated reply instead
    of human attention. This is the error to minimize.
  - A false POSITIVE (agent escalates something a human would have
    auto-handled) is a cost/efficiency problem, not a safety one -- overly
    cautious, but not actually harmful.
This asymmetry is worth stating explicitly in the report rather than
just reporting a single F1 and moving on.

Output: outputs/eval/escalation_metrics.txt
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config


def to_bool(series):
    # gold_escalate / escalate may come through as "True"/"False" strings,
    # 1/0, or actual bools depending on how the CSV was read/edited --
    # normalize before comparing so a formatting quirk doesn't silently
    # break the eval.
    return series.astype(str).str.strip().str.lower().map({"true": True, "1": True, "1.0": True,
                                                             "false": False, "0": False, "0.0": False})


def main():
    df = pd.read_csv(config.EVAL_DIR / "agent_outputs.csv")

    before = len(df)
    df = df.dropna(subset=["gold_escalate", "escalate"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[escalation_eval] dropped {dropped} row(s) with missing gold_escalate or escalate")

    df["gold_escalate_bool"] = to_bool(df["gold_escalate"])
    df["escalate_bool"] = to_bool(df["escalate"])

    bad_parse = df["gold_escalate_bool"].isna() | df["escalate_bool"].isna()
    if bad_parse.any():
        print(f"[escalation_eval] WARNING: {bad_parse.sum()} row(s) had an unparseable "
              f"escalate value, dropping: {df.loc[bad_parse, 'example_id'].tolist()}")
        df = df[~bad_parse]

    y_true = df["gold_escalate_bool"]
    y_pred = df["escalate_bool"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=True, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0]
    fp, tn = cm[1]

    lines = []
    lines.append(f"n = {len(df)}")
    lines.append(f"precision (of predicted-escalate, how many truly needed it): {precision:.3f}")
    lines.append(f"recall (of truly-needed-escalation, how many were caught):   {recall:.3f}  <-- most important: misses here are the dangerous direction")
    lines.append(f"f1: {f1:.3f}")
    lines.append("")
    lines.append("confusion matrix:")
    lines.append(f"                  gold=escalate   gold=auto-handle")
    lines.append(f"  pred=escalate   {tp:>13d}   {fp:>17d}")
    lines.append(f"  pred=auto       {fn:>13d}   {tn:>17d}")
    lines.append("")
    lines.append(f"FALSE NEGATIVES (agent said auto-handle, should have escalated) -- the dangerous direction: {fn}")
    lines.append(f"FALSE POSITIVES (agent escalated, human would have auto-handled) -- a cost/efficiency issue, not a safety one: {fp}")

    for line in lines:
        print(line)

    fn_examples = df[(df["gold_escalate_bool"] == True) & (df["escalate_bool"] == False)]
    if len(fn_examples):
        print(f"\n[escalation_eval] false-negative examples (real problems the agent auto-handled):")
        for _, row in fn_examples.head(10).iterrows():
            print(f"  {row.get('example_id', '?')}: \"{str(row['customer_text'])[:80]}\"")
            print(f"      agent's reasoning: {row.get('escalate_reason', '(none)')}")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "escalation_metrics.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write(f"\n\nfalse-negative example_ids: {fn_examples['example_id'].tolist() if len(fn_examples) else []}\n")
    print(f"\n[escalation_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
