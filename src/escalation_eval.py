"""
Evaluates the agent's auto-handle vs. escalate decisions against your
hand-labeled gold_escalate ground truth. Purely mechanical -- no LLM call,
reads directly from outputs/eval/agent_outputs.csv (produced by
reply_agent.py --golden-set).

Precision/recall framing matters here, not just accuracy:
  - False negative (agent said auto-handle, gold says escalate) is the
    costly direction -- a customer with a real problem gets a canned
    reply instead of a human. This is the error to weight more heavily.
  - False positive (agent escalates something gold says was fine to
    auto-handle) is a cost/efficiency loss, not a customer-harm risk.

Reports both directions separately rather than a single blended accuracy
number, since they have very different real-world costs.

Output: outputs/eval/escalation_metrics.txt, plus prints a handful of
false negatives to the console (the ones worth manually reading for the
failure analysis, since those are the actual risk cases).
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config

AGENT_OUTPUTS_PATH = config.EVAL_DIR / "agent_outputs.csv"


def main():
    df = pd.read_csv(AGENT_OUTPUTS_PATH)
    before = len(df)
    df = df.dropna(subset=["gold_escalate", "escalate"])
    dropped = before - len(df)
    if dropped:
        print(f"[escalation_eval] dropped {dropped} row(s) with missing escalate/gold_escalate")

    # normalize both to real booleans -- CSV round-trips can turn True/False
    # into the strings "True"/"False", and pandas' own bool inference on
    # mixed-type columns isn't always reliable, so this is done explicitly
    def to_bool(series):
        return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})

    y_true = to_bool(df["gold_escalate"])
    y_pred = to_bool(df["escalate"])

    valid = y_true.notna() & y_pred.notna()
    if (~valid).sum():
        print(f"[escalation_eval] dropped {(~valid).sum()} row(s) with unparseable True/False values")
    y_true, y_pred, df = y_true[valid], y_pred[valid], df[valid]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    accuracy = (y_true == y_pred).mean()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    lines = []
    lines.append(f"n = {len(df)}")
    lines.append(f"accuracy:  {accuracy:.3f}")
    lines.append(f"precision (of predicted escalations, how many were truly meant to escalate): {precision:.3f}")
    lines.append(f"recall (of true escalations, how many the agent caught):                      {recall:.3f}")
    lines.append(f"f1: {f1:.3f}")
    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(f"                    gold: auto-handle   gold: escalate")
    lines.append(f"  agent: auto-handle       {tn:5d}               {fn:5d}   <- FN: costly (missed real escalations)")
    lines.append(f"  agent: escalate          {fp:5d}               {tp:5d}")
    lines.append("")
    lines.append(f"False negatives (agent said auto-handle, should have escalated): {fn}")
    lines.append(f"False positives (agent escalated, gold said auto-handle was fine): {fp}")

    print("\n".join(lines))

    # surface the actual false-negative rows -- these are the real-risk cases
    # worth reading by hand for the failure analysis
    fn_rows = df[(y_true == True) & (y_pred == False)]  # noqa: E712
    if len(fn_rows):
        print(f"\n--- False negative examples (up to 10 shown, all in the output file) ---")
        for _, row in fn_rows.head(10).iterrows():
            print(f"  [{row.get('example_id', '?')}] {row['customer_text'][:100]}")
            print(f"      agent's reason for NOT escalating: {row.get('escalate_reason', '(none)')}")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "escalation_metrics.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write(f"\n\nFalse negative example_ids (read these for failure analysis): "
                f"{fn_rows.get('example_id', fn_rows.index).tolist()}\n")
    print(f"\n[escalation_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
