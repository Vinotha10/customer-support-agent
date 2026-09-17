"""
Fast interactive scoring for the judge calibration set. Shows one row at a
time; press a single digit 1-5 for each dimension (no typing into CSV
cells, no Enter needed on Windows). Saves after every row, so you can quit
(q) at any point and resume later exactly where you left off.

Deliberately does NOT show the LLM judge's own scores anywhere in this
tool -- see judge_agreement.py's docstring for why that matters. You are
the independent check; seeing the judge's opinion first would defeat the
entire point of measuring agreement.

Controls, at each prompt:
    1-5   score for that dimension
    b     go back and redo the previous dimension/row
    q     save progress and quit -- rerun this script later to resume

Run:
    python score_calibration.py

When every row has all three scores, it writes outputs/eval/judge_calibration_labeled.csv,
ready for judge_agreement.py.
"""
import sys

import numpy as np
import pandas as pd

import config

UNLABELED_PATH = config.EVAL_DIR / "judge_calibration_unlabeled.csv"
LABELED_PATH = config.EVAL_DIR / "judge_calibration_labeled.csv"

DIMENSIONS = [
    ("correctness", "Does the reply address what the customer actually said? (1=off-topic/generic, 5=clearly on-point)"),
    ("tone", "Professional, empathetic, brand-appropriate? (1=cold/inappropriate, 5=warm and professional)"),
    ("actionability", "Clear next step or resolution for the customer? (1=vague non-answer, 5=clear concrete action)"),
]

try:
    import msvcrt

    def get_key():
        return msvcrt.getch().decode(errors="ignore")
except ImportError:
    def get_key():
        # non-Windows fallback: requires Enter, but functionally equivalent
        return input().strip()[:1]


def load_progress():
    if LABELED_PATH.exists():
        return pd.read_csv(LABELED_PATH)
    return pd.read_csv(UNLABELED_PATH)


def prompt_dimension(row_num, total, customer_text, reply, dim_name, dim_desc, current_value):
    status = f"  [already scored: {int(current_value)}]" if pd.notna(current_value) and current_value != "" else ""
    print(f"\n--- row {row_num}/{total} -- {dim_name}{status} ---")
    print(f"Customer: {customer_text}")
    print(f"Reply:    {reply}")
    print(f"\n{dim_desc}")
    print("Press 1-5, 'b' to go back, or 'q' to save and quit: ", end="", flush=True)

    while True:
        key = get_key()
        print(key)
        if key == "q":
            return "QUIT"
        if key == "b":
            return "BACK"
        if key in "12345":
            return int(key)
        print("(invalid key -- press 1-5, 'b', or 'q'): ", end="", flush=True)


def main():
    df = load_progress()
    for col in ["human_correctness", "human_tone", "human_actionability"]:
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    i = 0
    while i < total:
        row = df.iloc[i]
        dim_idx = 0
        while dim_idx < len(DIMENSIONS):
            dim_name, dim_desc = DIMENSIONS[dim_idx]
            col = f"human_{dim_name}"
            current = row[col]
            if pd.notna(current) and str(current).strip() != "":
                dim_idx += 1  # already scored, skip silently on resume
                continue

            result = prompt_dimension(i + 1, total, row["customer_text"], row["reply"],
                                       dim_name, dim_desc, current)

            if result == "QUIT":
                df.to_csv(LABELED_PATH, index=False)
                done = df[["human_correctness", "human_tone", "human_actionability"]].notna().all(axis=1).sum()
                print(f"\nSaved progress: {done}/{total} rows fully scored. "
                      f"Rerun this script to resume.")
                sys.exit(0)

            if result == "BACK":
                if dim_idx > 0:
                    dim_idx -= 1
                    df.at[df.index[i], f"human_{DIMENSIONS[dim_idx][0]}"] = np.nan
                elif i > 0:
                    i -= 1
                    dim_idx = len(DIMENSIONS) - 1
                continue

            df.at[df.index[i], col] = result
            df.to_csv(LABELED_PATH, index=False)  # save after every single answer, not just every row
            dim_idx += 1

        i += 1

    print(f"\nAll {total} rows scored. Wrote {LABELED_PATH}")
    print("Next: python judge_agreement.py")


if __name__ == "__main__":
    main()
