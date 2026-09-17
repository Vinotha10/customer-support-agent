"""
Quick EDA on the reconstructed pairs. Deliberately lightweight (no heavy
NLP deps) so it runs in seconds and gives you what you need to sanity-check
data_prep and to write the "problem framing" section of the report:
  - how much data do we actually have for this brand
  - how noisy/short are customer messages
  - how fast/slow does the brand typically respond (context for what
    "auto-handle" latency expectations should be)
  - rough signal on how many threads look "resolved" vs. not, via the
    weak follow-up-text heuristic (NOT ground truth — flag this in report)
"""
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

STOPWORDS = set("""
the a an and or but if to of in on for is are was were be been being
this that these those i you he she it we they my your his her its our
their me him her us them am is are was were will would can could shall
should may might must with as at by from up down out about into over
after before dm please thanks thank you re your apple applesupport
""".split())

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]


def main():
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    n = len(df)
    print(f"[eda] total pairs: {n}")

    # --- response time ---------------------------------------------------
    rt = df["response_seconds"].dropna()
    if len(rt):
        print("[eda] response time (seconds) - median: %.0f, p90: %.0f, max: %.0f"
              % (rt.median(), rt.quantile(0.9), rt.max()))
        plt.figure(figsize=(6, 4))
        rt.clip(upper=rt.quantile(0.95)).hist(bins=40)
        plt.title(f"{config.BRAND}: brand response time (sec, clipped p95)")
        plt.xlabel("seconds")
        plt.ylabel("count")
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "response_time_hist.png", dpi=120)
        plt.close()

    # --- message length ---------------------------------------------------
    df["customer_len"] = df["customer_text"].str.len()
    df["brand_len"] = df["brand_text"].str.len()
    print("[eda] customer msg length - mean: %.0f, median: %.0f"
          % (df["customer_len"].mean(), df["customer_len"].median()))
    print("[eda] brand reply length  - mean: %.0f, median: %.0f"
          % (df["brand_len"].mean(), df["brand_len"].median()))

    # --- multi-turn proportion ---------------------------------------------------
    has_prior = df["prior_context"].notna().mean()
    has_followup = df["followup_text"].notna().mean()
    print(f"[eda] pct with prior customer context (multi-turn): {has_prior:.1%}")
    print(f"[eda] pct with a customer follow-up after brand reply: {has_followup:.1%}")

    # --- weak resolution heuristic (report this as a HEURISTIC, not ground truth) ---
    THANK_WORDS = ("thanks", "thank you", "resolved", "fixed", "works now", "appreciate")
    COMPLAINT_WORDS = ("still", "again", "worse", "no update", "not working", "ridiculous")
    fu = df["followup_text"].dropna().str.lower()
    looks_thankful = fu.str.contains("|".join(THANK_WORDS)).mean() if len(fu) else float("nan")
    looks_unresolved = fu.str.contains("|".join(COMPLAINT_WORDS)).mean() if len(fu) else float("nan")
    print(f"[eda] of follow-ups, pct matching thank/resolved words: {looks_thankful:.1%}")
    print(f"[eda] of follow-ups, pct matching still-broken words: {looks_unresolved:.1%}")

    # --- top tokens in customer messages (rough theme preview before clustering) ---
    all_tokens = Counter()
    for text in df["customer_text"]:
        all_tokens.update(tokenize(text))
    print("[eda] top 25 customer message tokens (stopwords removed):")
    for tok, cnt in all_tokens.most_common(25):
        print(f"    {tok:20s} {cnt}")

    # --- save a small summary file for the report ---------------------------
    summary_path = config.FIGURES_DIR.parent / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Brand: {config.BRAND}\n")
        f.write(f"Total pairs: {n}\n")
        if len(rt):
            f.write(f"Median response time (s): {rt.median():.0f}\n")
        f.write(f"Pct multi-turn (has prior context): {has_prior:.1%}\n")
        f.write(f"Pct with customer follow-up: {has_followup:.1%}\n")
        f.write(f"Follow-up looks thankful (heuristic): {looks_thankful:.1%}\n")
        f.write(f"Follow-up looks unresolved (heuristic): {looks_unresolved:.1%}\n")
        f.write("Top tokens: " + ", ".join(t for t, _ in all_tokens.most_common(25)) + "\n")
    print(f"[eda] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
