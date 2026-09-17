"""
Same few-shot classification approach as llm_baseline_classifier.py, but
using Google's Gemini API instead of Anthropic's -- Gemini's free tier
(via Google AI Studio) requires no credit card and no billing setup, unlike
a fresh Anthropic account. See setup steps below.

Setup:
  1. Go to https://aistudio.google.com/apikey and create a free API key
     (no credit card required for the free tier).
  2. pip install google-genai
  3. export GEMINI_API_KEY=...  (or on Windows: $env:GEMINI_API_KEY="...")

Free tier rate limits (these have been shifting -- Google recently cut the
newer Flash models like gemini-3.6-flash down to just 20 requests/day,
while the Flash-Lite variants still get roughly 500/day and 15/minute as
of this writing; check https://ai.google.dev/gemini-api/docs/rate-limits
or just watch the 429 error body, which states the exact quota that
tripped). At the default pacing, 200 rows takes about 20 minutes. Use
--limit for a quick smoke test first, and if you get a 429 with a low
daily quota number, that specific model has been tightened -- try
--model gemini-3.5-flash-lite as a fallback.

This script has NOT been run end-to-end in development (no Gemini key
available in that environment) -- written carefully from current SDK docs,
but expect to debug on your first real run, same as the Anthropic version.

Output: outputs/eval/llm_baseline_predictions_gemini.csv, appended metrics
in outputs/eval/baseline_metrics.txt (same file used by the other two
baselines, so all tiers end up together for the report).
"""
import argparse
import time

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config
import intent_taxonomy as tax

load_dotenv(override=True)  # .env always wins, even over a stale $env: var set earlier in this shell session

try:
    from google import genai
except ImportError:
    genai = None

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # confirmed working end-to-end on this project's key (full 200-row run succeeded)
DEFAULT_SLEEP_SECONDS = 5.0  # flash-lite allows ~15 requests/min on the free tier

FEW_SHOT_EXAMPLES = """
Examples:
- "@AppleSupport my battery drains so fast since I updated to iOS 11" -> battery_drain
- "wifi keeps disconnecting on my phone since the last update" -> connectivity_issue
- "can someone help me figure out why my photos won't sync" -> general_help_request
- "this update ruined my whole experience, everything is worse now" -> general_software_complaint
- "why does typing the letter i show a question mark in a box" -> keyboard_autocorrect_bug
- "just unboxed my new phone and it won't turn on at all" -> new_device_defect
- "my screen went completely black and buttons don't respond" -> screen_display_issue
- "apps keep crashing constantly since I installed the new update" -> software_bug_after_update
""".strip()


def build_prompt(customer_text: str, prior_context) -> str:
    intent_list = "\n".join(f"- {i}" for i in tax.INTENTS)
    context_block = ""
    if prior_context and pd.notna(prior_context):
        context_block = f'\nPrior message in the same thread: "{prior_context}"\n'
    return f"""You are classifying a customer support tweet sent to Apple Support into exactly one intent.

Intents:
{intent_list}
- other (use ONLY if truly none of the above fit -- e.g. developer-tool bugs, phishing reports, complaints about the support process itself rather than a product issue)

{FEW_SHOT_EXAMPLES}
{context_block}
Customer message: "{customer_text}"

Respond with ONLY the intent name, nothing else. No punctuation, no explanation."""


def classify_one(client, customer_text, prior_context, model, max_retries=4):
    prompt = build_prompt(customer_text, prior_context)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            raw = (resp.text or "").strip().lower().replace(" ", "_")
            valid = set(tax.INTENTS) | {"other"}
            if raw in valid:
                return raw
            for v in valid:
                if v in raw:
                    return v
            return "other"
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = DEFAULT_SLEEP_SECONDS * (attempt + 2)
                print(f"[gemini_baseline] rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                print(f"[gemini_baseline] failed after {max_retries} attempts: {e}")
                return "other"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only classify the first N golden rows, for a quick smoke test")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS,
                         help="seconds to wait between calls, to stay under the free-tier rate limit")
    args = parser.parse_args()

    if genai is None:
        raise SystemExit("pip install google-genai first")

    client = genai.Client()  # reads GEMINI_API_KEY from env

    golden = pd.read_csv(config.GOLDEN_SET_LABELED)
    if args.limit:
        golden = golden.head(args.limit)
        print(f"[gemini_baseline] --limit set, running on first {len(golden)} rows only")

    preds = []
    for i, row in golden.iterrows():
        pred = classify_one(client, row["customer_text"], row.get("prior_context"), args.model)
        preds.append(pred)
        if (i + 1) % 10 == 0:
            print(f"[gemini_baseline] classified {i + 1}/{len(golden)}")
        time.sleep(args.sleep)

    golden = golden.copy()
    golden["llm_pred"] = preds

    in_taxonomy = golden[golden["gold_intent"].isin(tax.INTENTS)]
    out_of_taxonomy = golden[~golden["gold_intent"].isin(tax.INTENTS)]

    y_true = in_taxonomy["gold_intent"]
    y_pred = in_taxonomy["llm_pred"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cls_report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== LLM FEW-SHOT ({args.model}, Gemini) ===")
    print(f"accuracy:  {acc:.3f}")
    print(f"macro-F1:  {macro_f1:.3f}")
    print(cls_report)
    print(f"out-of-taxonomy golden rows (not scored): {len(out_of_taxonomy)}")
    if len(out_of_taxonomy):
        correctly_flagged = (out_of_taxonomy["llm_pred"] == "other").sum()
        print(f"  of those, model correctly said 'other' for {correctly_flagged}/{len(out_of_taxonomy)}")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = config.EVAL_DIR / "llm_baseline_predictions_gemini.csv"
    golden.to_csv(pred_path, index=False)

    metrics_path = config.EVAL_DIR / "baseline_metrics.txt"
    with open(metrics_path, "a") as f:
        f.write(f"\n\n=== LLM FEW-SHOT ({args.model}, Gemini) ===\n")
        f.write(f"accuracy:  {acc:.3f}\nmacro-F1:  {macro_f1:.3f}\n")
        f.write(cls_report)
        f.write(f"\nout-of-taxonomy rows: {len(out_of_taxonomy)}\n")

    print(f"\n[gemini_baseline] wrote {pred_path}")
    print(f"[gemini_baseline] appended to {metrics_path}")


if __name__ == "__main__":
    main()
