"""
Third classifier tier: few-shot LLM classification. Unlike the trivial and
simple baselines, this uses NO training data at all -- just the intent
definitions and a couple of examples per intent in the prompt. That matters
methodologically: it isn't bounded by the ~27% label noise in the
cluster-derived training pool (see baseline_classifiers.py), so if this
tier doesn't clearly beat the simple baseline on the golden set, that's a
real, reportable finding -- not a bug.

Requires: pip install anthropic
          export ANTHROPIC_API_KEY=sk-...   (your own key -- not provided)

This script has NOT been run end-to-end in development (no API key
available in that environment) -- the request/parsing logic is written
carefully and mirrors the Anthropic API docs, but test it on a handful of
rows first (see --limit) before running the full golden set, and expect to
debug the response-parsing step in particular.

Output: outputs/eval/llm_baseline_predictions.csv, appended metrics in
outputs/eval/baseline_metrics.txt (same file the other two baselines write to,
so all three tiers end up side by side for the report).
"""
import argparse
import json
import time

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config
import intent_taxonomy as tax

load_dotenv(override=True)  # .env always wins, even over a stale $env: var set earlier in this shell session

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap/fast default; pass --model claude-sonnet-5 for the higher-quality tier

# A couple of hand-picked examples per intent for the prompt. These are
# deliberately NOT drawn from the golden set (that would leak eval data
# into the few-shot prompt) -- they're written by hand from patterns seen
# while reading through the golden set during labeling.
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


def build_prompt(customer_text: str, prior_context: str | None) -> str:
    intent_list = "\n".join(f"- {i}" for i in tax.INTENTS)
    context_block = f'\nPrior message in the same thread: "{prior_context}"\n' if prior_context and pd.notna(prior_context) else ""
    return f"""You are classifying a customer support tweet sent to Apple Support into exactly one intent.

Intents:
{intent_list}
- other (use ONLY if truly none of the above fit -- e.g. developer-tool bugs, phishing reports, complaints about the support process itself rather than a product issue)

{FEW_SHOT_EXAMPLES}
{context_block}
Customer message: "{customer_text}"

Respond with ONLY the intent name, nothing else. No punctuation, no explanation."""


def classify_one(client, customer_text, prior_context, model, max_retries=3):
    prompt = build_prompt(customer_text, prior_context)
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip().lower().replace(" ", "_")
            valid = set(tax.INTENTS) | {"other"}
            if raw in valid:
                return raw
            # loose match in case the model added punctuation/extra words
            for v in valid:
                if v in raw:
                    return v
            return "other"  # couldn't parse -> treat as out-of-taxonomy rather than crash
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[llm_baseline] failed after {max_retries} attempts: {e}")
                return "other"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only classify the first N golden rows, for a quick smoke test before the full run")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help=f"model string to use (default: {DEFAULT_MODEL}, cheap/fast). "
                              f"Use claude-sonnet-5 for the higher-quality tier once you're ready for the full run.")
    args = parser.parse_args()

    if anthropic is None:
        raise SystemExit("pip install anthropic first")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    golden = pd.read_csv(config.GOLDEN_SET_LABELED)
    if args.limit:
        golden = golden.head(args.limit)
        print(f"[llm_baseline] --limit set, running on first {len(golden)} rows only")

    preds = []
    for i, row in golden.iterrows():
        pred = classify_one(client, row["customer_text"], row.get("prior_context"), args.model)
        preds.append(pred)
        if (i + 1) % 20 == 0:
            print(f"[llm_baseline] classified {i + 1}/{len(golden)}")

    golden = golden.copy()
    golden["llm_pred"] = preds

    in_taxonomy = golden[golden["gold_intent"].isin(tax.INTENTS)]
    out_of_taxonomy = golden[~golden["gold_intent"].isin(tax.INTENTS)]

    y_true = in_taxonomy["gold_intent"]
    y_pred = in_taxonomy["llm_pred"]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cls_report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== LLM FEW-SHOT ({args.model}) ===")
    print(f"accuracy:  {acc:.3f}")
    print(f"macro-F1:  {macro_f1:.3f}")
    print(cls_report)
    print(f"out-of-taxonomy golden rows (not scored): {len(out_of_taxonomy)}")
    if len(out_of_taxonomy):
        correctly_flagged = (out_of_taxonomy["llm_pred"] == "other").sum()
        print(f"  of those, LLM correctly said 'other' for {correctly_flagged}/{len(out_of_taxonomy)} "
              f"-- worth reporting separately: can the model recognize when NONE of your intents fit?")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = config.EVAL_DIR / "llm_baseline_predictions.csv"
    golden.to_csv(pred_path, index=False)

    metrics_path = config.EVAL_DIR / "baseline_metrics.txt"
    with open(metrics_path, "a") as f:
        f.write(f"\n\n=== LLM FEW-SHOT ({args.model}) ===\n")
        f.write(f"accuracy:  {acc:.3f}\nmacro-F1:  {macro_f1:.3f}\n")
        f.write(cls_report)
        f.write(f"\nout-of-taxonomy rows: {len(out_of_taxonomy)}\n")

    print(f"\n[llm_baseline] wrote {pred_path}")
    print(f"[llm_baseline] appended to {metrics_path}")


if __name__ == "__main__":
    main()
