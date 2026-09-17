"""
The actual end-to-end agent: given a customer message, this
  1. classifies intent (using the Day 3 TF-IDF+LogReg classifier -- it beat
     the zero-shot LLM baseline, so it's the right choice here, not just in
     the results table),
  2. retrieves similar historically-resolved exchanges for that intent
     (retrieve_exemplars.py, no LLM call -- fast, free, deterministic),
  3. makes ONE LLM call that both drafts a grounded reply AND decides
     auto-handle vs. escalate with a stated reason.

One combined call (not two separate ones for draft + escalate) to control
API cost/quota given today's provider issues, and because the escalation
decision often depends on judgments the model needs to make while
reading the message anyway (severity, whether a standard fix likely
already failed, safety language, etc.) -- splitting it into two calls
would mean re-deriving the same read of the message twice.

The escalation rubric embedded in the prompt is drawn directly from
LABELING_GUIDE.md's own criteria, so the agent's escalation judgment is
methodologically consistent with how the golden set itself was labeled --
not a different, invented standard.

Known gap (see decision log): retrieval has no filter for historical
brand replies that were themselves poor/mismatched resolutions (see the
g0122 "lock-screen complaint answered about the Podcast app" finding from
golden-set labeling). A bad historical exemplar can currently be retrieved
and grounded on as if it were a good one.

Usage:
    export LLM_PROVIDER=anthropic  (or gemini)
    export ANTHROPIC_API_KEY=... (or GEMINI_API_KEY=...)

    python reply_agent.py --text "my battery drains so fast since the update"
    python reply_agent.py --golden-set --limit 10     # smoke test on real data
    python reply_agent.py --golden-set                # full run, for Day 5 eval
"""
import argparse
import json
import re
import time

import joblib
import pandas as pd

import config
import intent_taxonomy as tax
import llm_client
from retrieve_exemplars import load_retriever

CLF_VECTORIZER_PATH = config.EVAL_DIR / "intent_clf_vectorizer.joblib"
CLF_MODEL_PATH = config.EVAL_DIR / "intent_clf_model.joblib"

ESCALATION_RUBRIC = """
Escalate to a human if ANY of the following apply (based on the same
criteria used to build this project's hand-labeled evaluation set):
- Safety-relevant language (e.g. battery swelling, fire/smoke, injury)
- Explicit legal threats, or account security/hacking/phishing reports
- Financial asks: refund, replacement, billing disputes, warranty cost disputes
- Data loss (photos, contacts, messages) reported as already happened
- Customer states they already tried the standard fix (restart, update,
  DM'd support, etc.) and it did NOT work
- Severe impact: device completely unusable, DOA, or failing very frequently
- The message is too vague, fragment-like, or image-dependent to act on
  with confidence
- None of the above, but you are not genuinely confident a standard reply
  would resolve this

Otherwise, auto-handle: a clearly recognized issue type with a standard,
low-risk resolution (e.g. a known bug with a documented workaround, a
first-contact triage question, factual/expected-behavior clarification).
""".strip()


def load_classifier():
    vectorizer = joblib.load(CLF_VECTORIZER_PATH)
    clf = joblib.load(CLF_MODEL_PATH)
    return vectorizer, clf


def classify_intent(vectorizer, clf, customer_text: str) -> str:
    X = vectorizer.transform([customer_text])
    return clf.predict(X)[0]


def build_prompt(customer_text, prior_context, intent, exemplars):
    exemplar_block = ""
    if exemplars:
        lines = []
        for i, ex in enumerate(exemplars, 1):
            lines.append(f'{i}. Customer: "{ex["customer_text"]}"\n   Brand replied: "{ex["brand_text"]}"')
        exemplar_block = "Similar issues this brand has resolved before:\n" + "\n".join(lines)
    else:
        exemplar_block = "(No closely similar historical exchange was found for this intent.)"

    context_block = f'\nEarlier message in the same thread: "{prior_context}"\n' if prior_context and pd.notna(prior_context) else ""

    return f"""You are drafting an Apple Support Twitter reply and deciding how to handle a customer message.

Predicted intent: {intent}

{exemplar_block}
{context_block}
Customer message: "{customer_text}"

{ESCALATION_RUBRIC}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"reply": "<the drafted reply, in the brand's typical tone, grounded in the similar past resolutions above where relevant>", "escalate": <true or false>, "escalate_reason": "<one specific sentence -- if not escalating, briefly say why the standard reply is sufficient instead>"}}"""


def parse_json_response(raw: str) -> dict:
    # models sometimes wrap JSON in ```json fences despite instructions -- strip if present
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def run_agent(customer_text, prior_context, vectorizer, clf, retriever):
    intent = classify_intent(vectorizer, clf, customer_text)
    exemplars = retriever.retrieve(customer_text, intent, k=3)
    prompt = build_prompt(customer_text, prior_context, intent, exemplars)

    raw = llm_client.chat(prompt)
    try:
        parsed = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        # don't silently fail -- surface the bad response so it can be debugged,
        # and fall back to a safe default (escalate) rather than guessing
        print(f"[reply_agent] WARNING: couldn't parse model response as JSON ({e}). Raw response:\n{raw}")
        parsed = {"reply": None, "escalate": True, "escalate_reason": "agent response could not be parsed -- escalate as a safe default"}

    return {
        "intent": intent,
        "num_exemplars_found": len(exemplars),
        # top exemplar's text, logged for future judge-groundedness checks
        # (not available for any run completed before this was added --
        # see llm_judge.py's documented limitation)
        "top_exemplar_customer_text": exemplars[0]["customer_text"] if exemplars else None,
        "top_exemplar_brand_text": exemplars[0]["brand_text"] if exemplars else None,
        "reply": parsed.get("reply"),
        "escalate": parsed.get("escalate"),
        "escalate_reason": parsed.get("escalate_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="classify/draft/decide for a single ad-hoc message")
    parser.add_argument("--prior-context", type=str, default=None)
    parser.add_argument("--golden-set", action="store_true", help="run on the full hand-labeled golden set")
    parser.add_argument("--limit", type=int, default=None, help="only process first N golden-set rows")
    parser.add_argument("--sleep", type=float, default=None,
                         help="seconds to wait between calls in --golden-set mode; "
                              "defaults to a provider-appropriate pacing (avoids free-tier rate limits)")
    parser.add_argument("--restart", action="store_true",
                         help="ignore any existing outputs/eval/agent_outputs.csv and start fresh "
                              "instead of resuming from where a previous run left off")
    args = parser.parse_args()

    print("[reply_agent] loading classifier...")
    vectorizer, clf = load_classifier()
    print("[reply_agent] loading training pool + building retriever "
          "(first pyarrow import can take 20-60s on some Windows setups, especially "
          "with antivirus scanning -- this is normal, please wait rather than Ctrl+C)...")
    retriever = load_retriever()
    print("[reply_agent] ready")

    if args.text:
        result = run_agent(args.text, args.prior_context, vectorizer, clf, retriever)
        print(json.dumps(result, indent=2))
        return

    if args.golden_set:
        golden = pd.read_csv(config.GOLDEN_SET_LABELED)

        missing_text = golden["customer_text"].isna()
        if missing_text.any():
            bad_ids = golden.loc[missing_text, "example_id"].tolist() if "example_id" in golden.columns else golden.index[missing_text].tolist()
            print(f"[reply_agent] WARNING: dropping {missing_text.sum()} row(s) with missing "
                  f"customer_text (can't classify/draft/ground on empty text): {bad_ids}")
            print(f"[reply_agent] this is the same row flagged in day3_failure_analysis_notes.md "
                  f"-- worth a quick look at outputs/golden_set/golden_set_labeled.csv before "
                  f"finalizing golden-set counts in the report")
            golden = golden[~missing_text].reset_index(drop=True)

        if args.limit:
            golden = golden.head(args.limit)
            print(f"[reply_agent] --limit set, running on first {len(golden)} rows only")

        config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.EVAL_DIR / "agent_outputs.csv"

        # RESUME SUPPORT: this loop has now crashed mid-run three times
        # (NaN row, rate limit, transient 503) over a run that takes
        # 15-20+ minutes. Writing output only at the end means any crash
        # throws away everything already completed AND burns API quota
        # re-doing it on a restart. Instead: write each row to disk
        # immediately after it completes, and on startup skip any
        # example_id already present in that file -- a rerun of the exact
        # same command after a crash picks up where it left off.
        already_done = set()
        if out_path.exists() and not args.restart:
            existing = pd.read_csv(out_path)
            already_done = set(existing["example_id"].astype(str))
            print(f"[reply_agent] found {len(already_done)} already-completed rows in {out_path}, resuming...")
        elif out_path.exists() and args.restart:
            print(f"[reply_agent] --restart set, ignoring existing {out_path} and starting fresh")
            out_path.unlink()

        pace = args.sleep if args.sleep is not None else llm_client.pacing_seconds()
        n_done_this_run = 0
        for i, row in golden.iterrows():
            eid = str(row.get("example_id", i))
            if eid in already_done:
                continue

            result = run_agent(row["customer_text"], row.get("prior_context"), vectorizer, clf, retriever)
            result["example_id"] = eid
            result["customer_text"] = row["customer_text"]
            result["gold_intent"] = row.get("gold_intent")
            result["gold_escalate"] = row.get("gold_escalate")

            row_df = pd.DataFrame([result])
            row_df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            already_done.add(eid)
            n_done_this_run += 1

            if n_done_this_run % 20 == 0:
                print(f"[reply_agent] processed {len(already_done)}/{len(golden)} total "
                      f"({n_done_this_run} this run)")
            time.sleep(pace)

        print(f"\n[reply_agent] done -- {out_path} has {len(already_done)}/{len(golden)} rows")
        print("[reply_agent] this feeds Day 5's reply-quality eval (LLM-as-judge) and "
              "the escalation precision/recall comparison against gold_escalate")
        return

    parser.error("pass either --text \"...\" or --golden-set")


if __name__ == "__main__":
    main()
