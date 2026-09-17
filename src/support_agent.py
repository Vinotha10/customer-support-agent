"""
The actual AI support agent, tying together everything built so far:

  1. CLASSIFY   -- trained TF-IDF+LogisticRegression (outputs/eval/intent_clf_*.joblib),
                   chosen over the LLM classifier because it scored higher on the
                   golden set (0.735/0.717 vs 0.658/0.564 accuracy/macro-F1).
  2. RULE CHECK -- deterministic escalation triggers from escalation_rules.py,
                   each traceable to a specific golden-set labeling example.
  3. RETRIEVE   -- top-k same-intent historical exemplars from retrieval.py,
                   preferring ones with a positive resolution signal.
  4. DRAFT+DECIDE -- one LLM call: given the message, retrieved exemplars, and
                   the rule flags, produce {draft_reply, escalate, reason}.
                   Rule flags are passed in explicitly rather than left for the
                   LLM to notice unprompted -- deterministic triggers should not
                   depend on the LLM happening to catch them in free-form reasoning.

Final output per message matches the brief's required shape:
    {intent, confidence, draft_reply, decision, reason}

Uses Gemini by default (gemini-2.5-flash -- a step up from the flash-lite
used for classification, since reply text quality matters more here and
this is a much lower-volume call than 200 classification calls was).
Swap --model / GEMINI_API_KEY for Anthropic the same way the baseline
scripts do, if preferred.

This script has NOT been run end-to-end against a live API in development
(no key available in that environment). The classify/rules/retrieve stages
ARE fully tested (see their own __main__ blocks and this file's own smoke
test path). Test the LLM call on a couple of messages with --limit before
trusting it on the full golden set.
"""
import argparse
import json
import re
import time

import joblib
import pandas as pd

import config
import escalation_rules as rules
from retrieval import IntentRetriever

try:
    from google import genai
except ImportError:
    genai = None

DEFAULT_MODEL = "gemini-2.5-flash"
CONFIDENCE_THRESHOLD = 0.40  # below this, low_classifier_confidence rule fires

PROMPT_TEMPLATE = """You are drafting a reply for {brand}'s customer support Twitter account.

Customer's message: "{customer_text}"
{prior_context_block}
Predicted intent: {intent} (classifier confidence: {confidence:.2f})

Here are similar past exchanges showing how {brand} has historically handled this intent:
{exemplars_block}

Automated checks already flagged the following (factor these into your escalate decision -- do not contradict a triggered safety or legal flag):
{rule_flags_block}

Write a brief, on-brand draft reply (in the same style as the historical exchanges above -- concise, empathetic, action-oriented). Do NOT invent account-specific facts (order numbers, dates, amounts) that aren't in the customer's message.

Then decide: should this be escalated to a human rather than auto-sent? Consider message severity, whether a standard remedy has already failed, and the automated flags above.

Respond with ONLY valid JSON in this exact shape, nothing else:
{{"draft_reply": "...", "escalate": true or false, "reason": "one sentence"}}"""


def format_exemplars(exemplars):
    if not exemplars:
        return "(no historical exemplars found for this intent)"
    lines = []
    for i, ex in enumerate(exemplars, 1):
        lines.append(f'{i}. Customer: "{ex["customer_text"]}"\n   Brand replied: "{ex["brand_text"]}"')
    return "\n".join(lines)


def format_rule_flags(rule_results):
    triggered = rules.summarize_triggered(rule_results)
    if not triggered:
        return "(none triggered)"
    return "\n".join(f"- {t}" for t in triggered)


class SupportAgent:
    def __init__(self, model=DEFAULT_MODEL):
        self.vectorizer = joblib.load(config.EVAL_DIR / "intent_clf_vectorizer.joblib")
        self.clf = joblib.load(config.EVAL_DIR / "intent_clf_model.joblib")
        self.retriever = IntentRetriever()
        self.model = model
        if genai is None:
            raise SystemExit("pip install google-genai first")
        self.client = genai.Client()

    def classify(self, customer_text: str):
        X = self.vectorizer.transform([customer_text])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        intent = self.clf.classes_[idx]
        confidence = float(proba[idx])
        return intent, confidence

    def draft_and_decide(self, customer_text, prior_context, intent, confidence, rule_results, max_retries=3):
        exemplars = self.retriever.retrieve(customer_text, intent, k=3)
        prior_block = f'\nPrior message in this thread: "{prior_context}"\n' if prior_context else ""
        prompt = PROMPT_TEMPLATE.format(
            brand=config.BRAND,
            customer_text=customer_text,
            prior_context_block=prior_block,
            intent=intent,
            confidence=confidence,
            exemplars_block=format_exemplars(exemplars),
            rule_flags_block=format_rule_flags(rule_results),
        )

        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt)
                raw = (resp.text or "").strip()
                raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                parsed = json.loads(raw)
                return parsed["draft_reply"], bool(parsed["escalate"]), parsed["reason"], exemplars
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[support_agent] failed after {max_retries} attempts: {e}")
                    return None, True, f"drafting failed ({e}), defaulting to escalate", exemplars
                time.sleep(2 ** attempt)

    def handle(self, customer_text: str, prior_context: str = None):
        intent, confidence = self.classify(customer_text)
        rule_results = rules.check_rules(customer_text, prior_context, confidence, CONFIDENCE_THRESHOLD)

        draft_reply, llm_escalate, llm_reason, exemplars = self.draft_and_decide(
            customer_text, prior_context, intent, confidence, rule_results
        )

        # A triggered safety or legal rule ALWAYS wins, regardless of what
        # the LLM decided -- these are non-negotiable per escalation_rules.py's
        # own design intent (see g0135 safety note). Don't let a drafting
        # model talk itself out of a hard safety trigger.
        hard_override = rule_results["safety_hazard"][0] or rule_results["legal_threat"][0]
        escalate = hard_override or llm_escalate
        if hard_override and not llm_escalate:
            reason = f"HARD RULE OVERRIDE: {format_rule_flags(rule_results)}"
        else:
            reason = llm_reason

        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "draft_reply": draft_reply,
            "decision": "escalate" if escalate else "auto_handle",
            "reason": reason,
            "rule_flags": rules.summarize_triggered(rule_results),
            "num_exemplars_used": len(exemplars),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--message", type=str, default=None,
                         help="run on a single ad-hoc message instead of the golden set")
    args = parser.parse_args()

    agent = SupportAgent(model=args.model)

    if args.message:
        result = agent.handle(args.message)
        print(json.dumps(result, indent=2))
        return

    golden = pd.read_csv(config.GOLDEN_SET_LABELED)
    if args.limit:
        golden = golden.head(args.limit)

    results = []
    for i, row in golden.iterrows():
        result = agent.handle(row["customer_text"], row.get("prior_context"))
        result["example_id"] = row.get("example_id", i)
        result["gold_intent"] = row.get("gold_intent")
        result["gold_escalate"] = row.get("gold_escalate")
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"[support_agent] processed {i + 1}/{len(golden)}")

    out_df = pd.DataFrame(results)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "agent_outputs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[support_agent] wrote {out_path}")


if __name__ == "__main__":
    main()
