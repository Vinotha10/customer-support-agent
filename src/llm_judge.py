"""
Scores every drafted reply in agent_outputs.csv on a simple rubric, using
an LLM judge. Per the assignment brief, this needs a demonstrated human-
agreement check -- see build_judge_calibration_set.py and
judge_agreement.py, run after this.

Rubric (1-5 each):
  - correctness: does the reply plausibly and appropriately address what
    the customer actually said (not generic filler, not answering a
    different problem)
  - tone: professional, empathetic, matches a typical brand-support voice
  - actionability: does the customer come away with a clear next step or
    resolution (a troubleshooting step, a request for more info, a
    concrete answer) rather than a vague non-answer

Deliberately NOT scoring "groundedness against the specific retrieved
historical exemplar" -- agent_outputs.csv only stores how many exemplars
were found, not their text, so a strict grounding check isn't possible
against already-completed runs without a costly full re-run. This is a
documented limitation, not a silently skipped dimension -- see
DECISION_LOG.md. (reply_agent.py has been updated to log exemplar text
for any FUTURE run, so this can be closed later if there's time.)

Resumable, same pattern as reply_agent.py, given how much API instability
this project has already hit -- writes each score to disk immediately.

Output: outputs/eval/judge_scores.csv
"""
import argparse
import json
import re
import time

import pandas as pd

import config
import llm_client

RUBRIC_PROMPT = """You are evaluating an AI-drafted customer support reply for quality.

Customer message: "{customer_text}"

Drafted reply: "{reply}"

Score the reply on three dimensions, 1 (poor) to 5 (excellent):
- correctness: does it plausibly and specifically address what the customer actually said? (1 = generic/off-topic, 5 = clearly on-point)
- tone: professional, empathetic, appropriate for a brand support account? (1 = robotic/cold/inappropriate, 5 = warm and professional)
- actionability: does the customer get a clear next step or resolution? (1 = vague non-answer, 5 = clear concrete action or answer)

Respond with ONLY a JSON object, no other text:
{{"correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "rationale": "<one short sentence>"}}"""


def parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def judge_one(customer_text, reply):
    if not reply or pd.isna(reply):
        return {"correctness": 1, "tone": 1, "actionability": 1, "rationale": "no reply was drafted (agent failure)"}
    prompt = RUBRIC_PROMPT.format(customer_text=customer_text, reply=reply)
    raw = llm_client.chat(prompt)
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[llm_judge] WARNING: couldn't parse judge response ({e}): {raw[:100]}")
        return {"correctness": None, "tone": None, "actionability": None, "rationale": "unparseable judge response"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    agent_out = pd.read_csv(config.EVAL_DIR / "agent_outputs.csv")
    if args.limit:
        agent_out = agent_out.head(args.limit)

    out_path = config.EVAL_DIR / "judge_scores.csv"
    already_done = set()
    if out_path.exists() and not args.restart:
        existing = pd.read_csv(out_path)
        already_done = set(existing["example_id"].astype(str))
        print(f"[llm_judge] found {len(already_done)} already-scored rows, resuming...")
    elif out_path.exists() and args.restart:
        out_path.unlink()

    pace = llm_client.pacing_seconds()
    n_done = 0
    for i, row in agent_out.iterrows():
        eid = str(row.get("example_id", i))
        if eid in already_done:
            continue
        scores = judge_one(row["customer_text"], row.get("reply"))
        scores["example_id"] = eid
        scores["customer_text"] = row["customer_text"]
        scores["reply"] = row.get("reply")
        pd.DataFrame([scores]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        already_done.add(eid)
        n_done += 1
        if n_done % 20 == 0:
            print(f"[llm_judge] scored {len(already_done)}/{len(agent_out)} total ({n_done} this run)")
        time.sleep(pace)

    print(f"\n[llm_judge] done -- {out_path} has {len(already_done)}/{len(agent_out)} rows")
    print("[llm_judge] next: python build_judge_calibration_set.py, then hand-score it, "
          "then python judge_agreement.py")


if __name__ == "__main__":
    main()
