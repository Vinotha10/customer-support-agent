# AI Support Agent for AppleSupport — Hiver SDE Intern Take-Home

Classifies incoming AppleSupport customer tweets into one of 8 hand-derived
intents, drafts a reply grounded in similar historically-resolved
exchanges, and decides auto-handle vs. escalate with a stated reason.

**Start here:** [`REPORT.md`](REPORT.md) — problem framing, results vs.
baselines, failure analysis, and what's misleading about the headline
numbers. [`DECISION_LOG.md`](DECISION_LOG.md) — 15 non-obvious decisions
and why. [`LABELING_GUIDE.md`](LABELING_GUIDE.md) — how the 200-row
golden set was sampled and labeled.

## Setup (< 2 minutes)

```bash
git clone <this-repo>
cd hiver-support-agent
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data (~5 minutes, manual — Kaggle requires auth)

Download `twcs.csv` from
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
and place it at `data/raw/twcs.csv`. Can't be automated — needs your
Kaggle credentials.

## Reproduce everything (~20-30 minutes total, excluding LLM API calls)

```bash
cd src

# Day 1-2: data pipeline + taxonomy discovery
python data_prep.py              # raw twcs.csv -> reconstructed pairs (~1-3 min)
python eda.py                    # volume/response-time/token stats (~10s)
python cluster_intents.py        # TF-IDF+KMeans taxonomy discovery (~1-2 min)
python apply_intent_labels.py    # applies the taxonomy to all ~101K pairs
python build_golden_set.py       # samples 200 balanced examples for hand-labeling
python -m pytest ../tests/ -v    # thread-reconstruction unit tests

# Day 3: classifier baselines (golden set must already be hand-labeled --
# outputs/golden_set/golden_set_labeled.csv is included in this repo)
python baseline_classifiers.py           # trivial + TF-IDF/LogReg (~30s)
pip install anthropic  # or: pip install google-genai
python llm_baseline_classifier.py --limit 10   # or llm_baseline_classifier_gemini.py

# Day 4: the actual agent (needs an LLM -- see Provider setup below)
python reply_agent.py --text "my battery drains so fast since the update"
python reply_agent.py --golden-set       # full 200-row run, resumable if interrupted

# Day 5: evaluation harness
python evaluate_escalation.py            # no API needed
python llm_judge.py                      # scores all 200 replies
python build_judge_calibration_set.py    # samples rows for human calibration
python score_calibration.py              # fast keypress scoring tool
python judge_agreement.py                # judge-vs-human agreement
```

## Provider setup (pick one)

Copy `.env.example` to `.env` and fill in ONE of:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
```
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
```
```
LLM_PROVIDER=ollama
# no key needed -- install from ollama.com, then: ollama pull llama3.2:3b
```

`llm_client.py` is provider-agnostic and includes automatic retry/backoff
and model-fallback logic — this project hit real quota exhaustion and
model deprecations on Gemini's free tier during development, documented
in DECISION_LOG.md #14 and REPORT.md.

## Why AppleSupport

High message volume and a support style with recognizable, repeatable
resolution patterns (troubleshooting steps, "DM us," refund language) —
reply drafting is grounded in *how this brand has historically resolved
similar issues*, which needs a large-enough set of resolved-looking
exchanges per intent to retrieve from.

## Repo layout

```
src/
  config.py                      # brand + path + sampling knobs
  data_prep.py                   # raw twcs.csv -> reconstructed pairs
  eda.py                         # volume/response-time/token stats
  cluster_intents.py             # TF-IDF+KMeans taxonomy discovery
  intent_taxonomy.py             # loads the human-edited taxonomy mapping
  apply_intent_labels.py         # applies taxonomy to all pairs
  build_golden_set.py            # samples the 200-row golden set
  baseline_classifiers.py        # trivial + TF-IDF/LogReg, saves the trained model
  llm_baseline_classifier.py     # LLM few-shot baseline (Anthropic)
  llm_baseline_classifier_gemini.py  # LLM few-shot baseline (Gemini)
  llm_client.py                  # provider-agnostic LLM wrapper, retry/fallback
  retrieve_exemplars.py          # per-intent TF-IDF retrieval for grounding
  reply_agent.py                 # the actual agent: classify -> retrieve -> draft+escalate
  evaluate_escalation.py         # escalation precision/recall/F1
  llm_judge.py                   # LLM-as-judge reply quality scoring
  build_judge_calibration_set.py # samples rows for human calibration
  score_calibration.py           # fast keypress-driven scoring tool
  judge_agreement.py             # judge-vs-human agreement metrics
tests/
  test_data_prep.py              # unit tests for thread reconstruction
  fixtures/mini_twcs.csv
data/
  raw/                           # put twcs.csv here (gitignored, ~350MB)
  processed/                     # generated pair data (gitignored, regenerate via data_prep.py)
outputs/
  clusters/intent_taxonomy_mapping.csv   # hand-named taxonomy (committed)
  golden_set/golden_set_labeled.csv      # the 200-row hand-labeled golden set (committed)
  eval/                                  # metrics, predictions, judge scores (committed)
  figures/                               # EDA charts (committed)
```

## Key results (see REPORT.md for full analysis)

| Metric | Value |
|---|---|
| Intent classifier (TF-IDF+LogReg) accuracy / macro-F1 | 0.735 / 0.717 |
| Escalation precision / **recall** | 0.933 / **0.149** |

The escalation recall is the headline finding of this project — the
agent's precision looks excellent in isolation but hides that it misses
85% of messages that genuinely needed human attention. See REPORT.md
Sections 3-4 for the full failure analysis and why this number is easy
to misread.
