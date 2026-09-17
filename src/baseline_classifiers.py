"""
Two baselines for intent classification:
  1. Trivial: always predict the majority class from the training pool.
  2. Simple: TF-IDF + multinomial Logistic Regression.

IMPORTANT methodology notes (also goes in the report):

- TRAINING labels come from cluster_intents.py + intent_taxonomy.py -- i.e.
  unsupervised clustering, not human judgment. The golden-set labeling
  exercise found these disagree with actual human judgment on ~27% of
  cases (54/200). That means whatever a classifier trained on these labels
  scores, it has a ceiling around that same noise level -- it cannot be
  more "correct" than the labels it learned from. This is exactly why
  EVALUATION happens only against the hand-labeled golden set, never
  against the cluster labels themselves (that would just be checking
  clustering agrees with clustering).

- LEAKAGE GUARD: the golden set's 200 examples were sampled FROM the same
  labeled pool used for training. If they aren't excluded before training,
  the classifier would effectively be evaluated on data it already saw.
  This script filters them out by customer_tweet_id before fitting.

- Golden-set rows labeled 'other' (outside the 8-intent taxonomy -- Xcode
  bug reports, support-process complaints, phishing reports, etc.) cannot
  be predicted correctly by a classifier that only knows 8 classes. They
  are reported SEPARATELY as "out-of-taxonomy coverage" rather than folded
  into accuracy, so they don't silently make the model look artificially
  worse (or, if excluded quietly, artificially better) than it is.

Output: outputs/eval/baseline_predictions.csv, outputs/eval/baseline_metrics.txt
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config
import intent_taxonomy as tax

LABELED_PATH = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_labeled.parquet"


def normalize_id(series):
    """
    Tweet IDs should be plain integer strings, but a CSV that's been opened
    and re-saved in Excel often gets its ID column reformatted as a float
    (e.g. "2056267" -> 2056267.0 -> "2056267.0" after str()), which silently
    breaks any exact-match join against the original string-typed ids.
    Strip a trailing ".0" (and any decimal remainder) so both sides compare
    as the same plain digit string regardless of which file went through Excel.
    """
    return series.astype(str).str.replace(r"\.0+$", "", regex=True)


def load_data():
    train_pool = pd.read_parquet(LABELED_PATH)
    golden = pd.read_csv(config.GOLDEN_SET_LABELED)

    golden["customer_tweet_id"] = normalize_id(golden["customer_tweet_id"])
    train_pool["customer_tweet_id"] = normalize_id(train_pool["customer_tweet_id"])

    golden_ids = set(golden["customer_tweet_id"])
    before = len(train_pool)
    train_pool = train_pool[~train_pool["customer_tweet_id"].isin(golden_ids)]
    removed = before - len(train_pool)
    print(f"[baselines] removed {removed} rows from training pool to prevent golden-set leakage")

    # in-taxonomy vs out-of-taxonomy split for evaluation
    in_taxonomy = golden[golden["gold_intent"].isin(tax.INTENTS)].copy()
    out_of_taxonomy = golden[~golden["gold_intent"].isin(tax.INTENTS)].copy()
    print(f"[baselines] golden set: {len(in_taxonomy)} in-taxonomy, "
          f"{len(out_of_taxonomy)} out-of-taxonomy (reported separately, not scored)")

    return train_pool, in_taxonomy, out_of_taxonomy


def trivial_baseline(train_pool, golden_eval):
    majority_class = train_pool["intent"].value_counts().idxmax()
    preds = [majority_class] * len(golden_eval)
    print(f"\n[trivial] always predicts '{majority_class}'")
    return preds


def simple_baseline(train_pool, golden_eval):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X_train = vectorizer.fit_transform(train_pool["customer_text"].fillna(""))
    y_train = train_pool["intent"]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Persisted for reuse by reply_agent.py -- this classifier beat the
    # zero-shot LLM baseline (0.735/0.717 vs 0.658/0.564 accuracy/macro-F1),
    # so the actual agent pipeline uses THIS for intent classification
    # rather than re-asking an LLM to classify from scratch.
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, config.EVAL_DIR / "intent_clf_vectorizer.joblib")
    joblib.dump(clf, config.EVAL_DIR / "intent_clf_model.joblib")
    print(f"[simple] saved trained vectorizer + classifier to {config.EVAL_DIR}")

    X_eval = vectorizer.transform(golden_eval["customer_text"].fillna(""))
    preds = clf.predict(X_eval)
    print(f"\n[simple] TF-IDF + LogisticRegression trained on {len(train_pool)} noisy-labeled examples")
    return preds


def report(name, y_true, y_pred, out_lines):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    header = f"\n=== {name} ==="
    print(header)
    print(f"accuracy:  {acc:.3f}")
    print(f"macro-F1:  {macro_f1:.3f}  <- this is the number that matters given class imbalance")
    cls_report = classification_report(y_true, y_pred, zero_division=0)
    print(cls_report)
    out_lines.append(header)
    out_lines.append(f"accuracy:  {acc:.3f}")
    out_lines.append(f"macro-F1:  {macro_f1:.3f}")
    out_lines.append(cls_report)
    return acc, macro_f1


def main():
    train_pool, golden_eval, out_of_taxonomy = load_data()
    out_lines = []

    y_true = golden_eval["gold_intent"]

    trivial_preds = trivial_baseline(train_pool, golden_eval)
    report("TRIVIAL (majority class)", y_true, trivial_preds, out_lines)

    simple_preds = simple_baseline(train_pool, golden_eval)
    report("SIMPLE (TF-IDF + LogisticRegression)", y_true, simple_preds, out_lines)

    print(f"\n[out-of-taxonomy] {len(out_of_taxonomy)}/{len(golden_eval) + len(out_of_taxonomy)} "
          f"golden examples ({100*len(out_of_taxonomy)/(len(golden_eval)+len(out_of_taxonomy)):.1f}%) "
          f"had a gold_intent outside the 8-class taxonomy (e.g. 'other'). These are "
          f"NOT counted in the accuracy/F1 above -- a real system needs an explicit "
          f"low-confidence/fallback path for exactly this slice, not silence.")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    golden_eval = golden_eval.copy()
    golden_eval["trivial_pred"] = trivial_preds
    golden_eval["simple_pred"] = simple_preds
    pred_path = config.EVAL_DIR / "baseline_predictions.csv"
    golden_eval.to_csv(pred_path, index=False)

    metrics_path = config.EVAL_DIR / "baseline_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("\n".join(out_lines))
        f.write(f"\n\nout-of-taxonomy golden examples (not scored): {len(out_of_taxonomy)}\n")

    print(f"\n[baselines] wrote {pred_path}")
    print(f"[baselines] wrote {metrics_path}")


if __name__ == "__main__":
    main()
