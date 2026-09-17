"""
Labels every reconstructed (customer, brand_reply) pair with an intent,
using:
  1. the vectorizer + KMeans model fitted in cluster_intents.py (loaded, not
     refit -- refitting would change what each cluster_id means, silently
     invalidating outputs/clusters/intent_taxonomy_mapping.csv)
  2. the human-edited CLUSTER_TO_INTENT mapping from intent_taxonomy.py

Rows whose cluster is marked include_in_training=False (follow-ups/short
acknowledgments -- not real incoming-customer intents) are split out into a
separate file rather than dropped silently, since they're useful evidence
for the failure-analysis section of the report.

Output:
  data/processed/{brand}_pairs_labeled.parquet   -- has an `intent` column,
                                                     ready for the classifier
                                                     baselines and RAG exemplar
                                                     retrieval
  data/processed/{brand}_pairs_excluded.parquet  -- excluded rows, kept for
                                                     reference/failure analysis
"""
import joblib
import pandas as pd

import config
import intent_taxonomy as tax


def main():
    print(f"[apply_labels] loading fitted vectorizer + kmeans from {config.CLUSTERS_DIR}")
    vectorizer = joblib.load(config.CLUSTERS_DIR / "tfidf_vectorizer.joblib")
    kmeans = joblib.load(config.CLUSTERS_DIR / "kmeans_model.joblib")

    df = pd.read_parquet(config.PROCESSED_PARQUET)
    print(f"[apply_labels] labeling {len(df)} total pairs")

    # customer_text can't be empty/NaN for the vectorizer transform
    texts = df["customer_text"].fillna("")
    X = vectorizer.transform(texts)
    df["cluster_id"] = kmeans.predict(X)

    df["intent"] = df["cluster_id"].map(tax.CLUSTER_TO_INTENT)

    is_excluded = df["cluster_id"].isin(tax.EXCLUDED_CLUSTER_IDS)
    excluded = df[is_excluded].copy()
    labeled = df[~is_excluded].copy()

    print(f"[apply_labels] kept as real intents: {len(labeled)}")
    print(f"[apply_labels] excluded (follow-up/ack, not a new intent): {len(excluded)}")

    print("\n[apply_labels] intent distribution (this IS your class balance for "
          "the classifier -- check for anything wildly dominant before training):")
    counts = labeled["intent"].value_counts()
    for intent, n in counts.items():
        pct = 100 * n / len(labeled)
        print(f"    {intent:30s} {n:7d}  ({pct:5.1f}%)")

    labeled_path = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_labeled.parquet"
    excluded_path = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_excluded.parquet"
    labeled.to_parquet(labeled_path, index=False)
    excluded.to_parquet(excluded_path, index=False)
    print(f"\n[apply_labels] wrote {labeled_path}")
    print(f"[apply_labels] wrote {excluded_path}")


if __name__ == "__main__":
    main()
