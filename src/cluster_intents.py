"""
Discovers candidate intent clusters from customer messages so you're not
hand-guessing the taxonomy — you look at what's actually in the data, then
collapse/name clusters yourself (that manual step is documented in the
decision log, not automated, because that's the part Hiver is testing).

Default: TF-IDF + KMeans. This is fully offline (no model download), so a
fresh clone reproduces this step without any network dependency — matters
for the "<15 min from clone" requirement.

Swap-in option (commented below): sentence-transformers embeddings +
KMeans, if you already have a model cached locally (you've used
all-MiniLM-L6-v2 before). Better cluster quality, but not the default here
since it introduces a HuggingFace download dependency for anyone re-running
this fresh.

Output: outputs/clusters/cluster_samples.csv
    columns: cluster_id, top_terms, sample_text (5 per cluster)
You read this file, decide on ~6-9 named intents, and write the mapping
in src/intent_taxonomy.py (next step, not in this script) as
    CLUSTER_TO_INTENT = {0: "shipping_delay", 1: "billing_refund", ...}
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

import config

N_SAMPLES_PER_CLUSTER = 6
K_CANDIDATES = [6, 8, 10, 12]  # sweep to help you pick taxonomy granularity
SILHOUETTE_SUBSAMPLE = 1000     # silhouette is O(n^2)-ish; keep it cheap

# Generic sklearn English stopwords aren't enough here: the brand's own name,
# generic product nouns, and Kaggle's anonymized-mention placeholders (e.g.
# "115858") show up in nearly every tweet regardless of actual intent, and
# without stripping them TF-IDF partly clusters on shared boilerplate instead
# of on what the issue actually is. Numeric-only tokens are excluded via the
# token_pattern below rather than an explicit list, since anonymized IDs vary.
DOMAIN_STOPWORDS = {
    "apple", "applesupport", "iphone", "ipad", "ipod", "macbook", "imac",
    "ios", "https", "http", "com", "www", "rt", "amp", "co",
}
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))


def main():
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    texts_full = df["customer_text"].dropna()

    n = min(config.MAX_PAIRS_FOR_CLUSTERING, len(texts_full))
    texts = texts_full.sample(n=n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"[cluster] clustering {n} customer messages (of {len(texts_full)} total)")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=CUSTOM_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",  # drop pure-numeric tokens
                                                        # (e.g. anonymized "115858" ids)
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    best_k, best_score, best_model = None, -1, None
    for k in K_CANDIDATES:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)
        sub_idx = np.random.RandomState(config.RANDOM_SEED).choice(
            n, size=min(SILHOUETTE_SUBSAMPLE, n), replace=False
        )
        score = silhouette_score(X[sub_idx], labels[sub_idx])
        print(f"[cluster] k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    print(f"[cluster] selected k={best_k} (silhouette={best_score:.4f})")
    labels = best_model.labels_

    # top terms per cluster via centroid weights
    order_centroids = best_model.cluster_centers_.argsort()[:, ::-1]

    rows = []
    rng = np.random.RandomState(config.RANDOM_SEED)
    for c in range(best_k):
        top_terms = ", ".join(terms[order_centroids[c, :10]])
        idxs = np.where(labels == c)[0]
        sample_idxs = rng.choice(idxs, size=min(N_SAMPLES_PER_CLUSTER, len(idxs)), replace=False)
        for i in sample_idxs:
            rows.append({
                "cluster_id": c,
                "cluster_size": len(idxs),
                "top_terms": top_terms,
                "sample_text": texts.iloc[i],
            })

    out = pd.DataFrame(rows).sort_values(["cluster_id"])
    config.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CLUSTERS_DIR / "cluster_samples.csv"
    out.to_csv(out_path, index=False)
    print(f"[cluster] wrote {out_path}")

    # Persist the FITTED vectorizer + KMeans model, not just the sample output.
    # apply_intent_labels.py loads these to assign cluster_id (and therefore
    # intent, via intent_taxonomy.py) to the full ~106k-row dataset, WITHOUT
    # refitting. Refitting on the full corpus would reshuffle which cluster_id
    # means what, silently invalidating every name you just wrote into
    # outputs/clusters/intent_taxonomy_mapping.csv.
    vec_path = config.CLUSTERS_DIR / "tfidf_vectorizer.joblib"
    model_path = config.CLUSTERS_DIR / "kmeans_model.joblib"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(best_model, model_path)
    print(f"[cluster] saved fitted vectorizer -> {vec_path}")
    print(f"[cluster] saved fitted kmeans (k={best_k}) -> {model_path}")

    print("[cluster] next: open this file, read the samples per cluster_id, "
          "and name each one in src/intent_taxonomy.py")


if __name__ == "__main__":
    main()
