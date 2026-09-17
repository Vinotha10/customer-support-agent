"""
Retrieves grounding exemplars for reply drafting: past (customer_text,
brand_text) pairs of the SAME intent, ranked by text similarity to the new
message, with a preference for exemplars that have a positive resolution
signal in their followup_text (reusing the heuristic from eda.py).

Deliberately scoped per-intent rather than searched globally: retrieving
within the same intent keeps exemplars topically relevant and is cheap
(no need to re-rank across the full ~100k corpus for every query), at the
cost of being only as good as the upstream classifier's intent call --
documented tradeoff, not an oversight.

Uses TF-IDF + cosine similarity, consistent with the rest of this
pipeline's "no heavy model download" constraint (see cluster_intents.py).
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

LABELED_PATH = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_labeled.parquet"

# same weak heuristic introduced in eda.py -- NOT ground truth of a
# successful resolution, just a soft ranking signal for exemplar quality
_THANK_WORDS_RE = re.compile(
    r"\b(thanks|thank you|resolved|fixed|works now|appreciate)\b", re.IGNORECASE
)


class IntentRetriever:
    def __init__(self, exclude_ids=None):
        df = pd.read_parquet(LABELED_PATH)
        if exclude_ids:
            df = df[~df["customer_tweet_id"].astype(str).isin(exclude_ids)]
        self.df = df.reset_index(drop=True)
        self.df["_looks_resolved"] = self.df["followup_text"].fillna("").apply(
            lambda t: bool(_THANK_WORDS_RE.search(t))
        )

        # one TF-IDF space fit across everything, then filtered per-intent
        # at query time -- simpler than maintaining N separate vectorizers,
        # and the corpus is small enough (~100k short texts) that this is
        # fast either way.
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                           min_df=2, stop_words="english")
        self.X = self.vectorizer.fit_transform(self.df["customer_text"].fillna(""))

    def retrieve(self, query_text: str, intent: str, k: int = 3, resolved_bonus: float = 0.05):
        """
        Returns up to k exemplars as a list of dicts:
            {customer_text, brand_text, similarity, looks_resolved}
        Ranked by cosine similarity + a small bonus for exemplars with a
        positive-looking followup. The bonus is intentionally small --
        similarity to the actual query still dominates the ranking; this
        only breaks near-ties in favor of exemplars more likely to reflect
        a real resolution rather than a generic "please DM us."
        """
        mask = self.df["intent"] == intent
        if mask.sum() == 0:
            return []

        subset_idx = np.where(mask.values)[0]
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.X[subset_idx]).ravel()

        scores = sims + self.df.iloc[subset_idx]["_looks_resolved"].values * resolved_bonus
        top_k_local = np.argsort(-scores)[:k]
        top_idx = subset_idx[top_k_local]

        results = []
        for i, local_i in zip(top_idx, top_k_local):
            row = self.df.iloc[i]
            results.append({
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "similarity": float(sims[local_i]),
                "looks_resolved": bool(row["_looks_resolved"]),
            })
        return results


if __name__ == "__main__":
    # smoke test -- requires data/processed/applesupport_pairs_labeled.parquet to exist
    retriever = IntentRetriever()
    examples = retriever.retrieve("my phone battery drains so fast since the update", "battery_drain", k=3)
    for ex in examples:
        print(f"sim={ex['similarity']:.3f} resolved={ex['looks_resolved']}")
        print(f"  customer: {ex['customer_text'][:80]}")
        print(f"  brand:    {ex['brand_text'][:80]}")
