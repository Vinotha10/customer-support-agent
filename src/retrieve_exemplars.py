"""
Given a new customer message and its (predicted or given) intent, retrieves
the top-k most similar historically-resolved (customer_text, brand_text)
pairs from the SAME intent in the training pool. These become the grounding
exemplars for reply_agent.py's drafting prompt -- this is what makes the
draft "grounded in how the brand has historically resolved similar issues"
rather than a generic canned response.

Deliberately per-intent, not global: retrieving across all intents would
risk pulling a battery-drain exemplar for a screen-crack complaint just
because the wording happens to be superficially similar. Restricting the
candidate pool to the predicted intent first, then ranking by similarity
within it, keeps retrieval on-topic even when TF-IDF similarity alone
would be a weak signal.

Known limitation (documented, not silently ignored): not every historical
brand reply is a GOOD exemplar of a correct resolution -- failure analysis
during golden-set labeling found at least one case (a lock-screen
complaint that got a reply about an unrelated "Podcast app") where the
brand's own historical reply was itself mismatched to the customer's
actual issue. This retrieval has no quality filter for that yet -- see
DECISION_LOG.md and the report's "what I chose not to build" section.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

LABELED_PATH = config.PROCESSED_PARQUET.parent / f"{config.BRAND.lower()}_pairs_labeled.parquet"


class ExemplarRetriever:
    def __init__(self, train_pool: pd.DataFrame):
        self.train_pool = train_pool
        self._vectorizers = {}
        self._matrices = {}
        self._pools = {}
        for intent, group in train_pool.groupby("intent"):
            # skip rows with no brand reply text -- can't ground a draft on nothing
            group = group[group["brand_text"].notna() & (group["brand_text"].str.len() > 0)]
            if len(group) == 0:
                continue
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, stop_words="english")
            X = vec.fit_transform(group["customer_text"].fillna(""))
            self._vectorizers[intent] = vec
            self._matrices[intent] = X
            self._pools[intent] = group.reset_index(drop=True)

    def retrieve(self, customer_text: str, intent: str, k: int = 3):
        if intent not in self._vectorizers:
            return []  # e.g. intent == "other", or an intent with zero usable exemplars
        vec = self._vectorizers[intent]
        X = self._matrices[intent]
        pool = self._pools[intent]

        q = vec.transform([customer_text])
        sims = cosine_similarity(q, X)[0]
        top_idx = np.argsort(sims)[::-1][:k]

        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue  # no lexical overlap at all -- not a real match, don't force it
            row = pool.iloc[i]
            results.append({
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "similarity": float(sims[i]),
            })
        return results


def load_retriever(exclude_ids=None) -> ExemplarRetriever:
    train_pool = pd.read_parquet(LABELED_PATH)
    if exclude_ids:
        train_pool = train_pool[~train_pool["customer_tweet_id"].astype(str).isin(exclude_ids)]
    return ExemplarRetriever(train_pool)


if __name__ == "__main__":
    # quick manual smoke test
    retriever = load_retriever()
    test_text = "my battery drains so fast since the last update"
    results = retriever.retrieve(test_text, "battery_drain", k=3)
    print(f"query: {test_text}\n")
    for r in results:
        print(f"  sim={r['similarity']:.3f}  customer: {r['customer_text'][:80]}")
        print(f"              brand reply: {r['brand_text'][:80]}\n")
