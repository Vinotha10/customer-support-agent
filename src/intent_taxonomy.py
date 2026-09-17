"""
The intent taxonomy itself lives in one place, editable by a human without
touching code: outputs/clusters/intent_taxonomy_mapping.csv

This module just loads that file and exposes it in the shapes later stages
need:
    CLUSTER_TO_INTENT   {cluster_id: intent_name}      -- for clusters kept
    EXCLUDED_CLUSTER_IDS  {cluster_id, ...}             -- follow-ups/acks,
                                                            not real intents
    INTENTS             sorted list of unique intent names (the final taxonomy)
    TAXONOMY_DF         the raw mapping table, for reference/reporting

If you rename or merge a cluster's intent, edit the CSV — nothing here
needs to change.
"""
import pandas as pd

import config

TAXONOMY_CSV = config.CLUSTERS_DIR / "intent_taxonomy_mapping.csv"


def load_taxonomy(path=TAXONOMY_CSV):
    df = pd.read_csv(path)

    included = df[df["include_in_training"] == True]  # noqa: E712
    excluded = df[df["include_in_training"] == False]  # noqa: E712

    cluster_to_intent = dict(zip(included["cluster_id"], included["intent"]))
    excluded_cluster_ids = set(excluded["cluster_id"])
    intents = sorted(included["intent"].unique().tolist())

    return df, cluster_to_intent, excluded_cluster_ids, intents


TAXONOMY_DF, CLUSTER_TO_INTENT, EXCLUDED_CLUSTER_IDS, INTENTS = load_taxonomy()


def summary():
    print(f"[intent_taxonomy] {len(INTENTS)} intents from {len(CLUSTER_TO_INTENT)} clusters "
          f"({len(EXCLUDED_CLUSTER_IDS)} clusters excluded as non-intents):")
    for intent in INTENTS:
        cluster_ids = [c for c, i in CLUSTER_TO_INTENT.items() if i == intent]
        total_size = TAXONOMY_DF[TAXONOMY_DF["cluster_id"].isin(cluster_ids)]["cluster_size"].sum()
        print(f"    {intent:30s} <- clusters {cluster_ids}  (~{total_size} of 3000 sampled msgs)")
    print("  excluded (not real intents):")
    for _, row in TAXONOMY_DF[TAXONOMY_DF["include_in_training"] == False].iterrows():  # noqa: E712
        print(f"    cluster {row['cluster_id']}: {row['intent']} - {row['notes'][:70]}...")


if __name__ == "__main__":
    summary()
