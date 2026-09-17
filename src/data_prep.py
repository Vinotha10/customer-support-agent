"""
Reconstructs (customer_message -> brand_reply) pairs for one brand from the
raw Kaggle "Customer Support on Twitter" file (twcs.csv).

Raw schema (Kaggle):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

inbound == True  -> tweet sent BY a customer TO a company
inbound == False -> tweet sent BY a company (author_id == brand handle)

Design decisions (see decision log in README for the "why"):
  1. A valid pair requires: brand tweet whose in_response_to_tweet_id points
     at a tweet that is itself inbound==True (i.e. an actual customer, not
     another company tweet or a dangling id).
  2. We keep ONE customer->brand pair per brand tweet_id (brand tweets are
     the anchor), not per customer tweet, because a customer can send
     multiple tweets before getting one reply — the brand reply is the
     unit we're grounding on for reply drafting.
  3. We also pull:
       - prior_context: the customer's own previous tweet in the same
         thread, if any (helps with multi-turn cases), by walking the
         customer tweet's in_response_to_tweet_id.
       - followup_text: the customer's reply back to the brand, if any
         (response_tweet_id of the brand tweet) — used later as a weak
         resolution signal ("thanks!" vs. re-complaint), NOT as ground
         truth of a successful resolution. That distinction goes in the
         report's "what's misleading" section.
  4. Retweets/quote text and @mentions are left in `text` as-is (raw) —
     cleaning happens downstream per-task so we never destroy information
     we might need later (e.g. order numbers redacted with regex only
     when building prompts, not at ingestion).
"""
import pandas as pd
from tqdm import tqdm

import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    dtype = {
        "tweet_id": str,
        "author_id": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }
    df = pd.read_csv(path, dtype=dtype)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    # Twitter's raw export format is fixed, e.g. "Tue Oct 31 22:10:47 +0000 2017".
    # Passing format= makes this vectorized (seconds) instead of falling back to
    # dateutil's per-row parser (minutes+ on the full ~3M-row file, and looks
    # like the script has hung when it's really just parsing dates one at a time).
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    n_unparsed = df["created_at"].isna().sum()
    if n_unparsed:
        print(f"[data_prep] warning: {n_unparsed} created_at values didn't match "
              f"the expected format and were set to NaT (response_seconds will be "
              f"null for those rows, pairs are still kept)")
    return df


def build_pairs(df: pd.DataFrame, brand: str = config.BRAND) -> pd.DataFrame:
    df = df.set_index("tweet_id", drop=False)

    brand_tweets = df[(df["inbound"] == False) & (df["author_id"] == brand)]  # noqa: E712
    print(f"[data_prep] brand tweets for {brand}: {len(brand_tweets)}")

    records = []
    missing_parent = 0
    parent_not_customer = 0

    for _, brow in tqdm(brand_tweets.iterrows(), total=len(brand_tweets), desc="reconstructing pairs"):
        parent_id = brow["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in df.index:
            missing_parent += 1
            continue

        crow = df.loc[parent_id]
        # .loc can return a DataFrame if tweet_id isn't unique; guard for that
        if isinstance(crow, pd.DataFrame):
            crow = crow.iloc[0]

        if not crow["inbound"]:
            parent_not_customer += 1
            continue

        # prior customer context (one hop back), if it exists
        prior_context = None
        cparent_id = crow["in_response_to_tweet_id"]
        if pd.notna(cparent_id) and cparent_id in df.index:
            prow = df.loc[cparent_id]
            if isinstance(prow, pd.DataFrame):
                prow = prow.iloc[0]
            if prow["inbound"]:
                prior_context = prow["text"]

        # customer follow-up after the brand reply (weak resolution signal)
        followup_text = None
        followup_id = brow["response_tweet_id"]
        if pd.notna(followup_id):
            first_followup_id = str(followup_id).split(",")[0].strip()
            if first_followup_id in df.index:
                frow = df.loc[first_followup_id]
                if isinstance(frow, pd.DataFrame):
                    frow = frow.iloc[0]
                if frow["inbound"]:
                    followup_text = frow["text"]

        response_seconds = None
        if pd.notna(crow["created_at"]) and pd.notna(brow["created_at"]):
            response_seconds = (brow["created_at"] - crow["created_at"]).total_seconds()

        records.append({
            "customer_tweet_id": crow["tweet_id"],
            "customer_text": crow["text"],
            "customer_created_at": crow["created_at"],
            "prior_context": prior_context,
            "brand_tweet_id": brow["tweet_id"],
            "brand_text": brow["text"],
            "brand_created_at": brow["created_at"],
            "response_seconds": response_seconds,
            "followup_text": followup_text,
        })

    pairs = pd.DataFrame.from_records(records)
    print(f"[data_prep] valid pairs: {len(pairs)}")
    print(f"[data_prep] dropped (no/unresolved parent id): {missing_parent}")
    print(f"[data_prep] dropped (parent wasn't a customer tweet): {parent_not_customer}")
    return pairs


def main():
    df = load_raw()
    pairs = build_pairs(df)
    config.PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(config.PROCESSED_PARQUET, index=False)
    pairs.to_csv(config.PROCESSED_CSV, index=False)
    print(f"[data_prep] wrote {config.PROCESSED_PARQUET}")
    print(f"[data_prep] wrote {config.PROCESSED_CSV}")


if __name__ == "__main__":
    main()
