"""
Verifies the thread-reconstruction logic against a small hand-built fixture
where we know the correct answer by construction. This is the kind of thing
that's easy to get subtly wrong (off-by-one hops, treating a brand-to-brand
reply as a valid pair, dropping dangling ids silently) so it's covered here
rather than only eyeballed once.

Run: pytest tests/ -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_prep  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mini_twcs.csv"


@pytest.fixture
def pairs():
    df = data_prep.load_raw(FIXTURE)
    return data_prep.build_pairs(df, brand="AppleSupport")


def test_valid_pair_count(pairs):
    # 5 AppleSupport tweets in fixture: ids 2, 11, 13, 20, 30
    # - 2: valid (parent 1 is a customer tweet)
    # - 11: valid (parent 10 is a customer tweet)
    # - 13: valid (parent 12 is a customer tweet)
    # - 20: invalid (parent id 999 does not exist)
    # - 30: invalid (parent id 2 is a brand tweet, not a customer)
    assert len(pairs) == 3


def test_drops_dangling_parent(pairs):
    assert 20 not in pairs["brand_tweet_id"].astype(int).values
    assert "@ghost" not in " ".join(pairs["brand_text"])


def test_drops_brand_to_brand_reply(pairs):
    assert 30 not in pairs["brand_tweet_id"].astype(int).values
    assert "cross-posting" not in pairs["brand_text"].values


def test_prior_context_only_when_parent_is_customer(pairs):
    # tweet 12's parent (11) is a brand tweet -> prior_context must be None
    row = pairs[pairs["customer_tweet_id"] == "12"].iloc[0]
    assert row["prior_context"] is None


def test_followup_text_captured(pairs):
    # brand tweet 2's response_tweet_id (3) is a customer tweet -> followup captured
    row = pairs[pairs["brand_tweet_id"] == "2"].iloc[0]
    assert row["followup_text"] == "@AppleSupport that worked, thanks!"


def test_response_seconds_computed_correctly(pairs):
    row = pairs[pairs["brand_tweet_id"] == "2"].iloc[0]
    # 10:00:00 -> 10:05:00 = 300 seconds
    assert row["response_seconds"] == pytest.approx(300.0)
