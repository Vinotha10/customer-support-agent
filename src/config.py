"""
Single place to change the brand / paths / sampling knobs.
Everything else in src/ imports from here so the pipeline stays consistent
end to end (data_prep -> eda -> cluster_intents -> later: classifier, rag).
"""
from pathlib import Path

# ---- Brand selection -------------------------------------------------
# Pick ONE brand handle exactly as it appears in author_id for company tweets.
# Good candidates (high volume, pattern-heavy replies): AppleSupport, AmazonHelp
BRAND = "AppleSupport"

# ---- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_PARQUET = ROOT / "data" / "processed" / f"{BRAND.lower()}_pairs.parquet"
PROCESSED_CSV = ROOT / "data" / "processed" / f"{BRAND.lower()}_pairs.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
CLUSTERS_DIR = ROOT / "outputs" / "clusters"
GOLDEN_SET_DIR = ROOT / "outputs" / "golden_set"
GOLDEN_SET_LABELED = GOLDEN_SET_DIR / "golden_set_labeled.csv"
EVAL_DIR = ROOT / "outputs" / "eval"

# ---- Sampling knobs (kept small deliberately so the whole pipeline
# reproduces in well under 15 minutes on a laptop CPU) -------------------
MAX_PAIRS_FOR_CLUSTERING = 3000   # subsample for taxonomy discovery
RANDOM_SEED = 42
