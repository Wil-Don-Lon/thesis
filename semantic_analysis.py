"""
semantic_analysis.py  —  Semantic text analysis of telephone game captions.

Analyzes how caption language evolves across iterations and epochs, including:
  - Caption drift (absolute: vs iter 1 caption; relative: epoch vs previous epoch)
  - Sentiment trajectory (positive/neutral/negative valence across iterations)
  - Subject matter insertion (new vocabulary that emerges and persists mid-chain)
  - Refusal language detection (hedging, disclaimers, identity evasion)
  - Chunked word frequency by epoch with bubble chart visualization
  - Cross-seed vocabulary divergence (when shared lexicon splits)

Epoch structure (configurable via EPOCH_SIZE):
  Default epochs of 3: [1-3], [4-6], [7-9], [10-12], [13-15]

Outputs (written to --output-dir, default: semantic_output/):
  data/
    captions.csv              — all captions with metadata
    iteration_scores.csv      — per-iteration drift + sentiment scores
    epoch_scores.csv          — aggregated per-epoch scores
    epoch_relative_drift.csv  — epoch-to-epoch (Markov-style) drift
    word_freq_by_epoch.csv    — word frequencies per seed per epoch
    refusal_events.csv        — all detected refusal/hedging events
    insertion_events.csv      — detected subject matter insertions
    vocabulary_divergence.csv — cross-seed vocabulary overlap by epoch
  figures/
    drift_absolute.png        — absolute drift from iter 1 caption per seed
    drift_relative.png        — epoch-to-epoch relative drift per seed
    drift_combined.png        — absolute + relative drift on same axes
    sentiment_trajectory.png  — sentiment valence across iterations per seed
    bubble_charts/            — word frequency bubble chart per seed per epoch
    refusal_timeline.png      — when refusal language appears across seeds
    insertion_heatmap.png     — subject matter insertions by seed and epoch
    vocabulary_divergence.png — cross-seed vocabulary overlap over time

Dependencies:
  pip install sentence-transformers nltk textblob wordcloud matplotlib pandas scipy

Usage:
  python semantic_analysis.py                        # uses ./output
  python semantic_analysis.py --data-dir /path/out
  python semantic_analysis.py --output-dir /path/results
  python semantic_analysis.py --epoch-size 3         # default
"""

import os
import re
import sys
import json
import argparse
import warnings
import string
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")


# ================================
# SEED METADATA
# ================================

SEED_METADATA = {
    "test 1":  {"name": "JFK",               "category": "leader"},
    "test 2":  {"name": "Donald Trump",      "category": "leader"},
    "test 3":  {"name": "Gandhi",            "category": "leader"},
    "test 4":  {"name": "Fidel Castro",      "category": "leader"},
    "test 5":  {"name": "MLK",               "category": "leader"},
    "test 6":  {"name": "Obama",             "category": "leader"},
    "test 7":  {"name": "Bill Clinton",      "category": "leader"},
    "test 8":  {"name": "Queen Elizabeth",   "category": "leader"},
    "test 9":  {"name": "Joseph Stalin",     "category": "leader"},
    "test 10": {"name": "Kim Jong Un",       "category": "leader"},
    "test 11": {"name": "Adolf Hitler",      "category": "leader"},
    "test 12": {"name": "Vladimir Putin",    "category": "leader"},
    "test 13": {"name": "Xi Jinping",        "category": "leader"},
    "test 14": {"name": "George W. Bush",    "category": "leader"},
    "test 15": {"name": "Benjamin Netanyahu","category": "leader"},
}

CATEGORY_COLORS = {
    "leader":   "#2563eb",
    "celebrity":"#16a34a",
    "group":    "#dc2626",
    "unknown":  "#9ca3af",
}

FIGURE_DPI   = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# Stop words to exclude from word frequency analysis
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "his", "his", "her", "their", "him", "them", "we", "us", "our", "i",
    "my", "you", "your", "who", "which", "what", "where", "when", "how",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "as", "if", "than", "then", "there", "here", "into", "onto", "up",
    "out", "about", "over", "after", "before", "between", "through",
    "during", "while", "although", "because", "since", "though", "also",
    "just", "very", "more", "most", "also", "such", "other", "one", "two",
    "well", "back", "even", "still", "way", "toward", "towards", "upon",
}

# Refusal / hedging phrases to detect
REFUSAL_PATTERNS = [
    "i'm not able to",
    "i am not able to",
    "not able to confirm",
    "not able to verify",
    "unable to confirm",
    "cannot be confirmed",
    "cannot be verified",
    "unable to verify",
    "i can't identify",
    "i cannot identify",
    "unable to identify",
    "i'm unable to",
    "i am unable to",
    "cannot determine",
    "can't determine",
    "will not speculate",
    "will not guess",
    "will not assume",
    "i don't know who",
    "i do not know who",
    "it's not possible for me",
    "it is not possible for me",
    "i won't be able",
    "i will not be able",
    "without more context",
    "based on the image alone",
    "an unidentified",
    "unknown person",
    "i will not say",
    "i won't say",
    "i cannot say",
    "i can't say",
]

# Sentiment word lists (simple lexicon-based, no external dependency)
POSITIVE_WORDS = {
    "smiling", "happy", "joyful", "celebrating", "triumphant", "proud",
    "confident", "strong", "powerful", "inspiring", "historic", "dignified",
    "honored", "respected", "cheering", "applause", "victory", "success",
    "peaceful", "serene", "calm", "warm", "bright", "vibrant", "animated",
}
NEGATIVE_WORDS = {
    "angry", "aggressive", "threatening", "violent", "dark", "grim",
    "somber", "tense", "conflict", "war", "battle", "protest", "fire",
    "flames", "smoke", "chaos", "destruction", "burning", "explosion",
    "confrontational", "hostile", "oppressive", "authoritarian", "stern",
    "grave", "solemn", "troubled", "controversial", "disputed", "oppression",
}


# ================================
# MODEL LOADER
# ================================

def load_sentence_model():
    """
    Loads sentence-transformers model for semantic embedding.
    Uses all-MiniLM-L6-v2 — fast, good quality, standard for this kind of work.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nERROR: sentence-transformers is required.")
        print("Install with:  pip install sentence-transformers")
        sys.exit(1)

    print("Loading sentence embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model ready.\n")
    return model


def embed_text(texts: list[str], model) -> np.ndarray:
    """Returns a matrix of sentence embeddings, one row per text."""
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def semantic_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised embeddings."""
    return float(np.dot(a, b))


# ================================
# DATA DISCOVERY
# ================================

def _read_captions_from_dir(
    chain_dir: Path,
    seed_folder: str,
    seed_name: str,
    category: str,
    chain_num: int,
    rows: list,
):
    """Reads all iter_*_caption.txt files from a directory into rows."""
    for caption_file in sorted(chain_dir.glob("iter_*_caption.txt")):
        parts = caption_file.stem.split("_")
        if len(parts) >= 2:
            try:
                iter_num = int(parts[1])
            except ValueError:
                continue
            try:
                caption = caption_file.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except Exception:
                caption = ""
            if caption:
                rows.append({
                    "seed_folder": seed_folder,
                    "seed_name":   seed_name,
                    "category":    category,
                    "chain_num":   chain_num,
                    "iteration":   iter_num,
                    "caption":     caption,
                })


def discover_captions(data_dir: Path) -> pd.DataFrame:
    """
    Walks data_dir and reads all caption .txt files.
    Handles two output formats:

    New format (telephone.py current):
      output/<seed>/objective/chain_01/iter_*_caption.txt

    Old/flat format (telephone.py older):
      output/<seed>/iter_*_caption.txt

    Returns DataFrame with columns:
      seed_folder, seed_name, category, chain_num, iteration, caption
    """
    skip = {"violations", "post_mortem"}
    rows = []

    for seed_dir in sorted(data_dir.iterdir()):
        if not seed_dir.is_dir() or seed_dir.name in skip:
            continue
        meta      = SEED_METADATA.get(seed_dir.name, {})
        seed_name = meta.get("name", seed_dir.name)
        category  = meta.get("category", "unknown")

        # Detect format: if the seed dir contains caption files directly
        # it is the old flat format. Otherwise assume new nested format.
        direct_captions = list(seed_dir.glob("iter_*_caption.txt"))
        if direct_captions:
            # Old flat format — treat as chain 1
            _read_captions_from_dir(
                seed_dir, seed_dir.name, seed_name, category, 1, rows
            )
            continue

        # New format: seed_dir -> prompt_dir -> chain_dir -> captions
        for prompt_dir in sorted(seed_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            for chain_dir in sorted(prompt_dir.iterdir()):
                if not chain_dir.is_dir() or not chain_dir.name.startswith("chain_"):
                    continue
                try:
                    chain_num = int(chain_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                _read_captions_from_dir(
                    chain_dir, seed_dir.name, seed_name, category, chain_num, rows
                )

    if not rows:
        return pd.DataFrame(
            columns=["seed_folder", "seed_name", "category",
                     "chain_num", "iteration", "caption"]
        )

    return pd.DataFrame(rows).sort_values(
        ["seed_folder", "chain_num", "iteration"]
    ).reset_index(drop=True)


# ================================
# EPOCH ASSIGNMENT
# ================================

def assign_epoch(iteration: int, epoch_size: int) -> int:
    """Returns 1-based epoch number for a given iteration."""
    return (iteration - 1) // epoch_size + 1


def epoch_label(epoch_num: int, epoch_size: int) -> str:
    start = (epoch_num - 1) * epoch_size + 1
    end   = epoch_num * epoch_size
    return f"{start}-{end}"


# ================================
# TEXT UTILITIES
# ================================

def tokenize(text: str) -> list[str]:
    """Lowercases, strips punctuation, removes stop words and short tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [
        w for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]


def detect_refusals(caption: str) -> list[str]:
    """Returns list of matched refusal patterns found in the caption."""
    # Normalize smart/curly apostrophes and quotes to straight versions
    # so patterns match regardless of which apostrophe style the model used
    lower = caption.lower()
    lower = lower.replace("’", "'").replace("‘", "'")  # curly single quotes
    lower = lower.replace("“", '"').replace("”", '"')  # curly double quotes
    lower = lower.replace("–", "-").replace("—", "-")  # em/en dashes
    return [p for p in REFUSAL_PATTERNS if p in lower]


def sentiment_score(caption: str) -> tuple[float, str]:
    """
    Simple lexicon-based sentiment scoring.
    Returns (score, label) where score is in [-1, 1] and label is
    'positive', 'negative', or 'neutral'.
    Falls back to TextBlob if installed for better accuracy.
    """
    try:
        from textblob import TextBlob
        polarity = TextBlob(caption).sentiment.polarity
        if polarity > 0.05:
            label = "positive"
        elif polarity < -0.05:
            label = "negative"
        else:
            label = "neutral"
        return polarity, label
    except ImportError:
        pass

    # Fallback: simple word count approach
    words  = set(tokenize(caption))
    pos    = len(words & POSITIVE_WORDS)
    neg    = len(words & NEGATIVE_WORDS)
    total  = pos + neg
    if total == 0:
        return 0.0, "neutral"
    score = (pos - neg) / total
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return score, label


# ================================
# CORE ANALYSIS
# ================================

def compute_iteration_scores(captions_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    For each caption, computes:
      - absolute_drift: semantic distance from the first caption in that chain
      - sentiment_score, sentiment_label
      - refusal_detected, refusal_phrases
      - epoch number
    """
    rows = []
    total_chains = captions_df.groupby(["seed_folder", "chain_num"]).ngroups
    done = 0

    for (seed_folder, chain_num), chain_df in captions_df.groupby(
        ["seed_folder", "chain_num"]
    ):
        done += 1
        print(f"  Computing iteration scores: {done}/{total_chains} chains", end="\r")

        chain_df = chain_df.sort_values("iteration")
        captions = chain_df["caption"].tolist()
        iters    = chain_df["iteration"].tolist()

        if not captions:
            continue

        # Embed all captions in this chain at once (efficient)
        embeddings = embed_text(captions, model)
        anchor_emb = embeddings[0]  # iter 1 caption is the anchor

        for idx, (iter_num, caption, emb) in enumerate(
            zip(iters, captions, embeddings)
        ):
            # Absolute drift vs iter 1
            abs_drift = 1.0 - semantic_similarity(anchor_emb, emb)

            # Relative drift vs previous iteration
            if idx == 0:
                rel_drift = 0.0
            else:
                rel_drift = 1.0 - semantic_similarity(embeddings[idx - 1], emb)

            # Sentiment
            sent_score, sent_label = sentiment_score(caption)

            # Refusals
            refusals = detect_refusals(caption)

            rows.append({
                "seed_folder":     seed_folder,
                "seed_name":       chain_df["seed_name"].iloc[0],
                "category":        chain_df["category"].iloc[0],
                "chain_num":       chain_num,
                "iteration":       iter_num,
                "caption":         caption,
                "absolute_drift":  round(abs_drift, 4),
                "relative_drift":  round(rel_drift, 4),
                "sentiment_score": round(sent_score, 4),
                "sentiment_label": sent_label,
                "refusal_detected": len(refusals) > 0,
                "refusal_phrases": "|".join(refusals),
                "word_count":      len(caption.split()),
            })

    print()
    return pd.DataFrame(rows)


def compute_epoch_scores(iter_df: pd.DataFrame, epoch_size: int) -> pd.DataFrame:
    """
    Aggregates iteration scores into epochs.
    For each seed × epoch:
      - mean absolute drift, relative drift, sentiment
      - refusal rate
      - dominant sentiment label
    """
    iter_df = iter_df.copy()
    iter_df["epoch"] = iter_df["iteration"].apply(
        lambda x: assign_epoch(x, epoch_size)
    )

    rows = []
    for (seed_name, epoch), grp in iter_df.groupby(["seed_name", "epoch"]):
        rows.append({
            "seed_name":           seed_name,
            "category":            grp["category"].iloc[0],
            "epoch":               epoch,
            "epoch_label":         epoch_label(epoch, epoch_size),
            "mean_absolute_drift": round(grp["absolute_drift"].mean(), 4),
            "sd_absolute_drift":   round(grp["absolute_drift"].std(), 4),
            "mean_relative_drift": round(grp["relative_drift"].mean(), 4),
            "sd_relative_drift":   round(grp["relative_drift"].std(), 4),
            "mean_sentiment":      round(grp["sentiment_score"].mean(), 4),
            "refusal_rate":        round(grp["refusal_detected"].mean(), 4),
            "n_captions":          len(grp),
        })

    return pd.DataFrame(rows).sort_values(["seed_name", "epoch"])


def compute_relative_epoch_drift(
    iter_df: pd.DataFrame, epoch_size: int, model
) -> pd.DataFrame:
    """
    Computes epoch-to-epoch (Markov-style) semantic drift.
    For each seed, embeds the mean of all captions in epoch N and epoch N+1
    and measures the cosine distance between them.
    """
    iter_df = iter_df.copy()
    iter_df["epoch"] = iter_df["iteration"].apply(
        lambda x: assign_epoch(x, epoch_size)
    )

    rows = []
    total = iter_df["seed_name"].nunique()
    done  = 0

    for seed_name, seed_df in iter_df.groupby("seed_name"):
        done += 1
        print(f"  Computing epoch transitions: {done}/{total} seeds", end="\r")

        epochs = sorted(seed_df["epoch"].unique())
        # Build mean embedding per epoch (across all chains for that seed)
        epoch_embeddings: dict[int, np.ndarray] = {}
        for ep in epochs:
            ep_captions = seed_df[seed_df["epoch"] == ep]["caption"].tolist()
            if ep_captions:
                embs = embed_text(ep_captions, model)
                epoch_embeddings[ep] = embs.mean(axis=0)

        # Epoch N vs N+1 transition drift
        for i in range(len(epochs) - 1):
            ep_a = epochs[i]
            ep_b = epochs[i + 1]
            if ep_a in epoch_embeddings and ep_b in epoch_embeddings:
                drift = 1.0 - semantic_similarity(
                    epoch_embeddings[ep_a], epoch_embeddings[ep_b]
                )
                rows.append({
                    "seed_name":    seed_name,
                    "category":     seed_df["category"].iloc[0],
                    "epoch_from":   ep_a,
                    "epoch_to":     ep_b,
                    "transition":   f"{epoch_label(ep_a, epoch_size)}→{epoch_label(ep_b, epoch_size)}",
                    "epoch_drift":  round(drift, 4),
                })

    print()
    return pd.DataFrame(rows)


def compute_word_frequencies(
    captions_df: pd.DataFrame, epoch_size: int
) -> pd.DataFrame:
    """
    For each seed × epoch, computes word frequencies across all chains.
    Returns long-form DataFrame: seed_name, epoch, word, count, freq
    """
    captions_df = captions_df.copy()
    captions_df["epoch"] = captions_df["iteration"].apply(
        lambda x: assign_epoch(x, epoch_size)
    )

    rows = []
    for (seed_name, epoch), grp in captions_df.groupby(["seed_name", "epoch"]):
        all_words = []
        for caption in grp["caption"]:
            all_words.extend(tokenize(caption))

        total = len(all_words)
        for word, count in Counter(all_words).most_common(50):
            rows.append({
                "seed_name": seed_name,
                "category":  grp["category"].iloc[0],
                "epoch":     epoch,
                "word":      word,
                "count":     count,
                "freq":      round(count / total, 4) if total > 0 else 0,
            })

    return pd.DataFrame(rows)


def detect_insertions(
    word_freq_df: pd.DataFrame, min_freq: float = 0.02, min_epoch: int = 2
) -> pd.DataFrame:
    """
    Detects subject matter insertions: words that appear in epoch N+ with
    meaningful frequency but were absent or rare in epoch 1.

    A word is flagged as an insertion if:
      - It appears in epoch >= min_epoch
      - Its frequency in that epoch >= min_freq
      - Its frequency in epoch 1 was less than half of min_freq
    """
    rows = []
    for seed_name, seed_df in word_freq_df.groupby("seed_name"):
        epoch1 = seed_df[seed_df["epoch"] == 1].set_index("word")["freq"].to_dict()

        for _, row in seed_df[seed_df["epoch"] >= min_epoch].iterrows():
            word      = row["word"]
            freq_now  = row["freq"]
            freq_e1   = epoch1.get(word, 0.0)

            if freq_now >= min_freq and freq_e1 < min_freq / 2:
                rows.append({
                    "seed_name":       seed_name,
                    "category":        row["category"],
                    "word":            word,
                    "epoch_first_seen": row["epoch"],
                    "freq_epoch1":     round(freq_e1, 4),
                    "freq_at_insertion": round(freq_now, 4),
                })

    if not rows:
        return pd.DataFrame(columns=["seed_name", "category", "word",
                                      "epoch_first_seen", "freq_epoch1",
                                      "freq_at_insertion"])
    return pd.DataFrame(rows).drop_duplicates(
        subset=["seed_name", "word"]
    ).sort_values(["seed_name", "epoch_first_seen", "freq_at_insertion"],
                  ascending=[True, True, False])


def compute_vocabulary_divergence(
    word_freq_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measures cross-seed vocabulary overlap per epoch.
    For each epoch, computes pairwise Jaccard similarity between seeds'
    top-N word sets, then reports the mean overlap.

    Low overlap = seeds have diverged into distinct vocabularies.
    High overlap = seeds share a generic common lexicon.
    """
    rows = []
    top_n = 20

    for epoch, ep_df in word_freq_df.groupby("epoch"):
        seed_wordsets: dict[str, set] = {}
        for seed_name, sdf in ep_df.groupby("seed_name"):
            top_words = set(sdf.nlargest(top_n, "count")["word"].tolist())
            seed_wordsets[seed_name] = top_words

        seeds = list(seed_wordsets.keys())
        if len(seeds) < 2:
            continue

        # Pairwise Jaccard
        similarities = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a = seed_wordsets[seeds[i]]
                b = seed_wordsets[seeds[j]]
                if not a and not b:
                    continue
                jaccard = len(a & b) / len(a | b)
                similarities.append(jaccard)

        # Also compute shared words across ALL seeds
        all_words = [seed_wordsets[s] for s in seeds]
        universal = set.intersection(*all_words) if all_words else set()

        rows.append({
            "epoch":              epoch,
            "mean_pairwise_jaccard": round(np.mean(similarities), 4),
            "sd_pairwise_jaccard":   round(np.std(similarities), 4),
            "n_universal_words":  len(universal),
            "universal_words":    ", ".join(sorted(universal)),
        })

    return pd.DataFrame(rows)


# ================================
# FIGURES
# ================================

def fig_drift_combined(
    iter_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    epoch_size: int,
    out_path: Path,
):
    """
    Two-panel figure per seed showing absolute drift (line) and
    relative/step drift (bars) across iterations, with epoch boundaries marked.
    All seeds on one figure, faceted by seed.
    """
    seeds      = sorted(iter_df["seed_name"].unique())
    n_seeds    = len(seeds)
    n_cols     = 3
    n_rows     = (n_seeds + n_cols - 1) // n_cols
    max_iter   = iter_df["iteration"].max()

    plt.style.use(FIGURE_STYLE)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharex=True, sharey=False,
    )
    axes_flat = axes.flatten() if n_seeds > 1 else [axes]

    for ax, seed_name in zip(axes_flat, seeds):
        sdf   = iter_df[iter_df["seed_name"] == seed_name]
        cat   = sdf["category"].iloc[0]
        color = CATEGORY_COLORS.get(cat, "#555")

        # Mean absolute drift per iteration (across chains)
        abs_mean = sdf.groupby("iteration")["absolute_drift"].mean()
        rel_mean = sdf.groupby("iteration")["relative_drift"].mean()

        # Bar chart for relative drift
        ax.bar(rel_mean.index, rel_mean.values,
               color=color, alpha=0.3, width=0.8, label="step drift")

        # Line for absolute drift
        ax2 = ax.twinx()
        ax2.plot(abs_mean.index, abs_mean.values,
                 color=color, linewidth=2.0, label="absolute drift", zorder=3)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Absolute drift", fontsize=7, color=color)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=7)

        # Epoch boundary lines
        for ep_start in range(1, max_iter + 1, epoch_size):
            if ep_start > 1:
                ax.axvline(ep_start - 0.5, color="gray",
                           linestyle=":", linewidth=0.7, alpha=0.6)

        ax.set_title(seed_name, fontsize=10, fontweight="bold")
        ax.set_xlim(0.5, max_iter + 0.5)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Step drift", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for ax in axes_flat[n_seeds:]:
        ax.set_visible(False)

    fig.suptitle(
        "Caption Semantic Drift: Absolute (line) vs Step/Relative (bars)\n"
        "Vertical lines = epoch boundaries",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_epoch_relative_drift(
    rel_epoch_df: pd.DataFrame, epoch_size: int, out_path: Path
):
    """
    Heatmap: seeds × epoch transitions, color = epoch-to-epoch drift magnitude.
    High values = big semantic jump between those epochs.
    """
    pivot = rel_epoch_df.pivot(
        index="seed_name", columns="transition", values="epoch_drift"
    )

    # Sort columns chronologically by epoch_from, not alphabetically
    col_order = (
        rel_epoch_df.drop_duplicates("transition")
        .sort_values("epoch_from")["transition"]
        .tolist()
    )
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    # Sort rows by mean drift descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.5), max(5, len(pivot) * 0.45))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=0.5, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        "Epoch-to-Epoch Semantic Drift (Markov-style)\n"
        "High value = large vocabulary/semantic shift between adjacent epochs",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Cosine Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_sentiment_trajectory(iter_df: pd.DataFrame, out_path: Path):
    """
    Line plot of mean sentiment score per iteration per seed.
    Color-coded by category.
    """
    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))
    max_iter = iter_df["iteration"].max()

    for seed_name, sdf in iter_df.groupby("seed_name"):
        cat   = sdf["category"].iloc[0]
        color = CATEGORY_COLORS.get(cat, "#9ca3af")
        mean  = sdf.groupby("iteration")["sentiment_score"].mean()
        ax.plot(mean.index, mean.values, color=color,
                linewidth=1.5, alpha=0.8)
        ax.text(mean.index.max() + 0.15, mean.iloc[-1],
                seed_name, fontsize=7, va="center", color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean Sentiment Score", fontsize=12)
    ax.set_title("Caption Sentiment Trajectory Across Iterations",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, max_iter + 3)

    handles = [
        plt.Line2D([0], [0], color=CATEGORY_COLORS[c],
                   linewidth=2, label=c.capitalize())
        for c in CATEGORY_COLORS if c in iter_df["category"].values
    ]
    ax.legend(handles=handles, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_refusal_timeline(iter_df: pd.DataFrame, out_path: Path):
    """
    Heatmap: seeds × iterations, color = refusal detected (binary).
    Shows exactly when and where the caption model starts hedging.
    """
    pivot = (
        iter_df.groupby(["seed_name", "iteration"])["refusal_detected"]
        .max()
        .unstack(level="iteration")
        .fillna(0)
    )
    # Sort by first refusal iteration
    first_refusal = pivot.idxmax(axis=1)
    pivot = pivot.loc[first_refusal.sort_values().index]

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.7), max(5, len(pivot) * 0.45))
    )
    im = ax.imshow(pivot.values.astype(float), aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_title(
        "Refusal/Hedging Language Detection\n"
        "Red = refusal phrase detected in caption",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Refusal Detected",
                 ticks=[0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_insertion_heatmap(insertion_df: pd.DataFrame, out_path: Path):
    """
    Bar chart of detected insertions per seed, labelled with the inserted words.
    """
    if insertion_df.empty:
        print("  Skipping insertion heatmap (no insertions detected).")
        return

    counts = (
        insertion_df.groupby("seed_name")
        .size()
        .reset_index(name="n_insertions")
        .sort_values("n_insertions", ascending=False)
    )

    # Get top inserted words per seed for labels
    top_words = (
        insertion_df.sort_values("freq_at_insertion", ascending=False)
        .groupby("seed_name")["word"]
        .apply(lambda x: ", ".join(x.head(5)))
        .reset_index()
    )
    counts = counts.merge(top_words, on="seed_name")

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.5)))

    bars = ax.barh(counts["seed_name"], counts["n_insertions"],
                   color="#2563eb", alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Inserted Terms", fontsize=11)
    ax.set_title(
        "Subject Matter Insertions per Seed\n"
        "(Terms appearing in later epochs absent from epoch 1)",
        fontsize=12, fontweight="bold",
    )

    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            row["word"],
            va="center", fontsize=7, color="#555",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_vocabulary_divergence(vocab_div_df: pd.DataFrame,
                               epoch_size: int, out_path: Path):
    """
    Line plot of mean pairwise Jaccard similarity across epochs.
    Falling line = seeds are diverging into distinct vocabularies.
    Rising line = seeds converging to generic shared language.
    """
    plt.style.use(FIGURE_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [
        epoch_label(int(e), epoch_size)
        for e in vocab_div_df["epoch"]
    ]

    ax1.plot(vocab_div_df["epoch"], vocab_div_df["mean_pairwise_jaccard"],
             color="#2563eb", linewidth=2.5, marker="o")
    ax1.fill_between(
        vocab_div_df["epoch"],
        vocab_div_df["mean_pairwise_jaccard"] - vocab_div_df["sd_pairwise_jaccard"],
        vocab_div_df["mean_pairwise_jaccard"] + vocab_div_df["sd_pairwise_jaccard"],
        color="#2563eb", alpha=0.15,
    )
    ax1.set_xticks(vocab_div_df["epoch"])
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Pairwise Jaccard Similarity", fontsize=11)
    ax1.set_title(
        "Cross-Seed Vocabulary Overlap by Epoch\n"
        "Falling = seeds diverging; Rising = converging to generic language",
        fontsize=11, fontweight="bold",
    )
    ax1.set_ylim(0, 1)

    # Right panel: show the universal words as text per epoch since count is
    # always 0 (no single word appears in every seed's top-20). Instead show
    # the SD of pairwise Jaccard — spread tells us how much seeds vary from
    # each other, not just the mean.
    ax2.bar(vocab_div_df["epoch"], vocab_div_df["sd_pairwise_jaccard"],
            color="#16a34a", alpha=0.7)
    ax2.set_xticks(vocab_div_df["epoch"])
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("SD of Pairwise Jaccard Similarity", fontsize=11)
    ax2.set_title(
        "Vocabulary Overlap Variance by Epoch\n"
        "High SD = some seed pairs share vocabulary, others do not",
        fontsize=11, fontweight="bold",
    )
    ax2.set_ylim(0, max(0.5, vocab_div_df["sd_pairwise_jaccard"].max() * 1.2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_bubble_charts(word_freq_df: pd.DataFrame,
                       epoch_size: int, out_dir: Path):
    """
    For each seed, generates a grid of bubble charts (one per epoch)
    showing top words by frequency.
    Bubble size and color intensity = word frequency.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds  = sorted(word_freq_df["seed_name"].unique())
    epochs = sorted(word_freq_df["epoch"].unique())
    top_n  = 15

    for seed_name in seeds:
        seed_df = word_freq_df[word_freq_df["seed_name"] == seed_name]
        n_epochs = len(epochs)
        fig, axes = plt.subplots(
            1, n_epochs,
            figsize=(4 * n_epochs, 5),
        )
        if n_epochs == 1:
            axes = [axes]

        for ax, epoch in zip(axes, epochs):
            ep_df = seed_df[seed_df["epoch"] == epoch].nlargest(top_n, "count")
            if ep_df.empty:
                ax.set_visible(False)
                continue

            words  = ep_df["word"].tolist()
            counts = ep_df["count"].tolist()
            max_c  = max(counts) if counts else 1

            # Bubble chart: random layout in unit square
            np.random.seed(42)
            x = np.random.uniform(0.1, 0.9, len(words))
            y = np.random.uniform(0.1, 0.9, len(words))
            sizes  = [300 * (c / max_c) + 30 for c in counts]
            alphas = [0.4 + 0.6 * (c / max_c) for c in counts]

            for xi, yi, word, size, alpha in zip(x, y, words, sizes, alphas):
                ax.scatter(xi, yi, s=size, color="#2563eb", alpha=alpha, zorder=2)
                ax.text(xi, yi, word, ha="center", va="center",
                        fontsize=max(5, 9 - len(word) // 3),
                        fontweight="bold", color="white", zorder=3)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Epoch {epoch_label(epoch, epoch_size)}",
                         fontsize=10, fontweight="bold")
            ax.axis("off")

        fig.suptitle(f"{seed_name} — Word Frequency by Epoch",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe_name = seed_name.replace(" ", "_").replace(".", "")
        out_path  = out_dir / f"bubble_{safe_name}.png"
        plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()

    print(f"  Saved: bubble charts in {out_dir.name}/")


# ================================
# STATISTICAL SUMMARY
# ================================

def run_stats(
    iter_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    rel_epoch_df: pd.DataFrame,
    insertion_df: pd.DataFrame,
) -> str:
    lines = ["=" * 60, "SEMANTIC ANALYSIS STATISTICAL SUMMARY", "=" * 60, ""]

    # Overall absolute drift
    valid = iter_df[["iteration", "absolute_drift"]].dropna()
    r, p  = stats.pearsonr(valid["iteration"], valid["absolute_drift"])
    lines.append(f"Overall caption drift (Pearson r vs iteration): r={r:.3f}, p={p:.2e}")
    lines.append("")

    # Per-seed decay rate
    lines.append("Absolute drift slope per seed (linear regression):")
    for seed_name, sdf in iter_df.groupby("seed_name"):
        v = sdf[["iteration", "absolute_drift"]].dropna()
        if len(v) < 4 or v["iteration"].nunique() < 2:
            lines.append(f"  {seed_name:24s}  insufficient data")
            continue
        slope, _, r_val, p2, _ = stats.linregress(
            v["iteration"], v["absolute_drift"]
        )
        lines.append(
            f"  {seed_name:24s}  slope={slope:.4f}/iter  "
            f"r²={r_val**2:.3f}  p={p2:.2e}"
        )
    lines.append("")

    # Largest epoch-to-epoch transitions
    if not rel_epoch_df.empty and "seed_name" in rel_epoch_df.columns:
        lines.append("Largest epoch-to-epoch transitions:")
        top = rel_epoch_df.nlargest(10, "epoch_drift")
        for _, row in top.iterrows():
            lines.append(
                f"  {row['seed_name']:24s}  {row['transition']:20s}  "
                f"drift={row['epoch_drift']:.4f}"
            )
        lines.append("")

    # Refusal summary
    refusals = iter_df[iter_df["refusal_detected"]]
    lines.append(f"Refusal language summary:")
    lines.append(f"  Total refusal events:  {len(refusals)}")
    lines.append(f"  Seeds affected:        {refusals['seed_name'].nunique()}")
    if len(refusals):
        lines.append(f"  Mean iteration:        {refusals['iteration'].mean():.1f}")
        lines.append(f"  Earliest:              iter {refusals['iteration'].min()}")
        lines.append(f"  Latest:                iter {refusals['iteration'].max()}")
        lines.append("")
        lines.append("  Refusals per seed:")
        for seed, grp in refusals.groupby("seed_name"):
            lines.append(
                f"    {seed:24s}  {len(grp):3d} events  "
                f"(first at iter {grp['iteration'].min()})"
            )
    lines.append("")

    # Insertion summary
    if not insertion_df.empty:
        lines.append(f"Subject matter insertions:")
        lines.append(f"  Total inserted terms:  {len(insertion_df)}")
        lines.append(f"  Seeds with insertions: {insertion_df['seed_name'].nunique()}")
        lines.append("")
        lines.append("  Top insertions by seed:")
        for seed, grp in insertion_df.groupby("seed_name"):
            top5 = grp.nlargest(5, "freq_at_insertion")["word"].tolist()
            lines.append(f"    {seed:24s}  {', '.join(top5)}")
    lines.append("")

    # Sentiment
    lines.append("Sentiment summary:")
    for seed_name, sdf in iter_df.groupby("seed_name"):
        mean_sent = sdf["sentiment_score"].mean()
        trend_dir = "↑" if sdf.groupby("iteration")["sentiment_score"].mean().iloc[-1] > \
                           sdf.groupby("iteration")["sentiment_score"].mean().iloc[0] else "↓"
        lines.append(
            f"  {seed_name:24s}  mean={mean_sent:+.3f}  trend={trend_dir}"
        )

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic analysis of telephone game captions."
    )
    parser.add_argument("--data-dir",    default="output",
                        help="Path to telephone.py output dir (default: output)")
    parser.add_argument("--output-dir",  default="semantic_output",
                        help="Where to write results (default: semantic_output)")
    parser.add_argument("--epoch-size",  type=int, default=3,
                        help="Iterations per epoch (default: 3)")
    parser.add_argument("--captions-csv", default=None,
                        help="Load pre-extracted captions CSV to skip file discovery")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    out_dir    = Path(args.output_dir)
    data_out   = out_dir / "data"
    figs_out   = out_dir / "figures"
    data_out.mkdir(parents=True, exist_ok=True)
    figs_out.mkdir(parents=True, exist_ok=True)

    epoch_size = args.epoch_size

    print(f"\nTelephone Game — Semantic Analysis")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Epoch size: {epoch_size} iterations\n")

    # Load captions
    if args.captions_csv:
        print(f"Loading captions from {args.captions_csv}...")
        captions_df = pd.read_csv(args.captions_csv)
    else:
        print("Discovering captions...")
        captions_df = discover_captions(data_dir)
        captions_df.to_csv(data_out / "captions.csv", index=False)
        print(f"  Found {len(captions_df)} captions across "
              f"{captions_df['seed_name'].nunique()} seeds\n")

    if captions_df.empty:
        print("ERROR: No captions found. Check --data-dir path.")
        sys.exit(1)

    # Load sentence model
    model = load_sentence_model()

    # Iteration-level scores
    print("Computing iteration-level scores...")
    iter_df = compute_iteration_scores(captions_df, model)
    iter_df.to_csv(data_out / "iteration_scores.csv", index=False)
    print(f"  Saved: iteration_scores.csv\n")

    # Epoch aggregation
    print("Aggregating into epochs...")
    epoch_df = compute_epoch_scores(iter_df, epoch_size)
    epoch_df.to_csv(data_out / "epoch_scores.csv", index=False)
    print(f"  Saved: epoch_scores.csv")

    # Epoch-to-epoch (Markov) drift
    print("Computing epoch transition drift...")
    rel_epoch_df = compute_relative_epoch_drift(iter_df, epoch_size, model)
    rel_epoch_df.to_csv(data_out / "epoch_relative_drift.csv", index=False)
    print(f"  Saved: epoch_relative_drift.csv\n")

    # Word frequencies
    print("Computing word frequencies...")
    word_freq_df = compute_word_frequencies(captions_df, epoch_size)
    word_freq_df.to_csv(data_out / "word_freq_by_epoch.csv", index=False)
    print(f"  Saved: word_freq_by_epoch.csv")

    # Insertions
    print("Detecting subject matter insertions...")
    insertion_df = detect_insertions(word_freq_df)
    insertion_df.to_csv(data_out / "insertion_events.csv", index=False)
    print(f"  Found {len(insertion_df)} insertion events")
    print(f"  Saved: insertion_events.csv")

    # Vocabulary divergence
    print("Computing cross-seed vocabulary divergence...")
    vocab_div_df = compute_vocabulary_divergence(word_freq_df)
    vocab_div_df.to_csv(data_out / "vocabulary_divergence.csv", index=False)
    print(f"  Saved: vocabulary_divergence.csv\n")

    # Refusal events
    refusal_df = iter_df[iter_df["refusal_detected"]].copy()
    refusal_df.to_csv(data_out / "refusal_events.csv", index=False)
    print(f"  Found {len(refusal_df)} refusal events")
    print(f"  Saved: refusal_events.csv\n")

    # Figures
    print("Generating figures...")
    fig_drift_combined(iter_df, epoch_df, epoch_size,
                       figs_out / "drift_combined.png")
    if not rel_epoch_df.empty and "seed_name" in rel_epoch_df.columns:
        fig_epoch_relative_drift(rel_epoch_df, epoch_size,
                                  figs_out / "drift_epoch_transitions.png")
    else:
        print("  Skipping epoch transitions (insufficient iterations for epoch analysis)")
    fig_sentiment_trajectory(iter_df, figs_out / "sentiment_trajectory.png")
    fig_refusal_timeline(iter_df, figs_out / "refusal_timeline.png")
    fig_insertion_heatmap(insertion_df, figs_out / "insertion_heatmap.png")
    fig_vocabulary_divergence(vocab_div_df, epoch_size,
                               figs_out / "vocabulary_divergence.png")
    fig_bubble_charts(word_freq_df, epoch_size, figs_out / "bubble_charts")

    # Stats
    print("\nRunning statistical tests...")
    stats_text = run_stats(iter_df, epoch_df, rel_epoch_df, insertion_df)
    print(stats_text)
    (data_out / "statistical_summary.txt").write_text(
        stats_text, encoding="utf-8"
    )
    print(f"  Saved: statistical_summary.txt")

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()