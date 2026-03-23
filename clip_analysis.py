"""
clip_analysis.py  —  CLIP similarity analysis for the telephone game dataset.

Computes cosine similarity between each generated image and its chain's seed image,
then produces all tables and figures needed for the thesis Results section.

Outputs (written to --output-dir, default: clip_analysis_output/):
  data/
    clip_scores.csv           — raw per-chain per-iteration CLIP scores
    decay_summary.csv         — mean +/- SD by category and iteration
    preservation_table.csv    — per-seed CLIP at iters 1, 5, 10, 15 + block counts
    block_table.csv           — per-chain block characteristics
    statistical_summary.txt   — key stats for thesis
  figures/
    decay_curves.png          — mean CLIP decay faceted by category
    individual_heatmap.png    — per-seed mean CLIP score heatmap
    individual_decay.png      — all seed trajectories on one axes
    block_distribution.png    — block counts and first-block iteration per seed
    category_boxplot.png      — CLIP distribution by category at key checkpoints

Dependencies:
  pip install open_clip_torch torch torchvision pillow matplotlib pandas scipy

Usage:
  python clip_analysis.py                           # uses ./output and ./master_log.json
  python clip_analysis.py --data-dir /path/output   # custom telephone.py output dir
  python clip_analysis.py --log /path/master_log.json
  python clip_analysis.py --output-dir /path/out    # custom results dir
  python clip_analysis.py --no-clip                 # skip CLIP, block stats only
  python clip_analysis.py --scores-csv data/clip_scores.csv  # reuse saved scores
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")


# ================================
# SEED METADATA
# ================================
# Maps folder name (as it appears in output/) -> display name + category.
# Category must be one of: "leader", "celebrity", "group"
#
# To use on a different dataset: update this dict.
# Any seed folder not listed here is labelled by its folder name as "unknown".

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


# ================================
# CLIP LOADER
# ================================

def load_clip_model(device=None):
    """
    Loads OpenCLIP ViT-B-32 (LAION-2B weights).
    Falls back to ViT-B-16 if the preferred checkpoint is unavailable.
    Returns (model, preprocess, device).
    """
    try:
        import open_clip
        import torch
    except ImportError:
        print("\nERROR: open_clip_torch and torch are required for CLIP scoring.")
        print("Install with:  pip install open_clip_torch torch torchvision")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model on {device}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
    except Exception:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
    model = model.to(device).eval()
    print("CLIP ready.\n")
    return model, preprocess, device


def embed_image(image_path: Path, model, preprocess, device) -> np.ndarray:
    """Returns a unit-normalised CLIP embedding for a single image."""
    import torch
    from PIL import Image as PILImage

    img    = PILImage.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ================================
# DATA DISCOVERY
# ================================

def discover_chains(data_dir: Path) -> list[dict]:
    """
    Walks data_dir and returns one dict per chain:
      seed_folder, seed_name, category, prompt_type, chain_num,
      chain_dir, seed_image, generated_images [(iter_num, path), ...]

    Skips violations/ and post_mortem/ subdirectories.
    """
    skip   = {"violations", "post_mortem"}
    chains = []

    for seed_dir in sorted(data_dir.iterdir()):
        if not seed_dir.is_dir() or seed_dir.name in skip:
            continue
        meta      = SEED_METADATA.get(seed_dir.name, {})
        seed_name = meta.get("name", seed_dir.name)
        category  = meta.get("category", "unknown")

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

                seed_img = chain_dir / "iter_00_seed.jpg"
                if not seed_img.exists():
                    continue

                generated = []
                for f in sorted(chain_dir.glob("iter_*_generated.jpg")):
                    parts = f.stem.split("_")
                    if len(parts) >= 2:
                        try:
                            n = int(parts[1])
                            if f.stat().st_size > 0:
                                generated.append((n, f))
                        except ValueError:
                            continue

                chains.append({
                    "seed_folder":      seed_dir.name,
                    "seed_name":        seed_name,
                    "category":         category,
                    "prompt_type":      prompt_dir.name,
                    "chain_num":        chain_num,
                    "chain_dir":        chain_dir,
                    "seed_image":       seed_img,
                    "generated_images": generated,
                })

    return chains


# ================================
# CLIP SCORING
# ================================

def compute_clip_scores(
    chains: list[dict],
    model,
    preprocess,
    device,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """
    Computes cosine similarity of each generated image against its seed.
    Returns DataFrame: seed_folder, seed_name, category, prompt_type,
                       chain_num, iteration, clip_score

    Saves an incremental checkpoint every `checkpoint_every` images so that
    if the run is interrupted you can resume with --scores-csv and it will
    pick up where it left off rather than starting from scratch.
    """
    # Load any previously checkpointed rows to skip already-scored images
    done_keys: set[tuple] = set()
    rows: list[dict] = []

    if checkpoint_path and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        rows = existing.to_dict("records")
        for r in rows:
            done_keys.add((r["seed_folder"], r["chain_num"], r["iteration"]))
        print(f"  Resuming from checkpoint: {len(rows)} images already scored.")

    total   = sum(len(c["generated_images"]) for c in chains)
    scored  = len(rows)
    skipped = scored

    for chain in chains:
        chain_keys = {
            (chain["seed_folder"], chain["chain_num"], it)
            for it, _ in chain["generated_images"]
        }
        if chain_keys and chain_keys.issubset(done_keys):
            continue

        seed_emb = embed_image(chain["seed_image"], model, preprocess, device)

        for iter_num, img_path in chain["generated_images"]:
            key = (chain["seed_folder"], chain["chain_num"], iter_num)
            if key in done_keys:
                continue

            scored += 1
            print(f"  Scoring: {scored}/{total}  ({skipped} resumed from checkpoint)", end="\r")

            try:
                gen_emb = embed_image(img_path, model, preprocess, device)
                score   = cosine_sim(seed_emb, gen_emb)
            except Exception as e:
                print(f"\n  WARN: could not score {img_path.name}: {e}")
                score = np.nan

            rows.append({
                "seed_folder": chain["seed_folder"],
                "seed_name":   chain["seed_name"],
                "category":    chain["category"],
                "prompt_type": chain["prompt_type"],
                "chain_num":   chain["chain_num"],
                "iteration":   iter_num,
                "clip_score":  score,
            })
            done_keys.add(key)

            # Incremental checkpoint save
            if checkpoint_path and len(rows) % checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    print()
    return pd.DataFrame(rows)


# ================================
# LOG PARSING
# ================================

def parse_blocks(log_path: Path) -> pd.DataFrame:
    """
    Returns one row per policy block event from master_log.json:
      seed_folder, seed_name, category, chain_num, iteration,
      block_stage, had_refusal
    """
    if not log_path.exists():
        print(f"WARN: {log_path} not found — skipping block analysis.")
        return pd.DataFrame()

    with open(log_path, encoding="utf-8") as f:
        master = json.load(f)

    rows = []
    for chain in master.get("chains", []):
        seed_folder = chain.get("seed_name", "")
        meta        = SEED_METADATA.get(seed_folder, {})
        seed_name   = meta.get("name", seed_folder)
        category    = meta.get("category", "unknown")
        chain_num   = chain.get("chain_num", 0)

        for it in chain.get("iterations", []):
            iteration = it.get("iteration", 0)
            caption   = it.get("caption", "") or ""
            had_refusal = any(p in caption.lower() for p in [
                "i'm not able to identify", "i can't identify",
                "i cannot identify", "unable to identify",
            ])

            for stage in ("caption", "generation"):
                if it.get(f"{stage}_error") == "CONTENT_POLICY_VIOLATION":
                    rows.append({
                        "seed_folder": seed_folder,
                        "seed_name":   seed_name,
                        "category":    category,
                        "chain_num":   chain_num,
                        "iteration":   iteration,
                        "block_stage": stage,
                        "had_refusal": had_refusal,
                    })

    return pd.DataFrame(rows)


def parse_completion(log_path: Path) -> pd.DataFrame:
    """
    Returns per-chain completion stats from master_log.json.
    """
    if not log_path.exists():
        return pd.DataFrame()

    with open(log_path, encoding="utf-8") as f:
        master = json.load(f)

    rows = []
    for chain in master.get("chains", []):
        seed_folder = chain.get("seed_name", "")
        meta        = SEED_METADATA.get(seed_folder, {})
        rows.append({
            "seed_folder":          seed_folder,
            "seed_name":            meta.get("name", seed_folder),
            "category":             meta.get("category", "unknown"),
            "chain_num":            chain.get("chain_num", 0),
            "completed_iterations": chain.get("completed_iterations", 0),
            "terminated_early":     chain.get("chain_terminated_early", False),
            "termination_reason":   chain.get("termination_reason", None),
            "total_tokens":         chain.get("total_tokens_used", 0),
        })

    return pd.DataFrame(rows)


# ================================
# SUMMARY TABLES
# ================================

def make_decay_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_df.groupby(["category", "iteration"])["clip_score"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )


def make_preservation_table(scores_df: pd.DataFrame, blocks_df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed mean CLIP at key checkpoints plus block counts."""
    checkpoints = [cp for cp in [1, 5, 10, 15] if cp in scores_df["iteration"].values]
    rows = []

    for seed_name, grp in scores_df.groupby("seed_name"):
        row = {"seed_name": seed_name, "category": grp["category"].iloc[0]}
        for cp in checkpoints:
            cp_scores       = grp[grp["iteration"] == cp]["clip_score"].dropna()
            row[f"iter_{cp:02d}_mean"] = round(cp_scores.mean(), 4) if len(cp_scores) else np.nan
            row[f"iter_{cp:02d}_sd"]   = round(cp_scores.std(),  4) if len(cp_scores) > 1 else np.nan

        if not blocks_df.empty:
            sb = blocks_df[blocks_df["seed_name"] == seed_name]
            row["total_blocks"]      = len(sb)
            row["caption_blocks"]    = len(sb[sb["block_stage"] == "caption"])
            row["generation_blocks"] = len(sb[sb["block_stage"] == "generation"])
            row["first_block_iter"]  = sb["iteration"].min() if len(sb) else np.nan
        else:
            row.update({"total_blocks": 0, "caption_blocks": 0,
                        "generation_blocks": 0, "first_block_iter": np.nan})

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["category", "seed_name"])


def make_block_table(blocks_df: pd.DataFrame, completion_df: pd.DataFrame) -> pd.DataFrame:
    if blocks_df.empty:
        return pd.DataFrame()

    rows = []
    for (seed_name, chain_num), grp in blocks_df.groupby(["seed_name", "chain_num"]):
        comp = completion_df[
            (completion_df["seed_name"] == seed_name) &
            (completion_df["chain_num"] == chain_num)
        ] if not completion_df.empty else pd.DataFrame()

        rows.append({
            "seed_name":         seed_name,
            "category":          grp["category"].iloc[0],
            "chain_num":         chain_num,
            "total_blocks":      len(grp),
            "first_block_iter":  grp["iteration"].min(),
            "caption_blocks":    len(grp[grp["block_stage"] == "caption"]),
            "generation_blocks": len(grp[grp["block_stage"] == "generation"]),
            "had_refusal":       grp["had_refusal"].any(),
            "completed_iters":   comp["completed_iterations"].iloc[0] if len(comp) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["seed_name", "chain_num"])


# ================================
# FIGURES
# ================================

def fig_decay_curves(scores_df: pd.DataFrame, out_path: Path):
    """Mean CLIP decay per category with individual seed lines in background."""
    categories = sorted(scores_df["category"].unique())
    n_cats     = len(categories)

    plt.style.use(FIGURE_STYLE)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        color  = CATEGORY_COLORS.get(cat, "#555")
        cat_df = scores_df[scores_df["category"] == cat]

        for _, sdf in cat_df.groupby("seed_name"):
            seed_mean = sdf.groupby("iteration")["clip_score"].mean()
            ax.plot(seed_mean.index, seed_mean.values,
                    color=color, alpha=0.2, linewidth=1.0, zorder=2)

        cat_mean = cat_df.groupby("iteration")["clip_score"].mean()
        cat_sd   = cat_df.groupby("iteration")["clip_score"].std()
        ax.plot(cat_mean.index, cat_mean.values,
                color=color, linewidth=2.5, label=f"{cat} mean", zorder=3)
        ax.fill_between(cat_mean.index,
                        cat_mean - cat_sd, cat_mean + cat_sd,
                        color=color, alpha=0.12, zorder=1)

        ax.set_title(cat.capitalize(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("CLIP Cosine Similarity (vs seed)", fontsize=11)
        ax.set_xlim(1, scores_df["iteration"].max())
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("CLIP Similarity Decay by Category",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_individual_heatmap(scores_df: pd.DataFrame, out_path: Path):
    """Heatmap: seeds × iterations, color = mean CLIP score."""
    pivot = (
        scores_df.groupby(["seed_name", "iteration"])["clip_score"]
        .mean()
        .unstack(level="iteration")
    )

    cat_map       = scores_df.drop_duplicates("seed_name").set_index("seed_name")["category"]
    pivot["_cat"] = pivot.index.map(cat_map)
    pivot["_mu"]  = pivot.drop(columns=["_cat"]).mean(axis=1)
    pivot         = pivot.sort_values(["_cat", "_mu"], ascending=[True, False])
    pivot         = pivot.drop(columns=["_cat", "_mu"])

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.7), max(5, len(pivot) * 0.45))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.3, vmax=0.95, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_title("Mean CLIP Similarity to Seed Image",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_individual_decay(scores_df: pd.DataFrame, out_path: Path):
    """All seed trajectories on one axes, colour-coded by category."""
    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))
    max_iter = scores_df["iteration"].max()

    for seed_name, sdf in scores_df.groupby("seed_name"):
        cat   = sdf["category"].iloc[0]
        color = CATEGORY_COLORS.get(cat, "#9ca3af")
        mean  = sdf.groupby("iteration")["clip_score"].mean()
        ax.plot(mean.index, mean.values, color=color, linewidth=1.5, alpha=0.85)
        ax.text(mean.index.max() + 0.15, mean.iloc[-1],
                seed_name, fontsize=7, va="center", color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean CLIP Cosine Similarity (vs seed)", fontsize=12)
    ax.set_title("Individual Seed CLIP Decay Trajectories",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, max_iter + 3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    handles = [
        plt.Line2D([0], [0], color=CATEGORY_COLORS[c], linewidth=2, label=c.capitalize())
        for c in CATEGORY_COLORS if c in scores_df["category"].values
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_block_distribution(blocks_df: pd.DataFrame, out_path: Path):
    """Two-panel: total blocks per seed (bar) + first block iteration (scatter)."""
    if blocks_df.empty:
        print("  Skipping block distribution (no blocks found).")
        return

    seed_blocks = (
        blocks_df.groupby("seed_name")
        .agg(total_blocks=("iteration", "count"),
             first_block=("iteration", "min"),
             category=("category", "first"))
        .reset_index()
        .sort_values("total_blocks", ascending=False)
    )

    plt.style.use(FIGURE_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = [CATEGORY_COLORS.get(c, "#9ca3af") for c in seed_blocks["category"]]
    bars   = ax1.barh(seed_blocks["seed_name"], seed_blocks["total_blocks"],
                      color=colors, edgecolor="white")
    ax1.set_xlabel("Total Policy Blocks", fontsize=11)
    ax1.set_title("Policy Blocks per Seed", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    for bar, val in zip(bars, seed_blocks["total_blocks"]):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9)

    has_block     = seed_blocks[seed_blocks["first_block"].notna()]
    scatter_colors = [CATEGORY_COLORS.get(c, "#9ca3af") for c in has_block["category"]]
    ax2.scatter(has_block["first_block"], has_block["seed_name"],
                c=scatter_colors, s=80, zorder=3)
    ax2.set_xlabel("Iteration of First Block", fontsize=11)
    ax2.set_title("When Does Blocking First Occur?", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, blocks_df["iteration"].max() + 1)
    ax2.invert_yaxis()
    mean_iter = blocks_df["iteration"].mean()
    ax2.axvline(mean_iter, color="gray", linestyle="--", linewidth=1,
                label=f"mean = {mean_iter:.1f}")
    ax2.legend(fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c], label=c.capitalize())
        for c in CATEGORY_COLORS if c in seed_blocks["category"].values
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.06), fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_category_boxplot(scores_df: pd.DataFrame, out_path: Path):
    """Box plots of CLIP distribution by category at key iteration checkpoints."""
    checkpoints = [cp for cp in [1, 5, 10, 15] if cp in scores_df["iteration"].values]
    categories  = sorted(scores_df["category"].unique())

    plt.style.use(FIGURE_STYLE)
    fig, axes = plt.subplots(1, len(checkpoints),
                             figsize=(4 * len(checkpoints), 5), sharey=True)
    if len(checkpoints) == 1:
        axes = [axes]

    for ax, cp in zip(axes, checkpoints):
        cp_df  = scores_df[scores_df["iteration"] == cp]
        data   = [cp_df[cp_df["category"] == cat]["clip_score"].dropna().values
                  for cat in categories]
        colors = [CATEGORY_COLORS.get(cat, "#9ca3af") for cat in categories]

        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels([c.capitalize() for c in categories], fontsize=10)
        ax.set_title(f"Iteration {cp}", fontsize=12, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("CLIP Cosine Similarity", fontsize=11)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for i, (cat, d) in enumerate(zip(categories, data), start=1):
            ax.text(i, 0.03, f"n={len(d)}", ha="center", fontsize=8, color="gray")

    fig.suptitle("CLIP Score Distribution by Category at Key Iterations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ================================
# STATISTICAL TESTS
# ================================

def run_stats(scores_df: pd.DataFrame, blocks_df: pd.DataFrame) -> str:
    lines = ["=" * 60, "STATISTICAL SUMMARY", "=" * 60, ""]

    # Overall decay
    valid = scores_df[["iteration", "clip_score"]].dropna()
    r, p  = stats.pearsonr(valid["iteration"], valid["clip_score"])
    lines.append(f"Overall CLIP decay (Pearson r): r={r:.3f}, p={p:.2e}")
    lines.append("")

    # Per-category linear decay slope
    lines.append("Linear decay rate per category:")
    for cat, cdf in scores_df.groupby("category"):
        v = cdf[["iteration", "clip_score"]].dropna()
        if len(v) < 4:
            continue
        slope, _, r_val, p2, _ = stats.linregress(v["iteration"], v["clip_score"])
        lines.append(f"  {cat:14s}  slope={slope:.4f}/iter  r²={r_val**2:.3f}  p={p2:.2e}")
    lines.append("")

    # Iter 1 vs last iteration t-test
    max_iter = scores_df["iteration"].max()
    lines.append(f"CLIP at iter 1 vs iter {max_iter} (independent t-test per category):")
    for cat, cdf in scores_df.groupby("category"):
        i1   = cdf[cdf["iteration"] == 1]["clip_score"].dropna()
        ilast = cdf[cdf["iteration"] == max_iter]["clip_score"].dropna()
        if len(i1) < 2 or len(ilast) < 2:
            continue
        t, p3 = stats.ttest_ind(i1, ilast)
        lines.append(
            f"  {cat:14s}  iter1 μ={i1.mean():.3f}  "
            f"iter{max_iter} μ={ilast.mean():.3f}  t={t:.2f}  p={p3:.2e}"
        )
    lines.append("")

    # One-way ANOVA across categories at iter 1
    categories = sorted(scores_df["category"].unique())
    if len(categories) >= 2:
        lines.append("One-way ANOVA at iteration 1 (category effect on initial preservation):")
        iter1  = scores_df[scores_df["iteration"] == 1]
        groups = [iter1[iter1["category"] == c]["clip_score"].dropna().values
                  for c in categories if len(iter1[iter1["category"] == c]) >= 2]
        if len(groups) >= 2:
            f, pf = stats.f_oneway(*groups)
            lines.append(f"  F={f:.2f}  p={pf:.2e}")
        lines.append("")

    # Block summary
    if not blocks_df.empty:
        first_blocks = blocks_df.groupby(["seed_name", "chain_num"])["iteration"].min()
        lines.append("Block summary:")
        lines.append(f"  Total block events:        {len(blocks_df)}")
        lines.append(f"  Seeds with blocks:         {blocks_df['seed_name'].nunique()}")
        lines.append(f"  Mean first block iter:     {first_blocks.mean():.1f} "
                     f"(SD={first_blocks.std():.1f})")
        lines.append(f"  Caption-stage blocks:      "
                     f"{len(blocks_df[blocks_df['block_stage']=='caption'])}")
        lines.append(f"  Generation-stage blocks:   "
                     f"{len(blocks_df[blocks_df['block_stage']=='generation'])}")
        lines.append(f"  Preceded by refusal text:  "
                     f"{blocks_df['had_refusal'].sum()} / {len(blocks_df)}")
        lines.append("")
        lines.append("Blocks per seed (all chains combined):")
        for seed, grp in sorted(blocks_df.groupby("seed_name"),
                                key=lambda x: -len(x[1])):
            lines.append(f"  {seed:24s}  {len(grp):3d} blocks  "
                         f"(first at iter {grp['iteration'].min()})")
    else:
        lines.append("No policy blocks found in log.")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="CLIP similarity analysis for the telephone game dataset."
    )
    parser.add_argument("--data-dir",   default="output",
                        help="Path to telephone.py output dir (default: output)")
    parser.add_argument("--log",        default="output/master_log.json",
                        help="Path to master_log.json (default: output/master_log.json)")
    parser.add_argument("--output-dir", default="clip_analysis_output",
                        help="Where to write results (default: clip_analysis_output)")
    parser.add_argument("--no-clip",    action="store_true",
                        help="Skip CLIP scoring; only parse log for block stats")
    parser.add_argument("--device",     default=None,
                        help="Force device: cpu or cuda (default: auto)")
    parser.add_argument("--scores-csv", default=None,
                        help="Load pre-computed scores CSV to skip re-running CLIP")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    log_path  = Path(args.log)
    out_dir   = Path(args.output_dir)
    data_out  = out_dir / "data"
    figs_out  = out_dir / "figures"
    data_out.mkdir(parents=True, exist_ok=True)
    figs_out.mkdir(parents=True, exist_ok=True)

    print(f"\nTelephone Game — CLIP Analysis")
    print(f"  Data dir:   {data_dir}")
    print(f"  Master log: {log_path}")
    print(f"  Output dir: {out_dir}\n")

    # Discover chains
    print("Discovering chains...")
    chains = discover_chains(data_dir)
    n_seeds = len(set(c["seed_folder"] for c in chains))
    print(f"  Found {len(chains)} chains across {n_seeds} seeds\n")
    if not chains:
        print("ERROR: No chains found. Check --data-dir path.")
        sys.exit(1)

    # CLIP scores
    scores_csv_path = data_out / "clip_scores.csv"

    if args.scores_csv:
        print(f"Loading pre-computed scores from {args.scores_csv}...")
        scores_df = pd.read_csv(args.scores_csv)

    elif args.no_clip:
        print("--no-clip set: skipping CLIP scoring.")
        rows = []
        for c in chains:
            for it, _ in c["generated_images"]:
                rows.append({
                    "seed_folder": c["seed_folder"], "seed_name": c["seed_name"],
                    "category":    c["category"],    "prompt_type": c["prompt_type"],
                    "chain_num":   c["chain_num"],   "iteration": it,
                    "clip_score":  np.nan,
                })
        scores_df = pd.DataFrame(rows)

    else:
        print("Computing CLIP scores...")
        model, preprocess, device = load_clip_model(args.device)
        scores_df = compute_clip_scores(
            chains, model, preprocess, device,
            checkpoint_path=scores_csv_path,
        )
        scores_df.to_csv(scores_csv_path, index=False)
        print(f"  Saved raw scores: {scores_csv_path.name}\n")

    # Parse log
    print("Parsing master log...")
    blocks_df     = parse_blocks(log_path)
    completion_df = parse_completion(log_path)
    print(f"  Found {len(blocks_df)} policy block event(s)\n")

    # Summary tables
    print("Building summary tables...")
    has_scores = not scores_df["clip_score"].isna().all()

    if has_scores:
        make_decay_summary(scores_df).to_csv(data_out / "decay_summary.csv", index=False)
        print(f"  Saved: decay_summary.csv")

        pres_df = make_preservation_table(scores_df, blocks_df)
        pres_df.to_csv(data_out / "preservation_table.csv", index=False)
        print(f"  Saved: preservation_table.csv")
        print()
        print(pres_df.to_string(index=False))
        print()

    if not blocks_df.empty:
        make_block_table(blocks_df, completion_df).to_csv(
            data_out / "block_table.csv", index=False
        )
        print(f"  Saved: block_table.csv")

    # Figures
    if has_scores:
        print("\nGenerating figures...")
        fig_decay_curves(scores_df,       figs_out / "decay_curves.png")
        fig_individual_heatmap(scores_df, figs_out / "individual_heatmap.png")
        fig_individual_decay(scores_df,   figs_out / "individual_decay.png")
        fig_category_boxplot(scores_df,   figs_out / "category_boxplot.png")

    if not blocks_df.empty:
        fig_block_distribution(blocks_df, figs_out / "block_distribution.png")

    # Stats
    if has_scores:
        print("\nRunning statistical tests...")
        stats_text = run_stats(scores_df, blocks_df)
        print(stats_text)
        (data_out / "statistical_summary.txt").write_text(stats_text, encoding="utf-8")
        print(f"  Saved: statistical_summary.txt")

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()