"""
face_analysis.py  —  Face identity preservation analysis for the telephone game dataset.

Uses ArcFace (via insightface) to measure how well the identity of the person in each
generated image matches the original seed image, across all chains and iterations.

This is complementary to clip_analysis.py:
  - CLIP measures overall visual similarity (composition, color, setting)
  - Face analysis measures face identity similarity specifically

If no face is detected in a generated image, that is recorded as NaN and flagged —
it means the chain drifted far enough that the model stopped generating a recognizable face.

Outputs (written to --output-dir, default: face_analysis_output/):
  data/
    face_scores.csv           — raw per-chain per-iteration face similarity scores
    face_decay_summary.csv    — mean +/- SD by category and iteration
    face_preservation_table.csv — per-seed face similarity at iters 1, 5, 10, 15
    face_detection_table.csv  — detection rate per seed (how often a face was found)
    statistical_summary.txt   — key stats
  figures/
    face_decay_curves.png     — mean face similarity decay by category
    face_heatmap.png          — per-seed face similarity heatmap
    face_individual_decay.png — all seed trajectories
    face_vs_clip.png          — scatter comparing face vs CLIP scores (if clip_scores.csv provided)
    detection_rate.png        — face detection rate per seed per iteration

Dependencies:
  pip install insightface onnxruntime pillow numpy pandas matplotlib scipy

Usage:
  python face_analysis.py                             # uses ./output and ./master_log.json
  python face_analysis.py --data-dir /path/output
  python face_analysis.py --output-dir face_analysis_output
  python face_analysis.py --clip-scores analysis_output/data/clip_scores.csv  # enables comparison plot
  python face_analysis.py --scores-csv face_analysis_output/data/face_scores.csv  # reuse saved scores
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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


# ================================
# ARCFACE MODEL
# ================================

def load_face_model():
    """
    Loads insightface ArcFace model for face detection and embedding.
    Returns the app object ready for get() calls.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("\nERROR: insightface is required.")
        print("Install with:  pip install insightface onnxruntime")
        sys.exit(1)

    print("Loading ArcFace model...")
    app = FaceAnalysis(
        name="buffalo_l",           # best accuracy model bundle
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("ArcFace ready.\n")
    return app


def get_face_embedding(image_path: Path, app) -> tuple[np.ndarray | None, str]:
    """
    Detects the largest face in the image and returns its ArcFace embedding.

    Returns (embedding, status) where:
      embedding: unit-normalised 512-d vector, or None if no face detected
      status:    "ok", "no_face", or "error:<msg>"

    When multiple faces are detected (e.g. group photos), uses the largest
    bounding box on the assumption it's the primary subject.
    """
    from PIL import Image as PILImage

    try:
        img = np.array(PILImage.open(image_path).convert("RGB"))
        faces = app.get(img)

        if not faces:
            return None, "no_face"

        # Pick the largest detected face by bounding box area
        face = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))

        emb = face.normed_embedding  # already unit-normalised by insightface
        return emb.astype(np.float32), "ok"

    except Exception as e:
        return None, f"error:{str(e)[:80]}"


def face_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised embeddings."""
    return float(np.dot(a, b))


# ================================
# DATA DISCOVERY  (mirrors clip_analysis.py)
# ================================

def discover_chains(data_dir: Path) -> list[dict]:
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
# FACE SCORING
# ================================

def compute_face_scores(chains: list[dict], app) -> pd.DataFrame:
    """
    For each generated image, extracts the ArcFace embedding and computes
    cosine similarity against the seed image's face embedding.

    Records face_score (float or NaN) and detection_status for every image.
    """
    rows  = []
    total = sum(len(c["generated_images"]) for c in chains)
    done  = 0

    # Pre-compute seed embeddings (one per chain — same seed image)
    seed_cache: dict[Path, tuple] = {}

    for chain in chains:
        seed_path = chain["seed_image"]
        if seed_path not in seed_cache:
            seed_emb, seed_status = get_face_embedding(seed_path, app)
            seed_cache[seed_path] = (seed_emb, seed_status)
            if seed_status != "ok":
                print(f"\n  WARN: seed face not detected for "
                      f"{chain['seed_name']} chain {chain['chain_num']}: {seed_status}")
        seed_emb, seed_status = seed_cache[seed_path]

        for iter_num, img_path in chain["generated_images"]:
            done += 1
            print(f"  Scoring faces: {done}/{total}", end="\r")

            gen_emb, gen_status = get_face_embedding(img_path, app)

            if seed_emb is None or gen_emb is None:
                score = np.nan
            else:
                score = face_cosine_sim(seed_emb, gen_emb)

            rows.append({
                "seed_folder":      chain["seed_folder"],
                "seed_name":        chain["seed_name"],
                "category":         chain["category"],
                "prompt_type":      chain["prompt_type"],
                "chain_num":        chain["chain_num"],
                "iteration":        iter_num,
                "face_score":       score,
                "seed_status":      seed_status,
                "gen_status":       gen_status,
                "face_detected":    gen_status == "ok",
            })

    print()
    return pd.DataFrame(rows)


# ================================
# SUMMARY TABLES
# ================================

def make_face_decay_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_df.groupby(["category", "iteration"])["face_score"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )


def make_face_preservation_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [cp for cp in [1, 5, 10, 15] if cp in scores_df["iteration"].values]
    rows = []

    for seed_name, grp in scores_df.groupby("seed_name"):
        row = {"seed_name": seed_name, "category": grp["category"].iloc[0]}

        for cp in checkpoints:
            cp_scores = grp[grp["iteration"] == cp]["face_score"].dropna()
            row[f"iter_{cp:02d}_mean"] = round(cp_scores.mean(), 4) if len(cp_scores) else np.nan
            row[f"iter_{cp:02d}_sd"]   = round(cp_scores.std(),  4) if len(cp_scores) > 1 else np.nan

        # Detection rate across all iterations
        total_imgs   = len(grp)
        detected     = grp["face_detected"].sum()
        row["detection_rate"]      = round(detected / total_imgs, 3) if total_imgs else np.nan
        row["total_images"]        = total_imgs
        row["faces_detected"]      = int(detected)
        row["seed_face_detected"]  = (grp["seed_status"] == "ok").any()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["category", "seed_name"])


def make_detection_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Detection rate per seed per iteration — useful for seeing where faces disappear."""
    return (
        scores_df.groupby(["seed_name", "iteration"])["face_detected"]
        .agg(detection_rate="mean", n="count")
        .reset_index()
    )


# ================================
# FIGURES
# ================================

def fig_face_decay_curves(scores_df: pd.DataFrame, out_path: Path):
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
            seed_mean = sdf.groupby("iteration")["face_score"].mean()
            ax.plot(seed_mean.index, seed_mean.values,
                    color=color, alpha=0.2, linewidth=1.0, zorder=2)

        cat_mean = cat_df.groupby("iteration")["face_score"].mean()
        cat_sd   = cat_df.groupby("iteration")["face_score"].std()
        ax.plot(cat_mean.index, cat_mean.values,
                color=color, linewidth=2.5, zorder=3)
        ax.fill_between(cat_mean.index,
                        cat_mean - cat_sd, cat_mean + cat_sd,
                        color=color, alpha=0.12, zorder=1)

        ax.set_title(cat.capitalize(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("ArcFace Cosine Similarity (vs seed face)", fontsize=11)
        ax.set_xlim(1, scores_df["iteration"].max())
        ax.set_ylim(-0.2, 1)
        # ArcFace threshold: ~0.3 is commonly used as same-identity threshold
        ax.axhline(0.3, color="red",  linestyle="--", linewidth=0.9,
                   alpha=0.6, label="identity threshold (0.3)")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("ArcFace Identity Similarity Decay by Category",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_face_heatmap(scores_df: pd.DataFrame, out_path: Path):
    pivot = (
        scores_df.groupby(["seed_name", "iteration"])["face_score"]
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
                   vmin=-0.1, vmax=0.8, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_title("Mean ArcFace Identity Similarity to Seed Face",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="ArcFace Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_face_individual_decay(scores_df: pd.DataFrame, out_path: Path):
    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))
    max_iter = scores_df["iteration"].max()

    for seed_name, sdf in scores_df.groupby("seed_name"):
        cat   = sdf["category"].iloc[0]
        color = CATEGORY_COLORS.get(cat, "#9ca3af")
        mean  = sdf.groupby("iteration")["face_score"].mean()
        ax.plot(mean.index, mean.values, color=color, linewidth=1.5, alpha=0.85)
        ax.text(mean.index.max() + 0.15, mean.iloc[-1],
                seed_name, fontsize=7, va="center", color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Mean ArcFace Cosine Similarity (vs seed face)", fontsize=12)
    ax.set_title("Individual Seed Face Identity Decay Trajectories",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, max_iter + 3)
    ax.set_ylim(-0.2, 1)
    ax.axhline(0.3, color="red",  linestyle="--", linewidth=0.9,
               alpha=0.6, label="identity threshold (0.3)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)

    handles = [
        plt.Line2D([0], [0], color=CATEGORY_COLORS[c], linewidth=2, label=c.capitalize())
        for c in CATEGORY_COLORS if c in scores_df["category"].values
    ]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--",
                               linewidth=1, label="identity threshold"))
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_detection_rate(scores_df: pd.DataFrame, out_path: Path):
    """
    Heatmap of face detection rate per seed per iteration.
    White = face always detected, dark = face often missing.
    """
    pivot = (
        scores_df.groupby(["seed_name", "iteration"])["face_detected"]
        .mean()
        .unstack(level="iteration")
    )

    # Sort by mean detection rate descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.7), max(5, len(pivot) * 0.45))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_title("Face Detection Rate per Seed and Iteration\n"
                 "(1.0 = face detected in all chains, 0.0 = no faces detected)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Detection Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def fig_face_vs_clip(face_df: pd.DataFrame, clip_path: Path, out_path: Path):
    """
    Scatter plot: CLIP score vs face score for every image.
    Highlights the divergence between overall visual similarity and face identity.
    """
    if not clip_path.exists():
        print(f"  Skipping face vs CLIP plot (no CLIP scores at {clip_path})")
        return

    clip_df = pd.read_csv(clip_path)

    merged = pd.merge(
        face_df[["seed_folder", "chain_num", "iteration", "seed_name",
                 "category", "face_score", "face_detected"]],
        clip_df[["seed_folder", "chain_num", "iteration", "clip_score"]],
        on=["seed_folder", "chain_num", "iteration"],
        how="inner",
    ).dropna(subset=["face_score", "clip_score"])

    plt.style.use(FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(8, 7))

    for cat, grp in merged.groupby("category"):
        color = CATEGORY_COLORS.get(cat, "#9ca3af")
        ax.scatter(grp["clip_score"], grp["face_score"],
                   c=color, alpha=0.3, s=12, label=cat.capitalize(), zorder=2)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=0.8, alpha=0.5, label="x=y")
    ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8,
               alpha=0.6, label="face identity threshold")

    # Correlation
    r, p = stats.pearsonr(merged["clip_score"], merged["face_score"])
    ax.text(0.05, 0.95, f"r = {r:.3f}  (p = {p:.2e})",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("CLIP Cosine Similarity (overall visual)", fontsize=12)
    ax.set_ylabel("ArcFace Cosine Similarity (face identity)", fontsize=12)
    ax.set_title("CLIP vs Face Identity Similarity\n"
                 "Points below diagonal: visual similarity preserved but face identity lost",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ================================
# STATISTICAL TESTS
# ================================

def run_stats(scores_df: pd.DataFrame) -> str:
    lines = ["=" * 60, "FACE IDENTITY STATISTICAL SUMMARY", "=" * 60, ""]

    valid = scores_df[["iteration", "face_score"]].dropna()

    if len(valid) == 0:
        lines.append("No valid face scores found.")
        lines.append("This likely means faces were not detected in generated images.")
        lines.append("Check detection_rate.png and face_detection_table.csv.")
        lines += ["", "=" * 60]
        return "\n".join(lines)

    # Overall decay
    r, p = stats.pearsonr(valid["iteration"], valid["face_score"])
    lines.append(f"Overall face identity decay (Pearson r): r={r:.3f}, p={p:.2e}")
    lines.append("")

    # Per-category decay
    lines.append("Linear decay rate per category:")
    for cat, cdf in scores_df.groupby("category"):
        v = cdf[["iteration", "face_score"]].dropna()
        if len(v) < 4:
            continue
        slope, _, r_val, p2, _ = stats.linregress(v["iteration"], v["face_score"])
        lines.append(f"  {cat:14s}  slope={slope:.4f}/iter  r²={r_val**2:.3f}  p={p2:.2e}")
    lines.append("")

    # Iter 1 vs last
    max_iter = scores_df["iteration"].max()
    lines.append(f"Face score at iter 1 vs iter {max_iter} (independent t-test per category):")
    for cat, cdf in scores_df.groupby("category"):
        i1    = cdf[cdf["iteration"] == 1]["face_score"].dropna()
        ilast = cdf[cdf["iteration"] == max_iter]["face_score"].dropna()
        if len(i1) < 2 or len(ilast) < 2:
            continue
        t, p3 = stats.ttest_ind(i1, ilast)
        lines.append(
            f"  {cat:14s}  iter1 μ={i1.mean():.3f}  "
            f"iter{max_iter} μ={ilast.mean():.3f}  t={t:.2f}  p={p3:.2e}"
        )
    lines.append("")

    # Detection rates
    lines.append("Face detection rates per seed:")
    det = (
        scores_df.groupby("seed_name")["face_detected"]
        .agg(rate="mean", total="count")
        .reset_index()
        .sort_values("rate", ascending=False)
    )
    for _, row in det.iterrows():
        lines.append(
            f"  {row['seed_name']:24s}  "
            f"{row['rate']*100:5.1f}%  ({int(row['rate']*row['total'])}/{int(row['total'])} images)"
        )
    lines.append("")

    # Identity threshold analysis (ArcFace >0.3 = same identity)
    threshold = 0.3
    above = scores_df[scores_df["face_score"] > threshold]
    total_scored = scores_df["face_score"].notna().sum()
    lines.append(f"Images above identity threshold ({threshold}):")
    lines.append(f"  Overall: {len(above)}/{total_scored} "
                 f"({100*len(above)/total_scored:.1f}% of scored images)")
    lines.append("")
    lines.append(f"  By iteration:")
    for it, grp in scores_df.groupby("iteration"):
        scored  = grp["face_score"].notna().sum()
        n_above = (grp["face_score"] > threshold).sum()
        pct     = 100 * n_above / scored if scored > 0 else 0
        lines.append(f"    iter {it:2d}:  {n_above:3d}/{scored:3d}  ({pct:.0f}%)")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="ArcFace identity preservation analysis for the telephone game dataset."
    )
    parser.add_argument("--data-dir",    default="output",
                        help="Path to telephone.py output dir (default: output)")
    parser.add_argument("--output-dir",  default="face_analysis_output",
                        help="Where to write results (default: face_analysis_output)")
    parser.add_argument("--scores-csv",  default=None,
                        help="Load pre-computed face scores CSV to skip re-running ArcFace")
    parser.add_argument("--clip-scores", default="analysis_output/data/clip_scores.csv",
                        help="Path to CLIP scores CSV for comparison plot")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    data_out = out_dir / "data"
    figs_out = out_dir / "figures"
    data_out.mkdir(parents=True, exist_ok=True)
    figs_out.mkdir(parents=True, exist_ok=True)

    print(f"\nTelephone Game — Face Identity Analysis (ArcFace)")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {out_dir}\n")

    # Discover chains
    print("Discovering chains...")
    chains  = discover_chains(data_dir)
    n_seeds = len(set(c["seed_folder"] for c in chains))
    print(f"  Found {len(chains)} chains across {n_seeds} seeds\n")
    if not chains:
        print("ERROR: No chains found. Check --data-dir path.")
        sys.exit(1)

    # Face scores
    scores_csv_path = data_out / "face_scores.csv"

    if args.scores_csv:
        print(f"Loading pre-computed face scores from {args.scores_csv}...")
        scores_df = pd.read_csv(args.scores_csv)
        scores_df["face_detected"] = scores_df["gen_status"] == "ok"
    else:
        app = load_face_model()
        print("Computing face identity scores...")
        scores_df = compute_face_scores(chains, app)
        scores_df.to_csv(scores_csv_path, index=False)
        print(f"  Saved: {scores_csv_path.name}\n")

    # Report seed-level detection failures upfront
    seed_no_face = scores_df[scores_df["seed_status"] != "ok"]["seed_name"].unique()
    if len(seed_no_face):
        print(f"  WARNING: No face detected in seed image for: {', '.join(seed_no_face)}")
        print(f"  These seeds will have NaN face scores throughout.\n")

    # Summary tables
    print("Building summary tables...")
    has_scores = scores_df["face_score"].notna().any()

    if has_scores:
        make_face_decay_summary(scores_df).to_csv(
            data_out / "face_decay_summary.csv", index=False
        )
        print(f"  Saved: face_decay_summary.csv")

        pres_df = make_face_preservation_table(scores_df)
        pres_df.to_csv(data_out / "face_preservation_table.csv", index=False)
        print(f"  Saved: face_preservation_table.csv")
        print()
        print(pres_df[["seed_name", "category", "iter_01_mean", "iter_05_mean",
                        "iter_10_mean", "iter_15_mean",
                        "detection_rate", "faces_detected", "total_images"]].to_string(index=False))
        print()

    make_detection_table(scores_df).to_csv(
        data_out / "face_detection_table.csv", index=False
    )
    print(f"  Saved: face_detection_table.csv")

    # Figures
    print("\nGenerating figures...")
    if has_scores:
        fig_face_decay_curves(scores_df,      figs_out / "face_decay_curves.png")
        fig_face_heatmap(scores_df,           figs_out / "face_heatmap.png")
        fig_face_individual_decay(scores_df,  figs_out / "face_individual_decay.png")
        fig_face_vs_clip(scores_df, Path(args.clip_scores),
                         figs_out / "face_vs_clip.png")

    fig_detection_rate(scores_df, figs_out / "detection_rate.png")

    # Stats
    print("\nRunning statistical tests...")
    stats_text = run_stats(scores_df)
    print(stats_text)
    (data_out / "statistical_summary.txt").write_text(stats_text, encoding="utf-8")
    print(f"  Saved: statistical_summary.txt")

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
