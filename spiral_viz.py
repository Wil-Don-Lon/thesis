"""
spiral_viz.py  —  Golden ratio spiral visualization of telephone game image chains.

Places images along a Fermat spiral (sunflower spiral) using the golden angle,
ordered from center outward: seed image first, then round-robin across chains
by iteration (seed, chain1.1, chain2.1, ..., chain5.1, chain1.2, chain2.2, ...).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIGURATION  (edit these defaults here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SHAPE        = "full"     # "full"   — full rectangular image, corners intact
                            # "square" — cropped to square
                            # "circle" — cropped to circle

  CANVAS_SIZE  = 5000       # poster size in pixels (square canvas)

  CELL_GAP     = 0.02       # spacing between cells as a fraction of cell size
                            # 0.00 = touching, 0.10 = 10% gap, -0.05 = 5% overlap

  SEED_BOOST   = 1.6        # seed image size multiplier vs outer cells (e.g. 1.6 = 60% larger)

  ROTATE       = True       # rotate each image to follow its spiral angle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All of the above can also be set via CLI flags (which override these defaults).

Usage:
  python spiral_viz.py --seed "MLK"
  python spiral_viz.py --seed "test 5" --shape circle --gap 0.05
  python spiral_viz.py --seed "MLK" --size 8000 --gap 0.0 --no-rotate
  python spiral_viz.py --seed "MLK" --shape square --seed-boost 2.0
  python spiral_viz.py --list
"""

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ================================
# CONFIGURATION DEFAULTS
# ================================
# Edit these to change defaults without using CLI flags.

SHAPE       = "circle"    # "full", "square", or "circle"
CANVAS_SIZE = 6000      # poster size in pixels
CELL_GAP    = 0.0      # gap between cells (fraction of cell size; negative = overlap)
SEED_BOOST  = 1.6       # seed image size multiplier
ROTATE      = False      # rotate images to follow spiral angle

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

# Reverse lookup: display name -> folder name
NAME_TO_FOLDER = {v["name"].lower(): k for k, v in SEED_METADATA.items()}

# ================================
# GOLDEN RATIO CONSTANTS
# ================================

PHI          = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi * (2 - PHI)   # ~137.508 degrees in radians


# ================================
# DATA DISCOVERY
# ================================

def find_seed_folder(data_dir: Path, seed_arg: str) -> tuple[Path, str]:
    """
    Resolves seed_arg to a folder path and display name.
    Accepts either the folder name ("test 5") or display name ("MLK").
    """
    skip = {"violations", "post_mortem"}

    # Try direct folder name match
    candidate = data_dir / seed_arg
    if candidate.is_dir() and seed_arg not in skip:
        meta = SEED_METADATA.get(seed_arg, {})
        return candidate, meta.get("name", seed_arg)

    # Try display name match
    folder = NAME_TO_FOLDER.get(seed_arg.lower())
    if folder:
        candidate = data_dir / folder
        if candidate.is_dir():
            return candidate, seed_arg

    # Fuzzy: check if any folder's display name contains seed_arg
    for folder_name, meta in SEED_METADATA.items():
        if seed_arg.lower() in meta["name"].lower():
            candidate = data_dir / folder_name
            if candidate.is_dir():
                return candidate, meta["name"]

    raise ValueError(
        f"Could not find seed '{seed_arg}' in {data_dir}. "
        f"Use --list to see available seeds."
    )


def list_seeds(data_dir: Path):
    skip = {"violations", "post_mortem"}
    print(f"\nAvailable seeds in {data_dir}:\n")
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name not in skip:
            meta = SEED_METADATA.get(d.name, {})
            name = meta.get("name", d.name)
            chains = count_chains(d)
            print(f"  {d.name:12s}  →  {name:24s}  ({chains} chains)")
    print()


def count_chains(seed_dir: Path) -> int:
    count = 0
    for prompt_dir in seed_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
        for chain_dir in prompt_dir.iterdir():
            if chain_dir.is_dir() and chain_dir.name.startswith("chain_"):
                count += 1
    return count


def collect_images(seed_dir: Path) -> tuple[Path | None, list[list[Path]]]:
    """
    Discovers the seed image and all generated images, organised by chain.

    Returns:
      seed_path: Path to iter_00_seed.jpg (from any chain, they're identical)
      chains: list of lists, each inner list = generated images for one chain
              sorted by iteration number

    The chains list is sorted by chain number for consistent ordering.
    """
    seed_path = None
    chains: dict[int, list[tuple[int, Path]]] = {}

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

            # Grab seed from first chain that has it
            seed_candidate = chain_dir / "iter_00_seed.jpg"
            if seed_candidate.exists() and seed_path is None:
                seed_path = seed_candidate

            # Collect generated images
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

            if generated:
                chains[chain_num] = sorted(generated, key=lambda x: x[0])

    # Sort chains by chain number and strip the iteration numbers
    sorted_chains = [
        [path for _, path in chains[k]]
        for k in sorted(chains.keys())
    ]

    return seed_path, sorted_chains


def build_spiral_order(seed_path: Path, chains: list[list[Path]]) -> list[Path]:
    """
    Builds the spiral ordering:
      [seed, chain1_iter1, chain2_iter1, ..., chainN_iter1,
             chain1_iter2, chain2_iter2, ..., chainN_iter2, ...]

    Skips None entries (chains that terminated early and didn't reach that iteration).
    """
    if seed_path is None:
        raise ValueError("No seed image found.")

    images = [seed_path]

    # Find max iteration depth across all chains
    max_iters = max(len(c) for c in chains) if chains else 0

    for iter_idx in range(max_iters):
        for chain in chains:
            if iter_idx < len(chain):
                images.append(chain[iter_idx])

    return images


# ================================
# IMAGE PROCESSING
# ================================

def crop_to_circle(img: Image.Image, size: int) -> Image.Image:
    """Crops to circle of diameter size. Flattens onto black first so any
    light source background doesn't show through as a gap inside the circle."""
    # Flatten onto black so source background doesn't bleed through
    bg  = Image.new("RGB", img.size, (0, 0, 0))
    bg.paste(img.convert("RGB"))
    img = bg

    w, h    = img.size
    min_dim = min(w, h)
    left    = (w - min_dim) // 2
    top     = (h - min_dim) // 2
    img     = img.crop((left, top, left + min_dim, top + min_dim))
    img     = img.resize((size, size), Image.LANCZOS).convert("RGBA")
    # Anti-aliased circle mask
    scale    = 4
    big_mask = Image.new("L", (size * scale, size * scale), 0)
    ImageDraw.Draw(big_mask).ellipse((0, 0, size * scale - 1, size * scale - 1), fill=255)
    mask     = big_mask.resize((size, size), Image.LANCZOS)
    result   = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(img, mask=mask)
    return result


def crop_to_square(img: Image.Image, size: int) -> Image.Image:
    """Crops to square of size pixels."""
    w, h    = img.size
    min_dim = min(w, h)
    left    = (w - min_dim) // 2
    top     = (h - min_dim) // 2
    img     = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((size, size), Image.LANCZOS).convert("RGBA")


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """
    Resizes image so its longest dimension = size, preserving aspect ratio.
    Returns RGBA image with full corners intact — no cropping.
    """
    w, h     = img.size
    scale    = size / max(w, h)
    new_w    = int(w * scale)
    new_h    = int(h * scale)
    img      = img.resize((new_w, new_h), Image.LANCZOS)
    return img.convert("RGBA")


def rotate_image(img: Image.Image, angle_deg: float) -> Image.Image:
    """
    Rotates image by angle_deg with expand=True so corners are never clipped.
    Transparent background outside the rotated image.
    """
    return img.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)


def add_glow(img: Image.Image, radius: int = 3, intensity: float = 0.4) -> Image.Image:
    """Adds a subtle white glow around the circular image edge."""
    glow = img.filter(ImageFilter.GaussianBlur(radius))
    # Brighten the glow
    glow_arr = np.array(glow, dtype=np.float32)
    glow_arr[:, :, 3] = glow_arr[:, :, 3] * intensity
    glow = Image.fromarray(glow_arr.astype(np.uint8))

    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, img)
    return result


# ================================
# SPIRAL LAYOUT
# ================================

def compute_sunflower_layout(
    n: int,
    canvas_size: int,
    gap: float = CELL_GAP,
    margin: float = 0.90,
) -> tuple[list[tuple[float, float]], int]:
    """
    Computes tightly-packed Fermat spiral (sunflower) positions and uniform
    thumbnail diameter so that adjacent images just touch with a small gap.

    The spread constant c is derived from the canvas size and n so that
    the outermost point sits within margin * canvas_radius, and the cell
    diameter is set so adjacent cells (which are sqrt(i) apart radially)
    pack together without overlap.

    Returns (positions, diameter) where:
      positions: list of (x, y) canvas coordinates for each image
      diameter:  uniform thumbnail size in pixels for all images
    """
    cx = cy = canvas_size / 2
    max_r = canvas_size / 2 * margin

    # Spread constant: outermost point (index n-1) lands at max_r
    # r_max = c * sqrt(n - 1)  =>  c = max_r / sqrt(n - 1)
    c = max_r / math.sqrt(max(n - 1, 1))

    # Cell diameter: in the sunflower pattern the average nearest-neighbour
    # distance for point i is approximately c * (sqrt(i+1) - sqrt(i)).
    # We use the mean over the whole spiral as the cell size.
    # A simpler closed form: diameter ≈ 2 * c * (1 / (2*sqrt(n))) * sqrt(n)
    # which simplifies to c. But empirically c * 0.95 / sqrt(1) is too big.
    # Best fit: diameter = c * sqrt(2) * (1 - gap)
    diameter = int(c * math.sqrt(2) * (1 - gap))
    diameter = max(diameter, 4)

    positions = []
    for i in range(n):
        if i == 0:
            positions.append((cx, cy))
        else:
            r     = c * math.sqrt(i)
            theta = i * GOLDEN_ANGLE
            x     = cx + r * math.cos(theta)
            y     = cy + r * math.sin(theta)
            positions.append((x, y))

    return positions, diameter


# ================================
# RENDER
# ================================

def render_spiral(
    images: list[Path],
    seed_name: str,
    output_path: Path,
    canvas_size: int = CANVAS_SIZE,
    shape: str = SHAPE,
    gap: float = CELL_GAP,
    seed_boost: float = SEED_BOOST,
    rotate: bool = ROTATE,
    show_label: bool = True,
):
    """
    Renders the tightly-packed golden ratio sunflower spiral visualization.

    shape:      "full" | "square" | "circle"
    gap:        spacing between cells as fraction of cell size (0=touching, negative=overlap)
    seed_boost: size multiplier for the center seed image
    rotate:     whether to rotate each image to follow its spiral angle
    """
    n = len(images)
    shape = shape.lower()

    print(f"\nRendering spiral for: {seed_name}")
    print(f"  Images:      {n}")
    print(f"  Canvas:      {canvas_size}×{canvas_size}px")
    print(f"  Shape:       {shape}")
    print(f"  Gap:         {gap:+.0%}")
    print(f"  Rotate:      {rotate}")

    # Compute tight sunflower layout — all cells same size
    positions, diameter = compute_sunflower_layout(n, canvas_size, gap=gap)
    sizes = [diameter] * n

    # Seed image is slightly larger to stand out at center
    seed_size = min(int(diameter * seed_boost), canvas_size // 6)
    sizes[0]  = seed_size

    print(f"  Cell size:   {diameter}px")
    print(f"  Seed size:   {seed_size}px")

    # Create canvas
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 255))

    # Render images from outside in so center is on top
    render_order = list(range(n - 1, -1, -1))

    for idx in render_order:
        img_path = images[idx]
        size     = sizes[idx]
        cx, cy   = positions[idx]

        if idx % 50 == 0:
            print(f"  Compositing image {idx + 1}/{n}...", end="\r")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n  WARN: could not open {img_path.name}: {e}")
            continue

        # Apply shape
        if shape == "circle":
            thumb = crop_to_circle(img, size)
        elif shape == "square":
            thumb = crop_to_square(img, size)
        else:
            thumb = resize_image(img, size)

        # Rotate to follow the spiral angle
        if rotate:
            angle_deg = 0.0 if idx == 0 else math.degrees(idx * GOLDEN_ANGLE)
            thumb = rotate_image(thumb, angle_deg)

        # Paste centered on spiral position
        tw, th  = thumb.size
        paste_x = int(cx - tw / 2)
        paste_y = int(cy - th / 2)

        # Clip to canvas bounds
        if paste_x < -tw or paste_y < -th:
            continue
        if paste_x > canvas_size or paste_y > canvas_size:
            continue

        canvas.alpha_composite(thumb, dest=(paste_x, paste_y))

    print(f"\n  Compositing complete.")

    # Label
    if show_label:
        from PIL import ImageFont
        label_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(label_canvas)

        font_size = max(40, canvas_size // 80)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size // 2)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size // 2)
            except Exception:
                font = ImageFont.load_default()
                small_font = font

        # Title bottom-left
        margin = canvas_size // 40
        draw.text(
            (margin, canvas_size - margin - font_size * 2),
            seed_name,
            font=font,
            fill=(255, 255, 255, 200),
        )
        draw.text(
            (margin, canvas_size - margin - font_size),
            f"Telephone Game  ·  {n} images  ·  Golden ratio spiral",
            font=small_font,
            fill=(180, 180, 180, 160),
        )

        canvas = Image.alpha_composite(canvas, label_canvas)

    # Convert to RGB and save
    final = canvas.convert("RGB")
    final.save(str(output_path), quality=95, dpi=(300, 300))
    print(f"  Saved: {output_path}")
    print(f"  Size:  {output_path.stat().st_size / 1024 / 1024:.1f} MB")


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Golden ratio spiral visualization of telephone game image chains."
    )
    parser.add_argument("--data-dir",  default="output",
                        help="Path to telephone.py output dir (default: output)")
    parser.add_argument("--seed",      default=None,
                        help="Seed — folder name ('test 5') or display name ('MLK')")
    parser.add_argument("--output",    default=None,
                        help="Output file path (default: spiral_<seedname>.jpg)")
    parser.add_argument("--size",      type=int,   default=None,
                        help=f"Canvas size in pixels, square (default from config: {CANVAS_SIZE})")
    parser.add_argument("--shape",     default=None,
                        choices=["full", "square", "circle"],
                        help=f"Image shape: full / square / circle (default from config: {SHAPE})")
    parser.add_argument("--gap",       type=float, default=None,
                        help=f"Gap between cells as fraction of cell size, "
                             f"e.g. 0.0=touching 0.1=10%% gap -0.05=overlap (default from config: {CELL_GAP})")
    parser.add_argument("--seed-boost", type=float, default=None,
                        help=f"Center seed size multiplier vs outer cells (default from config: {SEED_BOOST})")
    parser.add_argument("--rotate", default=None, action="store_true",
                        help="Force rotation on (overrides config ROTATE=False)")
    parser.add_argument("--no-rotate", default=None, action="store_true",
                        help="Force rotation off (overrides config ROTATE=True)")
    parser.add_argument("--no-label",  action="store_true",
                        help="Omit title label")
    parser.add_argument("--list",      action="store_true",
                        help="List available seeds and exit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if args.list:
        list_seeds(data_dir)
        sys.exit(0)

    if not args.seed:
        print("ERROR: Specify a seed with --seed, or use --list to see options.")
        sys.exit(1)

    # Resolve seed
    try:
        seed_dir, seed_name = find_seed_folder(data_dir, args.seed)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Seed: {seed_name} ({seed_dir})")

    # Collect images
    seed_path, chains = collect_images(seed_dir)

    if seed_path is None:
        print("ERROR: No seed image found.")
        sys.exit(1)

    n_chains = len(chains)
    n_images = sum(len(c) for c in chains)
    print(f"Found: {n_chains} chains, {n_images} generated images")
    print(f"Total spiral images: {1 + n_images} (seed + generated)")

    # Build ordering
    ordered = build_spiral_order(seed_path, chains)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe = seed_name.lower().replace(" ", "_").replace(".", "")
        out_path = Path(f"spirals/spiral_{safe}.jpg")

    # Render — CLI flags override config constants; if not passed, config wins
    render_spiral(
        images=ordered,
        seed_name=seed_name,
        output_path=out_path,
        canvas_size=args.size       if args.size       is not None else CANVAS_SIZE,
        shape=args.shape            if args.shape      is not None else SHAPE,
        gap=args.gap                if args.gap        is not None else CELL_GAP,
        seed_boost=args.seed_boost  if args.seed_boost is not None else SEED_BOOST,
        rotate=(ROTATE if args.no_rotate is None and args.rotate is None
                else False if args.no_rotate else True),
        show_label=not args.no_label,
    )


if __name__ == "__main__":
    main()