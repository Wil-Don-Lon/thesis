# Multimodal AI Bias Detection via Iterative Feedback Loops
### Honors Thesis — William Donnell-Lonon, University of Arkansas (2026)

> *What happens when you ask an AI to describe its own output, over and over again — without ever changing the prompt?*

This research investigates emergent representational bias in generative AI systems by constructing **regurgitative multimodal feedback loops**: a seed image is captioned by a vision-language model, that caption is used to generate a new image, and the process repeats for up to 75 iterations. The drift you observe across iterations — in race, gender, age, setting, and identity — is the bias, surfacing without any explicit instruction.



## Repository Structure

```
telephone.py          # Core pipeline: runs the iterative feedback loop
clip_analysis.py      # CLIP similarity scoring — measures visual drift from seed
face_analysis.py      # ArcFace identity scoring — measures face identity preservation
semantic_analysis.py  # NLP analysis of caption text — drift, sentiment, refusals
spiral_viz.py         # Golden ratio spiral visualization of image chains
seed_images/          # Input seed images (15 high-publicity public figures)
output/               # Generated image chains (not committed — too large)
```

---

## How It Works

**telephone.py** is the core engine. It implements a configurable multi-chain telephone game:

1. A seed image is passed to a vision-language model (GPT), which generates an objective caption
2. That caption is passed to an image generation model (GPT Image), which generates a new image
3. Steps 1-2 repeat for N iterations across M independent chains per seed

The pipeline includes production-grade infrastructure: checkpoint/resume, atomic image writes, policy violation logging, automated post-mortem analysis, and a live terminal dashboard.

**The four analysis scripts** measure different dimensions of drift across the generated chains:

| Script | Method | What it measures |
|--------|--------|-----------------|
| `clip_analysis.py` | OpenCLIP ViT-B-32 cosine similarity | Overall visual similarity to seed |
| `face_analysis.py` | ArcFace (insightface) embedding similarity | Face identity preservation |
| `semantic_analysis.py` | Sentence-BERT + lexicon analysis | Caption language drift, sentiment, refusals |
| `spiral_viz.py` | Fermat/sunflower spiral layout | Visual summary of a full chain |

---

## Setup

```bash
pip install openai pillow requests

# For analysis scripts (install only what you need)
pip install open_clip_torch torch torchvision        # clip_analysis.py
pip install insightface onnxruntime                  # face_analysis.py
pip install sentence-transformers nltk textblob      # semantic_analysis.py
pip install numpy pandas matplotlib scipy            # all analysis scripts
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Usage

### Run the feedback loop

```bash
# Place seed images in seed_images/
python telephone.py

# Resume from checkpoint after interruption
python telephone.py checkpoint

# Custom parameters
python telephone.py --iterations 75 --chains 4
```

**Controls:** `Ctrl+C` once pauses after the current iteration (progress saved). `Ctrl+C` twice force-quits.

### Run analysis

```bash
# CLIP visual similarity analysis
python clip_analysis.py --data-dir output

# Face identity analysis
python face_analysis.py --data-dir output

# Semantic/caption analysis
python semantic_analysis.py --data-dir output --epoch-size 3

# Generate spiral visualization for a specific seed
python spiral_viz.py --seed "Queen Elizabeth" --shape circle
python spiral_viz.py --list   # see available seeds
```

---

## Configuration

Key parameters in `telephone.py`:

```python
ITERATIONS       = 75   # iterations per chain
CHAINS_PER_SEED  = 4    # independent chains per seed image
CAPTION_MODEL    = "gpt-5.2-chat-latest"
IMAGE_MODEL      = "gpt-image-1.5"
IMAGE_QUALITY    = "medium"   # low | medium | high
```

The captioning prompt is configurable via the `PROMPTS` dict. The default (`"objective"`) instructs the model to describe the image as factually as possible — no artistic framing, no softening.

---

## Output Structure

```
output/
  <seed_name>/
    <prompt_type>/
      chain_01/
        iter_00_seed.jpg          # original seed image
        iter_01_caption.txt       # caption generated at iteration 1
        iter_01_generated.jpg     # image generated from that caption
        iter_02_caption.txt
        iter_02_generated.jpg
        ...
        log.json                  # full chain log with per-iteration data
        summary.txt               # human-readable chain summary
        post_mortem/              # policy violation investigation reports (if any)
  master_log.json                 # aggregate log across all chains and seeds
  violations/                     # chains terminated by content policy blocks
  post_mortem/                    # global post-mortem reports
```

---

## Dataset

15 seed images of high-publicity public figures across political and historical categories. Seeds were selected to probe how the model handles figures with varying degrees of visual recognizability, racial diversity, gender, and historical context.

The full image dataset is not committed to this repository due to size. Analysis outputs (CSVs, figures) are in their respective `*_output/` directories.

---

## Key Findings

- Visual similarity (CLIP) decays measurably across iterations — the model systematically drifts from the original representation even under an identical, objective prompt
- Face identity (ArcFace) degrades faster than overall visual similarity, suggesting the model deprioritizes specific identity features while preserving general scene composition
- Caption language shifts in sentiment, vocabulary, and subject framing across epochs, with some seeds showing insertion of entirely new subject matter by iteration 15+
- Refusal language (hedging, identity evasion) appears inconsistently and correlates with policy block events
- Policy blocks are not uniformly distributed — certain seeds trigger significantly more content policy violations than others, which is itself a finding about model sensitivity

---

## License

Copyright William Donnell-Lonon, 2026. All rights reserved.

Use for further research or AI model development is permitted under the following conditions:
1. **Attribution** — any use must credit the original author and link to this repository
2. **Non-commercial** — no commercial use without explicit permission
3. **Open derivatives** — any publication using derivative work must release modified code publicly under the same terms
4. **Citation** — academic use must cite the original thesis

---

## Citation

```
Donnell-Lonon, W. (2026). Multimodal AI Bias Detection via Iterative Feedback Loops.
Honors Thesis, University of Arkansas.
```
