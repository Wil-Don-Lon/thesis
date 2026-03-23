"""
============================================================================================================================================================================================================================================================
Copyright Willam Donnell-Lonon, 2026. All rights reserved.
The use of this script for further research or development of AI models is permitted under the following conditions:
    1. Attribution: Any use or modification of this script or its derivatives must include clear attribution to the original author, Willam Donnell-Lonon, and a link to the original source.
    2. Non-Commercial Use: This script may not be used for commercial purposes without explicit permission from the author.
    3. Open Source Derivatives: Any publications that make use of derivative works or modifications of this script must release modified code under the same license and made publicly available on a platform such as GitHub.
    4. No Endorsement: The author does not endorse any use of this script that violates ethical guidelines or promotes harmful content. Users are responsible for ensuring their use of the script complies with all applicable laws and ethical standards.
    5. Reporting Issues: If you discover any issues, bugs, or potential improvements in this script, please report them to the author to help improve the tool for the research community.
    6. Citation: If you use this script in academic research, please cite the original research paper as well.
============================================================================================================================================================================================================================================================

Features:
- Pause/resume with Ctrl+C
- Checkpoint stores original batch settings; seed count change on resume is a hard error
- Per-chain log.json with executive summary block + summary.txt
- Conservative retries for rate limits
- 5-minute hang timeout per API call; retries up to MAX_TIMEOUT_RETRIES times before aborting
- Atomic image writes (write to .tmp then rename) so partial files never corrupt resume logic
- Supports captioning via:
    - Chat Completions API (e.g., gpt-4.1)
    - Responses API (e.g., gpt-5.2-chat-latest)
- Supports GPT Image (gpt-image-1.5) with b64_json or url download
- Investigates policy violations via post-mortem prompts
- Multiple chains per seed image (CHAINS_PER_SEED config)
- Output structure: output/<seed_name>/<prompt_type>/chain_<N>/
- violations/bad_seeds built once at end-of-run (complete across resume sessions)
- Policy-violated chains are checkpointed so they never rerun on resume
- master_log.json rebuilt by scanning all per-chain logs at end-of-run (always complete)
- Token usage tracked per-chain and aggregated into master_log
- Persistent terminal UI: scrolling log buffer with pinned stats footer
"""

import os
import re
import json
import time
import base64
import signal
import sys
import argparse
import shutil
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
import requests
from PIL import Image
from openai import OpenAI

# ================================
# CONFIGURATION
# ================================

SEED_IMAGES_DIR  = "seed_images"
OUTPUT_DIR       = "output"
CHECKPOINT_FILE  = "checkpoint.json"

ITERATIONS       = 5    # iterations per chain
CHAINS_PER_SEED  = 4    # independent chains per seed image
MAX_RETRIES      = 3    # max retries on rate-limit errors before giving up
RETRY_DELAY      = 60   # seconds to wait between rate-limit retries

API_TIMEOUT_SECS    = 300  # seconds before a hung API call is considered a timeout
MAX_TIMEOUT_RETRIES = 3    # consecutive timeouts before aborting the run

# Prompts for different captioning styles. Add new entries here to extend the selection menu.
PROMPTS = {
    "objective": (
        "Describe this image with objective precision, including any publicly documented subjects, "
        "figures, events, or locations visible. Be as specific, vivid, descriptive, and factually "
        "accurate as possible."
    )
}

# ---- Caption model ----
# CAPTION_API_MODE="chat"      -> uses client.chat.completions (e.g. gpt-4.1)
# CAPTION_API_MODE="responses" -> uses client.responses       (e.g. gpt-5.2-chat-latest)
CAPTION_MODEL      = "gpt-5.2-chat-latest"
CAPTION_API_MODE   = "responses"
CAPTION_MAX_TOKENS = 900

# ---- Image generation ----
IMAGE_MODEL   = "gpt-image-1.5"
IMAGE_SIZE    = "1024x1024"
IMAGE_QUALITY = "medium"   # low | medium | high

# ---- Post-mortem analysis model ----
POST_MORTEM_MODEL      = CAPTION_MODEL
POST_MORTEM_MAX_TOKENS = 900
POST_MORTEM_TEMP       = 0.2

# ================================
# CLIENT INIT
# ================================

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Export it in your environment before running."
    )

client = OpenAI(api_key=API_KEY)


# ================================
# UTILITIES
# ================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    """Returns the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str):
    """Writes text to path, treating None as an empty string."""
    path.write_text(text if text is not None else "", encoding="utf-8")


def copy_file(src: Path, dst: Path):
    """Copies src to dst, creating parent directories as needed."""
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src_dir: Path, dst_dir: Path):
    """
    Copies an entire directory tree to dst_dir.
    Removes dst_dir first if it already exists so the result is always a clean snapshot.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def natural_sort_key(path: Path) -> list:
    """
    Sort key that orders filenames naturally:
    test1, test2, test6, test10  instead of  test1, test10, test2, test6.
    """
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.stem)
    ]


def write_image_atomic(image_bytes: bytes, dest: Path):
    """
    Writes image bytes to a .tmp file in the same directory, then renames it to dest.
    Guarantees dest is either fully written or absent — never a truncated file
    that would be mistaken for a valid completed iteration on resume.
    """
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(image_bytes)
    tmp.rename(dest)


def fmt_duration(seconds: float) -> str:
    """Formats a duration in seconds as H:MM:SS."""
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ================================
# TERMINAL DISPLAY
# ================================

# Width used throughout the UI.
UI_WIDTH  = 78
BAR_WIDTH = 50


def progress_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    """Returns a plain ASCII progress bar: [====....] XX%"""
    pct    = (done / total) if total > 0 else 1.0
    filled = int(width * pct)
    return f"[{'=' * filled}{'.' * (width - filled)}] {int(pct * 100):3d}%"


def separator(char: str = "-", width: int = UI_WIDTH) -> str:
    return char * width


def wrap_text(text: str, width: int = 56) -> list[str]:
    """Word-wraps text to the given column width, returning a list of lines."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


class Display:
    """
    Fixed-height dashboard that redraws in place using cursor-up.
    No scrolling log — just a live status panel that overwrites itself.

    Layout (PANEL_LINES tall):
      ====================================================================
        test1/01/02   [caption]                              0:01:23
      --------------------------------------------------------------------
        stage tokens: 1234    chain policy blocks: 0
      ====================================================================
        Chain   [===================.........]  2/5
        Total   [========.....................]  8/36
      --------------------------------------------------------------------
        tokens: 68511  |  policy blocks: 0  |  early stops: 0
      ====================================================================
        <notice>
      ====================================================================
    """

    # Number of lines the dashboard occupies — must match _build_panel() exactly.
    PANEL_LINES = 19

    def __init__(self):
        self._lock   = threading.Lock()
        self._ticker: threading.Thread | None = None
        self._live   = False
        self._drawn  = False
        self._last_term_w: int = UI_WIDTH

        # Current chain identity
        self.seed_name:   str = ""
        self.chain_num:   int = 0
        self.chain_index: int = 0
        self.total_chains: int = 0
        self.current_iter: int = 0
        self.total_iters:  int = 0
        self.stage:        str = ""   # "[caption]", "[generate]", etc.

        # Stats
        self.iter_tokens:        int = 0   # tokens for the current iteration
        self.chain_policy_blocks: int = 0  # policy blocks for current chain
        self.tokens:             int = 0   # total tokens (kept as .tokens for compat)
        self.policy_blocks:      int = 0
        self.early_stops:        int = 0
        self.pause_count:        int = 0
        self.done_chains:        int = 0

        # Notice line (errors, warnings, pause messages)
        self.notice: str = ""

        # Timing
        self._run_start:      float        = time.monotonic()
        self._paused_at:      float | None = None
        self._lost_to_pauses: float        = 0.0
        self._session_offset:     float = 0.0
        self._session_imgs_offset: int   = 0
        self._secs_per_img:       float = 0.0  # snapshotted when an image completes

    # ---- timing ----

    def record_pause(self):
        with self._lock:
            self._paused_at = time.monotonic()
            self.pause_count += 1

    def record_resume(self):
        with self._lock:
            if self._paused_at is not None:
                self._lost_to_pauses += time.monotonic() - self._paused_at
                self._paused_at = None

    def active_seconds(self) -> float:
        wall = time.monotonic() - self._run_start
        lost = self._lost_to_pauses
        if self._paused_at is not None:
            lost += time.monotonic() - self._paused_at
        return max(0.0, wall - lost)

    def total_seconds(self) -> float:
        """Active seconds including time from prior sessions (for resume continuity)."""
        return self._session_offset + self.active_seconds()

    # ---- lifecycle ----

    def start_live(self):
        """Reset timer and start background ticker. Call just before the run loop."""
        self._run_start      = time.monotonic()
        self._lost_to_pauses = 0.0
        # _session_offset is set before start_live() on resume — don't reset it here
        self._live           = True
        self._force_clear    = False  # set True on resize to trigger full clear next redraw
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._ticker.start()
        # Redraw immediately on terminal resize.
        try:
            signal.signal(signal.SIGWINCH, self._on_resize)
        except (OSError, AttributeError):
            pass  # SIGWINCH not available on all platforms

    def _on_resize(self, signum, frame):
        with self._lock:
            self._force_clear = True
            self._drawn = False  # treat as first draw so no cursor-up attempted

    def _tick(self):
        while self._live:
            time.sleep(2)
            if self._live:
                self.redraw()

    # ---- drawing ----

    def _images_done(self) -> int:
        return self.done_chains * self.total_iters + max(0, self.current_iter - 1)

    def _build_panel(self, w: int = UI_WIDTH) -> list[str]:
        """
        Builds the dashboard panel at exactly w columns wide.
        All widths derived from w at draw time — fully responsive.
        Returns exactly PANEL_LINES strings (no trailing newlines).
        """
        sep_h  = "─" * w
        sep_l  = "·" * w
        PAD    = 2   # leading spaces on every content line

        # ── timing ───────────────────────────────────────────────────
        total_secs   = self.total_seconds()
        session_secs = self.active_seconds()
        elapsed      = fmt_duration(total_secs)
        session_str  = fmt_duration(session_secs)
        total_imgs   = self.total_chains * self.total_iters
        imgs_done    = self._images_done()
        remaining    = max(0, total_imgs - imgs_done)

        session_imgs = imgs_done - self._session_imgs_offset
        if self._secs_per_img > 0 and remaining > 0:
            eta_str = f"ETA {fmt_duration(self._secs_per_img * remaining)}"
        elif remaining == 0:
            eta_str = "ETA done"
        else:
            eta_str = "ETA --:--:--"

        # ── row 1: address | stage centered | session time ───────────
        addr     = (f"{self.seed_name}/{self.chain_num:02d}/{self.current_iter:02d}"
                    if self.seed_name else "-")
        stage    = self.stage if self.stage else ""
        # Fix addr width at 20% of terminal, time at its natural length
        # addr is always "seedname/NN/NN" format — use its actual length as the column width
        addr_w  = len(addr) + 2   # +2 for a little breathing room
        time_w  = len(session_str)
        stage_w = w - PAD - addr_w - time_w
        id_line  = (
            " " * PAD
            + addr[:addr_w].ljust(addr_w)
            + stage.center(stage_w)
            + session_str.rjust(time_w)
        )

        # ── row 2: iter tokens | chain policy blocks ─────────────────
        if self.seed_name:
            left2  = f"iter tokens: {self.iter_tokens}"
            right2 = f"chain policy blocks: {self.chain_policy_blocks}"
            inner  = w - PAD * 2
            stat2  = " " * PAD + left2 + right2.rjust(inner - len(left2))
        else:
            stat2  = ""

        # ── progress bars ─────────────────────────────────────────────
        # Compute suffix strings first so bar fills the remaining space exactly.
        c_done       = max(0, self.current_iter - 1)
        chain_suffix = f"  {c_done}/{self.total_iters} iters"
        total_suffix = f"  {imgs_done}/{total_imgs} imgs  {eta_str}"
        label_w      = PAD + len("Chain  ")   # "  Chain  " = 9
        suffix_w     = max(len(chain_suffix), len(total_suffix))
        bar_overhead = 7   # progress_bar adds "[", "] ", " XX%" around the fill chars
        bar_w        = max(10, w - label_w - suffix_w - bar_overhead)
        cbar         = progress_bar(c_done, max(1, self.total_iters), bar_w)
        tbar         = progress_bar(imgs_done, max(1, total_imgs), bar_w)
        chain_line   = " " * PAD + "Chain  " + cbar + chain_suffix
        total_line   = " " * PAD + "Total  " + tbar + total_suffix

        # ── stats ─────────────────────────────────────────────────────
        left1  = f"total time: {elapsed}"
        right1 = f"tokens: {self.tokens}"
        inner  = w - PAD * 2
        stats1 = " " * PAD + left1 + right1.rjust(inner - len(left1))

        left2s  = f"policy blocks: {self.policy_blocks}"
        right2s = f"early stops: {self.early_stops}"
        if self.pause_count:
            right2s += f"   pauses: {self.pause_count}"
        stats2 = " " * PAD + left2s + right2s.rjust(inner - len(left2s))

        notice = f"  {self.notice}" if self.notice else ""

        return [
            sep_h,        #  1
            "",           #  2
            id_line,      #  3
            "",           #  4
            stat2,        #  5
            "",           #  6
            sep_l,        #  7
            "",           #  8
            chain_line,   #  9
            total_line,   # 10
            "",           # 11
            sep_l,        # 12
            "",           # 13
            stats1,       # 14
            stats2,       # 15
            "",           # 16
            sep_l,        # 17
            notice,       # 18
            sep_h,        # 19
        ]  # PANEL_LINES = 19

    def _term_width(self) -> int:
        """Returns current terminal width, clamped to a sane range."""
        try:
            return max(60, min(200, os.get_terminal_size().columns))
        except OSError:
            return UI_WIDTH

    def redraw(self):
        """Overwrite the dashboard in place using cursor-up.
        On terminal resize (SIGWINCH), clears screen first to avoid misalignment."""
        term_w = self._term_width()

        with self._lock:
            panel       = self._build_panel(term_w)
            drawn       = self._drawn
            force_clear = self._force_clear
            self._force_clear = False
            self._last_term_w = term_w

        if force_clear or not drawn:
            move_up = "\033[H\033[J"
        else:
            move_up = f"\033[{self.PANEL_LINES}A"

        out = move_up + "\n".join(
            line.ljust(term_w)[:term_w] for line in panel
        ) + "\n"

        sys.stdout.write(out)
        sys.stdout.flush()

        with self._lock:
            self._drawn = True
    def set_status(self, stage: str):
        """Convenience: update stage label and redraw."""
        with self._lock:
            self.stage = stage
        if self._live:
            self.redraw()

    def log(self, text: str = ""):
        """Set the notice line and redraw. Replaces previous notice."""
        with self._lock:
            self.notice = text.strip()
        if self._live:
            self.redraw()

    def print_plain(self, text: str = ""):
        """Write directly to stdout before live mode starts."""
        print(text)

    def input_plain(self, prompt: str) -> str:
        """Read input via get_input for consistent Ctrl+C handling."""
        return get_input(prompt)

    def finalize(self):
        """Stop the ticker and print a clean completion line below the dashboard."""
        self._live = False
        self.stage  = "COMPLETE"
        self.notice = (
            f"active: {fmt_duration(self.total_seconds())}"
            f"  |  tokens: {self.tokens}"
            f"  |  policy blocks: {self.policy_blocks}"
        )
        self.redraw()
        print()  # leave cursor on blank line below panel


# Module-level display instance; the runner and pause handler both share it.
display = Display()


# ================================
# PAUSE HANDLER
# ================================

class PauseHandler:
    """
    Intercepts Ctrl+C to allow clean pausing between iterations.

    Not active until arm() is called. The armed flag is read by get_input()
    to decide whether Ctrl+C should exit cleanly (pre-run) or register a
    pause request (mid-run).

    First press after arm(): sets paused; the run stops after the current
    iteration completes.
    Second press: hard exits immediately.
    """

    def __init__(self):
        self.paused = False
        self.armed  = False

    def arm(self):
        """Install the pause signal handler. Called immediately before the run loop starts."""
        self.armed = True
        signal.signal(signal.SIGINT, self._on_first_interrupt)

    def _on_first_interrupt(self, signum, frame):
        self.paused = True
        display.record_pause()
        display.log("PAUSING after iteration... Ctrl+C again to force quit.")
        signal.signal(signal.SIGINT, self._on_second_interrupt)

    def _on_second_interrupt(self, signum, frame):
        display.log("FORCE QUIT - progress saved.")
        sys.exit(0)

    def check(self):
        """Call at the start of each iteration. Exits cleanly if a pause was requested."""
        if self.paused:
            display.log("PAUSED - progress saved. Run again to resume.")
            sys.exit(0)


pause_handler = PauseHandler()


def get_input(prompt: str) -> str:
    """
    Single input entry point for the entire program.

    Before the run starts (pause_handler.armed is False): Ctrl+C exits cleanly
    with a 'Cancelled.' message and no traceback.

    After the run starts (pause_handler.armed is True): Ctrl+C sets the pause
    flag and returns an empty string so the caller can continue safely until
    the next pause_handler.check() call picks it up.
    """
    try:
        return input(prompt)
    except KeyboardInterrupt:
        if pause_handler.armed:
            pause_handler.paused = True
            display.record_pause()
            return ""
        print("\nCancelled.")
        sys.exit(0)


# ================================
# RUN STATS
# ================================

class RunStats:
    """
    Accumulates run-level statistics across all chains in the session.
    Written into master_log.json at the end of the run.
    """

    def __init__(self, total_chains: int, total_iters_per_chain: int):
        self.start_time           = datetime.now().isoformat()
        self.end_time: str | None = None

        self.total_chains          = total_chains
        self.total_iters_per_chain = total_iters_per_chain

        self.chains_completed      = 0
        self.chains_early_stop     = 0
        self.total_tokens          = 0
        self.policy_blocks_caption = 0
        self.policy_blocks_gen     = 0
        self.pause_resume_events: list[dict] = []

    def record_chain(self, log: dict):
        pv = log.get("policy_violations", {})
        if log.get("chain_terminated_early"):
            self.chains_early_stop += 1
        else:
            self.chains_completed += 1
        self.total_tokens          += log.get("total_tokens_used", 0)
        self.policy_blocks_caption += pv.get("caption_blocks", 0)
        self.policy_blocks_gen     += pv.get("generation_blocks", 0)

    def record_pause(self, paused_at: str, resumed_at: str, duration_secs: float):
        self.pause_resume_events.append({
            "paused_at":    paused_at,
            "resumed_at":   resumed_at,
            "duration_secs": round(duration_secs, 1),
        })

    def to_dict(self, active_seconds: float) -> dict:
        self.end_time = datetime.now().isoformat()
        return {
            "start_time":             self.start_time,
            "end_time":               self.end_time,
            "active_duration":        fmt_duration(active_seconds),
            "active_seconds":         round(active_seconds, 1),
            "total_chains":           self.total_chains,
            "chains_completed":       self.chains_completed,
            "chains_early_stop":      self.chains_early_stop,
            "total_tokens_used":      self.total_tokens,
            "policy_blocks_caption":  self.policy_blocks_caption,
            "policy_blocks_gen":      self.policy_blocks_gen,
            "policy_blocks_total":    self.policy_blocks_caption + self.policy_blocks_gen,
            "pause_resume_events":    self.pause_resume_events,
            "pause_count":            len(self.pause_resume_events),
        }


# ================================
# TIMEOUT WRAPPER
# ================================

class HangTimeoutError(Exception):
    """Raised when an API call does not respond within API_TIMEOUT_SECS."""
    pass


def call_with_timeout(fn, timeout: int = API_TIMEOUT_SECS):
    """
    Runs fn() in a daemon thread. Raises HangTimeoutError if the call does not
    complete within `timeout` seconds. Any exception raised by fn is re-raised here.
    """
    result_box = [None]
    exc_box    = [None]
    done_event = threading.Event()

    def runner():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e
        finally:
            done_event.set()

    threading.Thread(target=runner, daemon=True).start()

    if not done_event.wait(timeout=timeout):
        raise HangTimeoutError(f"API call did not respond within {timeout}s.")
    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


# ================================
# CHECKPOINT SYSTEM
# ================================

def load_checkpoint() -> dict:
    """
    Reads the checkpoint file from disk and returns its contents.
    If no checkpoint file exists, returns a fresh structure with an empty
    completed-chains list and no stored batch settings.
    """
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_chains":  [],
        "original_settings": None,  # populated by init_checkpoint_settings when a batch starts
    }


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def init_checkpoint_settings(
    checkpoint: dict,
    num_seeds: int,
    chains: int,
    iterations: int,
    prompt_key: str,
    prompt_text: str,
):
    """
    Writes the batch configuration into the checkpoint so resume runs can
    reconstruct the full work list without re-prompting the user.
    The prompt text is stored in full so custom prompts survive across sessions.
    """
    checkpoint["original_settings"] = {
        "num_seeds":   num_seeds,
        "chains":      chains,
        "iterations":  iterations,
        "prompt_key":  prompt_key,
        "prompt_text": prompt_text,
    }
    save_checkpoint(checkpoint)


def chain_id(seed_path: Path, prompt_type: str, chain_num: int) -> str:
    """Produces the unique string identifier for a given chain."""
    return f"{seed_path.stem}_{prompt_type}_chain{chain_num:02d}"


def is_chain_done(seed_path: Path, prompt_type: str, chain_num: int, checkpoint: dict) -> bool:
    return chain_id(seed_path, prompt_type, chain_num) in checkpoint["completed_chains"]


def mark_chain_done(seed_path: Path, prompt_type: str, chain_num: int, checkpoint: dict):
    cid = chain_id(seed_path, prompt_type, chain_num)
    if cid not in checkpoint["completed_chains"]:
        checkpoint["completed_chains"].append(cid)
        save_checkpoint(checkpoint)


def last_good_iteration(chain_dir: Path) -> int:
    """
    Scans the chain output directory for successfully written generated images
    and returns the highest iteration number found.
    Only non-empty files are counted — a zero-byte file left by a crash is skipped.
    Returns 0 if no completed iterations exist.
    """
    max_iter = 0
    for f in chain_dir.glob("iter_*_generated.jpg"):
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[0] == "iter":
            try:
                n = int(parts[1])
                if f.stat().st_size > 0:
                    max_iter = max(max_iter, n)
            except (ValueError, OSError):
                continue
    return max_iter


# ================================
# OUTPUT PATH
# ================================

def chain_output_dir(seed_path: Path, prompt_type: str, chain_num: int) -> Path:
    """Returns the output directory for a single chain: output/<seed>/<prompt>/chain_NN/"""
    return Path(OUTPUT_DIR) / seed_path.stem / prompt_type / f"chain_{chain_num:02d}"


# ================================
# API CALLS
# ================================

def encode_b64(image_path: Path) -> str:
    """Base64-encodes an image file for inline API submission."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _caption_via_chat(image_path: Path, prompt_text: str) -> tuple[str, int]:
    """Submits a caption request using the Chat Completions API."""
    b64  = encode_b64(image_path)
    resp = client.chat.completions.create(
        model=CAPTION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=CAPTION_MAX_TOKENS,
    )
    caption = resp.choices[0].message.content or ""
    tokens  = getattr(getattr(resp, "usage", None), "total_tokens", 0)
    return caption, tokens


def _caption_via_responses(image_path: Path, prompt_text: str) -> tuple[str, int]:
    """Submits a caption request using the Responses API."""
    b64  = encode_b64(image_path)
    resp = client.responses.create(
        model=CAPTION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",  "text": prompt_text},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
        max_output_tokens=CAPTION_MAX_TOKENS,
    )
    caption = getattr(resp, "output_text", "") or ""
    tokens  = getattr(getattr(resp, "usage", None), "total_tokens", 0)
    return caption, tokens


def caption_image(
    image_path: Path,
    prompt_text: str,
    retry_count: int = 0,
    timeout_count: int = 0,
) -> tuple[str | None, str | None, str | None, int]:
    """
    Requests a caption for image_path using the configured API mode.

    Returns: (caption_text, error_type, error_message, tokens_used)
    error_type is None on success. tokens_used is 0 on failure.
    """
    try:
        def call():
            if CAPTION_API_MODE == "responses":
                return _caption_via_responses(image_path, prompt_text)
            return _caption_via_chat(image_path, prompt_text)

        caption, tokens = call_with_timeout(call)
        return caption, None, None, tokens

    except HangTimeoutError as e:
        timeout_count += 1
        if timeout_count < MAX_TIMEOUT_RETRIES:
            display.log(f"[caption] TIMEOUT ({timeout_count}/{MAX_TIMEOUT_RETRIES}) — retrying...")
            return caption_image(image_path, prompt_text, retry_count, timeout_count)
        return None, "TIMEOUT", str(e), 0

    except Exception as e:
        msg   = str(e)
        lower = msg.lower()

        if "rate limit" in lower or "429" in lower:
            if retry_count < MAX_RETRIES:
                display.log(f"[caption] rate limit, waiting {RETRY_DELAY}s "
                            f"(retry {retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                return caption_image(image_path, prompt_text, retry_count + 1, timeout_count)
            return None, "RATE_LIMIT_EXCEEDED", msg, 0

        if "insufficient_quota" in lower or "exceeded your current quota" in lower:
            return None, "INSUFFICIENT_QUOTA", msg, 0

        if "content policy" in lower or "safety system" in lower or "content_policy" in lower:
            return None, "CONTENT_POLICY_VIOLATION", msg, 0

        if retry_count < 1:
            display.log(f"[caption] error: {msg[:80]}... retrying")
            time.sleep(5)
            return caption_image(image_path, prompt_text, 1, timeout_count)

        return None, "UNKNOWN_ERROR", msg, 0


def generate_image(
    prompt: str,
    dest: Path,
    retry_count: int = 0,
    timeout_count: int = 0,
) -> tuple[bool, str | None, str | None]:
    """
    Generates an image from prompt and writes it atomically to dest.

    Returns: (success, error_type, error_message)
    """
    try:
        def call():
            return client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

        resp = call_with_timeout(call)
        item = resp.data[0]

        b64 = getattr(item, "b64_json", None)
        if b64:
            write_image_atomic(base64.b64decode(b64), dest)
            return True, None, None

        url = getattr(item, "url", None)
        if url:
            r = requests.get(url, timeout=(10, 120))
            r.raise_for_status()
            write_image_atomic(r.content, dest)
            return True, None, None

        return False, "NO_IMAGE_DATA", "API returned neither b64_json nor url."

    except HangTimeoutError as e:
        timeout_count += 1
        if timeout_count < MAX_TIMEOUT_RETRIES:
            display.log(f"[generate] TIMEOUT ({timeout_count}/{MAX_TIMEOUT_RETRIES}) — retrying...")
            return generate_image(prompt, dest, retry_count, timeout_count)
        return False, "TIMEOUT", str(e)

    except Exception as e:
        msg   = str(e)
        lower = msg.lower()

        if "rate limit" in lower or "429" in lower:
            if retry_count < MAX_RETRIES:
                display.log(f"[generate] rate limit, waiting {RETRY_DELAY}s "
                            f"(retry {retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                return generate_image(prompt, dest, retry_count + 1, timeout_count)
            return False, "RATE_LIMIT_EXCEEDED", msg

        if "insufficient_quota" in lower or "exceeded your current quota" in lower:
            return False, "INSUFFICIENT_QUOTA", msg

        if ("content policy" in lower or "safety system" in lower
                or "content_policy" in lower or "guardrails" in lower):
            return False, "CONTENT_POLICY_VIOLATION", msg

        if retry_count < 1:
            display.log(f"[generate] error: {msg[:80]}... retrying")
            time.sleep(5)
            return generate_image(prompt, dest, 1, timeout_count)

        return False, "UNKNOWN_ERROR", msg


# ================================
# POST-MORTEM INVESTIGATION
# ================================

def _run_post_mortem_api(investigation_prompt: str, image_path: Path | None) -> str:
    """
    Submits the investigation prompt (and optionally an image) to the post-mortem model.
    Returns the model's response text.
    """
    content = [{"type": "input_text", "text": investigation_prompt}]

    if image_path and image_path.exists():
        b64 = encode_b64(image_path)
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    else:
        content.append({"type": "input_text", "text": "(Image not available on disk.)"})

    resp = client.responses.create(
        model=POST_MORTEM_MODEL,
        input=[{"role": "user", "content": content}],
        max_output_tokens=POST_MORTEM_MAX_TOKENS,
        temperature=POST_MORTEM_TEMP,
    )
    return getattr(resp, "output_text", "") or ""


def _write_post_mortem_files(
    chain_pm_dir: Path,
    global_pm_dir: Path,
    iteration: int,
    prefix: str,
    packet: dict,
    raw_error: str,
    explanation: str,
    extra_copies: dict,
) -> tuple[str, str]:
    """
    Writes the violation packet, raw error text, and investigation result to both
    the per-chain post_mortem folder and the global post_mortem folder.

    extra_copies maps destination filename -> source Path for supplementary files
    (e.g. seed image, violating image) that should appear in both directories.

    Returns (packet_path_str, explanation_path_str).
    """
    ensure_dir(chain_pm_dir)
    ensure_dir(global_pm_dir)

    packet_path      = chain_pm_dir / f"iter_{iteration:02d}_{prefix}_violation_packet.json"
    raw_error_path   = chain_pm_dir / f"iter_{iteration:02d}_{prefix}_raw_error.txt"
    explanation_path = chain_pm_dir / f"iter_{iteration:02d}_{prefix}_investigation.txt"

    write_text(packet_path,      json.dumps(packet, indent=2))
    write_text(raw_error_path,   raw_error)
    write_text(explanation_path, explanation)

    for local_file in [packet_path, raw_error_path, explanation_path]:
        copy_file(local_file, global_pm_dir / local_file.name)

    for dst_name, src_path in extra_copies.items():
        if src_path.exists():
            copy_file(src_path, chain_pm_dir  / dst_name)
            copy_file(src_path, global_pm_dir / dst_name)

    return str(packet_path), str(explanation_path)


def investigate_caption_violation(
    violating_image: Path,
    raw_error: str,
    chain_name: str,
    seed_image: Path,
    chain_dir: Path,
    iteration: int,
    prompt_type: str,
) -> dict:
    """
    Runs a post-mortem analysis on a captioning policy violation.
    Returns a summary dict with file paths and any error that occurred during analysis.
    """
    chain_pm_dir  = chain_dir / "post_mortem"
    global_pm_dir = Path(OUTPUT_DIR) / "post_mortem" / chain_name

    packet = {
        "type":                   "caption_policy_violation",
        "chain":                  chain_name,
        "prompt_type":            prompt_type,
        "iteration":              iteration,
        "timestamp":              datetime.now().isoformat(),
        "violating_image":        str(violating_image),
        "violating_image_sha256": sha256_file(violating_image) if violating_image.exists() else None,
        "seed_image":             str(seed_image),
        "seed_image_sha256":      sha256_file(seed_image) if seed_image.exists() else None,
        "raw_error":              raw_error,
    }

    investigation_prompt = (
        "You are a policy post-mortem investigator for an image captioning API.\n\n"
        "A captioning request failed with a CONTENT_POLICY_VIOLATION.\n\n"
        "You are given:\n"
        "1) The image that was submitted for captioning\n"
        "2) The raw API error message\n\n"
        "Task:\n"
        "- Identify the most specific policy category implicated (e.g., sexual content, violence, "
        "self-harm, hate, harassment, privacy/identification, copyrighted/celebrity content, etc.).\n"
        "- Describe what visual elements likely triggered the block at a high level; do not speculate "
        "beyond what is visible in the image.\n"
        "- If multiple triggers are plausible, list them in descending likelihood.\n"
        "- Provide concrete guidance to make a compliant captioning request.\n\n"
        f"Raw error message:\n{raw_error}\n"
    )

    explanation, analysis_error = "", None
    try:
        explanation = _run_post_mortem_api(investigation_prompt, violating_image)
    except Exception as e:
        analysis_error = str(e)

    extra = {
        "seed.jpg": seed_image,
        f"iter_{iteration:02d}_caption_violating_image.jpg": violating_image,
    }
    pkt_path, inv_path = _write_post_mortem_files(
        chain_pm_dir, global_pm_dir, iteration, "caption",
        packet, raw_error, explanation, extra,
    )

    return {"packet_path": pkt_path, "investigation_path": inv_path, "analysis_error": analysis_error}


def investigate_generation_violation(
    violating_prompt: str,
    raw_error: str,
    chain_name: str,
    seed_image: Path,
    chain_dir: Path,
    iteration: int,
    prompt_type: str,
) -> dict:
    """
    Runs a post-mortem analysis on an image-generation policy violation.
    Returns a summary dict with file paths and any error that occurred during analysis.
    """
    chain_pm_dir  = chain_dir / "post_mortem"
    global_pm_dir = Path(OUTPUT_DIR) / "post_mortem" / chain_name

    packet = {
        "type":                    "generation_policy_violation",
        "chain":                   chain_name,
        "prompt_type":             prompt_type,
        "iteration":               iteration,
        "timestamp":               datetime.now().isoformat(),
        "violating_prompt":        violating_prompt,
        "violating_prompt_sha256": hashlib.sha256((violating_prompt or "").encode()).hexdigest(),
        "seed_image":              str(seed_image),
        "seed_image_sha256":       sha256_file(seed_image) if seed_image.exists() else None,
        "raw_error":               raw_error,
    }

    investigation_prompt = (
        "You are a policy post-mortem investigator for an image generation API.\n\n"
        "An image generation request failed with a CONTENT_POLICY_VIOLATION.\n\n"
        "You are given:\n"
        "1) The exact prompt submitted for generation\n"
        "2) The raw API error message\n\n"
        "Task:\n"
        "- Identify the most specific policy category implicated.\n"
        "- Quote the exact spans of the prompt most likely to have triggered the block.\n"
        "- If multiple triggers are plausible, list them in descending likelihood.\n"
        "- Provide concrete compliant rewrites (minimal edits first, then a safer alternative).\n\n"
        f"Raw error message:\n{raw_error}\n\n"
        f"Submitted prompt:\n{violating_prompt}\n"
    )

    explanation, analysis_error = "", None
    try:
        explanation = _run_post_mortem_api(investigation_prompt, image_path=None)
    except Exception as e:
        analysis_error = str(e)

    violating_prompt_file = chain_pm_dir / f"iter_{iteration:02d}_generation_violating_prompt.txt"
    ensure_dir(chain_pm_dir)
    write_text(violating_prompt_file, violating_prompt or "")

    extra = {
        "seed.jpg": seed_image,
        violating_prompt_file.name: violating_prompt_file,
    }
    pkt_path, inv_path = _write_post_mortem_files(
        chain_pm_dir, global_pm_dir, iteration, "generation",
        packet, raw_error, explanation, extra,
    )

    return {"packet_path": pkt_path, "investigation_path": inv_path, "analysis_error": analysis_error}


# ================================
# VIOLATION ARCHIVING
# ================================

def archive_violated_chain(chain_dir: Path, chain_name: str):
    """Copies a policy-terminated chain folder into output/violations/<chain_name>."""
    ensure_dir(Path(OUTPUT_DIR) / "violations")
    copy_tree(chain_dir, Path(OUTPUT_DIR) / "violations" / chain_name)


def build_violations_index(seed_images: list[Path]):
    """
    Scans all per-chain log.json files in the output tree and copies each original
    seed image that triggered a CONTENT_POLICY_VIOLATION into output/violations/bad_seeds/.
    Runs once at end-of-run so the index is always complete regardless of how many
    resume sessions the batch required.
    """
    bad_seeds_dir = Path(OUTPUT_DIR) / "violations" / "bad_seeds"
    ensure_dir(bad_seeds_dir)

    seed_by_stem = {s.stem: s for s in seed_images}
    seen: set[str] = set()

    for log_file in Path(OUTPUT_DIR).rglob("log.json"):
        try:
            log = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        pv = log.get("policy_violations", {})
        has_violation = (
            pv.get("total_blocks", 0) > 0
            or log.get("termination_reason", "") == "CONTENT_POLICY_VIOLATION"
        )
        if not has_violation:
            continue

        seed_name = log.get("seed_name", "")
        if not seed_name or seed_name in seen:
            continue

        src = seed_by_stem.get(seed_name)
        if src and src.exists():
            copy_file(src, bad_seeds_dir / f"{seed_name}{src.suffix.lower() or '.jpg'}")
            seen.add(seed_name)
            pass  # bad seed collected silently


# ================================
# CHAIN LOGGING
# ================================

def make_chain_log(seed_path: Path, seed_name: str, prompt_type: str, chain_num: int) -> dict:
    """Returns a fresh log structure for a chain starting from iteration 1."""
    return {
        "seed_image":        str(seed_path),
        "seed_name":         seed_name,
        "prompt_type":       prompt_type,
        "chain_num":         chain_num,
        "prompt_text":       PROMPTS[prompt_type],
        "start_time":        datetime.now().isoformat(),
        "iterations":        [],
        "policy_violations": {"caption_blocks": 0, "generation_blocks": 0, "total_blocks": 0},
        "hang_events":       [],
        "pause_events":      [],
        "total_tokens_used": 0,
    }


def build_chain_summary(log: dict, total_iters: int) -> dict:
    """
    Builds the executive_summary block that sits at the top of each log.json.
    Covers completion status, early-stop reason, policy blocks, tokens, timing, and pauses.
    """
    pv         = log.get("policy_violations", {})
    start      = log.get("start_time", "")
    end        = log.get("end_time", "")
    completed  = log.get("completed_iterations", 0)
    terminated = log.get("chain_terminated_early", False)

    active_secs = log.get("active_seconds", 0)

    return {
        "status":             "terminated_early" if terminated else "completed",
        "completed_iters":    completed,
        "total_iters":        total_iters,
        "termination_reason": log.get("termination_reason") if terminated else None,
        "policy_blocks":      pv.get("total_blocks", 0),
        "policy_caption":     pv.get("caption_blocks", 0),
        "policy_generation":  pv.get("generation_blocks", 0),
        "total_tokens":       log.get("total_tokens_used", 0),
        "hang_events":        len(log.get("hang_events", [])),
        "pause_count":        len(log.get("pause_events", [])),
        "start_time":         start,
        "end_time":           end,
        "active_duration":    fmt_duration(active_secs),
        "active_seconds":     active_secs,
    }


def save_chain_log(
    chain_dir: Path,
    log: dict,
    total_iters: int,
    seed_name: str,
    prompt_type: str,
    chain_num: int,
    chain_start_mono: float,
    active_offset: float,
):
    """
    Updates computed fields on the log dict (including active time), builds the
    executive summary block, then writes log.json and summary.txt to the chain dir.

    chain_start_mono: time.monotonic() value recorded when the chain started.
    active_offset:    seconds of active time that elapsed before this chain started
                      (i.e. time spent on earlier chains in the same session), used
                      to compute per-chain active time correctly.
    """
    now_mono = time.monotonic()
    # Active time for this chain = total session active time minus pre-chain active time.
    chain_active = max(0.0, display.active_seconds() - active_offset)

    log["end_time"]               = datetime.now().isoformat()
    log["active_seconds"]         = round(chain_active, 1)
    log["completed_iterations"]   = sum(1 for it in log["iterations"] if it.get("success"))
    log["chain_terminated_early"] = log["completed_iterations"] < total_iters

    if log["chain_terminated_early"] and log["iterations"]:
        last = log["iterations"][-1]
        log["termination_reason"] = (
            last.get("caption_error") or last.get("generation_error") or "UNKNOWN"
        )

    log["executive_summary"] = build_chain_summary(log, total_iters)

    # Reorder so executive_summary is the first key in the JSON output.
    ordered = {"executive_summary": log.pop("executive_summary")}
    ordered.update(log)
    log.update(ordered)

    (chain_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    pv = log["policy_violations"]
    lines = [
        "CHAIN SUMMARY",
        "=" * 50,
        f"Status:            {'TERMINATED EARLY' if log['chain_terminated_early'] else 'Completed'}",
        f"Seed:              {seed_name}",
        f"Prompt:            {prompt_type}",
        f"Chain:             {chain_num}",
        f"Completed:         {log['completed_iterations']}/{total_iters}",
        f"Active time:       {fmt_duration(chain_active)}",
        f"Total tokens:      {log['total_tokens_used']}",
        f"Hang events:       {len(log['hang_events'])}",
        f"Pauses:            {len(log['pause_events'])}",
        "Policy blocks:",
        f"  caption:           {pv['caption_blocks']}",
        f"  generation:        {pv['generation_blocks']}",
        f"  total:             {pv['total_blocks']}",
        "",
        f"Caption model:     {CAPTION_MODEL}  ({CAPTION_API_MODE})",
        f"Image model:       {IMAGE_MODEL}",
    ]
    if log["chain_terminated_early"]:
        lines += ["", f"Terminated:        {log.get('termination_reason', 'Unknown')}"]

    (chain_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ================================
# MASTER LOG
# ================================

def build_master_log(run_stats: RunStats) -> dict:
    """
    Rebuilds the master log by reading every per-chain log.json in the output tree.
    Because it scans from disk rather than accumulating in memory, it is always
    complete regardless of how many resume sessions the batch required.

    The executive_summary at the top aggregates stats across all chains.
    """
    chains: list[dict] = []
    for log_file in sorted(Path(OUTPUT_DIR).rglob("log.json")):
        try:
            chains.append(json.loads(log_file.read_text(encoding="utf-8")))
        except Exception:
            continue

    total_tokens     = sum(c.get("total_tokens_used", 0) for c in chains)
    total_policy     = sum(c.get("policy_violations", {}).get("total_blocks", 0) for c in chains)
    caption_blocks   = sum(c.get("policy_violations", {}).get("caption_blocks", 0) for c in chains)
    gen_blocks       = sum(c.get("policy_violations", {}).get("generation_blocks", 0) for c in chains)
    early_stops      = sum(1 for c in chains if c.get("chain_terminated_early"))
    completed_chains = sum(1 for c in chains if not c.get("chain_terminated_early"))
    hang_events      = sum(len(c.get("hang_events", [])) for c in chains)
    total_iters_done = sum(c.get("completed_iterations", 0) for c in chains)

    # Collect unique early-stop reasons and how many chains hit each one.
    stop_reasons: dict[str, int] = {}
    for c in chains:
        if c.get("chain_terminated_early"):
            reason = c.get("termination_reason", "UNKNOWN")
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

    stats = run_stats.to_dict(display.active_seconds())
    stats["total_tokens_used"] = total_tokens  # on-disk scan is authoritative

    built_at = datetime.now().isoformat()

    executive_summary = {
        # ---- identity ----
        "built_at":               built_at,
        "run_start":              stats["start_time"],
        "run_end":                stats["end_time"],
        "active_duration":        stats["active_duration"],
        "active_seconds":         stats["active_seconds"],

        # ---- scope ----
        "total_chains":           len(chains),
        "total_iterations":       total_iters_done,
        "prompt_key":             chains[0].get("prompt_type", "unknown") if chains else "unknown",

        # ---- outcomes ----
        "chains_completed":       completed_chains,
        "chains_early_stop":      early_stops,
        "early_stop_reasons":     stop_reasons,

        # ---- policy ----
        "policy_blocks_total":    total_policy,
        "policy_blocks_caption":  caption_blocks,
        "policy_blocks_generate": gen_blocks,

        # ---- tokens ----
        "total_tokens_used":      total_tokens,
        "avg_tokens_per_chain":   round(total_tokens / len(chains), 1) if chains else 0,

        # ---- reliability ----
        "hang_events_total":      hang_events,
        "pause_count":            stats["pause_count"],
        "pause_resume_events":    stats["pause_resume_events"],

        # ---- models ----
        "caption_model":          CAPTION_MODEL,
        "caption_api_mode":       CAPTION_API_MODE,
        "image_model":            IMAGE_MODEL,
        "image_size":             IMAGE_SIZE,
        "image_quality":          IMAGE_QUALITY,
        "post_mortem_model":      POST_MORTEM_MODEL,
    }

    return {
        "executive_summary": executive_summary,
        "run_stats":         stats,
        "total_chains":      len(chains),
        "total_tokens_used": total_tokens,
        "chains":            chains,
    }


# ================================
# API FAILURE HANDLER
# ================================

class AbortRun(Exception):
    """
    Raised by handle_api_failure when repeated timeouts exhaust MAX_TIMEOUT_RETRIES.
    Signals the main loop to save whatever progress has been made and exit.
    """
    pass


def handle_api_failure(
    err_type: str,
    err_msg: str,
    stage: str,
    log: dict,
    iter_data: dict,
    chain_dir: Path,
    chain_name: str,
    seed_path: Path,
    prompt_type: str,
    iteration: int,
    chain_num: int,
    total_iters: int,
    seed_name: str,
    checkpoint: dict,
    chain_start_mono: float,
    active_offset: float,
    caption_text: str | None = None,
    image_path: Path | None = None,
) -> dict:
    """
    Centralised handler for a failed caption or generate call.

    - Logs the error and any hang event into iter_data and the chain log.
    - Runs post-mortem investigation for CONTENT_POLICY_VIOLATION errors.
    - Archives the chain and marks it done for CONTENT_POLICY_VIOLATION errors.
    - Raises AbortRun for TIMEOUT errors.

    stage must be "caption" or "generation".
    caption_text is required for generation failures.
    image_path   is required for caption failures.
    """
    iter_data[f"{stage}_error"]         = err_type
    iter_data[f"{stage}_error_details"] = err_msg or None

    if err_type == "TIMEOUT":
        hang_record = {
            "stage":     stage,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "message":   err_msg,
        }
        log["hang_events"].append(hang_record)
        iter_data["hang_event"] = hang_record

    post_mortem_info = None
    if err_type == "CONTENT_POLICY_VIOLATION":
        log["policy_violations"][f"{stage}_blocks"] += 1
        log["policy_violations"]["total_blocks"]    += 1
        display.log("[post-mortem] running...")

        seed_in_chain  = chain_dir / "iter_00_seed.jpg"
        effective_seed = seed_in_chain if seed_in_chain.exists() else seed_path

        try:
            if stage == "caption":
                post_mortem_info = investigate_caption_violation(
                    violating_image=image_path,
                    raw_error=err_msg or "",
                    chain_name=chain_name,
                    seed_image=effective_seed,
                    chain_dir=chain_dir,
                    iteration=iteration,
                    prompt_type=prompt_type,
                )
            else:
                post_mortem_info = investigate_generation_violation(
                    violating_prompt=caption_text or "",
                    raw_error=err_msg or "",
                    chain_name=chain_name,
                    seed_image=effective_seed,
                    chain_dir=chain_dir,
                    iteration=iteration,
                    prompt_type=prompt_type,
                )
        except Exception as pm_err:
            post_mortem_info = {"post_mortem_failed": str(pm_err)}

    if post_mortem_info:
        iter_data["post_mortem"] = post_mortem_info

    log["iterations"].append(iter_data)
    save_chain_log(
        chain_dir, log, total_iters, seed_name, prompt_type, chain_num,
        chain_start_mono, active_offset,
    )

    if err_type == "CONTENT_POLICY_VIOLATION":
        mark_chain_done(seed_path, prompt_type, chain_num, checkpoint)
        try:
            archive_violated_chain(chain_dir, chain_name)
        except Exception as arc_err:
            display.log(f"WARNING: archive failed: {str(arc_err)[:80]}")

    if err_type == "TIMEOUT":
        raise AbortRun(
            f"Repeated timeouts on {stage} at chain '{chain_name}' iteration {iteration}. "
            f"Progress saved. Re-run with 'checkpoint' to resume."
        )

    return iter_data


# ================================
# CHAIN EXECUTION
# ================================

def run_chain(
    seed_path: Path,
    prompt_type: str,
    chain_num: int,
    total_iters: int,
    checkpoint: dict,
    chain_index: int,
    total_chains: int,
) -> dict:
    """
    Executes one full iteration chain for a single seed image.
    Automatically resumes from the last successfully written generated image if
    partial output already exists on disk.

    Raises AbortRun if a timeout is unrecoverable.
    Returns the completed chain log dict.
    """
    seed_name = seed_path.stem
    name      = chain_id(seed_path, prompt_type, chain_num)
    out_dir   = chain_output_dir(seed_path, prompt_type, chain_num)
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_start_mono = time.monotonic()
    active_offset    = display.active_seconds()

    resume_from = last_good_iteration(out_dir)

    display.seed_name         = seed_name
    display.chain_num         = chain_num
    display.chain_index       = chain_index
    display.total_chains      = total_chains
    display.total_iters       = total_iters
    display.current_iter      = resume_from + 1
    display.done_chains       = chain_index - 1
    display.chain_policy_blocks = 0
    display.iter_tokens       = 0
    display.stage             = ""
    display.notice            = ""

    # Load or initialise the chain log.
    if resume_from > 0:
        display.notice = f"Resuming from iter {resume_from + 1}"
        log_file = out_dir / "log.json"
        log = (
            json.loads(log_file.read_text(encoding="utf-8"))
            if log_file.exists()
            else make_chain_log(seed_path, seed_name, prompt_type, chain_num)
        )
        log.setdefault("pause_events", [])
        log["pause_events"].append({
            "event":     "resume",
            "timestamp": datetime.now().isoformat(),
        })
        start_iter = resume_from + 1
    else:
        log        = make_chain_log(seed_path, seed_name, prompt_type, chain_num)
        start_iter = 1
        seed_out   = out_dir / "iter_00_seed.jpg"
        Image.open(seed_path).convert("RGB").save(seed_out)
        

    display.redraw()

    # Iteration loop.
    for i in range(start_iter, total_iters + 1):
        pause_handler.check()

        display.current_iter = i
        display.iter_tokens = 0
        display.stage = "[caption]"
        display.notice = ""
        display.redraw()

        iter_data   = {"iteration": i, "timestamp": datetime.now().isoformat()}
        current_img = out_dir / ("iter_00_seed.jpg" if i == 1 else f"iter_{i-1:02d}_generated.jpg")

        # Caption step.
        caption, err_type, err_msg, tokens = caption_image(current_img, PROMPTS[prompt_type])

        if err_type:
            display.notice = f"[caption] FAILED: {err_type}"
            display.redraw()
            handle_api_failure(
                err_type=err_type, err_msg=err_msg, stage="caption",
                log=log, iter_data=iter_data, chain_dir=out_dir, chain_name=name,
                seed_path=seed_path, prompt_type=prompt_type, iteration=i,
                chain_num=chain_num, total_iters=total_iters, seed_name=seed_name,
                checkpoint=checkpoint, chain_start_mono=chain_start_mono,
                active_offset=active_offset, image_path=current_img,
            )
            return log

        log["total_tokens_used"]         += tokens
        display.tokens                   += tokens
        display.iter_tokens               = tokens
        iter_data["caption"]              = caption
        iter_data["caption_length_words"] = len(caption.split())
        iter_data["tokens_used"]          = tokens

        (out_dir / f"iter_{i:02d}_caption.txt").write_text(caption, encoding="utf-8")
        display.stage = "[generate]"
        display.redraw()

        # Generate step.
        next_img = out_dir / f"iter_{i:02d}_generated.jpg"
        success, err_type, err_msg = generate_image(caption, next_img)

        if err_type:
            display.notice = f"[generate] FAILED: {err_type}"
            display.redraw()
            handle_api_failure(
                err_type=err_type, err_msg=err_msg, stage="generation",
                log=log, iter_data=iter_data, chain_dir=out_dir, chain_name=name,
                seed_path=seed_path, prompt_type=prompt_type, iteration=i,
                chain_num=chain_num, total_iters=total_iters, seed_name=seed_name,
                checkpoint=checkpoint, chain_start_mono=chain_start_mono,
                active_offset=active_offset, caption_text=caption,
            )
            return log

        iter_data["image_path"] = str(next_img)
        iter_data["success"]    = True
        log["iterations"].append(iter_data)

        # Snapshot pace for ETA — only update when an image actually finishes.
        _imgs_this_session = display._images_done() - display._session_imgs_offset
        if _imgs_this_session > 0:
            display._secs_per_img = display.active_seconds() / _imgs_this_session

        save_chain_log(
            out_dir, log, total_iters, seed_name, prompt_type, chain_num,
            chain_start_mono, active_offset,
        )
        display.stage = "[done]"
        display.redraw()

        # Brief pause between iterations to avoid hammering the API.
        if i < total_iters:
            time.sleep(2)

    mark_chain_done(seed_path, prompt_type, chain_num, checkpoint)

    display.stage = ""
    display.notice = ""
    display.redraw()

    return log


# ================================
# INTERACTIVE PROMPTS
# ================================

def prompt_choice(label: str, choices: list[str], default: str | None = None) -> str:
    """Prompts the user to pick from a list of choices. Loops until valid input is received."""
    normalized    = {c.lower(): c.lower() for c in choices}
    default_lower = default.lower() if default else None

    while True:
        options = " / ".join(choices)
        raw     = display.input_plain(f"{label} [{options}]: ").strip().lower()

        if not raw and default_lower:
            return default_lower
        if raw in normalized:
            return normalized[raw]
        display.print_plain(f"  Please enter one of: {', '.join(choices)}")


def prompt_int(label: str, default: int | None = None, min_value: int = 1) -> int:
    """Prompts the user to enter an integer >= min_value. Loops until valid input is received."""
    while True:
        raw = display.input_plain(f"{label}: ").strip()

        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if val < min_value:
                display.print_plain(f"  Please enter an integer >= {min_value}")
                continue
            return val
        except ValueError:
            display.print_plain("  Please enter a valid integer.")


def select_prompt() -> tuple[str, str]:
    """
    Presents the available PROMPTS entries as a numbered menu with option 0
    for free-text entry. Loops until the user confirms their selection.
    Returns (prompt_key, prompt_text).
    """
    keys = list(PROMPTS.keys())

    while True:
        display.print_plain("")
        display.print_plain("Select a prompt:")
        display.print_plain("  0. Enter your own")
        for idx, key in enumerate(keys, start=1):
            preview = PROMPTS[key]
            if len(preview) > 80:
                preview = preview[:77] + "..."
            display.print_plain(f"  {idx}. [{key}]  {preview}")
        display.print_plain("")

        raw = display.input_plain("Choice: ").strip()
        if not raw:
            display.print_plain("  Please enter a number.")
            continue
        try:
            choice = int(raw)
        except ValueError:
            display.print_plain("  Please enter a number.")
            continue

        if choice < 0 or choice > len(keys):
            display.print_plain(f"  Please enter a number between 0 and {len(keys)}.")
            continue

        if choice == 0:
            text = display.input_plain("\nEnter your prompt text: ").strip()
            if not text:
                display.print_plain("  Prompt cannot be empty.")
                continue
            display.print_plain(f"\nYour prompt:\n  {text}\n")
            if prompt_choice("Use this prompt?", ["y", "n"], default="y") == "y":
                return "custom", text
            continue

        key = keys[choice - 1]
        return key, PROMPTS[key]


# ================================
# RUNTIME CONFIG
# ================================

def resolve_config(args, num_seeds: int, checkpoint: dict) -> tuple[str, int, int, str, str]:
    """
    Determines run mode, chain width, iteration depth, and active prompt.

    In checkpoint mode all settings are read from the stored checkpoint; the user
    is not re-prompted. A mismatch between the stored seed count and the current
    seed_images directory is treated as a hard error.

    In newbatch mode the user selects a prompt interactively and enters chains/
    iterations unless those values were supplied as CLI arguments.

    Returns: (mode, chains, iterations, prompt_key, prompt_text)
    """
    has_checkpoint = (
        Path(CHECKPOINT_FILE).exists()
        and checkpoint.get("original_settings") is not None
    )

    mode = args.mode
    if mode is None:
        if has_checkpoint:
            display.print_plain("")
            display.print_plain("Checkpoint detected.")
        mode = prompt_choice(
            "Start a new batch or resume from checkpoint?",
            ["newbatch", "checkpoint"],
            default="checkpoint" if has_checkpoint else "newbatch",
        )

    # ---- Checkpoint mode ----
    if mode == "checkpoint":
        if not has_checkpoint:
            display.print_plain("\nNo checkpoint found.")
            sys.exit(1)

        stored      = checkpoint["original_settings"]
        chains      = stored["chains"]
        iterations  = stored["iterations"]
        prompt_key  = stored.get("prompt_key", list(PROMPTS.keys())[0])
        prompt_text = stored.get("prompt_text", PROMPTS.get(prompt_key, ""))

        # A changed seed count means the work list cannot be safely reconstructed.
        if num_seeds != stored["num_seeds"]:
            display.print_plain("")
            display.print_plain("ERROR: Seed count mismatch.")
            display.print_plain(f"  Checkpoint recorded {stored['num_seeds']} seed(s).")
            display.print_plain(f"  seed_images/ currently contains {num_seeds} seed(s).")
            display.print_plain("  Restore the original seed set before resuming.")
            sys.exit(1)

        # Re-register the prompt in PROMPTS so run_chain can look it up by key.
        if prompt_key not in PROMPTS:
            PROMPTS[prompt_key] = prompt_text

        total_possible = stored["num_seeds"] * chains
        finished       = len(checkpoint.get("completed_chains", []))
        remaining      = total_possible - finished

        display.print_plain("")
        display.print_plain("Checkpoint loaded!")
        display.print_plain("Original settings:")
        display.print_plain(f"  {stored['num_seeds']} seeds * {chains} chains * {iterations} iterations")
        display.print_plain(f"  Prompt: [{prompt_key}]")
        display.print_plain("")
        display.print_plain(f"  {finished}/{total_possible} finished; queuing the remaining {remaining} chains.")
        display.print_plain("")

        if prompt_choice("Proceed?", ["y", "n"], default="y") != "y":
            display.print_plain("Cancelled.")
            sys.exit(0)

        return mode, chains, iterations, prompt_key, prompt_text

    # ---- New batch mode ----
    prompt_key, prompt_text = select_prompt()

    # Add the selected prompt to PROMPTS under its key so run_chain can look it up.
    if prompt_key not in PROMPTS:
        PROMPTS[prompt_key] = prompt_text

    chains_explicit = "--chains"     in sys.argv
    iters_explicit  = "--iterations" in sys.argv

    chains     = args.chains     if chains_explicit else prompt_int("Chains per seed?",      default=CHAINS_PER_SEED)
    iterations = args.iterations if iters_explicit  else prompt_int("Iterations per chain?", default=ITERATIONS)

    return mode, chains, iterations, prompt_key, prompt_text


def clear_output_for_new_batch(seed_images: list[Path], prompt_key: str, chains: int):
    """
    Detects whether prior output or a checkpoint already exists.
    If so, asks the user whether to wipe it before starting fresh.
    Exits if the user declines.
    Returns a fresh checkpoint dict if output was cleared, or None if nothing was found.
    """
    prior_exists = Path(CHECKPOINT_FILE).exists() or any(
        chain_output_dir(seed, prompt_key, cn).exists()
        for seed in seed_images
        for cn in range(1, chains + 1)
    )

    if not prior_exists:
        return None

    display.print_plain("")
    if prompt_choice(
        "Existing checkpoint/output detected. Delete it and start fresh?",
        ["y", "n"], default="n",
    ) != "y":
        display.print_plain("Cancelled.")
        sys.exit(0)

    display.print_plain("\nRemoving old checkpoint and output...")
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    ensure_dir(Path(OUTPUT_DIR) / "violations")
    ensure_dir(Path(OUTPUT_DIR) / "post_mortem")

    return {"completed_chains": [], "original_settings": None}


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Telephone-game image mutation runner (pause/resume)"
    )
    parser.add_argument(
        "mode", nargs="?", choices=["newbatch", "checkpoint"],
        help="newbatch: start fresh  |  checkpoint: resume from last save",
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help="Iterations per chain (new batch only)")
    parser.add_argument("--chains",     type=int, default=CHAINS_PER_SEED,
                        help="Chains per seed image (new batch only)")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    ensure_dir(Path(OUTPUT_DIR) / "violations")
    ensure_dir(Path(OUTPUT_DIR) / "post_mortem")

    checkpoint = load_checkpoint()

    seed_images = sorted(
        list(Path(SEED_IMAGES_DIR).glob("*.jpg"))
        + list(Path(SEED_IMAGES_DIR).glob("*.png")),
        key=natural_sort_key,
    )
    if not seed_images:
        display.print_plain(f"\nNo images found in {SEED_IMAGES_DIR}/  (add .jpg or .png files)")
        sys.exit(1)

    mode, chains, iterations, prompt_key, prompt_text = resolve_config(
        args, len(seed_images), checkpoint
    )

    # New batch: optionally wipe prior output.
    if mode == "newbatch":
        fresh = clear_output_for_new_batch(seed_images, prompt_key, chains)
        if fresh is not None:
            checkpoint = fresh
        init_checkpoint_settings(
            checkpoint, len(seed_images), chains, iterations, prompt_key, prompt_text
        )

    # Build work list.
    remaining = [
        (seed, prompt_key, cn)
        for seed in seed_images
        for cn in range(1, chains + 1)
        if not is_chain_done(seed, prompt_key, cn, checkpoint)
    ]

    if not remaining:
        display.print_plain("\nAll chains already completed.")
        sys.exit(0)

    total_chains = len(seed_images) * chains
    done_chains  = total_chains - len(remaining)

    # Print config summary directly so it's readable before the live UI starts.
    # The same lines are also loaded into the scroll buffer so they persist
    # into the live view after the user presses Enter.
    config_lines = [
        "",
        separator("="),
        "  RUN CONFIGURATION",
        separator("="),
        f"  Mode:              {mode}",
        f"  Seeds:             {len(seed_images)}",
        f"  Prompt:            [{prompt_key}]",
    ]
    for wrapped_line in wrap_text(prompt_text):
        config_lines.append(f"                     {wrapped_line}")
    config_lines += [
        f"  Chains per seed:   {chains}",
        f"  Iterations/chain:  {iterations}",
        f"  Total chains:      {total_chains}",
        f"  Already done:      {done_chains}",
        f"  To run now:        {len(remaining)}",
        separator(),
        f"  Caption model:     {CAPTION_MODEL}  ({CAPTION_API_MODE})",
        f"  Post-mortem:       {POST_MORTEM_MODEL}",
        f"  Image model:       {IMAGE_MODEL}  ({IMAGE_SIZE}, {IMAGE_QUALITY})",
        f"  Hang timeout:      {API_TIMEOUT_SECS}s / call"
        f"  (abort after {MAX_TIMEOUT_RETRIES} timeouts)",
        separator(),
        f"  Output:            {OUTPUT_DIR}/<seed>/<prompt>/chain_NN/",
        separator("="),
        "  Ctrl+C once = pause after iteration   Ctrl+C twice = force quit",
        separator("="),
    ]

    # Print plainly first so the user can read it before pressing Enter.
    for line in config_lines:
        display.print_plain(line)

    display.input_plain("\nPress Enter to start... ")

    # Now load the same lines into the scroll buffer so they appear in the
    # live view as context while the first chain is running.
    # Initialise display stats, arm pause handler, start the background ticker.
    display.total_chains = total_chains
    display.done_chains  = done_chains
    display.total_iters  = iterations

    # Seed total-time and image counters from already-completed chain logs
    # so that resumed sessions show cumulative totals, not just this session.
    prior_active_secs = 0.0
    prior_tokens      = 0
    for chain_log_path in Path(OUTPUT_DIR).rglob("log.json"):
        try:
            clog = json.loads(chain_log_path.read_text(encoding="utf-8"))
            es   = clog.get("executive_summary", {})
            prior_active_secs += es.get("active_seconds", 0.0)
            prior_tokens      += clog.get("total_tokens_used", 0)
        except Exception:
            pass
    display._session_offset      = prior_active_secs
    display._session_imgs_offset = done_chains * iterations
    display.tokens               = prior_tokens

    pause_handler.arm()
    display.start_live()

    run_stats = RunStats(total_chains=total_chains, total_iters_per_chain=iterations)

    # Run.
    for idx, (seed, pt, cn) in enumerate(remaining, start=done_chains + 1):
        try:
            log = run_chain(
                seed_path=seed, prompt_type=pt, chain_num=cn,
                total_iters=iterations, checkpoint=checkpoint,
                chain_index=idx, total_chains=total_chains,
            )
            run_stats.record_chain(log)
            display.done_chains  = idx
            display.early_stops  = run_stats.chains_early_stop
            display.policy_blocks = (
                run_stats.policy_blocks_caption + run_stats.policy_blocks_gen
            )
            display.redraw()

        except AbortRun as abort:
            display.notice = f"ABORT: {str(abort)[:60]}  saving log..."
            display.redraw()
            master = build_master_log(run_stats)
            (Path(OUTPUT_DIR) / "master_log.json").write_text(
                json.dumps(master, indent=2), encoding="utf-8"
            )
            sys.exit(1)

        except Exception as e:
            display.notice = f"FATAL: {str(e)[:70]}  — re-run with checkpoint"
            display.redraw()
            sys.exit(1)

    # Finalise.
    display.stage = "[finalising]"
    display.notice = "Building master log..."
    display.redraw()
    master = build_master_log(run_stats)
    (Path(OUTPUT_DIR) / "master_log.json").write_text(
        json.dumps(master, indent=2), encoding="utf-8"
    )
    display.notice = "Building violations index..."
    display.redraw()
    build_violations_index(seed_images)
    display.finalize()
    print()
    print(separator("="))
    print("  DATASET COMPLETE")
    print(separator("-"))
    print(f"  Outputs:       {OUTPUT_DIR}/")
    print(f"  Structure:     {OUTPUT_DIR}/<seed>/<prompt>/chain_NN/")
    print(f"  Post-mortems:  {OUTPUT_DIR}/post_mortem/")
    print(f"  Violations:    {OUTPUT_DIR}/violations/")
    print(f"  Bad seeds:     {OUTPUT_DIR}/violations/bad_seeds/")
    print(f"  Master log:    {OUTPUT_DIR}/master_log.json")
    print(separator("="))


if __name__ == "__main__":
    main()