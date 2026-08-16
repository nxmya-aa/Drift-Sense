# Drift-Sense — Navigation-Error Recovery for Wafer Inspection

Finds the location of a high-resolution Reference image (100x) inside a
lower-resolution Search image (10x, exactly 10x the field of view), for
DRAM-style and FinFET-style periodic die layouts. Built for the Applied
Materials "Drift-Sense" AI Hackathon challenge.

## What's in this repo

| File | Purpose |
|---|---|
| `dataset_generator.py` | Synthetic dataset generator (standalone `.py`) |
| `localize.py` | Localization inference script (standalone `.py`) — **this is the script Applied Materials will run on test data** |
| `eval_localization.py` | Self-evaluation harness: accuracy + speed metrics against a generated dataset |
| `citations.md` | Justification + references for every design choice (noise model, structural parameters, algorithm design) |
| `dev_notes.md` | Honest record of the empirical iteration process, including rejected approaches and known failure modes |
| `requirements.txt` | Exact dependency versions |

There is **no separate training script** — the localization algorithm is a
classical computer-vision method (normalized cross-correlation with an FFT-
based periodicity-aware tie-break), not a trained model, so there are no
weights to reproduce. This is documented explicitly per the "if applicable"
clause in the submission requirements.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. No GPU required — everything here runs on CPU in
well under a second per image pair.

## 1. Generate a dataset

```bash
python3 dataset_generator.py \
    --architecture both \
    --n_pairs 30 \
    --out_dir ./synthetic_data \
    --size 1000 \
    --zoom_ratio 10 \
    --seed 100
```

This creates:
```
synthetic_data/
  dram/
    reference/dram_0000.png ... dram_0029.png
    search/dram_0000.png ... dram_0029.png
    ground_truth.json
  finfet/
    reference/finfet_0000.png ... finfet_0029.png
    search/finfet_0000.png ... finfet_0029.png
    ground_truth.json
```

`--architecture` accepts `dram`, `finfet`, or `both`. Add
`--test_mode_extra_noise` to generate harder, noisier search images (used to
approximate what Applied Materials' hidden test set will look like, per
their note that "the wide-search image will be more noisy in test data").

## 2. Run localization on a single pair

```bash
python3 localize.py path/to/reference.png path/to/search.png --zoom_ratio 10
```

Prints a single line: `<x> <y>` — the predicted center coordinate of the
matching region in the search image, per the spec. Add `--json_out
result.json` to also save the full result (score, candidate list, ambiguity
flag) for debugging/analysis.

**This script is the one Applied Materials will run directly on their test
data.** It has no interactive steps, no manual edits required, and runs
end-to-end from just the two image paths.

## 3. Self-evaluate on a generated dataset

```bash
python3 eval_localization.py synthetic_data/dram --tolerance_px 5.0
python3 eval_localization.py synthetic_data/finfet --tolerance_px 5.0
```

Reports success rate within tolerance, mean/median pixel error, mean/median
inference time, and the rate of genuinely ambiguous (periodic tie) cases.
Also prints the worst 3 cases for failure-mode analysis.

**Current self-evaluation results** (30 pairs per architecture, seed=100):

| Architecture | Success @ 5px | Median error | Mean inference time |
|---|---|---|---|
| DRAM | 80% | 0.43 px | 64 ms |
| FinFET | 70% | 0.40 px | 47 ms |

See `dev_notes.md` for the full iteration history behind these numbers,
including two significant bugs that were caught and fixed (ground truth
silently invalidated by rotation; aliased downsampling erasing signal), and
one documented case where an attempted algorithm improvement was tested and
correctly reverted after it regressed overall accuracy.

## How the algorithm works (short version)

1. The reference is downsampled by the known 10x zoom ratio to match the
   scale it will appear at in the search image.
2. Normalized cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`)
   locates candidate matches.
3. Because DRAM/FinFET layouts are highly periodic, NCC produces many
   near-tied peaks (this is the core difficulty the challenge is built
   around). When top candidates are nearly tied, they're re-scored using a
   periodicity-suppressed signal (FFT notch filtering of the dominant
   repeating spatial frequency) to try to recover the non-periodic signal
   that actually discriminates the true site.
4. If still inconclusive, falls back to the spec's rule: return the
   candidate closest to the search image's center.

Full rationale and citations for every step are in `citations.md`.

## Known limitations (see dev_notes.md §6 for detail)

Under high noise, a wrong periodic repeat can occasionally score higher
than the true (noisy) site even after periodicity suppression — this is a
genuine limit of classical correlation-based matching on adversarially
periodic content, not a bug. A learned (Siamese/embedding) matcher is the
most promising next step beyond this baseline.
