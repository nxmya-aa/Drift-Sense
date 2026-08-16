"""
Drift-Sense Localization Algorithm
=====================================
Given a Reference image (high-res, "100x") and a Search image (low-res,
"10x", 10x the physical field of view), finds the center (x, y) in Search
image pixel coordinates where the Reference pattern appears.

Approach: scale-aware normalized cross-correlation (NCC).
  1. The 10x zoom ratio is a KNOWN system parameter (not something to infer),
     so we first downsample the reference by that exact factor to match the
     scale it will appear at in the search image. This avoids needing
     multi-scale template matching (which is slower and less accurate when
     the scale is already known).
  2. We run normalized cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED)
     between the scaled-down reference and the search image.
  3. Because the layout is highly periodic, NCC produces MANY near-equal
     peaks (one per repeating unit cell), not one clean peak. We detect all
     strong local maxima, not just the single global max, then apply the
     tie-break rule from the spec: if multiple candidate matches are within
     a small margin of the best score, return the one closest to the center
     of the search image.
  4. We additionally support light Gaussian smoothing before matching to
     reduce sensitivity to independent sensor noise on each capture.

This is intentionally a fast, dependency-light classical CV baseline —
no training required, runs in well under a second per pair on CPU. A
learned (Siamese / deep feature) matcher can be swapped in as `--method dl`
in a later iteration; the interface (ref_path, search_path) -> (x, y) stays
identical either way.
"""

import argparse
import time
import json
import numpy as np
import cv2
from scipy.ndimage import maximum_filter


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


def find_peaks(score_map, num_peaks=8, min_distance=15):
    """
    Non-maximum-suppressed peak detection over a correlation score map.
    Returns list of (score, y, x) sorted descending by score.
    """
    local_max = maximum_filter(score_map, size=min_distance) == score_map
    candidates = np.argwhere(local_max)
    scored = [(score_map[y, x], y, x) for y, x in candidates]
    scored.sort(key=lambda t: -t[0])
    return scored[:num_peaks]


def suppress_periodicity(img, num_peaks=6, exclude_radius=4, notch_radius=2):
    """
    Detects the dominant periodic spatial frequencies in `img` via FFT
    magnitude peaks (excluding the DC/low-frequency region, which carries
    the non-periodic die-signature / broad shading, not the repeating
    lattice) and notches them out.

    Rationale: plain NCC on a highly periodic DRAM/FinFET layout is
    dominated by the repeating lattice itself, so many false-positive
    locations score nearly as high (or higher, under noise) as the true
    site — this is exactly the failure mode Applied Materials describes
    ("template matching returns false positives across the entire
    array"). Since the periodic component is, by definition, concentrated
    at a small number of specific spatial frequencies, and it's the SAME
    dominant frequency almost everywhere in the image, suppressing it
    exposes the residual non-periodic signal (die signature + genuine
    local detail) that actually discriminates the true site. This is a
    standard comb/notch-filtering technique, not specific to our own
    synthetic generator's parameters — the dominant frequency is detected
    from the image itself, so it generalizes to test data with unknown
    (and possibly different) periodicity.
    """
    h, w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(F).copy()
    cy, cx = h // 2, w // 2
    mag[cy - exclude_radius:cy + exclude_radius + 1,
        cx - exclude_radius:cx + exclude_radius + 1] = 0

    peaks = []
    used = np.zeros_like(mag, dtype=bool)
    flat_idx = np.argsort(mag.ravel())[::-1]
    for idx in flat_idx:
        y, x = np.unravel_index(idx, mag.shape)
        if used[y, x]:
            continue
        peaks.append((y, x))
        y0, y1 = max(0, y - 5), min(h, y + 6)
        x0, x1 = max(0, x - 5), min(w, x + 6)
        used[y0:y1, x0:x1] = True
        if len(peaks) >= num_peaks:
            break

    Fn = F.copy()
    for (y, x) in peaks:
        y0, y1 = max(0, y - notch_radius), min(h, y + notch_radius + 1)
        x0, x1 = max(0, x - notch_radius), min(w, x + notch_radius + 1)
        Fn[y0:y1, x0:x1] = 0

    out = np.real(np.fft.ifft2(np.fft.ifftshift(Fn)))
    return out.astype(np.float32)


def localize(reference, search, zoom_ratio=10, smooth_sigma=0.6,
             tie_margin=0.03, num_peaks=10, use_notch_tiebreak=True):
    """
    Core localization routine (arrays in, coordinates out) so it can be
    reused/tested without file I/O.

    Approach (revised after empirical testing — see dev notes):
      1. Primary candidate ranking/positions come from plain NCC on the
         (lightly smoothed) images. An earlier attempt to run the FULL
         match on a periodicity-suppressed (FFT notch-filtered) signal was
         tested and REJECTED: it improved one hand-picked hard case but
         significantly hurt overall accuracy (70-77% -> 13-30% success in
         our own benchmark), because the notch filter also removes
         precise high-frequency alignment information that plain NCC
         relies on for the (majority) unambiguous cases.
      2. Periodicity suppression is instead used ONLY as a tie-breaker:
         when multiple NCC peaks are nearly tied (the genuinely ambiguous
         case periodic layouts create), we re-score just those few tied
         candidates using the periodicity-suppressed signal, which is
         specifically designed to expose the non-repeating (die-signature)
         component that plain NCC can't distinguish between repeats of.
         This is safer than replacing the primary signal because it only
         ever affects already-ambiguous decisions.
      3. If the notch tie-break itself is inconclusive (still tied), we
         fall back to the spec's "closest to search-image-center" rule.

    Returns dict with predicted (x, y), match score, candidate list, and
    ambiguity flag (True if periodic layout produced multiple near-tied
    peaks — useful for the failure-mode/explainability slide).
    """
    ref_size = reference.shape[0]
    template_size = max(8, int(round(ref_size / zoom_ratio)))
    template = cv2.resize(reference, (template_size, template_size),
                           interpolation=cv2.INTER_AREA)

    ref_proc = cv2.GaussianBlur(template, (0, 0), smooth_sigma)
    search_proc = cv2.GaussianBlur(search, (0, 0), smooth_sigma)

    score_map = cv2.matchTemplate(search_proc, ref_proc, cv2.TM_CCOEFF_NORMED)
    # score_map[y, x] corresponds to template's TOP-LEFT placed at (x, y)
    # in the search image. Convert candidate top-lefts to centers later.

    peaks = find_peaks(score_map, num_peaks=num_peaks,
                        min_distance=max(4, template_size // 15))

    if not peaks:
        # Fallback: global argmax
        y, x = np.unravel_index(np.argmax(score_map), score_map.shape)
        peaks = [(score_map[y, x], y, x)]

    best_score = peaks[0][0]
    tied = [p for p in peaks if (best_score - p[0]) <= tie_margin]

    search_center = np.array([search.shape[1] / 2.0, search.shape[0] / 2.0])
    half_t = template_size / 2.0

    def top_left_to_center(y, x):
        return np.array([x + half_t, y + half_t])

    ambiguous = len(tied) > 1

    if ambiguous and use_notch_tiebreak:
        # Re-score only the tied candidates using periodicity-suppressed
        # local correlation — exposes the non-repeating (die-signature)
        # component that plain NCC can't distinguish between repeats of.
        ref_notched = suppress_periodicity(ref_proc)
        sig_scores = []
        for s, y, x in tied:
            crop = search_proc[y:y + template_size, x:x + template_size]
            if crop.shape != ref_notched.shape:
                sig_scores.append(-1.0)
                continue
            crop_notched = suppress_periodicity(crop)
            a = ref_notched - ref_notched.mean()
            b = crop_notched - crop_notched.mean()
            denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-8
            sig_scores.append(float((a * b).sum() / denom))

        sig_scores = np.array(sig_scores)
        sig_margin = 0.05  # how much better a signature match needs to be to trust it
        best_sig = sig_scores.max()
        sig_tied = np.where((best_sig - sig_scores) <= sig_margin)[0]

        if len(sig_tied) == 1:
            chosen_idx = int(sig_tied[0])
        else:
            # still genuinely inconclusive -> fall back to spec's rule
            tied_centers = [top_left_to_center(y, x) for _, y, x in tied]
            dists = [np.linalg.norm(c - search_center) for c in tied_centers]
            # restrict fallback to the sig-tied subset only
            sub_dists = [dists[i] for i in sig_tied]
            chosen_idx = int(sig_tied[np.argmin(sub_dists)])
    else:
        tied_centers = [top_left_to_center(y, x) for _, y, x in tied]
        dists = [np.linalg.norm(c - search_center) for c in tied_centers]
        chosen_idx = int(np.argmin(dists))

    chosen_center = top_left_to_center(tied[chosen_idx][1], tied[chosen_idx][2])
    chosen_score = tied[chosen_idx][0]

    all_candidates = [
        {"x": float(top_left_to_center(y, x)[0]),
         "y": float(top_left_to_center(y, x)[1]),
         "score": float(s)}
        for s, y, x in peaks
    ]

    return {
        "x": float(chosen_center[0]),
        "y": float(chosen_center[1]),
        "score": float(chosen_score),
        "template_size": template_size,
        "ambiguous": ambiguous,
        "n_candidates": len(peaks),
        "candidates": all_candidates,
    }


def run_inference(reference_path, search_path, zoom_ratio=10):
    reference = load_gray(reference_path)
    search = load_gray(search_path)
    t0 = time.perf_counter()
    result = localize(reference, search, zoom_ratio=zoom_ratio)
    elapsed = time.perf_counter() - t0
    result["inference_time_sec"] = elapsed
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drift-Sense localization: find reference pattern center in search image")
    parser.add_argument("reference_path", type=str, help="Path to reference image")
    parser.add_argument("search_path", type=str, help="Path to search image")
    parser.add_argument("--zoom_ratio", type=float, default=10,
                         help="Known zoom ratio between reference and search capture")
    parser.add_argument("--json_out", type=str, default=None,
                         help="Optional path to write full result JSON")
    args = parser.parse_args()

    result = run_inference(args.reference_path, args.search_path, args.zoom_ratio)

    # Required output: single (x, y) coordinate
    print(f"{result['x']:.2f} {result['y']:.2f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
