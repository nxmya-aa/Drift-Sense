"""
Evaluation harness for the Drift-Sense localization algorithm.
Runs localize() over a generated dataset directory and reports:
  - per-sample pixel error vs ground truth
  - success rate within a stated tolerance
  - mean/median computation time per pair
  - ambiguity flag rate (periodic false-positive risk)
"""
import json
import time
import numpy as np
import cv2
import argparse
import os

from localize import localize


def evaluate(data_dir, tolerance_px=5.0):
    meta = json.load(open(os.path.join(data_dir, "ground_truth.json")))
    results = []
    times = []

    for m in meta:
        sid = m["sample_id"]
        ref = cv2.imread(os.path.join(data_dir, "reference", f"{sid}.png"),
                          cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(os.path.join(data_dir, "search", f"{sid}.png"),
                             cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        t0 = time.perf_counter()
        pred = localize(ref, search, zoom_ratio=m["zoom_ratio"])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        err = np.hypot(pred["x"] - m["gt_center_x"], pred["y"] - m["gt_center_y"])
        results.append({
            "sample_id": sid,
            "error_px": float(err),
            "success": bool(err <= tolerance_px),
            "ambiguous": pred["ambiguous"],
            "score": pred["score"],
            "time_sec": elapsed,
        })

    errors = np.array([r["error_px"] for r in results])
    successes = np.array([r["success"] for r in results])
    ambiguous_rate = np.mean([r["ambiguous"] for r in results])

    summary = {
        "n_samples": len(results),
        "success_rate_pct": float(100 * successes.mean()),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(errors.max()),
        "mean_time_sec": float(np.mean(times)),
        "median_time_sec": float(np.median(times)),
        "ambiguous_case_rate_pct": float(100 * ambiguous_rate),
        "tolerance_px": tolerance_px,
    }
    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--tolerance_px", type=float, default=5.0)
    args = parser.parse_args()

    summary, results = evaluate(args.data_dir, args.tolerance_px)
    print(json.dumps(summary, indent=2))

    # show worst cases for failure-analysis slide
    worst = sorted(results, key=lambda r: -r["error_px"])[:3]
    print("\nWorst cases:")
    for w in worst:
        print(f"  {w['sample_id']}: error={w['error_px']:.1f}px "
              f"ambiguous={w['ambiguous']} score={w['score']:.3f}")
