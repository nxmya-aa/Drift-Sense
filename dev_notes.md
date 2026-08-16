# Development Notes — Drift-Sense

This documents the actual empirical iteration process behind the dataset
generator and localization algorithm, including bugs found and design
choices that were tried and rejected. Kept deliberately honest and detailed
because (a) it's good engineering practice, and (b) the challenge explicitly
rewards identifying root causes and failure modes, not just presenting a
polished final result.

---

## 1. Ground truth was silently invalidated by rotation

**Symptom:** cropping the search image at the recorded ground-truth
coordinates and comparing it to the reference gave near-zero correlation
(mean NCC ≈ 0.006), even before any accuracy evaluation.

**Root cause:** the search image had a small random rotation (±2°) applied
*after* the ground-truth center was computed, so the recorded coordinate no
longer pointed at the pattern's actual (now-rotated) location.

**Fix:** the ground-truth point is now transformed through the exact same
rotation applied to the image (verified empirically against
`scipy.ndimage.rotate`'s actual coordinate convention, not assumed).

---

## 2. Correctly anti-aliased downsampling erased the FinFET signal

**Symptom:** even after fixing rotation, FinFET samples still showed near-
zero correlation at the true location (mean NCC ≈ 0.05), while DRAM was
better but still weak (≈ 0.23).

**Root cause:** the native fin pitch (18 px) becomes 1.8 output pixels after
a true 10x shrink — below the Nyquist sampling limit. A *correctly*
anti-aliased downsample (box-filter averaging, standard for any real optical
or detector downsampling) legitimately suppresses signal at that frequency;
this isn't a bug in the downsampling, it's what real physics would do to
such a fine a pitch at that demagnification.

**Fix:** increased native pitch (DRAM: 34→60, FinFET: 18→50) so the finest
repeating feature survives the shrink with several output pixels per
period. Verified via direct NCC measurement before/after (see citations.md
§6): FinFET mean NCC went from ≈0.05 to ≈0.47 across 30 samples.

---

## 3. Fully-random site placement made the "closest to center" rule meaningless

**Symptom:** even with correlation signal restored, the localization
algorithm's success rate was 0%, with the "closest to search-image-center"
tie-break never landing near the true site.

**Root cause:** the reference site was originally placed uniformly at
random anywhere within the full 10x field of view. With a fine periodic
pitch, there are many periodic repeats between the search-image center and
a randomly-placed true site — so the repeat *nearest the exact center* is
essentially always a different (wrong) instance than the true site. The
spec's own tie-break rule only makes sense if the true site is expected to
be near where the tool aimed, i.e., drift is a *bounded, small* perturbation
— not an arbitrary relocation.

**Fix:** site placement is now bounded to ~15% of the field of view around
the search window's center, consistent with real stage-drift magnitudes
(see citations.md §7) and with what the tie-break rule assumes.

---

## 4. A purely periodic pattern is mathematically ill-posed for localization

**Symptom:** even after fix #3, many samples still failed — an exactly
periodic lattice, by construction, looks identical at every repeat, so
there is no way to prefer the true site over a neighboring repeat from
texture alone.

**Fix:** added `die_signature`, a smooth non-periodic low-frequency field
(sum of a few random sinusoids), added identically to both the reference
and search generation at the same absolute coordinates. This models real
process variation (see citations.md §8) and is what makes the problem
solvable at all, not just harder.

**Result after this fix:** success rate (5px tolerance) jumped to 70%
(DRAM) / 76.7% (FinFET), with near-exact median error (~0.4px) on
successful cases.

---

## 5. Attempted improvement: primary matching on periodicity-suppressed signal

**Hypothesis:** since the die-signature is low-frequency and the lattice is
high-frequency, detecting and notching out the dominant periodic frequency
(via FFT) before correlating should isolate the discriminative signal and
fix the remaining failures.

**Result: this made things significantly WORSE overall** — success rate
dropped to 30% (DRAM) / 13% (FinFET). The notch filter also removes precise
high-frequency alignment information that plain NCC needs for the
(majority) *unambiguous* cases, so replacing the primary signal was a net
loss even though it helped the one adversarial case that motivated it.

**Revised approach:** use plain NCC as the primary ranking signal (restores
the 70-77% baseline), and apply periodicity suppression *only* as a
tie-breaker among already-near-tied top candidates — this only ever touches
decisions that were already ambiguous, so it can't regress the easy
majority. Net result: DRAM improved to 80%, FinFET slightly regressed to
70% (from 76.7%) — a small net average improvement (~73% → ~75%), and a much
more defensible algorithm design, even though it isn't a uniform win.

---

## 6. Known remaining failure mode (see also Results/Failure Analysis slide)

Sample `dram_0006` is a representative honest failure: the true site scores
0.846 under periodicity-suppressed local comparison, while a different
periodic repeat scores 0.881 — genuinely higher, under this sample's
particular noise realization. This is not a code bug; it's the fundamental
difficulty the challenge is built around: when noise is high enough, a
wrong periodic repeat can, by chance, resemble the reference better than
the true (noisy) site does. Plausible next steps beyond classical NCC:
- A learned (Siamese/embedding) matcher trained to be invariant to the
  periodic component while remaining sensitive to the signature — likely
  the most promising direction, since it can learn a better feature space
  than a fixed FFT notch.
- Multi-frame or multi-scale consistency checks (if additional captures at
  intermediate zoom were available).
- Explicitly modeling and marginalizing over the periodic ambiguity as a
  posterior distribution rather than a point estimate, and reporting
  calibrated uncertainty alongside the coordinate.
