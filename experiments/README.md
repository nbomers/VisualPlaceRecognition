# Side studies

**English** · [Deutsch](README.de.md)

[← back to the README](../README.md) ·
[Overview](#overview) ·
[Measuring](#measuring) ·
[Encoders](#encoders-and-descriptors) ·
[What it fails on](#what-it-fails-on) ·
[Post-processing](#post-processing-of-the-top-k) ·
[Confidence](#confidence) ·
[Six cities](#six-cities) ·
[Cost](#cost)

Every script here asks **one question** and answers it on existing
results. None of them belongs to the pipeline 01–08, and none changes it.
The [README](../README.md) summarises the answers; here you find **how**
every number came about, what it means and what it does not show.

Every section has the same structure:

| | |
|---|---|
| **Question** | what the script is meant to clarify |
| **In short** | the answer in one sentence |
| **Method** | what is computed, and the command |
| **Result** | figure and numbers, with source |
| **Limitations** | what the number does not show |

Terms such as R@1, solvable or Hard are explained in the README under
[terms in one sentence](../README.md#terms-in-one-sentence). Program output in code blocks is in German.

---

## Overview

| Question | Script | Answer |
|---|---|---|
| What does the system achieve with all the reference? | [`full_reference.py`](#full-reference--full_referencepy) | MegaLoc 0.798 instead of 0.568; the ranking holds |
| Which differences are established? | [`bootstrap_ci.py`](#confidence-intervals--bootstrap_cipy) | single numbers ±0.10, differences much narrower |
| How many matches are drives uploaded twice? | [`zwillinge.py`](#twin-drives--zwillingepy) | Osnabrück: 28.2 % with the full reference; without them MegaLoc 0.690 instead of 0.798. In Jena mostly second cameras, not copies |
| Is MegaLoc's lead just width? | [`pca_reduce.py`](#pca-and-whitening--pca_reducepy) | No: at 512 dimensions the ranking holds |
| Do two encoders help together? | [`concat_embeddings.py`](#concatenation--concat_embeddingspy) | On par with MegaLoc, not better |
| Why does the adapter hurt? | [`adapter_diagnose.py`](#adapter-diagnosis--adapter_diagnosepy) | The selection could never choose "no training" |
| Is it the margin or the learning rate? | [`adapter_sweep.py`](#adapter-grid--adapter_sweeppy) | The learning rate |
| Does more reference help? | [`database_density.py`](#reference-density--database_densitypy) | Yes, steadily |
| What makes a query hard? | [`recall_by_difficulty.py`](#difficulty-per-query--recall_by_difficultypy) | Above all the viewing direction |
| Where in the city does it fail? | [`recall_by_district.py`](#districts--recall_by_districtpy) | Districts from 0.07 to 0.83 |
| Where do the errors point? | [`confusion_atlas.py`](#confusion-atlas--confusion_atlaspy) | Just off or far away, hardly anything in between |
| Does averaging the matches help? | [`localization_aggregation.py`](#aggregation--localization_aggregationpy) | No |
| Do neighbouring images help? | [`sequence_retrieval.py`](#neighbouring-frames--sequence_retrievalpy) | No |
| Does the drive as a path help? | [`sequence_hmm.py`](#drive-as-a-path--sequence_hmmpy) | Yes, +0.030 |
| Does geometric re-checking help? | [`geometric_verification.py`](#geometric-verification--geometric_verificationpy) | Not at 25 m |
| Do detected objects help? | [`detection_rerank.py`](#detections--detection_rerankpy) | No |
| Is cos usable as a confidence? | [`rejection_curve.py`](#rejection--rejection_curvepy) | Yes, better than any other measure |
| Which city is suitable? | [`city_coverage.py`](#city-selection--city_coveragepy) | Coverage counts, not density |
| What carries over to other cities? | [`city_comparison.py`](#city-comparison--city_comparisonpy) | The gap between the encoders, not their level |
| What does each encoder cost? | [`timing.py`](#runtime-and-memory--timingpy) | Best trade-off: MegaLoc whitened to 512 |

**Common rules.**
- Every script takes configuration and paths from [`_common.py`](_common.py) and evaluates with
  [`src/evaluation.py`](../src/evaluation.py) — exactly like 07.
- Results go to `experiments/results/<city>/`; new rows for
  `compare.py` to `results/<city>/evaluation/`. Below abbreviated as `results/…`.
- **MegaLoc is the default**; the second figure shows EigenPlaces. Another
  encoder: `--method`.
- Most scripts only need the match lists from 06 (a few tens of MB).
  `full_reference.py`, `database_density.py` and `geometric_verification.py`
  need the embeddings or the images and run where these are.
  What is available where: `python run.py --bestand`.

---

## Measuring

### Full reference — `full_reference.py`

**Question.** What does the system achieve when it gets all the reference material —
`train` and `database` together, 279,453 instead of 48,321 images?

**In short.** Every encoder gains about 0.2; the ranking stays exactly the same.

**Method.** Searches all reference images block by block (65,536 per index,
top k merged) and evaluates like 07. Only encoders without an adapter — `train`
was the adapter's training material.

```bash
python experiments/full_reference.py
```

```bash
python experiments/bootstrap_ci.py --reference full
```

**Result.** R@1 at 25 m, 48,177 solvable of 53,414 queries (90.2 %):

| Encoder | Dim | full reference | 95 % | Benchmark | Gain |
|---|---:|---:|---|---:|---:|
| megaloc | 8448 | **0.798** | [0.739, 0.848] | 0.568 | +0.230 |
| megaloc_pcaw512 | 512 | 0.778 | [0.716, 0.831] | 0.541 | +0.237 |
| eigenplaces_megaloc_concat | 1024 | 0.778 | [0.714, 0.832] | 0.572 | +0.206 |
| eigenplaces_pcaw512 | 512 | 0.715 | [0.646, 0.777] | 0.507 | +0.208 |
| eigenplaces | 2048 | 0.701 | [0.630, 0.765] | 0.484 | +0.217 |
| mixvpr | 4096 | 0.653 | [0.575, 0.724] | 0.426 | +0.227 |
| anyloc_pcaw4096 | 4096 | 0.540 | [0.456, 0.624] | 0.321 | +0.219 |
| anyloc | 4096 | 0.435 | [0.343, 0.533] | 0.204 | +0.231 |
| clip_pcaw512 | 512 | 0.290 | [0.202, 0.402] | 0.105 | +0.185 |
| clip | 512 | 0.232 | [0.144, 0.349] | 0.073 | +0.159 |

<sub>From `results/osnabrueck/evaluation/*_fullref.json` and `results/bootstrap_ci_fullref.json`.
All 18 rows: `python compare.py --reference full --derived`.</sub>

- **The same gain for all** (+0.16 to +0.24): more reference lifts every encoder but does not change who is better.
- **Ranking established:** every neighbouring pair excludes 0, e.g. EigenPlaces → MegaLoc +0.096 [+0.070, +0.122].
- **Shrinking costs measurably here:** MegaLoc 8448 → 512 whitened −0.019 [−0.024, −0.015];
  the concatenation lies clearly *below* MegaLoc (−0.019 [−0.027, −0.011]).

**Limitations.** Part of the gain are [twin drives](#twin-drives--zwillingepy):
without them MegaLoc is at 0.690, EigenPlaces at 0.593.

---

### Confidence intervals — `bootstrap_ci.py`

**Question.** Which differences between two rows are real, which are noise?

**In short.** Single numbers are only accurate to ±0.03 to ±0.10, paired differences much more precisely.

**Method.** The 53,414 queries come from 198 drives, and images of one
drive fail together. Therefore the set of **drives** is resampled with replacement
1,000 times — the same draws for all 40 rows, so that
differences are paired. Every row is checked against its 07 JSON. The interval is the
2.5th and 97.5th percentile of the 1,000 values (percentile bootstrap, no normal approximation).

```bash
python experiments/bootstrap_ci.py
```

**Result.** Seed 42, R@1 at 25 m.

- **Single numbers:** ±0.03 (CLIP) to ±0.10 (MegaLoc) instead of binomial ±0.005 —
  a factor of 100 to 330 in the variance. Reason: the drives are extremely unequal in length
  (median 176 images, the longest 3,156; seven drives provide 19 % of the queries).
- **Differences:** ±0.005 to ±0.09, because hard drives are hard for every encoder.

| Comparison | Diff | 95 % | established |
|---|---:|---|:---:|
| clip → clip_linear | +0.050 | [+0.038, +0.064] | yes |
| clip_pcaw512 → clip_pcaw512_linear | +0.009 | [−0.002, +0.021] | no |
| anyloc → anyloc_linear | +0.130 | [+0.081, +0.187] | yes |
| anyloc → anyloc_pcaw4096 | +0.117 | [+0.078, +0.162] | yes |
| anyloc_pcaw4096 → anyloc_pcaw4096_linear | −0.029 | [−0.074, +0.012] | no |
| anyloc_pcaw512 → anyloc_pcaw512_linear | +0.045 | [+0.009, +0.088] | yes |
| mixvpr → mixvpr_linear | −0.063 | [−0.096, −0.032] | yes |
| eigenplaces → eigenplaces_pca512 | −0.003 | [−0.008, +0.002] | no |
| eigenplaces → eigenplaces_pcaw512 | +0.022 | [+0.011, +0.039] | yes |
| eigenplaces → eigenplaces_pcaw2048 | −0.025 | [−0.042, −0.010] | yes |
| eigenplaces → eigenplaces_linear | −0.040 | [−0.064, −0.017] | yes |
| eigenplaces_pcaw512 → seq3 | −0.009 | [−0.017, −0.001] | yes |
| megaloc → megaloc_pca512 | −0.023 | [−0.031, −0.017] | yes |
| megaloc → eigenplaces_megaloc_concat | +0.004 | [−0.005, +0.014] | no |
| megaloc → megaloc_hmm30-25 | +0.030 | [+0.020, +0.041] | yes |
| megaloc → megaloc_gv20 | −0.029 | [−0.062, +0.000] | no (borderline) |
| megaloc → megaloc_linear | −0.126 | [−0.179, −0.073] | yes |
| megaloc_pca512 → megaloc_pca512_linear | −0.127 | [−0.160, −0.099] | yes |

<sub>Excerpt from [`results/osnabrueck/bootstrap_ci.json`](results/osnabrueck/bootstrap_ci.json);
it contains all 39 pairs, also for R@5 to R@20. `seq3` = `eigenplaces_pcaw512_seq3`.</sub>

**Limitations.**
- 39 comparisons at 95 % are counted. One or two of them may be "established" by chance alone;
  no correction for multiple comparisons is computed. Read close cases (such as `seq3`) accordingly.
- The bounds depend slightly on the bootstrap seed (Monte Carlo error of 1,000 draws, a few tenths of a
  percentage point for single numbers). This only matters where a bound sits right at 0: the upper bound for
  `megaloc_gv20` is +0.0003.
- Reference density, difficulty classes and districts have no bootstrap of their own.

---

### Twin drives — `zwillinge.py`

**Question.** The split separates by sequence. How often does the same
drive still end up on both sides?

**In short.** Mapillary lists some drives as two sequences. In the benchmark
this affects few queries (−0.02 R@1), with the full reference more than a
quarter (−0.108). Counted without twins, MegaLoc reaches 0.690 with the full reference.
Not every twin is a copy: in Jena they are almost all second cameras
looking elsewhere — there R@1 even rises without them.

**What it looks like.** Query `117568923689680` and reference image
`489059405624874` come from two sequences, but from the same account,
0.2 seconds apart, with identical coordinates and compass direction.
Every encoder finds the reference image as its first match: 0 m away and far ahead of all
other candidates (example with MegaLoc: cos 0.56, rank 2 only 0.35). The split does not see this because it only knows the sequence ID.

**Method.** Twin means: same account (`creator_id`) and at most 60 s
apart. An account does not photograph two places at the same time — what is this
close in time is the same drive. The script counts how many queries have a
twin within 25 m and recomputes R@1 **as if the
copy had never been uploaded**: twins drop out of the match list, the
next candidates move up, and a twin does not make a query
solvable. The standard row must hit the number from 07, otherwise it aborts.
In addition it separates **copies** (viewing direction at most 30° apart, or
both panoramas) from simultaneous cameras of one rig that look in another
direction.

```bash
python experiments/zwillinge.py                                   # Osnabrück, MegaLoc und EigenPlaces
python experiments/zwillinge.py --methods megaloc,eigenplaces,mixvpr,anyloc,clip
VPR_CITY="Jena, Germany" python experiments/zwillinge.py          # andere Stadt
python experiments/zwillinge.py --nur-anteile                     # nur Anteile, ohne Trefferlisten
```

If a city's match lists are spread over two machines, a second run adds to the JSON instead of overwriting it.

**Result 1: Osnabrück, all five encoders.** R@1 at 25 m:

| Encoder | Benchmark | without twins | full reference | without twins |
|---|---:|---:|---:|---:|
| MegaLoc | 0.568 | 0.550 | 0.798 | **0.690** |
| EigenPlaces | 0.484 | 0.464 | 0.701 | **0.593** |
| MixVPR | 0.426 | 0.409 | 0.653 | 0.533 |
| AnyLoc | 0.204 | 0.167 | 0.435 | 0.273 |
| CLIP | 0.073 | 0.031 | 0.232 | 0.071 |

<sub>From `results/osnabrueck/zwillinge.json`. "Without twins" counts only queries that are solvable without a copy too
(full 47,800 instead of 48,177). The standard columns hit the numbers from 07. No interval.</sub>

- **Benchmark:** −0.02 for MegaLoc and EigenPlaces. The comparisons there hold.
- **Full reference:** MegaLoc −0.108; for 18.3 % of all queries its first match was a twin.
- **The ranking holds** in both protocols; the gap MegaLoc − EigenPlaces is +0.097 with the full reference, with and without twins.
- **The weaker the encoder, the more copies:** with the full reference CLIP keeps only 0.071 of 0.232.
- **More reference still helps:** 0.550 → 0.690 instead of 0.568 → 0.798.

**Result 2: six cities.**

| City | twin, benchmark | twin, full | of which copy | MegaLoc | MegaLoc full | EigenPlaces full |
|---|---:|---:|---:|---:|---:|---:|
| Osnabrück | 3.4 % | 28.2 % | 97 % | 0.568 → 0.550 | 0.798 → **0.690** | 0.701 → 0.593 |
| Karlsruhe | 4.7 % | 18.4 % | 97 % | 0.419 → 0.394 | 0.640 → **0.577** | 0.507 → 0.440 |
| Würzburg | 1.4 % | 7.0 % | 94 % | 0.336 → 0.323 | 0.476 → **0.426** | 0.384 → 0.325 |
| Fürth | 0.9 % | 6.0 % | 83 % | 0.549 → 0.548 | 0.699 → 0.692 | 0.627 → 0.621 |
| Kaiserslautern | 2.6 % | 7.4 % | 64 % | 0.651 → 0.647 | 0.810 → 0.807 | 0.733 → 0.732 |
| Jena | 6.4 % | 35.8 % | **12 %** | 0.417 → 0.425 | 0.622 → **0.638** | 0.520 → 0.528 |

<sub>R@1 at 25 m, each standard → without twins. Shares from the metadata, per protocol against its reference.
From `results/<city>/zwillinge.json`.</sub>

- **Copies raise the recall, second cameras do not.** Where almost all twins are copies (Osnabrück, Karlsruhe,
  Würzburg), removing them costs 0.05 to 0.11 with the full reference.
- **Jena** has the most twins, but only 12 % face the same direction. The others make a query "solvable" by
  the ground truth without any encoder being able to find it. Without them R@1 rises. A visually checked example
  (query `361562566053038`, twin `1342909849450252`): the front and rear camera of the same bicycle, 0.95 s and 3 m apart.
- **Prediction, recorded in advance:** many twins → a large jump to the full reference. ρ = +0.77, p = 0.10 (n = 6) —
  not established, and Jena contradicts it.
- **Found afterwards:** the share of *copies* orders the cities exactly by their loss (ρ = −1.00, p = 0.003
  exact). This is an explanation after looking at the data, not a passed test.
- **The gap between the models stays:** MegaLoc − EigenPlaces without twins +0.076 to +0.116 in the benchmark,
  +0.071 to +0.137 with the full reference.

**What it means for the other results.**

| Result | affected | why |
|---|---|---|
| Encoder comparison in the benchmark | barely | −0.02 for MegaLoc and EigenPlaces; under Hard the ranking MegaLoc > EigenPlaces > MixVPR > AnyLoc > CLIP holds |
| Full reference, headline 0.798 | **strongly** | without twins 0.690 (−0.108); under Hard 0.523 |
| Encoder comparison with the full reference | barely | MegaLoc and EigenPlaces both lose 0.108, the gap stays +0.097 |
| Reference density | yes | every piece of `train` also adds copies; the rise 0 % → 100 % shrinks from +0.23 to +0.14 |
| City comparison, column "full" | yes | without twins −0.108 (Osnabrück) to +0.016 (Jena); the order of the cities changes (result 2) |
| Hard penalty per city | explained in part | twins are part of d and of 1 − r |
| Adapter selection on val | yes, the other way round | val has practically no twins (0.2 % of the solvable val queries, test 5.3 %) — one reason why val is so much harder than test |
| Confidence (cos) | little | measured in the benchmark, where for 4.1 % of the queries the first match is a copy. If all of them were above cos 0.30 and were removed, "82 % correct" would drop to 78 % in the worst case |
| Geometric verification | little | measured in the benchmark (4.1 % copies at rank 1); copies have the largest image overlap, so part of the gain at 5 and 10 m may come from them |
| Drive as a path | barely | only re-ranks, twins stay where they are |
| "Same account, same day" in the difficulty breakdown | yes | the class contains the twins |
| Examples in the demo | fixed | only matches from a different account are shown |

**How to fix it.** Three levels, by effort:

| Level | What | Consequence | Status |
|---|---|---|---|
| 1 | Disclose: the number without twins next to every number with the full reference | no recomputation | **done** (this script) |
| 2 | A fourth ground truth "without twins" in `src/evaluation.py`, next to Standard, Hard and Sequence | the code identifier changes → evaluation (07) and bootstrap anew for all encoders and cities | open |
| 3 | Split by drives (account + time window) instead of by sequences | the embedding fingerprint hashes `image_id` + `split` → re-encode everything | open; the clean solution |

<sub>Shortcut for level 3: the embeddings do not depend on the split in content. If the split is removed from the
fingerprint (`metadata_digest` in `src/run_guard.py`), only adapter, search and evaluation have to run again.</sub>

**Limitations.**
- This does not make the split "wrong", but leaky: tight by sequence, not by drive.
- 60 s is a choice. To change it: `--fenster-s`.
- No interval. In the five other cities only MegaLoc and EigenPlaces.
- Copy via viewing direction (30°) is a choice; the compass of a Mapillary image is not always accurate.

---

## Encoders and descriptors

### PCA and whitening — `pca_reduce.py`

**Question.** How much of MegaLoc's lead is skill, and how much just width (8448 dimensions)?

**In short.** Skill: at 512 dimensions the ranking holds, and MegaLoc loses only 0.023.

**Method.** Writes `<encoder>_pca512` (PCA to 512) and `<encoder>_pcaw512`
(additionally whitened) as encoders of their own, exactly as 04 would. PCA fitted only
on `train`, 50,000 rows. Only the encoder's embeddings are needed — no image is encoded again.

```bash
python experiments/pca_reduce.py
```

```bash
python run.py --method derived --adapter all
```

![R@1 per variant, one panel per encoder](../results/osnabrueck/figures/evaluation/vergleich_r1_25m_derived.png)

<sub>All 40 benchmark rows, one panel per encoder; colour and shape = variant.
Created by `python compare.py --plot --derived`.</sub>

**Result.** R@1 at 25 m, benchmark:

| Encoder | full width | pca512 | pcaw512 | whitened at full width |
|---|---:|---:|---:|---:|
| megaloc (8448) | **0.568** | 0.545 | 0.541 | — |
| eigenplaces (2048) | 0.484 | 0.481 | **0.507** | 0.459 |
| mixvpr (4096) | 0.426 | 0.408 | 0.424 | — |
| anyloc (4096) | 0.204 | 0.175 | 0.263 | **0.321** |
| clip (512) | 0.073 | 0.074 | **0.105** | — |

- **The ranking does not depend on width:** whitened to 512, MegaLoc 0.541 > EigenPlaces 0.507 > MixVPR 0.424 > AnyLoc 0.263 > CLIP 0.105.
- **Whitening rescues AnyLoc** (+0.117, established): VLAD vectors are very unevenly weighted, and this was not balanced here.
- **For the place encoders** whitening helps slightly at 512 (EigenPlaces +0.022) and hurts at full width (−0.025).

**Limitations.** Why whitening hurts at full width is not measured. A plausible reason: it divides by the smallest
eigenvalues, and with 50,000 samples those are noise. The test would be shrinkage (`√(λ + ε·λ_max)` instead of `√λ`).

---

### Concatenation — `concat_embeddings.py`

**Question.** Do two good encoders see different things, so that both together are better?

**In short.** No: EigenPlaces + MegaLoc are on par with MegaLoc alone — at an eighth of the width.

**Method.** Both whitened to 512, concatenated, re-normalised, written as an encoder of its own.
Aligned via the `image_id`, not the row number. This yields a new combined encoder; it does not re-rank a match list.

```bash
python experiments/concat_embeddings.py
```

| | Dim | R@1 | R@5 |
|---|---:|---:|---:|
| megaloc | 8448 | 0.568 | 0.676 |
| megaloc_pcaw512 | 512 | 0.541 | 0.654 |
| eigenplaces_pcaw512 | 512 | 0.507 | 0.641 |
| **eigenplaces_megaloc_concat** | **1024** | **0.572** | **0.692** |

- R@1: +0.004 [−0.005, +0.014] versus MegaLoc — **not established**.
- R@5: +0.016 [+0.005, +0.028] — narrowly established. The right place slips into the top 5 more often, but not to rank 1.

**Limitations.** Why R@1 does not rise is not explained. Conjecture: both fail on the same hard drives.

---

### Adapter diagnosis — `adapter_diagnose.py`

**Question.** The adapter hurts all place encoders. Is this already visible during training?

**In short.** For 7 of the 12 harmful adapters, yes — but the selection in 05 could never choose "no training".

**Method.** 05 picks the best epoch by val R@1, but only among **trained** epochs: the untrained adapter
(the identity, i.e. exactly the encoder) is never measured. The script measures it on the same val split and
puts it next to the best epoch and the test number from 07.

```bash
python experiments/adapter_diagnose.py
```

**Result.** Osnabrück, all 18 adapters, R@1:

| Encoder | val without | val with | test without | test with | Reading |
|---|---:|---:|---:|---:|---|
| megaloc | 0.198 | 0.132 | 0.568 | 0.442 | already worse on val |
| megaloc_pca512 | 0.167 | 0.133 | 0.545 | 0.417 | already worse on val |
| megaloc_pcaw512 | 0.155 | 0.127 | 0.541 | 0.416 | already worse on val |
| eigenplaces_megaloc_concat | 0.220 | 0.165 | 0.572 | 0.479 | already worse on val |
| eigenplaces | 0.118 | 0.142 | 0.484 | 0.444 | val rises, test falls |
| eigenplaces_pca512 | 0.127 | 0.129 | 0.481 | 0.438 | val rises, test falls |
| eigenplaces_pcaw512 | 0.164 | 0.128 | 0.507 | 0.437 | already worse on val |
| eigenplaces_pcaw2048 | 0.097 | 0.129 | 0.459 | 0.422 | val rises, test falls |
| mixvpr | 0.120 | 0.112 | 0.426 | 0.363 | already worse on val |
| mixvpr_pca512 | 0.103 | 0.112 | 0.408 | 0.371 | val rises, test falls |
| mixvpr_pcaw512 | 0.119 | 0.110 | 0.424 | 0.366 | already worse on val |
| anyloc | 0.050 | 0.194 | 0.204 | 0.335 | helps |
| anyloc_pca512 | 0.048 | 0.184 | 0.175 | 0.305 | helps |
| anyloc_pcaw512 | 0.129 | 0.180 | 0.263 | 0.308 | helps |
| anyloc_pcaw4096 | 0.136 | 0.151 | 0.321 | 0.292 | val rises, test falls |
| clip | 0.015 | 0.033 | 0.073 | 0.123 | helps |
| clip_pca512 | 0.015 | 0.034 | 0.074 | 0.121 | helps |
| clip_pcaw512 | 0.026 | 0.027 | 0.105 | 0.113 | helps |

<sub>From `results/osnabrueck/adapter_diagnose_<encoder>.json`.</sub>

| Group | Count | which |
|---|---:|---|
| helps on test | 6 | CLIP and AnyLoc, except `anyloc_pcaw4096` |
| already worse on val | 7 | all MegaLoc variants, the concatenation, MixVPR, the whitened 512 variants of MixVPR and EigenPlaces |
| val rises, test falls | 5 | the remaining EigenPlaces and MixVPR variants, `anyloc_pcaw4096` |

- **The better an encoder is already trained for places, the earlier the adapter hurts.**
- With the identity as a candidate, MegaLoc would stay at 0.568 instead of 0.442.
- For the five "val rises, test falls" cases, the corrected selection would have taken the adapter too.

**Limitations.** val measures a harder task (MegaLoc 0.198 versus 0.568) and is small: 3,210 solvable queries from
only 46 drives. Only the direction counts; differences such as for MixVPR (0.120 versus 0.112) are noise.
The selection rule is not corrected in 05, because that would change every adapter row.

---

### Adapter grid — `adapter_sweep.py`

**Question.** Does the adapter hurt because the training values are too coarse? The margin 0.2 on the cosine distance corresponds
to 0.4 on the squared distance (‖a−b‖² = 2(1−cos)); NetVLAD uses 0.1 there.

**In short.** It is not the margin, but it is the learning rate. With 1e-4 the adapter no longer hurts EigenPlaces — but it does not help either.

**Method.** Grid over margin 0.2 / 0.1 / 0.05 and learning rate 1e-3 / 1e-4, otherwise as 05. Unlike 05, the
identity is available as epoch 0. Selection by val only; test is only reported. Writes nothing to `results/`.

```bash
python experiments/adapter_sweep.py --method eigenplaces
```

**Result.** Osnabrück, R@1 at 25 m; in bold the choice by val:

| Encoder | Margin | Learning rate | Epoch | val | test R@1 | test R@5 | test R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eigenplaces | identity | — | 0 | 0.118 | 0.484 | 0.608 | 0.695 |
| | 0.2 | 1e-3 | 2 | 0.137 | 0.446 | 0.592 | 0.695 |
| | 0.2 | 1e-4 | 3 | 0.150 | 0.468 | 0.613 | 0.715 |
| | 0.1 | 1e-3 | 3 | 0.146 | 0.458 | 0.610 | 0.712 |
| | 0.1 | 1e-4 | 3 | **0.153** | 0.483 | 0.630 | 0.728 |
| | 0.05 | 1e-3 | 2 | 0.142 | 0.452 | 0.609 | 0.711 |
| | 0.05 | 1e-4 | 3 | 0.149 | 0.483 | 0.636 | 0.733 |
| clip | identity | — | 0 | 0.015 | 0.073 | 0.106 | 0.158 |
| | 0.2 | 1e-3 | 1 | 0.033 | 0.115 | 0.207 | 0.332 |
| | 0.2 | 1e-4 | 1 | 0.038 | 0.133 | 0.232 | 0.357 |
| | 0.1 | 1e-3 | 1 | 0.031 | 0.115 | 0.208 | 0.334 |
| | 0.1 | 1e-4 | 1 | **0.039** | 0.137 | 0.235 | 0.364 |
| | 0.05 | 1e-3 | 1 | 0.032 | 0.115 | 0.208 | 0.337 |
| | 0.05 | 1e-4 | 1 | 0.034 | 0.136 | 0.234 | 0.365 |

<sub>From [`results/osnabrueck/adapter_sweep_eigenplaces.json`](results/osnabrueck/adapter_sweep_eigenplaces.json)
and [`adapter_sweep_clip.json`](results/osnabrueck/adapter_sweep_clip.json).</sub>

- **Margin:** at the same learning rate at most 0.015 difference — the conjecture does not hold.
- **Learning rate:** in all six pairs 1e-4 is better, on val as on test. EigenPlaces: loss −0.027…−0.038 → −0.001…−0.016. CLIP: +0.042 → up to +0.064.
- **Even the best setting does not beat the identity for EigenPlaces** (0.483 versus 0.484); only R@5 and R@20 rise slightly, without interval.
- **val ranks correctly but is wrong relative to the identity:** the order of the combinations matches test (ρ = 0.94),
  yet every one lies above the identity on val and below it on test.

**Limitations.** The row 0.2 / 1e-3 does not hit 05 bit-identically (0.446 versus 0.444, CLIP 0.115 versus 0.123):
do not interpret differences below 0.01. MegaLoc is not computed. The pipeline stays at 1e-3.

---

## What it fails on

### Reference density — `database_density.py`

**Question.** Does the system fail on Osnabrück or on too little reference? In the benchmark 36 % of the queries have no
reference image within 25 m.

**In short.** More reference helps steadily: more queries become solvable, and the recall among them rises.

**Method.** Adds `train` to the reference step by step (0 / 25 / 50 / 75 / 100 % of the train sequences). Only encoders without an adapter.

```bash
python experiments/database_density.py
```

<p align="center">
  <img src="results/osnabrueck/database_density_megaloc.png" width="49%" alt="R@1 versus reference density, MegaLoc">
  <img src="results/osnabrueck/database_density_eigenplaces.png" width="49%" alt="R@1 versus reference density, EigenPlaces">
</p>

| train added | reference images | solvable | MegaLoc | EigenPlaces | MegaLoc, whitened to 512 |
|---:|---:|---:|---:|---:|---:|
| 0 % | 48,321 | 63.9 % | 0.568 | 0.484 | 0.541 |
| 25 % | 107,449 | 78.4 % | 0.636 | 0.545 | 0.615 |
| 50 % | 160,981 | 84.9 % | 0.689 | 0.595 | 0.670 |
| 75 % | 222,300 | 88.4 % | 0.774 | 0.672 | 0.754 |
| 100 % | 279,453 | 90.2 % | 0.798 | 0.701 | 0.778 |

<sub>R@1 at 25 m among the solvable queries, from `results/osnabrueck/database_density_<encoder>.json`.
The last step is the full reference. The whitened column shows: shrinking shifts the curve, the shape stays.
MegaLoc computes with two fewer reference images (279,451); this changes nothing in the numbers.</sub>

- The newly solvable queries are the harder ones — and the recall rises nonetheless.
- What is gained is not only neighbours, but also matching viewing directions and capture times.

**Limitations.** No bootstrap. Part of the rise are [twins](#twin-drives--zwillingepy) from `train`:
without them MegaLoc rises from 0.550 to 0.690 instead of from 0.568 to 0.798 (only the end points are measured).
Memory: the last step searches block by block and so needs about 6 GB (Osnabrück) instead of 11.2 GB.

---

### Difficulty per query — `recall_by_difficulty.py`

**Question.** What makes a single query hard?

**In short.** Above all, whether a reference image faces the same direction. Time and number of neighbours act more weakly.

**Method.** Four properties per query from the metadata, R@1 per class over the solvable queries. Every query
lies in exactly one class per feature.

```bash
python experiments/recall_by_difficulty.py
```

<p align="center">
  <img src="results/osnabrueck/recall_by_difficulty_megaloc.png" width="100%" alt="R@1 by properties of the query, MegaLoc">
</p>
<p align="center">
  <img src="results/osnabrueck/recall_by_difficulty_eigenplaces.png" width="100%" alt="R@1 by properties of the query, EigenPlaces">
</p>

| Feature | Class | n | R@1 MegaLoc |
|---|---|---:|---:|
| A neighbour faces the same direction | yes | 29,167 | 0.653 |
| | no | 4,945 | **0.071** |
| Days to the nearest neighbour | 0–7 | 4,850 | 0.575 |
| | 8–30 | 3,675 | **0.819** |
| | 31–180 | 10,450 | 0.539 |
| | 181–365 | 5,772 | 0.612 |
| | over 365 | 9,365 | 0.472 |
| Neighbours within 25 m | 1–2 | 1,705 | 0.523 |
| | 3–5 | 3,496 | 0.483 |
| | 6–10 | 3,883 | 0.534 |
| | 11–20 | 6,808 | 0.448 |
| | 21–50 | 14,076 | 0.600 |
| | 51+ | 4,144 | **0.780** |
| A neighbour from the same account, same day | yes | 3,168 | 0.690 |
| | no | 30,944 | 0.556 |

<sub>From `results/osnabrueck/recall_by_difficulty_megaloc.json`; EigenPlaces in the second figure and
`recall_by_difficulty_eigenplaces.json`.</sub>

1. **Viewing direction.** 14.5 % of the solvable queries have no neighbour facing the same direction: R@1 0.071.
   Without them MegaLoc would be at 0.653.
2. **Time, not evenly.** 8–30 days is the best class, over a year the worst — but 0–7 days is only mid-table.
3. **Density.** Only clearly better from 51 neighbours on. Many neighbours help; few do not measurably hurt.

**Origin of the top-1 match.** The JSON also counts where the successful matches come from (`herkunft_top1`):
the share from the same account, of those within 180 days (`anteil_dublette`), time gap, viewing angle.
`anteil_dublette` is exactly the set the Hard filter discards — the [city comparison](#city-comparison--city_comparisonpy)
uses it to recompute the Hard recall exactly.

**Limitations.** No bootstrap; single classes can depend on few drives. Do not interpret differences below about 0.1.
"Same account" means the uploading account (`creator_id`) — an agency is many cameras under one account.

---

### Districts — `recall_by_district.py`

**Question.** Where in the city does the system fail — and is it due to the reference density?

**In short.** Districts range from 0.07 to 0.83. Images per km² do not explain this; the reasons lie in the captures.

**Method.** Every query assigned to an OSM district by point-in-polygon; per district R@1 and reference images per km².
Districts with fewer than 100 solvable queries in grey.

```bash
python experiments/recall_by_district.py
```

<p align="center">
  <img src="results/osnabrueck/recall_by_district_megaloc.png" width="100%" alt="R@1 per district, MegaLoc">
</p>
<p align="center">
  <img src="results/osnabrueck/recall_by_district_eigenplaces.png" width="100%" alt="R@1 per district, EigenPlaces">
</p>

| District | Queries | Drives | solvable | R@1 MegaLoc | Reference/km² |
|---|---:|---:|---:|---:|---:|
| Atter | 2,630 | 14 | 91 % | **0.83** | 244 |
| Schinkel | 311 | 4 | 48 % | 0.76 | 700 |
| Nahne | 4,956 | 16 | 88 % | 0.69 | 1,116 |
| Pye | 6,160 | 24 | 98 % | 0.67 | 287 |
| Innenstadt | 7,001 | 39 | 38 % | 0.67 | 2,623 |
| Darum-Gretesch-Lüstringen | 744 | 3 | 83 % | 0.63 | 95 |
| Schölerberg | 829 | 16 | 94 % | 0.62 | 565 |
| Fledder | 1,852 | 13 | 68 % | 0.61 | 599 |
| Voxtrup | 2,967 | 18 | 92 % | 0.59 | 228 |
| Wüste | 6,094 | 36 | 64 % | 0.57 | 1,915 |
| Dodesheide | 2,307 | 14 | 52 % | 0.48 | 412 |
| Kalkhügel | 2,853 | 14 | 63 % | 0.40 | 905 |
| Weststadt | 2,961 | 18 | 55 % | 0.38 | 970 |
| Westerberg | 744 | 14 | 26 % | 0.35 | 381 |
| Hafen | 2,014 | 20 | 54 % | 0.29 | 609 |
| Haste | 2,404 | 15 | 50 % | 0.23 | 206 |
| Hellern | 2,184 | 13 | 70 % | 0.21 | 147 |
| Sonnenhügel | 1,634 | 12 | 7 % | 0.18 | 403 |
| Sutthausen | 414 | 3 | 100 % | **0.07** | 230 |

<sub>From `results/osnabrueck/recall_by_district_megaloc.json`; 23 districts, four of them with fewer than 100 solvable queries.</sub>

- **Factor 12 with the same encoder.** The city value 0.568 is an average over very different neighbourhoods.
- **Density does not explain it:** ρ = 0.27 (p = 0.27). The city centre (Innenstadt) has the densest reference and only 38 % solvable queries.
- **The map shows the data, not the encoder:** EigenPlaces (second map) ranks the districts practically the same way (Spearman 0.96).
- **Reasons in the recordings:**
  - *Haste* (0.23): only 29 % of the solvable queries have a neighbour with the same viewing direction; median gap three years.
  - *Hellern* (0.21): median gap 338 days — a different season.
  - *Sutthausen* (0.07): 35 neighbours, six days apart, matching viewing direction — and still 208 of 414
    queries end up in Hellern, 4 km away. A single drive confuses one residential area with another.

**Limitations.** Districts with three drives show the fate of one drive, not of one place.

---

### Confusion atlas — `confusion_atlas.py`

**Question.** Where does the system guess when it is wrong?

**In short.** Either just off or in a different neighbourhood, hardly anything in between — and mostly in residential streets.

**Method.** For every solvable query whose top-1 is beyond 25 m, an arrow from the true to the estimated position,
plus district pairs and the road type from OSM.

```bash
python experiments/confusion_atlas.py
```

![Misses as arrows from the true to the estimated position](results/osnabrueck/confusion_atlas_megaloc.png)

**Result.** MegaLoc, 14,726 misses among 34,112 solvable queries:

| Error | Share |
|---|---:|
| under 100 m — same street | 45 % |
| 100 m to 1 km | 11 % |
| over 1 km — a different neighbourhood | 44 % |

The most frequent district pairs (true → estimated):

| Pair | n | Median |
|---|---:|---:|
| Voxtrup → Nahne | 240 | 540 m |
| Sutthausen → Hellern | 208 | 4,239 m |
| Hellern → Nahne | 147 | 6,894 m |
| Wüste → Weststadt | 141 | 63 m |
| Hellern → Kalkhügel | 139 | 3,590 m |

- **No dominant confusion:** no pair carries more than 1.6 % of the errors.
- **Targets on the outskirts:** Nahne and Hellern — residential and commercial areas that resemble each other.

![Misses by road type](results/osnabrueck/confusion_atlas_megaloc_strassentyp.png)

**The motorway is not the problem — even though the map suggests it.** A few hundred long arrows along
the A30 visually cover ten thousand short ones.

| Road type | R@1 | Share of solvable queries | Share of errors over 1 km |
|---|---:|---:|---:|
| Motorway | 0.570 | 32 % | 23 % |
| Main road | 0.576 | — | — |
| Residential and other | 0.565 | 51 % | 56 % |

<sub>From `results/osnabrueck/confusion_atlas_megaloc.json`.</sub>

**Limitations.** Solvable queries only. Over all queries (08) the median of a wrong top-1 is 1.7 km.

---

## Post-processing of the top-k

All methods here receive the same match list and re-rank it or
combine it. Common finding: **the errors are coherent** —
in a gross confusion the other matches are in the wrong place as well
([confusion atlas](#confusion-atlas--confusion_atlaspy)).

### Aggregation — `localization_aggregation.py`

**Question.** Does the coordinate get better if you average or cluster the top-10 instead of taking only the best match?

**In short.** No — five methods, all worse than top-1.

**Method.** Centroid (raw and spread), clustering, snap (best match of the strongest group), gated (group only
at 70 % agreement). Measured over all 53,414 queries: share under 25 m and median error.

```bash
python experiments/localization_aggregation.py
```

<p align="center">
  <img src="results/osnabrueck/localization_aggregation_megaloc.png" width="49%" alt="Aggregation methods against top-1, MegaLoc">
  <img src="results/osnabrueck/localization_aggregation_eigenplaces.png" width="49%" alt="Aggregation methods against top-1, EigenPlaces">
</p>

| Method | MegaLoc | EigenPlaces |
|---|---|---|
| **Top-1** | **0.363 / 94 m** | **0.309 / 364 m** |
| Clustering | 0.318 / 444 m | 0.265 / 656 m |
| Snap | 0.314 / 445 m | |
| Gated | 0.362 | |
| Centroid, spread | 0.267 / 575 m | |
| Centroid, raw | 0.206 / 961 m | |

<sub>Share of **all** queries under 25 m / median error; that is why top-1 is 0.363 here and not the benchmark R@1 0.568,
which counts solvable queries only.</sub>

- Even gated, which only departs from top-1 when agreement is high, only approaches top-1 from below.
- Reason: in gross confusions the strongest group lies in the wrong place as a whole — consensus confirms the error.

---

### Neighbouring frames — `sequence_retrieval.py`

**Question.** Single images are ambiguous, drives are not. Does it help to sum the match lists of neighbouring images?

**In short.** No, it hurts slightly.

**Method.** Match lists of the ±W neighbours within a drive summed with a triangular weight. The default is
`eigenplaces_pcaw512`; this is the row `seq3` in the benchmark.

```bash
python experiments/sequence_retrieval.py
```

| Window | R@1 | R@5 | R@20 |
|---|---:|---:|---:|
| single | 0.507 | 0.641 | 0.727 |
| ±1 | 0.507 | 0.644 | 0.731 |
| ±3 | 0.498 | 0.645 | 0.736 |
| ±5 | 0.488 | 0.641 | 0.739 |

- ±3 against single: −0.009 [−0.017, −0.001], established.
- Two reasons the method does not separate: neighbouring images make the same error, **and** it needs the same
  reference image in several lists — rare with 15 % reference.

---

### Drive as a path — `sequence_hmm.py`

**Question.** Does the drive help if you read it as a route instead of a sum?

**In short.** Yes — the only post-processing that beats top-1 with an established difference.

**Method.** A hidden Markov model. The candidates may differ from image to image; they only have to fit together geometrically.

| | |
|---|---|
| States | the top-k of an image |
| Emission | `β ×` cos from the search |
| Transition | does the distance between two candidates fit the elapsed time? `−\|d − v·Δt\| / σ` |
| Result | probability per candidate (forward-backward) and best path (Viterbi) |

A candidate 6 km off drops out, because nobody drives 6 km in 0.17 s. The speed (12.5 m/s) comes from the
**reference** drives; the positions of the queries are not used anywhere. β = 30 and σ = 25 m were fixed before the run —
this one setting is reported, not the best one from a grid.

```bash
python experiments/sequence_hmm.py --method megaloc
```

| | R@1 | R@5 | R@10 | R@20 | Path R@1 |
|---|---:|---:|---:|---:|---:|
| MegaLoc | 0.568 | 0.676 | 0.719 | 0.763 | |
| MegaLoc, HMM | **0.598** | **0.690** | **0.725** | **0.764** | 0.603 |
| EigenPlaces | 0.484 | **0.608** | **0.650** | **0.695** | |
| EigenPlaces, HMM | **0.501** | 0.603 | 0.633 | 0.675 | 0.511 |

- **Established:** MegaLoc +0.030 [+0.020, +0.041], EigenPlaces +0.017 [+0.006, +0.030].
- **Not for free:** with EigenPlaces the re-ranking costs R@5 to R@20.
- **Small, as expected:** the HMM only catches isolated outliers. A whole drive on the wrong street is just as consistent as a path.

**Limitations.** Forward-backward and Viterbi are checked in [`tests/test_sequence_hmm.py`](../tests/test_sequence_hmm.py) against a
complete enumeration.

---

### Geometric verification — `geometric_verification.py`

**Question.** Places that are globally similar but locally different are the typical gross error. Does it help to re-check the top-20 with local
features?

**In short.** Not at 25 m. It sharpens the position to a few metres but does not find the right place more often.

**Method.** SuperPoint finds keypoints, LightGlue matches them, RANSAC checks them against the geometry of two
cameras. The candidates are re-ranked by the number of consistent points (inliers); below 15 the old
order stays. All five parameters were fixed before the run:

| Parameter | Value | Origin |
|---|---|---|
| `--top-k` | 20 | project choice |
| `--min-inliers` | 15 | project choice |
| `--ransac-px` | 3.0 | OpenCV default |
| `--max-keypoints` | 1,024 | LightGlue's recommendation for speed |
| `--max-side` | 640 | project choice |

```bash
python experiments/geometric_verification.py --n-queries 0
```

Needs the images and a GPU; saves the inliers every two minutes and resumes after an interruption.
`bootstrap_ci.py` recomputes the row from the saved inliers without matching again.

**Result.** Osnabrück, MegaLoc, paired:

| | MegaLoc | + verification | Diff | 95 % |
|---|---:|---:|---:|---|
| R@1 | 0.568 | 0.539 | −0.029 | [−0.062, +0.000] |
| R@5 | 0.676 | 0.678 | +0.002 | [−0.018, +0.024] |
| R@10 | 0.719 | 0.726 | +0.007 | [−0.005, +0.019] |
| R@20 | 0.763 | 0.763 | 0 | — |

| R@1 per threshold | 5 m | 10 m | 25 m | 50 m | 100 m |
|---|---:|---:|---:|---:|---:|
| Difference | +0.013 | +0.010 | −0.029 | −0.034 | −0.039 |

<sub>From [`results/osnabrueck/bootstrap_ci.json`](results/osnabrueck/bootstrap_ci.json) and
`results/osnabrueck/evaluation/megaloc{,_gv20}.json`.</sub>

- R@20 necessarily stays the same: re-ranking happens only within the top-20.
- **The R@1 loss is a borderline case:** the upper bound is +0.0003, so it is just not established.
- **Interpretation, not checked individually:** inliers measure image overlap, not sameness of place. The candidate with the largest
  shared view moves up — often very close, but sometimes a wrong place with recurring geometry.
- **The threshold hardly separates:** 87 % of all candidate pairs have 15 inliers or more (median 40).
- **Runtime:** 840,200 image pairs in 8.6 hours, about 27 per second. The other five cities would have cost about 57 hours.

**Limitations.** One setting, one city, pure re-ranking by inliers. A higher threshold could be recomputed from the
saved inliers — on Osnabrück, however, that would be tuning on the reported sample.
SuperPoint is licensed for non-commercial research only ([NOTICE.md](../NOTICE.md)).

---

### Detections — `detection_rerank.py`

**Question.** Mapillary detects objects in every image (signs, street lights, cars). Does that add information that is missing
from the descriptor?

**In short.** No. The signal exists, but it is weaker than the descriptor and already contained in it.

**Method.** 2,000 queries with correct **and** wrong candidates in the top-10; per image a
weighted histogram of the detected classes. Measured: how well it separates right from wrong (AUC), and R@1 after
re-ranking. EigenPlaces.

```bash
python experiments/detection_rerank.py
```

| | all classes | without cars and persons |
|---|---:|---:|
| AUC detections | 0.562 | 0.559 |
| AUC descriptor | 0.735 | 0.735 |

| Weight of the detections | 0 | 0.05 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|
| R@1 (1,286 queries) | 0.7551 | 0.7558 | 0.7496 | 0.6998 |

- The best value is a single query above the baseline.
- Missing detections count as neutral, not as similarity 0 — otherwise the measurement would penalise missing data.

**Limitations.** Own sample, not comparable with the main table. The detections are versioned
(`cache/detections.jsonl`). The coverage pilot (`detection_probe.py`): 85 to 94 % of the images have detections.

---

## Confidence

### Rejection — `rejection_curve.py`

**Question.** How much better does the answer get if the system may stay silent at low confidence — and which measure works?

**In short.** The raw cos of the best match is the best measure.

**Method.** Three measures: cos of the best match, gap to rank 2 (margin), agreement of the top-10 (share within
25 m of rank 1). The threshold is lowered step by step and precision is plotted against the share of answered queries.

```bash
python experiments/rejection_curve.py
```

<p align="center">
  <img src="results/osnabrueck/rejection_curve_megaloc.png" width="100%" alt="Precision against coverage, MegaLoc">
</p>
<p align="center">
  <img src="results/osnabrueck/rejection_curve_eigenplaces.png" width="100%" alt="Precision against coverage, EigenPlaces">
</p>

**Result.** MegaLoc, precision at 100 / 80 / 50 / 20 % answered queries:

| View | Measure | 100 % | 80 % | 50 % | 20 % | Area |
|---|---|---:|---:|---:|---:|---:|
| solvable queries | cos | 0.568 | **0.689** | 0.793 | 0.887 | 0.791 |
| | margin | 0.568 | 0.618 | 0.724 | 0.860 | 0.741 |
| | agreement | 0.568 | 0.597 | 0.706 | 0.798 | 0.706 |
| all queries | cos | 0.363 | **0.451** | 0.673 | 0.828 | 0.659 |
| | margin | 0.363 | 0.410 | 0.526 | 0.766 | 0.580 |

<sub>From [`results/osnabrueck/rejection_curve_megaloc.json`](results/osnabrueck/rejection_curve_megaloc.json).
The concatenation EigenPlaces + MegaLoc is level with it (0.695 at 80 %, area 0.790).</sub>

- **Rule of thumb for MegaLoc:** cos ≥ 0.30 → 82 % correct (37 % of the solvable queries answered); cos ≥ 0.20 → 75 % (69 %).
- The margin (gap to rank 2) is less useful than the raw cos — unlike what is usual for classifiers.
- **EigenPlaces** shows the same order (second figure): cos 0.568 at 80 %, without rejection 0.484.
- That is why `locate.py` and the demo report cos as the confidence.
- cos is only comparable within one encoder: CLIP gives even wrong matches about 0.90.

**Limitations.** In operation the system does not know the reference. cos recognises unsolvable queries only partly
(all queries, 80 %: 0.451 instead of 0.363). The thresholds apply only to MegaLoc.

---

## Six cities

### City selection — `city_coverage.py`

**Question.** Which city is suitable as the second one — before spending days on images and embeddings?

**In short.** What matters is whether **every** street has an image, not how many images there are — and that no single account recorded everything.

**Method.** From metadata only, one minute per city: image points via the same tiles as 01, road network from OSM.
**Street coverage** = share of the street length with an image within 25 m, split into major roads and
residential streets. Plus drives, accounts, image age, share of panoramas.

```bash
python experiments/city_coverage.py "Heidelberg, Germany"
```

**Result.** 50 cities measured, 45 with street coverage. The candidates:

| City | Images/km² | Coverage | Residential | Largest account | Since 2022 | Computed |
|---|---:|---:|---:|---:|---:|:---:|
| Jena | 6,115 | **99 %** | 99 % | 51 % | 45 % | ✓ |
| Gütersloh | 5,363 | **99 %** | 99 % | **80 %** | 75 % | ✗ one account |
| Würzburg | 4,895 | **98 %** | 98 % | 51 % | 28 % | ✓ |
| Mainz | 6,596 | 95 % | 94 % | **74 %** | 44 % | ✗ one account |
| Halle (Saale) | 6,787 | 92 % | 91 % | 30 % | 87 % | open |
| Heidelberg | 4,879 | 87 % | 83 % | 33 % | 36 % | |
| Erlangen | **7,943** | 70 % | **63 %** | 36 % | 49 % | |
| Kaiserslautern | 2,792 | 80 % | 76 % | 46 % | | ✓ |
| Karlsruhe | 3,504 | 75 % | 68 % | 7 % | | ✓ |
| Fürth | 3,030 | 72 % | 66 % | 30 % | | ✓ |
| Osnabrück | 2,808 | 44 % | **38 %** | 47 % | 84 % | ✓ |

<sub>From [`results/city_coverage.json`](results/city_coverage.json).</sub>

- **Density is not coverage:** Erlangen has the most images per km² and covers only 63 % of its residential streets.
- **Osnabrück explains itself:** 38 % of the residential streets covered → 36 % of the queries without reference.
- **Gütersloh drops out,** although it looks perfect: one account provides 80 % of the images and 94 % of the drives.
  Query and reference would almost always be the same camera — a good result could not be separated from the camera.
  How much that matters is shown by Osnabrück: neighbour from the same account on the same day 0.690 instead of 0.556.

**Limitations.** No number in this table predicts the recall: Würzburg has the second-highest coverage and the
lowest recall. The computed cities are not free of large accounts either (Jena, Würzburg: 51 %).

---

### City comparison — `city_comparison.py`

**Question.** Is a result from Osnabrück a result about the method — or about Osnabrück?

**In short.** The level depends on the city, the gap between the encoders hardly does.

**Method.** Reads only versioned files (evaluations, intervals, difficulty profile, city coverage) and therefore runs in
any fresh clone. Cities **cannot be compared paired** — their queries are different; what remains is the
comparison of independent estimates with wide intervals.

```bash
python experiments/city_comparison.py
```

#### What carries over

```text
Was sich uebertraegt   R@1 bei 25 m (MegaLoc voll = volle Referenz)
Delta = MegaLoc - EigenPlaces, gepaart ueber dieselben Anfragen, 95-%-Intervall aus dem Bootstrap

Stadt            Anfragen Fahrten  EigenPl  MegaLoc   Delta [95 %]              voll   Delta voll [95 %]
--------------------------------------------------------------------------------------------------------
osnabrueck         53,414     198    0.484    0.568   +0.084 [+0.055, +0.116]  0.798   +0.096 [+0.070, +0.122]
fuerth             24,994     239    0.473    0.549   +0.076 [+0.057, +0.096]  0.699   +0.072 [+0.052, +0.092]
karlsruhe          87,181     584    0.306    0.419   +0.114 [+0.095, +0.134]  0.640   +0.132 [+0.111, +0.155]
kaiserslautern     58,916     249    0.552    0.651   +0.099 [+0.079, +0.122]  0.810   +0.077 [+0.058, +0.096]
wuerzburg          60,203     300    0.263    0.336   +0.073 [+0.045, +0.106]  0.476   +0.092 [+0.057, +0.130]
jena              114,558     677    0.332    0.417   +0.084 [+0.073, +0.096]  0.622   +0.102 [+0.091, +0.113]
  Niveau MegaLoc: 0.336 bis 0.651 (Spanne 0.315); Abstand: +0.073 bis +0.114 (Spanne 0.041)
  MegaLoc vorn mit Intervall ueber 0: 12 von 12 (Staedte x Protokolle)
```

<sub>Verbatim output of `python experiments/city_comparison.py`, first part (in German: Stadt = city, Anfragen = queries,
Fahrten = drives, voll = full reference, Niveau = level, Abstand = gap, Spanne = range). The intervals come from
`results/<city>/bootstrap_ci{,_fullref}.json`.</sub>

- **Level:** MegaLoc 0.336 to 0.651 — range 0.315.
- **Gap:** +0.073 to +0.114 — range 0.041, about eight times narrower. With the full reference 0.061.
- **MegaLoc ahead in all six cities and both protocols;** all twelve intervals exclude 0.
- **Same order of cities** for both encoders, except for Karlsruhe and Jena (MegaLoc 0.419 against 0.417).
- **Not constant:** Fürth and Karlsruhe do not overlap with the full reference.

#### What distinguishes the cities

```text
Encoder megaloc  |  R@1 bei 25 m  |  6 Staedte

Stadt              Abd  loesbar   Pano   Dubl   Tage   Alle    Hard   voll   ±boot
----------------------------------------------------------------------------------
osnabrueck        0.44    63.9%   0.0%  11.8%    317  0.568  -0.025  0.798   0.096
fuerth            0.72    62.0%   3.0%  39.6%    291  0.549  -0.141  0.699   0.059
karlsruhe         0.75    70.8%  17.9%  15.4%    408  0.419  -0.057  0.640   0.058
kaiserslautern    0.80    85.6%   0.3%  35.8%    279  0.651  -0.204  0.810   0.045
wuerzburg         0.98    45.5%   8.7%  17.9%    627  0.336  -0.034  0.476   0.059
jena              0.99    52.9%   0.3%  24.1%   2188  0.417  -0.010  0.622   0.033
```

<sub>Verbatim output of `python experiments/city_comparison.py`, second part.
Abd = street coverage, loesbar = solvable, Pano = share of panoramas, Dubl = share of correct top-1 matches from the same account within
180 days, Tage = median days between query and correct match, Alle = all queries (benchmark R@1), Hard = penalty from the hard ground truth,
voll = full reference, ±boot = half width of the 95 % interval.</sub>

**Würzburg:** 98 % coverage, but only 45.5 % solvable and the lowest recall. Coverage counts the entire collection;
the reference is 15 % of the drives. A street driven only once usually ends up in `train` — and still counts
as covered. On top of that the images are old (median 627 days between query and match).

#### The hard penalty, decomposed exactly

The hard ground truth costs between 0.010 (Jena) and 0.204 (Kaiserslautern). The share of matches from the same account
alone does not predict that (ρ = −0.54). The reason: two different quantities are both called "duplicate".

| | counts | Source |
|---|---|---|
| **d** | share of correct **matches** that come from the same account within 180 days | `recall_by_difficulty.py` |
| **1 − r** | share of solvable **queries** that were solvable **only** through such images | the two "solvable" figures from 07 |

The filter removes both at once — numerator and denominator. This yields an identity:

```math
\frac{\text{penalty}}{\mathrm{R@1}} = -\,\frac{d-(1-r)}{r}
```

| City | d | 1 − r | d − (1−r) | rel. penalty | Residual |
|---|---:|---:|---:|---:|---:|
| Kaiserslautern | 35.8 % | 6.6 % | +0.292 | −0.313 | 0.000 |
| Fürth | 39.6 % | 18.7 % | +0.209 | −0.257 | 0.000 |
| Karlsruhe | 15.4 % | 2.0 % | +0.134 | −0.137 | 0.000 |
| Würzburg | 17.9 % | 8.6 % | +0.093 | −0.102 | 0.000 |
| Osnabrück | 11.8 % | 7.7 % | +0.041 | −0.044 | 0.000 |
| Jena | 24.1 % | 22.3 % | +0.018 | −0.023 | 0.000 |

- **Jena:** many matches from the same account, but there is often no alternative either — hardly any penalty. The images
  from the same account there are mostly second cameras of the same rig ([twins per city](#twins-per-city)).
- **Kaiserslautern:** needs such images for only 6.6 % of the queries, but takes 36 % of its matches from them — overuse, largest penalty.
- **So the hard penalty measures how much a system uses images from the same account beyond what the dataset forces.**
- The `Residual` column checks that `recall_by_difficulty.py` and 07 measure the same thing: 0.000 in all six cities.
- Checked in advance: Osnabrück's d was predicted from the evaluation JSON (11.8 %) before the script ran there — measured 11.8 %.

#### Twins per city

How much of the full reference consists of rediscovered copies? `city_comparison.py` reads the results of
[`zwillinge.py`](#twin-drives--zwillingepy) for this:

```text
Zwillingsfahrten (selbes Konto, <= 60 s): R@1 mit und ohne
Stadt                Zw   Alle   ohne  Zw voll  Kopie   voll   ohne  Sprung   ohne
----------------------------------------------------------------------------------
osnabrueck         3.4%  0.568  0.550    28.2%  27.4%  0.798  0.690  +0.229 +0.139
fuerth             0.9%  0.549  0.548     6.0%   5.0%  0.699  0.692  +0.149 +0.144
karlsruhe          4.7%  0.419  0.394    18.4%  17.9%  0.640  0.577  +0.220 +0.183
kaiserslautern     2.6%  0.651  0.647     7.4%   4.7%  0.810  0.807  +0.159 +0.159
wuerzburg          1.4%  0.336  0.323     7.0%   6.6%  0.476  0.426  +0.141 +0.103
jena               6.4%  0.417  0.425    35.8%   4.3%  0.622  0.638  +0.206 +0.212
```

<sub>Verbatim output, MegaLoc. Zw = queries with a twin within 25 m; Kopie = copy, those of them with the same
viewing direction, as a share of all queries; Sprung = jump, voll − Alle; ohne = without twins.</sub>

- **Without copies Osnabrück is not an outlier:** jump +0.139 instead of +0.229, right among the others (+0.103 to +0.212).
- **The order with the full reference changes:** Kaiserslautern 0.807 ahead, then Fürth 0.692 and Osnabrück 0.690;
  Karlsruhe falls behind Jena.
- **Predicted in advance** was: twin share ↔ jump, ρ = +0.77, p = 0.10 — not established. **Afterwards** the
  copy share orders the cities exactly by their loss (ρ = −1.00, p = 0.003); found as an explanation, not passed as a test.

#### Panoramas

360° queries are hard: R@1 0.06 to 0.24 against 0.35 to 0.65 on the others. Recomputed exactly from the
two evaluations of 07:

| City | Panorama queries | R@1 panorama | R@1 others | Penalty |
|---|---:|---:|---:|---:|
| Karlsruhe | 8,024 | 0.220 | 0.449 | 0.229 |
| Würzburg | 2,412 | 0.170 | 0.352 | 0.182 |
| Kaiserslautern | 253 | 0.055 | 0.654 | 0.599 |
| Fürth | 192 | 0.208 | 0.553 | 0.345 |
| Jena | 85 | 0.235 | 0.417 | 0.182 |

Reliable only in Karlsruhe and Würzburg (four-digit case numbers).

> [!NOTE]
> **Until 18 September this section contained a wrong finding.** The penalty had been computed with the panorama share of *all images*
> instead of the *solvable queries*. As a result Karlsruhe was at 0.168 instead of 0.229, and the apparent
> agreement with Würzburg (0.184) was a coincidence.

#### A prediction that did not hold

Assumption: street coverage predicts the measurement uncertainty (patchy coverage → some drives find nothing
→ wide intervals).

![MegaLoc: half width of the 95 % interval against street coverage](results/city_comparison.png)

| Cities | ρ | p (exact) |
|---:|---:|---:|
| 4 | −1.00 | 0.083 |
| 5 | −0.40 | 0.517 |
| 6 | −0.83 | 0.058 |

- For the fifth city ±boot ≤ 0.045 had been predicted in advance — measured 0.059.
- **With so few cities a rank correlation fluctuates wildly.** With four cities even a perfect order is not
  significant (p = 2/4! = 0.083) — regardless of the data.
- Not decidable: the number of drives gives the same ρ = −0.83 and is even the more plausible candidate, because the
  bootstrap resamples drives.

**Limitations.** n = 6. Five cities only with MegaLoc and EigenPlaces. `scipy.stats.spearmanr` returns p = 0 for perfect
monotonicity; the script therefore counts the permutations exactly (up to n = 8).

---

## Cost

### Runtime and memory — `timing.py`

**Question.** What does each encoder cost — for encoding, for the search, in memory?

**In short.** **Best trade-off: MegaLoc whitened to 512** — 0.028 less R@1 than MegaLoc, but a 16.5 times
smaller index and a four times faster search.

**Method.** Encoding: 200 fixed images, including loading, after a warm-up run. Search: exact FAISS index over the
48,321 reference images, 1,000 queries in blocks of 256, median of five rounds.

```bash
python experiments/timing.py --skip-search
```

```bash
python experiments/timing.py --skip-encode
```

![Encoding throughput and search time against Recall@1](results/osnabrueck/timing.png)

```text
Encoder                        Dim     R@1  ms/Anfrage  Index MB  Bilder/s  alle Bilder
---------------------------------------------------------------------------------------
eigenplaces_megaloc_concat    1024   0.572        0.11       189         -            -
megaloc                       8448   0.568        0.74      1557      26.0        3.6 h
megaloc_pca512                 512   0.545        0.18        94         -            -
megaloc_pcaw512                512   0.541        0.18        94         -            -
eigenplaces_pcaw512            512   0.507        0.18        94         -            -
eigenplaces                   2048   0.484        0.21       378      33.8        2.7 h
eigenplaces_pca512             512   0.481        0.19        94         -            -
eigenplaces_pcaw2048          2048   0.459        0.21       378         -            -
mixvpr                        4096   0.426        0.44       755      77.9        1.2 h
mixvpr_pcaw512                 512   0.424        0.18        94         -            -
mixvpr_pca512                  512   0.408        0.17        94         -            -
anyloc_pcaw4096               4096   0.321        0.44       755         -            -
anyloc_pcaw512                 512   0.263        0.18        94         -            -
anyloc                        4096   0.204        0.45       755      14.6        6.3 h
anyloc_pca512                  512   0.175        0.18        94         -            -
clip_pcaw512                   512   0.105        0.18        94         -            -
clip_pca512                    512   0.074        0.18        94         -            -
clip                           512   0.073        0.19        94     178.6        31 min

  R@1 bei 25 m (Benchmark, 07). Suche: FAISS flach auf der CPU, Index ueber die Referenz.
  'alle Bilder' = 332,868 Bilder encodieren. Abgeleitete Varianten (PCA, Verkettung)
  encodieren mit dem Netz ihres Basis-Encoders.
  anyloc: ohne PCA-Projektion gemessen
```

<sub>Output of `python experiments/timing.py --skip-encode --skip-search`, read from
[`results/osnabrueck/timing.json`](results/osnabrueck/timing.json). In German: ms/Anfrage = ms per query, Bilder/s = images per second,
alle Bilder = time to encode all 332,868 images; derived variants (PCA, concatenation) encode with the network of their base encoder.
Search on the CPU with 8 threads; encoded on an Apple M1 Pro, AnyLoc (fp16) on a CUDA GPU. Image sizes: CLIP 224 px, MixVPR 320, MegaLoc and AnyLoc 322, EigenPlaces 512.</sub>

- **Winner: `megaloc_pcaw512`** — 0.028 less R@1 than MegaLoc, 94 instead of 1,557 MB index, 0.18 instead of 0.74 ms per query.
- **The concatenation** is at the top, but needs two networks for encoding and is not established as ahead of MegaLoc (+0.004 [−0.005, +0.014]).
- **Index linear, search not:** from 512 to 8448 dimensions 16.5 times more memory, but only 4 times slower.
- **Irrelevant at 48,321 reference images** (10 s against 39 s for all queries); at one million it would be a 32 GB against a 2 GB index.
- **Encoding depends on the network and the image size,** not on the PCA: the projection is a matrix product.

**Memory.** Everything follows from `images × dimension × 4 bytes`. MegaLoc: Osnabrück 11.2 GB, Jena 23.6 GB.

| Step (MegaLoc) | Osnabrück | Jena |
|---|---:|---:|
| 04, writing embeddings | 22.5 → not needed | 47.2 → not needed |
| 05, adapter | 7.8 GB | 16.2 GB |
| 06, search | 5.1 GB | 11.0 GB |
| `database_density.py`, last stage | 11.2 → 6 GB | 23.6 → 8.3 GB |

The arrows are two fixes after a crash in Jena: 04 now writes directly to disk (memmap) instead of
into memory first, and the search over large references runs in blocks.

---

## Tools

### Example images — `beispielbilder.py`

A contact sheet per scene type, then the selection for the README.

```bash
python experiments/beispielbilder.py --szenen
```

Every saved figure with Mapillary images records the author of each image in
`results/<city>/figures/demo/QUELLEN.md` ([`src/quellen.py`](../src/quellen.py)).
