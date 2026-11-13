# Flora experiment results

Updated: 13 Nov 2026. Sections 1–3 report the current main method (3-seed means). The older sections below ("Earlier method") are the 11 Nov snapshot and use seed 1 only.

Protocol: paper setting (1000 + 200 iterations), Colab L4, ZSL, `best` = best checkpoint on unseen. Flora is compared with the numbers reported in its paper; Flora itself was not rerun, apart from the STAR `none` / `compose` runs listed for reference.

## 1. Current main method: SeqKG + COA-Fuse (v1)

- **SeqKG**: KG v2 (82 primitives) plus 29 order primitives and 14 reverse-order forbids; class pairs with identical rules 7 → 0.
- **Signed Compose**: class semantics = (1+λ)·mean(require) − λ·mean(order-forbid), λ = 1. No learned parameters.
- **COA-Fuse**: Center (remove class-mean of the KG term), Orthogonal (remove the part along the class text), Adaptive (per-class β from text-vs-KG disagreement, clipped at 3β). No learned parameters.
- Code: `nesy4_seqkg.py`, `nesy4_1_forbid.py`, `nesy4_1_1_fuse.py`. Details: `progress/2026-11-12_flora_nesy_notes.md`.

### ZSL, STAR features (mean of seeds 1–3, best)

| Split | Flora paper | none (rerun) | compose v2 | SeqKG + forbid | **SeqKG + COA-Fuse** (mean ± std) | Δ vs paper |
|---|---:|---:|---:|---:|---:|---:|
| 55/5 | 85.8 | 85.14 | 85.28 | 85.16 | 85.24 ± 0.25 | −0.56 |
| 48/12 | 61.5 | 60.87 | 61.68 | 61.55 | **62.39 ± 1.27** | +0.89 |
| 110/10 | 80.7 | 78.42 | 79.70 | 80.14 | **81.10 ± 0.47** | +0.40 |
| 96/24 | 64.1 | 67.20 | 69.13 | 68.65 | 68.16 ± 1.23 | +4.06 |
| mean | | 72.91 | 73.95 | 73.88 | **74.22** | **+1.20** |

Per-seed COA-Fuse (s1 / s2 / s3): 55/5 85.09 / 85.09 / 85.53; 48/12 63.75 / 62.20 / 61.23; 110/10 80.63 / 81.10 / 81.57; 96/24 66.96 / 69.42 / 68.09.

Notes:
- The 96/24 gain over the paper is mostly the rerun itself (`none` is already +3.10). That split is unstable (best − last ≈ 8–12 points).
- COA-Fuse vs compose v2: +0.71 / +1.40 / −0.04 / −0.97 (48/12, 110/10, 55/5, 96/24); mean +0.28, inside seed noise. Only 110/10 is consistent (3/3 seeds).
- Centering alone (C-Fuse) gives 61.86 (48/12) and 80.94 (110/10). The shuffled-KG control (below) shows this is not a KG-independent shift: centering a shuffled KG gives 58.28 / 77.77.
- SeqKG + forbid is on par with compose (mean −0.08) but separates reversed-order class pairs: centered cosine 0.99 → 0.78–0.81.
- **The order claim cannot be tested by ZSL accuracy on the standard splits** (checked 13 Nov with `nesy4_1_1_3_diag.py`): in all four splits each reversed-order pair has only one unseen member (48/12 wear a shoe; 110/10 wear jacket, put something into a bag; 55/5 put on a hat/cap; 96/24 take off shoe / glasses / hat / bag), so in-pair confusion never happens in ZSL. Claim only the representational difference. A split with both members unseen would need new STAR features.

### ZSL, SynSE features (SeqKG + COA-Fuse, seeds 1–3, best)

| Split | s1 / s2 / s3 | mean ± std | Flora paper (SynSE) | Δ |
|---|---|---:|---:|---:|
| 55/5 | 85.32 / 84.66 / 86.50 | 85.49 ± 0.93 | 86.3 | −0.81 |
| 48/12 | 64.68 / 62.17 / 63.94 | 63.60 ± 1.29 | 65.3 | −1.70 |
| 110/10 | 81.37 / 80.71 / 82.27 | 81.45 ± 0.78 | 79.6 | **+1.85** |
| 96/24 | 66.78 / 66.30 / 65.16 | 66.08 ± 0.85 | 66.4 | −0.32 |
| mean | | | | −0.25 |

`last10` (mean of seeds): 82.87 / 61.65 / 79.00 / 60.56 for 55/5, 48/12, 110/10, 96/24. Seed-1 reference runs in the SynSE folder (none / compose, as found there): Δ best vs none −0.37 / +1.22 / +1.25 / +2.50, vs compose −0.23 / +0.15 / −0.50 / +0.18 (55/5, 48/12, 110/10, 96/24).

### GZSL

Not yet run for SeqKG + COA-Fuse (STAR, use `--eval-class-chunk 16`). The only GZSL numbers so far are in the "Earlier method" sections below.

### Control: shuffled KG (13 Nov, `nesy0_shuffle.py`, STAR, mean of seeds 1–3, best)

KG rows are permuted across classes (class c gets the rules of class π(c), no fixed points, π depends on the run seed); everything else is unchanged. Real-KG columns are the runs above.

| Fusion | 48/12 shuffled | 48/12 real | real − shuffled | 110/10 shuffled | 110/10 real | real − shuffled |
|---|---:|---:|---:|---:|---:|---:|
| raw add (SeqKG + forbid) | 57.95 | 61.55 | +3.60 | 77.79 | 80.14 | +2.35 |
| Center only (C-Fuse) | 58.28 | 61.86 | +3.58 | 77.77 | 80.94 | +3.17 |
| COA-Fuse | 58.48 | 62.39 | +3.91 | 77.96 | 81.10 | +3.14 |
| none (Flora, no KG) | 60.87 | | | 78.42 | | |

Per seed (s1 / s2 / s3) for COA-Fuse shuffled: 48/12 56.88 / 59.37 / 59.19; 110/10 77.81 / 76.89 / 79.17. The gap real − shuffled is about 3× the seed std. A wrong KG falls below no-KG (−2.4 to −2.9 at 48/12, −0.5 to −0.7 at 110/10), so the model reads the KG content, and Center / Orth / Adaptive do not rescue a wrong KG. Within the real KG, each added fusion step is small (+0.31 / +0.80 for Center over raw add, +0.53 / +0.16 for Orth + Adaptive over Center, at 48/12 / 110/10) and inside seed noise. The shuffle also corrupts the Adaptive step, so the drop contains both effects. Not run: 55/5 and 96/24.

### β sensitivity (13 Nov, C-Fuse = Center only, STAR, mean of seeds 1–3, best)

| β | 48/12 | 110/10 | mean of both | 48/12 − none | 110/10 − none |
|---|---:|---:|---:|---:|---:|
| 0.1 | 61.86 | 79.57 | 70.72 | +0.99 | +1.16 |
| **0.3** (fixed in advance) | 61.86 | 80.94 | **71.40** | +0.99 | +2.52 |
| 0.5 | 60.87 | **81.36** | 71.12 | 0.00 | +2.94 |
| 1.0 | 59.27 | 79.29 | 69.28 | −1.61 | +0.87 |
| none | 60.87 | 78.42 | 69.65 | | |

Inverted U on both splits; the good range is β ∈ [0.3, 0.5]. 110/10 peaks at 0.5, 48/12 at ≤ 0.3, and β = 1.0 hurts 48/12 (seeds 1, 2: −3.40, −3.16 vs none). The method is sensitive to β at the ±1-point level, so report it as a sensitivity analysis with β = 0.3 fixed beforehand (ZSL has no validation set). Per seed (s1 / s2 / s3), 48/12: β 0.1 63.69 / 61.01 / 60.89, β 0.5 61.96 / 59.80 / 60.86, β 1.0 58.71 / 57.95 / 61.14; 110/10: β 0.1 78.94 / 79.31 / 80.47, β 0.5 81.55 / 81.41 / 81.12, β 1.0 79.81 / 79.48 / 78.58. Not run for full COA-Fuse.

### Other 13 Nov analyses (details: `progress/2026-11-13_flora_nesy_notes.md`)

- **KG-only** (`nesy4_1_1_2_kgonly.py`, 48/12 seed 1, centered, no neighbours): 41.08 vs Flora text 61.5 (chance ≈ 8%).
- **Error diagnostic** (`nesy4_1_1_3_diag.py`, seed 1): top2 − top1 14.16 / 13.46, margin predicts errors (error ratio 2.34 / 23.7), but only 46.4% / 34.5% of wrong samples have the KG saying "more different than text" (criterion 50%). A margin-gated KG re-ranking loop is dropped; errors are skeleton-side. On 55/5 (3 seeds) almost all errors are reading ↔ writing and 90–94% of wrong samples have d > 0; on 96/24 (seed 1) 62–63%, with take off a hat/cap at ~20% accuracy (51% predicted as take off glasses). So criterion (c) passes on 55/5 and 96/24 but not on 48/12 and 110/10; 55/5 is a single pair, so this is not enough to revive the loop.
- **Name-free context around primitive sentences** (`nesy4_1_1_4_ctx.py`, seed 1, COA-Fuse 63.75 / 80.63): `action` 64.33 / 79.69, `person` 63.78 / 78.91. No gain; the template shifts all primitives along a common direction, which Center removes.

- **Retrieval → primitive → rule re-ranking** (`nesy4_1_1_5_retrieval.py`, 3 seeds, best): the 50 nearest seen training samples give a primitive profile (mean of their class rules), a fuzzy rule score (mean require − mean forbid) re-ranks the DiT top-3 with γ = 1. Result 46.20 (48/12) and 56.44 (110/10) vs COA-Fuse 62.39 / 81.10; with shuffled rules 42.04 / 52.77. Rule-only accuracy 31.2% at 48/12; no cell of the γ × top grid beats the base (best: γ 0.25, top-2, 54.60 vs base 57.22 at the same eval). The profile copies the rules of the nearest seen class (e.g. walking apart → staggering → falling). Real rules beat shuffled by ~4, so the signal is real but much weaker than DiT. **Negative; not pursued.**
- **Primitive probe** (`nesy0_probe.py`, no Flora training): measures whether the 256-d features carry primitive information that transfers to unseen classes (AUC on unseen test samples, seen-test reference, shuffled-rule control, kNN comparison). Not run yet.

## 2. Negative results (STAR, 48/12 unless noted)

- Every use of the KG on the skeleton side or at decision time failed: logic head, primitive tokens, margin-gated re-ranking (diagnostic), retrieval + rule re-ranking (−16 / −25). Only the semantic-side use works, confirmed by the shuffled-KG control.

- Compose from a joined class sentence (`--kg-compose-src text`): 62.53 vs compose 62.87 (seed 1; seeds 2–3 and 110/10 not run).
- Name-free context around primitive sentences: no gain (see above).

- ESA over per-class KG triples (`nesy7_esa.py`), Δ best vs compose: −5.99 (sim 1), −7.84 (sim 0), −0.82 (sim 0.05 + reg 1). R-GCN (`nesy6_gcn.py`) behaves the same: a learned delta on the KG memorises seen classes.
- Text+IDF+gate ≈ compose after 3 seeds; fixed mix (0.5 text + 0.5 IDF, no gate): 61.37 vs 61.68.
- Forbid of every kind (`--forbid-order-only 0`): unstable (48/12 −1.79, 110/10 +1.58 but last10 −0.98); dropped.
- Earlier (10 Nov): neighbors (hurts), logic head, hard negatives, primitive tokens, plain SeqKG: no gain.

## 3. Still to do

1. Shuffled-KG control on 55/5 and 96/24 if it goes into the paper; β sensitivity for full COA-Fuse (done for C-Fuse only).
2. GZSL for SeqKG + COA-Fuse.
3. ~~Per-pair accuracy for the reversed-order pairs~~: not measurable on the standard splits (see section 1).

---

# Earlier method (11 Nov snapshot, seed 1 only)

## Method under test

**SeqKG Text+IDF+gate** is the SeqKG Compose extension that fuses:

- text-compose of each class's required primitives;
- IDF-weighted primitive compose; and
- a learned, per-class semantic gate trained in Phase 1.

## ZSL (completed)

### STAR features

| Split | Best ZSL (%) |
|---|---:|
| 55/5 | 84.28 |
| 48/12 | 62.38 |
| 110/10 | 79.92 |
| 96/24 | 68.34 |

### SynSE features

| Split | Best ZSL (%) |
|---|---:|
| 55/5 | 86.06 |
| 48/12 | 64.31 |
| 110/10 | 80.24 |
| 96/24 | 64.19 |

## GZSL (completed)

### STAR features — SeqKG Text+IDF+gate

Scores are taken at the checkpoint with the largest harmonic mean, `H`.

| Split | Seen S (%) | Unseen U (%) | H (%) |
|---|---:|---:|---:|
| 55/5 | 77.76 | 73.98 | 75.82 |
| 48/12 | 70.17 | 48.22 | 57.16 |

### GZSL comparison status

- SeqKG Compose baseline for 55/5: **S 79.19, U 72.37, H 75.63**. The Text+IDF+gate method is therefore **+0.19 H** on this split.
- SeqKG Compose baseline for 48/12: **S 71.70, U 49.71, H 58.72** (best-H checkpoint at iteration 1160). Compared with the Flora paper, this is +4.80 S, +0.71 U, and **+2.12 H**.
- SeqKG Compose baseline for 110/10: **S 58.86, U 72.02, H 64.78** (best-H checkpoint at iteration 1080). The final checkpoint at iteration 1200 is S 54.86, U 67.43, H 60.50.
- Flora paper (STAR features) for 110/10 reports **S 59.80, U 70.50, H 64.70**. SeqKG Compose is therefore -0.94 S, +1.52 U, and **+0.08 H** versus the paper result.
- 110/10 Text+IDF+gate and 96/24 GZSL still require final results.

### GZSL versus Flora paper (STAR features)

| Split | Paper S / U / H | SeqKG Compose S / U / H | Compose ΔH | Text+IDF+gate S / U / H | Gate ΔH |
|---|---|---|---:|---|---:|
| 55/5 | 77.70 / 75.60 / 76.60 | 79.19 / 72.37 / 75.63 | -0.97 | 77.76 / 73.98 / 75.82 | -0.78 |
| 48/12 | 66.90 / 49.00 / 56.60 | 71.70 / 49.71 / 58.72 | +2.12 | 70.17 / 48.22 / 57.16 | +0.56 |
| 110/10 | 59.80 / 70.50 / 64.70 | 58.86 / 72.02 / 64.78 | +0.08 | pending | — |
| 96/24 | 53.70 / 52.20 / 52.90 | pending | — | pending | — |

## Runtime note

For GZSL-110/10, `--eval-class-chunk 16` reduces VRAM from about 31.8 GB (`chunk=0`) to about 8.4 GB and reduces one evaluation from about 11m54s to about 3m18s. GZSL evaluates every 10 iterations by default, so the complete 110/10 run remains evaluation-heavy.

## Reproducible commands

Run only the SeqKG Compose baseline and Text+IDF+gate method for GZSL:

```bash
python nesy4_seqkg.py --setting GZSL --tasks 48 --num-worker 0 --eval-class-chunk 16 --baselines compose --rerun-baselines --kg-compose-src text --kg-compose-idf --kg-learned-gate
```

For SynSE, add `--feat synse` and select the required splits with `--tasks`.
