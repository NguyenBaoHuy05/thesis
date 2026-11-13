# Flora experiment results

Updated: 12 Nov 2026. Sections 1–3 report the current main method (3-seed means). The older sections below ("Earlier method") are the 11 Nov snapshot and use seed 1 only.

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
- Centering alone (C-Fuse) gives 61.86 (48/12) and 80.94 (110/10), so most of the 110/10 gain is a KG-independent shift. A no-KG centering control is still missing.
- SeqKG + forbid is on par with compose (mean −0.08) but separates reversed-order class pairs: centered cosine 0.99 → 0.78–0.81.

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

## 2. Negative results (STAR, 48/12 unless noted)

- ESA over per-class KG triples (`nesy7_esa.py`), Δ best vs compose: −5.99 (sim 1), −7.84 (sim 0), −0.82 (sim 0.05 + reg 1). R-GCN (`nesy6_gcn.py`) behaves the same: a learned delta on the KG memorises seen classes.
- Text+IDF+gate ≈ compose after 3 seeds; fixed mix (0.5 text + 0.5 IDF, no gate): 61.37 vs 61.68.
- Forbid of every kind (`--forbid-order-only 0`): unstable (48/12 −1.79, 110/10 +1.58 but last10 −0.98); dropped.
- Earlier (10 Nov): neighbors (hurts), logic head, hard negatives, primitive tokens, plain SeqKG: no gain.

## 3. Still to do

1. No-KG centering control (`nesy0_center.py`), 48/12 and 110/10, 3 seeds.
2. SynSE 96/24, and GZSL for SeqKG + COA-Fuse.
3. Per-pair accuracy and confusion for the reversed-order pairs (evidence for the order/negation claim).

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
