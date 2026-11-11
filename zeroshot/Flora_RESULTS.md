# Flora experiment results

Snapshot: 11 Nov 2026. All reported scores use seed 1.

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
