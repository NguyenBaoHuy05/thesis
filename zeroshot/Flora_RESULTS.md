# Flora / PRIZM experiment results

Updated: 16 Nov 2026 (synced with `main.tex`). Sections 1–3 report the current main method (**PRIZM** = SeqKG [ORE + PPC] + COA-Fuse). Standard ZSL runs use 3-seed means on Colab L4 (with `main.tex` Table 1 reporting seed 1, fixed in advance); GZSL, low-shot, and random-split runs use seed 1 on RTX 5090. The older sections below ("Earlier method") are the 11 Nov snapshot and use seed 1 only.

Protocol: paper setting (1000 + 200 iterations), ZSL (`best` = best checkpoint on unseen) and GZSL (`best` = checkpoint with highest harmonic mean `H`). Because Flora is not deterministic across GPU types, controlled reruns (`No KG`, ablations) use the same GPU and seed as the PRIZM runs they are compared with; paper baselines are taken directly from Flora.

## 1. Current main method: PRIZM (SeqKG + COA-Fuse)

- **SeqKG (ORE)**: KG v2 (82 primitives) plus 29 order primitives and 14 reverse-order forbids; class pairs with identical rules 7 → 0 (111 primitives total).
- **Signed Compose (PPC)**: class semantics = (1+λ)·mean(require) − λ·mean(order-forbid), λ = 1. No learned parameters.
- **COA-Fuse**: Center (remove class-mean of the KG term), Orthogonal (remove the part along the class text), Adaptive (per-class β from text-vs-KG disagreement, β = 0.3, clipped at 3β). No learned parameters.
- Code: `nesy4_seqkg.py`, `nesy4_1_forbid.py`, `nesy4_1_1_fuse.py`. Details: `progress/2026-11-12_flora_nesy_notes.md`, `progress/2026-11-13_flora_nesy_notes.md`, `main.tex`.

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
In `main.tex` Table 1 (seed 1): **PRIZM†** is 85.3 / 64.7 / 81.4 / 66.8 (4-split mean **74.5%** vs Flora† paper **74.4%**).

### Component ablation (ZSL, STAR features, seed 1, same GPU — from `main.tex`)

| Semantics | 55/5 | 48/12 | 110/10 | 96/24 | Mean |
|---|---:|---:|---:|---:|---:|
| Flora (paper) | 85.8 | 61.5 | 80.7 | 64.1 | 73.0 |
| No KG (Flora, rerun, seed 1) | **85.5** | 62.1 | 77.6 | 67.1 | 73.1 |
| KG v2 compose (82 primitives, no ORE/PPC) | 85.4 | 62.9 | 79.2 | **68.5** | 74.0 |
| + ORE + PPC (raw add) | 84.7 | **64.0** | 79.2 | 67.5 | 73.9 |
| + COA-Fuse (= PRIZM, seed 1) | 85.1 | 63.8 | **80.6** | 67.0 | **74.1** |

### COA-Fuse step-by-step ablation (17 Nov, STAR features, seed 1, same GPU session)

Run in `ablation_b_samegpu/` (`best` / `last10`, %):

| Fusion | 48/12 (best / last10) | 110/10 (best / last10) |
|---|---:|---:|
| No KG | 62.96 / 60.38 | 78.11 / 74.18 |
| ORE + PPC | 63.96 / 61.94 | 79.22 / 76.64 |
| + Center (C-Fuse) | 63.69 / 61.45 | 80.16 / 76.41 |
| + Orth (CO-Fuse) | **64.08** / 61.71 | 80.16 / 76.91 |
| + Adaptive (COA-Fuse = PRIZM) | 63.75 / **61.95** | **80.63** / **77.24** |

Note: the KG runs (`ORE + PPC` and `COA-Fuse`) match the earlier L4 seed-1 numbers to 0.01 (`63.96/79.22` and `63.75/80.63`), confirming identical hardware/determinism; `C-Fuse` seed 1 (`63.69` / `80.16`) also completes the β = 0.3 entry for seed 1.

### GZSL (PRIZM = SeqKG + COA-Fuse, seed 1, RTX 5090 — from `main.tex`)

Scores are S / U / H (%) at the checkpoint with the largest harmonic mean `H`.

| Method | Feat | 55/5 (S / U / H) | 48/12 (S / U / H) | 110/10 (S / U / H) | 96/24 (S / U / H) | Mean H |
|---|---|---|---|---|---|---:|
| Flora (paper) | STAR | 77.7 / 75.6 / 76.6 | 66.9 / 49.0 / 56.6 | 59.8 / 70.5 / 64.7 | 53.7 / 52.2 / 52.9 | 62.7 |
| **PRIZM** (seed 1) | STAR | 78.0 / 74.4 / 76.2 | 70.8 / 49.7 / 58.4 | 59.6 / 72.4 / 65.4 | 50.5 / 61.6 / **55.5** | **63.9** |
| Flora† (paper) | SynSE | 75.9 / 78.8 / 77.4 | 63.7 / 57.5 / 60.5 | 66.2 / 66.0 / 66.1 | 55.9 / 50.7 / 53.2 | **64.3** |
| **PRIZM†** (seed 1) | SynSE | 78.3 / 74.0 / 76.1 | 68.4 / 52.3 / 59.3 | 62.9 / 72.9 / **67.5** | 53.1 / 53.8 / 53.4 | 64.1 |

Notes:
- With STAR features, PRIZM improves `H` over the Flora paper on 48/12 (+1.8), 110/10 (+0.7), and 96/24 (+2.6, best overall at 55.5%), and is −0.4 on 55/5 (4-split mean `H`: 63.9% vs 62.7%).
- Gains on NTU-120 come mainly from unseen accuracy `U` (highest `U` among compared methods on both 110/10 and 96/24 for both feature sets), partly traded off against seen accuracy `S` (e.g., 96/24 STAR `S` 50.5% vs 53.7% for Flora).
- With SynSE features, PRIZM† achieves the best `H` on 110/10 (67.5%, +1.4 over Flora†) and +0.2 on 96/24, while trailing on 55/5 (−1.3) and 48/12 (−1.2); 4-split mean `H` is 64.1% vs 64.3%. As in ZSL, 110/10 is the split where the gain holds for both feature sets.

#### GZSL fusion ablation (H %, best / last-10, STAR features, seed 1, same GPU)

| Semantics | 55/5 | 48/12 | 110/10 | 96/24 | Mean |
|---|---|---|---|---|---:|
| No KG (Flora, rerun) | 75.9 / **72.8** | 55.3 / 54.0 | — | — | — |
| KG, raw add (ORE, uniform compose) | 75.6 / 71.6 | **58.7** / **56.4** | 64.8 / 62.7 | **55.5** / **53.6** | 63.7 / **61.1** |
| COA-Fuse (= PRIZM) | **76.2** / **72.8** | 58.4 / 55.7 | **65.4** / **63.7** | **55.5** / 52.1 | **63.9** / **61.1** |

- On NTU-60 vs the `No KG` rerun, PRIZM improves `H` by +3.1 on 48/12 (58.4 vs 55.3) and is slightly above on 55/5 (76.2 vs 75.9; `main.tex` notes +0.2 from unrounded values).
- Between `raw add` and `COA-Fuse`, the mean difference is +0.2 best `H` and 0.0 last-10 `H` (61.1%). Only 110/10 improves on both best (+0.6) and last-10 (+1.0); on 96/24 best `H` ties (55.5%) while COA-Fuse drops faster in the last 10 evaluations (−1.5). The GZSL gains over Flora therefore come mainly from the KG prior itself, with COA-Fuse adding specifically on 110/10.

### Low-shot ZSL training (SynSE features, seed 1, RTX 5090 — from `main.tex`)

Trained on 1% or 10% of the seen training samples per class (at least 1 sample/class, sampled with seed 1), evaluated on the full unseen test set.

| Method | 55/5 (1% / 10%) | 48/12 (1% / 10%) | 110/10 (1% / 10%) | 96/24 (1% / 10%) | Mean (1% / 10%) |
|---|---|---|---|---|---|
| Flora (paper) | **82.8** / **85.6** | 46.5 / 56.1 | 77.4 / 78.1 | **58.0** / 65.9 | 66.2 / 71.4 |
| **PRIZM** (seed 1) | 81.6 / 84.8 | **55.0** / **62.6** | **81.2** / **83.4** | **58.0** / **66.8** | **68.9** / **74.4** |
| Δ vs Flora paper | −1.2 / −0.8 | **+8.5** / **+6.5** | **+3.8** / **+5.3** | 0.0 / +0.9 | **+2.7** / **+3.0** |

- With 10% training data, PRIZM stays close to its full-data mean (74.4% vs 74.5%); at 1% it drops 5.6 points on average vs 8.2 points for Flora.
- Indicative only: at 1% (5–6 samples per seen class) subset choice adds variance; on 55/5 at 1%, our `No KG` rerun reaches 82.6% vs 81.6% for PRIZM.

### Random splits (SA-DVAE ST-GCN features, 3-split mean, seed 1, RTX 5090 — from `main.tex`)

Average ZSL accuracy and GZSL `H` (%) over the three random seen/unseen splits of SA-DVAE. PKU-MMD I (46/5) uses rules mapped from NTU-120 (49 classes unchanged; *put something inside pocket* and *take out something from pocket* split from *reach into pocket* with a hand-in-front primitive and opposite sequences; ORE separates all 4 reversed pairs).

| Method | NTU-60 55/5 (ZSL / H) | NTU-120 110/10 (ZSL / H) | PKU-MMD I 46/5 (ZSL / H) |
|---|---|---|---|
| Flora (paper) | **88.6** / 80.2 | 71.2 / 63.0 | **71.6** / **59.5** |
| **PRIZM** (seed 1) | **88.6** / **81.4** | **72.9** / **64.8** | 69.4 / 58.1 |
| Δ vs Flora paper | 0.0 / +1.2 | **+1.7** / **+1.8** | −2.2 / −1.4 |

- Per-split PRIZM breakdown: NTU-60 ZSL 85.1% / 95.0% / 85.6%; NTU-120 GZSL `H` 71.5% / 56.3% / 66.7%; PKU-MMD ZSL 51.3% / 75.1% / 81.8% and GZSL `H` 47.7% / 59.5% / 67.2%.
- The three random splits vary widely in difficulty; the clearest gain over Flora again appears on NTU-120 (+1.7 ZSL, +1.8 `H`).

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
- **Primitive probe** (`nesy0_probe.py`, no Flora training, STAR): a probe trained on seen features (targets = class `require` rows) predicts primitives on unseen test samples with mean AUC 0.771 / 0.831 / 0.772 / 0.841 (linear) and 0.859 / 0.871 / 0.858 / 0.882 (MLP) for 55/5, 48/12, 110/10, 96/24; seen-test reference 0.985–0.997; shuffled-rule probe 0.744 / 0.631 / 0.522 / 0.531 (linear), so the real − shuffled gap is +0.18 to +0.32 except on 55/5 (only 11 evaluable primitives, mostly easy ones). kNN profile 0.775–0.786. **The pooled features do carry transferable primitive information** (the earlier explanation "pooled features hold no primitive information" was wrong). Rule-only ZSL accuracy stays low: probe 72.9 / 29.7 / 43.1 / 25.8, kNN 66.4 / 31.2 / 31.7 / 26.3 (chance 20 / 8.3 / 10 / 4.2), so per-primitive evidence is not enough to separate unseen classes that share most primitives.
- **DiT + probe log-probability re-ranking** (`nesy4_1_1_5_1_probe.py`, seed 1, score = z(−MSE) + γ·[mean log q(require) + mean log(1 − q(forbid))] in the DiT top-k). At the best-base eval (base 63.75 / 80.63): probe 48/12 γ 0.02 / 0.05 / 0.1 top-3 62.17 / 59.53 / 56.58 (every γ below base, monotone towards base as γ → 0); probe 110/10 80.80 / **81.43** / 81.29 (+0.17 / +0.80 / +0.66); kNN 48/12 61.20 / 57.49 / 53.08; kNN 110/10 81.17 / 80.70 / 77.99. Log-probability combination hurts far less than the z-score version and the probe beats kNN, but 48/12 is hurt at every γ (primitive coverage 0.58 vs 0.90 at 110/10). Fails the pre-set criterion (gain on both splits); **closed**. The 110/10 gain is a single seed.

## 2. Negative results (STAR, 48/12 unless noted)

- Every use of the KG on the skeleton side or at decision time failed: logic head, primitive tokens, margin-gated re-ranking (diagnostic), retrieval + rule re-ranking (−16 / −25). Only the semantic-side use works, confirmed by the shuffled-KG control.

- Compose from a joined class sentence (`--kg-compose-src text`): 62.53 vs compose 62.87 (seed 1; seeds 2–3 and 110/10 not run).
- Name-free context around primitive sentences: no gain (see above).

- ESA over per-class KG triples (`nesy7_esa.py`), Δ best vs compose: −5.99 (sim 1), −7.84 (sim 0), −0.82 (sim 0.05 + reg 1). R-GCN (`nesy6_gcn.py`) behaves the same: a learned delta on the KG memorises seen classes.
- Text+IDF+gate ≈ compose after 3 seeds; fixed mix (0.5 text + 0.5 IDF, no gate): 61.37 vs 61.68.
- Forbid of every kind (`--forbid-order-only 0`): unstable (48/12 −1.79, 110/10 +1.58 but last10 −0.98); dropped.
- Earlier (10 Nov): neighbors (hurts), logic head, hard negatives, primitive tokens, plain SeqKG: no gain.

## 3. Still to do

1. Shuffled-KG control on 55/5 and 96/24 if needed; β sensitivity for full COA-Fuse (done for C-Fuse only).
2. `No KG` GZSL reruns on NTU-120 (110/10 and 96/24) and seeds 2–3 for GZSL / low-shot / random splits if multi-seed reporting is needed beyond seed 1.
3. ~~GZSL for SeqKG + COA-Fuse~~: **Done (seed 1, STAR & SynSE)** — see Section 1.
4. ~~Per-pair accuracy for the reversed-order pairs~~: not measurable on the standard splits (see section 1).

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
