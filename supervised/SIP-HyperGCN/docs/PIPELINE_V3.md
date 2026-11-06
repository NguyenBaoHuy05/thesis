# SIP-HyperGCN v3 — authoritative spec

Approved design: `docs/architecture_v3.html` (read it). Summary of changes vs v2:

- Binary predicates (`sip/predicates.py`) are REMOVED. Replaced by ~18 **continuous measures**
  (`sip/measures.py`), each with a few temporal statistics and attached to body regions.
  No hard thresholds anywhere during measurement, erasure, probing or filtering.
- Thresholds appear ONLY inside induced rules, learned on TRAIN-FIT (decision stump).
- Two intervention tiers: **regions** (mean-ablation of joints, "where") and **measures**
  (LEACE with continuous multi-dim Z, "what"), plus a **cross-check**: a measure is accepted
  for class k only if one of its regions also matters for class k.
- Data split unchanged: TRAIN-FIT (80% train) / VAL (20% train, stratified, seed) / TEST.
- No CLIR rules anywhere. Concepts CSV (GCCR path) only for the final human-knowledge check.

## Workflow (Colab)

```bash
H=../GCCR-Skateformer-TreeRule/hypergcn; W=work_dir/sip
COMMON="--hypergcn_root $H --checkpoint $CK --data_path $D"
python scripts/extract_features.py $COMMON --split train --out_dir $W
python scripts/extract_features.py $COMMON --split test  --out_dir $W
for S in train test; do  # train => VAL rows
  python scripts/run_region_intervention.py $COMMON --train_feats $W/train_feats.npz --out_dir $W --eval_split $S
  python scripts/run_intervention.py        $COMMON --train_feats $W/train_feats.npz --out_dir $W --eval_split $S
done
python scripts/analyze_intervention.py --measures $W/intervention_val.npz --regions $W/region_intervention_val.npz \
    --feats $W/train_feats.npz --out_dir $W/causal_val
python scripts/analyze_intervention.py --measures $W/intervention_test.npz --regions $W/region_intervention_test.npz \
    --feats $W/test_feats.npz --out_dir $W/causal_test
python scripts/select_measures.py --train_feats $W/train_feats.npz --causal_dir $W/causal_val --out_dir $W/measures
python scripts/induce_rules.py --causal_dir $W/causal_val --strong $W/measures/strong_measures.json \
    --train_feats $W/train_feats.npz --out_dir $W/rules
python scripts/evaluate_rules.py --rules $W/rules/rules.json --test_feats $W/test_feats.npz \
    --causal_dir $W/causal_test --strong $W/measures/strong_measures.json --out_dir $W/rules_eval
```
Defaults: `--val_frac 0.2 --seed 0 --n_random 5`; measure layers `3 6 9`; region layers `0 3 6 9`.

## Interfaces

### `sip/measures.py` (agent 1)
```python
MEASURE_NAMES: list[str]                     # ~18, fixed order
MEASURE_REGIONS: dict[str, list[str]]        # measure -> region names (from sip.regions.REGION_NAMES)
MEASURE_STATS: dict[str, list[str]]          # measure -> stat names, e.g. ["min", "q10", "median"]
COLUMNS: list[str]                           # flat "measure:stat" in MEASURE_NAMES order, D entries
def measure_slices() -> dict[str, slice]     # measure -> column slice into COLUMNS
def compute_measures(x: np.ndarray, input_format: str = "feeder") -> np.ndarray
    # x (N, 3, T, 25, 2) as output by the HyperGCN test feeder (joint modality,
    # joint-20-relative except joint 20) -> (N, D) float32, finite everywhere (no NaN).
FAR_VALUE: float                             # fill for interaction measures when no 2nd person
```
Includes a `second_person` measure (region `other_person`) with stat `present_frac`.
Normalised by spine length L. Angles in degrees. Also `measure_table_markdown()` for docs.

### `sip/regions.py` (agent 1)
```python
REGION_NAMES = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "other_person"]
REGIONS: dict[str, list[int]]       # joint indices (0-based NTU); "other_person" -> all 25 joints of person index 1
def region_joint_mask(name) -> np.ndarray   # (25,) bool
def random_joint_sets(n: int, sizes: list[int], seed: int) -> list[list[int]]  # controls
def ablate_torch(h, joints, mean_mcv, num_person=2, person=None)
    # h: (N*M, C, T, V) torch. mean_mcv: torch/np (M, C, V) TRAIN-FIT mean of the layer output
    # (mean over samples and T, per person slot). Replaces h[:, :, :, joints] by the mean for
    # those joints (all persons if person is None; only person slot `person` otherwise —
    # "other_person" uses person=1 and all joints). Returns a new tensor.
```

### `sip/concept_map.py` (agent 1, update)
`MEASURE_TO_CONCEPTS: dict[str, dict[str, list[str]]]` — measure -> {"<": [...], ">": [...]}
(concepts implied when the measure is LOW / HIGH; real GCCR CSV column names only).
`REGION_TO_CONCEPTS: dict[str, list[str]]`. Keep `load_concept_csv`, `GCCR_CONCEPT_CSV`.
Remove everything predicate-specific and all CLIR / DNF helpers.

### `train_feats.npz` / `test_feats.npz` (agent 2, `extract_features.py`)
`feat_l1..feat_l9` (N, C), `featj_l3/6/9` (N, C, 25) f16, `logits`, `labels`, `index`,
`meas` (N, D) float32, `meas_cols` (D,) str (= measures.COLUMNS). No `pred_scores`.

### `intervention_{val,test}.npz` (agent 2, `run_intervention.py`, measures via LEACE)
Same layout as v2 but units are measures: `unit_names` (U = M_meas + R) with randoms
`__random_0__..` last, `n_random`, `layers`, `labels`, `base_logits`, `base_pred`,
`correct` (L, U, N) bool, `pred_label` (L, U, N) int16, `acc_per_class` (L, U, K),
`base_acc_per_class`, `class_count`, `skipped` (L, U), `eval_split`, `eval_index`, `val_index`,
optional `final_feat`, `base_final_feat`. LEACE fit per measure on TRAIN-FIT `feat_l{l}` vs
**standardised continuous** Z = `meas[:, measure_slices()[m]]` (multi-column). Random controls:
random subspace of rank = median measure rank.

### `region_intervention_{val,test}.npz` (agent 2, `run_region_intervention.py`)
Same keys as above with `unit_names` = REGION_NAMES + `__random_0__..` (random joint sets of
matched sizes), `layers` default (0, 3, 6, 9). Layer 0 = output of the joint embedding +
data_bn, before l1 (`sip/backbone.py` must support intervene[0]). Means (M, C, V) per layer
estimated on TRAIN-FIT (`--n_mean_samples`, default 4000, stratified) in a first pass.

### `causal_dir/` (agent 3, `analyze_intervention.py`)
`causal_measures.npz` and `causal_regions.npz`, each with: `unit_names` (U,) (no randoms),
`class_names` (K,), `layers` (L,), `dacc` (L,U,K), `drand` (L,K), `drand_sd` (L,K),
`excess` (L,U,K), `p_class` (L,U,K), `t1_p` (L,U), `t1_mean` (L,U), `confusion` (K,K),
`class_count` (K,), `base_acc` (K,), `val_index`, `eval_index`, `eval_split`.
`causal_measures.npz` additionally: `class_mean` (D,K), `class_std` (D,K) of `meas` columns
on the eval rows, `meas_cols` (D,). Also CSV summaries + heatmaps.

### `strong_measures.json` (agent 3, `select_measures.py`)
```json
{"measures": [{"name": "hand_head_dist", "layer": 6, "regions": ["left_arm","right_arm","head"],
   "tiers": {"grounded": true, "within_class": true, "encoded": true, "causal": true,
             "stable": true, "region_check": true},
   "metrics": {"within_frac": 0.4, "probe_r2": 0.5, "class_r2": 0.3, "gain_r2": 0.2,
               "within_class_r2": 0.1, "t1_p": 0.003, "t1_mean": 0.01, "stability": 0.9,
               "region_t1_p": 0.001, "best_region": "right_arm", "region_layer": 0}}],
 "selected": ["..."], "layer_of": {"hand_head_dist": 6}, "criteria": {}}
```
Tiers: within_class = mean over the measure's columns of (within-class var / total var) on
TRAIN-FIT ≥ 0.2. encoded = ridge probe feat_l -> standardised Z (TRAIN-FIT fit, VAL score);
gain_r2 = probe R² − class-mean baseline R² ≥ 0.02 and within-class R² ≥ 0.05; layer chosen
by max gain BEFORE reading causal results. causal = t1_p < 0.05 and t1_mean > 0.005 at that
layer. stable = bootstrap (200) over VAL samples for encoded gain and over classes for t1,
both ≥ 80 % agreement. region_check = some region of the measure has region t1_p < 0.05 at
layer 0 or 3.

### `rules.json` (agent 4, `induce_rules.py`)
```json
{"columns": ["..."], "rules": [{"class_idx": 0, "class_name": "drink_water",
   "literals": [["hand_head_dist:min", "<", 0.43], ["person_dist:median", ">", 2.1]],
   "evidence": {"hand_head_dist": {"layer": 6, "excess": 0.08, "q_class": 0.01,
                "region": "right_arm", "region_excess": 0.12, "region_p": 0.003}},
   "train_support": 0.71, "train_precision": 0.4,
   "learnable": {"data_supported": true, "encodable": true, "consistent": true, "learnable": true}}],
 "params": {}}
```
Candidate measure m for class k: m selected; excess[l_m, m, k] ≥ 0.02 with BH-FDR q < 0.05;
per-class region check: some region r of m has region excess[l, r, k] ≥ 0.01 and p_class < 0.05
at layer 0 or 3. Literal: best column of m + direction + θ by decision stump on TRAIN-FIT
(class k vs union of its top-3 confusable classes, maximise balanced accuracy), requiring
class-k coverage ≥ 0.5. Greedy add while TRAIN-FIT precision increases, coverage ≥ 0.3,
max 4 literals. `rule_eval` applies literals on TEST `meas` and reports as in v2, plus
knowledge check via `MEASURE_TO_CONCEPTS` (direction-aware).
