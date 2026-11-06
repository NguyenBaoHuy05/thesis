# SIP-HyperGCN v2: rules are induced from interventions

Decision (user): the CLIR DNF rules (`CLIR_NEW/rules_output/dnf_rules.json`) are dropped
entirely. No comparison with them anywhere. Rules are an OUTPUT of the intervention study.
The concepts CSV (`GCCR-Skateformer-TreeRule/concepts/ntu_rgbd_60_120.csv`) is kept only
as human knowledge to check induced rules against (agree = confirmation, disagree = finding).

```
geometric predicates -> strong-predicate filter (tiers 1-5)
  -> interventions on VAL -> causal dependency D[k, p]
  -> induce one rule per class from D + data support (train)
  -> evaluate rules on TEST: coverage, precision, fidelity, discrimination, necessity
  -> check against concepts CSV
  -> learnable rules list (input to step 5: loss design)
```

Anti-leakage: VAL = stratified 20% of the NTU train split (fixed seed). LEACE erasers and
probes are fit on TRAIN-FIT = train minus VAL. Selection and induction use VAL only.
TEST is used only for final evaluation.

## Workflow

```bash
# 1. features (unchanged)
extract_features.py --split train ... ; extract_features.py --split test ...
# 2. interventions on val (induction) and test (evaluation), same erasers
run_intervention.py --eval_split train --val_frac 0.2 --seed 0 --out_name intervention_val.npz ...
run_intervention.py --eval_split test  --val_frac 0.2 --seed 0 --out_name intervention_test.npz ...
# 3. causal analysis per split
analyze_intervention.py --intervention $W/intervention_val.npz  --feats $W/train_feats.npz --out_dir $W/causal_val
analyze_intervention.py --intervention $W/intervention_test.npz --feats $W/test_feats.npz  --out_dir $W/causal_test
# 4. strong predicates (uses VAL only)
select_predicates.py --train_feats $W/train_feats.npz --causal $W/causal_val/causal_matrix.npz --out_dir $W/predicates
# 5. rule induction (VAL causal + TRAIN-FIT data support)
induce_rules.py --causal $W/causal_val/causal_matrix.npz --strong $W/predicates/strong_predicates.json \
    --train_feats $W/train_feats.npz --out_dir $W/rules
# 6. rule evaluation (TEST)
evaluate_rules.py --rules $W/rules/rules.json --test_feats $W/test_feats.npz \
    --causal $W/causal_test/causal_matrix.npz --strong $W/predicates/strong_predicates.json --out_dir $W/rules_eval
```

## Interfaces

### `intervention_*.npz` (agent A, `scripts/run_intervention.py`)
Existing keys stay (`layers`, `pred_names`, `labels`, `base_logits`, `acc_per_class`,
`base_acc_per_class`, `class_count`, `skipped`, `final_feat`, `base_final_feat`) plus:
- `eval_split`: str `"train"` | `"test"`
- `eval_index`: `(N,)` int64 — row indices into `<eval_split>_feats.npz` of evaluated samples
- `val_index`: `(N_val,)` int64 — row indices into `train_feats.npz` forming VAL (same in both runs)
- `n_random`: int R. The P axis is `[*PREDICATE_NAMES, "__random_0__", ..., "__random_{R-1}__"]`
  (the old single `"__random__"` is gone). Default R = 5.
- `correct`: `(L, P+R, N)` bool — sample correct after erasing at layer l
- `pred_label`: `(L, P+R, N)` int16 — predicted class after erasing
- `base_pred`: `(N,)` int64
- `--no_final_feat` flag skips `final_feat` to save RAM.

### `causal_matrix.npz` (agent B, `scripts/analyze_intervention.py`)
- `pred_names` `(P,)`, `class_names` `(K,)`, `layers` `(L,)`
- `dacc` `(L, P, K)`: base_acc_k − acc_k(erase p)
- `drand` `(L, K)`: mean over the R random controls; `drand_sd` `(L, K)`
- `excess` `(L, P, K)`: dacc − drand
- `p_class` `(L, P, K)`: one-sided p that erasing p hurts class k more than random
  (bootstrap / permutation over the samples of class k)
- `t1_p` `(L, P)`: predicate-level test (class-sign-flip permutation on excess), `t1_mean` `(L, P)`
- `emp_rate` `(P, K)`: fraction of samples of class k with binarized predicate p
  (computed from the `--feats` file rows `eval_index`)
- `confusion` `(K, K)`: base confusion matrix on the evaluated samples
- `centroid_shift` `(L, P, K)` (if final_feat present)

### `strong_predicates.json` (agent B, `scripts/select_predicates.py`)
```json
{"predicates": [{"name": "...", "layer": 6,
   "tiers": {"grounded": true, "within_class": true, "encoded": true, "causal": true, "stable": true},
   "metrics": {"mixed_class_frac": 0.4, "probe_gain_auroc": 0.05, "within_class_auroc": 0.61,
               "t1_p": 0.003, "t1_mean": 0.012, "stability": 0.9}}],
 "selected": ["..."], "layer_of": {"name": 6}, "criteria": {...}}
```
`layer` = layer with largest probe gain on VAL (chosen before looking at causal results).
A predicate is selected when all five tiers hold.

### `rules.json` (agent C, `scripts/induce_rules.py`)
```json
{"predicates": ["..."], "rules": [{"class_idx": 0, "class_name": "drink_water",
   "literals": [["hand_near_head", true], ["two_person_present", false]],
   "evidence": {"hand_near_head": {"excess": 0.08, "p_class": 0.01, "emp_rate": 0.9}},
   "train_support": 0.71, "train_precision": 0.22,
   "learnable": {"data_supported": true, "encodable": true, "consistent": true}}],
 "params": {...}}
```
One conjunctive rule per class (empty `literals` allowed = no rule found).
