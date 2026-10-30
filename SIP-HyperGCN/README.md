# SIP-HyperGCN — Symbolic Intervention Probing (v3)

Symbolic knowledge is used as an **external probe / intervention tool** on a frozen
HyperGCN, never as a component in the forward path. Rules are an **output** of the study:
they are induced from what the network causally depends on, then checked against data
and human knowledge (concepts CSV). No external rule set is used.

- Spec and file formats: [docs/PIPELINE_V3.md](docs/PIPELINE_V3.md)
- Architecture: [docs/architecture_v3.html](docs/architecture_v3.html)
- Superseded: `docs/PIPELINE_V2.md`, `docs/architecture.html` (v2, binary predicates)

```
continuous measures (20, 40 columns) + body regions (7)
  -> tier 1: region mean-ablation (where)      -> causal_regions
  -> tier 2: LEACE on continuous measures (what) -> causal_measures
  -> strong measures (5 tiers + region cross-check)
  -> induced rules with data-learned thresholds  -> TEST evaluation + concepts CSV check
  -> learnable rules (input to step 7: loss design)
```

Data split: TRAIN-FIT (80% of train) fits erasers, region means, probes and rule thresholds;
VAL (20% of train, stratified, fixed seed) selects measures and induces rules; TEST only
evaluates.

Runs on Colab (GPU, torch, numpy, matplotlib). sklearn is not required.

## Layout

```
sip/
  backbone.py        HyperGCN loader, forward with tap / intervene at l0..l9, sweep helpers
  measures.py        20 continuous measures (40 "measure:stat" columns), region attached
  regions.py         7 body regions, mean-ablation (absent persons never ablated)
  concept_map.py     direction-aware measure -> concepts CSV mapping
  leace.py           closed-form LEACE (continuous multi-column Z)
  probe.py           ridge probes, class-mean baseline, within-class R2, cluster metrics
  causal.py          causal matrix for any unit set (excess vs random, p_class, t1)
  measure_filter.py  5 tiers + region cross-check
  rule_induction.py  one rule per class, stump thresholds, learnability flags
  rule_eval.py       coverage, precision, fidelity, necessity, rule classifier, CSV check
scripts/
  extract_features.py        features l1..l9 + measures
  run_region_intervention.py tier 1 (layers 0 3 6 9)
  run_intervention.py        tier 2 (layers 3 6 9)
  analyze_intervention.py    causal_measures.npz + causal_regions.npz
  select_measures.py         strong_measures.json
  induce_rules.py            rules.json / rules.md
  evaluate_rules.py          rule_report.md / learnable_rules.json
  run_probe.py               optional layer-wise probe table (train -> test)
```

## Running on Colab

Run from `SIP-HyperGCN/`, with `GCCR-Skateformer-TreeRule/` next to it.

```bash
H=../GCCR-Skateformer-TreeRule/hypergcn
CK=/content/drive/MyDrive/ntu60_xsub_joint.pt      # trained HyperGCN joint checkpoint
D=/content/data/NTU60_CS.npz
W=work_dir/sip
COMMON="--hypergcn_root $H --checkpoint $CK --data_path $D"

# 0. smoke test (checks forward_tapped == model.forward, layers 0..9)
python scripts/extract_features.py $COMMON --split test --out_dir work_dir/sip_dbg --max_samples 256

# 1. features + measures
python scripts/extract_features.py $COMMON --split train --out_dir $W --batch_size 128
python scripts/extract_features.py $COMMON --split test  --out_dir $W --batch_size 128

# 2. interventions on VAL (eval_split train) and TEST, same erasers / means / split
for S in train test; do
  python scripts/run_region_intervention.py $COMMON --train_feats $W/train_feats.npz --out_dir $W --eval_split $S
  python scripts/run_intervention.py        $COMMON --train_feats $W/train_feats.npz --out_dir $W --eval_split $S
done

# 3. causal matrices
python scripts/analyze_intervention.py --measures $W/intervention_val.npz --regions $W/region_intervention_val.npz \
    --feats $W/train_feats.npz --out_dir $W/causal_val
python scripts/analyze_intervention.py --measures $W/intervention_test.npz --regions $W/region_intervention_test.npz \
    --feats $W/test_feats.npz --out_dir $W/causal_test

# 4. strong measures (VAL only)
python scripts/select_measures.py --train_feats $W/train_feats.npz --causal_dir $W/causal_val --out_dir $W/measures

# 5. rule induction
python scripts/induce_rules.py --causal_dir $W/causal_val --strong $W/measures/strong_measures.json \
    --train_feats $W/train_feats.npz --out_dir $W/rules

# 6. rule evaluation (TEST)
python scripts/evaluate_rules.py --rules $W/rules/rules.json --test_feats $W/test_feats.npz \
    --causal_dir $W/causal_test --strong $W/measures/strong_measures.json --out_dir $W/rules_eval
# -> $W/rules_eval/rule_report.md, learnable_rules.json
```

Notes:
- Cost: about 25 full-forward equivalents per batch for each intervention script, run twice
  (VAL, TEST). Use `--max_samples` for a timing run first.
- RAM: add `--no_final_feat` to the intervention scripts if short (~0.6 GB on full TEST);
  centroid shift (`analyze_intervention.py --centroid`) then is unavailable.
- Single-person samples: interaction measures are set to FAR_VALUE = 5.0 L; region ablation
  never touches an absent person slot.
- Check `measures/measure_tiers.csv` after step 4: measures failing `within_class` or
  `encoded` explain an empty rule set.
