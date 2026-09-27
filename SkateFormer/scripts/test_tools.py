import sys
sys.path.insert(0, '.')
import numpy as np

# Test build_template_labels.py
from tools.build_template_labels import load_concept_csv, get_template_matrix
t = get_template_matrix(60)
print(f'Template loaded: {t.shape}')

# Test refine_pseudolabels.py
from tools.refine_pseudolabels import refine_labels
N, C = 100, 74
template_targets = np.random.randint(0, 2, (N, C)).astype(np.float32)
template_mask    = np.ones((N, C), dtype=bool)
oof_probs        = np.random.rand(N, C).astype(np.float32)
oof_confidence   = np.random.rand(N).astype(np.float32)
action_labels    = np.random.randint(0, 60, N)

refined, refined_mask, soft_targets, stats = refine_labels(
    template_targets, template_mask, oof_probs, oof_confidence, action_labels
)
print('Refined labels shape:', refined.shape)
print('Refinement stats keys:', list(stats.keys()))

# Test evaluate_concepts.py
from tools.evaluate_concepts import compute_concept_metrics
_, _, concept_names = load_concept_csv('./concepts/ntu_rgbd_60_120.csv', 60)
probs   = np.random.rand(N, C).astype(np.float32)
targets = template_targets.copy()
targets[:10, :5] = -1  # some unknown
cmets = compute_concept_metrics(probs, targets, concept_names)
g = cmets['global']
print('Concept metrics: macro_f1=', round(g['macro_f1'], 3), 'coverage=', round(cmets['coverage'], 3))

# Test select_audit_samples.py
from tools.select_audit_samples import select_audit_samples
sample_names = np.array(['train_' + str(i) for i in range(N)])
idx, groups = select_audit_samples(
    oof_probs, oof_confidence, template_targets, action_labels, sample_names,
    n_disagree=20, n_new_pos=20, n_low_conf=10, n_random=10
)
print('Audit samples:', len(idx), 'groups:', set(groups.tolist()))

print('All tool imports and basic tests passed!')
