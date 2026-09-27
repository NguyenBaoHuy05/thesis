import sys
sys.path.insert(0, '.')
from tools.build_template_labels import load_concept_csv, get_template_matrix
import numpy as np

csv_path = 'concepts/ntu_rgbd_60_120.csv'

# NTU60
t60, classes60, concepts = load_concept_csv(csv_path, 60)
print(f'NTU60 template shape: {t60.shape}')
print(f'  Classes: {len(classes60)} (first: {classes60[0]}, last: {classes60[-1]})')
print(f'  Concepts: {len(concepts)} (first: {concepts[0]}, last: {concepts[-1]})')
print(f'  Mean pos/class: {t60.sum(1).mean():.2f}')
print(f'  Mean pos/concept: {t60.sum(0).mean():.2f}')
print(f'  Value range: {t60.min():.0f} to {t60.max():.0f}')

# NTU120
t120, classes120, _ = load_concept_csv(csv_path, 120)
print(f'NTU120 template shape: {t120.shape}')
print(f'  Classes: {len(classes120)}')

# Quick sanity check
assert t60.shape == (60, 74), f'Expected (60,74) got {t60.shape}'
assert t120.shape == (120, 74), f'Expected (120,74) got {t120.shape}'
assert np.isin(t60, [0.0, 1.0]).all(), 'Non-binary values in template'
print('All template checks passed!')
