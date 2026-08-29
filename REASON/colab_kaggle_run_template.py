import os
from pathlib import Path

# ============================================================
# REASON Colab / Kaggle launcher
# ============================================================
# Paste this into a notebook and fill the path variables.
# This script runs the two ablation branches separately:
#   1) random_init
#   2) kg_primed
# ============================================================

ROOT = Path('/content') if os.path.exists('/content') else Path('/kaggle/working')
PROJECT_ROOT = ROOT / 'REASON'

# ------------------------------------------------------------
# 1) Set your dataset / checkpoint paths
# ------------------------------------------------------------
# Example:
# os.environ['REASON_PRETRAIN_HGCN'] = '/content/drive/MyDrive/REASON/checkpoints/hgcn_l_j.pt'
# os.environ['REASON_DATA_PATH'] = '/content/drive/MyDrive/NTU120/NTU120_CSub.npz'
# os.environ['REASON_CONCEPTS_CSV'] = '/content/REASON/concepts/ntu_rgbd_60_120.csv'

# If you have the repo already checked out in the workspace, leave these as-is:
if not os.environ.get('REASON_PRETRAIN_HGCN'):
    os.environ['REASON_PRETRAIN_HGCN'] = str((PROJECT_ROOT / 'checkpoints' / 'hgcn' / 'hgcn_l_j.pt').resolve())
if not os.environ.get('REASON_DATA_PATH'):
    os.environ['REASON_DATA_PATH'] = str((PROJECT_ROOT / 'data' / 'ntu120' / 'NTU120_CSub.npz').resolve())
if not os.environ.get('REASON_CONCEPTS_CSV'):
    os.environ['REASON_CONCEPTS_CSV'] = str((PROJECT_ROOT / 'concepts' / 'ntu_rgbd_60_120.csv').resolve())

# ------------------------------------------------------------
# 2) Optional: download or mount files here if needed
# ------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')

# ------------------------------------------------------------
# 3) Choose train mode
# ------------------------------------------------------------
RUN_RANDOM = True
RUN_KG = True

# ------------------------------------------------------------
# 4) Build command list
# ------------------------------------------------------------
PYTHON = 'python'
COMMON_ARGS = [
    str(PROJECT_ROOT / 'main.py'),
    '--device', '0',
]

# config files generated earlier
RANDOM_CFG = str(PROJECT_ROOT / 'config' / 'xsub' / 'config_skeleton_crl_random_pretrain.yaml')
KG_CFG = str(PROJECT_ROOT / 'config' / 'xsub' / 'config_skeleton_crl_kg_pretrain.yaml')


def run_cmd(cmd):
    import subprocess
    print('\n=== RUNNING ===')
    print(' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy())
    print(f'EXIT CODE: {result.returncode}')
    return result.returncode


if __name__ == '__main__':
    # ensure project root exists
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f'Project root not found: {PROJECT_ROOT}. Please clone or mount the repo first.')

    if RUN_RANDOM:
        run_cmd([PYTHON, *COMMON_ARGS, '--config', RANDOM_CFG])

    if RUN_KG:
        run_cmd([PYTHON, *COMMON_ARGS, '--config', KG_CFG])

    print('\nDone. Random and KG-primed runs were launched separately.')
