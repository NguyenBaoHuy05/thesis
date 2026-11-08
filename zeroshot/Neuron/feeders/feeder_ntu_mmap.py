"""Low-RAM version of Feeder_Shiftgcn_Match (feeders/feeder_ntu.py).

The upstream feeder loads the whole x_train into RAM and np.delete makes a second copy (~14 GB peak for
NTU60, ~35 GB for NTU120). This one memory-maps .npy files and keeps only the indices of the selected
samples, so RAM stays around 1-2 GB. Sample order, preprocessing and augmentation are the same as upstream
(it reuses Feeder_Shiftgcn_Match.__getitem__), so a seeded run gives the same samples.

data_path: a directory with x_train.npy, y_train.npy, x_test.npy, y_test.npy, or an .npz file (converted
once into the sibling directory <name>/ by npz_to_npy, streamed, without loading it into RAM).
"""
import os
import shutil
import zipfile

import numpy as np

from feeders.feeder_ntu import Feeder_Shiftgcn_Match

UNSEEN = {  # same lists as feeders/feeder_ntu.py
    'ntu60_seen55_unseen5': [10, 11, 19, 26, 56],
    'ntu60_seen48_unseen12': [3, 5, 9, 12, 15, 40, 42, 47, 51, 56, 58, 59],
    'ntu120_seen110_unseen10': [4, 13, 37, 43, 49, 65, 88, 95, 99, 106],
    'ntu120_seen96_unseen24': [5, 9, 11, 16, 18, 20, 22, 29, 35, 39, 45, 49, 59, 68, 70, 81, 84, 87, 93, 94, 104,
                               113, 114, 119],
    'as_ntu60_seen55_unseen5_split1': [4, 19, 31, 47, 51],
    'as_ntu60_seen55_unseen5_split2': [12, 29, 32, 44, 59],
    'as_ntu60_seen55_unseen5_split3': [7, 20, 28, 39, 58],
}
KEYS = ('x_train', 'y_train', 'x_test', 'y_test')


def npz_to_npy(npz_path, out_dir, chunk=64 << 20):
    """Extract every array of the .npz into out_dir/<key>.npy by streaming (no full load into RAM).
    Each file is written to .tmp then renamed, so a half-written file is never used."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        for name in zf.namelist():
            dst = os.path.join(out_dir, name if name.endswith('.npy') else name + '.npy')
            if os.path.exists(dst):
                continue
            with zf.open(name) as src, open(dst + '.tmp', 'wb') as f:
                shutil.copyfileobj(src, f, chunk)
            os.replace(dst + '.tmp', dst)
    return out_dir


def npy_dir(data_path):
    """Directory of .npy files for data_path (converting an .npz on first use)."""
    if os.path.isdir(data_path):
        return data_path
    assert data_path.endswith('.npz'), data_path
    out = data_path[:-4]
    if not all(os.path.exists(os.path.join(out, f'{k}.npy')) for k in KEYS):
        print(f'converting {data_path} -> {out}/*.npy (one time)', flush=True)
        npz_to_npy(data_path, out)
    return out


def select(labels, split, zero_setting, unseen):
    """Indices of the samples a Feeder_Shiftgcn_Match with these settings keeps, in the same order."""
    if split == 'train':
        return np.where(~np.isin(labels, unseen))[0]
    if zero_setting == 'ZSL':
        return np.where(np.isin(labels, unseen))[0]
    return np.arange(len(labels))


def count_samples(data_path, split='train', zero_setting='ZSL', zero_spilt_setting='ntu60_seen55_unseen5', **_):
    """Number of samples of a feeder config, from the labels only (cheap; an .npz is not converted)."""
    key = 'y_train' if split == 'train' else 'y_test'
    if os.path.isdir(data_path):
        labels = np.load(os.path.join(data_path, f'{key}.npy'))
    else:
        with np.load(data_path) as z:            # reads only this member of the npz
            labels = z[key]
    labels = labels.argmax(1)
    return len(select(labels, split, zero_setting, UNSEEN[zero_spilt_setting]))


class _Rows:
    """rows[i] -> sample idx[i] of a memory-mapped (N, T, 150) array, as (C, T, V, M) like the upstream feeder."""

    def __init__(self, arr, idx):
        self.arr, self.idx = arr, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = np.asarray(self.arr[self.idx[i]])                      # (T, 150), read from disk
        return x.reshape(x.shape[0], 2, 25, 3).transpose(3, 0, 2, 1)


class Feeder_Mmap(Feeder_Shiftgcn_Match):
    def load_data(self):
        d = npy_dir(self.data_path)
        part = 'train' if self.split == 'train' else 'test'
        if self.split not in ('train', 'test'):
            raise NotImplementedError('data split only supports train/test')
        labels = np.load(os.path.join(d, f'y_{part}.npy')).argmax(1)
        self.unseen_classes = UNSEEN[self.zero_spilt_setting]
        idx = select(labels, self.split, self.zero_setting, self.unseen_classes)
        self.label = labels[idx]
        self.data = _Rows(np.load(os.path.join(d, f'x_{part}.npy'), mmap_mode='r'), idx)
        self.sample_name = [f'{part}_{i}' for i in range(len(idx))]

    def get_mean_map(self):
        raise NotImplementedError('normalization=True is not supported by the memory-mapped feeder')
