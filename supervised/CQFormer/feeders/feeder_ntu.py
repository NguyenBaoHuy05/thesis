"""NTU feeder with the augmentation recipe of SkateFormer (feeders/feeder_ntu.py there).

Difference: it always returns joint coordinates in NTU order (the model computes the four
modalities per joint set).
"""
import random

import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, split='train', p_interval=(0.5, 1), window_size=64, thres=64,
                 uniform=True, aug_method='a123489', intra_p=0.5, inter_p=0.2):
        if split not in {'train', 'test'}:
            raise ValueError("split must be train or test")
        self.split = split
        self.p_interval = list(p_interval)
        self.window_size = window_size
        self.thres = thres
        self.uniform = uniform
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p

        npz = np.load(data_path)
        x_key, y_key = ('x_train', 'y_train') if split == 'train' else ('x_test', 'y_test')
        data = npz[x_key]
        N, T, _ = data.shape
        self.data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
        self.label = np.where(npz[y_key] > 0)[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        crop = tools.valid_crop_uniform if self.uniform else tools.valid_crop_resize
        data_numpy, index_t = crop(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)

        if self.split == 'train':
            p = np.random.rand(1)
            if p < self.intra_p:
                if 'a' in self.aug_method and np.random.rand(1) < 0.5:
                    data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method and num_people == 2 and np.random.rand(1) < 0.5:
                    data_numpy = data_numpy.copy()
                    data_numpy[:, :, :, 0] = 0
                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)
            elif p < self.intra_p + self.inter_p:
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = np.array(self.data[adain_idx])
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int64)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

        return np.array(data_numpy, dtype=np.float32), np.array(index_t, dtype=np.float32), int(label), index
