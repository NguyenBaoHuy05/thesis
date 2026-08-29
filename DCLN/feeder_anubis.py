import sys
sys.path.extend(['../'])
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import feeder_tools as tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, window_size=-1, normalization=False, 
                 debug=False, use_mmap=True, bone=False, vel=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        
        self.label = np.array(self.label)
        
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        print(f"shape: {self.data.shape}")
        print(f"label: {len(self.label)}")
        
        # debug
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100] if isinstance(self.sample_name, list) else list(range(100))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if self.window_size > 0:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            
        if self.bone:
            # Azure Kinect
            from bone_pair_anu import anubis_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in anubis_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            
            bone_data_numpy[:, :, 0] = data_numpy[:, :, 0]
            data_numpy = bone_data_numpy
        else:
            center_joint = 0  
            trajectory = data_numpy[:, :, center_joint]
            data_numpy = data_numpy - data_numpy[:, :, center_joint:center_joint+1]
            data_numpy[:, :, center_joint] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
            
        data_numpy = np.nan_to_num(data_numpy)
        data_numpy[data_numpy == -np.inf] = 0
        data_numpy[data_numpy == np.inf] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        if score is None or len(score) == 0:
            return 0.0
            
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label) if i < rank.shape[0]]
        return sum(hit_top_k) * 1.0 / len(hit_top_k) if len(hit_top_k) > 0 else 0.0