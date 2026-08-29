import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
)

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear', align_corners=False).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def downsample(data_numpy, step, random_sample=True):
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]

def temporal_slice(data_numpy, step):
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)

def mean_subtractor(data_numpy, mean):
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy

def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy

def random_choose(data_numpy, size, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy

def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = torch.zeros(rot.shape[0], 1)
    ones = torch.ones(rot.shape[0], 1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=1)

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def random_rot(data_numpy, theta=0.3):
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch.numpy() if isinstance(data_torch, torch.Tensor) else data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
