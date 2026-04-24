import argparse
import numpy as np
import scipy.io
from sklearn.decomposition import NMF
import time
import os

def load_dataset(name):
    """Load and normalize hyperspectral dataset."""
    name = name.lower()
    if name == "salinas":
        data = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        data = np.clip(data, 0, None)
        for i in range(204):
            data[:, :, i] /= np.max(data[:, :, i])
    elif name == "urban":
        data = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(162, 307, 307).transpose(2, 1, 0)
    elif name == "jasperridge":
        data = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(198, 100, 100).transpose(2, 1, 0)
    elif name == "paviau":
        data = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            data[:, :, i] /= np.max(data[:, :, i])
        data = data[-340:, :, :]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # (C, H, W) → (C, H*W)
    data = np.transpose(data, (2, 0, 1)).reshape(data.shape[2], -1)
    return data


def nmf_initialization(I, rank, dataset_name=None):
    """Perform NMF initialization from a tensor/array in CHW, HWC, BCHW, or (C, H*W) format."""
    if hasattr(I, "detach"):
        I = I.detach().float().cpu()
        if I.dim() == 4 and I.shape[0] == 1:
            I = I.squeeze(0).reshape(I.shape[1], -1)
        elif I.dim() == 3:
            if I.shape[0] <= I.shape[1] and I.shape[0] <= I.shape[2]:
                I = I.reshape(I.shape[0], -1)
            else:
                I = I.permute(2, 0, 1).reshape(I.shape[-1], -1)
        I = I.numpy()
    else:
        I = np.asarray(I)
        if I.ndim == 4 and I.shape[0] == 1:
            I = I.squeeze(0).reshape(I.shape[1], -1)
        elif I.ndim == 3:
            if I.shape[0] <= I.shape[1] and I.shape[0] <= I.shape[2]:
                I = I.reshape(I.shape[0], -1)
            else:
                I = np.transpose(I, (2, 0, 1)).reshape(I.shape[-1], -1)

    if dataset_name is not None:
        print(f"Running NMF initialization on {dataset_name} with rank={rank}")

    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T  # shape (rank, channels)
    abundance = nmf.components_.T       # shape (H*W, rank)

    return endmember.astype(np.float32), abundance.astype(np.float32)