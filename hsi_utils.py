import numpy as np
import scipy.io
import torch


def load_dataset(name):
    """Load a normalized hyperspectral cube in HWC layout."""
    name = name.lower()
    if name == "urban":
        cube = scipy.io.loadmat("HSI/data/Urban_R162.mat")["Y"].astype(np.float32)
        for channel in range(162):
            cube[channel, :] /= np.maximum(np.max(cube[channel, :]), 1e-8)
        cube = cube.reshape(162, 307, 307).transpose(2, 1, 0)
    elif name == "salinas":
        cube = scipy.io.loadmat("HSI/data/Salinas_crop.mat")["I"].astype(np.float32)
        cube = np.clip(cube, 0, None)
        for channel in range(204):
            cube[:, :, channel] /= np.maximum(np.max(cube[:, :, channel]), 1e-8)
    elif name == "jasperridge":
        cube = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")["Y"].astype(np.float32)
        for channel in range(198):
            cube[channel, :] /= np.maximum(np.max(cube[channel, :]), 1e-8)
        cube = cube.reshape(198, 100, 100).transpose(2, 1, 0)
    elif name == "paviau":
        cube = scipy.io.loadmat("HSI/data/PaviaU.mat")["paviaU"].astype(np.float32)
        for channel in range(103):
            cube[:, :, channel] /= np.maximum(np.max(cube[:, :, channel]), 1e-8)
        cube = cube[-340:, :, :]
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return cube


def compute_sam(target_image, image):
    if torch.is_tensor(target_image) or torch.is_tensor(image):
        target = target_image if torch.is_tensor(target_image) else torch.as_tensor(target_image)
        pred = image if torch.is_tensor(image) else torch.as_tensor(image, device=target.device)
        target = target.to(dtype=torch.float32)
        pred = pred.to(dtype=torch.float32, device=target.device)

        dot = torch.sum(target * pred, dim=-1)
        norm_target = torch.linalg.norm(target, dim=-1)
        norm_pred = torch.linalg.norm(pred, dim=-1)
        cos_angle = dot / torch.clamp(norm_target * norm_pred, min=1e-8)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angles = torch.acos(cos_angle) * (180.0 / np.pi)
        return float(angles.mean().item())

    target = np.asarray(target_image, dtype=np.float32)
    pred = np.asarray(image, dtype=np.float32)
    dot = np.sum(target * pred, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    cos_angle = dot / np.clip(norm_target * norm_pred, 1e-8, None)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle) * (180.0 / np.pi)
    return float(np.mean(angles))


def total_variation_loss(image):
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def spectral_smoothness_loss(image):
    if image.shape[1] <= 1:
        return image.new_zeros(())
    return (image[:, 1:, :, :] - image[:, :-1, :, :]).abs().mean()