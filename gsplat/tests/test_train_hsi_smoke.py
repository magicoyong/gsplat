from pathlib import Path

import numpy as np
import pytest
import torch

import train_hsi
import models.gaussianimage_covariance_hsi as covariance_hsi_module


@pytest.mark.parametrize(
    "argv",
    [
        ["--rank", "4"],
        ["--no_nmf"],
        ["--lora_rank", "2"],
        ["--lora_alpha", "0.1"],
        ["--freeze_endmember"],
    ],
)
def test_train_hsi_rejects_removed_nmf_args(argv):
    with pytest.raises(SystemExit):
        train_hsi.parse_args(argv)


def _fake_project_gaussians_2d_covariance(
    xys,
    cov2d,
    h,
    w,
    tile_bounds,
    coords_norm=False,
    clip_coe=3.0,
    radius_clip=1.0,
    isprint=False,
):
    del h, w, tile_bounds, coords_norm, clip_coe, radius_clip, isprint
    num_points = xys.shape[0]
    depths = torch.zeros(num_points, dtype=torch.float32, device=xys.device)
    radii = torch.ones(num_points, dtype=torch.int32, device=xys.device)
    num_tiles_hit = torch.ones(num_points, dtype=torch.int32, device=xys.device)
    return xys, depths, radii, cov2d, num_tiles_hit


def _fake_rasterize_gabor_sum(
    xys,
    depths,
    radii,
    conics,
    num_tiles_hit,
    colors,
    opacity,
    gabor_freqs_x,
    gabor_freqs_y,
    gabor_weights,
    num_freqs,
    h,
    w,
    block_h,
    block_w,
    background=None,
    isprint=False,
    radius_clip=1.0,
):
    del depths, radii, num_tiles_hit, block_h, block_w, background, isprint, radius_clip
    freq_x = gabor_freqs_x.view(-1, num_freqs)
    freq_y = gabor_freqs_y.view(-1, num_freqs)
    weight_grid = gabor_weights.view(-1, num_freqs)

    phase = xys[:, :1] * freq_x + xys[:, 1:] * freq_y
    modulation = 1.0 - weight_grid.sum(dim=1, keepdim=True) + (weight_grid * torch.cos(phase)).sum(dim=1, keepdim=True)
    gaussian = torch.exp(-0.1 * (conics[:, 0:1].abs() + conics[:, 2:3].abs()))
    coord_term = 1.0 + 0.01 * torch.tanh(xys[:, :1] + xys[:, 1:2])
    pooled = (colors * opacity * modulation * gaussian * coord_term).sum(dim=0)
    return pooled.view(1, 1, -1).expand(h, w, -1).contiguous()


def test_train_hsi_cpu_forward_backward_smoke(monkeypatch, tmp_path: Path):
    cube = np.random.rand(4, 3, 5).astype(np.float32)

    monkeypatch.setattr(train_hsi, "load_dataset", lambda dataset_name: cube)
    monkeypatch.setattr(train_hsi, "list_available_datasets", lambda: ["toy"])
    monkeypatch.setattr(covariance_hsi_module, "project_gaussians_2d_covariance", _fake_project_gaussians_2d_covariance)
    monkeypatch.setattr(covariance_hsi_module, "rasterize_gabor_sum", _fake_rasterize_gabor_sum)

    args = train_hsi.parse_args(
        [
            "--dataset",
            "toy",
            "--iterations",
            "1",
            "--num_points",
            "4",
            "--max_num_points",
            "4",
            "--grow_iter",
            "10",
            "--prune_iter",
            "10",
        ]
    )

    trainer = train_hsi.HSIFullTrainer(args, dataset_name="toy", experiment_root=tmp_path)
    loss, _psnr, out_image, recon_loss = trainer.model.train_iter(
        trainer.H,
        trainer.W,
        trainer.gt_image,
        isprint=False,
    )

    assert out_image.shape == (1, trainer.C, trainer.H, trainer.W)
    assert recon_loss.item() >= 0.0
    assert trainer.model._features_dc.grad is not None
    assert torch.isfinite(loss)