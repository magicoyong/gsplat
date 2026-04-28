import pytest
import torch

import gsplat.rasterize_sum_gabor as rasterize_gabor_module


def _reference_fake_gabor_image(
    xys,
    conics,
    colors,
    opacity,
    gabor_freqs_x,
    gabor_freqs_y,
    gabor_weights,
    num_freqs,
    img_height,
    img_width,
):
    weights = opacity.view(-1, 1)
    freq_x = gabor_freqs_x.view(-1, num_freqs)
    freq_y = gabor_freqs_y.view(-1, num_freqs)
    weight_grid = gabor_weights.view(-1, num_freqs)

    phase = xys[:, :1] * freq_x + xys[:, 1:] * freq_y
    modulation = 1.0 - weight_grid.sum(dim=1, keepdim=True) + (weight_grid * torch.cos(phase)).sum(dim=1, keepdim=True)
    sigma = 0.5 * (
        conics[:, 0:1] * xys[:, 0:1].square()
        + 2.0 * conics[:, 1:2] * xys[:, 0:1] * xys[:, 1:2]
        + conics[:, 2:3] * xys[:, 1:2].square()
    )
    gaussian = torch.exp(-sigma)
    pooled = (colors * weights * modulation * gaussian).sum(dim=0)
    return pooled.view(1, 1, -1).expand(img_height, img_width, -1).contiguous()


def _fake_forward_sum_gabor(
    tile_bounds,
    block,
    img_size,
    gaussian_ids_sorted,
    tile_bins,
    xys,
    conics,
    colors,
    opacity,
    background,
    gabor_weights,
    gabor_freqs_x,
    gabor_freqs_y,
    num_freqs,
):
    del tile_bounds, block, gaussian_ids_sorted, tile_bins, background
    img_width, img_height, _ = img_size
    out_img = _reference_fake_gabor_image(
        xys,
        conics,
        colors,
        opacity,
        gabor_freqs_x,
        gabor_freqs_y,
        gabor_weights,
        num_freqs,
        img_height,
        img_width,
    )
    final_ts = torch.ones(img_height, img_width, dtype=colors.dtype, device=colors.device)
    final_idx = torch.full((img_height, img_width), -1, dtype=torch.int32, device=colors.device)
    return out_img, final_ts, final_idx


def _fake_backward_sum_gabor(
    img_height,
    img_width,
    block_h,
    block_w,
    gaussian_ids_sorted,
    tile_bins,
    xys,
    conics,
    colors,
    opacity,
    background,
    gabor_weights,
    gabor_freqs_x,
    gabor_freqs_y,
    num_freqs,
    final_ts,
    final_idx,
    v_output,
    v_output_alpha,
):
    del block_h, block_w, gaussian_ids_sorted, tile_bins, background, final_ts, final_idx, v_output_alpha

    xys_ref = xys.detach().requires_grad_(True)
    conics_ref = conics.detach().requires_grad_(True)
    colors_ref = colors.detach().requires_grad_(True)
    opacity_ref = opacity.detach().requires_grad_(True)
    gabor_weights_ref = gabor_weights.detach().requires_grad_(True)
    gabor_freqs_x_ref = gabor_freqs_x.detach().requires_grad_(True)
    gabor_freqs_y_ref = gabor_freqs_y.detach().requires_grad_(True)

    out_img = _reference_fake_gabor_image(
        xys_ref,
        conics_ref,
        colors_ref,
        opacity_ref,
        gabor_freqs_x_ref,
        gabor_freqs_y_ref,
        gabor_weights_ref,
        num_freqs,
        img_height,
        img_width,
    )
    loss = (out_img * v_output).sum()
    grads = torch.autograd.grad(
        loss,
        [xys_ref, conics_ref, colors_ref, opacity_ref, gabor_weights_ref, gabor_freqs_x_ref, gabor_freqs_y_ref],
    )
    return grads


def test_rasterize_gabor_sum_multichannel_wrapper_cpu(monkeypatch):
    monkeypatch.setattr(rasterize_gabor_module._C, "rasterize_forward_sum_gabor", _fake_forward_sum_gabor)
    monkeypatch.setattr(rasterize_gabor_module._C, "rasterize_backward_sum_gabor", _fake_backward_sum_gabor)

    num_points = 3
    channels = 5
    num_freqs = 2
    img_height = 4
    img_width = 4

    xys = torch.tensor([[0.2, 0.1], [0.5, -0.3], [-0.4, 0.6]], dtype=torch.float32, requires_grad=True)
    depths = torch.zeros(num_points, dtype=torch.float32)
    radii = torch.ones(num_points, dtype=torch.int32)
    conics = torch.tensor([[0.9, 0.1, 1.2], [1.1, -0.05, 0.8], [0.7, 0.02, 1.0]], dtype=torch.float32, requires_grad=True)
    num_tiles_hit = torch.ones(num_points, dtype=torch.int32)
    colors = torch.randn(num_points, channels, dtype=torch.float32, requires_grad=True)
    opacity = torch.full((num_points, 1), 0.7, dtype=torch.float32, requires_grad=True)
    gabor_freqs_x = torch.tensor([0.2, -0.4, 0.1, 0.3, -0.2, 0.5], dtype=torch.float32, requires_grad=True)
    gabor_freqs_y = torch.tensor([-0.3, 0.1, 0.4, -0.2, 0.6, -0.5], dtype=torch.float32, requires_grad=True)
    gabor_weights = torch.tensor([0.1, 0.2, 0.15, 0.05, 0.12, 0.18], dtype=torch.float32, requires_grad=True)

    out = rasterize_gabor_module.rasterize_gabor_sum(
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
        img_height,
        img_width,
        BLOCK_H=4,
        BLOCK_W=4,
        background=torch.zeros(channels, dtype=torch.float32),
    )

    assert out.shape == (img_height, img_width, channels)

    loss = out.square().mean()
    loss.backward()

    assert colors.grad is not None and colors.grad.shape == colors.shape
    assert opacity.grad is not None and opacity.grad.shape == opacity.shape
    assert conics.grad is not None and conics.grad.shape == conics.shape
    assert xys.grad is not None and xys.grad.shape == xys.shape
    assert gabor_weights.grad is not None and gabor_weights.grad.shape == gabor_weights.shape
    assert gabor_freqs_x.grad is not None and gabor_freqs_x.grad.shape == gabor_freqs_x.shape
    assert gabor_freqs_y.grad is not None and gabor_freqs_y.grad.shape == gabor_freqs_y.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rasterize_gabor_sum_multichannel_cuda():
    device = torch.device("cuda:0")
    num_points = 4
    channels = 5
    num_freqs = 2
    img_height = 8
    img_width = 8

    xys = torch.tensor(
        [[1.5, 1.0], [3.0, 2.0], [4.0, 5.0], [6.0, 3.5]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    depths = torch.linspace(0.1, 0.4, num_points, dtype=torch.float32, device=device)
    radii = torch.full((num_points,), 2, dtype=torch.int32, device=device)
    conics = torch.tensor(
        [[0.6, 0.02, 0.7], [0.8, -0.03, 0.9], [0.7, 0.01, 0.6], [0.9, -0.02, 0.8]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    num_tiles_hit = torch.ones(num_points, dtype=torch.int32, device=device)
    colors = torch.rand(num_points, channels, dtype=torch.float32, device=device, requires_grad=True)
    opacity = torch.full((num_points, 1), 0.6, dtype=torch.float32, device=device, requires_grad=True)
    gabor_freqs_x = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.25, 0.3, -0.15], device=device, requires_grad=True)
    gabor_freqs_y = torch.tensor([-0.1, 0.35, 0.2, -0.25, 0.45, -0.3, 0.15, 0.05], device=device, requires_grad=True)
    gabor_weights = torch.tensor([0.1, 0.15, 0.08, 0.12, 0.07, 0.18, 0.11, 0.09], device=device, requires_grad=True)

    out = rasterize_gabor_module.rasterize_gabor_sum(
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
        img_height,
        img_width,
        BLOCK_H=8,
        BLOCK_W=8,
        background=torch.zeros(channels, dtype=torch.float32, device=device),
    )

    assert out.shape == (img_height, img_width, channels)

    loss = out.square().mean()
    loss.backward()

    assert colors.grad is not None and colors.grad.shape == colors.shape
    assert opacity.grad is not None and opacity.grad.shape == opacity.shape
    assert conics.grad is not None and conics.grad.shape == conics.shape
    assert xys.grad is not None and xys.grad.shape == xys.shape
    assert gabor_weights.grad is not None and gabor_weights.grad.shape == gabor_weights.shape
    assert gabor_freqs_x.grad is not None and gabor_freqs_x.grad.shape == gabor_freqs_x.shape
    assert gabor_freqs_y.grad is not None and gabor_freqs_y.grad.shape == gabor_freqs_y.shape