"""Python bindings for Gabor rasterization."""

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def _validate_gabor_inputs(
    xys: Float[Tensor, "*batch 2"],
    colors: Float[Tensor, "*batch channels"],
    background: Optional[Float[Tensor, "channels"]],
    gabor_freqs_x: Float[Tensor, "*batch num_freqs"],
    gabor_freqs_y: Float[Tensor, "*batch num_freqs"],
    gabor_weights: Float[Tensor, "*batch num_freqs"],
    num_freqs: int,
) -> Tuple[
    Float[Tensor, "channels"],
    Float[Tensor, "num_points_num_freqs"],
    Float[Tensor, "num_points_num_freqs"],
    Float[Tensor, "num_points_num_freqs"],
]:
    if background is not None:
        if background.shape[0] != colors.shape[-1]:
            raise ValueError(
                f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
            )
    else:
        background = torch.ones(colors.shape[-1], dtype=torch.float32, device=colors.device)

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    num_points = xys.size(0)
    expected_size = num_points * num_freqs
    if gabor_freqs_x.numel() != expected_size:
        raise ValueError(f"gabor_freqs_x size mismatch: expected {expected_size}, got {gabor_freqs_x.numel()}")
    if gabor_freqs_y.numel() != expected_size:
        raise ValueError(f"gabor_freqs_y size mismatch: expected {expected_size}, got {gabor_freqs_y.numel()}")
    if gabor_weights.numel() != expected_size:
        raise ValueError(f"gabor_weights size mismatch: expected {expected_size}, got {gabor_weights.numel()}")

    return (
        background.contiguous(),
        gabor_freqs_x.contiguous().view(-1),
        gabor_freqs_y.contiguous().view(-1),
        gabor_weights.contiguous().view(-1),
    )


def rasterize_gabor_sum(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    gabor_freqs_x: Float[Tensor, "*batch num_freqs"],
    gabor_freqs_y: Float[Tensor, "*batch num_freqs"],
    gabor_weights: Float[Tensor, "*batch num_freqs"],
    num_freqs: int,
    img_height: int,
    img_width: int,
    BLOCK_H: int = 16,
    BLOCK_W: int = 16,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
    radius_clip: float = 1.0,
    isprint: bool = False,
) -> Tensor:
    del return_alpha

    if colors.dtype == torch.uint8:
        colors = colors.float() / 255

    background, gabor_freqs_x, gabor_freqs_y, gabor_weights = _validate_gabor_inputs(
        xys,
        colors,
        background,
        gabor_freqs_x,
        gabor_freqs_y,
        gabor_weights,
        num_freqs,
    )

    return RasterizeGaborSum.apply(
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        gabor_freqs_x,
        gabor_freqs_y,
        gabor_weights,
        num_freqs,
        img_height,
        img_width,
        BLOCK_H,
        BLOCK_W,
        background,
        radius_clip,
        isprint,
    )


def rasterize_gabor_plus(*args, **kwargs):
    return rasterize_gabor_sum(*args, **kwargs)


class RasterizeGaborSum(Function):
    """Rasterizes 2D gaussians with Gabor filter modulation."""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        gabor_freqs_x: Float[Tensor, "num_points_num_freqs"],
        gabor_freqs_y: Float[Tensor, "num_points_num_freqs"],
        gabor_weights: Float[Tensor, "num_points_num_freqs"],
        num_freqs: int,
        img_height: int,
        img_width: int,
        BLOCK_H: int = 16,
        BLOCK_W: int = 16,
        background: Optional[Float[Tensor, "channels"]] = None,
        radius_clip: float = 1.0,
        isprint: bool = False,
    ):
        del isprint

        num_points = xys.size(0)
        BLOCK_X, BLOCK_Y = BLOCK_W, BLOCK_H
        tile_bounds = (
            (img_width + BLOCK_X - 1) // BLOCK_X,
            (img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        block = (BLOCK_X, BLOCK_Y, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = torch.ones(img_height, img_width, colors.shape[-1], device=xys.device) * background
            gaussian_ids_sorted = torch.zeros(0, dtype=torch.int32, device=xys.device)
            tile_bins = torch.zeros(0, 2, dtype=torch.int32, device=xys.device)
            final_Ts = torch.zeros(img_height * img_width, device=xys.device)
            final_idx = torch.zeros(img_height * img_width, dtype=torch.int32, device=xys.device)
        else:
            (
                _isect_ids_unsorted,
                _gaussian_ids_unsorted,
                _isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                radius_clip,
            )

            out_img, final_Ts, final_idx = _C.rasterize_forward_sum_gabor(
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
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.BLOCK_H = BLOCK_H
        ctx.BLOCK_W = BLOCK_W
        ctx.num_intersects = num_intersects
        ctx.num_freqs = num_freqs

        ctx.save_for_backward(
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
            final_Ts,
            final_idx,
        )
        return out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        img_height = ctx.img_height
        img_width = ctx.img_width
        BLOCK_H = ctx.BLOCK_H
        BLOCK_W = ctx.BLOCK_W
        num_intersects = ctx.num_intersects
        num_freqs = ctx.num_freqs

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
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
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
            v_weights = torch.zeros_like(gabor_weights)
            v_freqs_x = torch.zeros_like(gabor_freqs_x)
            v_freqs_y = torch.zeros_like(gabor_freqs_y)
        else:
            v_xy, v_conic, v_colors, v_opacity, v_weights, v_freqs_x, v_freqs_y = _C.rasterize_backward_sum_gabor(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
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
                final_Ts,
                final_idx,
                v_out_img.contiguous(),
                v_out_alpha.contiguous(),
            )

        return (
            v_xy,
            None,
            None,
            v_conic,
            None,
            v_colors,
            v_opacity,
            v_freqs_x,
            v_freqs_y,
            v_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )