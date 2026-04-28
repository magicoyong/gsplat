import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d_covariance import project_gaussians_2d_covariance
from gsplat.rasterize_sum_gabor import rasterize_gabor_sum

from optimizer import Adan

from .gaussianimage_covariance import GaussianImage_Covariance


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(values, min=1e-6)
    return torch.log(torch.expm1(clamped))


class GaussianImage_Covariance_HSI(GaussianImage_Covariance):
    GAUSSIAN_GROUP_NAMES = {"xyz", "f_dc", "cov2d", "gabor_freqs", "gabor_weights"}
    supports_pruning = True

    def __init__(self, loss_type="L2", **kwargs):
        self.spectral_channels = int(kwargs["C"])
        self.opt_type = kwargs.get("opt_type", "adam")
        super().__init__(loss_type=loss_type, **kwargs)

        feature_init = _inverse_softplus(
            torch.full((self.init_num_points, self.spectral_channels), 0.05, dtype=torch.float32, device=self.device)
        )
        self._features_dc = nn.Parameter(feature_init + 0.01 * torch.randn_like(feature_init))
        self.register_buffer("background", torch.zeros(self.spectral_channels, device=self.device))

        self.training_setup(lr=kwargs["lr"], update_optimizer=True, quantize=False)

    @property
    def get_features(self):
        return F.softplus(self._features_dc)

    @property
    def get_cov2d_elements(self):
        return self._cov2d + self.cholesky_bound

    @property
    def get_cov2d(self):
        return self._cov2d

    def prepare_new_xyz(self, xy: torch.Tensor) -> torch.Tensor:
        return xy.to(device=self.device, dtype=torch.float32)

    def initialize_new_covariance(self, num_points: int) -> torch.Tensor:
        bias = torch.tensor([0.5, 0.0, 0.5], device=self.device, dtype=torch.float32)
        return torch.rand(num_points, 3, device=self.device) + bias

    def training_setup(self, lr, update_optimizer=False, quantize=False):
        if not update_optimizer and hasattr(self, "optimizer"):
            return

        param_groups = [
            {"params": [self._xyz], "lr": lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": lr, "name": "f_dc"},
            {"params": [self._cov2d], "lr": lr, "name": "cov2d"},
            {"params": [self.gabor_freqs], "lr": lr, "name": "gabor_freqs"},
            {"params": [self.gabor_weights], "lr": lr, "name": "gabor_weights"},
        ]

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(param_groups, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.quantize = False

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            if group_name not in tensors_dict:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = torch.cat(
                        (stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)), dim=0
                    )
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = torch.cat(
                        (stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)), dim=0
                    )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def update_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            if group_name not in tensors_dict:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = torch.zeros_like(extension_tensor)
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = torch.zeros_like(extension_tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group["name"]
            if group_name not in self.GAUSSIAN_GROUP_NAMES:
                optimizable_tensors[group_name] = group["params"][0]
                continue

            group_mask = self._expand_gabor_mask(mask) if group_name in ("gabor_freqs", "gabor_weights") else mask
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][group_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][group_mask]
                if "exp_avg_diff" in stored_state:
                    stored_state["exp_avg_diff"] = stored_state["exp_avg_diff"][group_mask]
                if "neg_pre_grad" in stored_state:
                    stored_state["neg_pre_grad"] = stored_state["neg_pre_grad"][group_mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][group_mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][group_mask].requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def check_non_semi_definite(self, cov2d=None):
        if cov2d is None:
            cov2d = self.get_cov2d_elements
        valid_points_mask = torch.logical_and(
            cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 > 0,
            torch.logical_and(cov2d[:, 0] > 0, cov2d[:, 2] > 0),
        )
        prune_mask = ~valid_points_mask
        to_prune_nums = torch.sum(prune_mask).item()

        return to_prune_nums, valid_points_mask

    def non_semi_definite_prune(self, H, W, cov2d=None):
        to_prune_nums, valid_points_mask = self.check_non_semi_definite(cov2d=cov2d)

        if to_prune_nums and self.cur_num_points - to_prune_nums > 0:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self.gabor_freqs = optimizable_tensors["gabor_freqs"]
            self.gabor_weights = optimizable_tensors["gabor_weights"]
            new_num_points = self._xyz.shape[0]

            self._opacity = nn.Parameter(torch.ones((new_num_points, 1), device=self.device), requires_grad=False)
            self._cov2d = optimizable_tensors["cov2d"]
            if self.SLV:
                self.cholesky_bound = self.cholesky_bound[valid_points_mask]
            torch.cuda.empty_cache()
        pruned_num_points = self._xyz.shape[0]
        self.cur_num_points = pruned_num_points
        return to_prune_nums, pruned_num_points

    def densification_postfix(self, new_xyz, new_features_dc, new_cov2d, new_opacities=None, new_bkcolor=None):
        none_definite, valid_mask = self.check_non_semi_definite(new_cov2d)
        n_valid = int(valid_mask.sum().item())
        new_gabor_freqs, new_gabor_weights = self._init_gabor_params(n_valid)

        d = {
            "xyz": new_xyz[valid_mask],
            "f_dc": new_features_dc[valid_mask],
            "cov2d": new_cov2d[valid_mask],
            "gabor_freqs": new_gabor_freqs,
            "gabor_weights": new_gabor_weights,
        }

        original_points_nums = self.cur_num_points
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._cov2d = optimizable_tensors["cov2d"]
        self.gabor_freqs = optimizable_tensors["gabor_freqs"]
        self.gabor_weights = optimizable_tensors["gabor_weights"]
        new_num_points = self._xyz.shape[0]
        self.cur_num_points = new_num_points
        self.add_stage += 1

        self._opacity = nn.Parameter(torch.ones((new_num_points, 1), device=self.device), requires_grad=False)

        if self.SLV:
            added = new_num_points - original_points_nums
            if added > 0:
                low_pass = min(self.H * self.W / (9 * math.pi * max(self.cur_num_points, 1)), 300.0)
                new_cholesky_bound = torch.tensor(
                    [low_pass, 0.0, low_pass], device=self.device, dtype=torch.float32
                ).view(1, 3).repeat(added, 1)
                self.cholesky_bound = torch.cat((self.cholesky_bound, new_cholesky_bound), dim=0)

        return new_num_points, none_definite

    def forward(self, isprint=False, H=None, W=None):
        H = self.H if H is None else H
        W = self.W if W is None else W
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_covariance(
            self.get_xyz,
            self.get_cov2d_elements,
            H,
            W,
            tile_bounds,
            coords_norm=self.coords_norm,
            clip_coe=self.gs_clip_coe,
            radius_clip=self.radius_clip,
            isprint=isprint,
        )

        out_img = rasterize_gabor_sum(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            self.get_features,
            self.get_opacity,
            self.get_gabor_freqs[:, 0],
            self.get_gabor_freqs[:, 1],
            self.get_gabor_weights,
            self.num_gabor,
            H,
            W,
            self.BLOCK_H,
            self.BLOCK_W,
            background=self.background,
            isprint=isprint,
            radius_clip=self.radius_clip,
        )

        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(1, H, W, self.spectral_channels).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img}

    def train_iter(self, H, W, gt_image, isprint=False):
        render_pkg = self.forward(isprint=isprint, H=H, W=W)
        image = render_pkg["render"]

        if self.loss_type == "L1":
            recon_loss = F.l1_loss(image, gt_image)
        else:
            recon_loss = F.mse_loss(image, gt_image)

        loss = recon_loss
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / max(mse_loss.item(), 1e-12))
        return loss, psnr, image.detach(), recon_loss.detach()