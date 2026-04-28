import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum_gabor import rasterize_gabor_sum
from optimizer import Adan


def _inverse_softplus_cho(values: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(values, min=1e-6)
    return torch.log(torch.expm1(clamped))


class GaussianImage_Cholesky_HSI(nn.Module):
    supports_pruning = False

    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = int(kwargs["num_points"])
        self.cur_num_points = self.init_num_points
        self.H, self.W = int(kwargs["H"]), int(kwargs["W"])
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.device = kwargs["device"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.spectral_channels = int(kwargs["C"])
        self.opt_type = kwargs.get("opt_type", "adam")

        args = kwargs["args"]
        self.num_gabor = kwargs.get("num_gabor", getattr(args, "num_gabor", 2))
        self.radius_clip = float(getattr(args, "radius_clip", 1.0))

        self._xyz = nn.Parameter(
            torch.atanh(2 * (torch.rand(self.init_num_points, 2, device=self.device) - 0.5))
        )
        self._cholesky = nn.Parameter(torch.rand((self.init_num_points, 3), device=self.device))
        self.register_buffer("_opacity", torch.ones((self.init_num_points, 1), device=self.device))
        self.register_buffer(
            "cholesky_bound",
            torch.tensor([0.5, 0.0, 0.5], device=self.device).view(1, 3),
        )

        self.gabor_freqs = nn.Parameter(
            (torch.rand(self.init_num_points * self.num_gabor, 2, device=self.device) - 0.5) * 0.002
        )
        self.gabor_weights = nn.Parameter(
            torch.rand(self.init_num_points * self.num_gabor, 1, device=self.device) * (-5.0)
        )

        feature_init = _inverse_softplus_cho(
            torch.full((self.init_num_points, self.spectral_channels), 0.05, dtype=torch.float32, device=self.device)
        )
        self._features_dc = nn.Parameter(feature_init + 0.01 * torch.randn_like(feature_init))
        self.register_buffer("background", torch.zeros(self.spectral_channels, device=self.device))

        self.add_stage = 0
        self.training_setup(lr=kwargs["lr"], update_optimizer=True, quantize=False)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        return F.softplus(self._features_dc)

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_gabor_freqs(self):
        return torch.exp(self.gabor_freqs)

    @property
    def get_gabor_weights(self):
        return torch.sigmoid(self.gabor_weights)

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

    def prepare_new_xyz(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=self.device, dtype=torch.float32)
        if self.W > 1:
            x = 2.0 * xy[:, 0] / (self.W - 1) - 1.0
        else:
            x = torch.zeros_like(xy[:, 0])
        if self.H > 1:
            y = 2.0 * xy[:, 1] / (self.H - 1) - 1.0
        else:
            y = torch.zeros_like(xy[:, 1])
        return torch.atanh(torch.stack([x, y], dim=1).clamp(-0.999999, 0.999999))

    def _init_gabor_params(self, num_points: int):
        return (
            (torch.rand(num_points * self.num_gabor, 2, device=self.device) - 0.5) * 0.002,
            torch.rand(num_points * self.num_gabor, 1, device=self.device) * (-5),
        )

    def initialize_new_covariance(self, num_points: int) -> torch.Tensor:
        return torch.rand(num_points, 3, device=self.device)

    def training_setup(self, lr, update_optimizer=False, quantize=False):
        if not update_optimizer and hasattr(self, "optimizer"):
            return

        param_groups = [
            {"params": [self._xyz], "lr": lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": lr, "name": "f_dc"},
            {"params": [self._cholesky], "lr": lr, "name": "cholesky"},
            {"params": [self.gabor_freqs], "lr": lr, "name": "gabor_freqs"},
            {"params": [self.gabor_weights], "lr": lr, "name": "gabor_weights"},
        ]

        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(param_groups, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.quantize = False

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

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
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
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

    def check_non_semi_definite(self, cov2d=None):
        count = self.cur_num_points if cov2d is None else int(cov2d.shape[0])
        return 0, torch.ones(count, dtype=torch.bool, device=self.device)

    def non_semi_definite_prune(self, H, W, cov2d=None):
        return 0, self.cur_num_points

    def densification_postfix(self, new_xyz, new_features_dc, new_cov2d, new_opacities=None, new_bkcolor=None):
        new_gabor_freqs, new_gabor_weights = self._init_gabor_params(int(new_xyz.shape[0]))
        optimizable_tensors = self.cat_tensors_to_optimizer(
            {
                "xyz": new_xyz,
                "f_dc": new_features_dc,
                "cholesky": new_cov2d,
                "gabor_freqs": new_gabor_freqs,
                "gabor_weights": new_gabor_weights,
            }
        )

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self.gabor_freqs = optimizable_tensors["gabor_freqs"]
        self.gabor_weights = optimizable_tensors["gabor_weights"]
        self.cur_num_points = self._xyz.shape[0]
        self.add_stage += 1
        self._opacity = nn.Parameter(torch.ones((self.cur_num_points, 1), device=self.device), requires_grad=False)
        return self.cur_num_points, 0

    def forward(self, isprint=False, H=None, W=None):
        H = self.H if H is None else H
        W = self.W if W is None else W
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz,
            self.get_cholesky_elements,
            H,
            W,
            tile_bounds,
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

        out_img = torch.clamp(out_img, min=0.0, max=1.0)
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