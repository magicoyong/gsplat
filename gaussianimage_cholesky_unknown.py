from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum_gabor import rasterize_gabor_sum
from optimizer import Adan
from quantize import *
from utils import *


def _inverse_softplus(values):
    clamped = torch.clamp(values, min=1e-6)
    return torch.log(torch.expm1(clamped))


class GaussianImage_Cholesky_EA(nn.Module):
    def __init__(self, loss_type="L2", **kwargs): #L2 SSIM Fusion1 Fusion2
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W, self.rank, self.C = kwargs["H"], kwargs["W"], kwargs["rank"], kwargs["C"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.num_gabor = int(kwargs.get("num_gabor", 2))
        self.lora_rank = int(kwargs.get("lora_rank", 2))
        self.lora_alpha = float(kwargs.get("lora_alpha", 0.1))
        self.freeze_endmember = bool(kwargs.get("freeze_endmember", False))
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        #torch.seed(1234)
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        self.device = kwargs["device"]
        self.image = kwargs["GT"]#.to(torch.float32) # (1, C, H, W)
        self.register_buffer(
            "endmember",
            torch.as_tensor(kwargs["E"], dtype=torch.float32, device=self.device).contiguous(),
        )
        
        torch.cuda.synchronize()
        E_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"E0 GPU memory usage: {E_gpu_memory - gpu_memory} MB")

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        #self._xyz = self._initialize_xyz_from_abundance(self.abundance, self.init_num_points)
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3)) # 0.5 * torch.rand(self.init_num_points, 3)
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))

        feature_init = _inverse_softplus(
            torch.full((self.init_num_points, self.rank), 0.05, dtype=torch.float32, device=self.device)
        )
        self._features_dc = nn.Parameter(feature_init + 0.01 * torch.randn_like(feature_init))
        self.gabor_freqs = nn.Parameter(
            (torch.rand(self.init_num_points * self.num_gabor, 2, device=self.device) - 0.5) * 0.002
        )
        self.gabor_weights = nn.Parameter(
            torch.rand(self.init_num_points * self.num_gabor, 1, device=self.device) * (-5.0)
        )
        self.lora_U = nn.Parameter(torch.empty(self.rank, self.lora_rank, device=self.device))
        self.lora_V = nn.Parameter(torch.empty(self.lora_rank, self.C, device=self.device))
        self.lora_scale_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        nn.init.kaiming_uniform_(self.lora_U, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_V, a=np.sqrt(5.0))
        self.lora_V.data.mul_(1e-3)
        if self.freeze_endmember:
            self.lora_U.requires_grad_(False)
            self.lora_V.requires_grad_(False)
            self.lora_scale_logit.requires_grad_(False)

        self.coef = nn.Parameter(torch.tensor(0.0))

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(self.rank))
        self.register_buffer("abundance_background3", torch.zeros(3, device=self.device))
        self.register_buffer("abundance_background4", torch.zeros(4, device=self.device))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.channel_groups = self._build_channel_groups(self.rank)
        
        self.endmember_quantizer = FakeQuantizationHalf.apply #UniformQuantizer(signed=False, bits=6, learned=True, num_channels=self.rank)
        self.xyz_quantizer = FakeQuantizationHalf.apply 
        self.features_dc_quantizer = VectorQuantizer(codebook_dim=self.rank, codebook_size=72,num_quantizers=2, vector_type="vector", kmeans_iters=8) 
        self.cholesky_quantizer = UniformQuantizer(signed=False, bits=8, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @staticmethod
    @lru_cache(maxsize=None)
    def _plan_group_sizes(num_channels):
        if num_channels == 0:
            return tuple()
        if num_channels < 3:
            raise ValueError(f"rank={num_channels} cannot be decomposed into exact 3/4-channel groups")

        candidates = []
        if num_channels >= 3:
            try:
                tail = GaussianImage_Cholesky_EA._plan_group_sizes(num_channels - 3)
                candidates.append((1 + len(tail), ((3, 3),) + tail))
            except ValueError:
                pass
        if num_channels >= 4:
            try:
                tail = GaussianImage_Cholesky_EA._plan_group_sizes(num_channels - 4)
                candidates.append((1 + len(tail), ((4, 4),) + tail))
            except ValueError:
                pass
        if not candidates:
            raise ValueError(f"rank={num_channels} cannot be decomposed into exact 3/4-channel groups")
        return min(candidates, key=lambda item: item[0])[1]

    def _build_channel_groups(self, num_channels):
        groups = []
        start = 0
        for actual_channels, kernel_channels in self._plan_group_sizes(num_channels):
            groups.append((start, start + actual_channels, kernel_channels))
            start += actual_channels
        return groups
    
    @property
    def get_features(self):
        return F.softplus(self._features_dc)
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    @property
    def get_gabor_freqs(self):
        return torch.exp(self.gabor_freqs)

    @property
    def get_gabor_weights(self):
        return torch.sigmoid(self.gabor_weights)

    def get_calibrated_endmember(self):
        if self.freeze_endmember:
            return torch.clamp(self.endmember, min=1e-6)
        delta = torch.tanh(self.lora_U @ self.lora_V)
        scale = self.lora_alpha * torch.tanh(self.lora_scale_logit)
        return torch.clamp(self.endmember + scale * delta, min=1e-6)

    def get_endmember_delta_norm(self):
        with torch.no_grad():
            if self.freeze_endmember:
                return 0.0
            delta = torch.tanh(self.lora_U @ self.lora_V)
            scale = self.lora_alpha * torch.tanh(self.lora_scale_logit)
            return float((scale * delta).norm().item())

    def _render_abundance_groups(self, features, depths, conics, num_tiles_hit):
        rendered_groups = []
        gabor_freqs = self.get_gabor_freqs
        gabor_weights = self.get_gabor_weights
        for start, end, kernel_channels in self.channel_groups:
            feature_group = features[:, start:end].contiguous()
            background = self.abundance_background3 if kernel_channels == 3 else self.abundance_background4
            rendered = rasterize_gabor_sum(
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                feature_group,
                self.get_opacity,
                gabor_freqs[:, 0],
                gabor_freqs[:, 1],
                gabor_weights,
                self.num_gabor,
                self.H,
                self.W,
                self.BLOCK_H,
                self.BLOCK_W,
                background=background,
                return_alpha=False,
            )
            rendered_groups.append(rendered[..., : end - start])
        return torch.cat(rendered_groups, dim=2)

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz,
            self.get_cholesky_elements,
            self.H,
            self.W,
            self.tile_bounds,
        )
        abundance = self._render_abundance_groups(self.get_features, depths, conics, num_tiles_hit)
        abundance = torch.clamp(abundance, min=0.0)
        abundance_flat = abundance.view(self.H * self.W, self.rank).contiguous()
        endmember = self.get_calibrated_endmember()
        flatimage = abundance_flat @ endmember
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
        return {"render": image, "abundance": abundance_flat, "endmember": endmember}

    def train_iter(self, endmember_weight=0.0):
        render_pkg = self.forward()
        image = render_pkg["render"]
        endmember = render_pkg["endmember"]

        loss = loss_fn(image, self.image, self.loss_type, lambda_value=0.7)
        if endmember_weight > 0:
            loss = loss + endmember_weight * F.mse_loss(endmember, self.endmember)

        loss.backward()

        with torch.no_grad():
            mse_loss = F.mse_loss(image, self.image)
            psnr = 10 * math.log10(1.0 / max(mse_loss.item(), 1e-12))
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr, self.get_endmember_delta_norm()
    
    def forward_quantize(self):
        # quantize plan1: "GaussianImage"
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(FakeQuantizationHalf.apply(self._xyz))#FakeQuantizationHalf.apply(self._xyz)
        
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)#self._cholesky, 0, 32*self.init_num_points*3 32*self.init_num_points*3#self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        
        features, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, 
                cholesky_elements, self.H, self.W, self.tile_bounds)

        abundance = self._render_abundance_groups(features, depths, conics, num_tiles_hit)
        abundance = torch.clamp(abundance, min=0.0)
        out_img = abundance.view(self.H * self.W, self.rank).contiguous() * torch.exp(self.coef)
        
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {
            "render": out_img,
            "vq_loss": vq_loss,
            "unit_bit": [m_bit, s_bit, r_bit, c_bit],
            "endmember": self.get_calibrated_endmember(),
        }

    def train_iter_quantize(self, endmember_weight=0.0):
        render_pkg = self.forward_quantize()
        A = render_pkg["render"] # (H * W, rank)
        E = FakeQuantizationHalf.apply(render_pkg["endmember"])
        flatimage = A @ E # image: (H * W, C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =  loss_fn(image, self.image, self.loss_type, lambda_value=0.7) + 0.05 * render_pkg['vq_loss']
        if endmember_weight > 0:
            loss = loss + endmember_weight * F.mse_loss(render_pkg["endmember"], self.endmember)
        loss.backward()

        with torch.no_grad():
            I = (A @ render_pkg["endmember"]).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_per_channel = F.mse_loss(I, self.image, reduction='none')
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            psnr = psnr_per_channel.mean().item()
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr
