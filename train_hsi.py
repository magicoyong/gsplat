import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import yaml
from tqdm import tqdm

from hsi_utils import compute_sam, list_available_datasets, load_dataset
from models.gaussianimage_covariance_hsi import GaussianImage_Covariance_HSI
from models.gaussianimage_cholesky_hsi import GaussianImage_Cholesky_HSI
from models.utils import LogWriter

_MODEL_CLASSES = {
    "covariance": GaussianImage_Covariance_HSI,
    "cholesky": GaussianImage_Cholesky_HSI,
}

class HSIFullTrainer:
    def __init__(self, args, dataset_name: str, experiment_root: Path):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name.lower()
        self.experiment_root = experiment_root

        cube_hwc = load_dataset(self.dataset_name)
        self.H, self.W, self.C = cube_hwc.shape
        self.gt_cube = cube_hwc
        # Convert to tensor
        gt = torch.as_tensor(cube_hwc, dtype=torch.float32, device=self.device)
        gt = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        self.gt_image = torch.clamp(gt, 0, 1)

        self.log_dir = self._build_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        block_h, block_w = 16, 16
        model_cls = _MODEL_CLASSES[args.covariance_type]
        self.model = model_cls(
            loss_type=args.loss_type,
            opt_type=args.opt_type,
            num_points=args.num_points,
            H=self.H,
            W=self.W,
            C=self.C,
            BLOCK_H=block_h,
            BLOCK_W=block_w,
            device=self.device,
            lr=args.lr,
            num_gabor=args.num_gabor,
            quantize=False,
            args=args,
            logwriter=self.logwriter,
        ).to(self.device)
        self.max_num_points = args.max_num_points

    def _build_log_dir(self) -> Path:
        return self.experiment_root / self.dataset_name

    def _compute_bandwise_ssim(self, reconstruction: torch.Tensor) -> float:
        reconstruction = reconstruction.clamp(0.0, 1.0)
        scores = [
            ssim(
                reconstruction[:, band : band + 1],
                self.gt_image[:, band : band + 1],
                data_range=1.0,
                size_average=True,
            )
            for band in range(reconstruction.shape[1])
        ]
        return float(torch.stack(scores).mean().item())

    def _initialize_new_features(self, sampled_xy: torch.Tensor) -> torch.Tensor:
        # Match the model's init scheme (inverse_softplus(0.05) + jitter) so newly
        # densified points start near abundance=0.05 instead of softplus(0)=0.69.
        n = int(sampled_xy.shape[0])
        clamped = torch.full((n, self.C), 0.05, dtype=torch.float32, device=self.device)
        clamped = torch.clamp(clamped, min=1e-6)
        feat = torch.log(torch.expm1(clamped))
        feat = feat + 0.01 * torch.randn_like(feat)
        return feat

    def add_sample_positions(self, render_image: torch.Tensor, iteration: int) -> int:
        errors = torch.abs(render_image - self.gt_image).sum(dim=1)
        normalized = errors / torch.clamp(errors.sum(), min=1e-8)

        base_num_samples = 1000
        if iteration == self.args.iterations - self.args.grow_iter:
            dynamic_num_samples = max(0, self.max_num_points - self.model.cur_num_points)
        else:
            dynamic_num_samples = max(0, min(base_num_samples,
                                             self.max_num_points - self.model.cur_num_points))

        if dynamic_num_samples == 0:
            return 0

        flat = normalized.view(-1)
        dynamic_num_samples = min(dynamic_num_samples, flat.numel())
        _, sampled_indices = torch.topk(flat, dynamic_num_samples)
        sampled_y = sampled_indices // self.W
        sampled_x = sampled_indices % self.W
        sampled_xy = torch.stack([sampled_x, sampled_y], dim=1)

        new_features = self._initialize_new_features(sampled_xy)
        new_xyz = self.model.prepare_new_xyz(sampled_xy.float())
        new_cov2d = self.model.initialize_new_covariance(dynamic_num_samples)
        now_points, non_definite = self.model.densification_postfix(
            new_xyz=new_xyz,
            new_features_dc=new_features,
            new_cov2d=new_cov2d,
        )
        self.logwriter.write(
            f"iter:{iteration} add {dynamic_num_samples} points, filtered {non_definite}, current {now_points}"
        )
        return dynamic_num_samples

    def _snapshot_model_state(self):
        state_dict = copy.deepcopy(self.model.state_dict())
        if getattr(self.model, "SLV", False) and "cholesky_bound" not in state_dict:
            state_dict["cholesky_bound"] = self.model.cholesky_bound.detach().clone()
        return state_dict

    def _restore_model_state(self, state_dict):
        if "_xyz" in state_dict:
            self.model._xyz = nn.Parameter(state_dict["_xyz"].detach().clone())
        if "_features_dc" in state_dict:
            self.model._features_dc = nn.Parameter(state_dict["_features_dc"].detach().clone())
        if "_cov2d" in state_dict:
            self.model._cov2d = nn.Parameter(state_dict["_cov2d"].detach().clone())
        if "_cholesky" in state_dict:
            self.model._cholesky = nn.Parameter(state_dict["_cholesky"].detach().clone())
        if "_opacity" in state_dict:
            self.model._opacity = nn.Parameter(state_dict["_opacity"].detach().clone(), requires_grad=False)
        if "gabor_freqs" in state_dict:
            self.model.gabor_freqs = nn.Parameter(state_dict["gabor_freqs"].detach().clone())
        if "gabor_weights" in state_dict:
            self.model.gabor_weights = nn.Parameter(state_dict["gabor_weights"].detach().clone())
        self.model.load_state_dict(state_dict, strict=False)
        self.model.cur_num_points = state_dict["_xyz"].shape[0]
        if getattr(self.model, "SLV", False):
            if "cholesky_bound" in state_dict:
                self.model.cholesky_bound = state_dict["cholesky_bound"].detach().clone().to(self.model.device)
            else:
                n = self.model.cur_num_points
                low_pass = min(self.model.H * self.model.W / (9 * math.pi * max(n, 1)), 300.0)
                self.model.cholesky_bound = torch.tensor(
                    [low_pass, 0.0, low_pass], device=self.model.device, dtype=torch.float32
                ).view(1, 3).repeat(n, 1)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(H=self.H, W=self.W)
            reconstruction = outputs["render"]

        mse = F.mse_loss(reconstruction, self.gt_image).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        ssim_value = self._compute_bandwise_ssim(reconstruction)
        pred_hwc = reconstruction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        sam = compute_sam(self.gt_cube, pred_hwc)
        return {
            "psnr": psnr,
            "ssim": ssim_value,
            "sam": sam,
            "mse": mse,
            "reconstruction": pred_hwc,
        }

    def save_results(self, best_state_dict, metrics):
        checkpoint_path = self.log_dir / "gaussian_model_hsi.pth.tar"
        torch.save(
            {
                "gs": best_state_dict,
                "num_gs": int(best_state_dict["_xyz"].shape[0]),
                "spectral_channels": self.C,
                "metrics": {
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "sam": metrics["sam"],
                    "mse": metrics["mse"],
                },
            },
            checkpoint_path,
        )

        np.save(self.log_dir / "reconstruction.npy", metrics["reconstruction"].astype(np.float32))

        with open(self.log_dir / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "sam": metrics["sam"],
                    "mse": metrics["mse"],
                },
                fp,
                indent=2,
            )

    def train(self):
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="HSI fitting")
        best_psnr = -float("inf")
        best_iter = 0
        best_model_dict = self._snapshot_model_state()
        first_prune_logged = False
        first_grow_logged = False
        last_pruned = 0
        last_added = 0
        last_filtered = 0

        self.model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        for iteration in range(1, self.args.iterations + 1):
            loss, psnr, out_image, recon_loss = self.model.train_iter(
                self.H,
                self.W,
                self.gt_image,
                isprint=self.args.print,
            )

            with torch.no_grad():
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_iter = iteration
                    best_model_dict = self._snapshot_model_state()

            self.model.optimizer_step()

            with torch.no_grad():
                if self.args.prune and getattr(self.model, "supports_pruning", True) and iteration % self.args.prune_iter == 0:
                    before_prune = self.model.cur_num_points
                    to_prune_nums, after_prune = self.model.non_semi_definite_prune(self.H, self.W)
                    last_pruned = before_prune - after_prune
                    if not first_prune_logged and last_pruned > 0:
                        self.logwriter.write(
                            f"iter:{iteration} first prune: before={before_prune} after={after_prune} pruned={last_pruned}"
                        )
                        first_prune_logged = True

                if self.args.adaptive_add and iteration % self.args.grow_iter == 0 and iteration < self.args.iterations:
                    before_add = self.model.cur_num_points
                    last_added = self.add_sample_positions(out_image, iteration)
                    after_add = self.model.cur_num_points
                    last_filtered = max(0, before_add + last_added - after_add)
                    if not first_grow_logged:
                        self.logwriter.write(
                            f"iter:{iteration} first grow: before={before_add} added={last_added} "
                            f"filtered={last_filtered} after={after_add}"
                        )
                        first_grow_logged = True

                if iteration % 100 == 0:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.6f}",
                            "recon": f"{recon_loss.item():.6f}",
                            "psnr": f"{psnr:.4f}",
                            "best": f"{best_psnr:.4f}",
                            "num": f"{self.model.cur_num_points}",
                            "pruned": f"{last_pruned}",
                            "added": f"{last_added}",
                        }
                    )
                    progress_bar.update(100)

        torch.cuda.synchronize()
        train_time = time.time() - start_time
        progress_bar.close()

        self._restore_model_state(best_model_dict)
        metrics = self.evaluate()
        self.save_results(best_model_dict, metrics)
        self.logwriter.write(
            f"training complete in {train_time:.2f}s, best_iter={best_iter}, "
            f"psnr={metrics['psnr']:.4f}, ssim={metrics['ssim']:.4f}, sam={metrics['sam']:.4f}"
        )
        return metrics, train_time, best_iter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HSI full fitting with density-controlled Gabor++ Gaussian splatting")
    parser.add_argument('--dataset', type=str, default='all',
                        help="Dataset name or 'all': Urban | Salinas | JasperRidge | PaviaU | all")
    parser.add_argument("--output_root", type=str, default="./checkpoints_his")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--prune_iter", type=int, default=100)
    parser.add_argument("--grow_iter", type=int, default=5000)
    parser.add_argument("--num_points", type=int, default=2000)
    parser.add_argument("--max_num_points", type=int, default=5000)
    parser.add_argument("--opt_type", type=str, default="adam")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--no_prune", action="store_true", help="Disable covariance pruning; ignored for cholesky")
    parser.add_argument("--no_adaptive_add", action="store_true", help="Disable densification by error sampling")
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--num_gabor", type=int, default=2)
    parser.add_argument("--covariance_type", type=str, default="covariance",
                        choices=["covariance", "cholesky"],
                        help="Covariance parameterization: 'covariance' (direct) or 'cholesky' (L@L^T)")
    parser.add_argument("--color_norm", action="store_true")
    parser.add_argument("--coords_norm", action="store_true")
    parser.add_argument("--coords_act", type=str, default="tanh")
    parser.add_argument("--clip_coe", type=float, default=3.0)
    parser.add_argument("--radius_clip", type=float, default=1.0)
    parser.add_argument("--cov_quant", type=str, default="lsq")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12)
    parser.add_argument("--cov_bit", type=int, default=10)
    parser.add_argument("--color_bit", type=int, default=6)
    parser.add_argument("--SLV_init", action="store_true", default=True,
                        help="Enable per-point dynamic Cholesky low-pass bound (recommended).")
    parser.add_argument("--no_SLV_init", dest="SLV_init", action="store_false",
                        help="Disable SLV dynamic bound (fall back to fixed [0.5,0,0.5]).")
    args = parser.parse_args(argv)
    args.prune = not args.no_prune
    args.adaptive_add = not args.no_adaptive_add
    return args


def _build_experiment_root(args) -> Path:
    cov_tag = args.covariance_type
    prune_tag = "pruneOff" if not args.prune else f"prune{args.prune_iter}"
    add_tag = "addOff" if not args.adaptive_add else f"add{args.grow_iter}_max{args.max_num_points}"
    run_tag = f"g{args.num_gabor}_pts{args.num_points}_{prune_tag}_{add_tag}_{cov_tag}"
    root = Path(args.output_root) / run_tag
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_dataset_list(args):
    available = list_available_datasets()
    if not available:
        raise RuntimeError("No dataset found under HSI/data required by load_dataset")

    if args.dataset.lower() == "all":
        return available

    selected = args.dataset.lower()
    if selected not in available:
        raise ValueError(f"Dataset '{args.dataset}' not found. Available: {available}")
    return [selected]


def _write_train_summary(experiment_root: Path, rows):
    train_txt = experiment_root / "train.txt"
    with open(train_txt, "w", encoding="utf-8") as fp:
        fp.write("dataset\tpsnr\tssim\tsam\tmse\ttime_sec\tbest_iter\n")
        for row in rows:
            fp.write(
                f"{row['dataset']}\t{row['psnr']:.6f}\t{row['ssim']:.6f}\t{row['sam']:.6f}\t{row['mse']:.8f}\t"
                f"{row['time']:.2f}\t{row['best_iter']}\n"
            )

        avg_psnr = float(np.mean([row["psnr"] for row in rows]))
        avg_ssim = float(np.mean([row["ssim"] for row in rows]))
        avg_sam = float(np.mean([row["sam"] for row in rows]))
        avg_mse = float(np.mean([row["mse"] for row in rows]))
        avg_time = float(np.mean([row["time"] for row in rows]))
        fp.write("\n")
        fp.write(
            f"AVG\t{avg_psnr:.6f}\t{avg_ssim:.6f}\t{avg_sam:.6f}\t{avg_mse:.8f}\t{avg_time:.2f}\t-\n"
        )

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "sam": avg_sam,
        "mse": avg_mse,
        "time": avg_time,
    }


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    experiment_root = _build_experiment_root(args)
    dataset_list = _resolve_dataset_list(args)
    summary_rows = []

    for dataset_name in dataset_list:
        trainer = HSIFullTrainer(args, dataset_name=dataset_name, experiment_root=experiment_root)
        config_path = trainer.log_dir / "config.yaml"
        config_payload = dict(vars(args))
        config_payload["dataset"] = dataset_name
        with open(config_path, "w", encoding="utf-8") as fp:
            fp.write(yaml.safe_dump(config_payload, default_flow_style=False))

        metrics, train_time, best_iter = trainer.train()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "sam": metrics["sam"],
                "mse": metrics["mse"],
                "time": train_time,
                "best_iter": best_iter,
            }
        )
        print(
            f"[{dataset_name}] best_iter={best_iter}, psnr={metrics['psnr']:.4f}, "
            f"ssim={metrics['ssim']:.4f}, sam={metrics['sam']:.4f}, time={train_time:.2f}s"
        )

    averages = _write_train_summary(experiment_root, summary_rows)
    print(
        f"All datasets finished: avg_psnr={averages['psnr']:.4f}, "
        f"avg_ssim={averages['ssim']:.4f}, avg_sam={averages['sam']:.4f}, avg_mse={averages['mse']:.8f}, "
        f"avg_time={averages['time']:.2f}s"
    )


if __name__ == "__main__":
    main(sys.argv[1:])