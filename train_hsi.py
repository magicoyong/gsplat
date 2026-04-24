import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from endmember import nmf_initialization
from gaussianimage_cholesky_unknown import GaussianImage_Cholesky_EA
from hsi_utils import compute_sam, load_dataset
from utils import LogWriter


class HSIFitter:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("HSI fitting requires CUDA-enabled PyTorch and the compiled gsplat CUDA extension.")

        self.args = args
        self.device = torch.device("cuda:0")
        self.dataset_name = args.dataset.lower()

        cube_hwc = load_dataset(self.dataset_name)
        self.gt_cube = cube_hwc
        self.H, self.W, self.C = cube_hwc.shape

        gt = torch.as_tensor(cube_hwc, dtype=torch.float32, device=self.device)
        self.gt_image = torch.clamp(gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous(), 0, 1)

        if args.endmember_init:
            self.nmf_time = 0.0
            endmember_init = np.load(args.endmember_init).astype(np.float32)
        else:
            nmf_start_time = time.time()
            endmember_init, _ = nmf_initialization(self.gt_image, args.rank, dataset_name=args.dataset)
            self.nmf_time = time.time() - nmf_start_time

        self.rank = int(endmember_init.shape[0])
        self.log_dir = self._build_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir)

        self.model = GaussianImage_Cholesky_EA(
            loss_type=args.loss_type,
            opt_type=args.opt_type,
            num_points=args.num_points,
            GT=self.gt_image,
            E=endmember_init,
            H=self.H,
            W=self.W,
            C=self.C,
            rank=self.rank,
            BLOCK_H=16,
            BLOCK_W=16,
            device=self.device,
            lr=args.lr,
            quantize=args.quantize,
            num_gabor=args.num_gabor,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_endmember=args.freeze_endmember,
        ).to(self.device)

    def _build_log_dir(self):
        endmember_tag = "freezeE" if self.args.freeze_endmember else f"lora{self.args.lora_rank}_a{self.args.lora_alpha}"
        quant_tag = "quant" if self.args.quantize else "fp"
        return (
            Path(self.args.output_root)
            / self.dataset_name
            / f"rank{self.args.rank}_{endmember_tag}_g{self.args.num_gabor}_{quant_tag}_pts{self.args.num_points}"
        )

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward()
            reconstruction = outputs["render"]
            abundance = outputs["abundance"]
            endmember = outputs["endmember"]

        mse = F.mse_loss(reconstruction, self.gt_image).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        pred_hwc = reconstruction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        sam = compute_sam(self.gt_cube, pred_hwc)
        return {
            "psnr": psnr,
            "sam": sam,
            "mse": mse,
            "reconstruction": pred_hwc,
            "abundance": abundance.detach().cpu().numpy(),
            "endmember": endmember.detach().cpu().numpy(),
            "delta_norm": self.model.get_endmember_delta_norm(),
        }

    def save_results(self, best_state_dict, metrics, best_iter, train_time):
        checkpoint_path = self.log_dir / "gaussian_model_hsi.pth.tar"
        torch.save(
            {
                "gs": best_state_dict,
                "num_gs": int(best_state_dict["_xyz"].shape[0]),
                "rank": self.rank,
                "spectral_channels": self.C,
                "metrics": {
                    "psnr": metrics["psnr"],
                    "sam": metrics["sam"],
                    "mse": metrics["mse"],
                    "delta_norm": metrics["delta_norm"],
                    "best_iter": best_iter,
                    "train_time": train_time,
                    "nmf_time": self.nmf_time,
                },
            },
            checkpoint_path,
        )

        np.save(self.log_dir / "reconstruction.npy", metrics["reconstruction"].astype(np.float32))
        np.save(self.log_dir / "abundance.npy", metrics["abundance"].astype(np.float32))
        np.save(self.log_dir / "endmember.npy", metrics["endmember"].astype(np.float32))
        np.save(self.log_dir / "endmember_init.npy", self.model.endmember.detach().cpu().numpy().astype(np.float32))

        with open(self.log_dir / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "psnr": metrics["psnr"],
                    "sam": metrics["sam"],
                    "mse": metrics["mse"],
                    "delta_norm": metrics["delta_norm"],
                    "best_iter": best_iter,
                    "train_time": train_time,
                    "nmf_time": self.nmf_time,
                },
                fp,
                indent=2,
            )

    def train(self):
        progress_bar = tqdm(range(1, self.args.iterations + 1), desc="HSI fitting")
        self.model.train()
        best_psnr = -float("inf")
        best_iter = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())

        torch.cuda.synchronize()
        start_time = time.time()
        for iteration in range(1, self.args.iterations + 1):
            if self.args.quantize:
                loss, psnr = self.model.train_iter_quantize(endmember_weight=self.args.endmember_weight)
                delta_norm = self.model.get_endmember_delta_norm()
            else:
                loss, psnr, delta_norm = self.model.train_iter(endmember_weight=self.args.endmember_weight)

            if psnr > best_psnr:
                best_psnr = psnr
                best_iter = iteration
                best_state_dict = copy.deepcopy(self.model.state_dict())

            if iteration % 50 == 0 or iteration == self.args.iterations:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.6f}",
                        "psnr": f"{psnr:.4f}",
                        "best": f"{best_psnr:.4f}",
                        "deltaE": f"{delta_norm:.4f}",
                    }
                )
                progress_bar.update(50 if iteration % 50 == 0 else iteration % 50)

        torch.cuda.synchronize()
        optimization_time = time.time() - start_time
        train_time = self.nmf_time + optimization_time
        progress_bar.close()

        self.model.load_state_dict(best_state_dict)
        metrics = self.evaluate()
        self.save_results(best_state_dict, metrics, best_iter, train_time)
        self.logwriter.write(
            f"training complete in {train_time:.2f}s (nmf={self.nmf_time:.2f}s, optimize={optimization_time:.2f}s), best_iter={best_iter}, psnr={metrics['psnr']:.4f}, sam={metrics['sam']:.4f}, deltaE={metrics['delta_norm']:.4f}"
        )
        return metrics, train_time, best_iter


def parse_args():
    parser = argparse.ArgumentParser(description="HSI fitting with cholesky Gaussian splatting, Gabor rendering, and LoRA endmember tuning")
    parser.add_argument("--dataset", type=str, default="jasperridge", help="Dataset name: Urban | Salinas | JasperRidge | PaviaU")
    parser.add_argument("--endmember_init", type=str, default=None, help="Optional .npy file for initial endmember matrix E")
    parser.add_argument("--output_root", type=str, default="./checkpoints_hsi")
    parser.add_argument("--rank", type=int, default=10, help="NMF rank / abundance channels")
    parser.add_argument("--lora_rank", type=int, default=2, help="LoRA rank for endmember correction")
    parser.add_argument("--lora_alpha", type=float, default=0.1, help="Maximum endmember correction scale")
    parser.add_argument("--freeze_endmember", action="store_true", help="Disable LoRA and keep E fixed")
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--num_points", type=int, default=600)
    parser.add_argument("--num_gabor", type=int, default=2)
    parser.add_argument("--opt_type", type=str, default="adam")
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--endmember_weight", type=float, default=0.0, help="Optional regularization toward the initial endmember")
    parser.add_argument("--quantize", action="store_true", help="Train through the quantized abundance path")
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    fitter = HSIFitter(args)
    metrics, train_time, best_iter = fitter.train()
    print(
        f"HSI fitting finished: best_iter={best_iter}, psnr={metrics['psnr']:.4f}, sam={metrics['sam']:.4f}, deltaE={metrics['delta_norm']:.4f}, time={train_time:.2f}s"
    )


if __name__ == "__main__":
    main()