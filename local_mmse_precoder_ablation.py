"""Small local-MMSE precoder ablation for Reviewer 3.

The historical federated checkpoints store their clean MRT validation SE.  The
current source tree does not exactly reproduce that clean value, so this script
reports both the raw current-code values and an anchored local-MMSE value using
the same calibration protocol as robustness_beta_sweep_corrected.py.
"""

import argparse
import csv
from typing import Dict, List

import numpy as np
import torch

from robustness_beta_sweep_corrected import (
    compute_se_per_user,
    load_model,
    load_validation_split,
)


def compute_local_mmse_precoder(
    h: torch.Tensor,
    mask: torch.Tensor,
    p_max: float,
    num_users: int,
    regularization: float,
    served_only: bool,
    beam_norm: float,
) -> torch.Tensor:
    """Return AP-local regularized MMSE precoders with MRT-style norm."""
    if served_only:
        h_for_cov = h * mask.unsqueeze(-1)
        rhs = h_for_cov
    else:
        h_for_cov = h
        rhs = h

    eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
    cov = torch.einsum("blkm,blkn->blmn", h_for_cov, h_for_cov.conj())
    cov = cov + regularization * eye.view(1, 1, h.shape[-1], h.shape[-1])
    solved = torch.linalg.solve(cov, rhs.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    norm = torch.linalg.vector_norm(solved, dim=-1, keepdim=True).clamp(min=1e-12)
    w = beam_norm * solved / norm
    if served_only:
        w = w * mask.unsqueeze(-1)
    return w


@torch.no_grad()
def evaluate_precoder(model, bundle, args, device, precoder: str, regularization: float, served_only: bool) -> Dict[str, float]:
    total_sum_se = 0.0
    total_samples = 0
    power_totals: List[float] = []

    for start in range(0, bundle.h.shape[0], args.batch_size):
        end = min(start + args.batch_size, bundle.h.shape[0])
        h = bundle.h[start:end].to(device)
        w_mrt = bundle.w[start:end].to(device)
        beta = bundle.beta[start:end].to(device)
        mask = bundle.mask[start:end].to(device)

        if precoder == "mrt":
            w = w_mrt
            if args.unit_norm_precoder:
                w = w / torch.linalg.vector_norm(w, dim=-1, keepdim=True).clamp(min=1e-12)
        elif precoder == "local_mmse":
            beam_norm = 1.0 if args.unit_norm_precoder else (args.Pmax / args.K) ** 0.5
            w = compute_local_mmse_precoder(h, mask, args.Pmax, args.K, regularization, served_only, beam_norm)
        else:
            raise ValueError(f"Unknown precoder: {precoder}")

        power, _ = model(h, w, beta, mask)
        rates = compute_se_per_user(power, h, w)
        total_sum_se += rates.sum(dim=1).sum().item()
        total_samples += h.shape[0]
        power_totals.append(power.sum(dim=(1, 2)).mean().item())

    return {
        "raw_sum_se": total_sum_se / max(total_samples, 1),
        "mean_total_power": float(np.mean(power_totals)) if power_totals else 0.0,
    }


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRT vs AP-local MMSE precoding for beta=0 IB-FL.")
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="federated_cluster_beta_0.0.pth")
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument("--served_only", action="store_true", default=False)
    parser.add_argument("--unit_norm_precoder", action="store_true", default=False)
    parser.add_argument("--deterministic_latent", action="store_true", default=True)
    parser.add_argument("--stochastic_latent", dest="deterministic_latent", action="store_false")
    parser.add_argument("--output_csv", type=str, default="local_mmse_precoder_ablation.csv")
    args = parser.parse_args()

    if args.regularization is None:
        args.regularization = 1.0 / (args.Pmax / args.K)

    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Regularization lambda: {args.regularization:g}")
    print(f"Served-only local MMSE: {args.served_only}")

    bundle = load_validation_split(args)
    model, checkpoint = load_model(args.checkpoint, args, device)
    checkpoint_clean_mrt = float(checkpoint.get("val_se", np.nan))

    mrt = evaluate_precoder(model, bundle, args, device, "mrt", args.regularization, args.served_only)
    lmmse = evaluate_precoder(model, bundle, args, device, "local_mmse", args.regularization, args.served_only)
    scale = checkpoint_clean_mrt / mrt["raw_sum_se"] if mrt["raw_sum_se"] > 0 else 1.0

    rows = [
        {
            "precoder": "MRT",
            "raw_sum_se": mrt["raw_sum_se"],
            "anchored_sum_se": checkpoint_clean_mrt,
            "se_ratio_vs_mrt_percent": 100.0,
            "calibration_scale": scale,
            "mean_total_power": mrt["mean_total_power"],
            "regularization": args.regularization,
            "served_only": int(args.served_only),
        },
        {
            "precoder": "AP-local MMSE",
            "raw_sum_se": lmmse["raw_sum_se"],
            "anchored_sum_se": lmmse["raw_sum_se"] * scale,
            "se_ratio_vs_mrt_percent": 100.0 * lmmse["raw_sum_se"] / mrt["raw_sum_se"],
            "calibration_scale": scale,
            "mean_total_power": lmmse["mean_total_power"],
            "regularization": args.regularization,
            "served_only": int(args.served_only),
        },
    ]
    write_csv(args.output_csv, rows)

    for row in rows:
        print(
            f"{row['precoder']}: raw={row['raw_sum_se']:.4f}, "
            f"anchored={row['anchored_sum_se']:.4f}, "
            f"vs_MRT={row['se_ratio_vs_mrt_percent']:.2f}%"
        )
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
