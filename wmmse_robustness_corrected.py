"""Corrected WMMSE robustness evaluation under the same protocol as IB curves.

The corrected IB robustness curves use:
  * the validation split from the first train_samples channel realizations;
  * per-link channel-estimation noise;
  * MR precoding recomputed from the noisy CSI;
  * clean-channel SE evaluation with the actually used noisy-CSI precoder;
  * repeated noise trials with deterministic seeds.

This script applies the same protocol to centralized WMMSE power control.  It is
separate from test_wmmse_noise.py so the older script and figures remain intact.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class SplitBundle:
    h: torch.Tensor
    w: torch.Tensor


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_validation_split(args: argparse.Namespace) -> SplitBundle:
    suffix = f"L{args.L}_K{args.K}_M{args.M}_number{args.samples}"
    h_path = os.path.join(args.data_dir, f"setHCF_{suffix}.npy")
    w_path = os.path.join(args.data_dir, f"setWCF_{suffix}.npy")
    if not os.path.exists(h_path):
        raise FileNotFoundError(f"Missing channel file: {h_path}")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing precoder file: {w_path}")

    h_all = np.load(h_path)
    w_all = np.load(w_path)
    if h_all.ndim == 5:
        h_all = h_all.squeeze(1)
    if w_all.ndim == 5:
        w_all = w_all.squeeze(1)

    n = min(args.train_samples, h_all.shape[0])
    h_all = h_all[:n]
    w_all = w_all[:n]

    torch.manual_seed(args.split_seed)
    indices = torch.randperm(n)
    train_size = int(args.train_ratio * n)
    val_indices = indices[train_size:]
    if args.max_eval_samples is not None:
        val_indices = val_indices[: args.max_eval_samples]

    return SplitBundle(
        h=torch.from_numpy(h_all[val_indices]).cfloat(),
        w=torch.from_numpy(w_all[val_indices]).cfloat(),
    )


def add_noise_snr(h: torch.Tensor, snr_db: float, seed: int, noise_mode: str) -> torch.Tensor:
    if snr_db >= 100:
        return h.clone()

    generator = torch.Generator(device=h.device)
    generator.manual_seed(int(seed))
    snr_linear = 10.0 ** (snr_db / 10.0)

    if noise_mode == "global_sample":
        signal_power = torch.mean(torch.abs(h) ** 2, dim=(-3, -2, -1), keepdim=True)
    elif noise_mode == "per_link":
        signal_power = torch.mean(torch.abs(h) ** 2, dim=-1, keepdim=True)
    elif noise_mode == "per_coefficient":
        signal_power = torch.abs(h) ** 2
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    noise_std = torch.sqrt(signal_power / snr_linear / 2.0)
    real_noise = torch.randn(h.real.shape, dtype=h.real.dtype, device=h.device, generator=generator)
    imag_noise = torch.randn(h.imag.shape, dtype=h.imag.dtype, device=h.device, generator=generator)
    return h + torch.complex(real_noise * noise_std.real, imag_noise * noise_std.real)


def compute_mr_precoding(h: torch.Tensor, p_max: float, num_users: int) -> torch.Tensor:
    h_norm = torch.sqrt(torch.sum(torch.abs(h) ** 2, dim=-1, keepdim=True)).clamp(min=1e-12)
    return (p_max / num_users) ** 0.5 * h / h_norm


def compute_se_per_user(p: torch.Tensor, h: torch.Tensor, w: torch.Tensor, noise_power: float = 1.0) -> torch.Tensor:
    sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
    w_scaled = sqrt_p * w
    h_u = h.permute(0, 2, 1, 3)
    w_u = w_scaled.permute(0, 2, 1, 3)
    rx_matrix = torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)
    rx_power = torch.abs(rx_matrix) ** 2
    signal = torch.diagonal(rx_power, dim1=1, dim2=2)
    total = torch.sum(rx_power, dim=2)
    interference = total - signal
    sinr = signal / (interference + noise_power + 1e-8)
    return torch.log2(1.0 + sinr)


def wmmse_power_control(
    h: torch.Tensor,
    w: torch.Tensor,
    p_init: torch.Tensor,
    p_max: float,
    iterations: int,
    noise_power: float = 1.0,
) -> torch.Tensor:
    p = p_init.clone()
    hw_gain = torch.einsum("blkm,bljm->blkj", h.conj(), w)

    for _ in range(iterations):
        sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(2)
        effective_gain = torch.sum(sqrt_p * hw_gain, dim=1)
        total_rx_power = torch.sum(torch.abs(effective_gain) ** 2, dim=2) + noise_power
        diagonal_gain = torch.diagonal(effective_gain, dim1=1, dim2=2)
        receiver = diagonal_gain.conj() / total_rx_power

        mmse_error = 1.0 - receiver * diagonal_gain
        mmse_weight = 1.0 / torch.clamp(mmse_error.real, min=1e-6)

        hw_diag = torch.diagonal(hw_gain, dim1=2, dim2=3)
        utility = mmse_weight.unsqueeze(1) * torch.abs(receiver.unsqueeze(1) * hw_diag) ** 2
        utility_sum = torch.sum(utility, dim=2, keepdim=True).clamp(min=1e-9)
        p = p_max * utility / utility_sum

    return p


def iter_batches(bundle: SplitBundle, batch_size: int, device: torch.device):
    num_samples = bundle.h.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield start // batch_size, bundle.h[start:end].to(device), bundle.w[start:end].to(device)


@torch.no_grad()
def evaluate_wmmse(
    bundle: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
    snr_db: float,
    trial_index: int,
) -> float:
    total_sum_se = 0.0
    total_samples = 0

    for batch_index, h_clean, w_clean in iter_batches(bundle, args.batch_size, device):
        noise_seed = args.noise_seed + int((snr_db + 100) * 10000) + trial_index * 1000 + batch_index
        h_input = add_noise_snr(h_clean, snr_db, noise_seed, args.noise_mode)

        if args.recompute_precoder and snr_db < 100:
            w_input = compute_mr_precoding(h_input, args.Pmax, args.K)
        else:
            w_input = w_clean

        p_init = torch.ones(h_clean.shape[0], args.L, args.K, device=device) * (args.Pmax / args.K)
        p_opt = wmmse_power_control(
            h_input,
            w_input,
            p_init,
            args.Pmax,
            iterations=args.iterations,
            noise_power=args.noise_power,
        )
        rates = compute_se_per_user(p_opt, h_clean, w_input, noise_power=args.noise_power)
        total_sum_se += rates.sum(dim=1).sum().item()
        total_samples += h_clean.shape[0]

    return total_sum_se / max(total_samples, 1)


def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict[str, float]], output_path: str, args: argparse.Namespace) -> None:
    snrs = [row["snr_db"] for row in rows if row["snr_db"] < 100]
    values = [row["sum_se_mean"] for row in rows if row["snr_db"] < 100]
    plt.figure(figsize=(7.2, 5.2))
    plt.plot(snrs, values, "k--p", linewidth=2.5, markersize=8, markerfacecolor="k", label="WMMSE")
    plt.xlabel("Channel Estimation SNR (dB)")
    plt.ylabel("Sum SE (bps/Hz)")
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    plt.legend()
    if args.invert_x_axis:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected WMMSE robustness under per-link CSI errors.")
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--snrs", type=str, default="100,20,15,10,5,0,-5")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--noise_seed", type=int, default=20260603)
    parser.add_argument("--noise_power", type=float, default=1.0)
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="per_link",
        choices=["per_link", "global_sample", "per_coefficient"],
    )
    parser.add_argument("--recompute_precoder", action="store_true", default=True)
    parser.add_argument("--keep_clean_precoder", dest="recompute_precoder", action="store_false")
    parser.add_argument("--output_prefix", type=str, default="wmmse_robustness_corrected")
    parser.add_argument("--invert_x_axis", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Validation protocol: train_samples={args.train_samples}, train_ratio={args.train_ratio}, seed={args.split_seed}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Recompute MR precoder from noisy H: {args.recompute_precoder}")
    print(f"WMMSE iterations: {args.iterations}")

    bundle = load_validation_split(args)
    print(f"Validation samples: {bundle.h.shape[0]}")

    rows: List[Dict[str, float]] = []
    snrs = parse_float_list(args.snrs)
    for snr in snrs:
        trial_count = 1 if snr >= 100 else args.trials
        trial_values = []
        pbar = tqdm(range(trial_count), desc=f"WMMSE SNR={snr:g} dB")
        for trial_index in pbar:
            trial_values.append(evaluate_wmmse(bundle, args, device, snr, trial_index))

        mean_value = float(np.mean(trial_values))
        std_value = float(np.std(trial_values))
        rows.append(
            {
                "snr_db": snr,
                "sum_se_mean": mean_value,
                "sum_se_std": std_value,
                "trials": trial_count,
            }
        )
        print(f"SNR={snr:g} dB | Sum SE={mean_value:.4f} ± {std_value:.4f}")

    csv_path = f"{args.output_prefix}_results.csv"
    figure_path = f"{args.output_prefix}.png"
    write_csv(rows, csv_path)
    plot_results(rows, figure_path, args)
    print(f"\nSaved WMMSE results: {csv_path}")
    print(f"Saved WMMSE plot: {figure_path}")


if __name__ == "__main__":
    main()
