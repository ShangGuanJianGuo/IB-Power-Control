"""Corrected beta-sweep robustness evaluation for old federated checkpoints.

This script is intentionally separate from test_noise_federated.py.  The old
checkpoints store their clean validation SE in the checkpoint metadata, while
the current model code can no longer reproduce those clean values exactly.  To
avoid silently mixing inconsistent code/data states, this evaluator reports:

1) raw_current_se: SE recomputed with the current code;
2) anchored_se: checkpoint clean SE multiplied by the measured degradation
   ratio raw_noisy_se / raw_clean_se.

Anchored values reproduce a historical-clean-normalized curve; they are not
direct forward SE and do not resolve the code/checkpoint or physical-power
issues documented in KNOWN_ISSUES.md. Both raw and anchored columns are retained.
"""

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model_architecture_v2 import Iterative_IB_System


@dataclass
class SplitBundle:
    h: torch.Tensor
    w: torch.Tensor
    beta: torch.Tensor
    mask: torch.Tensor


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def beta_key_from_path(path: str) -> float:
    match = re.search(r"beta_([0-9.]+)\.pth$", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse beta from checkpoint name: {path}")
    return float(match.group(1))


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

    beta_linear = np.mean(np.abs(h_all) ** 2, axis=-1)
    beta_db = 10.0 * np.log10(beta_linear + 1e-15)
    beta_norm = (beta_db - beta_db.mean()) / (beta_db.std() + 1e-8)

    mask = np.zeros_like(beta_linear, dtype=np.float32)
    for sample_idx in range(beta_linear.shape[0]):
        top_idx = np.argsort(beta_linear[sample_idx], axis=0)[-args.n_neighbors :, :]
        for user_idx in range(args.K):
            mask[sample_idx, top_idx[:, user_idx], user_idx] = 1.0

    torch.manual_seed(args.split_seed)
    indices = torch.randperm(n)
    train_size = int(args.train_ratio * n)
    val_indices = indices[train_size:]
    if args.max_eval_samples is not None:
        val_indices = val_indices[: args.max_eval_samples]

    return SplitBundle(
        h=torch.from_numpy(h_all[val_indices]).cfloat(),
        w=torch.from_numpy(w_all[val_indices]).cfloat(),
        beta=torch.from_numpy(beta_norm[val_indices]).float(),
        mask=torch.from_numpy(mask[val_indices]).float(),
    )


def compute_mr_precoding(h: torch.Tensor, p_max: float, num_users: int) -> torch.Tensor:
    h_norm = torch.sqrt(torch.sum(torch.abs(h) ** 2, dim=-1, keepdim=True)).clamp(min=1e-12)
    return (p_max / num_users) ** 0.5 * h / h_norm


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
    noise = torch.complex(real_noise * noise_std.real, imag_noise * noise_std.real)
    return h + noise


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


def iter_batches(bundle: SplitBundle, batch_size: int, device: torch.device):
    num_samples = bundle.h.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield (
            start // batch_size,
            bundle.h[start:end].to(device),
            bundle.w[start:end].to(device),
            bundle.beta[start:end].to(device),
            bundle.mask[start:end].to(device),
        )


def load_model(path: str, args: argparse.Namespace, device: torch.device) -> Tuple[Iterative_IB_System, Dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    model = Iterative_IB_System(
        int(model_args.get("L", args.L)),
        int(model_args.get("K", args.K)),
        int(model_args.get("M", args.M)),
        int(model_args.get("z_dim", args.z_dim)),
        float(model_args.get("Pmax", args.Pmax)),
        int(model_args.get("num_iterations", args.num_iterations)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if args.deterministic_latent:
        model.encoder.reparameterize = lambda mu, logvar: mu
    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: Iterative_IB_System,
    bundle: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
    snr_db: float,
    trial_index: int,
) -> float:
    total_sum_se = 0.0
    total_samples = 0

    for batch_index, h_clean, w_clean, beta, mask in iter_batches(bundle, args.batch_size, device):
        noise_seed = args.noise_seed + int((snr_db + 100) * 10000) + trial_index * 1000 + batch_index
        h_input = add_noise_snr(h_clean, snr_db, noise_seed, args.noise_mode)
        if args.recompute_precoder and snr_db < 100:
            w_input = compute_mr_precoding(h_input, args.Pmax, args.K)
        else:
            w_input = w_clean

        power, _ = model(h_input, w_input, beta, mask)
        rates = compute_se_per_user(power, h_clean, w_input)
        total_sum_se += rates.sum(dim=1).sum().item()
        total_samples += h_clean.shape[0]

    return total_sum_se / max(total_samples, 1)


def discover_checkpoints(args: argparse.Namespace) -> List[Tuple[float, str]]:
    selected = set(parse_float_list(args.betas)) if args.betas else None
    paths = glob.glob(os.path.join(args.checkpoint_dir, "federated_cluster_beta_*.pth"))
    beta_paths = []
    for path in paths:
        beta = beta_key_from_path(path)
        if selected is None or beta in selected:
            beta_paths.append((beta, path))
    beta_paths.sort(key=lambda item: item[0])
    if not beta_paths:
        raise FileNotFoundError("No matching federated_cluster_beta_*.pth checkpoints found.")
    return beta_paths


def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict[str, float]], args: argparse.Namespace, output_path: str) -> None:
    betas = sorted({row["beta"] for row in rows})
    snrs = sorted({row["snr_db"] for row in rows}, reverse=False)
    plt.figure(figsize=(7.2, 5.2))
    marker_cycle = ["o", "s", "^", "D", "v", ">", "<", "p"]
    color_cycle = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    for idx, beta in enumerate(betas):
        beta_rows = {row["snr_db"]: row for row in rows if row["beta"] == beta}
        value_key = "anchored_se" if args.plot_value == "anchored" else "raw_noisy_se_mean"
        y_values = [beta_rows[snr][value_key] for snr in snrs]
        plt.plot(
            snrs,
            y_values,
            marker=marker_cycle[idx % len(marker_cycle)],
            linewidth=2.2,
            markersize=6,
            color=color_cycle[idx % len(color_cycle)],
            label=rf"$\beta_{{IB}}={beta:g}$",
        )

    plt.xlabel("Channel Estimation SNR (dB)")
    plt.ylabel("Historical-clean-anchored sum SE (bps/Hz)" if args.plot_value == "anchored" else "Direct sum SE (bps/Hz)")
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    plt.legend(fontsize=9)
    if args.invert_x_axis:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected robustness beta sweep for federated IB checkpoints.")
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--betas", type=str, default="0,1,4,16,32,64")
    parser.add_argument("--snrs", type=str, default="20,15,10,5,0,-5")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--plot_value", choices=["raw", "anchored"], default="anchored", help="Plot a direct SE or explicitly historical-clean-anchored curve; both remain in the CSV")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--noise_seed", type=int, default=20260603)
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="per_link",
        choices=["per_link", "global_sample", "per_coefficient"],
        help=(
            "per_link sets the error variance from each AP-UE link average power; "
            "global_sample reproduces the older script's whole-sample average."
        ),
    )
    parser.add_argument("--deterministic_latent", action="store_true", default=True)
    parser.add_argument("--stochastic_latent", dest="deterministic_latent", action="store_false")
    parser.add_argument("--recompute_precoder", action="store_true", default=True)
    parser.add_argument("--keep_clean_precoder", dest="recompute_precoder", action="store_false")
    parser.add_argument("--cap_degradation_at_one", action="store_true", default=True)
    parser.add_argument("--allow_noisy_gain", dest="cap_degradation_at_one", action="store_false")
    parser.add_argument("--output_prefix", type=str, default="robustness_beta_corrected")
    parser.add_argument("--invert_x_axis", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Plot value: {args.plot_value}; anchored values are historical normalizations, not direct SE")
    print(f"Deterministic latent: {args.deterministic_latent}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Recompute MR precoder from noisy H: {args.recompute_precoder}")
    print(f"Cap degradation ratio at one: {args.cap_degradation_at_one}")

    bundle = load_validation_split(args)
    print(f"Validation samples: {bundle.h.shape[0]}")

    snrs = parse_float_list(args.snrs)
    checkpoints = discover_checkpoints(args)
    rows: List[Dict[str, float]] = []
    clean_rows: List[Dict[str, float]] = []

    for beta, path in checkpoints:
        print(f"\nEvaluating beta={beta:g}: {os.path.basename(path)}")
        model, checkpoint = load_model(path, args, device)
        checkpoint_clean = float(checkpoint.get("val_se", np.nan))
        raw_clean = evaluate_model(model, bundle, args, device, snr_db=100.0, trial_index=0)
        scale = checkpoint_clean / raw_clean if raw_clean > 0 and np.isfinite(checkpoint_clean) else 1.0
        mismatch = abs(raw_clean - checkpoint_clean) / max(abs(checkpoint_clean), 1e-9)
        clean_rows.append(
            {
                "beta": beta,
                "checkpoint_clean_se": checkpoint_clean,
                "raw_current_clean_se": raw_clean,
                "calibration_scale": scale,
                "relative_mismatch": mismatch,
            }
        )
        print(
            f"  clean checkpoint={checkpoint_clean:.4f}, raw_current={raw_clean:.4f}, "
            f"scale={scale:.4f}, mismatch={100*mismatch:.2f}%"
        )

        for snr in snrs:
            trial_values = [
                evaluate_model(model, bundle, args, device, snr_db=snr, trial_index=trial)
                for trial in range(args.trials)
            ]
            raw_noisy_mean = float(np.mean(trial_values))
            raw_noisy_std = float(np.std(trial_values))
            raw_ratio = raw_noisy_mean / raw_clean if raw_clean > 0 else 0.0
            degradation_ratio = min(raw_ratio, 1.0) if args.cap_degradation_at_one else raw_ratio
            anchored_se = checkpoint_clean * degradation_ratio if np.isfinite(checkpoint_clean) else raw_noisy_mean
            rows.append(
                {
                    "beta": beta,
                    "snr_db": snr,
                    "checkpoint_clean_se": checkpoint_clean,
                    "raw_current_clean_se": raw_clean,
                    "raw_noisy_se_mean": raw_noisy_mean,
                    "raw_noisy_se_std": raw_noisy_std,
                    "raw_degradation_ratio": raw_ratio,
                    "used_degradation_ratio": degradation_ratio,
                    "anchored_se": anchored_se,
                }
            )
            print(
                f"  SNR={snr:>5g} dB | raw={raw_noisy_mean:.4f}±{raw_noisy_std:.4f} | "
                f"ratio={raw_ratio:.4f} | anchored={anchored_se:.4f}"
            )

    clean_csv = f"{args.output_prefix}_clean_audit.csv"
    result_csv = f"{args.output_prefix}_results.csv"
    figure_path = f"{args.output_prefix}.png"
    write_csv(clean_rows, clean_csv)
    write_csv(rows, result_csv)
    plot_results(rows, args, figure_path)
    print(f"\nSaved clean audit: {clean_csv}")
    print(f"Saved robustness results: {result_csv}")
    print(f"Saved plot: {figure_path}")


if __name__ == "__main__":
    main()
