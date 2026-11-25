import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model_architecture_v2 import Iterative_IB_System


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_se_batch(p, h, w, noise_power: float = 1.0) -> torch.Tensor:
    sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
    w_scaled = sqrt_p * w

    h_u = h.permute(0, 2, 1, 3)
    w_u = w_scaled.permute(0, 2, 1, 3)

    rx_matrix = torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)
    rx_power = torch.abs(rx_matrix) ** 2

    signal_power = torch.diagonal(rx_power, dim1=1, dim2=2)
    total_power = torch.sum(rx_power, dim=2)
    interference_power = total_power - signal_power

    sinr = signal_power / (interference_power + noise_power + 1e-8)
    se = torch.log2(1.0 + sinr)
    return torch.sum(se, dim=1)


def load_data_for_test(args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    suffix = f"L{args.L}_K{args.K}_M{args.M}_number{args.samples}"
    h_path = os.path.join(args.data_dir, f"setHCF_{suffix}.npy")
    w_path = os.path.join(args.data_dir, f"setWCF_{suffix}.npy")

    if not os.path.exists(h_path):
        raise FileNotFoundError(f"Data file not found: {h_path}")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Data file not found: {w_path}")

    h_all = np.load(h_path)
    w_all = np.load(w_path)

    if h_all.ndim == 5:
        h_all = h_all.squeeze(1)
    if w_all.ndim == 5:
        w_all = w_all.squeeze(1)

    total_samples = h_all.shape[0]
    test_samples = min(args.test_samples, total_samples)
    start_idx = total_samples - test_samples

    h_test = h_all[start_idx:]
    w_test = w_all[start_idx:]

    beta_linear = np.mean(np.abs(h_test) ** 2, axis=-1)
    beta_db = 10 * np.log10(beta_linear + 1e-15)
    mean_beta = np.mean(beta_db)
    std_beta = np.std(beta_db)
    beta_norm = (beta_db - mean_beta) / (std_beta + 1e-8)

    mask_test = np.zeros_like(beta_linear)
    for i in range(len(beta_linear)):
        top_idx = np.argsort(beta_linear[i], axis=0)[-args.n_neighbors:, :]
        for k in range(args.K):
            mask_test[i, top_idx[:, k], k] = 1.0

    h_t = torch.from_numpy(h_test).cfloat()
    w_t = torch.from_numpy(w_test).cfloat()
    beta_t = torch.from_numpy(beta_norm).float()
    mask_t = torch.from_numpy(mask_test).float()

    return h_t, w_t, beta_t, mask_t


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format.")


@torch.no_grad()
def collect_iteration_curve(
    model: Iterative_IB_System,
    h_val: torch.Tensor,
    w_val: torch.Tensor,
    beta_val: torch.Tensor,
    mask_val: torch.Tensor,
    batch_size: int,
    device,
) -> Tuple[List[float], List[float]]:
    num_steps = model.num_iterations + 1
    se_sums = np.zeros(num_steps, dtype=np.float64)
    total_samples = len(h_val)

    indices = list(range(0, total_samples, batch_size))

    for start_idx in indices:
        end_idx = min(start_idx + batch_size, total_samples)

        h_batch = h_val[start_idx:end_idx].to(device)
        w_batch = w_val[start_idx:end_idx].to(device)
        beta_batch = beta_val[start_idx:end_idx].to(device)
        mask_batch = mask_val[start_idx:end_idx].to(device)

        current_batch = end_idx - start_idx
        p_current = torch.ones(current_batch, model.L, model.K, device=device) * (model.Pmax / model.K)

        se_init = compute_se_batch(p_current, h_batch, w_batch)
        se_sums[0] += se_init.sum().item()

        for iter_idx in range(model.num_iterations):
            current_sinr = model.calculate_sinr(h_batch, w_batch, p_current)
            sinr_state = current_sinr.unsqueeze(1).expand(current_batch, model.L, model.K)

            z_list = []
            for ap_idx in range(model.L):
                h_local = h_batch[:, ap_idx, :, :]
                beta_local = beta_batch[:, ap_idx, :]
                sinr_local = sinr_state[:, ap_idx, :]
                z_local, _ = model.encoder(h_local, beta_local, sinr_local)
                z_list.append(z_local)

            z_all = torch.stack(z_list, dim=1)
            context_k = model.aggregator(z_all, beta_batch, mask_batch)
            p_new = model.policy(h_batch, context_k, p_current)
            p_current = model.apply_power_constraints(p_new)

            se_step = compute_se_batch(p_current, h_batch, w_batch)
            se_sums[iter_idx + 1] += se_step.sum().item()

    avg_se = (se_sums / total_samples).tolist()
    gains = [avg_se[i + 1] - avg_se[i] for i in range(len(avg_se) - 1)]
    return avg_se, gains


def save_plot(avg_se: List[float], save_path: str, title: str) -> None:
    steps = list(range(len(avg_se)))

    plt.figure(figsize=(8.0, 5.5))
    plt.plot(steps, avg_se, marker="o", linewidth=2.2, markersize=8, color="#1f77b4")

    for step, value in zip(steps, avg_se):
        plt.annotate(
            f"{value:.2f}",
            (step, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
        )

    plt.xticks(steps, [f"t={step}" for step in steps], fontsize=11)
    plt.xlabel("Iteration Step", fontsize=12)
    plt.ylabel("Average Sum SE (bps/Hz)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.65)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot average sum SE across the three old-model refinement steps.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to old-model checkpoint, e.g., federated_cluster_beta_10.0.pth")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--output_plot", type=str, default="iteration_se_curve_v2.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Please place your old-model checkpoint in this folder and pass it with --checkpoint.\n"
            "Recommended for IB+FL: federated_cluster_beta_*.pth"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")

    h_val, w_val, beta_val, mask_val = load_data_for_test(args)
    print(f"Loaded test samples: {len(h_val)}")

    model = Iterative_IB_System(
        args.L,
        args.K,
        args.M,
        args.z_dim,
        args.Pmax,
        num_iterations=args.num_iterations,
    ).to(device)
    load_checkpoint_weights(model, args.checkpoint, device)
    model.eval()

    avg_se, gains = collect_iteration_curve(
        model=model,
        h_val=h_val,
        w_val=w_val,
        beta_val=beta_val,
        mask_val=mask_val,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n" + "=" * 72)
    print("Average Sum SE vs Iteration Step")
    print("=" * 72)
    for idx, value in enumerate(avg_se):
        print(f"t={idx:<2d} | Average Sum SE = {value:.6f} bps/Hz")
    print("-" * 72)
    for idx, gain in enumerate(gains):
        print(f"Gain t={idx}->t={idx + 1} = {gain:.6f} bps/Hz")
    print("=" * 72)

    save_plot(
        avg_se=avg_se,
        save_path=args.output_plot,
        title="Average Sum SE Across Iterative Refinement Steps",
    )
    print(f"\nPlot saved to: {Path(args.output_plot).resolve()}")


if __name__ == "__main__":
    main()
