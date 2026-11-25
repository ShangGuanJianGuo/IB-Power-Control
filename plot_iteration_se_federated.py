import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_federated_cluster import ClusterFederatedSystem, load_data
from model_architecture_v2 import Iterative_IB_System


def compute_sum_se_batch(p: torch.Tensor, h: torch.Tensor, w: torch.Tensor, noise_power: float = 1.0) -> torch.Tensor:
    sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
    w_scaled = sqrt_p * w
    h_u = h.permute(0, 2, 1, 3)
    w_u = w_scaled.permute(0, 2, 1, 3)
    rx = torch.abs(torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)) ** 2
    signal = torch.diagonal(rx, dim1=1, dim2=2)
    interference = torch.sum(rx, dim=2) - signal
    sinr = signal / (interference + noise_power + 1e-8)
    return torch.sum(torch.log2(1.0 + sinr), dim=1)


def load_checkpoint(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt or "args" not in ckpt:
        raise ValueError("Checkpoint format is not compatible with train_federated_cluster.py")
    return ckpt


def build_args_from_checkpoint(ckpt_args: dict, override_batch_size: int = None):
    args = SimpleNamespace(**ckpt_args)
    if override_batch_size is not None:
        args.batch_size = override_batch_size
    return args


def encode_one_step(model: Iterative_IB_System, h_local, beta_local, sinr_local, deterministic_latent: bool):
    encoder = model.encoder
    features = encoder.extract_csi_features(h_local, beta_local, sinr_local)
    hidden = encoder.encoder_net(features)
    mu = encoder.fc_mu(hidden)
    logvar = torch.clamp(encoder.fc_logvar(hidden), min=-10.0, max=10.0)
    z = mu if deterministic_latent else encoder.reparameterize(mu, logvar)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return z, kl


@torch.no_grad()
def collect_iteration_curve(
    fed_system: ClusterFederatedSystem,
    val_loader,
    deterministic_latent: bool,
    device,
) -> Tuple[List[float], List[float]]:
    model = fed_system.global_model
    model.eval()

    num_steps = model.num_iterations + 1
    se_sums = np.zeros(num_steps, dtype=np.float64)
    total_samples = 0

    for h, w, beta, mask in val_loader:
        h = h.to(device)
        w = w.to(device)
        beta = beta.to(device)
        mask = mask.to(device)

        batch_size = h.shape[0]
        total_samples += batch_size

        p_current = torch.ones(batch_size, model.L, model.K, device=device) * (model.Pmax / model.K)
        se_sums[0] += compute_sum_se_batch(p_current, h, w).sum().item()

        for iter_idx in range(model.num_iterations):
            current_sinr = model.calculate_sinr(h, w, p_current)
            sinr_state = current_sinr.unsqueeze(1).expand(batch_size, model.L, model.K)

            z_list = []
            for ap_idx in range(model.L):
                z_ap, _ = encode_one_step(
                    model,
                    h[:, ap_idx, :, :],
                    beta[:, ap_idx, :],
                    sinr_state[:, ap_idx, :],
                    deterministic_latent=deterministic_latent,
                )
                z_list.append(z_ap)

            z_all = torch.stack(z_list, dim=1)
            context_k = model.aggregator(z_all, beta, mask)
            p_new = model.policy(h, context_k, p_current)
            p_current = model.apply_power_constraints(p_new)

            se_sums[iter_idx + 1] += compute_sum_se_batch(p_current, h, w).sum().item()

    avg_se = (se_sums / total_samples).tolist()
    gains = [avg_se[idx + 1] - avg_se[idx] for idx in range(len(avg_se) - 1)]
    return avg_se, gains


def save_plot(avg_se: List[float], save_path: str, title: str):
    steps = list(range(len(avg_se)))
    plt.figure(figsize=(8.0, 5.6))
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
    parser = argparse.ArgumentParser(description="Plot SE-vs-step using federated_cluster_beta_*.pth with the exact train_federated_cluster.py setup.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to federated_cluster_beta_*.pth")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for evaluation batch size")
    parser.add_argument("--output_plot", type=str, default="iteration_se_curve_federated.png")
    parser.add_argument("--stochastic_latent", action="store_true", help="Use stochastic latent sampling instead of deterministic z=mu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")

    ckpt = load_checkpoint(args.checkpoint, device)
    train_args = build_args_from_checkpoint(ckpt["args"], override_batch_size=args.batch_size)

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader = load_data(train_args)

    model = Iterative_IB_System(
        train_args.L,
        train_args.K,
        train_args.M,
        train_args.z_dim,
        train_args.Pmax,
        train_args.num_iterations,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    fed_system = ClusterFederatedSystem(model, train_args.K, device)
    fed_system.cluster_weights = ckpt["fed_system_weights"]
    fed_system.cluster_active = ckpt["fed_system_active"]

    avg_se, gains = collect_iteration_curve(
        fed_system=fed_system,
        val_loader=val_loader,
        deterministic_latent=not args.stochastic_latent,
        device=device,
    )

    print("\n" + "=" * 72)
    print("Average Sum SE vs Iteration Step (Federated Validation Split)")
    print("=" * 72)
    print(f"Latent mode: {'stochastic' if args.stochastic_latent else 'deterministic (z=mu)'}")
    for idx, value in enumerate(avg_se):
        print(f"t={idx:<2d} | Average Sum SE = {value:.6f} bps/Hz")
    print("-" * 72)
    for idx, gain in enumerate(gains):
        print(f"Gain t={idx}->t={idx + 1} = {gain:.6f} bps/Hz")
    print("=" * 72)

    save_plot(
        avg_se=avg_se,
        save_path=args.output_plot,
        title="Average Sum SE Across Federated Iterative Refinement Steps",
    )
    print(f"\nPlot saved to: {Path(args.output_plot).resolve()}")


if __name__ == "__main__":
    main()
