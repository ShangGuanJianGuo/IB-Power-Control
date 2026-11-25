import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_architecture_v2 import Iterative_IB_System


def compute_se_per_sample(p, h, w, noise_power=1.0):
    sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
    w_scaled = sqrt_p * w
    h_u = h.permute(0, 2, 1, 3)
    w_u = w_scaled.permute(0, 2, 1, 3)
    rx = torch.abs(torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)) ** 2
    signal = torch.diagonal(rx, dim1=1, dim2=2)
    interference = torch.sum(rx, dim=2) - signal
    sinr = signal / (interference + noise_power + 1e-8)
    return torch.sum(torch.log2(1 + sinr), dim=1)


def make_final_model_validation_loader(train_args, split_name, max_eval_samples):
    suffix = f"L{train_args['L']}_K{train_args['K']}_M{train_args['M']}_number{train_args['samples']}"
    h_all = np.load(os.path.join(train_args["data_dir"], f"setHCF_{suffix}.npy"))
    w_all = np.load(os.path.join(train_args["data_dir"], f"setWCF_{suffix}.npy"))
    if h_all.ndim == 5:
        h_all = h_all.squeeze(1)
        w_all = w_all.squeeze(1)

    n = min(train_args["train_samples"], h_all.shape[0])
    h_all = h_all[:n]
    w_all = w_all[:n]

    beta_linear = np.mean(np.abs(h_all) ** 2, axis=-1)
    beta_db = 10 * np.log10(beta_linear + 1e-15)
    beta_norm = (beta_db - beta_db.mean()) / (beta_db.std() + 1e-8)

    mask_all = np.zeros_like(beta_linear)
    for i in range(len(beta_linear)):
        top_idx = np.argsort(beta_linear[i], axis=0)[-train_args["n_neighbors"] :, :]
        for k in range(train_args["K"]):
            mask_all[i, top_idx[:, k], k] = 1.0

    torch.manual_seed(42)
    idx = torch.randperm(n)
    train_size = int(0.9 * n)
    selected = idx[:train_size] if split_name == "train" else idx[train_size:]
    if max_eval_samples is not None:
        selected = selected[: min(max_eval_samples, selected.numel())]

    dataset = TensorDataset(
        torch.from_numpy(h_all[selected]).cfloat(),
        torch.from_numpy(w_all[selected]).cfloat(),
        torch.from_numpy(beta_norm[selected]).float(),
        torch.from_numpy(mask_all[selected]).float(),
    )
    return DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=False, num_workers=0)


def load_final_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    train_args = ckpt["args"]
    model = Iterative_IB_System(
        train_args["L"],
        train_args["K"],
        train_args["M"],
        train_args["z_dim"],
        train_args["Pmax"],
        train_args["num_iterations"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, train_args, ckpt


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, split_name, max_eval_samples, device):
    model, train_args, ckpt = load_final_model(checkpoint_path, device)
    loader = make_final_model_validation_loader(train_args, split_name, max_eval_samples)

    model.eval()
    all_se = []
    all_kl = []
    for h, w, beta, mask in loader:
        h = h.to(device)
        w = w.to(device)
        beta = beta.to(device)
        mask = mask.to(device)
        p, kl_div = model(h, w, beta, mask)
        all_se.append(compute_se_per_sample(p, h, w).cpu())
        all_kl.append(kl_div.cpu())

    se = torch.cat(all_se, dim=0)
    kl = torch.cat(all_kl, dim=0)
    metrics = {
        "sum_rate": float(se.mean()),
        "checkpoint_val_se": float(ckpt.get("val_se", float("nan"))),
        "kl_nats": float(kl.mean()),
        "kl_bits": float(kl.mean()) / 0.6931471805599453,
        "num_eval_samples": int(se.shape[0]),
    }
    return metrics, train_args


def beta_from_name(path):
    stem = Path(path).stem
    return float(stem.split("_beta_")[-1])


def main():
    parser = argparse.ArgumentParser(description="Measure online overhead proxies for WCL baselines.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output", default="overhead_measurement_table.csv")
    parser.add_argument("--float_bits", type=int, default=32)
    parser.add_argument("--quantized_bits", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    model_args = None
    proposed_metrics = []
    for checkpoint in sorted(args.checkpoints, key=beta_from_name):
        metrics, train_args = evaluate_checkpoint(checkpoint, args.split, args.max_eval_samples, device)
        model_args = train_args
        proposed_metrics.append((beta_from_name(checkpoint), metrics))

    if model_args is None:
        raise RuntimeError("No checkpoint was evaluated.")

    L = int(model_args["L"])
    K = int(model_args["K"])
    M = int(model_args["M"])
    z_dim = int(model_args["z_dim"])
    num_power_iterations = int(model_args["num_iterations"])

    raw_csi_real_scalars = L * K * M * 2
    latent_real_scalars = L * z_dim * num_power_iterations
    raw_csi_bits = raw_csi_real_scalars * args.float_bits
    latent_float_bits = latent_real_scalars * args.float_bits
    latent_quantized_bits = latent_real_scalars * args.quantized_bits

    def add_row(method, se_ratio, real_scalars, bits, note, kl_bits_per_ap="-"):
        if isinstance(kl_bits_per_ap, (int, float)):
            network_kl_bits = float(kl_bits_per_ap) * L
            kl_overhead = 100.0 * network_kl_bits / raw_csi_bits
            kl_bits_per_ap_value = f"{float(kl_bits_per_ap):.4f}"
            network_kl_bits_value = f"{network_kl_bits:.4f}"
            kl_overhead_value = f"{kl_overhead:.4f}"
        else:
            kl_bits_per_ap_value = "-"
            network_kl_bits_value = "-"
            kl_overhead_value = "-"
        rows.append(
            {
                "method": method,
                "se_ratio_percent": f"{se_ratio:.2f}" if isinstance(se_ratio, (int, float)) else se_ratio,
                "online_real_scalars": real_scalars,
                "online_bits_per_inference": bits,
                "overhead_vs_centralized_percent": (
                    f"{100.0 * float(bits) / raw_csi_bits:.2f}" if isinstance(bits, (int, float)) else bits
                ),
                "measured_kl_bits_per_ap": kl_bits_per_ap_value,
                "measured_network_kl_bits": network_kl_bits_value,
                "measured_kl_overhead_percent": kl_overhead_value,
                "note": note,
            }
        )

    add_row(
        "Centralized WMMSE reference",
        100.00,
        raw_csi_real_scalars,
        raw_csi_bits,
        "global instantaneous CSI, 32-bit real representation",
    )
    add_row(
        "Large-scale-fading local control",
        83.78,
        0,
        0,
        "zero online fronthaul; slow statistical information is assumed pre-known",
    )
    add_row(
        "Local neural control without message exchange",
        79.06,
        0,
        0,
        "local inference only",
    )
    add_row(
        "Federated neural control without bottleneck penalty",
        68.18,
        latent_real_scalars,
        latent_float_bits,
        "deterministic latent messages, 32-bit real representation",
    )

    for beta, metrics in proposed_metrics:
        if beta not in (0.0, 64.0):
            continue
        add_row(
            f"Proposed bottleneck federated control, beta={beta:g}",
            100.0 * metrics["checkpoint_val_se"] / 136.8435,
            latent_real_scalars,
            latent_float_bits,
            "fixed-width latent messages; measured KL bits report the effective IB rate proxy",
            metrics["kl_bits"],
        )

    # Keep the original historical ratios, but make their provenance explicit.
    for result_row in rows:
        result_row['se_ratio_source'] = 'historical table/reference, not a new paired SE experiment'
        result_row['direct_forward_sum_se'] = ''
        result_row['checkpoint_saved_sum_se'] = ''
        for beta, measured in proposed_metrics:
            if result_row['method'] == f'Proposed bottleneck federated control, beta={beta:g}':
                result_row['se_ratio_source'] = 'stored checkpoint val_se / historical WMMSE'
                result_row['direct_forward_sum_se'] = measured['sum_rate']
                result_row['checkpoint_saved_sum_se'] = measured['checkpoint_val_se']

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"device={device}")
    print(f"L={L}, K={K}, M={M}, z_dim={z_dim}, iterations={num_power_iterations}")
    print(f"centralized raw CSI: {raw_csi_real_scalars} real scalars, {raw_csi_bits} bits")
    print(f"latent messages: {latent_real_scalars} real scalars, {latent_float_bits} bits at 32-bit, {latent_quantized_bits} bits at {args.quantized_bits}-bit")
    print(f"wrote {args.output}")
    for beta, metrics in proposed_metrics:
        print(
            f"beta={beta:g}, split={args.split}, samples={metrics['num_eval_samples']}, "
            f"forward_sum_se={metrics['sum_rate']:.4f}, checkpoint_val_se={metrics['checkpoint_val_se']:.4f}, "
            f"se_ratio={100.0 * metrics['checkpoint_val_se'] / 136.8435:.2f}%, "
            f"kl_bits={metrics['kl_bits']:.4f}"
        )


if __name__ == "__main__":
    main()
