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
    parser = argparse.ArgumentParser(description='Evaluate IB SE, KL rate proxy and uplink payload dimensions')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--output', default='overhead_measurement.csv')
    parser.add_argument('--float_bits', type=int, default=32)
    parser.add_argument('--quantized_bits', type=int, default=8)
    args = parser.parse_args()
    if args.float_bits <= 0 or args.quantized_bits <= 0:
        parser.error('Bit widths must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for path in sorted(args.checkpoints, key=beta_from_name):
        metrics, cfg = evaluate_checkpoint(path, args.split, args.max_eval_samples, device)
        L, K, M, dz, steps = (int(cfg[name]) for name in ['L', 'K', 'M', 'z_dim', 'num_iterations'])
        raw_bits = 2 * L * K * M * args.float_bits
        latent_count = steps * L * dz
        rows.append(dict(beta=beta_from_name(path), scenes=metrics['num_eval_samples'],
                         direct_sum_se=metrics['sum_rate'], checkpoint_saved_sum_se=metrics['checkpoint_val_se'],
                         raw_CSI_bits=raw_bits, uplink_latent_bits=latent_count * args.float_bits,
                         uplink_latent_percent=100 * latent_count * args.float_bits / raw_bits,
                         quantized_uplink_bits=latent_count * args.quantized_bits,
                         mean_round_KL_bits_per_AP=metrics['kl_bits'],
                         mean_round_network_KL_bits=L * metrics['kl_bits'],
                         mean_round_KL_percent=100 * L * metrics['kl_bits'] / raw_bits))
    with open(args.output, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f'Saved numeric results to {args.output}')


if __name__ == '__main__':
    main()
