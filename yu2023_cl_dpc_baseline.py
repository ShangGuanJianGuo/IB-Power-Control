"""Yu2023-style cooperative-learning decentralized power-control baseline.

This is an adaptation of:
    D. Yu et al., "Learning decentralized power control in cell-free massive
    MIMO networks," IEEE TVT, 2023.

No official code was found.  The implementation follows the paper's scalable
CL structure:
    1) AP-shared uplink-message DNN V(rho_i);
    2) CP-shared message processor F(m_i) followed by average pooling;
    3) AP-shared decision DNN D(f, rho_i, h_i);
    4) AP-wise ReLU6-style total-power gate and ratio scaling.

It is adapted to the current project where each AP has M antennas and MR
precoding is fixed by setWCF.  The AP-local short-term feature is the antenna
averaged channel power, while the local long-term feature is the normalized
AP-UE channel gain used for association in the existing baseline suite.
"""

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ibfl_data import compute_beta_and_mask, load_raw_channel_dataset, split_indices


@dataclass
class Split:
    h: torch.Tensor
    w: torch.Tensor
    beta_norm: torch.Tensor
    beta_linear: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.h.shape[0])


class ChannelDataset(Dataset):
    def __init__(self, split: Split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        return (
            self.split.h[idx],
            self.split.w[idx],
            self.split.beta_norm[idx],
            self.split.beta_linear[idx],
            self.split.mask[idx],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_hidden: int, batch_norm: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    dim = input_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def compute_rates(h: torch.Tensor, w: torch.Tensor, p: torch.Tensor, noise_power: float = 1.0) -> torch.Tensor:
    sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
    w_scaled = sqrt_p * w
    h_u = h.permute(0, 2, 1, 3)
    w_u = w_scaled.permute(0, 2, 1, 3)
    rx_matrix = torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)
    rx_power = torch.abs(rx_matrix) ** 2
    signal = torch.diagonal(rx_power, dim1=1, dim2=2)
    total = rx_power.sum(dim=2)
    interference = total - signal
    sinr = signal / (interference + noise_power + 1e-8)
    return torch.log2(1.0 + sinr)


class Yu2023CLDPC(nn.Module):
    def __init__(
        self,
        num_users: int,
        p_max: float,
        uplink_dim: int,
        downlink_dim: int,
        hidden_dim: int,
        num_hidden: int,
        batch_norm: bool,
        use_service_mask: bool,
    ):
        super().__init__()
        self.K = num_users
        self.p_max = p_max
        self.use_service_mask = use_service_mask

        # Paper uses rho_i' in [0, sqrt(P)] as AP-local long-term CSI input.
        self.uplink_dnn = make_mlp(num_users, hidden_dim, uplink_dim, num_hidden, batch_norm)
        self.cp_dnn = make_mlp(uplink_dim, hidden_dim, downlink_dim, num_hidden, batch_norm)

        # AP decision DNN input: downlink message f, rho_i', local short-term CSI.
        decision_input_dim = downlink_dim + 2 * num_users
        self.decision_dnn = make_mlp(decision_input_dim, hidden_dim, num_users + 1, num_hidden, batch_norm)

    def normalized_lsf_feature(self, beta_norm: torch.Tensor) -> torch.Tensor:
        # Bounded, positive AP-local feature analogous to rho_i' in the paper.
        return torch.sigmoid(beta_norm) * (self.p_max ** 0.5)

    def local_short_term_feature(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,L,K,M).  Use antenna-averaged instantaneous channel power.
        h_power = torch.mean(torch.abs(h) ** 2, dim=-1)
        h_log = torch.log10(h_power + 1e-12)
        mean = h_log.mean(dim=(1, 2), keepdim=True)
        std = h_log.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (h_log - mean) / std

    def forward(self, h: torch.Tensor, beta_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_aps, num_users, _ = h.shape
        rho_feat = self.normalized_lsf_feature(beta_norm)
        short_feat = self.local_short_term_feature(h)

        rho_flat = rho_feat.reshape(batch_size * num_aps, num_users)
        uplink_message = self.uplink_dnn(rho_flat)
        cp_feature = self.cp_dnn(uplink_message).reshape(batch_size, num_aps, -1)
        downlink_message = cp_feature.mean(dim=1)
        downlink_expanded = downlink_message.unsqueeze(1).expand(batch_size, num_aps, -1)

        decision_input = torch.cat([downlink_expanded, rho_feat, short_feat], dim=-1)
        raw = self.decision_dnn(decision_input.reshape(batch_size * num_aps, -1))
        raw = raw.reshape(batch_size, num_aps, num_users + 1)

        ratio_logits = torch.nn.functional.softplus(raw[..., :num_users])
        if self.use_service_mask:
            ratio_logits = ratio_logits * mask
        ratio_sum = ratio_logits.sum(dim=2, keepdim=True).clamp(min=1e-9)
        ratios = ratio_logits / ratio_sum

        # Bounded total-power gate, serving the same role as the paper's ReLU6
        # gate while avoiding dead APs during adaptation to this dataset.
        total_power = self.p_max * torch.sigmoid(raw[..., num_users : num_users + 1])
        power = total_power * ratios

        if self.use_service_mask:
            power = power * mask
        return power


def build_split(
    h_all: np.ndarray,
    w_all: np.ndarray,
    beta_norm: np.ndarray,
    beta_linear: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
) -> Split:
    return Split(
        h=torch.from_numpy(h_all[indices]).cfloat(),
        w=torch.from_numpy(w_all[indices]).cfloat(),
        beta_norm=torch.from_numpy(beta_norm[indices]).float(),
        beta_linear=torch.from_numpy(beta_linear[indices]).float(),
        mask=torch.from_numpy(mask[indices]).float(),
    )


def load_data(args) -> Dict[str, object]:
    h_all, w_all, beta_lsf = load_raw_channel_dataset(args.data_dir, args.L, args.K, args.M, args.samples)
    if args.max_samples is not None:
        n = min(args.max_samples, h_all.shape[0])
        h_all = h_all[:n]
        w_all = w_all[:n]
        if beta_lsf is not None:
            beta_lsf = beta_lsf[:n]

    beta_linear, beta_norm, mask = compute_beta_and_mask(h_all, args.n_neighbors, beta_lsf=beta_lsf)
    if args.legacy_torch_split:
        torch.manual_seed(args.seed)
        indices = torch.randperm(h_all.shape[0]).numpy()
        train_end = int(args.train_ratio * h_all.shape[0])
        split_map = {
            "train": indices[:train_end],
            "val": indices[train_end:],
            "test": indices[train_end:],
        }
    else:
        split_map = split_indices(h_all.shape[0], args.train_ratio, args.val_ratio, args.seed)
    return {
        "train": build_split(h_all, w_all, beta_norm, beta_linear, mask, split_map["train"]),
        "val": build_split(h_all, w_all, beta_norm, beta_linear, mask, split_map["val"]),
        "test": build_split(h_all, w_all, beta_norm, beta_linear, mask, split_map["test"]),
        "beta_source": "setBetaCF" if beta_lsf is not None else "instantaneous_channel_power_fallback",
    }


def truncate_split(split: Split, max_items: Optional[int]) -> Split:
    if max_items is None:
        return split
    n = min(max_items, len(split))
    return Split(
        h=split.h[:n],
        w=split.w[:n],
        beta_norm=split.beta_norm[:n],
        beta_linear=split.beta_linear[:n],
        mask=split.mask[:n],
    )


def make_loader(split: Split, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ChannelDataset(split), batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


@torch.no_grad()
def evaluate_model(model: Yu2023CLDPC, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_rates = []
    seen = 0
    start = time.perf_counter()
    for h, w, beta_norm, _, mask in loader:
        h = h.to(device)
        w = w.to(device)
        beta_norm = beta_norm.to(device)
        mask = mask.to(device)
        power = model(h, beta_norm, mask)
        all_rates.append(compute_rates(h, w, power).cpu())
        seen += h.shape[0]
    elapsed = time.perf_counter() - start
    rates = torch.cat(all_rates, dim=0)
    sum_rate = rates.sum(dim=1)
    return {
        "sum_se": float(sum_rate.mean()),
        "std_sum_se": float(sum_rate.std(unbiased=False)),
        "min_ue_se": float(rates.mean(dim=0).min()),
        "runtime_ms_per_sample": 1000.0 * elapsed / max(seen, 1),
        "num_samples": int(seen),
    }


def train_model(args, train_loader, val_loader, device: torch.device, checkpoint_path: Path) -> Yu2023CLDPC:
    model = Yu2023CLDPC(
        num_users=args.K,
        p_max=args.Pmax,
        uplink_dim=args.uplink_dim,
        downlink_dim=args.downlink_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        batch_norm=args.batch_norm,
        use_service_mask=args.use_service_mask,
    ).to(device)

    if checkpoint_path.exists() and not args.retrain:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)
    best_val = -1e9
    history = []
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_idx, (h, w, beta_norm, _, mask) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            h = h.to(device)
            w = w.to(device)
            beta_norm = beta_norm.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            power = model(h, beta_norm, mask)
            rates = compute_rates(h, w, power)
            loss = -rates.sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["sum_se"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_sum_se": val_metrics["sum_se"],
            }
        )
        print(
            f"Yu2023-CL epoch {epoch:03d} | "
            f"train loss={history[-1]['train_loss']:.4f} | val SE={val_metrics['sum_se']:.4f}"
        )

        if val_metrics["sum_se"] > best_val + args.min_delta:
            best_val = val_metrics["sum_se"]
            wait = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_val_sum_se": best_val,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  -> saved best checkpoint ({best_val:.4f})")
        else:
            wait += 1
            if args.early_stop_patience > 0 and wait >= args.early_stop_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def save_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Yu2023-style CL-DPC baseline for the current CF-mMIMO dataset.")
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--legacy_torch_split",
        action="store_true",
        help="Use the original train_federated_cluster.py split: torch.randperm, train then remaining validation.",
    )
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_train_eval_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--uplink_dim", type=int, default=30)
    parser.add_argument("--downlink_dim", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden", type=int, default=3)
    parser.add_argument("--batch_norm", action="store_true", default=False)
    parser.add_argument("--use_service_mask", action="store_true", default=True)
    parser.add_argument("--full_ap_output", dest="use_service_mask", action="store_false")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=16)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--output_dir", type=str, default="yu2023_cl_results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Yu2023 CL-DPC adaptation: shared AP uplink DNN, CP average-pooling message, shared AP decision DNN.")

    bundle = load_data(args)
    train_split = truncate_split(bundle["train"], args.max_train_eval_samples)
    val_split = truncate_split(bundle["val"], args.max_eval_samples)
    test_split = truncate_split(bundle["test"], args.max_eval_samples)
    train_loader = make_loader(train_split, args.batch_size, shuffle=True)
    val_loader = make_loader(val_split, args.batch_size, shuffle=False)
    test_loader = make_loader(test_split, args.batch_size, shuffle=False)
    print(
        f"Data: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}, "
        f"beta_source={bundle['beta_source']}"
    )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "yu2023_cl_dpc_best.pth"
    model = train_model(args, train_loader, val_loader, device, checkpoint_path)
    train_metrics = evaluate_model(model, make_loader(train_split, args.batch_size, shuffle=False), device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    rows = []
    for split_name, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        rows.append(
            {
                "method": "Yu2023 CL-DPC adaptation",
                "split": split_name,
                **metrics,
                "uplink_dim": args.uplink_dim,
                "downlink_dim": args.downlink_dim,
                "hidden_dim": args.hidden_dim,
                "num_hidden": args.num_hidden,
                "use_service_mask": args.use_service_mask,
            }
        )
    save_summary(output_dir / "yu2023_cl_dpc_summary.csv", rows)

    print("\nFinal metrics:")
    for row in rows:
        print(
            f"{row['split']:<5} | Sum SE={row['sum_se']:.4f} | "
            f"Std={row['std_sum_se']:.4f} | Min UE SE={row['min_ue_se']:.4f} | "
            f"Runtime={row['runtime_ms_per_sample']:.4f} ms/sample"
        )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved summary: {output_dir / 'yu2023_cl_dpc_summary.csv'}")


if __name__ == "__main__":
    main()
