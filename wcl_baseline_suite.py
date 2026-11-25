import argparse
import copy
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ibfl_data import compute_beta_and_mask, load_raw_channel_dataset, split_indices
from ibfl_models import TaskOrientedIBFederatedController, summarize_rate_metrics


@dataclass
class BaselineSplit:
    H: torch.Tensor
    W: torch.Tensor
    beta_norm: torch.Tensor
    beta_linear: torch.Tensor
    mask: torch.Tensor
    client_ids: torch.Tensor

    def __len__(self) -> int:
        return int(self.H.shape[0])


class BaselineDataset(Dataset):
    def __init__(self, split: BaselineSplit):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        return (
            self.split.H[index],
            self.split.W[index],
            self.split.beta_norm[index],
            self.split.beta_linear[index],
            self.split.mask[index],
            self.split.client_ids[index],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_region_clients(beta_linear: np.ndarray, num_clients: int) -> np.ndarray:
    num_samples, num_aps, _ = beta_linear.shape
    ap_regions = np.array_split(np.arange(num_aps), num_clients)
    region_scores = []
    for region in ap_regions:
        region_scores.append(beta_linear[:, region, :].sum(axis=(1, 2)))
    region_scores = np.stack(region_scores, axis=1)
    client_ids = np.argmax(region_scores, axis=1).astype(np.int64)

    counts = np.bincount(client_ids, minlength=num_clients)
    for missing_client in np.where(counts == 0)[0]:
        donor_client = int(np.argmax(counts))
        donor_indices = np.where(client_ids == donor_client)[0]
        client_ids[donor_indices[-1]] = missing_client
        counts = np.bincount(client_ids, minlength=num_clients)
    return client_ids


def build_split(
    h_all: np.ndarray,
    w_all: np.ndarray,
    beta_norm: np.ndarray,
    beta_linear: np.ndarray,
    mask: np.ndarray,
    client_ids: np.ndarray,
    indices: np.ndarray,
) -> BaselineSplit:
    return BaselineSplit(
        H=torch.from_numpy(h_all[indices]).cfloat(),
        W=torch.from_numpy(w_all[indices]).cfloat(),
        beta_norm=torch.from_numpy(beta_norm[indices]).float(),
        beta_linear=torch.from_numpy(beta_linear[indices]).float(),
        mask=torch.from_numpy(mask[indices]).float(),
        client_ids=torch.from_numpy(client_ids[indices]).long(),
    )


def load_baseline_bundle(args) -> Dict[str, object]:
    h_all, w_all, beta_lsf = load_raw_channel_dataset(args.data_dir, args.L, args.K, args.M, args.samples)
    if args.max_samples is not None:
        n = min(args.max_samples, h_all.shape[0])
        h_all = h_all[:n]
        w_all = w_all[:n]
        if beta_lsf is not None:
            beta_lsf = beta_lsf[:n]

    beta_source = "setBetaCF"
    if beta_lsf is None:
        beta_source = "instantaneous_channel_power_fallback"

    beta_linear, beta_norm, mask = compute_beta_and_mask(h_all, args.n_neighbors, beta_lsf=beta_lsf)
    client_ids = assign_region_clients(beta_linear, args.num_clients)
    split_map = split_indices(h_all.shape[0], args.train_ratio, args.val_ratio, args.seed)

    return {
        "train": build_split(h_all, w_all, beta_norm, beta_linear, mask, client_ids, split_map["train"]),
        "val": build_split(h_all, w_all, beta_norm, beta_linear, mask, client_ids, split_map["val"]),
        "test": build_split(h_all, w_all, beta_norm, beta_linear, mask, client_ids, split_map["test"]),
        "beta_source": beta_source,
    }


def truncate_split(split: BaselineSplit, max_items: Optional[int]) -> BaselineSplit:
    if max_items is None:
        return split
    n = min(max_items, len(split))
    return BaselineSplit(
        H=split.H[:n],
        W=split.W[:n],
        beta_norm=split.beta_norm[:n],
        beta_linear=split.beta_linear[:n],
        mask=split.mask[:n],
        client_ids=split.client_ids[:n],
    )


def make_loader(split: BaselineSplit, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(BaselineDataset(split), batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def create_client_loaders(split: BaselineSplit, num_clients: int, batch_size: int) -> List[DataLoader]:
    client_ids_np = split.client_ids.numpy()
    loaders = []
    for client_id in range(num_clients):
        indices = np.where(client_ids_np == client_id)[0]
        client_split = BaselineSplit(
            H=split.H[indices],
            W=split.W[indices],
            beta_norm=split.beta_norm[indices],
            beta_linear=split.beta_linear[indices],
            mask=split.mask[indices],
            client_ids=split.client_ids[indices],
        )
        loaders.append(make_loader(client_split, batch_size, shuffle=True))
    return loaders


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


def project_power_per_ap(p_raw: torch.Tensor, mask: torch.Tensor, p_max: float) -> torch.Tensor:
    p_masked = p_raw.clamp(min=0.0) * mask
    ap_total = p_masked.sum(dim=2, keepdim=True).clamp(min=1e-9)
    scale = torch.clamp(ap_total / p_max, min=1.0)
    return p_masked / scale


def masked_equal_power(mask: torch.Tensor, p_max: float) -> torch.Tensor:
    active_users = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
    return mask * (p_max / active_users)


def lsfc_local_power(
    beta_linear: torch.Tensor,
    mask: torch.Tensor,
    p_max: float,
    exponent: float = 1.0,
) -> torch.Tensor:
    if exponent <= 0.0:
        return masked_equal_power(mask, p_max)
    gain = torch.pow(beta_linear.clamp(min=1e-30), exponent) * mask
    denom = gain.sum(dim=2, keepdim=True).clamp(min=1e-12)
    return p_max * gain / denom


def lsfc_effective_exponent(args) -> float:
    if getattr(args, "beta_source", "") == "setBetaCF":
        return args.lsfc_exponent
    if args.lsfc_fallback_policy == "uniform":
        return 0.0
    if args.lsfc_fallback_policy == "instant_proxy":
        return 1.0
    return args.lsfc_exponent




def masked_wmmse_power_control(
    h: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    p_max: float,
    iterations: int,
    noise_power: float = 1.0,
) -> torch.Tensor:
    p = mask * (p_max / mask.sum(dim=2, keepdim=True).clamp(min=1.0))
    hw_gain = torch.einsum("blkm,bljm->blkj", h.conj(), w)
    best_p = p.clone()
    best_rate = compute_rates(h, w, p, noise_power=noise_power).sum(dim=1)

    for _ in range(iterations):
        sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(2)
        gain = torch.sum(sqrt_p * hw_gain, dim=1)
        total_rx_power = torch.sum(torch.abs(gain) ** 2, dim=2) + noise_power
        gain_diag = torch.diagonal(gain, dim1=1, dim2=2)
        receiver = gain_diag.conj() / total_rx_power
        mmse_error = 1.0 - receiver * gain_diag
        mmse_weight = 1.0 / torch.clamp(mmse_error.real, min=1e-6)
        hw_diag = torch.diagonal(hw_gain, dim1=2, dim2=3)
        utility = mmse_weight.unsqueeze(1) * torch.abs(receiver.unsqueeze(1) * hw_diag) ** 2
        utility = utility * mask
        p = p_max * utility / utility.sum(dim=2, keepdim=True).clamp(min=1e-9)
        current_rate = compute_rates(h, w, p, noise_power=noise_power).sum(dim=1)
        improved = current_rate > best_rate
        if torch.any(improved):
            best_rate = torch.where(improved, current_rate, best_rate)
            best_p[improved] = p[improved]
    return best_p * mask


def centralized_wmmse_power_control(
    h: torch.Tensor,
    w: torch.Tensor,
    p_max: float,
    iterations: int,
    noise_power: float = 1.0,
) -> torch.Tensor:
    mask = torch.ones(h.shape[0], h.shape[1], h.shape[2], device=h.device)
    return masked_wmmse_power_control(h, w, mask, p_max, iterations, noise_power=noise_power)


class DecentralizedLocalDNN(nn.Module):
    def __init__(self, m_antennas: int, p_max: float, num_iterations: int = 3, damping: float = 0.6):
        super().__init__()
        self.p_max = p_max
        self.num_iterations = num_iterations
        self.damping = damping
        input_dim = m_antennas + 4
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def initialize_power(self, mask: torch.Tensor) -> torch.Tensor:
        active_users = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        return mask * (self.p_max / active_users)

    def forward(self, h: torch.Tensor, w: torch.Tensor, beta_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p_current = self.initialize_power(mask)
        for _ in range(self.num_iterations):
            rates = compute_rates(h, w, p_current)
            sinr_proxy = torch.clamp(torch.pow(2.0, rates) - 1.0, min=0.0)
            sinr_expanded = torch.log1p(sinr_proxy).unsqueeze(1).unsqueeze(-1).expand(-1, h.shape[1], -1, 1)
            features = torch.cat(
                [
                    torch.abs(h),
                    beta_norm.unsqueeze(-1),
                    mask.unsqueeze(-1),
                    torch.log1p(p_current).unsqueeze(-1),
                    sinr_expanded,
                ],
                dim=-1,
            )
            p_candidate = self.policy(features).squeeze(-1)
            p_projected = project_power_per_ap(p_candidate, mask, self.p_max)
            p_current = (1.0 - self.damping) * p_current + self.damping * p_projected
        return p_current


def build_ib_model(args) -> TaskOrientedIBFederatedController:
    return TaskOrientedIBFederatedController(
        L=args.L,
        K=args.K,
        M=args.M,
        z_dim=args.z_dim,
        p_max=args.Pmax,
        num_clients=args.num_clients,
        num_power_iterations=args.num_power_iterations,
        message_rounds=args.message_rounds,
        damping=args.damping,
    )




def summarize_metrics(rates: torch.Tensor, elapsed: float, samples: int, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    metrics = summarize_rate_metrics(rates)
    metrics["runtime_ms_per_sample"] = 1000.0 * elapsed / max(samples, 1)
    if extra:
        metrics.update(extra)
    return metrics


@torch.no_grad()
def evaluate_static_baseline(loader, device, args, method: str, max_wmmse_samples: Optional[int] = None) -> Dict[str, float]:
    all_rates = []
    seen = 0
    start = time.perf_counter()
    for h, w, beta_norm, beta_linear, mask, _ in loader:
        if max_wmmse_samples is not None and seen >= max_wmmse_samples:
            break
        if max_wmmse_samples is not None and seen + h.shape[0] > max_wmmse_samples:
            n = max_wmmse_samples - seen
            h, w, beta_norm, beta_linear, mask = h[:n], w[:n], beta_norm[:n], beta_linear[:n], mask[:n]

        h = h.to(device)
        w = w.to(device)
        beta_linear = beta_linear.to(device)
        mask = mask.to(device)

        if method == "lsfc":
            power = lsfc_local_power(beta_linear, mask, args.Pmax, exponent=lsfc_effective_exponent(args))
        elif method == "central_wmmse":
            power = centralized_wmmse_power_control(h, w, args.Pmax, iterations=args.central_wmmse_iter)
        else:
            raise ValueError(f"Unknown static baseline: {method}")
        all_rates.append(compute_rates(h, w, power).cpu())
        seen += h.shape[0]

    elapsed = time.perf_counter() - start
    if not all_rates:
        raise RuntimeError(f"No samples evaluated for {method}.")
    return summarize_metrics(torch.cat(all_rates, dim=0), elapsed, seen)


@torch.no_grad()
def evaluate_dec_dnn(model: DecentralizedLocalDNN, loader, device, args) -> Dict[str, float]:
    model.eval()
    all_rates = []
    seen = 0
    start = time.perf_counter()
    for h, w, beta_norm, _, mask, _ in loader:
        h = h.to(device)
        w = w.to(device)
        beta_norm = beta_norm.to(device)
        mask = mask.to(device)
        power = model(h, w, beta_norm, mask)
        all_rates.append(compute_rates(h, w, power).cpu())
        seen += h.shape[0]
    elapsed = time.perf_counter() - start
    return summarize_metrics(torch.cat(all_rates, dim=0), elapsed, seen)


@torch.no_grad()
def evaluate_ib_model(model: TaskOrientedIBFederatedController, loader, device, deterministic: bool = True) -> Dict[str, float]:
    model.eval()
    all_rates = []
    all_kl = []
    seen = 0
    start = time.perf_counter()
    for h, w, beta_norm, _, mask, client_ids in loader:
        h = h.to(device)
        w = w.to(device)
        beta_norm = beta_norm.to(device)
        mask = mask.to(device)
        client_ids = client_ids.to(device)
        _, kl_div, rates = model(h, w, beta_norm, mask, client_ids, deterministic=deterministic)
        all_rates.append(rates.cpu())
        all_kl.append(kl_div.cpu())
        seen += h.shape[0]
    elapsed = time.perf_counter() - start
    return summarize_metrics(
        torch.cat(all_rates, dim=0),
        elapsed,
        seen,
        extra={"kl_nats": float(torch.cat(all_kl, dim=0).mean())},
    )


def train_dec_dnn(args, train_loader, val_loader, device, checkpoint_path: Path) -> DecentralizedLocalDNN:
    model = DecentralizedLocalDNN(args.M, args.Pmax, args.dec_iterations, args.damping).to(device)
    if checkpoint_path.exists() and not args.retrain:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.dec_lr, weight_decay=args.weight_decay)
    best_val = -1e9
    history = []

    for epoch in range(1, args.dec_epochs + 1):
        model.train()
        for batch_idx, (h, w, beta_norm, _, mask, _) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            h = h.to(device)
            w = w.to(device)
            beta_norm = beta_norm.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            power = model(h, w, beta_norm, mask)
            rates = compute_rates(h, w, power)
            loss = -rates.sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_metrics = evaluate_dec_dnn(model, val_loader, device, args)
        history.append({"epoch": epoch, "val_sum_rate": val_metrics["sum_rate"]})
        print(f"DecDNN epoch {epoch:03d} | Val SE={val_metrics['sum_rate']:.3f}")
        if val_metrics["sum_rate"] > best_val:
            best_val = val_metrics["sum_rate"]
            torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return model


def aggregate_state_dicts(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    aggregated = {}
    for key in states[0].keys():
        aggregated[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
    return aggregated


def train_fl_noib(args, client_loaders, val_loader, device, checkpoint_path: Path) -> TaskOrientedIBFederatedController:
    model = build_ib_model(args).to(device)
    if checkpoint_path.exists() and not args.retrain:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        return model

    best_val = -1e9
    history = []
    for round_idx in range(1, args.noib_fed_rounds + 1):
        local_states = []
        local_weights = []
        for client_id, client_loader in enumerate(client_loaders):
            if len(client_loader.dataset) == 0:
                continue
            local_model = copy.deepcopy(model).to(device)
            local_model.train()
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.noib_lr, weight_decay=args.weight_decay)
            sample_count = 0
            for _ in range(args.noib_local_epochs):
                for batch_idx, (h, w, beta_norm, _, mask, client_ids) in enumerate(client_loader):
                    if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                        break
                    h = h.to(device)
                    w = w.to(device)
                    beta_norm = beta_norm.to(device)
                    mask = mask.to(device)
                    client_ids = client_ids.to(device)
                    optimizer.zero_grad()
                    _, _, rates = local_model(h, w, beta_norm, mask, client_ids, deterministic=True)
                    loss = -rates.sum(dim=1).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()
                    sample_count += h.shape[0]
            local_states.append({k: v.detach().cpu().clone() for k, v in local_model.shared_state_dict().items()})
            local_weights.append(max(sample_count, 1))
            adapter_state = {k: v.detach().cpu().clone() for k, v in local_model.get_adapter_state(client_id).items()}
            model.load_adapter_state(client_id, adapter_state)

        if local_states:
            model.load_shared_state_dict(aggregate_state_dicts(local_states, local_weights))

        val_metrics = evaluate_ib_model(model, val_loader, device, deterministic=True)
        history.append({"round": round_idx, "val_sum_rate": val_metrics["sum_rate"]})
        print(f"FL-noIB round {round_idx:03d} | Val SE={val_metrics['sum_rate']:.3f}")
        if val_metrics["sum_rate"] > best_val:
            best_val = val_metrics["sum_rate"]
            torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return model




def overhead_info(method: str, args, split: BaselineSplit, kl_nats: Optional[float] = None) -> Dict[str, str]:
    raw_global_scalars = 2.0 * args.L * args.K * args.M
    latent_scalars = args.L * args.z_dim * args.message_rounds * args.num_power_iterations

    if method == "LSFC local":
        rho = lsfc_effective_exponent(args)
        if getattr(args, "beta_source", "") == "setBetaCF":
            info = f"LSFC only, rho={rho:.2f}"
        elif args.lsfc_fallback_policy == "instant_proxy":
            info = "full beta proxy, rho=1.00"
        elif args.lsfc_fallback_policy == "fractional_proxy":
            info = f"fractional LSFC proxy, rho={rho:.2f}"
        else:
            info = "service mask only"
        return {"online_info": info, "overhead_norm": "0.0000", "rate_proxy_bits": "-"}
    if method == "Decentralized DNN w/o IB":
        return {"online_info": "local observation", "overhead_norm": "0.0000", "rate_proxy_bits": "-"}
    if method == "FL w/o IB":
        return {
            "online_info": "deterministic latent",
            "overhead_norm": f"{latent_scalars / raw_global_scalars:.4f}",
            "rate_proxy_bits": "-",
        }
    if method == "Centralized WMMSE":
        return {"online_info": "global instantaneous CSI", "overhead_norm": "1.0000", "rate_proxy_bits": "-"}
    return {"online_info": "-", "overhead_norm": "-", "rate_proxy_bits": "-"}


def row(method: str, category: str, metrics: Dict[str, float], args, split: BaselineSplit) -> Dict[str, str]:
    info = overhead_info(method, args, split, metrics.get("kl_nats"))
    complexity = {
        "LSFC local": "Low",
        "Decentralized DNN w/o IB": "Low, fixed-depth",
        "FL w/o IB": "Low, fixed-depth",
        "Centralized WMMSE": f"Very high, {args.central_wmmse_iter} iter.",
    }.get(method, "-")
    return {
        "method": method,
        "category": category,
        "online_info": info["online_info"],
        "sum_se": f"{metrics['sum_rate']:.4f}",
        "mean_ue_se": f"{metrics['mean_rate']:.4f}",
        "p05_ue_se": f"{metrics['p05_rate']:.4f}",
        "min_ue_se": f"{metrics['min_rate']:.4f}",
        "overhead_norm": info["overhead_norm"],
        "rate_proxy_bits": info["rate_proxy_bits"],
        "complexity": complexity,
        "runtime_ms_per_sample": f"{metrics['runtime_ms_per_sample']:.4f}",
    }


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)




def parse_args():
    parser = argparse.ArgumentParser(description="WCL reviewer baseline suite for IB-FL CF-mMIMO power control.")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Pmax", type=float, default=1000.0)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--wmmse_samples", type=int, default=80)
    parser.add_argument("--central_wmmse_iter", type=int, default=25)
    parser.add_argument(
        "--lsfc_fallback_policy",
        type=str,
        default="fractional_proxy",
        choices=["uniform", "fractional_proxy", "instant_proxy"],
    )
    parser.add_argument("--lsfc_exponent", type=float, default=0.6)
    parser.add_argument("--z_dim", type=int, default=24)
    parser.add_argument("--num_power_iterations", type=int, default=3)
    parser.add_argument("--message_rounds", type=int, default=2)
    parser.add_argument("--damping", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dec_epochs", type=int, default=8)
    parser.add_argument("--dec_iterations", type=int, default=3)
    parser.add_argument("--dec_lr", type=float, default=1e-3)
    parser.add_argument("--noib_fed_rounds", type=int, default=8)
    parser.add_argument("--noib_local_epochs", type=int, default=1)
    parser.add_argument("--noib_lr", type=float, default=1e-3)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="wcl_baseline_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip_training_baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    bundle = load_baseline_bundle(args)
    print(f"Beta source: {bundle['beta_source']}")
    args.beta_source = bundle["beta_source"]

    train_split = bundle["train"]
    val_split = truncate_split(bundle["val"], args.eval_samples)
    test_split = truncate_split(bundle["test"], args.eval_samples)
    train_loader = make_loader(train_split, args.batch_size, shuffle=True)
    val_loader = make_loader(val_split, args.batch_size, shuffle=False)
    test_loader = make_loader(test_split, args.batch_size, shuffle=False)
    client_loaders = create_client_loaders(train_split, args.num_clients, args.batch_size)

    rows = []

    print("Evaluating LSFC local power...")
    metrics = evaluate_static_baseline(test_loader, device, args, "lsfc")
    rows.append(row("LSFC local", "zero-fronthaul heuristic", metrics, args, test_split))

    dec_ckpt = ckpt_dir / "decentralized_dnn_noib.pth"
    if args.skip_training_baselines and not dec_ckpt.exists():
        print("Skipping Decentralized DNN w/o IB because checkpoint is missing.")
    else:
        print("Training/evaluating Decentralized DNN w/o IB...")
        dec_model = train_dec_dnn(args, train_loader, val_loader, device, dec_ckpt)
        metrics = evaluate_dec_dnn(dec_model, test_loader, device, args)
        rows.append(row("Decentralized DNN w/o IB", "learning baseline", metrics, args, test_split))

    noib_ckpt = ckpt_dir / "fl_without_ib.pth"
    if args.skip_training_baselines and not noib_ckpt.exists():
        print("Skipping FL w/o IB because checkpoint is missing.")
    else:
        print("Training/evaluating FL w/o IB...")
        noib_model = train_fl_noib(args, client_loaders, val_loader, device, noib_ckpt)
        metrics = evaluate_ib_model(noib_model, test_loader, device, deterministic=True)
        rows.append(row("FL w/o IB", "learning ablation", metrics, args, test_split))

    print("Evaluating Centralized WMMSE benchmark...")
    metrics = evaluate_static_baseline(test_loader, device, args, "central_wmmse", max_wmmse_samples=args.wmmse_samples)
    rows.append(row("Centralized WMMSE", "centralized benchmark", metrics, args, test_split))

    write_csv(output_dir / "wcl_baseline_table.csv", rows)
    print(f"Saved comparison results to {output_dir / 'wcl_baseline_table.csv'}")



if __name__ == "__main__":
    main()
