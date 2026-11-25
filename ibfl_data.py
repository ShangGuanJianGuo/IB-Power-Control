import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitTensors:
    H: torch.Tensor
    W: torch.Tensor
    beta: torch.Tensor
    mask: torch.Tensor
    client_ids: torch.Tensor

    def __len__(self) -> int:
        return int(self.H.shape[0])


class ChannelTensorDataset(Dataset):
    def __init__(self, split: SplitTensors):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        return (
            self.split.H[idx],
            self.split.W[idx],
            self.split.beta[idx],
            self.split.mask[idx],
            self.split.client_ids[idx],
        )


def load_raw_channel_dataset(data_dir: str, L: int, K: int, M: int, samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = f"L{L}_K{K}_M{M}_number{samples}"
    h_path = os.path.join(data_dir, f"setHCF_{suffix}.npy")
    w_path = os.path.join(data_dir, f"setWCF_{suffix}.npy")
    beta_path = os.path.join(data_dir, f"setBetaCF_{suffix}.npy")

    if not os.path.exists(h_path):
        raise FileNotFoundError(f"Missing dataset file: {h_path}")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing dataset file: {w_path}")

    h_all = np.load(h_path)
    w_all = np.load(w_path)

    if h_all.ndim == 5:
        h_all = h_all.squeeze(1)
    if w_all.ndim == 5:
        w_all = w_all.squeeze(1)

    beta_lsf = np.load(beta_path) if os.path.exists(beta_path) else None

    return h_all, w_all, beta_lsf


def compute_beta_and_mask(h_all: np.ndarray, n_neighbors: int, beta_lsf: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if beta_lsf is not None:
        beta_linear = beta_lsf
    else:
        beta_linear = np.mean(np.abs(h_all) ** 2, axis=-1)
    beta_db = 10.0 * np.log10(beta_linear + 1e-15)
    beta_norm = (beta_db - beta_db.mean()) / (beta_db.std() + 1e-8)

    mask = np.zeros_like(beta_linear, dtype=np.float32)
    for sample_idx in range(beta_linear.shape[0]):
        top_idx = np.argsort(beta_linear[sample_idx], axis=0)[-n_neighbors:, :]
        for user_idx in range(beta_linear.shape[2]):
            mask[sample_idx, top_idx[:, user_idx], user_idx] = 1.0

    return beta_linear, beta_norm.astype(np.float32), mask.astype(np.float32)


def assign_region_clients(beta_linear: np.ndarray, num_clients: int) -> np.ndarray:
    num_samples, num_aps, _ = beta_linear.shape
    ap_regions = np.array_split(np.arange(num_aps), num_clients)

    region_scores = []
    for region in ap_regions:
        score = beta_linear[:, region, :].sum(axis=(1, 2))
        region_scores.append(score)
    region_scores = np.stack(region_scores, axis=1)

    client_ids = np.argmax(region_scores, axis=1).astype(np.int64)

    counts = np.bincount(client_ids, minlength=num_clients)
    for missing_client in np.where(counts == 0)[0]:
        donor_client = int(np.argmax(counts))
        donor_indices = np.where(client_ids == donor_client)[0]
        move_index = donor_indices[-1]
        client_ids[move_index] = missing_client
        counts = np.bincount(client_ids, minlength=num_clients)

    return client_ids


def split_indices(num_samples: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.0")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _build_split(
    h_all: np.ndarray,
    w_all: np.ndarray,
    beta_norm: np.ndarray,
    mask: np.ndarray,
    client_ids: np.ndarray,
    indices: np.ndarray,
) -> SplitTensors:
    return SplitTensors(
        H=torch.from_numpy(h_all[indices]).cfloat(),
        W=torch.from_numpy(w_all[indices]).cfloat(),
        beta=torch.from_numpy(beta_norm[indices]).float(),
        mask=torch.from_numpy(mask[indices]).float(),
        client_ids=torch.from_numpy(client_ids[indices]).long(),
    )


def prepare_data_bundle(
    data_dir: str,
    L: int,
    K: int,
    M: int,
    samples: int,
    n_neighbors: int,
    num_clients: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int = None,
) -> Dict[str, object]:
    h_all, w_all, beta_lsf = load_raw_channel_dataset(data_dir, L, K, M, samples)

    if max_samples is not None:
        max_samples = min(max_samples, h_all.shape[0])
        h_all = h_all[:max_samples]
        w_all = w_all[:max_samples]
        if beta_lsf is not None:
            beta_lsf = beta_lsf[:max_samples]

    beta_linear, beta_norm, mask = compute_beta_and_mask(h_all, n_neighbors, beta_lsf=beta_lsf)
    client_ids = assign_region_clients(beta_linear, num_clients)
    split_map = split_indices(h_all.shape[0], train_ratio, val_ratio, seed)

    train_split = _build_split(h_all, w_all, beta_norm, mask, client_ids, split_map["train"])
    val_split = _build_split(h_all, w_all, beta_norm, mask, client_ids, split_map["val"])
    test_split = _build_split(h_all, w_all, beta_norm, mask, client_ids, split_map["test"])

    train_counts = np.bincount(train_split.client_ids.numpy(), minlength=num_clients)

    return {
        "train": train_split,
        "val": val_split,
        "test": test_split,
        "client_counts": train_counts,
        "num_clients": num_clients,
        "beta_linear_mean": float(beta_linear.mean()),
        "beta_linear_std": float(beta_linear.std()),
        "beta_source": "setBetaCF" if beta_lsf is not None else "instantaneous_channel_power_fallback",
    }


def create_loader(split: SplitTensors, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ChannelTensorDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def create_client_loaders(train_split: SplitTensors, num_clients: int, batch_size: int) -> List[DataLoader]:
    loaders = []
    client_ids_np = train_split.client_ids.numpy()

    for client_id in range(num_clients):
        indices = np.where(client_ids_np == client_id)[0]
        if len(indices) == 0:
            empty_split = SplitTensors(
                H=train_split.H[:0],
                W=train_split.W[:0],
                beta=train_split.beta[:0],
                mask=train_split.mask[:0],
                client_ids=train_split.client_ids[:0],
            )
            loaders.append(create_loader(empty_split, batch_size, shuffle=False))
            continue

        client_split = SplitTensors(
            H=train_split.H[indices],
            W=train_split.W[indices],
            beta=train_split.beta[indices],
            mask=train_split.mask[indices],
            client_ids=train_split.client_ids[indices],
        )
        loaders.append(create_loader(client_split, batch_size, shuffle=True))

    return loaders


def summarize_bundle(bundle: Dict[str, object]) -> str:
    train_len = len(bundle["train"])
    val_len = len(bundle["val"])
    test_len = len(bundle["test"])
    counts = bundle["client_counts"]
    count_str = ", ".join(f"c{idx}={int(count)}" for idx, count in enumerate(counts))
    return f"train={train_len}, val={val_len}, test={test_len}, clients: {count_str}"
