import os
from typing import Dict, Tuple
import numpy as np

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


