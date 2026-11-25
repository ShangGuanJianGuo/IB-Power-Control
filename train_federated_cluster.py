"""Cluster-labelled IB power-control training and numeric validation outputs."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import copy
from tqdm import tqdm
import argparse
import csv

# ===== 直接导入原有模型 =====
from model_architecture_v2 import (
    Iterative_IB_System, 
    Enhanced_IB_Encoder,
    Attention_Aggregator,
    Iterative_Power_Policy,
    unsupervised_se_loss
)


def compute_se(P, H, W, noise_power=1.0):
    """计算SE"""
    sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1)
    W_scaled = sqrt_P * W
    H_u, W_u = H.permute(0, 2, 1, 3), W_scaled.permute(0, 2, 1, 3)
    Rx = torch.abs(torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)) ** 2
    signal = torch.diagonal(Rx, dim1=1, dim2=2)
    interference = torch.sum(Rx, dim=2) - signal
    sinr = signal / (interference + noise_power + 1e-8)
    return torch.sum(torch.log2(1 + sinr), dim=1)


# ==================== 服务簇联邦系统 ====================

class ClusterFederatedSystem:
    """
    基于服务簇的联邦系统
    
    物理含义：
    - 每个UE由一个"服务簇"负责（主AP + 辅助APs）
    - 各簇独立编码本地CSI，只交换压缩表示z
    - 隐私保护：原始CSI H不出簇
    """
    
    def __init__(self, model, K, device):
        """
        model: 全局IB模型
        K: UE数量（=簇数量）
        """
        self.global_model = model
        self.K = K
        self.device = device
        
        # 每个UE簇的贡献权重
        self.cluster_weights = torch.ones(K) / K
        self.cluster_active = torch.ones(K, dtype=torch.bool)  # 哪些簇是活跃的
        
        
    def get_cluster_mask(self, connectivity_mask, ue_id):
        """获取服务特定UE的AP掩码"""
        # connectivity_mask: (B, L, K)
        # 返回服务UE ue_id的AP掩码
        return connectivity_mask[:, :, ue_id]  # (B, L)
    
    def cluster_forward(self, H, W, Beta, Mask, ue_ids=None):
        """
        按簇进行前向传播
        
        ue_ids: 要处理的UE列表（None表示全部）
        返回：P, kl, 每个簇的损失贡献
        """
        if ue_ids is None:
            ue_ids = range(self.K)
        
        # 只处理活跃的UE簇
        active_ues = [k for k in ue_ids if self.cluster_active[k]]
        
        # 正常前向传播
        P, kl = self.global_model(H, W, Beta, Mask)
        
        # 计算每个簇的SE贡献
        se_per_ue = self._compute_se_per_ue(P, H, W)  # (B, K)
        
        # 按簇权重汇总
        weighted_se = se_per_ue * self.cluster_weights.to(self.device)
        
        return P, kl, weighted_se, se_per_ue
    
    def _compute_se_per_ue(self, P, H, W, noise_power=1.0):
        """计算每个UE的SE"""
        sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1)
        W_scaled = sqrt_P * W
        H_u, W_u = H.permute(0, 2, 1, 3), W_scaled.permute(0, 2, 1, 3)
        Rx = torch.abs(torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)) ** 2
        signal = torch.diagonal(Rx, dim1=1, dim2=2)
        interference = torch.sum(Rx, dim=2) - signal
        sinr = signal / (interference + noise_power + 1e-8)
        return torch.log2(1 + sinr)  # (B, K)
    


# ==================== 数据加载 ====================

def load_data(args):
    """加载数据"""
    print("Loading data...")
    suffix = f"L{args.L}_K{args.K}_M{args.M}_number{args.samples}"
    H_all = np.load(os.path.join(args.data_dir, f"setHCF_{suffix}.npy"))
    W_all = np.load(os.path.join(args.data_dir, f"setWCF_{suffix}.npy"))
    
    if H_all.ndim == 5:
        H_all, W_all = H_all.squeeze(1), W_all.squeeze(1)
    
    n = min(args.train_samples, H_all.shape[0])
    H_all, W_all = H_all[:n], W_all[:n]
    
    # Beta归一化
    Beta_linear = np.mean(np.abs(H_all)**2, axis=-1)
    Beta_db = 10 * np.log10(Beta_linear + 1e-15)
    Beta_norm = (Beta_db - Beta_db.mean()) / (Beta_db.std() + 1e-8)
    
    # Mask
    Mask_all = np.zeros_like(Beta_linear)
    for i in range(len(Beta_linear)):
        top_idx = np.argsort(Beta_linear[i], axis=0)[-args.n_neighbors:, :]
        for k in range(args.K):
            Mask_all[i, top_idx[:, k], k] = 1.0
    
    # 转Tensor并划分
    H_t = torch.from_numpy(H_all).cfloat()
    W_t = torch.from_numpy(W_all).cfloat()
    Beta_t = torch.from_numpy(Beta_norm).float()
    Mask_t = torch.from_numpy(Mask_all).float()
    
    train_size = int(0.9 * n)
    torch.manual_seed(42)
    idx = torch.randperm(n)
    
    train_ds = TensorDataset(H_t[idx[:train_size]], W_t[idx[:train_size]], 
                             Beta_t[idx[:train_size]], Mask_t[idx[:train_size]])
    val_ds = TensorDataset(H_t[idx[train_size:]], W_t[idx[train_size:]], 
                           Beta_t[idx[train_size:]], Mask_t[idx[train_size:]])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader


# ==================== 训练与评估 ====================

def train_epoch(fed_system, optimizer, loader, beta_ib, device):
    """训练一个epoch"""
    fed_system.global_model.train()
    total_se, total_kl, count = 0, 0, 0
    
    for H, W, Beta, Mask in loader:
        H, W = H.to(device), W.to(device)
        Beta, Mask = Beta.to(device), Mask.to(device)
        
        optimizer.zero_grad()
        
        # 联邦前向传播
        P, kl, weighted_se, se_per_ue = fed_system.cluster_forward(H, W, Beta, Mask)
        
        # 损失：与原版完全一致（训练时不加权）
        total_se_batch = se_per_ue.sum(dim=1)  # (B,)
        loss = -total_se_batch.mean() + beta_ib * kl.mean()
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fed_system.global_model.parameters(), 1.0)
            optimizer.step()
            
            total_se += total_se_batch.mean().item()
            total_kl += kl.mean().item()
            count += 1
    
    return total_se / max(count, 1), total_kl / max(count, 1)


def validate(fed_system, loader, device):
    """验证"""
    fed_system.global_model.eval()
    total_se, count = 0, 0
    se_per_ue_all = []
    
    with torch.no_grad():
        for H, W, Beta, Mask in loader:
            H, W = H.to(device), W.to(device)
            Beta, Mask = Beta.to(device), Mask.to(device)
            
            P, _, _, se_per_ue = fed_system.cluster_forward(H, W, Beta, Mask)
            
            total_se += se_per_ue.sum(dim=1).mean().item()
            se_per_ue_all.append(se_per_ue.mean(dim=0).cpu())
            count += 1
    
    avg_se_per_ue = torch.stack(se_per_ue_all).mean(dim=0)
    return total_se / max(count, 1), avg_se_per_ue


def train_federated(args):
    """联邦训练主流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader = load_data(args)
    
    # 创建模型和联邦系统
    model = Iterative_IB_System(
        args.L, args.K, args.M, args.z_dim, args.Pmax, args.num_iterations
    ).to(device)
    
    fed_system = ClusterFederatedSystem(model, args.K, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    # 训练记录
    best_se = 0
    history = {'train_se': [], 'val_se': []}
    
    print(f"\n{'='*60}")
    print(f"Federated Training (Cluster-based): β={args.beta}")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        train_se, train_kl = train_epoch(fed_system, optimizer, train_loader, args.beta, device)
        val_se, se_per_ue = validate(fed_system, val_loader, device)
        
        scheduler.step(val_se)
        
        history['train_se'].append(train_se)
        history['val_se'].append(val_se)
        
        print(f"Epoch {epoch:03d} | Train SE: {train_se:.2f} | Val SE: {val_se:.2f} | KL: {train_kl:.4f}")
        
        if val_se > best_se:
            best_se = val_se
            torch.save({
                'model_state_dict': model.state_dict(),
                'fed_system_weights': fed_system.cluster_weights,
                'fed_system_active': fed_system.cluster_active,
                'epoch': epoch,
                'val_se': val_se,
                'args': vars(args)
            }, f"federated_cluster_beta_{args.beta}.pth")
            print(f"  -> Saved (best SE: {best_se:.2f})")
    
    with open(f'federated_training_history_beta_{args.beta}.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_sum_se', 'validation_sum_se'])
        writer.writerows((i + 1, tr, va) for i, (tr, va) in enumerate(zip(history['train_se'], history['val_se'])))

    print(f"\nBest Val SE: {best_se:.2f}")
    return fed_system




# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=40)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--Pmax', type=float, default=1000)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--train_samples', type=int, default=4000)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=10.0)
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_federated(args)
