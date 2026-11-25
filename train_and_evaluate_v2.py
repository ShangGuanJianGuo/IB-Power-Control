"""
训练脚本 v2 - 配合迭代IB系统

改进：
1. 支持迭代优化
2. 更细致的学习率调度
3. 早停机制
4. 性能监控
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse

from model_architecture_v2 import Iterative_IB_System, unsupervised_se_loss


def load_and_process_data(args):
    """加载和预处理数据（与原始版本相同）"""
    print("Loading raw channel data...")
    suffix = f"L{args.L}_K{args.K}_M{args.M}_number{args.samples}"
    H_path = os.path.join(args.data_dir, f"setHCF_{suffix}.npy")
    W_path = os.path.join(args.data_dir, f"setWCF_{suffix}.npy")
    
    if not os.path.exists(H_path):
        raise FileNotFoundError(f"Data file not found: {H_path}")

    H_all = np.load(H_path)
    W_all = np.load(W_path)
    
    if H_all.ndim == 5:
        H_all = H_all.squeeze(1)
    if W_all.ndim == 5:
        W_all = W_all.squeeze(1)
    
    # 截取训练数据
    train_samples = min(args.train_samples, H_all.shape[0])
    H_all = H_all[:train_samples]
    W_all = W_all[:train_samples]
    
    print(f"Data Shape: H {H_all.shape}")

    # 计算大尺度衰落并归一化
    Beta_linear = np.mean(np.abs(H_all)**2, axis=-1)
    Beta_db = 10 * np.log10(Beta_linear + 1e-15)
    mean_beta = np.mean(Beta_db)
    std_beta = np.std(Beta_db)
    Beta_norm = (Beta_db - mean_beta) / (std_beta + 1e-8)
    
    print(f"Beta (dB) normalized stats: Mean={np.mean(Beta_norm):.2f}, Std={np.std(Beta_norm):.2f}")

    # 构建稀疏图（连接性mask）
    N_neighbor = args.n_neighbors
    Mask_all = np.zeros_like(Beta_linear)
    
    for i in range(len(Beta_linear)):
        top_indices = np.argsort(Beta_linear[i], axis=0)[-N_neighbor:, :]
        for k in range(args.K):
            Mask_all[i, top_indices[:, k], k] = 1.0
    
    # 转为Tensor
    H_tensor = torch.from_numpy(H_all).cfloat()
    W_tensor = torch.from_numpy(W_all).cfloat()
    Beta_tensor = torch.from_numpy(Beta_norm).float()
    Mask_tensor = torch.from_numpy(Mask_all).float()
    
    # 划分训练集和验证集
    total_size = len(H_tensor)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = TensorDataset(
        H_tensor[train_indices],
        W_tensor[train_indices],
        Beta_tensor[train_indices],
        Mask_tensor[train_indices]
    )
    
    val_dataset = TensorDataset(
        H_tensor[val_indices],
        W_tensor[val_indices],
        Beta_tensor[val_indices],
        Mask_tensor[val_indices]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def train_one_epoch(model, optimizer, loader, beta_ib, device, epoch):
    """训练一个epoch"""
    model.train()
    total_se_loss = 0
    total_kl_loss = 0
    valid_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    
    for H, W, Beta, Mask in pbar:
        H, W, Beta, Mask = H.to(device), W.to(device), Beta.to(device), Mask.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播（迭代优化）
        P_alloc, kl_divs = model(H, W, Beta, Mask)
        
        # 计算损失
        loss_se = unsupervised_se_loss(P_alloc, H, W)
        loss_kl = torch.mean(kl_divs)
        
        # 总损失
        loss = loss_se + beta_ib * loss_kl
        
        # 检查NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_se_loss += -loss_se.item()
        total_kl_loss += loss_kl.item()
        valid_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            "SE": f"{-loss_se.item():.2f}",
            "KL": f"{loss_kl.item():.2f}",
            "Loss": f"{loss.item():.4f}"
        })
    
    if valid_batches == 0:
        return 0.0, 0.0
    
    avg_se = total_se_loss / valid_batches
    avg_kl = total_kl_loss / valid_batches
    
    return avg_se, avg_kl


def validate(model, loader, beta_ib, device):
    """验证"""
    model.eval()
    total_se_loss = 0
    total_kl_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for H, W, Beta, Mask in loader:
            H, W, Beta, Mask = H.to(device), W.to(device), Beta.to(device), Mask.to(device)
            
            P_alloc, kl_divs = model(H, W, Beta, Mask)
            
            loss_se = unsupervised_se_loss(P_alloc, H, W)
            loss_kl = torch.mean(kl_divs)
            
            if not (torch.isnan(loss_se) or torch.isinf(loss_se)):
                total_se_loss += -loss_se.item()
                total_kl_loss += loss_kl.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return 0.0, 0.0
    
    avg_se = total_se_loss / valid_batches
    avg_kl = total_kl_loss / valid_batches
    
    return avg_se, avg_kl


def run_experiment(args):
    """运行完整实验"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        train_loader, val_loader = load_and_process_data(args)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    results = []
    
    print(f"\n{'='*10} Starting Iterative IB Training {'='*10}")
    print(f"Num iterations per forward pass: {args.num_iterations}")
    print(f"Betas to test: {args.beta_list}")
    
    for beta in args.beta_list:
        print(f"\n{'='*60}")
        print(f"Training with Beta (Compression Weight) = {beta}")
        print(f"{'='*60}")
        
        # 初始化模型
        model = Iterative_IB_System(
            args.L, args.K, args.M,
            args.z_dim, args.Pmax,
            num_iterations=args.num_iterations
        ).to(device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # 早停
        best_val_se = 0
        patience_counter = 0
        best_epoch = 0
        
        train_se_history = []
        val_se_history = []
        
        # 训练循环
        for epoch in range(1, args.epochs + 1):
            # 训练
            train_se, train_kl = train_one_epoch(model, optimizer, train_loader, beta, device, epoch)
            
            # 验证
            val_se, val_kl = validate(model, val_loader, beta, device)
            
            train_se_history.append(train_se)
            val_se_history.append(val_se)
            
            # 更新学习率
            scheduler.step(val_se)
            
            # 打印进度
            print(f"Epoch {epoch:03d} | Train SE: {train_se:.4f} | Val SE: {val_se:.4f} | Val KL: {val_kl:.4f}")
            
            # 早停检查
            if val_se > best_val_se:
                best_val_se = val_se
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                save_path = f"best_model_beta_{beta}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved (Val SE: {best_val_se:.4f})")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break
        
        print(f"\n{'='*60}")
        print(f"Training finished for beta={beta}")
        print(f"Best Val SE: {best_val_se:.4f} at epoch {best_epoch}")
        print(f"{'='*60}")
        
        # 记录结果
        results.append({
            'beta': beta,
            'best_val_se': best_val_se,
            'best_epoch': best_epoch,
            'train_history': train_se_history,
            'val_history': val_se_history
        })
        
    # 保存所有结果
    np.save('training_results_v2.npy', results)
    print("\nAll results saved to training_results_v2.npy")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iterative IB System v2")
    
    # 系统参数
    parser.add_argument('--L', type=int, default=40, help='Number of APs')
    parser.add_argument('--K', type=int, default=30, help='Number of UEs')
    parser.add_argument('--M', type=int, default=8, help='Antennas per AP')
    parser.add_argument('--Pmax', type=float, default=1000, help='Max power per AP (mW)')
    
    # 数据参数
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--train_samples', type=int, default=4000)
    parser.add_argument('--data_dir', type=str, default='.')
    
    # 模型参数
    parser.add_argument('--z_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--num_iterations', type=int, default=3, help='Iterations per forward pass')
    parser.add_argument('--n_neighbors', type=int, default=10)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Beta列表
    parser.add_argument('--beta_list', type=float, nargs='+',
                       default=[4.0],
                       help='List of beta values to test')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    results = run_experiment(args)