import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm import tqdm

# ==========================================
# 1. 导入依赖
# ==========================================
try:
    from model_architecture_v2 import Iterative_IB_System
except ImportError:
    print("Error: model_architecture_v2.py not found.")
    exit()

# ==========================================
# 2. 攻击者模型
# ==========================================
class AdversaryDecoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super().__init__()
        # 保持强大的网络结构
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# ==========================================
# 3. 自定义损失函数 (核心修改)
# ==========================================
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # target (Normalized Beta): (Batch, K)
        # 我们希望攻击者重点关注那些数值大的点（强用户）
        # 计算权重：基于 target 的绝对值（值越大，权重越大）
        # 使用 Softmax 或 简单的指数 来放大强信号的权重
        
        # 将 target 还原回正数空间近似计算权重 (假设标准化后 > 0 的是强信号)
        weights = torch.abs(target) 
        weights = torch.clamp(weights, min=0.1) # 避免权重为0
        
        # 归一化权重，保持 loss 的量级一致
        weights = weights / weights.mean()
        
        loss = (pred - target) ** 2
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

# ==========================================
# 4. 攻击执行逻辑
# ==========================================
def run_attack(model, data_loader, device, beta_val, fixed_sample, epochs=500):
    model.eval()
    
    # --- Phase 1: 数据生成 ---
    z_list = []
    truth_list = []
    
    with torch.no_grad():
        for i, (H, W, Beta, Mask) in enumerate(data_loader):
            H, Beta = H.to(device), Beta.to(device)
            B, L, K, M = H.shape
            
            for l in range(L):
                H_l = H[:, l, :, :]      
                beta_l = Beta[:, l, :]   
                sinr_dummy = torch.zeros(B, K).to(device)
                z_l, _ = model.encoder(H_l, beta_l, sinr_dummy)
                z_list.append(z_l)
                truth_list.append(beta_l)

    X_attack = torch.cat(z_list, dim=0) 
    Y_attack = torch.cat(truth_list, dim=0) 
    
    # --- 双向标准化 (Z-score) ---
    z_mean = X_attack.mean(dim=0)
    z_std = X_attack.std(dim=0) + 1e-8
    X_attack = (X_attack - z_mean) / z_std
    
    y_mean = Y_attack.mean(dim=0)
    y_std = Y_attack.std(dim=0) + 1e-8
    Y_attack = (Y_attack - y_mean) / y_std
    
    # 划分数据集
    total_len = len(X_attack)
    split = int(0.85 * total_len) 
    train_x, test_x = X_attack[:split], X_attack[split:]
    train_y, test_y = Y_attack[:split], Y_attack[split:]
    
    # --- Phase 2: 训练攻击者 ---
    K_dim = train_y.shape[1]
    attacker = AdversaryDecoder(z_dim=model.encoder.z_dim, output_dim=K_dim).to(device)
    
    optimizer = optim.Adam(attacker.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30)
    
    # [修改] 使用加权 Loss，迫使攻击者学习波峰
    criterion = WeightedMSELoss() 
    # 或者如果不想用自定义Loss，可以用 nn.MSELoss() 但结果可能就像之前一样
    # criterion = nn.MSELoss() 
    
    attack_batch_size = 1024 
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    attack_loader = torch.utils.data.DataLoader(train_dataset, batch_size=attack_batch_size, shuffle=True)
    
    attacker.train()
    for ep in range(epochs):
        for bx, by in attack_loader:
            optimizer.zero_grad()
            pred = attacker(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        
        if ep % 50 == 0:
            attacker.eval()
            with torch.no_grad():
                val_pred = attacker(test_x)
                val_loss = criterion(val_pred, test_y)
            scheduler.step(val_loss)
            attacker.train()
    
    # --- Phase 3: 评估 ---
    attacker.eval()
    with torch.no_grad():
        # 计算最终 MSE (使用标准 MSE 进行公平报告)
        std_criterion = nn.MSELoss()
        pred_test = attacker(test_x)
        mse_score = std_criterion(pred_test, test_y).item()
        
        # 计算相关系数 (Pearson Correlation) 作为辅助指标
        # 如果 MSE 很难降下来，相关系数通常能更好体现“形状还原度”
        vx = pred_test - pred_test.mean(dim=1, keepdim=True)
        vy = test_y - test_y.mean(dim=1, keepdim=True)
        corr = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8)
        avg_corr = corr.mean().item()
        
        # 导出样本
        H_fix, Beta_fix = fixed_sample
        H_fix, Beta_fix = H_fix.to(device), Beta_fix.to(device)
        B_fix = H_fix.shape[0]
        K_fix = H_fix.shape[2] 
        
        l_idx = 0 
        H_l_fix = H_fix[:, l_idx, :, :]      
        beta_l_fix = Beta_fix[:, l_idx, :]   
        sinr_dummy = torch.zeros(B_fix, K_fix).to(device)
        
        z_vis, _ = model.encoder(H_l_fix, beta_l_fix, sinr_dummy)
        z_vis = (z_vis - z_mean) / z_std # 标准化
        pred_vis = attacker(z_vis)
        
        # 目标反标准化以便观察真实物理意义? 不，还是保持标准化画图比较清晰
        gt_vis_norm = (beta_l_fix - y_mean) / y_std
        
        pred_vis_np = pred_vis[0].cpu().numpy()
        gt_vis_np = gt_vis_norm[0].cpu().numpy()
    
    # 我们主要看 MSE，但也打印 Correlation 参考
    print(f"  > Beta={beta_val:<4} | MSE: {mse_score:.4f} | Corr: {avg_corr:.4f}")
    
    return mse_score, gt_vis_np, pred_vis_np, avg_corr

# ==========================================
# 4. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Final copy-3 supervised latent reconstruction attack (not gradient inversion)')
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--checkpoint_dir', default='.')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--train_samples', type=int, default=4500)
    parser.add_argument('--L', type=int, default=40)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--Pmax', type=float, default=1000.)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--betas', default='0,1,2,4,8,16,32,64')
    parser.add_argument('--output_prefix', default='privacy_attack')
    args = parser.parse_args()
    L, K, M, z_dim, Pmax = args.L, args.K, args.M, args.z_dim, args.Pmax
    beta_list = [float(value) for value in args.betas.split(',')]
    if len(set(beta_list)) != len(beta_list):
        parser.error('Each beta must occur once')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        from train_federated_cluster import load_data
        print("Loading large dataset...")
        train_loader, _ = load_data(args)
    except ImportError:
        print("Error: load_data failed.")
        return

    fixed_batch = next(iter(train_loader))
    H_batch, _, Beta_batch, _ = fixed_batch
    fixed_sample = (H_batch[0:1], Beta_batch[0:1]) 
    
    results_mse = []
    results_corr = []
    csv_rows = []
    reconstruction_dict = {}
    ground_truth_vector = None

    print(f"\n{'='*60}")
    print("STARTING ATTACK V6 (Weighted Loss for Strong Users)")
    print(f"{'='*60}")

    for beta in beta_list:
        model_path = os.path.join(args.checkpoint_dir, f"federated_cluster_beta_{beta}.pth")
        if not os.path.exists(model_path):
            print(f"Skipping {model_path}")
            results_mse.append(None)
            results_corr.append(None)
            continue
            
        model = Iterative_IB_System(L, K, M, z_dim, Pmax).to(device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
            
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) 
            
        mse, gt_vec, pred_vec, mean_corr = run_attack(model, train_loader, device, beta, fixed_sample, epochs=args.epochs)
        results_corr.append(mean_corr)
        csv_rows.append(dict(beta=beta, mse=mse, mean_pearson=mean_corr, epochs=args.epochs, scenario_pool=args.train_samples))
        results_mse.append(mse)
        
        reconstruction_dict[f"pred_beta_{int(beta)}"] = pred_vec
        if ground_truth_vector is None:
            ground_truth_vector = gt_vec

    valid_indices = [i for i, x in enumerate(results_mse) if x is not None]
    valid_betas = [beta_list[i] for i in valid_indices]
    valid_mses = [results_mse[i] for i in valid_indices]
    
    mat_data = {
        "beta_values": np.array(valid_betas),
        "mse_values": np.array(valid_mses),
        "mean_pearson_values": np.array([results_corr[i] for i in valid_indices]),
        "ground_truth": ground_truth_vector
    }
    mat_data.update(reconstruction_dict)
    
    if not valid_indices:
        raise FileNotFoundError('No requested checkpoints found')
    sio.savemat(f"{args.output_prefix}_data.mat", mat_data)
    with open(f"{args.output_prefix}_results.csv", 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Success] Saved {args.output_prefix}_data.mat and {args.output_prefix}_results.csv")

if __name__ == "__main__":
    main()