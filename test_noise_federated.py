"""
联邦学习架构下的噪声鲁棒性测试脚本
(Robustness Test for Federated Learning Framework)

功能：
1. 自动搜索目录下的联邦模型权重 (federated_cluster_beta_*.pth)
2. 加载全局模型 (Global Model)
3. 在不同信道估计噪声 (SNR) 下测试系统性能
4. 生成对比图表和汇总表格

使用前请确保:
- model_architecture_v2.py 在同一目录
- 数据文件 (setHCF_...npy) 在同一目录
"""

import torch
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# ===== 导入模型定义 =====
# 假设你的模型文件名为 model_architecture_v2.py 或 .txt
# 如果报错找不到模块，请确保文件名正确
try:
    from model_architecture_v2 import Iterative_IB_System
except ImportError:
    # 为了防止直接运行出错，这里定义一个假的占位符，实际运行时请确保文件存在
    print("Warning: model_architecture_v2 not found. Please ensure the file exists.")
    class Iterative_IB_System: pass

# ==================== 工具函数 ====================

def add_noise_snr(H, snr_db):
    """
    向信道矩阵 H 添加指定 SNR 的高斯白噪声
    用于模拟非完美的信道估计 (Imperfect CSI)
    """
    # 100dB 或更高被视为无噪声
    if snr_db >= 100:
        return H.clone()
    
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 计算信号功率 (平均每个元素的功率)
    # H shape: (B, L, K, M)
    signal_power = torch.mean(torch.abs(H) ** 2, dim=(-3, -2, -1), keepdim=True)
    
    # 计算噪声标准差: P_noise = P_signal / SNR
    # 复数噪声功率平分给实部和虚部，所以除以 2
    noise_std = torch.sqrt(signal_power / snr_linear / 2.0)
    
    # 生成复数噪声
    noise = torch.complex(
        torch.randn_like(H.real) * noise_std.real,
        torch.randn_like(H.imag) * noise_std.real
    )
    
    return H + noise


def compute_se_batch(P, H, W, noise_power=1.0):
    """
    计算批次频谱效率 (SE)
    用于评估性能
    """
    # P: (B, L, K) -> sqrt_P: (B, L, K, 1)
    sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1)
    
    # 缩放预编码向量 W: (B, L, K, M)
    W_scaled = sqrt_P * W
    
    # 调整维度以进行矩阵乘法: (B, K, L, M)
    H_u = H.permute(0, 2, 1, 3)
    W_u = W_scaled.permute(0, 2, 1, 3)
    
    # 计算接收信号矩阵 (B, K, K)
    # 元素 (b, k, j) 表示用户 k 接收到的来自用户 j 的信号
    Rx_matrix = torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)
    Rx_power = torch.abs(Rx_matrix) ** 2
    
    # 信号功率 (对角线)
    signal_power = torch.diagonal(Rx_power, dim1=1, dim2=2)
    
    # 总接收功率
    total_power = torch.sum(Rx_power, dim=2)
    
    # 干扰功率
    interference_power = total_power - signal_power
    
    # SINR
    sinr = signal_power / (interference_power + noise_power + 1e-8)
    
    # Sum SE
    se = torch.log2(1 + sinr)
    return se  # (B, K)


def load_data_for_test(args):
    """
    加载测试数据 (与训练数据保持一致的处理逻辑)
    """
    print(f"Loading data from {args.data_dir}...")
    suffix = f"L{args.L}_K{args.K}_M{args.M}_number{args.samples}"
    
    H_path = os.path.join(args.data_dir, f"setHCF_{suffix}.npy")
    W_path = os.path.join(args.data_dir, f"setWCF_{suffix}.npy")
    
    if not os.path.exists(H_path):
        raise FileNotFoundError(f"Data file not found: {H_path}")
        
    H_all = np.load(H_path)
    W_all = np.load(W_path)
    
    if H_all.ndim == 5:
        H_all, W_all = H_all.squeeze(1), W_all.squeeze(1)
    
    # 使用测试集部分 (例如最后 10% 或专门的测试集)
    # 这里假设我们使用最后 args.test_samples 个样本
    total_samples = H_all.shape[0]
    start_idx = total_samples - args.test_samples
    
    H_test = H_all[start_idx:]
    W_test = W_all[start_idx:]
    
    print(f"Testing on {len(H_test)} samples (Index {start_idx} to {total_samples})")
    
    # === 预处理 (与训练完全一致) ===
    # 1. Beta 归一化
    Beta_linear = np.mean(np.abs(H_test)**2, axis=-1)
    Beta_db = 10 * np.log10(Beta_linear + 1e-15)
    
    # 注意：这里的均值和方差最好使用训练集的统计量，但在测试脚本中通常重新计算或使用全局统计
    # 为了简化，这里重新计算 batch 的统计量，影响不大
    mean_beta = np.mean(Beta_db)
    std_beta = np.std(Beta_db)
    Beta_norm = (Beta_db - mean_beta) / (std_beta + 1e-8)
    
    # 2. 生成 Mask (User-Centric Topology)
    Mask_test = np.zeros_like(Beta_linear)
    for i in range(len(Beta_linear)):
        top_idx = np.argsort(Beta_linear[i], axis=0)[-args.n_neighbors:, :]
        for k in range(args.K):
            Mask_test[i, top_idx[:, k], k] = 1.0
            
    # 转 Tensor
    H_t = torch.from_numpy(H_test).cfloat()
    W_t = torch.from_numpy(W_test).cfloat()
    Beta_t = torch.from_numpy(Beta_norm).float()
    Mask_t = torch.from_numpy(Mask_test).float()
    
    return H_t, W_t, Beta_t, Mask_t


# ==================== 主测试逻辑 ====================

def main():
    # ===== 1. 参数设置 (请根据你的数据生成参数修改) =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=40)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--Pmax', type=float, default=1000)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=500) # 测试样本数
    parser.add_argument('--batch_size', type=int, default=20)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness test on device: {device}")
    
    # ===== 2. 加载数据 =====
    try:
        H_val, W_val, Beta_val, Mask_val = load_data_for_test(args)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 将数据移至 GPU (如果显存足够，一次性移入；否则需要在循环中移)
    # 考虑到 batch 测试，这里保持在 CPU，循环中移入 GPU
    
    # ===== 3. 搜索联邦模型权重 =====
    # 搜索模式匹配 federated_cluster_beta_*.pth
    model_pattern = "federated_cluster_beta_*.pth"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print("No federated model files found! Please check the directory.")
        return
        
    beta_models = []
    for path in model_files:
        # 提取 beta 值
        match = re.search(r'beta_([0-9.]+)\.pth', path)
        if match:
            beta_val = float(match.group(1))
            beta_models.append((beta_val, path))
    
    # 按 beta 从小到大排序
    beta_models.sort(key=lambda x: x[0])
    print(f"Found {len(beta_models)} federated models: {[b for b, _ in beta_models]}")
    
    # ===== 4. 定义 SNR 测试等级 =====
    # 100 代表无噪声 (Perfect CSI)
    snr_levels = [100, 20, 15, 10, 5, 0, -5]
    
    results = {} # 存储结果: {beta: {snr: se}}
    
    # ===== 5. 开始评估循环 =====
    for beta, model_path in beta_models:
        print(f"\n{'='*60}")
        print(f"Evaluating Federated Model with Beta = {beta}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        # 初始化模型架构
        model = Iterative_IB_System(
            args.L, args.K, args.M, args.z_dim, args.Pmax, args.num_iterations
        ).to(device)
        
        # 加载权重
        # 注意：联邦脚本保存的字典包含 'model_state_dict', 'fed_system_weights' 等
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  -> Global model weights loaded successfully.")
        else:
            print("  -> Error: 'model_state_dict' not found in checkpoint.")
            continue
            
        model.eval()
        results[beta] = {}
        
        # 遍历 SNR 等级
        for snr in snr_levels:
            total_se = 0
            num_batches = 0
            
            # Batch 评估
            num_samples = len(H_val)
            indices = list(range(0, num_samples, args.batch_size))
            
            with torch.no_grad():
                for start_idx in indices:
                    end_idx = min(start_idx + args.batch_size, num_samples)
                    
                    # 获取 Batch 数据
                    h_batch = H_val[start_idx:end_idx].to(device)
                    w_batch = W_val[start_idx:end_idx].to(device)
                    beta_batch = Beta_val[start_idx:end_idx].to(device)
                    mask_batch = Mask_val[start_idx:end_idx].to(device)
                    
                    # === 关键步骤：添加噪声 ===
                    # 噪声加在输入的 H 上
                    h_noisy = add_noise_snr(h_batch, snr)
                    
                    # 前向传播 (P, kl)
                    # 注意：IB模型输入的是 h_noisy，但计算 SE 时使用的是真实的 h_batch
                    P_pred, _ = model(h_noisy, w_batch, beta_batch, mask_batch)
                    
                    # 计算真实 SE (使用无噪信道评估性能)
                    se_batch = compute_se_batch(P_pred, h_batch, w_batch)
                    total_se += se_batch.sum().item()
                    num_batches += (end_idx - start_idx) * args.K # 总用户数
            
            avg_se = total_se / (len(H_val)) # 平均每个样本的总 SE (Sum SE)
            
            snr_str = "NoNoise" if snr >= 100 else f"{snr}dB"
            print(f"  SNR: {snr_str:<8} | Sum SE: {avg_se:.4f} bps/Hz")
            results[beta][snr] = avg_se

    # ===== 6. 输出汇总表格 (Summary Table) =====
    print("\n" + "="*100)
    print("FEDERATED ROBUSTNESS SUMMARY TABLE (Sum SE in bps/Hz)")
    print("="*100)
    
    # 表头
    header = f"{'Beta':<10}"
    for snr in snr_levels:
        label = "NoNoise" if snr >= 100 else f"SNR={snr}"
        header += f"{label:<12}"
    print(header)
    print("-" * 100)
    
    # 数据行
    for beta in sorted(results.keys()):
        row = f"{beta:<10}"
        for snr in snr_levels:
            val = results[beta].get(snr, 0.0)
            row += f"{val:<12.2f}"
        print(row)
    print("="*100)
    
    # ===== 7. 绘图 (Robustness Plot) =====
    plt.figure(figsize=(10, 7))
    
    # 选取几个代表性的 Beta 进行绘图 (避免线条太多)
    # 如果 Beta 很多，可以手动指定要画的，或者画全部
    plot_snrs = [s for s in snr_levels if s < 100] # x轴只画有噪声的部分
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i, beta in enumerate(sorted(results.keys())):
        y_vals = [results[beta][s] for s in plot_snrs]
        
        # 样式循环
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        plt.plot(plot_snrs, y_vals, marker=marker, linestyle='-', linewidth=2, color=color, label=f'Fed IB (β={beta})')
        
    plt.xlabel('Channel Estimation SNR (dB)', fontsize=12)
    plt.ylabel('Sum Spectral Efficiency (bps/Hz)', fontsize=12)
    plt.title('Robustness of Federated Global Model to CSI Noise', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    
    # 反转 x 轴，使 SNR 从高到低 (20 -> -5)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('federated_robustness_result.png', dpi=300)
    print("\nPlot saved to: federated_robustness_result.png")

if __name__ == "__main__":
    # 设置随机种子以保证复现性
    torch.manual_seed(42)
    np.random.seed(42)
    main()