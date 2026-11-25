"""
WMMSE (Imperfect CSI) 独立鲁棒性测试脚本

功能：
1. 加载信道数据
2. 模拟不同 SNR 下的信道估计误差
3. 关键修正：基于含噪信道重新计算预编码 W (MR Precoding)，模拟真实的非完美 CSI 场景
4. 运行 WMMSE 算法在含噪信道上优化 P
5. 在真实信道上评估性能

使用方法：
python test_wmmse_noise.py --data_dir .
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# ==================== 1. 基础工具函数 ====================

def add_noise_snr(H, snr_db):
    """向信道矩阵 H 添加指定 SNR 的高斯白噪声"""
    if snr_db >= 100:
        return H.clone()
    
    snr_linear = 10 ** (snr_db / 10.0)
    # H: (B, L, K, M)
    signal_power = torch.mean(torch.abs(H) ** 2, dim=(-3, -2, -1), keepdim=True)
    noise_std = torch.sqrt(signal_power / snr_linear / 2.0)
    
    noise = torch.complex(
        torch.randn_like(H.real) * noise_std.real,
        torch.randn_like(H.imag) * noise_std.real
    )
    return H + noise


def compute_mr_precoding(H, Pmax, K):
    """
    基于当前信道 H 计算最大比传输 (MR) 预编码
    W = sqrt(Pmax/K) * H / |H|
    """
    # H: (B, L, K, M)
    # Norm over antenna dimension M: (B, L, K, 1)
    # complex norm: sqrt(real^2 + imag^2)
    H_norm = torch.abs(H) # magnitude of complex number? No, we need vector norm
    # torch.norm is deprecated, use linalg.vector_norm or manual
    H_mag = torch.sqrt(torch.sum(torch.abs(H)**2, dim=-1, keepdim=True))
    
    # Avoid division by zero
    H_mag = torch.clamp(H_mag, min=1e-12)
    
    # MR Precoding direction: H / |H|
    # Note: In CF-mMIMO generation code, W is H.conj() or H? 
    # Usually W is H.conj() for DL? Let's check generation code logic.
    # The uploaded generation code: W = ... * H / norm(H)
    # But DL channel is H_{lk}. Received signal y = h^H * w.
    # So if w = h, then h^H * h = |h|^2. Correct.
    # So W direction is same as H direction.
    
    W_dir = H / H_mag
    
    # Power scaling: sqrt(Pmax / K)
    scale = np.sqrt(Pmax / K)
    W = scale * W_dir
    
    return W


def compute_se_batch(P, H, W, noise_power=1.0):
    """
    计算批次频谱效率 (SE)
    P: (B, L, K)
    H: (B, L, K, M)
    W: (B, L, K, M)
    """
    sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1) # (B, L, K, 1)
    W_scaled = sqrt_P * W # (B, L, K, M)
    
    # H_u: (B, K, L, M) - 用户 k 的信道
    H_u = H.permute(0, 2, 1, 3)
    # W_u: (B, K, L, M) - 发给用户 j 的预编码
    W_u = W_scaled.permute(0, 2, 1, 3)
    
    # Rx_matrix: (B, K_rx, K_tx)
    # 用户 k 接收到来自 用户 j 的信号 (所有 AP 的叠加)
    Rx_matrix = torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)
    Rx_power = torch.abs(Rx_matrix) ** 2
    
    signal_power = torch.diagonal(Rx_power, dim1=1, dim2=2)
    total_power = torch.sum(Rx_power, dim=2)
    interference_power = total_power - signal_power
    
    sinr = signal_power / (interference_power + noise_power + 1e-8)
    se = torch.log2(1 + sinr)
    return se  # (B, K)


def load_data_for_test(args):
    """加载测试数据"""
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
    
    # 使用测试集部分
    total_samples = H_all.shape[0]
    start_idx = total_samples - args.test_samples
    H_test = H_all[start_idx:]
    W_test = W_all[start_idx:]
    
    print(f"Testing on {len(H_test)} samples")
            
    H_t = torch.from_numpy(H_test).cfloat()
    W_t = torch.from_numpy(W_test).cfloat()
    return H_t, W_t


# ==================== 2. WMMSE 核心逻辑 (复现 benchmark) ====================

def wmmse_power_control(H, W, P_init, Pmax, iterations=50, noise_power=1.0):
    """
    PyTorch 批量实现的 WMMSE 功率控制
    逻辑完全复现 benchmark_upper_bound.txt 中的 WMMSE_Optimizer
    """
    B, L, K, M = H.shape
    device = H.device
    P = P_init.clone()
    
    # Pre-compute beamforming gains: (B, L, K_rx, K_tx)
    # HW_gain[b, l, k, j] = h_{b,l,k}^H * w_{b,l,j}
    HW_gain = torch.einsum('blkm,bljm->blkj', H.conj(), W)
    
    for _ in range(iterations):
        # 1. Update Receiver (U)
        sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(2) # (B, L, 1, K_tx)
        
        # G: (B, K_rx, K_tx)
        G = torch.sum(sqrt_P * HW_gain, dim=1)
        
        # Signal+Interference Power: sum_j |g_kj|^2 + sigma^2
        total_rx_power = torch.sum(torch.abs(G)**2, dim=2) + noise_power # (B, K)
        
        # u_k = g_kk^H / total_rx_power
        # g_kk: diagonal of G
        G_diag = torch.diagonal(G, dim1=1, dim2=2) # (B, K)
        U = G_diag.conj() / total_rx_power # (B, K)
        
        # 2. Update MMSE Weight (W_mmse)
        # mmse = 1 - u_k * g_kk
        mmse_error = 1.0 - U * G_diag # (B, K)
        # w_k = 1 / real(mmse)
        W_mmse = 1.0 / torch.clamp(mmse_error.real, min=1e-6) # (B, K)
        
        # 3. Update Power (P)
        
        # HW_diag: h_{lk}^H w_{lk} -> diagonal of HW_gain over (k, j) axes
        HW_diag = torch.diagonal(HW_gain, dim1=2, dim2=3) # (B, L, K)
        
        # Term inside square: u_k * h_w_lk
        term = U.unsqueeze(1) * HW_diag # (B, L, K)
        
        # Utility: w_k * |term|^2
        utility = W_mmse.unsqueeze(1) * (torch.abs(term)**2) # (B, L, K)
        
        # Allocate Power per AP
        sum_utility = torch.sum(utility, dim=2, keepdim=True)
        sum_utility = torch.clamp(sum_utility, min=1e-9)
        
        # P_new = (utility / sum_utility) * Pmax
        P = (utility / sum_utility) * Pmax
        
    return P


# ==================== 3. 主测试逻辑 ====================

def main():
    parser = argparse.ArgumentParser(description="WMMSE Robustness Test")
    parser.add_argument('--L', type=int, default=40)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--Pmax', type=float, default=1000)
    
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100, help='WMMSE iterations')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        H_val, W_val = load_data_for_test(args)
    except Exception as e:
        print(f"Error: {e}")
        return

    snr_levels = [100, 20, 15, 10, 5, 0, -5]
    results = {}
    
    print("\n" + "="*60)
    print("Starting WMMSE Robustness Evaluation")
    print("NOTE: Simulating Realistic Imperfect CSI (W is re-calculated from Noisy H)")
    print("="*60)
    
    for snr in snr_levels:
        total_se = 0
        indices = list(range(0, len(H_val), args.batch_size))
        
        snr_label = "NoNoise" if snr >= 100 else f"{snr}dB"
        pbar = tqdm(indices, desc=f"Evaluating SNR={snr_label}")
        
        for start_idx in pbar:
            end_idx = min(start_idx + args.batch_size, len(H_val))
            current_batch_size = end_idx - start_idx
            
            # 1. 准备真实数据
            h_batch = H_val[start_idx:end_idx].to(device)
            w_batch_perfect = W_val[start_idx:end_idx].to(device)
            
            # 2. 加噪
            h_noisy = add_noise_snr(h_batch, snr)
            
            # 3. 关键修改: 基于 h_noisy 重新计算 W (MR Precoding)
            # 这样 W 和 P 都受到噪声影响，这才是真正的 Imperfect CSI
            if snr >= 100:
                w_input = w_batch_perfect
            else:
                w_input = compute_mr_precoding(h_noisy, args.Pmax, args.K)
            
            # 3. 运行 WMMSE 优化 P
            # 算法输入: h_noisy, w_input (都是含噪的)
            P_init = torch.ones(current_batch_size, args.L, args.K, device=device) * (args.Pmax / args.K)
            P_opt = wmmse_power_control(h_noisy, w_input, P_init, args.Pmax, iterations=args.iterations)
            
            # 4. 评估 (使用真实信道 h_batch 和 实际使用的 w_input)
            # 注意: 系统实际上使用的是基于含噪信道计算出的 w_input
            with torch.no_grad():
                se_real = compute_se_batch(P_opt, h_batch, w_input)
                total_se += se_real.sum().item()
        
        avg_se = total_se / len(H_val)
        results[snr] = avg_se
        print(f"  -> Average Sum SE: {avg_se:.4f} bps/Hz")

    # ===== 输出表格 =====
    print("\n" + "="*50)
    print("WMMSE ROBUSTNESS RESULTS (Recalculated W)")
    print("="*50)
    print(f"{'SNR (dB)':<15} | {'Sum SE (bps/Hz)':<20}")
    print("-" * 50)
    for snr in snr_levels:
        label = "NoNoise" if snr >= 100 else str(snr)
        print(f"{label:<15} | {results[snr]:<20.4f}")
    print("="*50)
    
    # ===== 绘图 =====
    plt.figure(figsize=(8, 6))
    plot_snrs = [s for s in snr_levels if s < 100]
    y_vals = [results[s] for s in plot_snrs]
    
    plt.plot(plot_snrs, y_vals, 'k--o', linewidth=2, markersize=8, label='WMMSE (Imperfect CSI)')
    
    plt.xlabel('Channel Estimation SNR (dB)', fontsize=12)
    plt.ylabel('Sum Spectral Efficiency (bps/Hz)', fontsize=12)
    plt.title('Robustness of WMMSE to Channel Noise', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('wmmse_robustness_only.png', dpi=300)
    print("\nPlot saved to: wmmse_robustness_only.png")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()