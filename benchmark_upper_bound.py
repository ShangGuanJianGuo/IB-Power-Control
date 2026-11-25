"""
WMMSE (Weighted Minimum Mean Square Error) Baseline
用于评估IB模型的性能上限

配置：
- 每个AP功率约束：sum_k p_{lk} <= Pmax
- 与IB模型使用相同的H, W数据
"""

import numpy as np
import torch
import os
from tqdm import tqdm
import argparse

class WMMSE_Optimizer:
    """
    WMMSE算法实现（每AP功率约束版本）
    
    参考：
    Q. Shi et al., "An Iteratively Weighted MMSE Approach to Distributed 
    Sum-Utility Maximization for a MIMO Interfering Broadcast Channel"
    """
    
    def __init__(self, L, K, M, Pmax, noise_power=1.0):
        self.L = L  # 基站数
        self.K = K  # 用户数
        self.M = M  # 天线数
        self.Pmax = Pmax  # 每个AP的功率上限
        self.noise_power = noise_power
        
    def calculate_effective_channel(self, H, W, P):
        """
        计算有效信道矩阵
        H: (L, K, M) - 信道矩阵
        W: (L, K, M) - 预编码矩阵
        P: (L, K) - 功率分配
        
        返回：
        G: (K, K) - 有效信道增益矩阵，G[i,j] = sum_l sqrt(p_{lj}) * h_i^H * w_{lj}
        """
        K = self.K
        L = self.L
        
        # 计算每个用户接收到的信号
        G = np.zeros((K, K), dtype=complex)
        
        for i in range(K):  # 接收用户
            h_i = H[:, i, :].reshape(-1)  # (L*M,) 用户i的信道
            
            for j in range(K):  # 发射用户
                # 计算 sum_l sqrt(p_{lj}) * h_i^H * w_{lj}
                signal = 0
                for l in range(L):
                    w_lj = W[l, j, :]
                    p_lj = P[l, j]
                    signal += np.sqrt(p_lj) * np.vdot(h_i[l*self.M:(l+1)*self.M], w_lj)
                G[i, j] = signal
                
        return G
    
    def calculate_sinr_and_se(self, H, W, P):
        """计算SINR和频谱效率"""
        G = self.calculate_effective_channel(H, W, P)
        G_power = np.abs(G) ** 2
        
        # 信号功率：对角线元素
        signal_power = np.diag(G_power)
        
        # 总接收功率
        total_power = np.sum(G_power, axis=1)
        
        # 干扰功率
        interference_power = total_power - signal_power
        
        # SINR
        sinr = signal_power / (interference_power + self.noise_power)
        
        # SE (bps/Hz)
        se = np.log2(1 + sinr)
        
        return sinr, se, np.sum(se)
    
    def wmmse_iteration(self, H, W, P, max_iter=100, tol=1e-4):
        """
        WMMSE迭代算法
        
        返回：
        P_opt: 最优功率分配
        sum_se: 总频谱效率
        """
        L, K = self.L, self.K
        P_current = P.copy()
        
        prev_sum_se = 0
        
        for iteration in range(max_iter):
            # Step 1: 更新接收滤波器 u_k
            G = self.calculate_effective_channel(H, W, P_current)
            
            u = np.zeros(K, dtype=complex)
            for k in range(K):
                # u_k = g_kk^* / (sum_j |g_kj|^2 + sigma^2)
                denominator = np.sum(np.abs(G[k, :]) ** 2) + self.noise_power
                u[k] = np.conj(G[k, k]) / denominator
            
            # Step 2: 更新MMSE权重 w_k
            w = np.zeros(K)
            for k in range(K):
                # w_k = 1 / (1 - u_k * g_kk)
                mmse = 1 - u[k] * G[k, k]
                w[k] = 1.0 / np.real(mmse)
            
            # Step 3: 更新功率分配（考虑每AP约束）
            P_new = np.zeros((L, K))
            
            for l in range(L):
                # 对于每个AP，优化 sum_k w_k * |u_k * h_{lk}^H * w_{lk}|^2 * p_{lk}
                # s.t. sum_k p_{lk} <= Pmax, p_{lk} >= 0
                
                # 计算每个用户的效用系数
                utility = np.zeros(K)
                
                for k in range(K):
                    h_k = H[:, k, :].reshape(-1)
                    w_lk = W[l, k, :]
                    h_w_lk = np.vdot(h_k[l*self.M:(l+1)*self.M], w_lk)
                    
                    # WMMSE的效用函数系数
                    utility[k] = w[k] * np.abs(u[k] * h_w_lk) ** 2
                
                # 贪婪分配：按效用从大到小排序
                if np.sum(utility) > 1e-10:
                    # 归一化并分配功率
                    P_new[l, :] = (utility / np.sum(utility)) * self.Pmax
                else:
                    # 均匀分配
                    P_new[l, :] = self.Pmax / K
                
                # 确保非负
                P_new[l, :] = np.maximum(P_new[l, :], 0)
            
            # 检查收敛
            _, _, sum_se = self.calculate_sinr_and_se(H, W, P_new)
            
            if abs(sum_se - prev_sum_se) < tol:
                break
                
            P_current = P_new
            prev_sum_se = sum_se
        
        return P_current, sum_se


def evaluate_wmmse_on_dataset(args):
    """在数据集上评估WMMSE性能"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
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
    
    # 使用与训练相同的样本数
    test_samples = min(args.test_samples, H_all.shape[0])
    H_test = H_all[:test_samples]
    W_test = W_all[:test_samples]
    
    print(f"Testing on {test_samples} samples")
    print(f"Data shape: H {H_test.shape}, W {W_test.shape}")
    
    # 初始化WMMSE优化器
    wmmse = WMMSE_Optimizer(args.L, args.K, args.M, args.Pmax, noise_power=1.0)
    
    # 评估
    sum_se_list = []
    
    for idx in tqdm(range(test_samples), desc="WMMSE Optimization"):
        H = H_test[idx]
        W = W_test[idx]
        
        # 初始功率分配（均匀分配）
        P_init = np.ones((args.L, args.K)) * (args.Pmax / args.K)
        
        # WMMSE优化
        P_opt, sum_se = wmmse.wmmse_iteration(H, W, P_init, 
                                              max_iter=args.wmmse_iter,
                                              tol=args.tol)
        
        sum_se_list.append(sum_se)
    
    # 统计结果
    sum_se_array = np.array(sum_se_list)
    mean_se = np.mean(sum_se_array)
    std_se = np.std(sum_se_array)
    
    print("\n" + "="*50)
    print("WMMSE Baseline Results")
    print("="*50)
    print(f"Mean Sum SE: {mean_se:.4f} bps/Hz")
    print(f"Std Sum SE:  {std_se:.4f} bps/Hz")
    print(f"Max Sum SE:  {np.max(sum_se_array):.4f} bps/Hz")
    print(f"Min Sum SE:  {np.min(sum_se_array):.4f} bps/Hz")
    print("="*50)
    
    # 保存结果
    results = {
        'sum_se_list': sum_se_list,
        'mean_se': mean_se,
        'std_se': std_se,
        'args': vars(args)
    }
    
    save_path = f"wmmse_baseline_L{args.L}_K{args.K}_M{args.M}.npy"
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")
    
    return mean_se, std_se


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WMMSE Baseline Evaluation")
    
    # 系统参数（与IB模型完全一致）
    parser.add_argument('--L', type=int, default=40, help='Number of APs')
    parser.add_argument('--K', type=int, default=30, help='Number of UEs')
    parser.add_argument('--M', type=int, default=8, help='Number of antennas per AP')
    parser.add_argument('--Pmax', type=float, default=1000.0, help='Max power per AP (mW)')
    
    # 数据参数
    parser.add_argument('--samples', type=int, default=5000, help='Total samples in dataset')
    parser.add_argument('--test_samples', type=int, default=500, help='Number of samples to test')
    parser.add_argument('--data_dir', type=str, default='.', help='Data directory')
    
    # WMMSE参数
    parser.add_argument('--wmmse_iter', type=int, default=100, help='Max WMMSE iterations')
    parser.add_argument('--tol', type=float, default=1e-4, help='Convergence tolerance')
    
    args = parser.parse_args()
    
    evaluate_wmmse_on_dataset(args)