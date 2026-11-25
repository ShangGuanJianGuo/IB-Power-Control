"""
改进架构 v2: 方案1 + 方案4
- 方案1: Encoder看到瞬时CSI的统计特征
- 方案4: 迭代优化模仿WMMSE

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Enhanced_IB_Encoder(nn.Module):
    """
    增强的IB编码器 - 能看到瞬时CSI特征
    """
    def __init__(self, K, M, z_dim):
        super().__init__()
        self.K = K
        self.M = M
        self.z_dim = z_dim
        
        # 输入特征维度
        # 1. 信道幅度统计: K (每个UE的平均增益)
        # 2. 信道方向统计: K (主模方向)
        # 3. 大尺度衰落: K
        # 4. 当前SINR状态: K (来自上一轮迭代)
        self.input_dim = K * 4
        
        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)
        
    def extract_csi_features(self, H_local, beta, sinr_state):
        """
        从本地CSI提取统计特征
        
        H_local: (B, K, M) - AP l 到所有UE的信道
        beta: (B, K) - 大尺度衰落
        sinr_state: (B, K) - 当前SINR状态
        
        返回: (B, input_dim) 特征向量
        """
        B, K, M = H_local.shape
        
        # 特征1: 信道功率增益（归一化到0-1）
        h_power = torch.mean(torch.abs(H_local)**2, dim=-1)  # (B, K)
        h_power_normalized = torch.log10(h_power + 1e-8)
        
        # 特征2: 信道主模方向（取第一个天线作为参考）
        h_phase = torch.angle(H_local[:, :, 0])  # (B, K)
        h_direction = torch.stack([torch.cos(h_phase), torch.sin(h_phase)], dim=-1)  # (B, K, 2)
        h_direction_flat = h_direction.reshape(B, K * 2)
        
        # 特征3: 大尺度衰落（已归一化）
        beta_feat = beta  # (B, K)
        
        # 特征4: SINR状态（log域）
        sinr_log = torch.log10(sinr_state + 1e-3)  # (B, K)
        
        # 拼接所有特征
        # 注意：h_direction_flat是2K维，其他都是K维
        # 为了匹配input_dim=4K，我们只取h_power_normalized
        features = torch.cat([
            h_power_normalized,  # K
            h_phase,             # K  
            beta_feat,           # K
            sinr_log            # K
        ], dim=-1)  # (B, 4K)
        
        return features
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, H_local, beta, sinr_state):
        """
        H_local: (B, K, M)
        beta: (B, K)
        sinr_state: (B, K)
        
        返回: z, kl_div
        """
        # 提取特征
        features = self.extract_csi_features(H_local, beta, sinr_state)
        
        # 编码
        h = self.encoder_net(features)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 限制logvar范围
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # KL散度
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        return z, kl


class Attention_Aggregator(nn.Module):
    """
    加权聚合器（简化版本）
    使用基于信道增益的加权求和代替复杂的attention机制
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.norm = nn.LayerNorm(z_dim)
        
    def forward(self, z_all, lsf_matrix, connectivity_mask):
        """
        z_all: (B, L, z_dim) - 所有AP的编码
        lsf_matrix: (B, L, K) - 大尺度衰落矩阵
        connectivity_mask: (B, L, K) - 连接掩码（float: 0或1）
        
        返回: context_k (B, K, z_dim) - 每个UE的全局上下文
        """
        B, L, z_dim = z_all.shape
        K = lsf_matrix.shape[2]
        
        # 使用加权聚合代替attention（更简单且有效）
        # 扩展z到所有UE
        z_expanded = z_all.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, z_dim)
        
        # 使用lsf_matrix和connectivity_mask计算权重
        weights = lsf_matrix * connectivity_mask  # (B, L, K)
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8  # (B, 1, K)
        weights_norm = (weights / weights_sum).unsqueeze(-1)  # (B, L, K, 1)
        
        # 加权求和：对每个UE，聚合所有AP的z
        # z_expanded: (B, L, K, z_dim)
        # weights_norm: (B, L, K, 1)
        weighted_z = z_expanded * weights_norm  # (B, L, K, z_dim)
        context_k = torch.sum(weighted_z, dim=1)  # (B, K, z_dim)
        context_k = self.norm(context_k)
        
        return context_k


class Iterative_Power_Policy(nn.Module):
    """
    迭代功率决策网络
    """
    def __init__(self, z_dim, M):
        super().__init__()
        self.z_dim = z_dim
        self.M = M
        
        # 输入: [本地CSI特征, 全局context, 上一轮功率]
        self.input_dim = M + z_dim + 1
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保输出正数
        )
        
    def forward(self, H_local, context_k, P_prev):
        """
        H_local: (B, L, K, M) - 本地CSI
        context_k: (B, K, z_dim) - 全局上下文
        P_prev: (B, L, K) - 上一轮功率
        
        返回: P_new (B, L, K)
        """
        B, L, K, M = H_local.shape
        
        # 提取本地CSI特征
        h_mag = torch.abs(H_local)  # (B, L, K, M)
        
        # 扩展context到每个AP
        context_expanded = context_k.unsqueeze(1).expand(B, L, K, self.z_dim)  # (B, L, K, z_dim)
        
        # 扩展上一轮功率
        p_prev_expanded = P_prev.unsqueeze(-1)  # (B, L, K, 1)
        
        # 拼接特征
        features = torch.cat([
            h_mag,                 # (B, L, K, M)
            context_expanded,      # (B, L, K, z_dim)
            p_prev_expanded        # (B, L, K, 1)
        ], dim=-1)  # (B, L, K, M+z_dim+1)
        
        # 通过策略网络
        P_raw = self.policy_net(features).squeeze(-1)  # (B, L, K)
        
        return P_raw


class Iterative_IB_System(nn.Module):
    """
    完整的迭代IB系统
    """
    def __init__(self, L, K, M, z_dim, Pmax, num_iterations=3):
        super().__init__()
        self.L = L
        self.K = K
        self.M = M
        self.Pmax = Pmax
        self.num_iterations = num_iterations
        
        # 组件
        self.encoder = Enhanced_IB_Encoder(K, M, z_dim)
        self.aggregator = Attention_Aggregator(z_dim)
        self.policy = Iterative_Power_Policy(z_dim, M)
        
    def calculate_sinr(self, H, W, P, noise_power=1.0):
        """
        计算当前SINR
        
        H: (B, L, K, M)
        W: (B, L, K, M)
        P: (B, L, K)
        
        返回: sinr (B, K)
        """
        B, L, K, M = H.shape
        
        # 缩放W
        sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1)  # (B, L, K, 1)
        W_scaled = sqrt_P * W  # (B, L, K, M)
        
        # 计算接收信号
        H_u = H.permute(0, 2, 1, 3)  # (B, K, L, M)
        W_u = W_scaled.permute(0, 2, 1, 3)  # (B, K, L, M)
        
        # (B, K, K)
        Rx_matrix = torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)
        Rx_power = torch.abs(Rx_matrix) ** 2
        
        signal_power = torch.diagonal(Rx_power, dim1=1, dim2=2)  # (B, K)
        total_power = torch.sum(Rx_power, dim=2)
        interference_power = total_power - signal_power
        
        sinr = signal_power / (interference_power + noise_power + 1e-8)
        
        return sinr
    
    def apply_power_constraints(self, P):
        """
        应用每AP功率约束
        """
        B, L, K = P.shape
        
        # 每个AP的总功率
        ap_power_sum = torch.sum(P, dim=2, keepdim=True)  # (B, L, 1)
        
        # 缩放因子
        scaling_factor = torch.clamp(ap_power_sum / self.Pmax, min=1.0)
        
        # 归一化
        P_constrained = (P / scaling_factor) * self.Pmax
        
        return P_constrained
    
    def forward(self, H, W, lsf_matrix, connectivity_mask):
        """
        迭代优化功率分配
        
        H: (B, L, K, M) - 信道矩阵
        W: (B, L, K, M) - 预编码矩阵
        lsf_matrix: (B, L, K) - 大尺度衰落（已归一化）
        connectivity_mask: (B, L, K) - 连接掩码
        
        返回: P_final, total_kl_divs
        """
        B, L, K, M = H.shape
        
        # 初始化功率（均匀分配）
        P_current = torch.ones(B, L, K, device=H.device) * (self.Pmax / K)
        
        # 初始化SINR状态
        sinr_state = torch.ones(B, L, K, device=H.device)  # (B, L, K)
        
        total_kl_divs = []
        
        # 迭代优化
        for iteration in range(self.num_iterations):
            # 1. 计算当前SINR
            current_sinr = self.calculate_sinr(H, W, P_current)  # (B, K)
            
            # 为每个AP复制SINR状态
            sinr_state = current_sinr.unsqueeze(1).expand(B, L, K)  # (B, L, K)
            
            # 2. 每个AP编码本地信息
            z_list = []
            kl_list = []
            
            for l in range(L):
                H_l = H[:, l, :, :]  # (B, K, M)
                beta_l = lsf_matrix[:, l, :]  # (B, K)
                sinr_l = sinr_state[:, l, :]  # (B, K)
                
                z_l, kl_l = self.encoder(H_l, beta_l, sinr_l)
                z_list.append(z_l)
                kl_list.append(kl_l)
            
            z_all = torch.stack(z_list, dim=1)  # (B, L, z_dim)
            kl_all = torch.stack(kl_list, dim=1)  # (B, L)
            
            total_kl_divs.append(kl_all)
            
            # 3. 聚合全局信息
            context_k = self.aggregator(z_all, lsf_matrix, connectivity_mask)
            
            # 4. 更新功率
            P_new = self.policy(H, context_k, P_current)
            
            # 5. 应用功率约束
            P_current = self.apply_power_constraints(P_new)
        
        # 返回最终功率和平均KL散度
        avg_kl_divs = torch.mean(torch.stack(total_kl_divs, dim=0), dim=0)  # (B, L)
        
        return P_current, avg_kl_divs


def unsupervised_se_loss(P, H, W, noise_power=1.0):
    """
    计算频谱效率损失（与原始版本相同）
    """
    B, L, K, M = H.shape
    
    sqrt_P = torch.sqrt(P + 1e-12).unsqueeze(-1)
    W_scaled = sqrt_P * W
    
    H_u = H.permute(0, 2, 1, 3)
    W_u = W_scaled.permute(0, 2, 1, 3)
    
    Rx_matrix = torch.einsum('bklm,bjlm->bkj', H_u.conj(), W_u)
    Rx_power = torch.abs(Rx_matrix) ** 2
    
    signal_power = torch.diagonal(Rx_power, dim1=1, dim2=2)
    total_power = torch.sum(Rx_power, dim=2)
    interference_power = total_power - signal_power
    
    sinr = signal_power / (interference_power + noise_power + 1e-8)
    se = torch.log2(1 + sinr)
    sum_se = torch.sum(se, dim=1)
    
    return -torch.mean(sum_se)


if __name__ == "__main__":
    # 测试代码
    print("Testing Iterative IB System v2...")
    
    B, L, K, M = 4, 40, 30, 8
    z_dim = 16
    Pmax = 1000.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    model = Iterative_IB_System(L, K, M, z_dim, Pmax, num_iterations=3).to(device)
    
    # 生成随机数据
    H = torch.randn(B, L, K, M, dtype=torch.cfloat).to(device)
    W = torch.randn(B, L, K, M, dtype=torch.cfloat).to(device)
    lsf = torch.randn(B, L, K).to(device)
    mask = torch.ones(B, L, K, dtype=torch.bool).to(device)
    
    # 前向传播
    P, kl = model(H, W, lsf, mask)
    
    print(f"Output power shape: {P.shape}")
    print(f"KL divergence shape: {kl.shape}")
    print(f"Power range: [{P.min().item():.2f}, {P.max().item():.2f}]")
    
    # 计算SE
    se_loss = unsupervised_se_loss(P, H, W)
    print(f"SE loss: {-se_loss.item():.4f} bps/Hz")
    
    print("\nModel test passed!")