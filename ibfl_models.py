from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p
        - logvar_q
        + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8)
        - 1.0
    )
    return kl.sum(dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked_logits = logits.masked_fill(mask <= 0, -1e4)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask
    return weights / (weights.sum(dim=dim, keepdim=True) + 1e-8)


class SuccessiveRefinementIBEncoder(nn.Module):
    def __init__(self, K: int, M: int, z_dim: int, message_rounds: int):
        super().__init__()
        self.K = K
        self.M = M
        self.z_dim = z_dim
        self.message_rounds = message_rounds

        self.obs_dim = 5 * K
        self.slow_dim = 2 * K + 4
        post_dim = self.obs_dim + self.slow_dim + z_dim
        prior_dim = self.slow_dim + z_dim

        self.posterior_nets = nn.ModuleList()
        self.posterior_mu = nn.ModuleList()
        self.posterior_logvar = nn.ModuleList()
        self.posterior_gate = nn.ModuleList()
        self.prior_nets = nn.ModuleList()
        self.prior_mu = nn.ModuleList()
        self.prior_logvar = nn.ModuleList()

        for _ in range(message_rounds):
            self.posterior_nets.append(
                nn.Sequential(
                    nn.Linear(post_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                )
            )
            self.posterior_mu.append(nn.Linear(128, z_dim))
            self.posterior_logvar.append(nn.Linear(128, z_dim))
            self.posterior_gate.append(nn.Linear(128, z_dim))

            self.prior_nets.append(
                nn.Sequential(
                    nn.Linear(prior_dim, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                )
            )
            self.prior_mu.append(nn.Linear(128, z_dim))
            self.prior_logvar.append(nn.Linear(128, z_dim))

        self.output_norm = nn.LayerNorm(z_dim)

    def _observation_features(self, h_local, beta_local, mask_local, p_prev_local, sinr_state):
        h_power = torch.log1p(torch.mean(torch.abs(h_local) ** 2, dim=-1))
        h_phase = torch.angle(h_local[:, :, 0]) / torch.pi
        beta_feat = beta_local
        power_feat = torch.log1p(p_prev_local)
        sinr_feat = torch.log1p(sinr_state)
        return torch.cat([h_power, h_phase, beta_feat, power_feat, sinr_feat], dim=-1)

    def _slow_context(self, beta_local, mask_local, p_prev_local, sinr_state):
        beta_mean = beta_local.mean(dim=-1, keepdim=True)
        beta_std = beta_local.std(dim=-1, keepdim=True)
        mask_density = mask_local.mean(dim=-1, keepdim=True)
        power_mean = torch.log1p(p_prev_local.mean(dim=-1, keepdim=True))
        return torch.cat([beta_local, mask_local, beta_mean, beta_std, mask_density, power_mean], dim=-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_local, beta_local, mask_local, p_prev_local, sinr_state, deterministic: bool = False):
        obs_feat = self._observation_features(h_local, beta_local, mask_local, p_prev_local, sinr_state)
        slow_feat = self._slow_context(beta_local, mask_local, p_prev_local, sinr_state)

        z_state = torch.zeros(h_local.shape[0], self.z_dim, device=h_local.device)
        total_kl = torch.zeros(h_local.shape[0], device=h_local.device)

        for round_idx in range(self.message_rounds):
            posterior_input = torch.cat([obs_feat, slow_feat, z_state], dim=-1)
            posterior_hidden = self.posterior_nets[round_idx](posterior_input)
            mu_q = self.posterior_mu[round_idx](posterior_hidden)
            logvar_q = torch.clamp(self.posterior_logvar[round_idx](posterior_hidden), min=-8.0, max=8.0)
            gate = torch.sigmoid(self.posterior_gate[round_idx](posterior_hidden))

            prior_input = torch.cat([slow_feat, z_state], dim=-1)
            prior_hidden = self.prior_nets[round_idx](prior_input)
            mu_p = self.prior_mu[round_idx](prior_hidden)
            logvar_p = torch.clamp(self.prior_logvar[round_idx](prior_hidden), min=-8.0, max=8.0)

            z_delta = mu_q if deterministic else self.reparameterize(mu_q, logvar_q)
            z_state = z_state + gate * z_delta
            total_kl = total_kl + gaussian_kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        return self.output_norm(z_state), total_kl / self.message_rounds


class UserContextAggregator(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, z_all: torch.Tensor, beta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = masked_softmax(beta, mask, dim=1).unsqueeze(-1)
        context = torch.sum(z_all.unsqueeze(2) * weights, dim=1)
        return self.norm(context)


class PersonalizedContextAdapter(nn.Module):
    def __init__(self, z_dim: int, num_clients: int):
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(z_dim, z_dim),
                    nn.ReLU(),
                    nn.Linear(z_dim, z_dim),
                )
                for _ in range(num_clients)
            ]
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, context: torch.Tensor, client_ids: torch.Tensor) -> torch.Tensor:
        adapted = context.clone()
        for client_id in torch.unique(client_ids).tolist():
            index_mask = client_ids == client_id
            adapted[index_mask] = adapted[index_mask] + self.adapters[client_id](adapted[index_mask])
        return self.norm(adapted)


class PowerPolicyHead(nn.Module):
    def __init__(self, M: int, z_dim: int):
        super().__init__()
        input_dim = M + z_dim + 4
        self.policy = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(128, 64)),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, h, beta, mask, context, p_prev, sinr_state):
        batch_size, num_aps, num_users, num_antennas = h.shape
        context_expanded = context.unsqueeze(1).expand(batch_size, num_aps, num_users, context.shape[-1])
        sinr_expanded = torch.log1p(sinr_state).unsqueeze(1).unsqueeze(-1).expand(batch_size, num_aps, num_users, 1)
        features = torch.cat(
            [
                torch.abs(h),
                context_expanded,
                beta.unsqueeze(-1),
                torch.log1p(p_prev).unsqueeze(-1),
                sinr_expanded,
                mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.policy(features).squeeze(-1)


class TaskOrientedIBFederatedController(nn.Module):
    def __init__(
        self,
        L: int,
        K: int,
        M: int,
        z_dim: int,
        p_max: float,
        num_clients: int,
        num_power_iterations: int = 3,
        message_rounds: int = 2,
        damping: float = 0.6,
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.M = M
        self.z_dim = z_dim
        self.p_max = p_max
        self.num_clients = num_clients
        self.num_power_iterations = num_power_iterations
        self.message_rounds = message_rounds
        self.damping = damping

        self.encoder = SuccessiveRefinementIBEncoder(K, M, z_dim, message_rounds)
        self.aggregator = UserContextAggregator(z_dim)
        self.personalized_adapter = PersonalizedContextAdapter(z_dim, num_clients)
        self.policy_head = PowerPolicyHead(M, z_dim)

    def initialize_power(self, mask: torch.Tensor) -> torch.Tensor:
        active_users = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        return mask * (self.p_max / active_users)

    def apply_power_constraints(self, p_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p_masked = p_raw * mask
        ap_total = p_masked.sum(dim=2, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(ap_total / self.p_max, min=1.0)
        return p_masked / scale

    def calculate_sinr(self, h: torch.Tensor, w: torch.Tensor, p: torch.Tensor, noise_power: float = 1.0) -> torch.Tensor:
        sqrt_p = torch.sqrt(p + 1e-12).unsqueeze(-1)
        w_scaled = sqrt_p * w
        h_u = h.permute(0, 2, 1, 3)
        w_u = w_scaled.permute(0, 2, 1, 3)
        rx_matrix = torch.einsum("bklm,bjlm->bkj", h_u.conj(), w_u)
        rx_power = torch.abs(rx_matrix) ** 2
        signal = torch.diagonal(rx_power, dim1=1, dim2=2)
        total = rx_power.sum(dim=2)
        interference = total - signal
        return signal / (interference + noise_power + 1e-8)

    def compute_rates(self, h: torch.Tensor, w: torch.Tensor, p: torch.Tensor, noise_power: float = 1.0) -> torch.Tensor:
        sinr = self.calculate_sinr(h, w, p, noise_power=noise_power)
        return torch.log2(1.0 + sinr)

    def set_personalized_trainable(self, active_client: int):
        for client_id, adapter in enumerate(self.personalized_adapter.adapters):
            requires_grad = client_id == active_client
            for param in adapter.parameters():
                param.requires_grad = requires_grad

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if not key.startswith("personalized_adapter.adapters."):
                state[key] = value.detach().clone()
        return state

    def load_shared_state_dict(self, shared_state: Dict[str, torch.Tensor]):
        current_state = self.state_dict()
        current_state.update(shared_state)
        self.load_state_dict(current_state)

    def get_adapter_state(self, client_id: int) -> Dict[str, torch.Tensor]:
        prefix = f"personalized_adapter.adapters.{client_id}."
        return {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefix)
        }

    def load_adapter_state(self, client_id: int, adapter_state: Dict[str, torch.Tensor]):
        current_state = self.state_dict()
        current_state.update(adapter_state)
        self.load_state_dict(current_state)

    def forward(self, h, w, beta, mask, client_ids, deterministic: bool = False):
        p_current = self.initialize_power(mask)
        total_kl = torch.zeros(h.shape[0], device=h.device)

        for _ in range(self.num_power_iterations):
            sinr_state = self.calculate_sinr(h, w, p_current)
            z_list = []
            kl_list = []

            for ap_idx in range(self.L):
                z_ap, kl_ap = self.encoder(
                    h[:, ap_idx, :, :],
                    beta[:, ap_idx, :],
                    mask[:, ap_idx, :],
                    p_current[:, ap_idx, :],
                    sinr_state,
                    deterministic=deterministic,
                )
                z_list.append(z_ap)
                kl_list.append(kl_ap)

            z_all = torch.stack(z_list, dim=1)
            kl_all = torch.stack(kl_list, dim=1)
            total_kl = total_kl + kl_all.mean(dim=1)

            context = self.aggregator(z_all, beta, mask)
            context = self.personalized_adapter(context, client_ids)

            p_candidate = self.policy_head(h, beta, mask, context, p_current, sinr_state)
            p_projected = self.apply_power_constraints(p_candidate, mask)
            p_current = (1.0 - self.damping) * p_current + self.damping * p_projected

        rates = self.compute_rates(h, w, p_current)
        return p_current, total_kl / self.num_power_iterations, rates






@torch.no_grad()
def summarize_rate_metrics(rates: torch.Tensor) -> Dict[str, float]:
    return {
        "sum_rate": float(rates.sum(dim=1).mean().cpu()),
        "mean_rate": float(rates.mean().cpu()),
        "min_rate": float(rates.min(dim=1).values.mean().cpu()),
        "p05_rate": float(torch.quantile(rates, q=0.05).cpu()),
    }
