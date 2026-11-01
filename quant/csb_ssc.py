"""
Channel-wise Salience Balancing (CSB) and Spearman's rho-guided Salience Calibration (SSC)
Implements Section 4.1, Section 4.2, and Algorithm 1 (Appendix B) of PTQ4DiT (NeurIPS 2024).
"""

import torch


def compute_spearman_rho(x_vec: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
    """
    Computes Spearman's rank correlation coefficient rho(s(X^(t)), s(W))
    across input channels j = 1, ..., d_in (Eq. 11).
    """
    x_flat = x_vec.detach().float().view(-1)
    w_flat = w_vec.detach().float().view(-1)
    n = x_flat.numel()
    if n <= 1:
        return torch.tensor(0.0, device=x_vec.device)

    # Compute ranks via argsort of argsort
    rank_x = torch.argsort(torch.argsort(x_flat)).float()
    rank_w = torch.argsort(torch.argsort(w_flat)).float()

    rank_x = rank_x - rank_x.mean()
    rank_w = rank_w - rank_w.mean()

    denom = torch.sqrt((rank_x ** 2).sum() * (rank_w ** 2).sum()).clamp(min=1e-12)
    rho = (rank_x * rank_w).sum() / denom
    return rho.clamp(-1.0, 1.0)


@torch.no_grad()
def compute_csb_ssc_balancing_vectors(
    act_salience_per_t: torch.Tensor,
    weight_matrix: torch.Tensor,
    use_ssc: bool = True,
    eps: float = 1e-6,
):
    """
    Implements Algorithm 1 (Steps 1-3) of PTQ4DiT:
      Args:
        act_salience_per_t: Tensor of shape (T, d_in) containing s(X^(t)) for each
                            calibration timestep t = 1..T (Eq. 9).
        weight_matrix:      Tensor of shape (d_out, d_in) from nn.Linear.weight.
        use_ssc:            If True, applies Spearman's rho-guided Salience Calibration (Eq. 10-11).
                            If False, uses midpoint/mean salience (CSB-only ablation).
      Returns:
        bx: Tensor of shape (d_in,) representing diag(B_rho^X) (Eq. 6).
        bw: Tensor of shape (d_in,) representing diag(B_rho^W) (Eq. 7).
        eta: Tensor of shape (T,) representing temporal weights eta_t (Eq. 11).
        rhos: Tensor of shape (T,) representing Spearman correlations per timestep.
    """
    device = weight_matrix.device
    # Weight salience s(W_j) = max(|W_j|) along output dimension (Eq. 4 & Eq. 9)
    s_w = weight_matrix.detach().float().abs().amax(dim=0).clamp(min=eps)  # (d_in,)

    act_sal = act_salience_per_t.detach().float().to(device).clamp(min=eps)  # (T, d_in)
    T_steps = act_sal.shape[0]

    rhos = torch.zeros(T_steps, device=device, dtype=torch.float32)
    for t_idx in range(T_steps):
        rhos[t_idx] = compute_spearman_rho(act_sal[t_idx], s_w)

    if use_ssc and T_steps > 1:
        # Eq. 11: eta_t = exp(-rho_t) / sum_tau exp(-rho_tau)
        eta = torch.softmax(-rhos, dim=0)
        # Eq. 10: s_rho(X^(1:T)) = sum_t eta_t * s(X^(t))
        s_x = torch.matmul(eta.unsqueeze(0), act_sal).squeeze(0).clamp(min=eps)
    else:
        eta = torch.full((T_steps,), 1.0 / max(1, T_steps), device=device)
        mid_t = T_steps // 2
        s_x = act_sal[mid_t].clamp(min=eps)

    # Eq. 5 & Eq. 14: Balanced salience s_tilde_rho = (s_rho(X_j) * s(W_j))^(1/2)
    s_tilde = torch.sqrt(s_x * s_w).clamp(min=eps)

    # Eq. 6 & Eq. 7: Diagonal entries of B_rho^X and B_rho^W
    bx = (s_tilde / s_x).clamp(min=1e-4, max=1e4)
    # Enforce exact inverse bw = 1.0 / bx so B_rho^X * B_rho^W = I (Eq. 15)
    bw = 1.0 / bx

    return bx.to(weight_matrix.dtype), bw.to(weight_matrix.dtype), eta, rhos
