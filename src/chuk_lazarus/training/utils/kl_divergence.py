"""
KL divergence computation for RL training.

Used for:
- DPO implicit reward computation
- PPO KL penalty (optional)
- Monitoring policy drift from reference
"""

import mlx.core as mx


def compute_kl_divergence(
    log_probs_p: mx.array, log_probs_q: mx.array, mask: mx.array = None
) -> mx.array:
    """
    Compute KL divergence: KL(P || Q) = E_P[log P - log Q]

    For language models, this measures how much the policy has
    drifted from a reference distribution.

    Args:
        log_probs_p: Log probs under distribution P (usually current policy)
        log_probs_q: Log probs under distribution Q (usually reference)
        mask: Optional mask for valid tokens

    Returns:
        kl: Scalar KL divergence (averaged over valid tokens)
    """
    kl_per_token = log_probs_p - log_probs_q

    if mask is not None:
        kl_per_token = kl_per_token * mask
        total_tokens = mx.sum(mask)
        kl = mx.sum(kl_per_token) / (total_tokens + 1e-8)
    else:
        kl = mx.mean(kl_per_token)

    return kl


def compute_approx_kl(
    old_log_probs: mx.array, new_log_probs: mx.array, mask: mx.array = None
) -> mx.array:
    """
    Compute approximate KL divergence for PPO early stopping.

    Uses the approximation: KL = 0.5 * E[(log_ratio)^2]
    where log_ratio = new_log_probs - old_log_probs

    This is faster and works well for monitoring drift.

    Args:
        old_log_probs: Log probs from old policy (before update)
        new_log_probs: Log probs from current policy
        mask: Optional mask for valid tokens

    Returns:
        approx_kl: Scalar approximate KL divergence
    """
    log_ratio = new_log_probs - old_log_probs
    approx_kl = 0.5 * mx.mean(log_ratio**2)

    if mask is not None:
        masked_ratio = log_ratio * mask
        total_tokens = mx.sum(mask)
        approx_kl = 0.5 * mx.sum(masked_ratio**2) / (total_tokens + 1e-8)

    return approx_kl
