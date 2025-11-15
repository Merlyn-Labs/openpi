"""Loss utilities for Pi0 model to address action group imbalance."""

import jax.numpy as jnp
from openpi.shared import array_typing as at


def _delta_action_weights(actions, *, alpha=0.2, beta=1.0, cap=0.5):
    """Original delta action weights based on overall action magnitude.
    
    actions: [B, H, D] absolute commands
    returns: w in [B, H] with mean ≈ 1
    """
    da = actions[:, 1:, :] - actions[:, :-1, :]
    da_mag = jnp.linalg.norm(da, axis=-1)                 # [B, H-1]
    da_mag = jnp.pad(da_mag, ((0, 0), (1, 0)))            # align to [B, H]; first step has no Δ → 0
    w = alpha + beta * jnp.minimum(da_mag, cap)           # gentle slope + cap
    w = jnp.clip(w, 0.1, 3.0)                             # guardrails
    w = w / (jnp.mean(w) + 1e-8)                          # mean ≈ 1 per batch
    return w                                              # [B, H]


def _per_dimension_group_weights(
    actions: at.Float[at.Array, "b h d"],
    *,
    action_groups: dict[str, tuple[int, int]] | None = None,
    group_weights: dict[str, float] | None = None,
    use_delta_weighting: bool = True,
    alpha: float = 0.2,
    beta: float = 1.0,
    cap: float = 0.5,
) -> at.Float[at.Array, "b h d"]:
    """Compute per-dimension weights that balance different action groups.
    
    This addresses the issue where large navigation movements dominate the loss,
    preventing the model from learning arm control.
    
    Args:
        actions: [B, H, D] absolute action commands
        action_groups: Dictionary mapping group names to (start_idx, end_idx).
            Example for B1K robot:
            {
                "base": (0, 3),      # base velocity
                "trunk": (3, 7),     # trunk position  
                "left_arm": (7, 14), # left arm joints
                "right_arm": (14, 21), # right arm joints
                "grippers": (21, 23), # gripper widths
            }
        group_weights: Relative importance of each group (will be normalized).
            Example: {"base": 1.0, "trunk": 1.5, "left_arm": 2.0, "right_arm": 2.0, "grippers": 1.5}
        use_delta_weighting: If True, also apply delta-based weighting within each group.
        alpha, beta, cap: Parameters for delta weighting.
        
    Returns:
        weights: [B, H, D] per-dimension weights
    """
    batch_size, horizon, action_dim = actions.shape
    
    # Default action groups for B1K robot (23 dims)
    if action_groups is None:
        action_groups = {
            "base": (0, 3),
            "trunk": (3, 7),
            "left_arm": (7, 14),
            "right_arm": (14, 21),
            "grippers": (21, 23),
        }
    
    # Default group weights: emphasize arms more
    if group_weights is None:
        group_weights = {
            "base": 1.0,
            "trunk": 1.5,
            "left_arm": 2.5,  # Increase arm importance
            "right_arm": 2.5,  # Increase arm importance
            "grippers": 2.0,
        }
    
    # Initialize weights
    weights = jnp.ones((batch_size, horizon, action_dim))
    
    # Compute delta actions for delta weighting
    if use_delta_weighting:
        da = actions[:, 1:, :] - actions[:, :-1, :]
        da = jnp.pad(da, ((0, 0), (1, 0), (0, 0)))  # [B, H, D]
    
    # Apply weights per group
    for group_name, (start_idx, end_idx) in action_groups.items():
        # Skip if group is outside action dimensions (for padded actions)
        if start_idx >= action_dim:
            continue
        end_idx = min(end_idx, action_dim)
        
        base_weight = group_weights.get(group_name, 1.0)
        
        if use_delta_weighting:
            # Compute delta magnitude for this group
            group_da = da[:, :, start_idx:end_idx]
            group_da_mag = jnp.linalg.norm(group_da, axis=-1, keepdims=True)  # [B, H, 1]
            
            # Apply delta weighting
            group_w = alpha + beta * jnp.minimum(group_da_mag, cap)
            group_w = jnp.clip(group_w, 0.1, 3.0)
            
            # Normalize within group and apply base weight
            group_w = group_w / (jnp.mean(group_w) + 1e-8)
            group_w = group_w * base_weight
            
            # Broadcast to all dimensions in group
            group_w = jnp.broadcast_to(group_w, (batch_size, horizon, end_idx - start_idx))
        else:
            group_w = jnp.full((batch_size, horizon, end_idx - start_idx), base_weight)
        
        weights = weights.at[:, :, start_idx:end_idx].set(group_w)
    
    # Normalize so mean weight is 1.0
    weights = weights / (jnp.mean(weights) + 1e-8)
    
    return weights


def _action_dimension_mask_weights(
    actions: at.Float[at.Array, "b h d"],
    *,
    dimension_weights: at.Float[at.Array, "d"] | None = None,
) -> at.Float[at.Array, "b h d"]:
    """Apply fixed per-dimension weights (simplest approach).
    
    Args:
        actions: [B, H, D] absolute action commands
        dimension_weights: [D] per-dimension weight multipliers.
            Example for 32-dim action space:
            - dims 0-2 (base): 1.0
            - dims 3-6 (trunk): 1.5
            - dims 7-20 (arms): 2.5
            - dims 21-22 (grippers): 2.0
            - dims 23-31 (padding): 0.0
    
    Returns:
        weights: [B, H, D] per-dimension weights
    """
    if dimension_weights is None:
        # Default weights for B1K (32-dim padded action space)
        dimension_weights = jnp.array([
            1.0, 1.0, 1.0,  # base (0-2)
            1.5, 1.5, 1.5, 1.5,  # trunk (3-6)
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,  # left arm (7-13)
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,  # right arm (14-20)
            2.0, 2.0,  # grippers (21-22)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # padding (23-31)
        ])
    
    # Broadcast to [B, H, D]
    weights = jnp.broadcast_to(dimension_weights, actions.shape)
    
    # Normalize
    weights = weights / (jnp.mean(weights) + 1e-8)
    
    return weights


def compute_weighted_loss(
    base_loss: at.Float[at.Array, "b h d"],
    actions: at.Float[at.Array, "b h d"],
    *,
    weighting_strategy: str = "per_group",
    action_groups: dict[str, tuple[int, int]] | None = None,
    group_weights: dict[str, float] | None = None,
    dimension_weights: at.Float[at.Array, "d"] | None = None,
    use_delta_weighting: bool = True,
) -> at.Float[at.Array, "b h"]:
    """Compute weighted loss with configurable weighting strategies.
    
    Args:
        base_loss: [B, H, D] per-dimension losses
        actions: [B, H, D] action values
        weighting_strategy: One of:
            - "original": Original L2-norm-based weighting (navigation-biased)
            - "per_group": Per-group weighting with optional delta (recommended)
            - "per_dimension": Fixed per-dimension weights (simple)
            - "uniform": No weighting (mean across all dimensions)
        action_groups: For "per_group" strategy
        group_weights: For "per_group" strategy
        dimension_weights: For "per_dimension" strategy
        use_delta_weighting: For "per_group" strategy
        
    Returns:
        loss: [B, H] weighted loss per timestep
    """
    if weighting_strategy == "original":
        # Original approach: weight by overall action magnitude
        temporal_weights = _delta_action_weights(actions)  # [B, H]
        return temporal_weights * jnp.mean(base_loss, axis=-1)  # [B, H]
    
    elif weighting_strategy == "per_group":
        # Recommended: weight different action groups differently
        dim_weights = _per_dimension_group_weights(
            actions,
            action_groups=action_groups,
            group_weights=group_weights,
            use_delta_weighting=use_delta_weighting,
        )  # [B, H, D]
        return jnp.mean(base_loss * dim_weights, axis=-1)  # [B, H]
    
    elif weighting_strategy == "per_dimension":
        # Simple: fixed weights per dimension
        dim_weights = _action_dimension_mask_weights(
            actions,
            dimension_weights=dimension_weights,
        )  # [B, H, D]
        return jnp.mean(base_loss * dim_weights, axis=-1)  # [B, H]
    
    elif weighting_strategy == "uniform":
        # No weighting: treat all dimensions equally
        return jnp.mean(base_loss, axis=-1)  # [B, H]
    
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

