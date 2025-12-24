"""
Selective State Space Model (Selective SSM).

Core selective scan operation used in Mamba.
The key innovation: input-dependent (selective) state transitions.

Reference: https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import SSMConfig


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model.

    Implements the selective scan algorithm where state transitions
    depend on the input sequence (not fixed like in S4).

    The continuous SSM is:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)

    Discretized with input-dependent delta:
        h_t = Ā h_{t-1} + B̄ x_t
        y_t = C h_t + D x_t

    Where Ā = exp(Δ A), B̄ = Δ B (simplified ZOH discretization).

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Local convolution width
        expand: Expansion factor for inner dimension
        dt_rank: Rank of delta projection ("auto" = ceil(d_model/16))
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Initialization strategy for delta ("random" or "constant")
        dt_scale: Scale for delta initialization
        dt_init_floor: Floor for delta initialization
        bias: Whether to use bias in projections
        conv_bias: Whether to use bias in conv layer

    Example:
        >>> ssm = SelectiveSSM(d_model=768, d_state=16)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y = ssm(x)  # (2, 100, 768)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Inner dimension
        self.d_inner = int(expand * d_model)

        # Delta rank
        if dt_rank == "auto":
            self.dt_rank = max(1, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)

        # Store dt params for initialization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor

        # Input projection: x -> (z, x_proj) where x_proj feeds into conv/SSM
        # z is the gating path
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution for local context
        # Groups = d_inner for depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            padding=d_conv - 1,
        )

        # SSM parameters projections
        # x -> (B, C, delta) projections
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + d_state * 2,  # dt, B, C
            bias=False,
        )

        # Delta projection (low-rank)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for proper dt range
        self._init_dt_proj_bias()

        # A parameter (log space for stability)
        # Shape: (d_inner, d_state)
        self.A_log = self._init_A()

        # D parameter (skip connection)
        self.D = mx.ones((self.d_inner,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _init_A(self) -> mx.array:
        """Initialize A in log space."""
        # A is initialized as -exp(uniform) per the paper
        # Using simple repeat initialization
        A = mx.arange(1, self.d_state + 1, dtype=mx.float32)
        A = mx.broadcast_to(A, (self.d_inner, self.d_state))
        return mx.log(A)

    def _init_dt_proj_bias(self) -> None:
        """Initialize dt_proj bias for proper delta range."""
        # Initialize bias so that softplus(bias) is in [dt_min, dt_max]
        import math

        dt = mx.exp(
            mx.random.uniform(shape=(self.d_inner,))
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = mx.clip(dt, a_min=self.dt_init_floor, a_max=None)
        # Inverse softplus
        inv_dt = dt + mx.log(-mx.expm1(-dt))
        self.dt_proj.bias = inv_dt

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Forward pass through selective SSM.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            cache: Optional (conv_state, ssm_state) for inference

        Returns:
            - Output, shape (batch, seq_len, d_model)
            - Updated cache if provided
        """
        batch, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = mx.split(xz, 2, axis=-1)  # Each (B, L, d_inner)

        # MLX Conv1d expects (B, L, C) format - no transpose needed!

        # Handle cache for inference
        if cache is not None:
            conv_state, ssm_state = cache
            # Prepend conv state along sequence dimension
            x_proj = mx.concatenate([conv_state, x_proj], axis=1)

        # Apply convolution - input is (B, L, d_inner)
        x_conv = self.conv1d(x_proj)

        # Handle caching
        if cache is not None:
            # Keep last d_conv-1 positions for next step
            new_conv_state = x_proj[:, -(self.d_conv - 1) :, :]
            # Trim to seq_len
            x_conv = x_conv[:, :seq_len, :]
        else:
            # Trim causal padding
            x_conv = x_conv[:, :seq_len, :]
            new_conv_state = None

        # Apply activation
        x_conv = nn.silu(x_conv)

        # Project to get dt, B, C
        x_dbc = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = mx.split(
            x_dbc,
            [self.dt_rank, self.dt_rank + self.d_state],
            axis=-1,
        )

        # Project dt to full dimension and apply softplus
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = nn.softplus(dt)  # Ensure positive

        # Get A (negative for stability)
        A = -mx.exp(self.A_log)  # (d_inner, d_state)

        # Run selective scan
        if cache is not None:
            y, new_ssm_state = selective_scan_step(x_conv, dt, A, B, C, self.D, ssm_state)
            new_cache = (new_conv_state, new_ssm_state)
        else:
            y = selective_scan(x_conv, dt, A, B, C, self.D)
            new_cache = None

        # Gate with z
        y = y * nn.silu(z)

        # Output projection
        y = self.out_proj(y)

        return y, new_cache

    @classmethod
    def from_config(cls, config: SSMConfig, d_model: int) -> SelectiveSSM:
        """Create from SSMConfig."""
        return cls(
            d_model=d_model,
            d_state=config.state_size,
            d_conv=config.conv_kernel,
            expand=config.expand,
        )


def selective_scan(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
) -> mx.array:
    """
    Selective scan (parallel implementation for training).

    This is the core Mamba operation: input-dependent state transitions.

    Args:
        x: Input, shape (B, L, D)
        dt: Delta (timestep), shape (B, L, D)
        A: State transition, shape (D, N)
        B: Input projection, shape (B, L, N)
        C: Output projection, shape (B, L, N)
        D: Skip connection, shape (D,)

    Returns:
        Output, shape (B, L, D)
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Discretize A and B
    # dA = exp(dt * A), shape (B, L, D, N)
    dt_expanded = mx.expand_dims(dt, axis=-1)  # (B, L, D, 1)
    A_expanded = mx.expand_dims(A, axis=(0, 1))  # (1, 1, D, N)
    dA = mx.exp(dt_expanded * A_expanded)  # (B, L, D, N)

    # dB = dt * B, need to match dimensions
    # B is (B, L, N), dt is (B, L, D)
    # dB should be (B, L, D, N)
    B_expanded = mx.expand_dims(B, axis=2)  # (B, L, 1, N)
    dB = dt_expanded * B_expanded  # (B, L, D, N)

    # x needs to be (B, L, D, 1) for broadcasting
    x_expanded = mx.expand_dims(x, axis=-1)  # (B, L, D, 1)

    # Input term: dB * x
    dBx = dB * x_expanded  # (B, L, D, N)

    # Sequential scan through time
    # h[t] = dA[t] * h[t-1] + dBx[t]
    # y[t] = C[t] @ h[t]

    # Initialize state
    h = mx.zeros((batch, d_inner, d_state))

    outputs = []
    for t in range(seq_len):
        # Update state
        h = dA[:, t] * h + dBx[:, t]  # (B, D, N)

        # Compute output: C[t] @ h
        # C[:, t] is (B, N), h is (B, D, N)
        # Want (B, D) output
        y_t = mx.sum(h * mx.expand_dims(C[:, t], axis=1), axis=-1)  # (B, D)
        outputs.append(y_t)

    y = mx.stack(outputs, axis=1)  # (B, L, D)

    # Add skip connection
    y = y + D * x

    return y


def selective_scan_step(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    ssm_state: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Single step of selective scan for inference.

    Args:
        x: Input, shape (B, 1, D)
        dt: Delta, shape (B, 1, D)
        A: State transition, shape (D, N)
        B: Input projection, shape (B, 1, N)
        C: Output projection, shape (B, 1, N)
        D: Skip connection, shape (D,)
        ssm_state: Previous state, shape (B, D, N)

    Returns:
        - Output, shape (B, 1, D)
        - New state, shape (B, D, N)
    """
    # Squeeze time dimension
    x = x[:, 0]  # (B, D)
    dt = dt[:, 0]  # (B, D)
    B = B[:, 0]  # (B, N)
    C = C[:, 0]  # (B, N)

    # Discretize
    dt_expanded = mx.expand_dims(dt, axis=-1)  # (B, D, 1)
    A_expanded = mx.expand_dims(A, axis=0)  # (1, D, N)
    dA = mx.exp(dt_expanded * A_expanded)  # (B, D, N)

    B_expanded = mx.expand_dims(B, axis=1)  # (B, 1, N)
    dB = dt_expanded * B_expanded  # (B, D, N)

    x_expanded = mx.expand_dims(x, axis=-1)  # (B, D, 1)
    dBx = dB * x_expanded  # (B, D, N)

    # Update state
    new_state = dA * ssm_state + dBx  # (B, D, N)

    # Compute output
    C_expanded = mx.expand_dims(C, axis=1)  # (B, 1, N)
    y = mx.sum(new_state * C_expanded, axis=-1)  # (B, D)

    # Add skip connection
    y = y + D * x

    # Add time dimension back
    y = mx.expand_dims(y, axis=1)  # (B, 1, D)

    return y, new_state
