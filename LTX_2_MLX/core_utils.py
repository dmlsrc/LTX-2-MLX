"""Utility functions for LTX-2 MLX."""

import mlx.core as mx


def rms_norm(
    x: mx.array, weight: mx.array | None = None, eps: float = 1e-6
) -> mx.array:
    """
    Root-mean-square (RMS) normalize `x` over its last dimension.

    Uses optimized mx.fast.rms_norm Metal kernel for efficiency.

    Args:
        x: Input tensor to normalize.
        weight: Optional learnable scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        RMS normalized tensor.
    """
    return mx.fast.rms_norm(x, weight, eps)


def to_velocity(
    sample: mx.array,
    sigma: float | mx.array,
    denoised_sample: mx.array,
) -> mx.array:
    """
    Convert the sample and its denoised version to velocity.

    Args:
        sample: The noisy sample.
        sigma: The noise level (sigma).
        denoised_sample: The predicted denoised sample.

    Returns:
        Velocity prediction.
    """
    if isinstance(sigma, mx.array):
        sigma_val = sigma.astype(mx.float32)
    else:
        if sigma == 0:
            raise ValueError("Sigma can't be 0.0")
        sigma_val = sigma

    # Compute in float32 for stability
    sample_f32 = sample.astype(mx.float32)
    denoised_f32 = denoised_sample.astype(mx.float32)

    return (sample_f32 - denoised_f32) / sigma_val


