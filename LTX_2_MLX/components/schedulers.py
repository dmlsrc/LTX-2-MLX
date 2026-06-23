"""Sigma schedule generators for LTX-2 diffusion sampling."""

import math

import mlx.core as mx

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    """

    def execute(
        self,
        steps: int,
        latent: mx.array | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> mx.array:
        """
        Generate sigma schedule.

        Args:
            steps: Number of denoising steps.
            latent: Optional latent tensor to compute token count from shape.
            max_shift: Maximum shift value for large token counts.
            base_shift: Base shift value for small token counts.
            stretch: Whether to stretch sigmas to match terminal value.
            terminal: Target terminal sigma value.

        Returns:
            Sigma schedule as an MLX array of shape (steps + 1,).
        """
        # Compute token count from latent shape or use default
        if latent is not None:
            tokens = math.prod(latent.shape[2:])
        else:
            tokens = MAX_SHIFT_ANCHOR

        # Linear spacing from 1.0 to 0.0
        sigmas = mx.linspace(1.0, 0.0, steps + 1)

        # Compute shift based on token count (linear interpolation)
        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        # Apply sigmoid-like transformation
        power = 1
        exp_shift = math.exp(sigma_shift)

        # Avoid division by zero for sigmas == 0
        # sigmas_transformed = exp_shift / (exp_shift + (1/sigmas - 1)^power)
        sigmas_transformed = mx.where(
            sigmas != 0,
            exp_shift / (exp_shift + mx.power(1.0 / sigmas - 1.0, power)),
            mx.zeros_like(sigmas),
        )

        sigmas = sigmas_transformed

        # Stretch sigmas so that final non-zero value matches terminal
        if stretch and steps > 0:
            # The last sigma is always 0 after transformation, so last non-zero is at index steps-1
            one_minus_sigmas = 1.0 - sigmas

            # Get the last non-zero sigma's (1 - sigma) value (second to last element)
            last_one_minus = float(one_minus_sigmas[steps - 1])

            # Scale factor to stretch to terminal
            scale_factor = last_one_minus / (1.0 - terminal)

            # Apply stretching: new_sigma = 1 - (1 - sigma) / scale_factor
            stretched = 1.0 - (one_minus_sigmas / scale_factor)

            # Only apply to non-zero positions (all except the last element which is 0)
            non_zero_mask = sigmas != 0
            sigmas = mx.where(non_zero_mask, stretched, sigmas)

        return sigmas.astype(mx.float32)




# Predefined sigma values for distilled models
# Official distilled sigma schedule from LTX-2 ComfyUI (9 values for 8 steps)
DISTILLED_SIGMA_VALUES = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

STAGE_2_DISTILLED_SIGMA_VALUES = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]


def get_sigma_schedule(
    num_steps: int,
    distilled: bool = False,
    latent: mx.array | None = None,
) -> mx.array:
    """
    Get sigma schedule for diffusion sampling.

    Convenience function to get the appropriate sigma schedule.

    Args:
        num_steps: Number of denoising steps.
        distilled: If True, use predefined distilled sigma values.
        latent: Optional latent for token-count-dependent scheduling.

    Returns:
        Sigma schedule as MLX array.
    """
    if distilled:
        return mx.array(DISTILLED_SIGMA_VALUES, dtype=mx.float32)

    scheduler = LTX2Scheduler()
    return scheduler.execute(steps=num_steps, latent=latent)
