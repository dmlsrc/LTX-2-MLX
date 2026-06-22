"""Diffusion stepping strategies for LTX-2 sampling."""

import mlx.core as mx

from LTX_2_MLX.core_utils import to_velocity


class EulerDiffusionStep:
    """
    First-order Euler method for diffusion sampling.

    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying:
        sample = sample + velocity * dt

    where dt = sigma_next - sigma (negative, moving toward less noise).
    """

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        *,
        sigma: float | None = None,
        sigma_next: float | None = None,
    ) -> mx.array:
        """
        Take a single Euler diffusion step.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample from the model.
            sigmas: Full sigma schedule array.
            step_index: Current step index in the schedule.
            sigma, sigma_next: Optional pre-extracted host floats for this
                step.  When provided, the two scalars are not re-read from
                ``sigmas`` (avoids a redundant device->host sync per step on
                the hot path); when ``None`` they are extracted as before.

        Returns:
            Updated sample at the next sigma level, cast back to ``sample``'s
            dtype (typically bf16).

        Math: x_{t+1} = denoised + (sigma_next/sigma) * (x_t - denoised),
        algebraically equivalent to the velocity form ``sample + velocity *
        dt`` but expressed as a single fused fp32 lerp instead of routing
        through ``to_velocity`` + a redundant cast.  No measured speed
        difference under MLX 0.31.2 (lazy graph fusion appears to elide the
        redundancy already), but reads more directly.
        """
        if sigma is None:
            sigma = float(sigmas[step_index])
        if sigma_next is None:
            sigma_next = float(sigmas[step_index + 1])

        sample_dtype = sample.dtype
        sample_f32 = sample if sample.dtype == mx.float32 else sample.astype(mx.float32)
        denoised_f32 = (
            denoised_sample if denoised_sample.dtype == mx.float32
            else denoised_sample.astype(mx.float32)
        )

        result = denoised_f32 + (sigma_next / sigma) * (sample_f32 - denoised_f32)

        if sample_dtype == mx.float32:
            return result
        return result.astype(sample_dtype)


class HeunDiffusionStep:
    """
    Second-order Heun method for diffusion sampling.

    Uses a two-stage predictor-corrector approach for more accurate stepping,
    at the cost of requiring two model evaluations per step.

    Note: This implementation requires a callback for the model evaluation
    at the predicted point. For simpler use cases, prefer EulerDiffusionStep.
    """

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        denoised_at_predicted: mx.array | None = None,
    ) -> mx.array:
        """
        Take a single Heun diffusion step.

        If denoised_at_predicted is not provided, falls back to Euler step.

        Args:
            sample: Current noisy sample.
            denoised_sample: Predicted denoised sample from the model.
            sigmas: Full sigma schedule array.
            step_index: Current step index in the schedule.
            denoised_at_predicted: Optional denoised sample evaluated at the
                predicted point (for the corrector step).

        Returns:
            Updated sample at the next sigma level.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma

        velocity = to_velocity(sample, sigma, denoised_sample)

        # Predictor step (Euler)
        sample_f32 = sample.astype(mx.float32)
        velocity_f32 = velocity.astype(mx.float32)
        predicted = sample_f32 + velocity_f32 * float(dt)

        # If no corrector evaluation provided, return Euler result
        if denoised_at_predicted is None:
            return predicted.astype(sample.dtype)

        # Corrector step: average the velocities
        velocity_at_predicted = to_velocity(
            predicted.astype(sample.dtype), sigma_next, denoised_at_predicted
        )
        velocity_avg = 0.5 * (velocity_f32 + velocity_at_predicted.astype(mx.float32))

        result = sample_f32 + velocity_avg * float(dt)

        return result.astype(sample.dtype)
