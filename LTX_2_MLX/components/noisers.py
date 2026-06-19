"""Noise injection strategies for LTX-2 diffusion sampling."""

import mlx.core as mx

from LTX_2_MLX.types import LatentState


class GaussianNoiser:
    """
    Adds Gaussian noise to a latent state, scaled by the denoise mask.

    Two-step blend matching Lightricks' GaussianNoiser: the generative latent is
    noised by ``noise_scale``, then composited onto ``clean_latent`` per the
    denoise mask (mask=1 -> fully noised, mask=0 -> clean_latent preserved). The
    conditioning values therefore come from ``clean_latent``; the noisy ``latent``
    field holds zeros at conditioned positions.
    """

    def __init__(self, key: mx.array | None = None):
        """
        Initialize the Gaussian noiser.

        Args:
            key: Optional MLX random key for reproducibility. If None, uses
                 the global random state.
        """
        self.key = key

    def __call__(
        self, latent_state: LatentState, noise_scale: float = 1.0
    ) -> LatentState:
        """
        Add Gaussian noise to the latent state.

        Args:
            latent_state: The current latent state to add noise to.
            noise_scale: Scale factor for the noise (typically sigma value).

        Returns:
            New LatentState with noise added according to the denoise mask.
        """
        # Generate Gaussian noise
        if self.key is not None:
            noise = mx.random.normal(
                shape=latent_state.latent.shape,
                dtype=latent_state.latent.dtype,
                key=self.key,
            )
            # Update key for next call
            self.key = mx.random.split(self.key)[0]
        else:
            noise = mx.random.normal(
                shape=latent_state.latent.shape,
                dtype=latent_state.latent.dtype,
            )

        # Handle both 2D (B, T) and 3D (B, T, 1) masks from patchification
        mask = latent_state.denoise_mask
        if mask.ndim == 2:
            # Expand mask to broadcast with latent: (B, T) -> (B, T, 1)
            mask = mx.expand_dims(mask, axis=-1)

        # Two-step blend (matches Lightricks GaussianNoiser, computed in float32):
        #   1. noise the generative latent by noise_scale at every position;
        #   2. composite that onto clean_latent per the denoise mask, so the
        #      conditioning values come from clean_latent (not the noisy latent
        #      field, which holds zeros at conditioned positions).
        # For a hard mask (0/1) or noise_scale==1 this equals the old single-step
        # form; it differs only for fractional masks at noise_scale<1 (stage 2).
        base = latent_state.latent.astype(mx.float32)
        clean = latent_state.clean_latent.astype(mx.float32)
        noised = base + noise_scale * (noise.astype(mx.float32) - base)
        latent = clean + mask.astype(mx.float32) * (noised - clean)

        return latent_state.replace(latent=latent.astype(latent_state.latent.dtype))


class DeterministicNoiser:
    """
    A noiser that uses a fixed noise pattern for deterministic generation.

    Useful for reproducible results when debugging or testing.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize with a seed.

        Args:
            seed: Random seed for generating the fixed noise pattern.
        """
        self.seed = seed

    def __call__(
        self, latent_state: LatentState, noise_scale: float = 1.0
    ) -> LatentState:
        """
        Add deterministic noise to the latent state.

        Args:
            latent_state: The current latent state to add noise to.
            noise_scale: Scale factor for the noise.

        Returns:
            New LatentState with deterministic noise added.
        """
        key = mx.random.key(self.seed)
        noise = mx.random.normal(
            shape=latent_state.latent.shape,
            dtype=latent_state.latent.dtype,
            key=key,
        )

        # Handle both 2D (B, T) and 3D (B, T, 1) masks from patchification
        mask = latent_state.denoise_mask
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=-1)
        # Same two-step clean_latent blend as GaussianNoiser (see there).
        base = latent_state.latent.astype(mx.float32)
        clean = latent_state.clean_latent.astype(mx.float32)
        noised = base + noise_scale * (noise.astype(mx.float32) - base)
        latent = clean + mask.astype(mx.float32) * (noised - clean)

        return latent_state.replace(latent=latent.astype(latent_state.latent.dtype))
