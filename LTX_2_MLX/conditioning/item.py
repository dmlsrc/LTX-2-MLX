"""Conditioning item protocol for modifying latent state during diffusion."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from LTX_2_MLX.conditioning.tools import VideoLatentTools
    from LTX_2_MLX.types import LatentState


class ConditioningItem(Protocol):
    """Protocol for conditioning items that modify latent state during diffusion."""

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """
        Apply the conditioning to the latent state.

        Args:
            latent_state: The latent state to apply the conditioning to.
                         This state is always patchified.

        Returns:
            The latent state after the conditioning has been applied.

        IMPORTANT: If the conditioning needs to add extra tokens to the latent,
        it should add them to the end of the latent.
        """
        ...
