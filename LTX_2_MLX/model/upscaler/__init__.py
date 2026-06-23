"""Upscaler modules for LTX-2 MLX."""

from .spatial import SpatialUpscaler, load_spatial_upscaler_weights

__all__ = [
    "SpatialUpscaler",
    "load_spatial_upscaler_weights",
]
