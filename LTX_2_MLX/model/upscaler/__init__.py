"""Upscaler modules for LTX-2 MLX."""

from .spatial import SpatialUpscaler, load_spatial_upscaler_weights
from .temporal import TemporalUpscaler, load_temporal_upscaler_weights

__all__ = [
    "SpatialUpscaler",
    "TemporalUpscaler",
    "load_spatial_upscaler_weights",
    "load_temporal_upscaler_weights",
]
