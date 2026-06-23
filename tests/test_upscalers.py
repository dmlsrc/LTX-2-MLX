"""Unit tests for LTX-2 upscaler components.

Tests spatial and temporal upscalers without requiring pre-trained weights.

Run with: pytest tests/test_upscalers.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.model.upscaler.spatial import (
    PixelShuffle2d,
    ResBlock3d,
    SpatialUpscaler,
)


class TestResBlock3d:
    """Test 3D residual block."""

    def test_basic_resblock(self):
        """Test basic 3D residual block."""
        block = ResBlock3d(channels=1024, num_groups=32)
        # NCTHW format
        x = mx.random.normal((1, 1024, 4, 8, 8))

        output = block(x)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert np.all(np.isfinite(np.array(output)))

    def test_resblock_stability(self):
        """Test that ResBlock doesn't cause exponential amplification."""
        block1 = ResBlock3d(channels=512, num_groups=32)
        block2 = ResBlock3d(channels=512, num_groups=32)
        block3 = ResBlock3d(channels=512, num_groups=32)

        x = mx.random.normal((1, 512, 4, 8, 8))

        # Track variance through blocks
        initial_var = float(mx.var(x))

        x1 = block1(x)
        var1 = float(mx.var(x1))

        x2 = block2(x1)
        var2 = float(mx.var(x2))

        x3 = block3(x2)
        var3 = float(mx.var(x3))

        # Variance should stay bounded
        assert var1 < initial_var * 10, f"Variance exploded: {initial_var} -> {var1}"
        assert var2 < initial_var * 10, f"Variance exploded: {initial_var} -> {var2}"
        assert var3 < initial_var * 10, f"Variance exploded: {initial_var} -> {var3}"

    def test_resblock_different_sizes(self):
        """Test ResBlock with different channel sizes."""
        for channels in [256, 512, 1024]:
            block = ResBlock3d(channels=channels, num_groups=min(32, channels))
            x = mx.random.normal((1, channels, 4, 8, 8))  # NCTHW format

            output = block(x)
            assert output.shape == x.shape


class TestPixelShuffle2d:
    """Test pixel shuffle upsampler."""

    def test_basic_upsampling(self):
        """Test basic 2x spatial upsampling."""
        shuffle = PixelShuffle2d(upscale_factor=2)
        # Input: (B, H, W, C*r*r) where C*r*r = 2048, r=2, so C=512
        x = mx.random.normal((4, 8, 8, 2048))

        output = shuffle(x)

        # Output should be (B, H*2, W*2, C) = (4, 16, 16, 512)
        assert output.shape == (4, 16, 16, 512)
        assert np.all(np.isfinite(np.array(output)))

    def test_pixel_shuffle_rearrangement(self):
        """Test the pixel shuffle rearrangement."""
        shuffle = PixelShuffle2d(upscale_factor=2)

        # Create input with known pattern
        b, h, w = 1, 2, 2
        c_out = 4
        r = 2
        c_in = c_out * r * r  # 16

        x = mx.arange(b * h * w * c_in).reshape(b, h, w, c_in).astype(mx.float32)
        output = shuffle(x)

        # Output should be (1, 4, 4, 4)
        assert output.shape == (b, h * r, w * r, c_out)


class TestSpatialUpscaler:
    """Test complete spatial upscaler."""

    def test_basic_spatial_upscaling(self):
        """Test basic 2x spatial upscaling."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,  # Use smaller for testing
            num_blocks_per_stage=2,  # Fewer blocks for faster testing
        )
        # Input: (B, C, F, H, W) - NCFHW format
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Spatial dimensions should be doubled
        assert output.shape == (1, 128, 4, 16, 16)
        assert np.all(np.isfinite(np.array(output)))

    def test_spatial_upscaler_different_sizes(self):
        """Test spatial upscaler with different input sizes."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=1,  # Fewer blocks for faster testing
        )

        test_shapes = [
            (1, 128, 2, 4, 4),
            (1, 128, 4, 8, 8),
        ]

        for shape in test_shapes:
            x = mx.random.normal(shape)
            output = upscaler(x)

            # Check spatial 2x upsampling
            b, c, f, h, w = shape
            assert output.shape == (b, c, f, h * 2, w * 2)

    def test_spatial_upscaler_numerical_stability(self):
        """Test that spatial upscaler produces stable outputs."""
        upscaler = SpatialUpscaler(
            in_channels=128,
            mid_channels=512,
            num_blocks_per_stage=2,
        )
        x = mx.random.normal((1, 128, 4, 8, 8))

        output = upscaler(x)

        # Check for NaN or Inf
        output_np = np.array(output)
        assert np.all(np.isfinite(output_np)), "Output should not contain NaN or Inf"

        # Variance should be reasonable (not exploded)
        input_var = float(mx.var(x))
        output_var = float(mx.var(output))

        # Allow up to 100x variance increase
        assert output_var < input_var * 100, \
            f"Variance exploded: {input_var} -> {output_var}"
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
