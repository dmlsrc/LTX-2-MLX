"""Tests for pipeline module: configuration, utilities, and generation flow."""

import os
import tempfile

import mlx.core as mx
import pytest

from LTX_2_MLX.conditioning.keyframe import VideoConditionByKeyframeIndex
from LTX_2_MLX.conditioning.latent import VideoConditionByLatentIndex

# Import the modules under test
from LTX_2_MLX.pipelines.common import (
    ImageCondition,
    create_image_conditionings,
    load_image_tensor,
    post_process_latent,
    timesteps_from_mask,
)
from LTX_2_MLX.types import NATIVE_FPS, VideoLatentShape


def _solid_png(path, mode, w, h, color):
    """Write a flat-color PNG in a PIL-style mode natively (no Pillow).

    Exercises load_image_tensor's decode path on genuine grayscale ("L"),
    alpha ("RGBA"), and "RGB" sources without depending on Pillow.
    """
    from LTX_2_MLX.videotoolbox._compat import Foundation, Quartz

    if mode == "L":
        buf = bytearray([int(color)] * (w * h))
        cs = Quartz.CGColorSpaceCreateDeviceGray()
        ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w, cs, Quartz.kCGImageAlphaNone)
    else:
        if isinstance(color, int):
            color = (color, color, color)
        if mode == "RGBA":
            r, g, b, a = color
            info = Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrderDefault
        else:  # "RGB"
            r, g, b, a = color[0], color[1], color[2], 255
            info = Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault
        buf = bytearray(bytes([r, g, b, a]) * (w * h))
        cs = Quartz.CGColorSpaceCreateDeviceRGB()
        ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w * 4, cs, info)
    cg = Quartz.CGBitmapContextCreateImage(ctx)
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(dest, cg, None)
    Quartz.CGImageDestinationFinalize(dest)


# ============================================================================
# ImageCondition Tests
# ============================================================================

class TestImageCondition:
    """Tests for ImageCondition dataclass."""

    def test_default_strength(self):
        """Test default conditioning strength."""
        cond = ImageCondition(image_path="/path/to/image.png", frame_index=0)
        assert cond.strength == 0.95

    def test_custom_strength(self):
        """Test custom conditioning strength."""
        cond = ImageCondition(
            image_path="/path/to/image.png",
            frame_index=5,
            strength=0.8,
        )
        assert cond.frame_index == 5
        assert cond.strength == 0.8


class TestCreateImageConditionings:
    """Tests for image-conditioning item selection."""

    class _FakeEncoder:
        def __call__(self, image_tensor):
            return mx.ones(
                (
                    image_tensor.shape[0],
                    128,
                    image_tensor.shape[2],
                    image_tensor.shape[3] // 32,
                    image_tensor.shape[4] // 32,
                ),
                dtype=image_tensor.dtype,
            )

    def test_first_frame_uses_latent_replacement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "RGB", 64, 64, (128, 64, 32))

            conditionings = create_image_conditionings(
                [ImageCondition(img_path, frame_index=0, strength=0.9)],
                self._FakeEncoder(),
                height=64,
                width=64,
            )

        assert isinstance(conditionings[0], VideoConditionByLatentIndex)
        assert conditionings[0].latent_idx == 0

    def test_nonzero_frame_uses_keyframe_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "RGB", 64, 64, (128, 64, 32))

            conditionings = create_image_conditionings(
                [ImageCondition(img_path, frame_index=9, strength=0.8)],
                self._FakeEncoder(),
                height=64,
                width=64,
            )

        assert isinstance(conditionings[0], VideoConditionByKeyframeIndex)
        assert conditionings[0].frame_idx == 9


# ============================================================================
# Common Pipeline Utility Tests
# ============================================================================

class TestLoadImageTensor:
    """Tests for load_image_tensor function."""

    def test_load_valid_rgb_image(self):
        """Test loading a valid RGB image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "RGB", 256, 256, (128, 64, 32))

            # Load and validate
            tensor = load_image_tensor(img_path, height=128, width=128)

            assert tensor.shape == (1, 3, 1, 128, 128)
            assert tensor.dtype == mx.float32
            # Values should be in [-1, 1] range
            assert float(mx.min(tensor)) >= -1.0
            assert float(mx.max(tensor)) <= 1.0

    def test_load_valid_rgba_image(self):
        """Test loading an RGBA image (converted to RGB)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "RGBA", 256, 256, (128, 64, 32, 255))

            tensor = load_image_tensor(img_path, height=64, width=64)
            assert tensor.shape == (1, 3, 1, 64, 64)

    def test_load_valid_grayscale_image(self):
        """Test loading a grayscale image (converted to RGB)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "L", 256, 256, 128)

            tensor = load_image_tensor(img_path, height=64, width=64)
            assert tensor.shape == (1, 3, 1, 64, 64)

    def test_nonexistent_file_raises(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_tensor("/nonexistent/path.png", height=64, width=64)

    def test_custom_dtype(self):
        """Test loading with custom dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _solid_png(img_path, "RGB", 64, 64, (0, 0, 0))

            tensor = load_image_tensor(img_path, 32, 32, dtype=mx.float16)
            assert tensor.dtype == mx.float16


class TestPostProcessLatent:
    """Tests for post_process_latent function."""

    def test_full_denoise(self):
        """Test with full denoise mask (all 1s)."""
        denoised = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
        clean = mx.array([[[5.0, 6.0], [7.0, 8.0]]])
        mask = mx.ones_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert mx.allclose(result, denoised)

    def test_no_denoise(self):
        """Test with no denoise mask (all 0s)."""
        denoised = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
        clean = mx.array([[[5.0, 6.0], [7.0, 8.0]]])
        mask = mx.zeros_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert mx.allclose(result, clean)

    def test_partial_denoise(self):
        """Test with partial denoise mask."""
        denoised = mx.array([[[1.0, 2.0]]])
        clean = mx.array([[[5.0, 6.0]]])
        mask = mx.array([[[1.0, 0.0]]])  # First element denoised, second clean

        result = post_process_latent(denoised, mask, clean)
        expected = mx.array([[[1.0, 6.0]]])
        assert mx.allclose(result, expected)

    def test_preserves_dtype(self):
        """Test that output dtype matches input."""
        denoised = mx.array([[[1.0]]]).astype(mx.float16)
        clean = mx.array([[[2.0]]]).astype(mx.float16)
        mask = mx.ones_like(denoised)

        result = post_process_latent(denoised, mask, clean)
        assert result.dtype == mx.float16


class TestTimestepsFromMask:
    """Tests for timesteps_from_mask function."""

    def test_full_mask(self):
        """Test with full denoise mask."""
        mask = mx.ones((1, 10))
        sigma = 0.5

        result = timesteps_from_mask(mask, sigma)
        expected = mx.full((1, 10), 0.5)
        assert mx.allclose(result, expected)

    def test_zero_mask(self):
        """Test with zero mask."""
        mask = mx.zeros((1, 10))
        sigma = 0.5

        result = timesteps_from_mask(mask, sigma)
        expected = mx.zeros((1, 10))
        assert mx.allclose(result, expected)

    def test_partial_mask(self):
        """Test with partial mask."""
        mask = mx.array([[1.0, 0.5, 0.0]])
        sigma = 2.0

        result = timesteps_from_mask(mask, sigma)
        expected = mx.array([[2.0, 1.0, 0.0]])
        assert mx.allclose(result, expected)


# ============================================================================
# VideoLatentShape Tests
# ============================================================================

class TestVideoLatentShape:
    """Tests for VideoLatentShape class."""

    def test_from_pixel_shape(self):
        """Test creating latent shape from pixel shape."""
        from LTX_2_MLX.types import VideoPixelShape

        pixel_shape = VideoPixelShape(
            batch=1,
            frames=97,
            height=480,
            width=704,
            fps=NATIVE_FPS,
        )

        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=128
        )

        # VAE compression: 32x spatial, 8x temporal
        assert latent_shape.batch == 1
        assert latent_shape.channels == 128
        assert latent_shape.frames == (97 - 1) // 8 + 1  # 13
        assert latent_shape.height == 480 // 32  # 15
        assert latent_shape.width == 704 // 32  # 22


# ============================================================================
# Component Integration Tests
# ============================================================================

class TestPipelineComponents:
    """Tests for pipeline component integration."""

    def test_patchifier_import(self):
        """Test VideoLatentPatchifier can be imported and instantiated."""
        from LTX_2_MLX.components.patchifiers import VideoLatentPatchifier

        patchifier = VideoLatentPatchifier(patch_size=1)
        # patch_size is a tuple (1, 1, 1) for spatial dimensions
        assert patchifier.patch_size == (1, 1, 1)

    def test_noiser_import(self):
        """Test GaussianNoiser can be imported and instantiated."""
        from LTX_2_MLX.components.noisers import GaussianNoiser

        noiser = GaussianNoiser()
        assert noiser is not None

    def test_guider_import(self):
        """Test CFGGuider can be imported and instantiated."""
        from LTX_2_MLX.components.guiders import CFGGuider

        # CFGGuider is a dataclass with 'scale' attribute
        guider = CFGGuider(scale=7.5)
        assert guider.scale == 7.5

    def test_diffusion_step_import(self):
        """Test EulerDiffusionStep can be imported and instantiated."""
        from LTX_2_MLX.components.diffusion_steps import EulerDiffusionStep

        step = EulerDiffusionStep()
        assert step is not None

# ============================================================================

class TestSchedulerIntegration:
    """Tests for scheduler integration with pipelines."""

    def test_get_sigma_schedule(self):
        """Test sigma schedule generation."""
        from LTX_2_MLX.components.schedulers import get_sigma_schedule

        sigmas = get_sigma_schedule(num_steps=10, distilled=False)

        assert len(sigmas) == 11  # num_steps + 1
        assert float(sigmas[-1]) == 0.0  # Should end at 0

    def test_distilled_sigma_values(self):
        """Test distilled sigma values are available."""
        from LTX_2_MLX.components import DISTILLED_SIGMA_VALUES

        assert len(DISTILLED_SIGMA_VALUES) > 0
        assert DISTILLED_SIGMA_VALUES[-1] == 0.0  # Should end at 0
