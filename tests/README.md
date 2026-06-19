# LTX-2-MLX Test Suite

This directory contains the test suite for the LTX-2-MLX project, organized into unit tests and integration tests.

## Test Structure

### Test Inventory

230 tests across 26 files, grouped by area. All run fast and weight-free except
the loader tests (`test_loaders.py`), which are marked `requires_weights`.

**Schedulers & precision (30)**
- **test_scheduler.py** (22) - sigma schedulers, distilled sigma values, Euler diffusion step
- **test_precision_plumbing.py** (8) - compute/storage dtype plumbing

**Conditioning & noise (21)**
- **test_conditioning.py** (17) - keyframe and latent-index conditioning, VideoLatentTools
- **test_noiser_parity.py** (4) - GaussianNoiser clean_latent blend (upstream parity)

**Transformer & fused ops (23)**
- **test_v2_architecture.py** (7) - LTX-2.3 transformer structure pins
- **test_fused_ops.py** (11) - fused-op correctness
- **test_transformer_cache.py** (5) - transformer cache (quantize / FF layout)

**Upscalers (17)**
- **test_upscalers.py** (15) - spatial/temporal upscaler blocks and pipelines
- **test_spatial_upscaler.py** (2) - spatial upscaler

**Audio (22)**
- **test_audio_onset.py** (10) - waveform onset-spike detection and trim
- **test_audio_onset_latent.py** (3) - latent-domain onset detector
- **test_audio_vae_encoder_load.py** (3) - audio VAE encoder weight load
- **test_vocoder_filters.py** (4) - vocoder filter ops
- **test_audio_noise_parity.py** (2) - audio noise-normalization removal lock-in

**Text encoder (6)**
- **test_gemma_tokenizer.py** (6) - Gemma 3 tokenizer

**Native I/O & encoding (20)**
- **test_image_io.py** (9) - ImageIO/CoreImage load, resize, save
- **test_ffmpeg_encoder.py** (3) - ffmpeg fallback encoder
- **test_videotoolbox_audio.py** (3) - AVFoundation audio read/write
- **test_cut_detect.py** (3) - cut detection
- **test_pixel_buffer_chroma.py** (1) - CVPixelBuffer chroma handling
- **test_videotoolbox_comparison.py** (1) - VideoToolbox frame comparison

**Loaders & streaming (47)**
- **test_loaders.py** (43, requires weights) - weight loaders and key mapping
- **test_safetensors_header.py** (2) - safetensors header parsing
- **test_streaming_converters.py** (2) - streaming-transformer converters

**Progress UI (21)**
- **test_progress.py** (21) - stacked phase-bar progress rendering

**End-to-end pipelines (23)**
- **test_pipelines.py** (23) - pipeline wiring (text-to-video, image-to-video, two-stage); runs in placeholder mode, no real weights

### Manual Verification (Placeholder Mode)

For quick verification without loading weights (works on low-RAM machines):

```bash
# Verify Two-Stage Pipeline Logic
ltx2mlx "test" \
  --pipeline two-stage \
  --placeholder \
  --distilled-lora weights/dummy.safetensors \
  --height 512 --width 768 \
  --no-gemma
```

## Running Tests

### With `uv` (Recommended - Fast and Isolated)

`uv` automatically creates an isolated environment and runs tests:

```bash
# Run the fast, weight-free suite
uv run pytest tests/ -m unit -v

# Run specific test file
uv run pytest tests/test_scheduler.py -v

# Run all tests
uv run pytest tests/ -v
```

### With `pytest` Directly

If you have dependencies installed:

```bash
# Run all unit tests
PYTHONPATH=. pytest tests/ -m unit -v

# Or if installed in editable mode
pytest tests/ -m unit -v
```

### Run All Unit Tests (Fast)
```bash
uv run pytest tests/ -m unit -v
```

### Run Specific Test File
```bash
# Scheduler tests only
pytest tests/test_scheduler.py -v

# Conditioning tests only
pytest tests/test_conditioning.py -v

# Upscaler tests only
pytest tests/test_upscalers.py -v
```

### Run Weight-Loading Tests
```bash
pytest tests/ -m requires_weights -v
```

### Run All Tests
```bash
pytest tests/ -v
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow" -v
```

### Skip Tests Requiring Weights
```bash
pytest tests/ -m "not requires_weights" -v
```

## Test Markers

Tests are automatically marked based on their characteristics:

- `@pytest.mark.unit` - Fast unit tests without external dependencies
- `@pytest.mark.integration` - End-to-end integration tests
- `@pytest.mark.slow` - Tests that take >10 seconds
- `@pytest.mark.requires_weights` - Tests requiring model weights

## Configuration

Test configuration is managed in:

- **conftest.py** - Pytest configuration, fixtures, and shared utilities
- **pyproject.toml** - Test settings under `[tool.pytest.ini_options]`

## Fixtures

Shared fixtures available to all tests:

- `test_logger` - Verbose test logging with timing
- `temp_output_dir` - Temporary directory for test outputs
- `weights_dir` - Path to model weights directory
- `examples_dir` - Path to example images directory

## Test Coverage

Current test coverage by component:

| Area | Tests |
|------|-------|
| Schedulers & precision | 30 |
| Conditioning & noise | 21 |
| Transformer & fused ops | 23 |
| Upscalers | 17 |
| Audio | 22 |
| Text encoder | 6 |
| Native I/O & encoding | 20 |
| Loaders & streaming | 47 |
| Progress UI | 21 |
| End-to-end pipelines | 23 |
| **Total** | **230** |

## Writing New Tests

### Unit Test Template

```python
"""Unit tests for <component name>.

Tests <component> without requiring model weights.

Run with: pytest tests/test_<component>.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.components import YourComponent


class TestYourComponent:
    """Test YourComponent functionality."""

    def test_basic_functionality(self):
        """Test basic component operation."""
        component = YourComponent()
        input_data = mx.random.normal((1, 128, 8, 8))

        output = component(input_data)

        assert output.shape == input_data.shape
        assert np.all(np.isfinite(np.array(output)))
```

### Integration Test Template

```python
@pytest.mark.integration
@pytest.mark.requires_weights
@pytest.mark.slow
def test_full_pipeline(test_logger):
    """Test complete video generation pipeline."""
    test_logger.log("Starting full pipeline test")

    # Your test implementation

    test_logger.log("Pipeline test completed")
```

## CI/CD Integration

For CI/CD pipelines, run only fast unit tests:

```bash
# Run unit tests only (no model weights needed)
pytest tests/ -m "unit and not slow" -v --maxfail=3
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'LTX_2_MLX'`:

```bash
# Set PYTHONPATH before running tests
PYTHONPATH=/path/to/LTX-2-MLX pytest tests/ -v

# Or install the package in development mode
pip install -e .
```

### Test Collection Errors

If pytest fails to collect tests:

1. Check that conftest.py exists in tests/
2. Verify all test files start with `test_`
3. Ensure all test functions start with `test_`

### Slow Test Execution

The fast suite should complete in a few seconds. If slower:

1. Check you're not running the weight-loading tests by accident
2. Use `-m unit` to run only the fast, weight-free tests
3. Verify the test isn't loading model weights

## Future Work

Coverage still missing (the areas above are otherwise covered):

1. **Video VAE roundtrip** - encoder/decoder numerical roundtrip (the audio VAE
   encoder load is covered; the video VAE has no dedicated roundtrip test)
2. **Gemma embedding parity** - end-to-end text-embedding parity vs the PyTorch reference
3. **Weight-loaded integration** - a full generate run against real weights (the
   pipeline tests currently run in placeholder mode)
