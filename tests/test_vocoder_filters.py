import mlx.core as mx
import pytest

from LTX_2_MLX.model.audio_vae.vocoder import (
    LowPassFilter1d,
    UpSample1d,
    _load_lowpass_filter_weights,
    _load_upsample1d_weights,
)


def test_missing_checkpoint_lowpass_filter_raises() -> None:
    lowpass = LowPassFilter1d(kernel_size=12)

    with pytest.raises(ValueError, match="Missing required vocoder filter tensor"):
        _load_lowpass_filter_weights(
            {},
            "vocoder.resblocks.0.acts1.0.downsample.lowpass",
            lowpass,
        )


def test_mismatched_checkpoint_upsample_filter_shape_raises() -> None:
    upsample = UpSample1d(ratio=2, kernel_size=12)
    weights = {"vocoder.resblocks.0.acts1.0.upsample.filter": mx.zeros((1, 1, 10))}

    with pytest.raises(ValueError, match="Unexpected shape for vocoder filter tensor"):
        _load_upsample1d_weights(
            weights,
            "vocoder.resblocks.0.acts1.0.upsample",
            upsample,
        )


def test_checkpoint_filter_placeholder_cannot_run_before_load() -> None:
    upsample = UpSample1d(ratio=2, kernel_size=12)
    x = mx.zeros((1, 1, 8))

    with pytest.raises(
        ValueError, match="requires a filter loaded from the vocoder checkpoint"
    ):
        upsample(x)


def test_bwe_hann_resampler_still_constructs_filter() -> None:
    upsample = UpSample1d(ratio=3, window_type="hann")

    assert upsample.filter.shape == (1, 1, 43)
