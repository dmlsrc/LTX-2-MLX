"""Guards for the audio VAE encoder weight loader.

Regression cover for a bug where `load_audio_encoder_weights` silently loaded
only 5/58 params and reported success: it built conv keys without the `.conv`
nesting the checkpoint uses (`...conv1.weight` vs `...conv1.conv.weight`), read
per-channel stats from the wrong path, and `AudioEncoder` defaulted to 3
res-blocks/level where the LTX-2.3 checkpoint has 2.  A half-loaded encoder
(mostly zeros) is worse than a loud failure, so the loader now raises on any
missing key / structural mismatch.
"""

import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from LTX_2_MLX.model.audio_vae import AudioEncoder
from LTX_2_MLX.model.audio_vae.encoder import _load_conv_weights, load_audio_encoder_weights

_CACHE = (
    "/Users/Shared/huggingface/mlx/LTX-2-MLX-cache/"
    "ltx-2.3-22b-dev-9ab2c718983937ba4674/audio_vae.safetensors"
)


def test_default_res_blocks_matches_checkpoint():
    # LTX-2.3 audio VAE checkpoint has 2 res-blocks per level, not 3.
    assert AudioEncoder().num_res_blocks == 2


def test_conv_loader_raises_on_missing_key():
    # The anti-silent-success guarantee: a missing weight key must raise, never
    # leave the conv at its uninitialized value.
    enc = AudioEncoder()
    with pytest.raises(KeyError):
        _load_conv_weights(enc.conv_in, {}, "audio_vae.encoder.conv_in.conv")


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="audio VAE cache not present")
def test_encoder_fully_loads_from_checkpoint():
    enc = AudioEncoder()
    load_audio_encoder_weights(enc, _CACHE)  # raises if any key is missing
    params = dict(tree_flatten(enc.parameters()))
    nonzero = sum(float(mx.sum(mx.abs(v))) > 0 for v in params.values())
    assert nonzero == len(params), f"only {nonzero}/{len(params)} encoder params populated"
