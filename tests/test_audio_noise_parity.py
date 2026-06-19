"""Lock-in guard: the audio-noise channelwise-normalization bandaid stays removed.

`AVPipeline._channelwise_normalize_audio_noise` was an April-2026 workaround for a
duration-dependent amplitude bug.  The bug was later fixed properly (gate /
denoise-broadcast / audio-VAE-vocoder / timestep-scale fixes; see
docs/AUDIO_ISSUES.md), and the normalization was a parity divergence from upstream
Lightricks (un-normalized N(0,1)*sigma_max noise) that also perturbed the video
branch via a2v cross-attention.  It was env-gated (default OFF) on 2026-05-21 and
removed outright on 2026-06-18.

These tests fail loudly if the bandaid is reintroduced, enforcing the docs'
"DO NOT RE-ADD the normalization" directive.  The actual amplitude regression is a
manual full-generation check (the empirical table in docs/AUDIO_ISSUES.md); it
needs model weights and is not unit-testable here.
"""

import pathlib

from LTX_2_MLX.pipelines.av_pipeline import AVPipeline

_PKG = pathlib.Path(__file__).resolve().parents[1] / "LTX_2_MLX"
_BANDAID_TOKENS = ("NORMALIZE_AUDIO_NOISE", "channelwise_normalize_audio_noise")


def test_normalize_method_is_gone():
    assert not hasattr(AVPipeline, "_channelwise_normalize_audio_noise"), (
        "audio-noise whitening bandaid was reintroduced on AVPipeline; see "
        "docs/AUDIO_ISSUES.md -> Duration-Dependent Amplitude Bug"
    )


def test_no_bandaid_tokens_in_package_source():
    offenders = []
    for path in _PKG.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(tok in text for tok in _BANDAID_TOKENS):
            offenders.append(str(path.relative_to(_PKG.parent)))
    assert not offenders, (
        "audio-noise normalization bandaid reintroduced in: " + ", ".join(sorted(offenders))
    )
