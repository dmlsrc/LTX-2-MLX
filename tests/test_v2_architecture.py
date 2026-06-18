"""Lock-in tests for the LTX-2.3 (v2 / AudioVideo) transformer architecture.

These pin the v2-constant structure - caption projection absent (the V2 feature
extractor projects directly), 9-parameter AdaLN tables, and gated attention - so
the removal of the LTX-2 (v1 / 19B) code paths cannot silently alter the live 22B
transformer. They are written against the current tree and must keep passing
through the v1 removal.
"""

import mlx.core as mx
import pytest

from LTX_2_MLX.generate import require_ltx23_checkpoint
from LTX_2_MLX.model.transformer.attention import Attention
from LTX_2_MLX.model.transformer.model import LTXModel, LTXModelType


def _build_av_transformer(num_layers: int = 2) -> LTXModel:
    """A small AudioVideo (v2) transformer with the LTX-2.3 config flags."""
    return LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=8,
        out_channels=8,
        num_layers=num_layers,
        cross_attention_dim=64,
        caption_channels=None,        # v2: feature extractor projects directly
        cross_attention_adaln=True,   # v2
        apply_gated_attention=True,   # v2
    )


def test_v2_transformer_has_no_caption_projection():
    m = _build_av_transformer()
    assert m.caption_projection is None
    assert m.audio_caption_projection is None


def test_v2_transformer_prompt_adaln_present():
    m = _build_av_transformer()
    assert m.prompt_adaln_single is not None
    assert m.audio_prompt_adaln_single is not None


def test_v2_blocks_use_9_param_adaln():
    blk = _build_av_transformer().transformer_blocks[0]
    assert blk.cross_attention_adaln is True
    # 9 = 6 base + 3 cross-attention Q-modulation params (v2; v1 was 6).
    m = _build_av_transformer()
    blk = m.transformer_blocks[0]
    assert blk.scale_shift_table.shape == (9, m.video_inner_dim)
    assert blk.audio_scale_shift_table.shape == (9, m.audio_inner_dim)


def test_v2_blocks_use_gated_attention():
    blk = _build_av_transformer().transformer_blocks[0]
    assert blk.attn1.to_gate_logits is not None
    assert blk.attn2.to_gate_logits is not None
    assert blk.audio_attn1.to_gate_logits is not None


def test_attention_gating_toggle():
    gated = Attention(query_dim=32, heads=2, dim_head=16, apply_gated_attention=True)
    assert gated.to_gate_logits is not None
    ungated = Attention(query_dim=32, heads=2, dim_head=16, apply_gated_attention=False)
    assert ungated.to_gate_logits is None


def _write_checkpoint(tmp_path, version):
    p = tmp_path / f"ckpt_{version or 'none'}.safetensors"
    metadata = {"model_version": version} if version else {}
    mx.save_safetensors(str(p), {"w": mx.zeros((1,))}, metadata=metadata)
    return str(p)


def test_guard_accepts_ltx23(tmp_path):
    # v2.3 checkpoint and the empty (placeholder) path must not raise.
    require_ltx23_checkpoint(_write_checkpoint(tmp_path, "2.3.0"))
    require_ltx23_checkpoint("")


@pytest.mark.parametrize("version", ["2.0.0", "1.0.0", None])
def test_guard_rejects_non_ltx23(tmp_path, version):
    with pytest.raises(SystemExit, match="only LTX-2.3"):
        require_ltx23_checkpoint(_write_checkpoint(tmp_path, version))
