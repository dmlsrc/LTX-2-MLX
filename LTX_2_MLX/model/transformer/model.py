"""LTX-2 Transformer Model for MLX (Unified Video/Audio)."""

import gc
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .transformer import BasicTransformerBlock, BasicAVTransformerBlock, TransformerArgs, TransformerConfig
from ...components.perturbations import BatchedPerturbationConfig


def _pack_transformer_args(args: Optional[TransformerArgs]) -> Optional[tuple]:
    """Flatten TransformerArgs for mx.compile boundaries."""
    if args is None:
        return None
    return (
        args.x,
        args.context,
        args.timesteps,
        args.positional_embeddings,
        args.context_mask,
        args.embedded_timestep,
        args.cross_positional_embeddings,
        args.cross_scale_shift_timestep,
        args.cross_gate_timestep,
        args.prompt_timestep,
    )


def _unpack_transformer_args(packed: Optional[tuple]) -> Optional[TransformerArgs]:
    """Restore TransformerArgs after an mx.compile boundary."""
    if packed is None:
        return None
    (
        x,
        context,
        timesteps,
        positional_embeddings,
        context_mask,
        embedded_timestep,
        cross_positional_embeddings,
        cross_scale_shift_timestep,
        cross_gate_timestep,
        prompt_timestep,
    ) = packed
    return TransformerArgs(
        x=x,
        context=context,
        timesteps=timesteps,
        positional_embeddings=positional_embeddings,
        context_mask=context_mask,
        embedded_timestep=embedded_timestep,
        cross_positional_embeddings=cross_positional_embeddings,
        cross_scale_shift_timestep=cross_scale_shift_timestep,
        cross_gate_timestep=cross_gate_timestep,
        enabled=True,
        prompt_timestep=prompt_timestep,
    )


def _compile_transformer_block_group(blocks: List[nn.Module]):
    """Compile a resident block window while treating parameters as dynamic inputs."""
    blocks = list(blocks)

    def _call(video_packed: Optional[tuple], audio_packed: Optional[tuple]) -> tuple:
        video_args = _unpack_transformer_args(video_packed)
        audio_args = _unpack_transformer_args(audio_packed)
        for block in blocks:
            video_args, audio_args = block(
                video_args,
                audio_args,
                perturbations=None,
                profile_events=None,
            )
        return _pack_transformer_args(video_args), _pack_transformer_args(audio_args)

    return mx.compile(_call, inputs=blocks)


class LTXModelType(Enum):
    """Model type variants."""

    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings with GELU activation.

    Adapted from PixArt-alpha implementation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def __call__(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear_1(caption)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass
class Modality:
    """Input modality data (video or audio)."""

    latent: mx.array  # Shape: (B, T, C) - patchified latents
    context: mx.array  # Shape: (B, S, C_ctx) - text context
    context_mask: Optional[mx.array]  # Shape: (B, S) or (B, 1, S, S)
    timesteps: mx.array  # Shape: (B,) or (B, T) - timestep values
    positions: mx.array  # Shape: (B, n_dims, T) - position indices
    enabled: bool = True
    sigma: Optional[mx.array] = None  # Shape: (B,) - scalar noise level for V2 prompt_adaln


class TransformerArgsPreprocessor:
    """
    Preprocesses inputs for transformer blocks.

    Handles:
    - Patchify projection (linear embedding)
    - Timestep embedding via AdaLN
    - Caption projection
    - Position embedding computation (RoPE)
    """

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: Optional[PixArtAlphaTextProjection],
        inner_dim: int,
        max_pos: List[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool = True,
        timestep_scale_multiplier: int = 1000,
        positional_embedding_theta: float = 10000.0,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,  # LTX-2 distilled uses SPLIT
        compute_dtype: mx.Dtype = mx.bfloat16,
        prompt_adaln: Optional[AdaLayerNormSingle] = None,
        use_double_precision: bool = False,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.compute_dtype = compute_dtype
        self.prompt_adaln = prompt_adaln
        self.use_double_precision = use_double_precision

    def _prepare_timestep(
        self,
        timestep: mx.array,
        adaln: AdaLayerNormSingle,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare timestep embeddings.

        Args:
            timestep: Timestep values, shape (B,) or (B, T).
            adaln: AdaLayerNormSingle to use.
            batch_size: Batch size.

        Returns:
            Tuple of (timestep_emb, embedded_timestep).
        """
        timestep = timestep * self.timestep_scale_multiplier
        emb, embedded_timestep = adaln(timestep.flatten())

        # Reshape processed emb to (B, num_tokens, num_embeddings, inner_dim)
        num_embeddings = emb.shape[-1] // self.inner_dim
        emb = emb.reshape(batch_size, -1, num_embeddings, self.inner_dim)

        # Reshape raw embedded_timestep to (B, num_tokens, inner_dim)
        embedded_timestep = embedded_timestep.reshape(batch_size, -1, self.inner_dim)

        return emb, embedded_timestep

    def _prepare_context(
        self,
        context: mx.array,
        x: mx.array,
    ) -> mx.array:
        """
        Prepare context (caption) for cross-attention.

        Args:
            context: Caption embeddings, shape (B, S, C_ctx).
            x: Projected hidden states (for batch size).

        Returns:
            Projected context, shape (B, S, inner_dim).
        """
        batch_size = x.shape[0]
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        context = context.reshape(batch_size, -1, x.shape[-1])
        return context

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[mx.array],
        target_dtype: mx.Dtype = mx.float32,
    ) -> Optional[mx.array]:
        """
        Prepare attention mask for cross-attention.

        Converts boolean mask to additive mask for softmax.
        Uses dtype-appropriate masking values to match PyTorch finfo behavior.

        Args:
            attention_mask: Boolean or float mask of shape (B, S).
            target_dtype: Target dtype for the mask (determines mask value).

        Returns:
            Additive attention mask of shape (B, 1, 1, S) or None.
        """
        if attention_mask is None:
            return None

        # If already a float mask, return as-is
        if attention_mask.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return attention_mask

        # Use dtype-appropriate max value (matches PyTorch finfo behavior)
        # PyTorch uses (mask - 1) * finfo(dtype).max
        if target_dtype == mx.float16:
            mask_value = -65504.0  # ~-finfo(float16).max
        elif target_dtype == mx.bfloat16:
            mask_value = -3.38e38  # ~-finfo(bfloat16).max
        else:
            mask_value = -3.40e38  # ~-finfo(float32).max

        # Convert boolean mask to additive mask
        # True = attend (0), False = don't attend (large negative)
        mask = (1 - attention_mask.astype(mx.float32)) * mask_value
        mask = mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
        return mask.astype(target_dtype)

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare RoPE positional embeddings.

        No caching - matches PyTorch behavior and avoids stale cache bugs when
        fps, causal_fix, or max_pos change while shape stays constant.

        Args:
            positions: Position indices, shape (B, n_dims, T, 2) where last dim is [start, end].

        Returns:
            Tuple of (cos_freq, sin_freq) for RoPE.
        """
        pe = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=mx.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
        )
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        """
        Prepare all inputs for transformer blocks.

        Args:
            modality: Input modality data.

        Returns:
            TransformerArgs ready for transformer blocks.
        """
        # Project latents to inner dimension
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]

        # Prepare timestep embeddings
        timestep_emb, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size
        )

        # Prepare prompt timestep (V2 cross-attention adaln)
        prompt_timestep = None
        if self.prompt_adaln is not None:
            # Use sigma if provided, otherwise fall back to timesteps (same for scalar case)
            sigma = modality.sigma if modality.sigma is not None else modality.timesteps
            if sigma.ndim > 1:
                sigma = sigma[:, 0]  # Per-token timesteps: use first token's sigma
            prompt_emb, _ = self._prepare_timestep(
                sigma, self.prompt_adaln, batch_size
            )
            prompt_timestep = prompt_emb  # Shape: (B, 1, 2, D)

        # Prepare context (caption projection)
        context = self._prepare_context(modality.context, x)

        # Prepare attention mask with dtype-appropriate masking values
        attention_mask = self._prepare_attention_mask(
            modality.context_mask, target_dtype=self.compute_dtype
        )

        # Prepare positional embeddings (RoPE)
        pe = self._prepare_positional_embeddings(modality.positions)

        return TransformerArgs(
            x=x,
            context=context,
            timesteps=timestep_emb,
            positional_embeddings=pe,
            context_mask=attention_mask,
            embedded_timestep=embedded_timestep,
            prompt_timestep=prompt_timestep,
        )


class MultiModalTransformerArgsPreprocessor:
    """
    Preprocesses inputs for AudioVideo transformer blocks.

    Extends TransformerArgsPreprocessor to handle cross-modal attention:
    - Separate positional embeddings for cross-attention
    - Cross-attention timestep embeddings (scale/shift and gate)
    """

    def __init__(
        self,
        simple_preprocessor: TransformerArgsPreprocessor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        cross_pe_max_pos: int,
        audio_cross_attention_dim: int,
        av_ca_timestep_scale_multiplier: int = 1,  # PyTorch default (av_ca_factor = 1/1000)
    ):
        """
        Initialize multi-modal preprocessor.

        Args:
            simple_preprocessor: Base preprocessor for video/audio.
            cross_scale_shift_adaln: AdaLN for cross-attention scale/shift.
            cross_gate_adaln: AdaLN for cross-attention gate.
            cross_pe_max_pos: Max position for cross-modal RoPE.
            audio_cross_attention_dim: Dimension for audio cross-attention.
            av_ca_timestep_scale_multiplier: Scale for cross-attention timestep.
        """
        self.simple_preprocessor = simple_preprocessor
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def _prepare_cross_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare cross-modal positional embeddings.

        No caching - matches PyTorch behavior and avoids stale cache bugs.
        Uses only the temporal dimension for cross-modal attention.
        """
        # Use only the first dimension (temporal) for cross-modal attention
        temporal_positions = positions[:, 0:1, :]

        pe = precompute_freqs_cis(
            indices_grid=temporal_positions,
            dim=self.audio_cross_attention_dim,
            out_dtype=mx.float32,
            theta=self.simple_preprocessor.positional_embedding_theta,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            rope_type=self.simple_preprocessor.rope_type,
        )
        return pe

    def _prepare_cross_attention_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare cross-attention timestep embeddings.

        Returns scale/shift and gate embeddings separately.
        """
        scaled_timestep = timestep * self.simple_preprocessor.timestep_scale_multiplier

        # Scale/shift timestep (AdaLayerNormSingle returns tuple, we only need processed emb)
        scale_shift_emb, _ = self.cross_scale_shift_adaln(scaled_timestep.flatten())
        scale_shift_emb = scale_shift_emb.reshape(batch_size, -1, 4, self.simple_preprocessor.inner_dim)

        # Gate timestep (with AV CA scale)
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.simple_preprocessor.timestep_scale_multiplier
        gate_emb, _ = self.cross_gate_adaln((scaled_timestep * av_ca_factor).flatten())
        gate_emb = gate_emb.reshape(batch_size, -1, 1, self.simple_preprocessor.inner_dim)

        return scale_shift_emb, gate_emb

    def prepare(
        self,
        modality: Modality,
        cross_modality: Optional[Modality] = None,
    ) -> TransformerArgs:
        """
        Prepare all inputs for AudioVideo transformer blocks.

        Args:
            modality: Input modality data (video or audio).
            cross_modality: The OTHER modality, used to compute cross-attention
                timestep embeddings. When preparing audio, pass video here
                (and vice versa). Matches PyTorch: prepare(audio, video).

        Returns:
            TransformerArgs with cross-modal attention fields populated.
        """
        # Get basic transformer args
        args = self.simple_preprocessor.prepare(modality)

        if cross_modality is None:
            return args

        # Cross-modal positional embeddings use THIS modality's temporal positions
        cross_pe = self._prepare_cross_positional_embeddings(modality.positions)

        # Cross-attention timestep uses the OTHER modality's sigma
        # This is critical: audio cross-attn timestep comes from video's sigma,
        # and video cross-attn timestep comes from audio's sigma
        cross_sigma = cross_modality.sigma if cross_modality.sigma is not None else cross_modality.timesteps
        if cross_sigma.ndim > 1:
            cross_sigma = cross_sigma[:, 0]  # Per-token: use first token's sigma

        batch_size = args.x.shape[0]
        cross_scale_shift, cross_gate = self._prepare_cross_attention_timestep(
            cross_sigma, batch_size
        )

        return args.replace(
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift,
            cross_gate_timestep=cross_gate,
        )


class LTXModel(nn.Module):
    """
    LTX-2 Transformer Model (Unified Video/Audio).

    Wrapper for both VideoOnly and AudioVideo variants.
    Architecture:
    - Input: Patchified video latents (and optional audio latents)
    - 48 transformer blocks with self-attention, cross-attention, and FFN
    - AdaLN conditioning on timestep
    - Output: Velocity predictions for diffusion

    This is the core denoising model that predicts velocities.
    """

    # Audio configuration constants
    AUDIO_ATTENTION_HEADS = 32
    AUDIO_HEAD_DIM = 64
    AUDIO_IN_CHANNELS = 128  # Audio VAE latent channels
    AUDIO_OUT_CHANNELS = 128
    # Audio cross-PE max position - PyTorch uses 20 for audio positional max
    # and max(video_t, audio_t) for cross-modal positions (typically 20)
    AUDIO_CROSS_PE_MAX_POS = 20

    def __init__(
        self,
        model_type: LTXModelType = LTXModelType.VideoOnly,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-6,
        caption_channels: Optional[int] = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: int = 1000,
        # AV cross-attn timestep scale: PyTorch defaults to 1, giving av_ca_factor = 1/1000
        # This controls the timestep scaling for audio-video cross attention
        av_ca_timestep_scale_multiplier: int = 1,
        use_middle_indices_grid: bool = True,
        # RoPE type: LTX-2 distilled weights use SPLIT
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        compute_dtype: mx.Dtype = mx.bfloat16,
        low_memory: bool = False,
        fast_mode: bool = False,
        profile_transformer_once: bool = False,
        cross_attention_adaln: bool = False,
        apply_gated_attention: bool = False,
    ):
        """
        Initialize LTX model.

        Args:
            model_type: Type of model (VideoOnly, AudioVideo, AudioOnly).
            num_attention_heads: Number of attention heads (32).
            attention_head_dim: Dimension per head (128).
            in_channels: Input channels from VAE (128).
            out_channels: Output channels (128).
            num_layers: Number of transformer blocks (48).
            cross_attention_dim: Text context dimension (4096).
            norm_eps: Epsilon for normalization.
            caption_channels: Caption embedding dimension (3840 from Gemma).
            positional_embedding_theta: Base theta for RoPE.
            positional_embedding_max_pos: Max positions [time, height, width].
            timestep_scale_multiplier: Scale for timestep (1000).
            av_ca_timestep_scale_multiplier: Scale for AV cross-attention timestep.
                PyTorch default is 1, giving av_ca_factor = 1/1000.
            use_middle_indices_grid: Use middle of position bounds for RoPE.
            rope_type: Type of RoPE. LTX-2 distilled uses SPLIT.
            compute_dtype: Dtype for computation.
            low_memory: If True, use aggressive memory optimization (eval every 4 layers).
            fast_mode: If True, skip intermediate evals for faster inference (uses more memory).
            profile_transformer_once: If True, print one forced-eval transformer timing trace.
        """
        super().__init__()

        self.model_type = model_type
        self.rope_type = rope_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.norm_eps = norm_eps
        self.compute_dtype = compute_dtype
        self.low_memory = low_memory
        self.fast_mode = fast_mode
        self.profile_transformer_once = profile_transformer_once
        self.profile_transformer_label: Optional[str] = None
        self.profile_transformer_blocks: Tuple[int, ...] = ()
        self.transformer_block_streamer = None
        self.transformer_block_compile = False
        object.__setattr__(self, "_compiled_transformer_block_groups", {})
        self._transformer_block_compile_disabled = False
        self.cross_attention_adaln = cross_attention_adaln
        
        # Eval frequency setup
        if fast_mode:
            self._eval_frequency = 0
        elif low_memory:
            self._eval_frequency = 4
        else:
            self._eval_frequency = 8

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.num_attention_heads = num_attention_heads
        
        # Video dimensions
        self.video_inner_dim = num_attention_heads * attention_head_dim
        # Map generic inner_dim to video_inner_dim for compatibility
        self.inner_dim = self.video_inner_dim
        
        # Audio dimensions
        self.audio_inner_dim = self.AUDIO_ATTENTION_HEADS * self.AUDIO_HEAD_DIM

        # V2: 9 adaln params (6 base + 3 for cross-attention Q modulation)
        adaln_num_embeddings = 9 if cross_attention_adaln else 6

        # =================
        # VIDEO COMPONENTS
        # =================
        if self.model_type.is_video_enabled():
            # Input projection
            self.patchify_proj = nn.Linear(in_channels, self.video_inner_dim, bias=True)

            # AdaLN
            self.adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=adaln_num_embeddings
            )

            # V2: prompt adaln for cross-attention KV modulation
            self.prompt_adaln_single = (
                AdaLayerNormSingle(self.video_inner_dim, num_embeddings=2)
                if cross_attention_adaln else None
            )

            # Caption projection (None for V2 models where feature extractor projects directly)
            if caption_channels is not None:
                self.caption_projection = PixArtAlphaTextProjection(
                    in_features=caption_channels,
                    hidden_size=self.video_inner_dim,
                )
            else:
                self.caption_projection = None

            # Output projection
            self.scale_shift_table = mx.zeros((2, self.video_inner_dim), dtype=mx.float32)
            self.norm_out = nn.LayerNorm(self.video_inner_dim, affine=False, eps=norm_eps)
            self.proj_out = nn.Linear(self.video_inner_dim, out_channels)

        # =================
        # AUDIO COMPONENTS
        # =================
        if self.model_type.is_audio_enabled():
            # Input projection
            self.audio_patchify_proj = nn.Linear(
                self.AUDIO_IN_CHANNELS, self.audio_inner_dim, bias=True
            )

            # AdaLN
            self.audio_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=adaln_num_embeddings
            )

            # V2: prompt adaln for audio cross-attention KV modulation
            self.audio_prompt_adaln_single = (
                AdaLayerNormSingle(self.audio_inner_dim, num_embeddings=2)
                if cross_attention_adaln else None
            )

            # Caption projection (None for V2 models where feature extractor projects directly)
            if caption_channels is not None:
                self.audio_caption_projection = PixArtAlphaTextProjection(
                    in_features=caption_channels,
                    hidden_size=self.audio_inner_dim,
                )
            else:
                self.audio_caption_projection = None

            # Output projection
            self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim), dtype=mx.float32)
            self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, affine=False, eps=norm_eps)
            self.audio_proj_out = nn.Linear(self.audio_inner_dim, self.AUDIO_OUT_CHANNELS)

        # =================
        # CROSS-MODAL COMPONENTS
        # =================
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            # Video side
            self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=4
            )
            self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=1
            )
            # Audio side
            self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=4
            )
            self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=1
            )

        # =================
        # TRANSFORMER BLOCKS
        # =================
        video_config = None
        if self.model_type.is_video_enabled():
            video_config = TransformerConfig(
                dim=self.video_inner_dim,
                heads=num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                cross_attention_adaln=cross_attention_adaln,
                apply_gated_attention=apply_gated_attention,
            )

        audio_config = None
        if self.model_type.is_audio_enabled():
            audio_config = TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.AUDIO_ATTENTION_HEADS,
                d_head=self.AUDIO_HEAD_DIM,
                context_dim=self.audio_inner_dim,  # 2048, not 4096 — matches PyTorch audio_cross_attention_dim
                cross_attention_adaln=cross_attention_adaln,
                apply_gated_attention=apply_gated_attention,
            )

        self.transformer_blocks = [
            BasicAVTransformerBlock(
                idx=i,
                video_config=video_config,
                audio_config=audio_config,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for i in range(num_layers)
        ]
        
        # =================
        # PREPROCESSORS
        # =================
        if self.model_type.is_video_enabled():
            video_simple_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.video_inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                compute_dtype=self.compute_dtype,
                prompt_adaln=self.prompt_adaln_single,
            )
            if self.model_type.is_audio_enabled():
                self._video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=video_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._video_args_preprocessor = video_simple_preprocessor

        if self.model_type.is_audio_enabled():
            audio_simple_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=[self.AUDIO_CROSS_PE_MAX_POS],
                num_attention_heads=self.AUDIO_ATTENTION_HEADS,
                use_middle_indices_grid=True,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                compute_dtype=self.compute_dtype,
                prompt_adaln=self.audio_prompt_adaln_single,
            )
            if self.model_type.is_video_enabled():
                self._audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=audio_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._audio_args_preprocessor = audio_simple_preprocessor

    def enable_video_ff_quantization(
        self,
        quantization_specs: Tuple[Tuple[str, str], ...],
        group_size: int | None = None,
        bits: int | None = None,
        layers: Tuple[int, ...] = (),
    ) -> int:
        """Quantize selected video feed-forward projections in-place for experiments."""
        arrays: List[mx.array] = []
        count = 0
        selected_layers = set(layers)
        for i, block in enumerate(self.transformer_blocks):
            if selected_layers and i not in selected_layers:
                continue
            ff = getattr(block, "ff", None)
            if ff is None:
                continue
            for target, target_mode in quantization_specs:
                arrays.extend(ff.quantize_projections(
                    targets=(target,),
                    mode=target_mode,
                    group_size=group_size,
                    bits=bits,
                ))
            count += len(quantization_specs)

        if arrays:
            mx.eval(*arrays)
        return count

    def apply_video_ff_layout(
        self,
        layout_specs: Tuple[Tuple[str, str], ...],
        layers: Tuple[int, ...] = (),
    ) -> int:
        """Apply selected same-math video feed-forward layout transforms."""
        count = 0
        selected_layers = set(layers)
        for i, block in enumerate(self.transformer_blocks):
            if selected_layers and i not in selected_layers:
                continue
            ff = getattr(block, "ff", None)
            if ff is None:
                continue
            arrays = ff.apply_layouts(layout_specs)
            if arrays:
                mx.eval(*arrays)
                ff.drop_layout_sources(layout_specs)
                gc.collect()
                mx.clear_cache()
            count += len(layout_specs)

        return count

    def apply_video_attn_layout(
        self,
        layout_specs: Tuple[Tuple[str, str], ...],
        layers: Tuple[int, ...] = (),
    ) -> int:
        """Apply selected same-math video attention layout transforms."""
        count = 0
        selected_layers = set(layers)
        output_specs = tuple(
            spec for spec in layout_specs
            if spec[0] == "to_out"
        )
        unsupported_specs = tuple(
            spec for spec in layout_specs
            if spec[0] != "to_out"
        )
        if unsupported_specs:
            spec_str = ",".join(f"{target}:{layout}" for target, layout in unsupported_specs)
            raise ValueError(f"Unsupported video attention layout specs: {spec_str}")

        for i, block in enumerate(self.transformer_blocks):
            if selected_layers and i not in selected_layers:
                continue

            if output_specs:
                attention_modules = [
                    getattr(block, "attn1", None),
                    getattr(block, "attn2", None),
                    getattr(block, "audio_to_video_attn", None),
                ]
                for attn in attention_modules:
                    if attn is None:
                        continue
                    arrays = attn.apply_layouts(output_specs)
                    if arrays:
                        mx.eval(*arrays)
                        attn.drop_layout_sources(output_specs)
                        gc.collect()
                        mx.clear_cache()
                    count += len(output_specs)

        return count

    # Audio layer debug: set to a directory path to capture per-layer audio states
    _audio_layer_debug_dir: Optional[str] = None
    _audio_layer_debug_done: bool = False

    def _collect_profile_arrays(
        self,
        args: Optional[TransformerArgs],
    ) -> List[mx.array]:
        """Collect representative arrays that materialize a preprocessor/output stage."""
        if args is None:
            return []

        arrays = [args.x, args.context, args.timesteps]
        if args.embedded_timestep is not None:
            arrays.append(args.embedded_timestep)
        if args.positional_embeddings is not None:
            arrays.extend(args.positional_embeddings)
        if args.cross_positional_embeddings is not None:
            arrays.extend(args.cross_positional_embeddings)
        if args.cross_scale_shift_timestep is not None:
            arrays.append(args.cross_scale_shift_timestep)
        if args.cross_gate_timestep is not None:
            arrays.append(args.cross_gate_timestep)
        if args.prompt_timestep is not None:
            arrays.append(args.prompt_timestep)
        return arrays

    def _print_transformer_profile(self, events: List[Tuple[str, float]]) -> None:
        """Print a compact selected-call transformer profile."""
        total = sum(seconds for _, seconds in events)
        label = f" {self.profile_transformer_label}" if self.profile_transformer_label else ""
        print(f"\n  Transformer profile{label} (forced eval diagnostics):")
        for name, seconds in events:
            pct = (seconds / total * 100.0) if total > 0 else 0.0
            print(f"    {name:<24} {seconds:7.2f}s  {pct:5.1f}%")
        print(f"    {'profiled total':<24} {total:7.2f}s")
        self.profile_transformer_label = None

    def _print_transformer_block_profiles(
        self,
        traces: List[Tuple[int, List[Tuple[str, float]]]],
    ) -> None:
        """Print detailed profiles for selected transformer blocks."""
        if not traces:
            return
        label = f" {self.profile_transformer_label}" if self.profile_transformer_label else ""
        for block_idx, events in traces:
            total = sum(seconds for _, seconds in events)
            print(f"\n  Transformer block {block_idx:02d} profile{label} (forced eval diagnostics):")
            for name, seconds in events:
                pct = (seconds / total * 100.0) if total > 0 else 0.0
                print(f"    {name:<24} {seconds:7.2f}s  {pct:5.1f}%")
            print(f"    {'profiled total':<24} {total:7.2f}s")

    def _process_transformer_blocks(
        self,
        video_args: Optional[TransformerArgs] = None,
        audio_args: Optional[TransformerArgs] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
        profile_events: Optional[List[Tuple[str, float]]] = None,
        profile_block_traces: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Process transformer blocks."""
        capture = (
            self._audio_layer_debug_dir is not None
            and not self._audio_layer_debug_done
            and audio_args is not None
            and audio_args.enabled
        )
        profile_group_size = 8
        profile_group_start = 0
        profile_group_started_at = time.perf_counter() if profile_events is not None else 0.0
        profile_block_ids = (
            set(self.profile_transformer_blocks)
            if profile_events is not None
            else set()
        )

        block_streamer = self.transformer_block_streamer
        total_blocks = (
            block_streamer.block_count
            if block_streamer is not None
            else len(self.transformer_blocks)
        )
        resident_blocks = len(self.transformer_blocks)
        previous_slot_layers: List[Optional[int]] = [None] * resident_blocks
        use_compiled_groups = (
            block_streamer is not None
            and self.transformer_block_compile
            and not self._transformer_block_compile_disabled
            and perturbations is None
            and not profile_block_ids
            and not capture
            and (video_args is None or video_args.enabled)
            and (audio_args is None or audio_args.enabled)
        )
        if use_compiled_groups:
            compiled_groups = object.__getattribute__(self, "_compiled_transformer_block_groups")
            if not isinstance(compiled_groups, dict):
                compiled_groups = {}
                object.__setattr__(self, "_compiled_transformer_block_groups", compiled_groups)

            compiled_group_enabled = True
            for group_start in range(0, total_blocks, resident_blocks):
                group_size = min(resident_blocks, total_blocks - group_start)
                for offset in range(group_size):
                    block_idx = group_start + offset
                    block_streamer.bind(
                        self.transformer_blocks[offset],
                        block_idx,
                        evict_block_idx=previous_slot_layers[offset],
                    )
                    previous_slot_layers[offset] = block_idx

                if compiled_group_enabled:
                    group_callable = compiled_groups.get(group_size)
                    if group_callable is None:
                        group_callable = _compile_transformer_block_group(
                            self.transformer_blocks[:group_size]
                        )
                        compiled_groups[group_size] = group_callable

                    try:
                        video_packed, audio_packed = group_callable(
                            _pack_transformer_args(video_args),
                            _pack_transformer_args(audio_args),
                        )
                        video_args = _unpack_transformer_args(video_packed)
                        audio_args = _unpack_transformer_args(audio_packed)
                    except (RuntimeError, TypeError, ValueError) as exc:
                        compiled_group_enabled = False
                        self._transformer_block_compile_disabled = True
                        object.__setattr__(self, "_compiled_transformer_block_groups", {})
                        print(
                            "  WARNING: Transformer resident group compile failed; "
                            f"falling back to eager streaming ({exc})"
                        )

                if not compiled_group_enabled:
                    for offset in range(group_size):
                        block = self.transformer_blocks[offset]
                        video_args, audio_args = block(
                            video_args,
                            audio_args,
                            perturbations=perturbations,
                            profile_events=None,
                        )

                # Streaming can only evict/rebind a resident slot after the
                # previous window's graph is materialized.
                arrays = []
                if video_args is not None:
                    arrays.append(video_args.x)
                if audio_args is not None:
                    arrays.append(audio_args.x)
                if arrays:
                    mx.eval(*arrays)
                if profile_events is not None:
                    now = time.perf_counter()
                    profile_events.append((
                        f"blocks {group_start:02d}-{group_start + group_size - 1:02d}",
                        now - profile_group_started_at,
                    ))
                    profile_group_started_at = now

            return video_args, audio_args

        for i in range(total_blocks):
            slot = i
            if block_streamer is not None:
                slot = i % resident_blocks
                block = block_streamer.bind(
                    self.transformer_blocks[slot],
                    i,
                    evict_block_idx=previous_slot_layers[slot],
                )
                previous_slot_layers[slot] = i
            else:
                block = self.transformer_blocks[i]
            block_profile_events = [] if i in profile_block_ids else None
            if block_profile_events is not None:
                arrays = []
                if video_args is not None:
                    arrays.append(video_args.x)
                if audio_args is not None and audio_args.enabled:
                    arrays.append(audio_args.x)
                started_at = time.perf_counter()
                if arrays:
                    mx.eval(*arrays)
                block_profile_events.append(("entry sync", time.perf_counter() - started_at))
            video_args, audio_args = block(
                video_args,
                audio_args,
                perturbations=perturbations,
                profile_events=block_profile_events,
            )
            if (
                block_profile_events is not None
                and profile_block_traces is not None
            ):
                profile_block_traces.append((i, block_profile_events))

            # Reduce eval frequency for performance
            eval_frequency = self._eval_frequency
            if block_streamer is not None:
                eval_frequency = resident_blocks

            if eval_frequency > 0 and (i + 1) % eval_frequency == 0:
                arrays = []
                if video_args is not None:
                    arrays.append(video_args.x)
                if audio_args is not None:
                    arrays.append(audio_args.x)
                if arrays:
                    mx.eval(*arrays)

            if profile_events is not None and (
                (i + 1) % profile_group_size == 0 or i == total_blocks - 1
            ):
                arrays = []
                if video_args is not None:
                    arrays.append(video_args.x)
                if audio_args is not None and audio_args.enabled:
                    arrays.append(audio_args.x)
                if arrays:
                    mx.eval(*arrays)
                now = time.perf_counter()
                profile_events.append((
                    f"blocks {profile_group_start:02d}-{i:02d}",
                    now - profile_group_started_at,
                ))
                profile_group_start = i + 1
                profile_group_started_at = now

            # Capture audio state after each block (first denoising step only)
            if capture and audio_args is not None:
                import numpy as _np
                mx.eval(audio_args.x)
                path = f"{self._audio_layer_debug_dir}/audio_layer_{i:04d}.npy"
                _np.save(path, _np.array(audio_args.x.astype(mx.float32)))
                if i == 0:
                    print(f"  [debug] Capturing audio layer states...")
                if i == total_blocks - 1:
                    print(f"  [debug] Saved {i+1} layer states to {self._audio_layer_debug_dir}")
                    self._audio_layer_debug_done = True

        return video_args, audio_args

    def _process_video_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process video output."""
        scale_shift_values = (
            self.scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        ).astype(x.dtype)
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)
        return x
        
    def _process_audio_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process audio output."""
        scale_shift_values = (
            self.audio_scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        ).astype(x.dtype)
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = self.audio_norm_out(x)
        x = x * (1 + scale) + shift
        x = self.audio_proj_out(x)
        return x

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            video: Input video modality (required for VideoOnly/AudioVideo).
            audio: Input audio modality (required for AudioVideo/AudioOnly).
            perturbations: Optional perturbation config for STG guidance.
                Supports 4 types: skip_video_self_attn, skip_audio_self_attn,
                skip_a2v_cross_attn, skip_v2a_cross_attn.

        Returns:
            VideoOnly: video_velocity
            AudioOnly: audio_velocity
            AudioVideo: (video_velocity, audio_velocity)
        """
        profile_this_call = self.profile_transformer_once
        self.profile_transformer_once = False
        profile_events: Optional[List[Tuple[str, float]]] = [] if profile_this_call else None
        profile_block_traces: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = (
            [] if profile_this_call and self.profile_transformer_blocks else None
        )
        profile_started_at = time.perf_counter() if profile_events is not None else 0.0

        # --- Type Casting ---
        if self.compute_dtype != mx.float32:
            if video is not None:
                video = Modality(
                    latent=video.latent.astype(self.compute_dtype),
                    context=video.context.astype(self.compute_dtype),
                    context_mask=video.context_mask,
                    timesteps=video.timesteps,
                    positions=video.positions,
                    enabled=video.enabled,
                    sigma=video.sigma,
                )
            if audio is not None:
                audio = Modality(
                    latent=audio.latent.astype(self.compute_dtype),
                    context=audio.context.astype(self.compute_dtype),
                    context_mask=audio.context_mask,
                    timesteps=audio.timesteps,
                    positions=audio.positions,
                    enabled=audio.enabled,
                    sigma=audio.sigma,
                )
        if profile_events is not None:
            now = time.perf_counter()
            profile_events.append(("input dtype wrappers", now - profile_started_at))
            profile_started_at = now

        # --- Preprocessing ---
        video_args = None
        if self.model_type.is_video_enabled():
            if video is None:
                raise ValueError("Video modality required for video-enabled model")
            video_args = self._video_args_preprocessor.prepare(video, audio)

        audio_args = None
        if self.model_type.is_audio_enabled():
            if audio is None:
                # Create empty audio modality (video-only inference on AV model)
                batch_size = video_args.x.shape[0] if video_args else 1
                audio = Modality(
                    latent=mx.zeros((batch_size, 0, self.audio_inner_dim), dtype=self.compute_dtype),
                    context=mx.zeros((batch_size, 0, self.audio_inner_dim), dtype=self.compute_dtype),
                    context_mask=None,
                    timesteps=mx.zeros((batch_size,), dtype=self.compute_dtype),
                    positions=mx.zeros((batch_size, 3, 0), dtype=self.compute_dtype),
                    enabled=False,
                )
            # Only preprocess if has tokens
            if audio.latent.size > 0:
                audio_args = self._audio_args_preprocessor.prepare(audio, video)
            else:
                 # Minimal dummy args (should be handled by block enabled check, but safe fallback)
                audio_args = TransformerArgs(
                    x=mx.zeros(
                        (video_args.x.shape[0] if video_args else 1, 0, self.audio_inner_dim),
                        dtype=self.compute_dtype,
                    ),
                    context=mx.zeros((1, 0, self.audio_inner_dim), dtype=self.compute_dtype),
                    timesteps=mx.zeros((1, 0, 6, self.audio_inner_dim), dtype=self.compute_dtype),
                    positional_embeddings=(
                        mx.zeros((1,), dtype=self.compute_dtype),
                        mx.zeros((1,), dtype=self.compute_dtype),
                    ),
                    enabled=False,
                )
        if profile_events is not None:
            arrays = self._collect_profile_arrays(video_args) + self._collect_profile_arrays(audio_args)
            if arrays:
                mx.eval(*arrays)
            now = time.perf_counter()
            profile_events.append(("preprocess", now - profile_started_at))
            profile_started_at = now

        # --- Transformer Blocks ---
        video_args, audio_args = self._process_transformer_blocks(
            video_args,
            audio_args,
            perturbations=perturbations,
            profile_events=profile_events,
            profile_block_traces=profile_block_traces,
        )
        if profile_events is not None:
            profile_started_at = time.perf_counter()

        # --- Output Processing ---
        video_out = None
        if self.model_type.is_video_enabled():
            video_out = self._process_video_output(video_args.x, video_args.embedded_timestep)

        audio_out = None
        current_batch_size = video_out.shape[0] if video_out is not None else (audio.latent.shape[0] if audio else 1)
        if self.model_type.is_audio_enabled():
            if audio_args.enabled and audio_args.x.size > 0:
                audio_out = self._process_audio_output(audio_args.x, audio_args.embedded_timestep)
            else:
                audio_out = mx.zeros(
                    (current_batch_size, 0, self.AUDIO_OUT_CHANNELS),
                    dtype=self.compute_dtype,
                )
        if profile_events is not None:
            arrays = []
            if video_out is not None:
                arrays.append(video_out)
            if audio_out is not None:
                arrays.append(audio_out)
            if arrays:
                mx.eval(*arrays)
            now = time.perf_counter()
            profile_events.append(("output projection", now - profile_started_at))
            if profile_block_traces is not None:
                self._print_transformer_block_profiles(profile_block_traces)
            self._print_transformer_profile(profile_events)
            self.profile_transformer_blocks = ()

        # --- Return Logic ---
        if self.model_type == LTXModelType.VideoOnly:
            return video_out
        elif self.model_type == LTXModelType.AudioOnly:
            return audio_out
        else:
            return video_out, audio_out


class X0Model(nn.Module):
    """
    Wrapper that returns denoised outputs instead of velocities.
    
    Handles both VideoOnly (returns tensor) and AudioVideo (returns tuple).
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
        perturbations: Optional[BatchedPerturbationConfig] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Compute denoised outputs.

        Args:
            video: Video modality.
            audio: Audio modality.
            perturbations: Optional perturbation config for STG guidance.
        """
        output = self.velocity_model(video, audio, perturbations=perturbations)

        # Helper to denoise
        def denoise(modality, velocity):
            timesteps = modality.timesteps
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
            elif timesteps.ndim == 2:
                timesteps = timesteps[:, :, None]
            return modality.latent - timesteps * velocity

        if isinstance(output, tuple):
            # AudioVideo case
            video_vel, audio_vel = output
            denoised_video = denoise(video, video_vel)
            if audio is not None:
                denoised_audio = denoise(audio, audio_vel)
            else:
                # Video-only inference on AV model — return video only
                return denoised_video
            return denoised_video, denoised_audio
        else:
            # VideoOnly or AudioOnly case
            if video is not None:
                return denoise(video, output)
            elif audio is not None:
                return denoise(audio, output)
            return output


# Aliases for backward compatibility
LTXAVModel = LTXModel
X0AVModel = X0Model
