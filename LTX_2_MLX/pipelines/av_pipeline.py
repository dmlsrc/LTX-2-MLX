"""Single-stage text/image-to-video generation pipeline for LTX-2 MLX.

This pipeline provides standard CFG-based video generation in a single pass:
  - Uses LTX2Scheduler for dev sigma schedules, fixed sigmas for distilled
  - Classifier-free guidance with positive/negative prompts
  - Optional image conditioning via latent replacement
  - Optional audio generation via AudioVideo transformer

This is the most common pipeline for high-quality video generation.
"""

import gc
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# LTX_VELOCITY_MODE=1 makes the simple AV distilled loop bypass X0Model and
# do the velocity-form Euler update inline:
#     denoised = latent - sigma * velocity   (skipped when no mask blend needed)
#     next     = latent + (sigma_next - sigma) * velocity
# Mathematically identical to the X0+stepper path; the goal is a leaner MLX
# graph (fewer wrapper-class boundaries between transformer call and stepper).
# Falls back to the X0 path automatically when state has a non-uniform mask
# (image conditioning) or when the wrapped transformer isn't an X0Model.
_USE_VELOCITY_MODE = bool(os.environ.get("LTX_VELOCITY_MODE"))
_NORMALIZE_AUDIO_NOISE = bool(os.environ.get("LTX_NORMALIZE_AUDIO_NOISE"))


def _env_enabled(name: str, disable_name: str | None = None) -> bool:
    false_values = {"", "0", "false", "no", "off"}
    if (
        disable_name is not None
        and os.environ.get(disable_name, "").strip().lower() not in false_values
    ):
        return False
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in false_values


_PRECOMPUTE_ROPE = _env_enabled(
    "LTX_ROPE_PRECOMPUTE",
    "LTX_DISABLE_ROPE_PRECOMPUTE",
)

from ..components import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    CFGGuider,
    CFGStarRescalingGuider,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    VideoLatentPatchifier,
)
from ..components.diffusion_steps import HeunDiffusionStep
from ..components.guiders import STGGuider
from ..components.patchifiers import AudioPatchifier
from ..components.perturbations import BatchedPerturbationConfig, create_batched_stg_config
from ..conditioning.tools import AudioLatentTools, VideoLatentTools
from ..loader import (
    LoRAConfig,
    format_lora_stage_scale_lines,
    fuse_loras_into_model,
    get_transformer_cache_restore_state,
    lora_configs_for_stage,
    lora_configs_for_stage_delta,
    restore_transformer_cache_state,
)
from ..model.audio_vae import AudioDecoder, Vocoder
from ..model.transformer import LTXModel, LTXModelType, X0Model
from ..model.video_vae.decode_utils import decode_latent
from ..model.video_vae.native_decoder import NativeConv3dVideoDecoder
from ..model.video_vae.native_encoder import NativeConv3dVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..types import NATIVE_FPS, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from .common import (
    ImageCondition,
    apply_conditionings,
    audio_modality_from_state,
    create_image_conditionings,
    maybe_post_process_latent,
    modality_from_state,
)


@dataclass
class AVCFGConfig:
    """Configuration for single-stage CFG pipeline."""

    # Video dimensions
    height: int = 480
    width: int = 704
    num_frames: int = 97  # Must be 8k + 1

    # Generation parameters
    seed: int = 42
    fps: float = NATIVE_FPS
    num_inference_steps: int = 30
    use_distilled_sigmas: bool = False

    # CFG parameters (matching LTX-2.3 reference defaults)
    cfg_scale: float = 3.0           # Video text guidance
    audio_cfg_scale: float = 7.0     # Audio text guidance (higher for better audio conditioning)
    rescale_scale: float = 0.7       # Restore MLX known-good short-clip baseline

    # Tiling for VAE decoding (enabled by default to prevent Metal watchdog crashes on long videos)
    tiling_config: TilingConfig | None = None
    auto_tiling: bool = True

    def _get_tiling_config(self) -> TilingConfig | None:
        """Return tiling config, auto-enabling."""
        if self.tiling_config is not None:
            return self.tiling_config
        if not self.auto_tiling:
            return None
        return TilingConfig.auto(self.height, self.width, self.num_frames)

    # Compute settings
    dtype: mx.Dtype = mx.bfloat16
    profile_transformer_once: bool = False
    profile_transformer_steps: tuple[int, ...] = ()
    profile_transformer_blocks: tuple[int, ...] = ()

    # Audio configuration
    audio_enabled: bool = False
    use_internal_audio_branch: bool = True
    audio_vae_channels: int = 8
    audio_mel_bins: int = 16
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_downsample_factor: int = 4
    audio_output_sample_rate: int = 24000

    def __post_init__(self):
        if self.num_frames % 8 != 1:
            raise ValueError(
                f"num_frames must be 8*k + 1, got {self.num_frames}. "
                f"Valid values: 1, 9, 17, 25, 33, ..., 121"
            )
        # For single-stage, resolution must be divisible by 32
        if self.height % 32 != 0 or self.width % 32 != 0:
            raise ValueError(
                f"Resolution ({self.height}x{self.width}) "
                f"must be divisible by 32 for single-stage pipeline."
            )


class AVPipeline:
    """
    Single-stage text/image-to-video generation pipeline.

    This pipeline generates video at target resolution in a single diffusion pass
    with classifier-free guidance (CFG). Supports optional image conditioning.

    Features:
    - Uses LTX2Scheduler for dev sigma schedules, fixed sigmas for distilled
    - CFG with positive/negative prompts for quality on dev checkpoints
    - Optional image conditioning via latent replacement
    - Optional joint audio-video generation via AudioVideo transformer
    """

    def __init__(
        self,
        transformer: LTXModel,
        video_encoder: NativeConv3dVideoEncoder | None,
        video_decoder: NativeConv3dVideoDecoder | None,
        audio_decoder: AudioDecoder | None = None,
        vocoder: Vocoder | None = None,
        video_decoder_loader: Callable[[], NativeConv3dVideoDecoder] | None = None,
        audio_decoder_loader: Callable[[], tuple[AudioDecoder, object, int]] | None = None,
        audio_sample_rate: int = 24000,
    ):
        """
        Initialize the single-stage pipeline.

        Args:
            transformer: LTX transformer model (LTXModel for video-only, LTXAVModel for audio+video).
            video_encoder: Optional VAE encoder for image conditioning.
            video_decoder: Optional VAE decoder for decoding latents to video.
            audio_decoder: Optional audio VAE decoder for decoding audio latents to mel spectrograms.
            vocoder: Optional vocoder for converting mel spectrograms to waveforms.
            video_decoder_loader: Optional lazy loader for the VAE decoder.
            audio_decoder_loader: Optional lazy loader for the audio VAE decoder + vocoder.
            audio_sample_rate: Output sample rate to use before a lazy vocoder has been loaded.
        """
        # Wrap transformer in X0Model if needed
        # LTXModel outputs velocity, but denoising expects denoised (X0) predictions
        if isinstance(transformer, X0Model):
            self.transformer = transformer
        else:
            self.transformer = X0Model(transformer)
        inner = self.transformer.velocity_model if hasattr(self.transformer, "velocity_model") else transformer
        self.is_av_model = getattr(inner, "model_type", None) == LTXModelType.AudioVideo
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.video_decoder_loader = video_decoder_loader
        self.audio_decoder_loader = audio_decoder_loader
        self.audio_sample_rate = (
            getattr(vocoder, "output_sample_rate", audio_sample_rate)
            if vocoder is not None
            else audio_sample_rate
        )
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
        self.diffusion_step = EulerDiffusionStep()
        self.scheduler = LTX2Scheduler()
        self.last_timing_sections: list[tuple[str, float]] = []

    def _velocity_transformer(self):
        """Return the wrapped velocity model when the pipeline uses X0Model."""
        return (
            self.transformer.velocity_model
            if hasattr(self.transformer, "velocity_model")
            else self.transformer
        )

    def _profile_next_transformer_call(
        self,
        label: str,
        blocks: tuple[int, ...] = (),
    ) -> None:
        """Arm the inner transformer profiler for the next model call."""
        model = self._velocity_transformer()
        if hasattr(model, "profile_transformer_once"):
            model.profile_transformer_once = True
            model.profile_transformer_label = label
            model.profile_transformer_blocks = blocks

    @staticmethod
    def _latent_to_numpy(latent: mx.array) -> np.ndarray:
        """Convert an MLX latent to a NumPy array suitable for npz storage."""
        mx.eval(latent)
        try:
            return np.array(latent)
        except (TypeError, RuntimeError):
            return np.array(latent.astype(mx.float32))

    @classmethod
    def _save_final_latents(
        cls,
        path: str,
        video_latent: mx.array,
        audio_latent: mx.array | None = None,
        progress_message: Callable[[str], None] | None = None,
    ) -> None:
        """Save final video/audio latents as a sidecar npz file.

        See `_save_distilled_two_stage_latents` for the
        `progress_message` rationale - same fallback semantics.
        """
        arrays = {
            "final_video_latent": cls._latent_to_numpy(video_latent),
            "final_video_latent_mlx_dtype": str(video_latent.dtype),
        }
        if audio_latent is not None:
            arrays["final_audio_latent"] = cls._latent_to_numpy(audio_latent)
            arrays["final_audio_latent_mlx_dtype"] = str(audio_latent.dtype)

        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(path, **arrays)
        message = f"  Saved final latents: {path}"
        if progress_message is not None:
            progress_message(message)
        else:
            print(message)

    @classmethod
    def _save_distilled_two_stage_latents(
        cls,
        path: str,
        stage_1_video_latent: mx.array,
        stage_2_video_latent: mx.array,
        stage_1_audio_latent: mx.array | None = None,
        stage_2_audio_latent: mx.array | None = None,
        progress_message: Callable[[str], None] | None = None,
    ) -> None:
        """Save both distilled stages while preserving the final-latent keys.

        When `progress_message` is supplied the "Saved distilled stage
        latents: ..." confirmation is routed through it instead of a
        direct `print()` - necessary when the caller is rendering
        progress bars that would otherwise be stomped by a raw stdout
        write.  Default (no callable) falls back to `print()` for
        callers that don't have a progress UI active.
        """
        stage_1_video_np = cls._latent_to_numpy(stage_1_video_latent)
        stage_2_video_np = cls._latent_to_numpy(stage_2_video_latent)
        arrays = {
            "pipeline": np.array("distilled_two_stage"),
            "stage_1_video_latent": stage_1_video_np,
            "stage_1_video_latent_mlx_dtype": str(stage_1_video_latent.dtype),
            "stage_2_video_latent": stage_2_video_np,
            "stage_2_video_latent_mlx_dtype": str(stage_2_video_latent.dtype),
            "final_video_latent": stage_2_video_np,
            "final_video_latent_mlx_dtype": str(stage_2_video_latent.dtype),
        }
        if stage_1_audio_latent is not None:
            arrays["stage_1_audio_latent"] = cls._latent_to_numpy(stage_1_audio_latent)
            arrays["stage_1_audio_latent_mlx_dtype"] = str(stage_1_audio_latent.dtype)
        if stage_2_audio_latent is not None:
            stage_2_audio_np = cls._latent_to_numpy(stage_2_audio_latent)
            arrays["stage_2_audio_latent"] = stage_2_audio_np
            arrays["stage_2_audio_latent_mlx_dtype"] = str(stage_2_audio_latent.dtype)
            arrays["final_audio_latent"] = stage_2_audio_np
            arrays["final_audio_latent_mlx_dtype"] = str(stage_2_audio_latent.dtype)

        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(path, **arrays)
        message = f"  Saved distilled stage latents: {path}"
        if progress_message is not None:
            progress_message(message)
        else:
            print(message)

    def _create_video_tools(
        self,
        target_shape: VideoLatentShape,
        fps: float,
    ) -> VideoLatentTools:
        """Create video latent tools for the target shape."""
        return VideoLatentTools(
            patchifier=self.patchifier,
            target_shape=target_shape,
            fps=fps,
        )

    def _create_audio_tools(
        self,
        target_shape: AudioLatentShape,
    ) -> AudioLatentTools:
        """Create audio latent tools for the target shape."""
        return AudioLatentTools(
            patchifier=self.audio_patchifier,
            target_shape=target_shape,
        )

    def _ensure_video_decoder(
        self,
        progress_message: Callable[[str], None] | None = None,
    ) -> tuple[float, bool]:
        """Load the video VAE decoder on demand.

        Returns ``(elapsed_seconds, loaded_now)`` so callers can time and
        release only the components that this helper materialized.
        """
        if self.video_decoder is not None:
            return 0.0, False
        if self.video_decoder_loader is None:
            raise ValueError("Video decoder required for VAE decode.")

        emit = progress_message or print
        emit("  Loading VAE decoder...")
        load_start = time.perf_counter()
        self.video_decoder = self.video_decoder_loader()
        load_elapsed = time.perf_counter() - load_start
        if self.video_decoder is None:
            raise ValueError("Video decoder loader returned None.")
        emit(f"  VAE decoder load complete in {load_elapsed:.1f}s")
        return load_elapsed, True

    def _ensure_audio_decoder(
        self,
        progress_message: Callable[[str], None] | None = None,
    ) -> tuple[float, bool]:
        """Load the audio VAE decoder + vocoder on demand."""
        if self.audio_decoder is not None and self.vocoder is not None:
            return 0.0, False
        if self.audio_decoder_loader is None:
            raise ValueError("Audio decoder and vocoder required for audio decode.")

        emit = progress_message or print
        emit("  Loading audio decoder/vocoder...")
        load_start = time.perf_counter()
        audio_decoder, vocoder, audio_sample_rate = self.audio_decoder_loader()
        load_elapsed = time.perf_counter() - load_start
        if audio_decoder is None or vocoder is None:
            raise ValueError("Audio decoder loader returned incomplete decode stack.")
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.audio_sample_rate = audio_sample_rate
        emit(f"  Audio decoder/vocoder load complete in {load_elapsed:.1f}s")
        return load_elapsed, True

    def _decode_audio(self, audio_latent: mx.array) -> mx.array:
        """
        Decode audio latent to waveform via VAE decoder + vocoder.

        Args:
            audio_latent: Audio latent tensor [B, C, F, mel_bins].

        Returns:
            Audio waveform tensor [B, channels, samples].
        """
        if self.audio_decoder is None or self.vocoder is None:
            raise ValueError("Audio decoder and vocoder required for audio decode.")

        # Decode latent to mel spectrogram (output is log-mel, which vocoder expects)
        mel_spectrogram = self.audio_decoder(audio_latent)
        mx.eval(mel_spectrogram)

        # Convert mel spectrogram to waveform (vocoder takes log-mel directly)
        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.astype(mx.float32)
        mx.eval(waveform)

        return waveform

    @staticmethod
    def _channelwise_normalize_audio_noise(latent: mx.array) -> mx.array:
        """Normalize pure audio noise so duration changes do not suppress amplitude."""
        x = (latent - mx.mean(latent)) / (mx.std(latent) + 1e-8)
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    def _apply_cross_attn_scales(self, scale: float, start_block: int):
        """Set cross-attention scale on late transformer blocks."""
        model = self.transformer.model if hasattr(self.transformer, 'model') else self.transformer
        blocks = getattr(model, 'transformer_blocks', [])
        for i, block in enumerate(blocks):
            if i >= start_block:
                block._cross_attn_scale = scale
            else:
                block._cross_attn_scale = None

    def _clear_cross_attn_scales(self):
        """Remove cross-attention scaling from all blocks."""
        model = self.transformer.model if hasattr(self.transformer, 'model') else self.transformer
        blocks = getattr(model, 'transformer_blocks', [])
        for block in blocks:
            block._cross_attn_scale = None

    def _denoise_loop_cfg(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        negative_context: mx.array,
        guider,
        stepper: EulerDiffusionStep,
        callback: Callable[[int, int], None] | None = None,
        stg_guider: STGGuider | None = None,
        stg_perturbations: BatchedPerturbationConfig | None = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> LatentState:
        """
        Run the denoising loop with guidance and optional STG + GE.

        Args:
            video_state: Initial noisy video latent state.
            sigmas: Sigma schedule.
            positive_context: Positive text context.
            negative_context: Negative text context.
            guider: Any guider implementing GuiderProtocol (CFG, CFG*, APG, etc.).
            stepper: Diffusion stepper.
            callback: Optional callback(step, total_steps).
            stg_guider: Optional STG guider for temporal coherence.
            stg_perturbations: Perturbation config for STG (skip video self-attn).
            stg_cutoff: Apply STG for first N fraction of steps (0.0-1.0).
            ge_gamma: GE (Gradient Estimation) velocity correction factor. 0.0 = disabled.

        Returns:
            Denoised latent state.
        """
        num_steps = len(sigmas) - 1
        prev_velocity = None  # GE velocity tracking

        # Apply cross-attention scaling if non-default
        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)
            print(f"  Cross-attn scaling: {cross_attn_scale}x on blocks {cross_attn_start_block}-47")

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])

            # Run positive (conditioned) prediction
            pos_modality = modality_from_state(
                video_state, positive_context, sigma
            )
            pos_denoised = self.transformer(pos_modality)

            # Run negative (unconditioned) prediction for CFG
            if guider.enabled():
                neg_modality = modality_from_state(
                    video_state, negative_context, sigma
                )
                neg_denoised = self.transformer(neg_modality)

                # Apply guidance (CFG, CFG*, APG, etc.)
                denoised = guider.guide(pos_denoised, neg_denoised)
            else:
                denoised = pos_denoised

            # Apply STG (Spatio-Temporal Guidance) if enabled and within cutoff
            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_denoised = self.transformer(pos_modality, perturbations=stg_perturbations)
                denoised = stg_guider.guide(denoised, perturbed_denoised)
                del perturbed_denoised

            # Apply GE (Gradient Estimation) velocity correction
            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            # Post-process with denoise mask
            denoised = maybe_post_process_latent(denoised, video_state)

            # Euler step
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.async_eval(video_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        # Clean up cross-attention scaling
        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state

    def _denoise_loop_heun(
        self,
        video_state: LatentState,
        sigmas: mx.array,
        positive_context: mx.array,
        negative_context: mx.array,
        guider,
        stepper: HeunDiffusionStep,
        callback: Callable[[int, int], None] | None = None,
        stg_guider: STGGuider | None = None,
        stg_perturbations: BatchedPerturbationConfig | None = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> LatentState:
        """
        Run the denoising loop with Heun (predictor-corrector) stepping.

        Heun uses two model evaluations per step:
        1. Euler prediction at current sigma
        2. Model eval at predicted point (next sigma)
        3. Average velocities for corrector step

        This gives higher accuracy per step at 2x compute cost.
        """
        num_steps = len(sigmas) - 1
        prev_velocity = None

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])

            # -- First evaluation: denoised at current sigma --
            pos_modality = modality_from_state(
                video_state, positive_context, sigma
            )
            pos_denoised = self.transformer(pos_modality)

            if guider.enabled():
                neg_modality = modality_from_state(
                    video_state, negative_context, sigma
                )
                neg_denoised = self.transformer(neg_modality)
                denoised = guider.guide(pos_denoised, neg_denoised)
            else:
                denoised = pos_denoised

            # STG on first evaluation
            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_denoised = self.transformer(pos_modality, perturbations=stg_perturbations)
                denoised = stg_guider.guide(denoised, perturbed_denoised)
                del perturbed_denoised

            # GE velocity correction
            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            denoised = maybe_post_process_latent(denoised, video_state)

            # -- Euler prediction to get predicted sample --
            from ..core_utils import to_velocity
            velocity = to_velocity(video_state.latent, sigmas[step_idx], denoised)
            dt = sigma_next - sigma
            predicted = video_state.latent.astype(mx.float32) + velocity.astype(mx.float32) * float(dt)
            predicted = predicted.astype(video_state.latent.dtype)
            mx.eval(predicted)

            # Skip corrector on last step (sigma_next == 0)
            if sigma_next == 0:
                video_state = video_state.replace(latent=denoised)
                mx.async_eval(video_state.latent)
                if callback:
                    callback(step_idx + 1, num_steps)
                continue

            # -- Second evaluation: denoised at predicted point --
            predicted_state = video_state.replace(latent=predicted)
            pos_modality_2 = modality_from_state(
                predicted_state, positive_context, sigma_next
            )
            pos_denoised_2 = self.transformer(pos_modality_2)

            if guider.enabled():
                neg_modality_2 = modality_from_state(
                    predicted_state, negative_context, sigma_next
                )
                neg_denoised_2 = self.transformer(neg_modality_2)
                denoised_at_predicted = guider.guide(pos_denoised_2, neg_denoised_2)
            else:
                denoised_at_predicted = pos_denoised_2

            denoised_at_predicted = maybe_post_process_latent(denoised_at_predicted, video_state)

            # -- Heun corrector step --
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=denoised_at_predicted,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.async_eval(video_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state

    def _denoise_loop_cfg_av(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        positive_video_context: mx.array,
        negative_video_context: mx.array,
        positive_audio_context: mx.array,
        negative_audio_context: mx.array,
        video_guider,
        audio_guider,
        stepper: EulerDiffusionStep,
        callback: Callable[[int, int], None] | None = None,
        stg_guider: STGGuider | None = None,
        stg_perturbations: BatchedPerturbationConfig | None = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
        profile_steps: tuple[int, ...] = (),
        profile_blocks: tuple[int, ...] = (),
    ) -> tuple[LatentState, LatentState]:
        """Run joint audio-video denoising loop with separate guidance per modality."""
        num_steps = len(sigmas) - 1
        need_cfg = video_guider.enabled() or audio_guider.enabled()
        prev_velocity = None
        profile_step_set = set(profile_steps)

        def print_profile(step: int, events: list[tuple[str, float]]) -> None:
            total = sum(seconds for _, seconds in events)
            print(f"\n  AV denoise step {step} profile (forced eval diagnostics):")
            for name, seconds in events:
                pct = (seconds / total * 100.0) if total > 0 else 0.0
                print(f"    {name:<24} {seconds:7.2f}s  {pct:5.1f}%")
            print(f"    {'profiled total':<24} {total:7.2f}s")

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)
            print(f"  Cross-attn scaling: {cross_attn_scale}x on blocks {cross_attn_start_block}-47")

        for step_idx in range(num_steps):
            profile_step = step_idx + 1
            profile_events = [] if profile_step in profile_step_set else None
            profile_started_at = time.perf_counter() if profile_events is not None else 0.0

            def mark_profile(
                name: str,
                *arrays: mx.array,
                _profile_events=profile_events,
            ) -> None:
                nonlocal profile_started_at
                if _profile_events is None:
                    return
                if arrays:
                    mx.eval(*arrays)
                now = time.perf_counter()
                _profile_events.append((name, now - profile_started_at))
                profile_started_at = now

            sigma = float(sigmas[step_idx])

            pos_video_modality = modality_from_state(video_state, positive_video_context, sigma)
            pos_audio_modality = audio_modality_from_state(audio_state, positive_audio_context, sigma)
            mark_profile("modality setup")

            if profile_events is not None:
                self._profile_next_transformer_call(
                    f"step {profile_step}",
                    blocks=profile_blocks,
                )
            pos_video_denoised, pos_audio_denoised = self.transformer(
                pos_video_modality, pos_audio_modality
            )
            mark_profile("positive transformer", pos_video_denoised, pos_audio_denoised)

            if need_cfg:
                neg_video_modality = modality_from_state(video_state, negative_video_context, sigma)
                neg_audio_modality = audio_modality_from_state(audio_state, negative_audio_context, sigma)
                neg_video_denoised, neg_audio_denoised = self.transformer(
                    neg_video_modality, neg_audio_modality
                )
                mark_profile("negative transformer", neg_video_denoised, neg_audio_denoised)
                video_denoised = video_guider.guide(pos_video_denoised, neg_video_denoised)
                audio_denoised = audio_guider.guide(pos_audio_denoised, neg_audio_denoised)
            else:
                video_denoised = pos_video_denoised
                audio_denoised = pos_audio_denoised

            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_video_denoised, _ = self.transformer(
                    pos_video_modality, pos_audio_modality, perturbations=stg_perturbations
                )
                mark_profile("stg transformer", perturbed_video_denoised)
                video_denoised = stg_guider.guide(video_denoised, perturbed_video_denoised)

            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - video_denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    video_denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            video_denoised = maybe_post_process_latent(video_denoised, video_state)
            audio_denoised = maybe_post_process_latent(audio_denoised, audio_state)
            mark_profile("guidance/postprocess", video_denoised, audio_denoised)

            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            new_audio_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=audio_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            mark_profile("scheduler step", new_video_latent, new_audio_latent)

            video_state = video_state.replace(latent=new_video_latent)
            audio_state = audio_state.replace(latent=new_audio_latent)

            mx.async_eval(video_state.latent, audio_state.latent)
            mark_profile("state eval")
            if profile_events is not None:
                print_profile(profile_step, profile_events)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state, audio_state

    def _denoise_loop_heun_av(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: mx.array,
        positive_video_context: mx.array,
        negative_video_context: mx.array,
        positive_audio_context: mx.array,
        negative_audio_context: mx.array,
        video_guider,
        audio_guider,
        stepper: HeunDiffusionStep,
        callback: Callable[[int, int], None] | None = None,
        stg_guider: STGGuider | None = None,
        stg_perturbations: BatchedPerturbationConfig | None = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> tuple[LatentState, LatentState]:
        """Run joint audio-video denoising loop with Heun stepping."""
        from ..core_utils import to_velocity

        num_steps = len(sigmas) - 1
        need_cfg = video_guider.enabled() or audio_guider.enabled()
        prev_velocity = None

        if cross_attn_scale != 1.0:
            self._apply_cross_attn_scales(cross_attn_scale, cross_attn_start_block)

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])

            pos_video_modality = modality_from_state(video_state, positive_video_context, sigma)
            pos_audio_modality = audio_modality_from_state(audio_state, positive_audio_context, sigma)
            pos_video_denoised, pos_audio_denoised = self.transformer(
                pos_video_modality, pos_audio_modality
            )

            if need_cfg:
                neg_video_modality = modality_from_state(video_state, negative_video_context, sigma)
                neg_audio_modality = audio_modality_from_state(audio_state, negative_audio_context, sigma)
                neg_video_denoised, neg_audio_denoised = self.transformer(
                    neg_video_modality, neg_audio_modality
                )
                video_denoised = video_guider.guide(pos_video_denoised, neg_video_denoised)
                audio_denoised = audio_guider.guide(pos_audio_denoised, neg_audio_denoised)
            else:
                video_denoised = pos_video_denoised
                audio_denoised = pos_audio_denoised

            stg_active = (
                stg_guider is not None
                and stg_guider.enabled()
                and stg_perturbations is not None
                and (step_idx + 1) / num_steps <= stg_cutoff
            )
            if stg_active:
                perturbed_video_denoised, _ = self.transformer(
                    pos_video_modality, pos_audio_modality, perturbations=stg_perturbations
                )
                video_denoised = stg_guider.guide(video_denoised, perturbed_video_denoised)

            if ge_gamma > 0 and sigma > 0:
                current_velocity = (video_state.latent - video_denoised) / sigma
                if prev_velocity is not None:
                    delta_v = current_velocity - prev_velocity
                    total_velocity = ge_gamma * delta_v + prev_velocity
                    video_denoised = video_state.latent - total_velocity * sigma
                prev_velocity = current_velocity

            video_denoised = maybe_post_process_latent(video_denoised, video_state)
            audio_denoised = maybe_post_process_latent(audio_denoised, audio_state)

            video_velocity = to_velocity(video_state.latent, sigmas[step_idx], video_denoised)
            audio_velocity = to_velocity(audio_state.latent, sigmas[step_idx], audio_denoised)
            dt = sigma_next - sigma
            predicted_video = video_state.latent.astype(mx.float32) + video_velocity.astype(mx.float32) * float(dt)
            predicted_audio = audio_state.latent.astype(mx.float32) + audio_velocity.astype(mx.float32) * float(dt)
            predicted_video = predicted_video.astype(video_state.latent.dtype)
            predicted_audio = predicted_audio.astype(audio_state.latent.dtype)
            mx.eval(predicted_video, predicted_audio)

            if sigma_next == 0:
                video_state = video_state.replace(latent=video_denoised)
                audio_state = audio_state.replace(latent=audio_denoised)
                mx.async_eval(video_state.latent, audio_state.latent)
                if callback:
                    callback(step_idx + 1, num_steps)
                continue

            predicted_video_state = video_state.replace(latent=predicted_video)
            predicted_audio_state = audio_state.replace(latent=predicted_audio)
            pos_video_modality_2 = modality_from_state(
                predicted_video_state, positive_video_context, sigma_next
            )
            pos_audio_modality_2 = audio_modality_from_state(
                predicted_audio_state, positive_audio_context, sigma_next
            )
            pos_video_denoised_2, pos_audio_denoised_2 = self.transformer(
                pos_video_modality_2, pos_audio_modality_2
            )

            if need_cfg:
                neg_video_modality_2 = modality_from_state(
                    predicted_video_state, negative_video_context, sigma_next
                )
                neg_audio_modality_2 = audio_modality_from_state(
                    predicted_audio_state, negative_audio_context, sigma_next
                )
                neg_video_denoised_2, neg_audio_denoised_2 = self.transformer(
                    neg_video_modality_2, neg_audio_modality_2
                )
                video_denoised_at_predicted = video_guider.guide(pos_video_denoised_2, neg_video_denoised_2)
                audio_denoised_at_predicted = audio_guider.guide(pos_audio_denoised_2, neg_audio_denoised_2)
            else:
                video_denoised_at_predicted = pos_video_denoised_2
                audio_denoised_at_predicted = pos_audio_denoised_2

            video_denoised_at_predicted = maybe_post_process_latent(video_denoised_at_predicted, video_state)
            audio_denoised_at_predicted = maybe_post_process_latent(audio_denoised_at_predicted, audio_state)

            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=video_denoised_at_predicted,
            )
            new_audio_latent = stepper.step(
                sample=audio_state.latent,
                denoised_sample=audio_denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=audio_denoised_at_predicted,
            )

            video_state = video_state.replace(latent=new_video_latent)
            audio_state = audio_state.replace(latent=new_audio_latent)
            mx.async_eval(video_state.latent, audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state, audio_state

    def _denoise_loop_simple_av(
        self,
        video_state: LatentState,
        audio_state: LatentState | None,
        sigmas: mx.array,
        video_context: mx.array,
        audio_context: mx.array | None,
        stepper: EulerDiffusionStep,
        callback: Callable[[int, int], None] | None = None,
        profile_steps: tuple[int, ...] = (),
        profile_blocks: tuple[int, ...] = (),
    ) -> tuple[LatentState, LatentState | None]:
        """Run distilled denoising without CFG for video-only or AV transformers."""
        num_steps = len(sigmas) - 1
        profile_step_set = set(profile_steps)

        def print_profile(step: int, events: list[tuple[str, float]]) -> None:
            total = sum(seconds for _, seconds in events)
            print(f"\n  AV denoise step {step} profile (forced eval diagnostics):")
            for name, seconds in events:
                pct = (seconds / total * 100.0) if total > 0 else 0.0
                print(f"    {name:<24} {seconds:7.2f}s  {pct:5.1f}%")
            print(f"    {'profiled total':<24} {total:7.2f}s")

        # Velocity-mode eligibility: requires X0Model wrapper (so we can reach
        # the underlying velocity_model) and uniform masks on both states (so
        # we can skip the explicit denoised computation).
        velocity_mode = (
            _USE_VELOCITY_MODE
            and hasattr(self.transformer, "velocity_model")
            and video_state.uniform_mask
            and (audio_state is None or audio_state.uniform_mask)
        )
        velocity_transformer = (
            self._velocity_transformer() if velocity_mode else None
        )

        # Optional per-stage RoPE precompute.  Positions are fixed for the
        # whole stage, while sigma/timestep changes each denoise step, so the
        # cos/sin tables are safe to reuse until the stage shape/positions
        # change.  This covers self-RoPE and A/V cross temporal RoPE.
        #
        # M1 Max stage-2 A/B with metadata-float64 RoPE measured neutral to
        # slight-regression, likely because the large resident FP32 tables add
        # memory pressure while MLX already handles the per-step setup cheaply.
        # Keep it opt-in for diagnostics:
        #   LTX_ROPE_PRECOMPUTE=1
        def _simple_preprocessor_of(preproc):
            # Multi-modal preprocessor wraps a simple preprocessor; video-only
            # is the simple preprocessor itself.
            return getattr(preproc, "simple_preprocessor", preproc)

        video_rope = None
        audio_rope = None
        video_cross_rope = None
        audio_cross_rope = None
        if _PRECOMPUTE_ROPE:
            velocity_model = (
                self.transformer.velocity_model
                if hasattr(self.transformer, "velocity_model")
                else self.transformer
            )
            rope_arrays: list[mx.array] = []
            video_pre = getattr(velocity_model, "_video_args_preprocessor", None)
            if video_pre is not None:
                video_rope = _simple_preprocessor_of(video_pre)._prepare_positional_embeddings(
                    video_state.positions,
                )
                rope_arrays.extend(video_rope)
                prepare_cross = getattr(video_pre, "_prepare_cross_positional_embeddings", None)
                if self.is_av_model and audio_state is not None and prepare_cross is not None:
                    video_cross_rope = prepare_cross(video_state.positions)
                    rope_arrays.extend(video_cross_rope)
            audio_pre = getattr(velocity_model, "_audio_args_preprocessor", None)
            if audio_state is not None and audio_pre is not None:
                audio_rope = _simple_preprocessor_of(audio_pre)._prepare_positional_embeddings(
                    audio_state.positions,
                )
                rope_arrays.extend(audio_rope)
                prepare_cross = getattr(audio_pre, "_prepare_cross_positional_embeddings", None)
                if self.is_av_model and prepare_cross is not None:
                    audio_cross_rope = prepare_cross(audio_state.positions)
                    rope_arrays.extend(audio_cross_rope)
            if rope_arrays:
                mx.eval(*rope_arrays)

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])
            dt = sigma_next - sigma

            profile_step = step_idx + 1
            profile_events: list[tuple[str, float]] | None = (
                [] if profile_step in profile_step_set else None
            )
            profile_started_at = time.perf_counter() if profile_events is not None else 0.0

            def mark_profile(
                name: str,
                *arrays: mx.array,
                _profile_events=profile_events,
            ) -> None:
                nonlocal profile_started_at
                if _profile_events is None:
                    return
                if arrays:
                    mx.eval(*arrays)
                now = time.perf_counter()
                _profile_events.append((name, now - profile_started_at))
                profile_started_at = now

            video_modality = modality_from_state(
                video_state, video_context, sigma,
                positional_embeddings=video_rope,
                cross_positional_embeddings=video_cross_rope,
            )
            mark_profile("modality setup")

            # ----- Transformer forward -----
            if self.is_av_model:
                audio_modality = (
                    audio_modality_from_state(
                        audio_state, audio_context, sigma,
                        positional_embeddings=audio_rope,
                        cross_positional_embeddings=audio_cross_rope,
                    )
                    if audio_state is not None
                    else None
                )
                if profile_events is not None:
                    self._profile_next_transformer_call(
                        f"step {profile_step}",
                        blocks=profile_blocks,
                    )
                if velocity_mode:
                    # Raw velocity outputs - skip the X0 wrapper.
                    result = (
                        velocity_transformer(video_modality, audio_modality)
                        if audio_modality is not None
                        else velocity_transformer(video_modality)
                    )
                else:
                    result = (
                        self.transformer(video_modality, audio_modality)
                        if audio_modality is not None
                        else self.transformer(video_modality)
                    )
                if isinstance(result, tuple):
                    video_out, audio_out = result
                else:
                    video_out = result
                    audio_out = None
            else:
                if profile_events is not None:
                    self._profile_next_transformer_call(
                        f"step {profile_step}",
                        blocks=profile_blocks,
                    )
                if velocity_mode:
                    video_out = velocity_transformer(video_modality)
                else:
                    video_out = self.transformer(video_modality)
                audio_out = None
            mark_profile("transformer call", video_out, audio_out) if audio_out is not None else mark_profile("transformer call", video_out)

            # ----- Latent update -----
            if velocity_mode:
                # Inline velocity-form Euler step:
                #     next = latent + (sigma_next - sigma) * velocity
                # Cast pattern matches EulerDiffusionStep: do the lerp in
                # the source dtype path; sigmas come in as Python floats.
                new_video_latent = video_state.latent + dt * video_out
                if os.environ.get("LTX_STEP_PROBE"):
                    try:
                        import stage2_step_contribution_probe as _ssp
                        _ssp.record(step_idx, sigma, sigma_next,
                                    video_state.latent, new_video_latent, velocity=video_out)
                    except Exception:
                        pass
                video_state = video_state.replace(latent=new_video_latent)
                if audio_state is not None and audio_out is not None:
                    new_audio_latent = audio_state.latent + dt * audio_out
                    audio_state = audio_state.replace(latent=new_audio_latent)
                    mx.eval(video_state.latent, audio_state.latent)
                else:
                    mx.eval(video_state.latent)
            else:
                video_denoised = maybe_post_process_latent(video_out, video_state)
                new_video_latent = stepper.step(
                    sample=video_state.latent,
                    denoised_sample=video_denoised,
                    sigmas=sigmas,
                    step_index=step_idx,
                )
                video_state = video_state.replace(latent=new_video_latent)

                if audio_state is not None and audio_out is not None:
                    audio_denoised = maybe_post_process_latent(audio_out, audio_state)
                    new_audio_latent = stepper.step(
                        sample=audio_state.latent,
                        denoised_sample=audio_denoised,
                        sigmas=sigmas,
                        step_index=step_idx,
                    )
                    audio_state = audio_state.replace(latent=new_audio_latent)
                    # Sync eval (one call, both arrays).  Profiled vs async_eval:
                    # same wall clock, but ~38% fewer set_copy_output_data calls
                    # (125 -> 78) so the lazy graph stays cleaner.  Profiled vs
                    # separate calls: combined is slightly faster at small T.
                    mx.eval(video_state.latent, audio_state.latent)
                else:
                    mx.eval(video_state.latent)
            mark_profile("post + stepper", video_state.latent)

            if profile_events is not None:
                print_profile(profile_step, profile_events)

            if callback:
                callback(step_idx + 1, num_steps)

            # Profiling helper: LTX_PROFILE_STOP_AFTER_STEPS=N exits cleanly
            # after step N of the current loop.  Combine with
            # LTX_PROFILE_PAUSE_BEFORE_DENOISE=1 to capture a precise window
            # - e.g. N=2 gives one warmup step + one stable step.
            _stop_n_str = os.environ.get("LTX_PROFILE_STOP_AFTER_STEPS")
            if _stop_n_str:
                try:
                    _stop_n = int(_stop_n_str)
                except ValueError:
                    _stop_n = 0
                if _stop_n > 0 and (step_idx + 1) >= _stop_n:
                    # Make sure the last step's GPU work is materialized
                    # before we exit (mx.eval was called inside the loop,
                    # but be defensive).
                    import sys as _sys
                    print(
                        f"\n  [LTX_PROFILE_STOP_AFTER_STEPS={_stop_n}] "
                        f"completed step {step_idx + 1}/{num_steps}, exiting.",
                        flush=True,
                    )
                    _sys.exit(0)

        return video_state, audio_state

    def generate_distilled_two_stage(
        self,
        positive_encoding: mx.array,
        config: AVCFGConfig,
        spatial_upscaler=None,
        spatial_upscaler_loader: Callable[[], object] | None = None,
        images: list[ImageCondition] | None = None,
        callback: Callable[[int, int], None] | None = None,
        stage_callback: Callable[[str, int, int], None] | None = None,
        progress_message: Callable[[str], None] | None = None,
        positive_audio_encoding: mx.array | None = None,
        latent_save_path: str | None = None,
        stage_1_sigmas: Sequence[float] | None = None,
        stage_2_sigmas: Sequence[float] | None = None,
        decode_video: bool = True,
        stage_lora_configs: list[LoRAConfig] | None = None,
        stage2_lora_fuse_mode: str = "fresh-total",
    ) -> tuple[mx.array, mx.array | None]:
        """
        Generate with the distilled AV checkpoint in two stages.

        Stage 1 denoises at half spatial resolution with the official distilled
        sigmas, then stage 2 spatially upscales and refines with the stage-2
        distilled sigmas. There is intentionally no CFG in either stage.

        When `decode_video=True` (default) the pipeline runs VAE decode
        internally and returns `(video, audio_waveform)` where `video` is the
        fully decoded `(B, 3, T, H, W)` float tensor in [-1, 1].

        When `decode_video=False` the pipeline skips the internal VAE decode +
        chunk concatenation and returns `(final_video_latent, audio_waveform)`
        instead.  The caller is responsible for running decode (typically via
        `LTX_2_MLX.pipelines.streaming.decode_video_chunks_streaming`) and
        consuming chunks as they arrive - useful for streaming-encoder paths
        that can start encoding chunk 1 while chunk 2 is still in the VAE,
        and for high-resolution clips where the full decoded tensor would
        otherwise consume gigabytes of unified memory.
        """
        call_start = time.perf_counter()
        self.last_timing_sections = []
        images = images or []

        if config.height % 64 != 0 or config.width % 64 != 0:
            raise ValueError(
                f"Distilled two-stage requires resolution divisible by 64, "
                f"got {config.height}x{config.width}."
            )
        if self.video_encoder is None:
            raise ValueError("Video encoder required for distilled two-stage upscaling.")
        if (
            decode_video
            and self.video_decoder is None
            and self.video_decoder_loader is None
        ):
            raise ValueError("Video decoder required for VAE decode.")
        if spatial_upscaler is None and spatial_upscaler_loader is None:
            raise ValueError("Spatial upscaler required for distilled two-stage generation.")

        internal_audio_active = self.is_av_model and (
            config.use_internal_audio_branch or config.audio_enabled
        )
        if internal_audio_active and positive_audio_encoding is None:
            raise ValueError(
                "Positive audio encoding required for AudioVideo distilled two-stage generation."
            )
        if (
            config.audio_enabled
            and (self.audio_decoder is None or self.vocoder is None)
            and self.audio_decoder_loader is None
        ):
            raise ValueError("Audio decoder and vocoder required when audio_enabled is True.")

        mx.random.seed(config.seed)
        noiser = GaussianNoiser()
        stepper = self.diffusion_step
        stage_1_sigmas = mx.array(
            DISTILLED_SIGMA_VALUES if stage_1_sigmas is None else stage_1_sigmas,
            dtype=mx.float32,
        )
        stage_2_sigmas = mx.array(
            STAGE_2_DISTILLED_SIGMA_VALUES if stage_2_sigmas is None else stage_2_sigmas,
            dtype=mx.float32,
        )
        _s1o = os.environ.get("LTX_STAGE1_SIGMAS")
        if _s1o:
            stage_1_sigmas = mx.array([float(x) for x in _s1o.split(",") if x.strip()], dtype=mx.float32)
            print(f"  [LTX_STAGE1_SIGMAS] stage-1 schedule override: {len(stage_1_sigmas) - 1} steps")
        _s2o = os.environ.get("LTX_STAGE2_SIGMAS")
        if _s2o:
            stage_2_sigmas = mx.array([float(x) for x in _s2o.split(",") if x.strip()], dtype=mx.float32)
            print(f"  [LTX_STAGE2_SIGMAS] stage-2 schedule override: {len(stage_2_sigmas) - 1} steps")
        if len(stage_1_sigmas) < 2:
            raise ValueError("stage_1_sigmas must contain at least two sigma values.")
        if len(stage_2_sigmas) < 2:
            raise ValueError("stage_2_sigmas must contain at least two sigma values.")
        total_steps = len(stage_1_sigmas) + len(stage_2_sigmas) - 2
        if callback:
            callback(0, total_steps)

        def emit_progress_message(message: str) -> None:
            if progress_message is not None:
                progress_message(message)
            else:
                print(message)

        stage_1_loras = lora_configs_for_stage(stage_lora_configs, 1)
        stage_2_total_loras = lora_configs_for_stage(stage_lora_configs, 2)
        stage_2_delta_loras = lora_configs_for_stage_delta(
            stage_lora_configs,
            from_stage=1,
            to_stage=2,
        )
        if stage2_lora_fuse_mode not in {"delta", "fresh-total"}:
            raise ValueError(
                "stage2_lora_fuse_mode must be 'delta' or 'fresh-total', "
                f"got {stage2_lora_fuse_mode!r}"
            )
        stage_lora_restore_state = None
        stage_1_lora_elapsed = 0.0
        spatial_upscaler_load_elapsed = 0.0
        stage_2_lora_elapsed = 0.0
        stage_2_lora_base_restore_elapsed = 0.0
        stage_2_lora_label = "stage 2 lora delta fuse"
        lora_restore_elapsed = 0.0
        if stage_1_loras or stage_2_delta_loras:
            stage_lora_restore_state = get_transformer_cache_restore_state(self.transformer)
        if stage_1_loras:
            stage_lines = format_lora_stage_scale_lines(stage_lora_configs, 1)
            if stage_lines:
                emit_progress_message("  Stage 1 LoRA scales:")
                for line in stage_lines:
                    emit_progress_message(line)
            lora_fuse_start = time.perf_counter()
            fuse_loras_into_model(
                self.transformer,
                stage_1_loras,
            )
            stage_1_lora_elapsed = time.perf_counter() - lora_fuse_start
            emit_progress_message(
                f"  Stage 1 LoRA fuse complete in {stage_1_lora_elapsed:.1f}s"
            )

        stage_1_height = config.height // 2
        stage_1_width = config.width // 2
        stage_1_pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=stage_1_height,
            width=stage_1_width,
            fps=config.fps,
        )
        stage_1_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_1_pixel_shape, latent_channels=128
        )
        video_tools = self._create_video_tools(stage_1_latent_shape, config.fps)
        conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            stage_1_height,
            stage_1_width,
            config.dtype,
        )

        video_state = video_tools.create_initial_state(dtype=config.dtype)
        video_state = apply_conditionings(video_state, conditionings, video_tools)
        video_state = noiser(video_state, noise_scale=1.0)

        audio_state = None
        audio_tools = None
        if internal_audio_active:
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                stage_1_pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools = self._create_audio_tools(audio_shape)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)
            emit_progress_message(
                "  Audio noise normalization: "
                f"{'enabled' if _NORMALIZE_AUDIO_NOISE else 'disabled'}"
            )
            if _NORMALIZE_AUDIO_NOISE:
                audio_state = audio_state.replace(
                    latent=self._channelwise_normalize_audio_noise(audio_state.latent)
                )

        emit_progress_message(
            f"  Distilled stage 1: {len(stage_1_sigmas) - 1} steps at "
            f"{stage_1_height}x{stage_1_width}"
        )
        # Profiling helper: LTX_PROFILE_PAUSE_BEFORE_DENOISE=1 blocks on
        # stdin before stage-1 starts.  Use this to attach Instruments to
        # the running process at a precise point (after model load / prompt
        # encoding, just before the first denoise step).  The announcement
        # routes through emit_progress_message so a caller with active
        # bars doesn't get its layout stomped; `input()` then blocks until
        # the user hits Enter.
        if os.environ.get("LTX_PROFILE_PAUSE_BEFORE_DENOISE"):
            try:
                emit_progress_message(
                    f"\n  [LTX_PROFILE_PAUSE_BEFORE_DENOISE] pid={os.getpid()}  "
                    "attach Instruments now, then press Enter to start stage 1."
                )
                input()
            except EOFError:
                pass
        if stage_callback:
            stage_callback("stage_1", 0, len(stage_1_sigmas) - 1)

        def stage_1_callback(step: int, _total: int):
            if stage_callback:
                stage_callback("stage_1", step, _total)
            if callback:
                callback(step, total_steps)

        # Map global profile step numbers to per-stage local step numbers.
        # Stage 1 covers global steps 1..(len(stage_1_sigmas)-1).
        # Stage 2 covers global steps len(stage_1_sigmas)..total_steps.
        global_profile_steps = tuple(config.profile_transformer_steps or ())
        global_profile_blocks = tuple(config.profile_transformer_blocks or ())
        stage_1_step_count = len(stage_1_sigmas) - 1
        stage_1_profile_steps = tuple(
            s for s in global_profile_steps if 1 <= s <= stage_1_step_count
        )
        stage_2_profile_steps = tuple(
            s - stage_1_step_count for s in global_profile_steps if s > stage_1_step_count
        )

        stage_1_start = time.perf_counter()
        video_state, audio_state = self._denoise_loop_simple_av(
            video_state=video_state,
            audio_state=audio_state,
            sigmas=stage_1_sigmas,
            video_context=positive_encoding,
            audio_context=positive_audio_encoding,
            stepper=stepper,
            callback=stage_1_callback,
            profile_steps=stage_1_profile_steps,
            profile_blocks=global_profile_blocks,
        )
        stage_1_elapsed = time.perf_counter() - stage_1_start

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        stage_1_video_latent = video_state.latent

        stage_1_audio_latent = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            stage_1_audio_latent = audio_state.latent

        stage_1_video_latent_for_save = (
            stage_1_video_latent if latent_save_path is not None else None
        )
        stage_1_audio_latent_for_save = (
            stage_1_audio_latent if latent_save_path is not None else None
        )

        active_spatial_upscaler = spatial_upscaler
        if active_spatial_upscaler is None:
            emit_progress_message("  Loading spatial upscaler...")
            load_start = time.perf_counter()
            active_spatial_upscaler = spatial_upscaler_loader()
            spatial_upscaler_load_elapsed = time.perf_counter() - load_start
            emit_progress_message(
                f"  Spatial upscaler load complete in {spatial_upscaler_load_elapsed:.1f}s"
            )

        upscale_start = time.perf_counter()
        emit_progress_message("  Upsampling latent 2x with spatial upscaler...")
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(
            stage_1_video_latent
        )
        upscaled_unnorm = active_spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)
        upscaled_video_latent = self.video_encoder.per_channel_statistics.normalize(
            upscaled_unnorm
        )
        mx.eval(upscaled_video_latent)
        del latent_unnorm, upscaled_unnorm
        del active_spatial_upscaler
        # Keep only the latents needed by stage 2 before LoRA fusion starts.
        del video_state, video_tools, conditionings
        del stage_1_video_latent
        if audio_state is not None:
            del audio_state
        if audio_tools is not None:
            del audio_tools
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        upscale_elapsed = time.perf_counter() - upscale_start

        if stage2_lora_fuse_mode == "fresh-total" and stage_lora_restore_state is not None:
            lora_restore_start = time.perf_counter()
            restore_transformer_cache_state(self.transformer, stage_lora_restore_state)
            stage_2_lora_base_restore_elapsed = time.perf_counter() - lora_restore_start
            emit_progress_message(
                "  Stage 2 LoRA base restore complete in "
                f"{stage_2_lora_base_restore_elapsed:.1f}s"
            )

        stage_2_loras_to_fuse = (
            stage_2_total_loras
            if stage2_lora_fuse_mode == "fresh-total"
            else stage_2_delta_loras
        )
        if stage_2_loras_to_fuse:
            if stage2_lora_fuse_mode == "fresh-total":
                stage_lines = format_lora_stage_scale_lines(stage_lora_configs, 2)
                line_header = "  Stage 2 LoRA total scales:"
                complete_label = "  Stage 2 LoRA total fuse complete in "
                stage_2_lora_label = "stage 2 lora total fuse"
            else:
                stage_lines = format_lora_stage_scale_lines(
                    stage_lora_configs,
                    2,
                    from_stage=1,
                    include_unchanged=True,
                )
                line_header = "  Stage 2 LoRA scales:"
                complete_label = "  Stage 2 LoRA delta fuse complete in "
                stage_2_lora_label = "stage 2 lora delta fuse"
            if stage_lines:
                emit_progress_message(line_header)
                for line in stage_lines:
                    emit_progress_message(line)
            lora_fuse_start = time.perf_counter()
            fuse_loras_into_model(
                self.transformer,
                stage_2_loras_to_fuse,
            )
            stage_2_lora_elapsed = time.perf_counter() - lora_fuse_start
            emit_progress_message(
                f"{complete_label}{stage_2_lora_elapsed:.1f}s"
            )

        stage_2_pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.fps,
        )
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_2_pixel_shape, latent_channels=128
        )
        video_tools_2 = self._create_video_tools(stage_2_latent_shape, config.fps)
        conditionings_2 = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )
        video_state_2 = video_tools_2.create_initial_state(
            dtype=config.dtype,
            initial_latent=upscaled_video_latent,
        )
        video_state_2 = apply_conditionings(video_state_2, conditionings_2, video_tools_2)
        video_state_2 = noiser(video_state_2, noise_scale=float(stage_2_sigmas[0]))

        audio_state_2 = None
        audio_tools_2 = None
        if internal_audio_active:
            audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
                stage_2_pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools_2 = self._create_audio_tools(audio_shape_2)
            audio_state_2 = audio_tools_2.create_initial_state(
                dtype=config.dtype,
                initial_latent=stage_1_audio_latent,
            )
            audio_state_2 = noiser(audio_state_2, noise_scale=float(stage_2_sigmas[0]))

        emit_progress_message(
            f"  Distilled stage 2: {len(stage_2_sigmas) - 1} steps at "
            f"{config.height}x{config.width}"
        )
        if stage_callback:
            stage_callback("stage_2", 0, len(stage_2_sigmas) - 1)

        def stage_2_callback(step: int, _total: int):
            if stage_callback:
                stage_callback("stage_2", step, _total)
            if callback:
                callback((len(stage_1_sigmas) - 1) + step, total_steps)

        stage_2_start = time.perf_counter()
        video_state_2, audio_state_2 = self._denoise_loop_simple_av(
            video_state=video_state_2,
            audio_state=audio_state_2,
            sigmas=stage_2_sigmas,
            video_context=positive_encoding,
            audio_context=positive_audio_encoding,
            stepper=stepper,
            callback=stage_2_callback,
            profile_steps=stage_2_profile_steps,
            profile_blocks=global_profile_blocks,
        )
        stage_2_elapsed = time.perf_counter() - stage_2_start
        if stage_lora_restore_state is not None:
            lora_restore_start = time.perf_counter()
            restore_transformer_cache_state(self.transformer, stage_lora_restore_state)
            lora_restore_elapsed = time.perf_counter() - lora_restore_start
            emit_progress_message(
                f"  Transformer cache restore complete in {lora_restore_elapsed:.1f}s"
            )
        post_denoise_start = time.perf_counter()

        video_state_2 = video_tools_2.clear_conditioning(video_state_2)
        video_state_2 = video_tools_2.unpatchify(video_state_2)
        final_video_latent = video_state_2.latent

        final_audio_latent = None
        if config.audio_enabled and audio_state_2 is not None and audio_tools_2 is not None:
            audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
            audio_state_2 = audio_tools_2.unpatchify(audio_state_2)
            final_audio_latent = audio_state_2.latent

        del self.transformer
        self.transformer = None
        gc.collect()
        mx.clear_cache()

        if latent_save_path is not None:
            self._save_distilled_two_stage_latents(
                latent_save_path,
                stage_1_video_latent=stage_1_video_latent_for_save,
                stage_2_video_latent=final_video_latent,
                stage_1_audio_latent=stage_1_audio_latent_for_save,
                stage_2_audio_latent=final_audio_latent,
                progress_message=progress_message,
            )
            del stage_1_video_latent_for_save, stage_1_audio_latent_for_save
            gc.collect()
            mx.clear_cache()

        effective_tiling = config._get_tiling_config()
        post_denoise_elapsed = time.perf_counter() - post_denoise_start
        video_decoder_load_elapsed = 0.0
        if decode_video:
            video_decoder_load_elapsed, video_decoder_loaded_now = self._ensure_video_decoder(
                emit_progress_message
            )
            try:
                decoder_name = self.video_decoder.__class__.__name__
                tiling_desc = "tiled" if effective_tiling else "not tiled"
                emit_progress_message(
                    f"  VAE decode started ({decoder_name}, {tiling_desc})..."
                )
                decode_start = time.perf_counter()
                if effective_tiling:
                    emit_progress_message(
                        "  Using tiled VAE decoding (preventing GPU watchdog timeout)"
                    )
                    video = None
                    for chunk in decode_tiled(final_video_latent, self.video_decoder, effective_tiling):
                        video = chunk if video is None else mx.concatenate([video, chunk], axis=2)
                        mx.eval(video)
                        del chunk
                        gc.collect()
                        mx.clear_cache()
                else:
                    video = decode_latent(final_video_latent, self.video_decoder)
                mx.eval(video)
                decode_elapsed = time.perf_counter() - decode_start
                emit_progress_message(f"  VAE decode complete in {decode_elapsed:.1f}s")
            finally:
                if video_decoder_loaded_now:
                    self.video_decoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            # Streaming-decode path: caller will run VAE decode on the
            # latent.  Return the latent so the caller can iterate chunks
            # straight into the downstream consumer (typically an
            # AVAssetWriter+VSR streaming sink) without ever materializing
            # the full decoded video tensor.
            video = final_video_latent
            decode_elapsed = 0.0

        audio_waveform = None
        audio_decoder_load_elapsed = 0.0
        if final_audio_latent is not None:
            audio_decoder_load_elapsed, audio_decoder_loaded_now = self._ensure_audio_decoder(
                emit_progress_message
            )
            try:
                audio_decode_start = time.perf_counter()
                audio_waveform = self._decode_audio(final_audio_latent)
                audio_decode_elapsed = time.perf_counter() - audio_decode_start
            finally:
                if audio_decoder_loaded_now:
                    self.audio_decoder = None
                    self.vocoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            audio_decode_elapsed = 0.0

        setup_elapsed = max(0.0, stage_1_start - call_start - stage_1_lora_elapsed)
        self.last_timing_sections = [("pipeline setup", setup_elapsed)]
        if stage_1_lora_elapsed:
            self.last_timing_sections.append(("stage 1 lora fuse", stage_1_lora_elapsed))
        self.last_timing_sections.extend([
            ("distilled stage 1 denoise", stage_1_elapsed),
        ])
        if spatial_upscaler_load_elapsed:
            self.last_timing_sections.append(
                ("spatial upscaler load", spatial_upscaler_load_elapsed)
            )
        self.last_timing_sections.append(("spatial upscale", upscale_elapsed))
        if stage_2_lora_elapsed:
            if stage_2_lora_base_restore_elapsed:
                self.last_timing_sections.append(
                    ("stage 2 lora base restore", stage_2_lora_base_restore_elapsed)
                )
            self.last_timing_sections.append((stage_2_lora_label, stage_2_lora_elapsed))
        elif stage_2_lora_base_restore_elapsed:
            self.last_timing_sections.append(
                ("stage 2 lora base restore", stage_2_lora_base_restore_elapsed)
            )
        self.last_timing_sections.append(("distilled stage 2 denoise", stage_2_elapsed))
        if lora_restore_elapsed:
            self.last_timing_sections.append(("lora restore", lora_restore_elapsed))
        self.last_timing_sections.append(("post-denoise prep", post_denoise_elapsed))
        if video_decoder_load_elapsed:
            self.last_timing_sections.append(("vae decoder load", video_decoder_load_elapsed))
        if decode_video:
            self.last_timing_sections.append(("vae decode", decode_elapsed))
        if audio_decoder_load_elapsed:
            self.last_timing_sections.append(("audio decoder load", audio_decoder_load_elapsed))
        if audio_decode_elapsed:
            self.last_timing_sections.append(("audio decode", audio_decode_elapsed))

        return video, audio_waveform

    def generate_distilled_stage2_from_latents(
        self,
        stage_1_video_latent: mx.array,
        positive_encoding: mx.array,
        config: AVCFGConfig,
        spatial_upscaler,
        stage_1_audio_latent: mx.array | None = None,
        images: list[ImageCondition] | None = None,
        callback: Callable[[int, int], None] | None = None,
        stage_callback: Callable[[str, int, int], None] | None = None,
        progress_message: Callable[[str], None] | None = None,
        positive_audio_encoding: mx.array | None = None,
        latent_save_path: str | None = None,
        burn_stage1_rng: bool = True,
        decode_video: bool = True,
        stage2_state_probe: Callable[[LatentState, LatentState | None, mx.array], None] | None = None,
        stage_lora_configs: list[LoRAConfig] | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Resume the distilled two-stage pipeline from saved stage-1 latents.

        The input latents must be the unpatchified stage-1 latents saved by
        `_save_distilled_two_stage_latents`. By default the method burns the
        stage-1 video/audio RNG draws before stage-2 noising so a same-seed
        harness run can match a full two-stage run's stage-2 noise stream.

        `decode_video` mirrors the same kwarg on `generate_distilled_two_stage`:
        when False, the pipeline skips the internal VAE decode and returns
        `(final_video_latent, audio_waveform)` so the caller can stream
        chunks into a downstream encoder.
        """
        call_start = time.perf_counter()
        self.last_timing_sections = []
        images = images or []

        if config.height % 64 != 0 or config.width % 64 != 0:
            raise ValueError(
                f"Distilled stage-2 harness requires resolution divisible by 64, "
                f"got {config.height}x{config.width}."
            )
        if self.video_encoder is None:
            raise ValueError("Video encoder required for distilled stage-2 upscaling.")
        if (
            decode_video
            and self.video_decoder is None
            and self.video_decoder_loader is None
        ):
            raise ValueError("Video decoder required for VAE decode.")
        if spatial_upscaler is None:
            raise ValueError("Spatial upscaler required for distilled stage-2 generation.")
        if stage_1_video_latent.ndim != 5:
            raise ValueError(
                "stage_1_video_latent must be a 5D unpatchified latent "
                f"(B,C,F,H,W), got shape {stage_1_video_latent.shape}."
            )

        stage_1_frames = (stage_1_video_latent.shape[2] - 1) * 8 + 1
        expected_height = stage_1_video_latent.shape[3] * 64
        expected_width = stage_1_video_latent.shape[4] * 64
        if config.num_frames != stage_1_frames:
            raise ValueError(
                f"Config frames ({config.num_frames}) do not match stage-1 "
                f"latent frames ({stage_1_frames})."
            )
        if config.height != expected_height or config.width != expected_width:
            raise ValueError(
                f"Config resolution ({config.height}x{config.width}) does not "
                f"match stage-1 latent-derived resolution "
                f"({expected_height}x{expected_width})."
            )

        internal_audio_active = self.is_av_model and (
            config.use_internal_audio_branch or config.audio_enabled
        )
        if internal_audio_active:
            if positive_audio_encoding is None:
                raise ValueError(
                    "Positive audio encoding required for AudioVideo distilled stage-2 generation."
                )
            if stage_1_audio_latent is None:
                raise ValueError(
                    "stage_1_audio_latent required when the AV audio branch is active."
                )
        if (
            config.audio_enabled
            and (self.audio_decoder is None or self.vocoder is None)
            and self.audio_decoder_loader is None
        ):
            raise ValueError("Audio decoder and vocoder required when audio_enabled is True.")

        mx.random.seed(config.seed)
        if burn_stage1_rng:
            burn_arrays = [
                mx.random.normal(
                    shape=stage_1_video_latent.shape,
                    dtype=config.dtype,
                )
            ]
            if internal_audio_active and stage_1_audio_latent is not None:
                burn_arrays.append(
                    mx.random.normal(
                        shape=stage_1_audio_latent.shape,
                        dtype=config.dtype,
                    )
                )
            mx.eval(*burn_arrays)
            del burn_arrays

        noiser = GaussianNoiser()
        stepper = self.diffusion_step
        stage_2_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)
        _s2_override = os.environ.get("LTX_STAGE2_SIGMAS")
        if _s2_override:
            stage_2_sigmas = mx.array([float(x) for x in _s2_override.split(",") if x.strip()])
            print(f"  [LTX_STAGE2_SIGMAS] stage-2 schedule override: "
                  f"{[round(float(s), 6) for s in stage_2_sigmas]} "
                  f"({len(stage_2_sigmas) - 1} steps)")
        total_steps = len(stage_2_sigmas) - 1
        if callback:
            callback(0, total_steps)

        def emit_progress_message(message: str) -> None:
            if progress_message is not None:
                progress_message(message)
            else:
                print(message)

        stage_2_loras = lora_configs_for_stage(stage_lora_configs, 2)
        stage_lora_restore_state = None
        stage_2_lora_elapsed = 0.0
        lora_restore_elapsed = 0.0
        if stage_2_loras:
            stage_lora_restore_state = get_transformer_cache_restore_state(self.transformer)
            stage_lines = format_lora_stage_scale_lines(stage_lora_configs, 2)
            if stage_lines:
                emit_progress_message("  Stage 2 LoRA scales:")
                for line in stage_lines:
                    emit_progress_message(line)
            lora_fuse_start = time.perf_counter()
            fuse_loras_into_model(
                self.transformer,
                stage_2_loras,
            )
            stage_2_lora_elapsed = time.perf_counter() - lora_fuse_start
            emit_progress_message(
                f"  Stage 2 LoRA fuse complete in {stage_2_lora_elapsed:.1f}s"
            )

        stage_1_video_latent = stage_1_video_latent.astype(config.dtype)
        stage_1_audio_latent_for_save = None
        if stage_1_audio_latent is not None:
            stage_1_audio_latent = stage_1_audio_latent.astype(config.dtype)
            stage_1_audio_latent_for_save = (
                stage_1_audio_latent if latent_save_path is not None else None
            )
        stage_1_video_latent_for_save = (
            stage_1_video_latent if latent_save_path is not None else None
        )

        upscale_start = time.perf_counter()
        emit_progress_message("  Upsampling saved stage-1 latent 2x with spatial upscaler...")
        latent_unnorm = self.video_encoder.per_channel_statistics.un_normalize(
            stage_1_video_latent
        )
        upscaled_unnorm = spatial_upscaler(latent_unnorm)
        mx.eval(upscaled_unnorm)
        upscaled_video_latent = self.video_encoder.per_channel_statistics.normalize(
            upscaled_unnorm
        )
        mx.eval(upscaled_video_latent)
        del latent_unnorm, upscaled_unnorm
        del stage_1_video_latent
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        upscale_elapsed = time.perf_counter() - upscale_start

        stage_2_pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.fps,
        )
        stage_2_latent_shape = VideoLatentShape.from_pixel_shape(
            stage_2_pixel_shape, latent_channels=128
        )
        video_tools_2 = self._create_video_tools(stage_2_latent_shape, config.fps)
        conditionings_2 = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )
        video_state_2 = video_tools_2.create_initial_state(
            dtype=config.dtype,
            initial_latent=upscaled_video_latent,
        )
        video_state_2 = apply_conditionings(video_state_2, conditionings_2, video_tools_2)
        video_state_2 = noiser(video_state_2, noise_scale=float(stage_2_sigmas[0]))

        audio_state_2 = None
        audio_tools_2 = None
        if internal_audio_active:
            audio_shape_2 = AudioLatentShape.from_video_pixel_shape(
                stage_2_pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools_2 = self._create_audio_tools(audio_shape_2)
            audio_state_2 = audio_tools_2.create_initial_state(
                dtype=config.dtype,
                initial_latent=stage_1_audio_latent,
            )
            audio_state_2 = noiser(audio_state_2, noise_scale=float(stage_2_sigmas[0]))

        if stage2_state_probe is not None:
            stage2_state_probe(video_state_2, audio_state_2, stage_2_sigmas)

        emit_progress_message(
            f"  Distilled stage 2: {len(stage_2_sigmas) - 1} steps at "
            f"{config.height}x{config.width}"
        )
        if stage_callback:
            stage_callback("stage_2", 0, len(stage_2_sigmas) - 1)

        def stage_2_callback(step: int, _total: int):
            if stage_callback:
                stage_callback("stage_2", step, _total)
            if callback:
                callback(step, total_steps)

        profile_steps = set(config.profile_transformer_steps or ())
        if config.profile_transformer_once:
            profile_steps.add(1)
        profile_blocks = tuple(sorted(set(config.profile_transformer_blocks or ())))

        stage_2_start = time.perf_counter()
        video_state_2, audio_state_2 = self._denoise_loop_simple_av(
            video_state=video_state_2,
            audio_state=audio_state_2,
            sigmas=stage_2_sigmas,
            video_context=positive_encoding,
            audio_context=positive_audio_encoding,
            stepper=stepper,
            callback=stage_2_callback,
            profile_steps=tuple(sorted(profile_steps)),
            profile_blocks=profile_blocks,
        )
        stage_2_elapsed = time.perf_counter() - stage_2_start
        if stage_lora_restore_state is not None:
            lora_restore_start = time.perf_counter()
            restore_transformer_cache_state(self.transformer, stage_lora_restore_state)
            lora_restore_elapsed = time.perf_counter() - lora_restore_start
            emit_progress_message(
                f"  Transformer cache restore complete in {lora_restore_elapsed:.1f}s"
            )
        post_denoise_start = time.perf_counter()

        video_state_2 = video_tools_2.clear_conditioning(video_state_2)
        video_state_2 = video_tools_2.unpatchify(video_state_2)
        final_video_latent = video_state_2.latent

        final_audio_latent = None
        if config.audio_enabled and audio_state_2 is not None and audio_tools_2 is not None:
            audio_state_2 = audio_tools_2.clear_conditioning(audio_state_2)
            audio_state_2 = audio_tools_2.unpatchify(audio_state_2)
            final_audio_latent = audio_state_2.latent

        del self.transformer
        self.transformer = None
        gc.collect()
        mx.clear_cache()

        if latent_save_path is not None:
            self._save_distilled_two_stage_latents(
                latent_save_path,
                stage_1_video_latent=stage_1_video_latent_for_save,
                stage_2_video_latent=final_video_latent,
                stage_1_audio_latent=stage_1_audio_latent_for_save,
                stage_2_audio_latent=final_audio_latent,
                progress_message=progress_message,
            )
            del stage_1_video_latent_for_save, stage_1_audio_latent_for_save
            gc.collect()
            mx.clear_cache()

        effective_tiling = config._get_tiling_config()
        post_denoise_elapsed = time.perf_counter() - post_denoise_start
        video_decoder_load_elapsed = 0.0
        if decode_video:
            video_decoder_load_elapsed, video_decoder_loaded_now = self._ensure_video_decoder(
                emit_progress_message
            )
            try:
                decoder_name = self.video_decoder.__class__.__name__
                tiling_desc = "tiled" if effective_tiling else "not tiled"
                emit_progress_message(
                    f"  VAE decode started ({decoder_name}, {tiling_desc})..."
                )
                decode_start = time.perf_counter()
                if effective_tiling:
                    emit_progress_message(
                        "  Using tiled VAE decoding (preventing GPU watchdog timeout)"
                    )
                    video = None
                    for chunk in decode_tiled(final_video_latent, self.video_decoder, effective_tiling):
                        video = chunk if video is None else mx.concatenate([video, chunk], axis=2)
                        mx.eval(video)
                        del chunk
                        gc.collect()
                        mx.clear_cache()
                else:
                    video = decode_latent(final_video_latent, self.video_decoder)
                mx.eval(video)
                decode_elapsed = time.perf_counter() - decode_start
                emit_progress_message(f"  VAE decode complete in {decode_elapsed:.1f}s")
            finally:
                if video_decoder_loaded_now:
                    self.video_decoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            # Streaming-decode path: see generate_distilled_two_stage's
            # decode_video=False docstring.  Returning the latent lets the
            # caller iterate VAE chunks straight into a streaming encoder.
            video = final_video_latent
            decode_elapsed = 0.0

        audio_waveform = None
        audio_decoder_load_elapsed = 0.0
        if final_audio_latent is not None:
            audio_decoder_load_elapsed, audio_decoder_loaded_now = self._ensure_audio_decoder(
                emit_progress_message
            )
            try:
                audio_decode_start = time.perf_counter()
                audio_waveform = self._decode_audio(final_audio_latent)
                audio_decode_elapsed = time.perf_counter() - audio_decode_start
            finally:
                if audio_decoder_loaded_now:
                    self.audio_decoder = None
                    self.vocoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            audio_decode_elapsed = 0.0

        setup_elapsed = max(0.0, upscale_start - call_start - stage_2_lora_elapsed)
        self.last_timing_sections = [("pipeline setup", setup_elapsed)]
        if stage_2_lora_elapsed:
            self.last_timing_sections.append(("stage 2 lora fuse", stage_2_lora_elapsed))
        self.last_timing_sections.extend([
            ("spatial upscale", upscale_elapsed),
            ("distilled stage 2 denoise", stage_2_elapsed),
        ])
        if lora_restore_elapsed:
            self.last_timing_sections.append(("lora restore", lora_restore_elapsed))
        self.last_timing_sections.append(("post-denoise prep", post_denoise_elapsed))
        if video_decoder_load_elapsed:
            self.last_timing_sections.append(("vae decoder load", video_decoder_load_elapsed))
        if decode_video:
            self.last_timing_sections.append(("vae decode", decode_elapsed))
        if audio_decoder_load_elapsed:
            self.last_timing_sections.append(("audio decoder load", audio_decoder_load_elapsed))
        if audio_decode_elapsed:
            self.last_timing_sections.append(("audio decode", audio_decode_elapsed))

        return video, audio_waveform

    def __call__(
        self,
        positive_encoding: mx.array,
        negative_encoding: mx.array | None = None,
        config: AVCFGConfig = None,
        images: list[ImageCondition] | None = None,
        callback: Callable[[int, int], None] | None = None,
        progress_message: Callable[[str], None] | None = None,
        positive_audio_encoding: mx.array | None = None,
        negative_audio_encoding: mx.array | None = None,
        stg_scale: float = 0.0,
        stg_blocks: list[int] | None = None,
        stg_cutoff: float = 1.0,
        guider_override=None,
        ge_gamma: float = 0.0,
        sampler: str = "euler",
        temporal_upscaler=None,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
        latent_save_path: str | None = None,
        decode_video: bool = True,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Generate video (and optionally audio) using single-stage CFG pipeline.

        When `decode_video=False`, returns `(final_video_latent, audio_waveform)`
        instead of decoding inside the pipeline - see
        `generate_distilled_two_stage` for the streaming-decode rationale.

        Args:
            positive_encoding: Encoded positive prompt for video [B, T, D].
            negative_encoding: Encoded negative prompt for video [B, T, D].
            config: Pipeline configuration.
            images: Optional list of image conditions.
            callback: Optional callback(step, total_steps).
            positive_audio_encoding: Encoded positive prompt for audio [B, T, D].
                Required when config.audio_enabled is True.
            negative_audio_encoding: Encoded negative prompt for audio [B, T, D].
                Required when config.audio_enabled is True.
            latent_save_path: Optional npz sidecar path for final video/audio latents.

        Returns:
            Tuple of (video, audio) where:
                - video: Generated video tensor [F, H, W, C] in pixel space (0-255).
                - audio: Audio waveform [B, channels, samples] at output_sample_rate,
                         or None if audio_enabled is False.
        """
        call_start = time.perf_counter()
        self.last_timing_sections = []
        images = images or []

        def emit_progress_message(message: str) -> None:
            """Route a status message through `progress_message` when the
            caller supplied one (so any live progress bars stay coherent);
            fall back to `print()` for callers without a bar UI."""
            if progress_message is not None:
                progress_message(message)
            else:
                print(message)

        internal_audio_active = self.is_av_model and (config.use_internal_audio_branch or config.audio_enabled)

        # Detect no-CFG fast path: when both video and audio CFG scales are 1.0,
        # the negative encoding is mathematically a no-op (out = neg + 1*(pos-neg)
        # = pos).  Skip negative encoding altogether and route through the
        # simple no-CFG loop, saving one full transformer pass per step.
        # This is the default for the distilled model (cfg_scale = 1.0).
        skip_cfg = (
            config.cfg_scale == 1.0
            and config.audio_cfg_scale == 1.0
            and config.rescale_scale == 0.0  # rescale needs both passes
        )

        # AV checkpoints can use audio context internally even for silent video output.
        if config.audio_enabled or internal_audio_active:
            if positive_audio_encoding is None:
                raise ValueError(
                    "Positive audio encoding required for AudioVideo generation."
                )
            if not skip_cfg and negative_audio_encoding is None:
                raise ValueError(
                    "Negative audio encoding required when audio_cfg_scale != 1.0."
                )
        if not skip_cfg and negative_encoding is None:
            raise ValueError(
                "Negative video encoding required when cfg_scale != 1.0."
            )
        if config.audio_enabled:
            if (
                (self.audio_decoder is None or self.vocoder is None)
                and self.audio_decoder_loader is None
            ):
                raise ValueError(
                    "Audio decoder and vocoder required when audio_enabled is True."
                )

        # Set seed
        mx.random.seed(config.seed)

        # Create components
        noiser = GaussianNoiser()
        stepper = self.diffusion_step

        # Create separate guiders for video and audio (LTX-2.3 reference uses different scales)
        if guider_override is not None:
            video_guider = guider_override
        elif config.rescale_scale > 0:
            video_guider = CFGStarRescalingGuider(scale=config.cfg_scale)
        else:
            video_guider = CFGGuider(scale=config.cfg_scale)
        # Audio always uses standard guiders (APG not tested with audio)
        if config.rescale_scale > 0:
            audio_guider = CFGStarRescalingGuider(scale=config.audio_cfg_scale)
        else:
            audio_guider = CFGGuider(scale=config.audio_cfg_scale)
        # Legacy single guider for video-only path
        guider = video_guider

        # Create output shape
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.fps,
        )
        latent_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=128
        )

        # Create video tools
        video_tools = self._create_video_tools(latent_shape, config.fps)

        # Create image conditionings
        if images and self.video_encoder is None:
            raise ValueError("Video encoder required when image conditioning is provided")
        conditionings = create_image_conditionings(
            images,
            self.video_encoder,
            config.height,
            config.width,
            config.dtype,
        )

        # Create initial video state
        video_state = video_tools.create_initial_state(dtype=config.dtype)

        # Apply conditionings
        video_state = apply_conditionings(video_state, conditionings, video_tools)

        # Get sigma schedule. Distilled checkpoints use their trained fixed
        # schedule; dev checkpoints use the token-count shifted scheduler.
        if config.use_distilled_sigmas:
            sigmas = mx.array(DISTILLED_SIGMA_VALUES)
        else:
            sigmas = self.scheduler.execute(steps=config.num_inference_steps)

        # Add noise to video
        video_state = noiser(video_state, noise_scale=1.0)

        profile_steps = set(config.profile_transformer_steps)
        if config.profile_transformer_once:
            profile_steps.add(1)
        profile_blocks = tuple(sorted(set(config.profile_transformer_blocks)))

        # Keep an internal audio state for AV checkpoints when requested.
        audio_state = None
        audio_tools = None
        if internal_audio_active:
            # Create audio latent shape from video duration
            audio_shape = AudioLatentShape.from_video_pixel_shape(
                pixel_shape,
                channels=config.audio_vae_channels,
                mel_bins=config.audio_mel_bins,
                sample_rate=config.audio_sample_rate,
                hop_length=config.audio_hop_length,
                audio_latent_downsample_factor=config.audio_downsample_factor,
            )
            audio_tools = self._create_audio_tools(audio_shape)
            audio_state = audio_tools.create_initial_state(dtype=config.dtype)
            audio_state = noiser(audio_state, noise_scale=1.0)

        if callback:
            callback(0, len(sigmas) - 1)

        # Run denoising loop
        denoise_start = time.perf_counter()
        if internal_audio_active and audio_state is not None:
            # Joint audio-video denoising with separate guidance per modality
            _stg_guider = None
            _stg_perturbations = None
            if stg_scale > 0:
                _stg_guider = STGGuider(scale=stg_scale)
                _stg_perturbations = create_batched_stg_config(
                    batch_size=1,
                    skip_video_self_attn=True,
                    blocks=stg_blocks,
                )
                cutoff_pct = int(stg_cutoff * 100)
                emit_progress_message(
                    f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%"
                )

            if skip_cfg and sampler != "heun" and _stg_guider is None:
                # Fast path: no CFG, no STG, no Heun - just the positive-only
                # transformer pass per step.  Saves one full transformer forward
                # per step vs the CFG loop.
                video_state, audio_state = self._denoise_loop_simple_av(
                    video_state=video_state,
                    audio_state=audio_state,
                    sigmas=sigmas,
                    video_context=positive_encoding,
                    audio_context=positive_audio_encoding,
                    stepper=stepper,
                    callback=callback,
                    profile_steps=tuple(sorted(profile_steps)),
                    profile_blocks=profile_blocks,
                )
            elif sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                emit_progress_message("  Using Heun sampler (2x model evals per step)")
                video_state, audio_state = self._denoise_loop_heun_av(
                    video_state=video_state,
                    audio_state=audio_state,
                    sigmas=sigmas,
                    positive_video_context=positive_encoding,
                    negative_video_context=negative_encoding,
                    positive_audio_context=positive_audio_encoding,
                    negative_audio_context=negative_audio_encoding,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    stepper=heun_stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                )
            else:
                video_state, audio_state = self._denoise_loop_cfg_av(
                    video_state=video_state,
                    audio_state=audio_state,
                    sigmas=sigmas,
                    positive_video_context=positive_encoding,
                    negative_video_context=negative_encoding,
                    positive_audio_context=positive_audio_encoding,
                    negative_audio_context=negative_audio_encoding,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    stepper=stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                    profile_steps=tuple(sorted(profile_steps)),
                    profile_blocks=profile_blocks,
                )
        else:
            # Set up STG if enabled
            _stg_guider = None
            _stg_perturbations = None
            if stg_scale > 0:
                _stg_guider = STGGuider(scale=stg_scale)
                _stg_perturbations = create_batched_stg_config(
                    batch_size=1,
                    skip_video_self_attn=True,
                    blocks=stg_blocks,
                )
                cutoff_pct = int(stg_cutoff * 100)
                emit_progress_message(
                    f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%"
                )

            # Video-only denoising - select loop based on sampler
            if sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                emit_progress_message("  Using Heun sampler (2x model evals per step)")
                video_state = self._denoise_loop_heun(
                    video_state=video_state,
                    sigmas=sigmas,
                    positive_context=positive_encoding,
                    negative_context=negative_encoding,
                    guider=guider,
                    stepper=heun_stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
                )
            else:
                video_state = self._denoise_loop_cfg(
                    video_state=video_state,
                    sigmas=sigmas,
                    positive_context=positive_encoding,
                    negative_context=negative_encoding,
                    guider=guider,
                    stepper=stepper,
                    callback=callback,
                    stg_guider=_stg_guider,
                    stg_perturbations=_stg_perturbations,
                    stg_cutoff=stg_cutoff,
                    ge_gamma=ge_gamma,
                    cross_attn_scale=cross_attn_scale,
                    cross_attn_start_block=cross_attn_start_block,
            )
        denoise_elapsed = time.perf_counter() - denoise_start
        post_denoise_start = time.perf_counter()

        # Clear conditioning and unpatchify video
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        final_video_latent = video_state.latent
        final_audio_latent = None

        # Apply temporal upscaler (2x frame interpolation) if provided
        if temporal_upscaler is not None:
            if self.video_decoder is None:
                raise ValueError("Video decoder required for temporal upscaling.")
            input_frames = final_video_latent.shape[2]
            emit_progress_message(
                f"  Temporal upscaling: {input_frames} -> {input_frames * 2 - 1} latent frames..."
            )
            # Un-normalize latent (upscaler trained on raw latents)
            std = self.video_decoder.std_of_means.reshape(1, -1, 1, 1, 1)
            mean = self.video_decoder.mean_of_means.reshape(1, -1, 1, 1, 1)
            latent_unnorm = final_video_latent * std + mean
            # Upscale
            latent_upscaled = temporal_upscaler(latent_unnorm)
            mx.eval(latent_upscaled)
            # Re-normalize
            final_video_latent = (latent_upscaled - mean) / std
            mx.eval(final_video_latent)
            del latent_unnorm, latent_upscaled
            output_frames = final_video_latent.shape[2]
            emit_progress_message(
                f"  Temporal upscale complete: {output_frames} latent frames"
            )

        # Prepare final audio latent before unloading transformer and decoding.
        if config.audio_enabled and audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            final_audio_latent = audio_state.latent

        # Unload transformer before VAE decode to free memory
        del self.transformer
        self.transformer = None
        gc.collect()
        mx.clear_cache()

        if latent_save_path is not None:
            self._save_final_latents(
                latent_save_path,
                video_latent=final_video_latent,
                audio_latent=final_audio_latent,
                progress_message=progress_message,
            )
            gc.collect()
            mx.clear_cache()

        # Decode video (auto-tile for large generations to prevent Metal watchdog timeout)
        if (
            decode_video
            and self.video_decoder is None
            and self.video_decoder_loader is None
        ):
            raise ValueError("Video decoder required for VAE decode.")
        effective_tiling = config._get_tiling_config()
        post_denoise_elapsed = time.perf_counter() - post_denoise_start
        video_decoder_load_elapsed = 0.0
        if decode_video:
            video_decoder_load_elapsed, video_decoder_loaded_now = self._ensure_video_decoder(
                emit_progress_message
            )
            try:
                decoder_name = self.video_decoder.__class__.__name__
                tiling_desc = "tiled" if effective_tiling else "not tiled"
                emit_progress_message(
                    f"  VAE decode started ({decoder_name}, {tiling_desc})..."
                )
                decode_start = time.perf_counter()
                if effective_tiling:
                    emit_progress_message(
                        "  Using tiled VAE decoding (preventing GPU watchdog timeout)"
                    )
                    video = None
                    for chunk in decode_tiled(final_video_latent, self.video_decoder, effective_tiling):
                        video = chunk if video is None else mx.concatenate([video, chunk], axis=2)
                        mx.eval(video)
                        del chunk
                        gc.collect()
                        mx.clear_cache()
                else:
                    video = decode_latent(final_video_latent, self.video_decoder)
                mx.eval(video)
                decode_elapsed = time.perf_counter() - decode_start
                emit_progress_message(f"  VAE decode complete in {decode_elapsed:.1f}s")
            finally:
                if video_decoder_loaded_now:
                    self.video_decoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            # Streaming-decode path: see generate_distilled_two_stage's
            # decode_video=False docstring.
            video = final_video_latent
            decode_elapsed = 0.0

        # Decode audio if enabled
        audio_waveform = None
        audio_decoder_load_elapsed = 0.0
        if final_audio_latent is not None:
            audio_decoder_load_elapsed, audio_decoder_loaded_now = self._ensure_audio_decoder(
                emit_progress_message
            )
            try:
                audio_decode_start = time.perf_counter()
                audio_waveform = self._decode_audio(final_audio_latent)
                audio_decode_elapsed = time.perf_counter() - audio_decode_start
            finally:
                if audio_decoder_loaded_now:
                    self.audio_decoder = None
                    self.vocoder = None
                    gc.collect()
                    mx.clear_cache()
        else:
            audio_decode_elapsed = 0.0

        self.last_timing_sections = [
            ("pipeline setup", denoise_start - call_start),
            ("denoise", denoise_elapsed),
            ("post-denoise prep", post_denoise_elapsed),
        ]
        if video_decoder_load_elapsed:
            self.last_timing_sections.append(("vae decoder load", video_decoder_load_elapsed))
        if decode_video:
            self.last_timing_sections.append(("vae decode", decode_elapsed))
        if audio_decoder_load_elapsed:
            self.last_timing_sections.append(("audio decoder load", audio_decoder_load_elapsed))
        if audio_decode_elapsed:
            self.last_timing_sections.append(("audio decode", audio_decode_elapsed))

        return video, audio_waveform


def create_av_pipeline(
    transformer: LTXModel,
    video_encoder: NativeConv3dVideoEncoder | None,
    video_decoder: NativeConv3dVideoDecoder | None,
    audio_decoder: AudioDecoder | None = None,
    vocoder: Vocoder | None = None,
    video_decoder_loader: Callable[[], NativeConv3dVideoDecoder] | None = None,
    audio_decoder_loader: Callable[[], tuple[AudioDecoder, object, int]] | None = None,
    audio_sample_rate: int = 24000,
) -> AVPipeline:
    """
    Create a single-stage CFG pipeline.

    Args:
        transformer: LTX transformer model (LTXModel or LTXAVModel).
        video_encoder: Optional VAE encoder.
        video_decoder: Optional VAE decoder.
        audio_decoder: Optional audio VAE decoder (required for audio generation).
        vocoder: Optional vocoder (required for audio generation).
        video_decoder_loader: Optional lazy VAE decoder loader.
        audio_decoder_loader: Optional lazy audio decode stack loader.
        audio_sample_rate: Output sample rate before lazy audio decode stack load.

    Returns:
        Configured AVPipeline.
    """
    return AVPipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
        video_decoder_loader=video_decoder_loader,
        audio_decoder_loader=audio_decoder_loader,
        audio_sample_rate=audio_sample_rate,
    )
