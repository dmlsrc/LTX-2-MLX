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
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .common import (
    ImageCondition,
    apply_conditionings,
    create_image_conditionings,
    modality_from_state,
    audio_modality_from_state,
    post_process_latent,
)
from ..components import (
    CFGGuider,
    CFGStarRescalingGuider,
    DISTILLED_SIGMA_VALUES,
    EulerDiffusionStep,
    GaussianNoiser,
    LTX2Scheduler,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    VideoLatentPatchifier,
)
from ..components.guiders import GuiderProtocol
from ..components.diffusion_steps import HeunDiffusionStep
from ..components.patchifiers import AudioPatchifier
from ..conditioning.tools import VideoLatentTools, AudioLatentTools
from ..model.transformer import LTXModel, LTXAVModel, LTXModelType, X0Model, Modality
from ..components.guiders import STGGuider
from ..components.perturbations import create_batched_stg_config, BatchedPerturbationConfig
from ..model.video_vae.simple_decoder import SimpleVideoDecoder, decode_latent
from ..model.video_vae.simple_encoder import SimpleVideoEncoder
from ..model.video_vae.tiling import TilingConfig, decode_tiled
from ..model.audio_vae import AudioDecoder, Vocoder
from ..types import (
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
    NATIVE_FPS
)


@dataclass
class OneStageCFGConfig:
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
    tiling_config: Optional[TilingConfig] = None
    auto_tiling: bool = True

    def _get_tiling_config(self) -> Optional[TilingConfig]:
        """Return tiling config, auto-enabling."""
        if self.tiling_config is not None:
            return self.tiling_config
        if not self.auto_tiling:
            return None
        return TilingConfig.auto(self.height, self.width, self.num_frames)

    # Compute settings
    dtype: mx.Dtype = mx.bfloat16
    profile_transformer_once: bool = False
    profile_transformer_steps: Tuple[int, ...] = ()
    profile_transformer_blocks: Tuple[int, ...] = ()

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


class OneStagePipeline:
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
        video_encoder: Optional[SimpleVideoEncoder],
        video_decoder: Optional[SimpleVideoDecoder],
        audio_decoder: Optional[AudioDecoder] = None,
        vocoder: Optional[Vocoder] = None,
    ):
        """
        Initialize the single-stage pipeline.

        Args:
            transformer: LTX transformer model (LTXModel for video-only, LTXAVModel for audio+video).
            video_encoder: Optional VAE encoder for image conditioning.
            video_decoder: Optional VAE decoder for decoding latents to video.
            audio_decoder: Optional audio VAE decoder for decoding audio latents to mel spectrograms.
            vocoder: Optional vocoder for converting mel spectrograms to waveforms.
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
        blocks: Tuple[int, ...] = (),
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
        audio_latent: Optional[mx.array] = None,
    ) -> None:
        """Save final video/audio latents as a sidecar npz file."""
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
        print(f"  Saved final latents: {path}")

    @classmethod
    def _save_distilled_two_stage_latents(
        cls,
        path: str,
        stage_1_video_latent: mx.array,
        stage_2_video_latent: mx.array,
        stage_1_audio_latent: Optional[mx.array] = None,
        stage_2_audio_latent: Optional[mx.array] = None,
    ) -> None:
        """Save both distilled stages while preserving the final-latent keys."""
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
        print(f"  Saved distilled stage latents: {path}")

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
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
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
            denoised = post_process_latent(
                denoised, video_state.denoise_mask, video_state.clean_latent
            )

            # Euler step
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.eval(video_state.latent)

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
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
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

            # ── First evaluation: denoised at current sigma ──
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

            denoised = post_process_latent(
                denoised, video_state.denoise_mask, video_state.clean_latent
            )

            # ── Euler prediction to get predicted sample ──
            from ..core_utils import to_velocity
            velocity = to_velocity(video_state.latent, sigmas[step_idx], denoised)
            dt = sigma_next - sigma
            predicted = video_state.latent.astype(mx.float32) + velocity.astype(mx.float32) * float(dt)
            predicted = predicted.astype(video_state.latent.dtype)
            mx.eval(predicted)

            # Skip corrector on last step (sigma_next == 0)
            if sigma_next == 0:
                video_state = video_state.replace(latent=denoised)
                mx.eval(video_state.latent)
                if callback:
                    callback(step_idx + 1, num_steps)
                continue

            # ── Second evaluation: denoised at predicted point ──
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

            denoised_at_predicted = post_process_latent(
                denoised_at_predicted, video_state.denoise_mask, video_state.clean_latent
            )

            # ── Heun corrector step ──
            new_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=denoised,
                sigmas=sigmas,
                step_index=step_idx,
                denoised_at_predicted=denoised_at_predicted,
            )

            video_state = video_state.replace(latent=new_latent)
            mx.eval(video_state.latent)

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
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
        profile_steps: Tuple[int, ...] = (),
        profile_blocks: Tuple[int, ...] = (),
    ) -> Tuple[LatentState, LatentState]:
        """Run joint audio-video denoising loop with separate guidance per modality."""
        num_steps = len(sigmas) - 1
        need_cfg = video_guider.enabled() or audio_guider.enabled()
        prev_velocity = None
        profile_step_set = set(profile_steps)

        def print_profile(step: int, events: List[Tuple[str, float]]) -> None:
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

            def mark_profile(name: str, *arrays: mx.array) -> None:
                nonlocal profile_started_at
                if profile_events is None:
                    return
                if arrays:
                    mx.eval(*arrays)
                now = time.perf_counter()
                profile_events.append((name, now - profile_started_at))
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

            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )
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

            mx.eval(video_state.latent, audio_state.latent)
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
        callback: Optional[Callable[[int, int], None]] = None,
        stg_guider: Optional[STGGuider] = None,
        stg_perturbations: Optional[BatchedPerturbationConfig] = None,
        stg_cutoff: float = 1.0,
        ge_gamma: float = 0.0,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
    ) -> Tuple[LatentState, LatentState]:
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

            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised = post_process_latent(
                audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
            )

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
                mx.eval(video_state.latent, audio_state.latent)
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

            video_denoised_at_predicted = post_process_latent(
                video_denoised_at_predicted, video_state.denoise_mask, video_state.clean_latent
            )
            audio_denoised_at_predicted = post_process_latent(
                audio_denoised_at_predicted, audio_state.denoise_mask, audio_state.clean_latent
            )

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
            mx.eval(video_state.latent, audio_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        if cross_attn_scale != 1.0:
            self._clear_cross_attn_scales()

        return video_state, audio_state

    def _denoise_loop_simple_av(
        self,
        video_state: LatentState,
        audio_state: Optional[LatentState],
        sigmas: mx.array,
        video_context: mx.array,
        audio_context: Optional[mx.array],
        stepper: EulerDiffusionStep,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[LatentState, Optional[LatentState]]:
        """Run distilled denoising without CFG for video-only or AV transformers."""
        num_steps = len(sigmas) - 1

        for step_idx in range(num_steps):
            sigma = float(sigmas[step_idx])
            video_modality = modality_from_state(video_state, video_context, sigma)

            if self.is_av_model:
                audio_modality = (
                    audio_modality_from_state(audio_state, audio_context, sigma)
                    if audio_state is not None
                    else None
                )
                result = (
                    self.transformer(video_modality, audio_modality)
                    if audio_modality is not None
                    else self.transformer(video_modality)
                )
                if isinstance(result, tuple):
                    video_denoised, audio_denoised = result
                else:
                    video_denoised = result
                    audio_denoised = None
            else:
                video_denoised = self.transformer(video_modality)
                audio_denoised = None

            video_denoised = post_process_latent(
                video_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            new_video_latent = stepper.step(
                sample=video_state.latent,
                denoised_sample=video_denoised,
                sigmas=sigmas,
                step_index=step_idx,
            )
            video_state = video_state.replace(latent=new_video_latent)

            if audio_state is not None and audio_denoised is not None:
                audio_denoised = post_process_latent(
                    audio_denoised, audio_state.denoise_mask, audio_state.clean_latent
                )
                new_audio_latent = stepper.step(
                    sample=audio_state.latent,
                    denoised_sample=audio_denoised,
                    sigmas=sigmas,
                    step_index=step_idx,
                )
                audio_state = audio_state.replace(latent=new_audio_latent)
                mx.eval(video_state.latent, audio_state.latent)
            else:
                mx.eval(video_state.latent)

            if callback:
                callback(step_idx + 1, num_steps)

        return video_state, audio_state

    def generate_distilled_two_stage(
        self,
        positive_encoding: mx.array,
        config: OneStageCFGConfig,
        spatial_upscaler,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[int, int], None]] = None,
        stage_callback: Optional[Callable[[str, int, int], None]] = None,
        progress_message: Optional[Callable[[str], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        latent_save_path: Optional[str] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Generate with the distilled AV checkpoint in two stages.

        Stage 1 denoises at half spatial resolution with the official distilled
        sigmas, then stage 2 spatially upscales and refines with the stage-2
        distilled sigmas. There is intentionally no CFG in either stage.
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
        if self.video_decoder is None:
            raise ValueError("Video decoder required for VAE decode.")
        if spatial_upscaler is None:
            raise ValueError("Spatial upscaler required for distilled two-stage generation.")

        internal_audio_active = self.is_av_model and (
            config.use_internal_audio_branch or config.audio_enabled
        )
        if internal_audio_active and positive_audio_encoding is None:
            raise ValueError(
                "Positive audio encoding required for AudioVideo distilled two-stage generation."
            )
        if config.audio_enabled and (self.audio_decoder is None or self.vocoder is None):
            raise ValueError("Audio decoder and vocoder required when audio_enabled is True.")

        mx.random.seed(config.seed)
        noiser = GaussianNoiser()
        stepper = self.diffusion_step
        stage_1_sigmas = mx.array(DISTILLED_SIGMA_VALUES)
        stage_2_sigmas = mx.array(STAGE_2_DISTILLED_SIGMA_VALUES)
        total_steps = len(stage_1_sigmas) + len(stage_2_sigmas) - 2
        if callback:
            callback(0, total_steps)

        def emit_progress_message(message: str) -> None:
            if progress_message is not None:
                progress_message(message)
            else:
                print(message)

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
            audio_state = audio_state.replace(
                latent=self._channelwise_normalize_audio_noise(audio_state.latent)
            )

        emit_progress_message(
            f"  Distilled stage 1: {len(stage_1_sigmas) - 1} steps at "
            f"{stage_1_height}x{stage_1_width}"
        )
        if stage_callback:
            stage_callback("stage_1", 0, len(stage_1_sigmas) - 1)

        def stage_1_callback(step: int, _total: int):
            if stage_callback:
                stage_callback("stage_1", step, _total)
            if callback:
                callback(step, total_steps)

        stage_1_start = time.perf_counter()
        video_state, audio_state = self._denoise_loop_simple_av(
            video_state=video_state,
            audio_state=audio_state,
            sigmas=stage_1_sigmas,
            video_context=positive_encoding,
            audio_context=positive_audio_encoding,
            stepper=stepper,
            callback=stage_1_callback,
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

        upscale_start = time.perf_counter()
        emit_progress_message("  Upsampling latent 2x with spatial upscaler...")
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
        if stage_1_video_latent_for_save is None:
            del stage_1_video_latent
        gc.collect()
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
        )
        stage_2_elapsed = time.perf_counter() - stage_2_start
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
            )
            del stage_1_video_latent_for_save, stage_1_audio_latent_for_save
            gc.collect()
            mx.clear_cache()

        effective_tiling = config._get_tiling_config()
        post_denoise_elapsed = time.perf_counter() - post_denoise_start
        decoder_name = self.video_decoder.__class__.__name__
        tiling_desc = "tiled" if effective_tiling else "not tiled"
        print(f"  VAE decode started ({decoder_name}, {tiling_desc})...")
        decode_start = time.perf_counter()
        if effective_tiling:
            print("  Using tiled VAE decoding (preventing GPU watchdog timeout)")
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
        print(f"  VAE decode complete in {decode_elapsed:.1f}s")

        audio_waveform = None
        if final_audio_latent is not None:
            audio_decode_start = time.perf_counter()
            audio_waveform = self._decode_audio(final_audio_latent)
            audio_decode_elapsed = time.perf_counter() - audio_decode_start
        else:
            audio_decode_elapsed = 0.0

        self.last_timing_sections = [
            ("pipeline setup", stage_1_start - call_start),
            ("distilled stage 1 denoise", stage_1_elapsed),
            ("spatial upscale", upscale_elapsed),
            ("distilled stage 2 denoise", stage_2_elapsed),
            ("post-denoise prep", post_denoise_elapsed),
            ("vae decode", decode_elapsed),
        ]
        if audio_decode_elapsed:
            self.last_timing_sections.append(("audio decode", audio_decode_elapsed))

        return video, audio_waveform

    def __call__(
        self,
        positive_encoding: mx.array,
        negative_encoding: mx.array,
        config: OneStageCFGConfig,
        images: Optional[List[ImageCondition]] = None,
        callback: Optional[Callable[[int, int], None]] = None,
        positive_audio_encoding: Optional[mx.array] = None,
        negative_audio_encoding: Optional[mx.array] = None,
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        stg_cutoff: float = 1.0,
        guider_override=None,
        ge_gamma: float = 0.0,
        sampler: str = "euler",
        temporal_upscaler=None,
        cross_attn_scale: float = 1.0,
        cross_attn_start_block: int = 40,
        latent_save_path: Optional[str] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Generate video (and optionally audio) using single-stage CFG pipeline.

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

        internal_audio_active = self.is_av_model and (config.use_internal_audio_branch or config.audio_enabled)

        # AV checkpoints can use audio context internally even for silent video output.
        if config.audio_enabled or internal_audio_active:
            if positive_audio_encoding is None or negative_audio_encoding is None:
                raise ValueError(
                    "Audio encoding required for AudioVideo generation. "
                    "Provide positive_audio_encoding and negative_audio_encoding."
                )
        if config.audio_enabled:
            if self.audio_decoder is None or self.vocoder is None:
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
                print(f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%")

            if sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                print(f"  Using Heun sampler (2x model evals per step)")
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
                print(f"  STG guidance: scale={stg_scale}, blocks={stg_blocks or 'all'}, cutoff={cutoff_pct}%")

            # Video-only denoising — select loop based on sampler
            if sampler == "heun":
                heun_stepper = HeunDiffusionStep()
                print(f"  Using Heun sampler (2x model evals per step)")
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
            print(f"  Temporal upscaling: {input_frames} → {input_frames * 2 - 1} latent frames...")
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
            print(f"  Temporal upscale complete: {output_frames} latent frames")

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
            )
            gc.collect()
            mx.clear_cache()

        # Decode video (auto-tile for large generations to prevent Metal watchdog timeout)
        if self.video_decoder is None:
            raise ValueError("Video decoder required for VAE decode.")
        effective_tiling = config._get_tiling_config()
        post_denoise_elapsed = time.perf_counter() - post_denoise_start
        decoder_name = self.video_decoder.__class__.__name__
        tiling_desc = "tiled" if effective_tiling else "not tiled"
        print(f"  VAE decode started ({decoder_name}, {tiling_desc})...")
        decode_start = time.perf_counter()
        if effective_tiling:
            print(f"  Using tiled VAE decoding (preventing GPU watchdog timeout)")
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
        print(f"  VAE decode complete in {decode_elapsed:.1f}s")

        # Decode audio if enabled
        audio_waveform = None
        if final_audio_latent is not None:
            audio_decode_start = time.perf_counter()
            audio_waveform = self._decode_audio(final_audio_latent)
            audio_decode_elapsed = time.perf_counter() - audio_decode_start
        else:
            audio_decode_elapsed = 0.0

        self.last_timing_sections = [
            ("pipeline setup", denoise_start - call_start),
            ("denoise", denoise_elapsed),
            ("post-denoise prep", post_denoise_elapsed),
            ("vae decode", decode_elapsed),
        ]
        if audio_decode_elapsed:
            self.last_timing_sections.append(("audio decode", audio_decode_elapsed))

        return video, audio_waveform


def create_one_stage_pipeline(
    transformer: LTXModel,
    video_encoder: Optional[SimpleVideoEncoder],
    video_decoder: Optional[SimpleVideoDecoder],
    audio_decoder: Optional[AudioDecoder] = None,
    vocoder: Optional[Vocoder] = None,
) -> OneStagePipeline:
    """
    Create a single-stage CFG pipeline.

    Args:
        transformer: LTX transformer model (LTXModel or LTXAVModel).
        video_encoder: Optional VAE encoder.
        video_decoder: Optional VAE decoder.
        audio_decoder: Optional audio VAE decoder (required for audio generation).
        vocoder: Optional vocoder (required for audio generation).

    Returns:
        Configured OneStagePipeline.
    """
    return OneStagePipeline(
        transformer=transformer,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
    )
