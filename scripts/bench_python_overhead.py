"""Isolation micro-bench: measure pure Python overhead in BasicAVTransformerBlock.

Hypothesis: at stage-1 (small T), our per-block Python overhead is ~30ms x 48
blocks = ~1.4 s/step, which would entirely explain the 42% gap with mlx-video
at small T.  This script simulates the Python work without any MLX ops, so
the result is a hard upper bound on what refactoring could reclaim.

Per-block Python work (no MLX), counted from BasicAVTransformerBlock.__call__:
  - ~15 unconditional `mark_profile(...)` closure calls
  - 2-4 `assert` statements
  - 4-5 `getattr(self, '_some_attr', None)` lookups
  - 1 method call to `_apply_text_cross_attention` per modality (2 total)
  - 2 `dataclass.replace(x=vx)` calls (video + audio)
  - ~10 hasattr/isinstance checks
  - ~5 perturbation flag computations

If the timed cost per step here is dramatically less than the observed 1.4 s
gap, Python overhead can't be the cause and we look elsewhere.
"""

import time
from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class FakeArgs:
    """Mimics TransformerArgs."""
    x: int = 0
    context: int = 0
    timesteps: int = 0
    extra1: int = 0
    extra2: int = 0
    extra3: int = 0
    extra4: int = 0


class FakeBlock:
    """Mimics the per-block Python machinery in BasicAVTransformerBlock."""

    def __init__(self, idx: int):
        self.idx = idx
        self.norm_eps = 1e-6
        self.cross_attention_adaln = True
        # _cross_attn_scale intentionally NOT set, so getattr returns None.

    def _apply_text_cross_attention(
        self,
        x,
        context,
        attn,
        scale_shift_table,
        prompt_scale_shift_table,
        timestep,
        prompt_timestep,
        context_mask,
        profile_name=None,
        mark_profile=None,
    ):
        # mimic the helper method body for non-profile path
        if self.cross_attention_adaln:
            return x  # would do real MLX work
        return x

    def __call__(
        self,
        video: Optional[FakeArgs] = None,
        audio: Optional[FakeArgs] = None,
        perturbations=None,
        profile_events=None,
    ):
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        profile_started_at = time.perf_counter() if profile_events is not None else 0.0

        def mark_profile(name, *arrays):
            nonlocal profile_started_at
            if profile_events is None:
                return
            now = time.perf_counter()
            profile_events.append((name, now - profile_started_at))
            profile_started_at = now

        run_vx = vx is not None and video is not None and video is not None and vx >= 0
        run_ax = ax is not None and audio is not None and audio is not None and ax >= 0
        run_a2v = run_vx and ax is not None and ax >= 0
        run_v2a = run_ax and vx is not None and vx >= 0

        skip_video_self = perturbations is not None and False
        skip_audio_self = perturbations is not None and False
        skip_a2v = perturbations is not None and False
        skip_v2a = perturbations is not None and False

        if run_vx:
            assert video is not None and vx is not None
            # mock the body — just stamp profile names + lookups
            mark_profile("video self-attn adaln")
            mark_profile("video self-attn residual", vx)
            self._apply_text_cross_attention(
                vx, video.context, None, None, None, video.timesteps, None, None,
                profile_name=None, mark_profile=None,
            )
            ca_scale = getattr(self, "_cross_attn_scale", None)
            if ca_scale is not None:
                vx = vx * ca_scale
            mark_profile("video text-attn residual", vx)

        if run_ax:
            assert audio is not None and ax is not None
            mark_profile("audio self-attn adaln")
            mark_profile("audio self-attn residual", ax)
            self._apply_text_cross_attention(
                ax, audio.context, None, None, None, audio.timesteps, None, None,
                profile_name=None, mark_profile=None,
            )
            mark_profile("audio text-attn residual", ax)

        if run_a2v or run_v2a:
            mark_profile("av ca setup", vx, ax, None, None)
            if run_a2v and not skip_a2v:
                mark_profile("audio->video residual", vx)
            if run_v2a and not skip_v2a:
                mark_profile("video->audio residual", ax)

        if run_vx:
            mark_profile("video ff residual", vx)
        if run_ax:
            mark_profile("audio ff residual", ax)

        video_out = replace(video, x=vx) if video is not None else None
        audio_out = replace(audio, x=ax) if audio is not None else None
        return video_out, audio_out


def main() -> None:
    n_blocks = 48
    n_steps = 1000  # many fake "steps" for stable timing

    blocks = [FakeBlock(i) for i in range(n_blocks)]

    # Warmup
    for _ in range(50):
        v = FakeArgs(x=1, context=2, timesteps=3)
        a = FakeArgs(x=4, context=5, timesteps=6)
        for b in blocks:
            v_args, a_args = b(video=v, audio=a)
            v, a = v_args, a_args

    t0 = time.perf_counter()
    for step in range(n_steps):
        v = FakeArgs(x=1, context=2, timesteps=3)
        a = FakeArgs(x=4, context=5, timesteps=6)
        for b in blocks:
            v_args, a_args = b(video=v, audio=a)
            v, a = v_args, a_args
    elapsed = time.perf_counter() - t0

    per_step_ms = (elapsed / n_steps) * 1000
    per_block_us = per_step_ms / n_blocks * 1000

    print(f"Total: {elapsed:.3f}s for {n_steps} steps x {n_blocks} blocks")
    print(f"Per step: {per_step_ms:.2f} ms")
    print(f"Per block: {per_block_us:.1f} µs")
    print()
    print("Hypothesis: ~1.4 s/step gap at stage 1")
    print(f"Observed Python-only cost per step: {per_step_ms:.2f} ms")
    if per_step_ms < 100:
        print("=> Python overhead can't explain a 1400 ms gap.  Look elsewhere.")
    elif per_step_ms < 700:
        print(f"=> Python overhead is non-trivial but not the full gap.  Refactor"
              f" might recover {per_step_ms / 1400 * 100:.0f}% of it.")
    else:
        print("=> Python overhead alone could explain a large fraction of the gap.")


if __name__ == "__main__":
    main()
