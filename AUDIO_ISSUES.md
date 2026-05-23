# LTX-2.3 MLX Audio Issues — Historical Investigation Log

**Date:** April 5, 2026  
**Status:** Historical notes. The current `main`/dtype-cleanup path has moved beyond this state.
**Reference:** ComfyUI (PyTorch+MPS) produces clear speech with same model/prompts

> **Current note (May 2026):** This file is retained as a debugging record, not
> as the current audio status. Recent LTX-2.3 AudioVideo smoke tests produce
> clear audio, and the old echo artifact was traced to audio latent positions
> being downcast. Those positions intentionally stay float32. Audio VAE decode
> now follows the requested compute dtype; the LTX-2.3 Vocoder+BWE path keeps a
> scoped float32 island, matching the Lightricks BWE precision caution.

---

## Symptom

- Speech audio is ~40% garbled across all pipelines (distilled, two-stage, i2v)
- Quality is best in first 2-3 seconds, degrades by 7-8 seconds
- Video quality is fine — only audio/speech affected
- Short prompts (20-30 tokens) work better than long prompts (200+ tokens)
- Lip sync tracks correctly despite garbled speech
- Ambient/environmental audio generates fine — speech specifically is weak

## Root Cause (Confirmed)

**The MLX diffusion transformer generates audio latents where speech is underrepresented relative to ambient sounds.**

Evidence chain:
1. Exported ComfyUI text embeddings → fed to MLX pipeline → still no clear speech (rules out text encoder)
2. Saved MLX mel spectrogram → fed to PyTorch vocoder → speech very quiet, mostly ambient (rules out MLX vocoder)
3. Therefore: the 48-layer MLX transformer's audio cross-attention produces weaker speech content than PyTorch

This is a numerical precision divergence across 48 transformer layers. Audio speech generation requires higher fidelity conditioning than video or ambient audio.

---

## Issues Found & Fixed

### 1. Gemma Per-Layer RoPE (FIXED — Backup 0577)
- **Problem:** Single RoPE config (theta=1M, scaling=8.0) applied to all 48 Gemma layers
- **Fix:** Two configs — 40 sliding layers (theta=10k, no scaling, 1024 window) + 8 full layers (theta=1M, scaling=8.0)
- **Impact:** Cosine similarity went from 0.05 to 0.934 at final layer. First working speech.
- **File:** `LTX_2_MLX/model/text_encoder/gemma3.py`

### 2. Gemma Boolean Attention Masks (FIXED — Backup 0577)
- **Problem:** Additive float masks caused NaN for all-padded rows
- **Fix:** Switched to boolean masks matching HuggingFace behavior
- **File:** `LTX_2_MLX/model/text_encoder/gemma3.py`

### 3. Connector Register Handling (FIXED — Backup 0578, not in 0577)
- **Problem:** MLX connector replaced padding within sequence (256 positions) instead of appending registers to extend to 1024
- **Fix:** Changed `_replace_padded_with_learnable_registers` → `_append_learnable_registers` matching ComfyUI behavior
- **Impact:** Correct sequence length but didn't fix speech quality
- **File:** `LTX_2_MLX/model/text_encoder/connector.py`

### 4. Double Precision RoPE for Connector (FIXED — Backup 0578, not in 0577)
- **Problem:** Connector used float32 for frequency grid; checkpoint specifies `frequencies_precision: float64`
- **Fix:** Added `double_precision_rope` flag, reads from checkpoint metadata
- **Impact:** Matches ComfyUI behavior but didn't fix speech quality
- **Files:** `connector.py`, `encoder.py`, `rope.py`

### 5. Padding Trim (FIXED — Backup 0578, not in 0577)
- **Problem:** Padding trimmed to next multiple of 128 instead of exact real token count
- **Fix:** Trim to exact real token count (matching ComfyUI)
- **Impact:** No audio impact
- **File:** `scripts/generate.py`

---

## Things Tried That Didn't Help

### 6. Gemma Model Swap (QAT → Standard)
- **Tried:** Replaced `gemma-3-12b-it-qat-q4_0-unquantized` with standard `gemma-3-12b-it` (bf16)
- **Result:** Audio slightly better, video slightly worse. Neither model matches ComfyUI exactly.
- **Note:** Standard model downloaded to `/Users/steveross/Documents/Development/Source Models/gemma-3-12b-it`
- ComfyUI uses `gemma_3_12B_it_fp4_mixed.safetensors` (standard model in fp4 quantization)

### 7. ComfyUI Embedding Injection (Definitive Test)
- **Tried:** Exported text embeddings from ComfyUI's `preprocess_text_embeds`, loaded into MLX pipeline
- **Result:** Ambient audio present, NO speech. Proves issue is in MLX diffusion transformer, not text encoder.
- **Tools:** `save_embeddings_hook.py`, `_load_comfyui_embeddings()` in app.py

### 8. PyTorch Vocoder on MLX Mel (Definitive Test)
- **Tried:** Saved mel spectrogram from MLX VAE decoder, ran through ComfyUI's PyTorch vocoder
- **Result:** Speech very quiet, mostly ambient. Proves mel spectrogram itself has weak speech content.
- **Conclusion:** Issue is upstream of vocoder — in the diffusion transformer's audio latent generation.

---

## Remaining Differences (Not Yet Investigated)

### Transformer-Level
- 48-layer transformer numerical divergence between MLX (Metal) and PyTorch (MPS)
- Audio cross-attention may weight text tokens differently due to floating-point arithmetic differences
- AdaLN modulation (shift/scale/gate) may accumulate differently across layers
- Audio-video cross-modal attention could have subtle differences

### Connector-Level (Backup 0577 state)
- Connector uses replace-and-sort behavior (old), not append (ComfyUI behavior)
- The append fix (backup 0578) was correct but didn't help because the bottleneck is the transformer
- `connector_positional_embedding_max_pos: [4096]` from checkpoint metadata — verify MLX reads this
- RMSNorm epsilon: ComfyUI uses 1e-5, MLX uses 1e-6 in connector attention

### Audio Token Count Off-by-One (FOUND — round vs ceil)
- ComfyUI produces 502 audio tokens, MLX produces 501 for same duration (481 frames)
- Cause: `round()` in MLX types.py vs `math.ceil()` in ComfyUI audio_vae.py
- Changing to ceil() made audio WORSE (voices super quiet) — reverted
- The model may have been trained with round() behavior, or the off-by-one matters less than expected

### Layer-by-Layer Comparison Results (Same Seed)
- Layer 0: 0.135 cosine — initial audio states already differ (noise + positions)
- Layers 1-47: gradual convergence from 0.05 to 0.945 — transformer blocks work correctly
- Final latent: 0.186 cosine — output projection diverges again
- MLX std consistently lower (0.56 vs 0.79) — MLX output is compressed/muted
- This pattern suggests the issue is in INPUTS (noise, positions) not transformer blocks

### av_ca_timestep_scale_multiplier Bug (FIXED)
- **`av_ca_timestep_scale_multiplier` was 1 instead of 1000** (checkpoint metadata value)
- This made the audio-video cross-attention gate factor 0.001 instead of 1.0 — effectively zeroing cross-modal gates
- Cross-modal attention carries speech information (lip sync → audio), so speech was weak while ambient was fine
- **Fix**: Added `av_ca_timestep_scale_multiplier=1000` to `load_av_transformer` in generate.py
- **Result**: 5-second clips now nearly perfect speech. Significant improvement.

### GPU Watchdog Crash on Audio Decode (FIXED)
- Added mx.eval() inside AMPBlock1 after each dilation iteration
- Added mx.clear_cache() between vocoder stages and BWE stages
- Added mx.eval() after STFT conv1d and conv_post
- 10-second clips now complete without kernel panic

### Duration-Dependent Amplitude Bug (RESOLVED — May 21, 2026)

**Original symptom (April 2026):**
- 5-second clip: RMS 5535, peak 31977 — loud, healthy audio, near-perfect speech
- 10-second clip: RMS 1137, peak 9899 — **5x quieter**, speech mumbles from second 0
- Audio latent amplitude appeared to scale inversely with duration
- Video quality unaffected, only audio

**Workaround applied at the time** (commit `3ce9ab7`, April 13 2026):
`OneStageAVPipeline._channelwise_normalize_audio_noise` was added and called
unconditionally at stage-1 audio init, whitening the per-channel statistics
of the audio noise tensor so that long clips wouldn't suffer the amplitude
collapse.  This made the audio init *not match* the distribution the LTX-2
audio LoRA was trained on (un-normalized `N(0,1) * sigma_max`, per upstream
Lightricks `ltx-pipelines/utils/blocks.py`).

**Empirical re-test on May 21, 2026** confirms the underlying bug is no
longer present and the workaround is no longer needed:

| Mode                                | 20-second clip RMS | Peak | dBFS RMS | Speech burst RMS |
|-------------------------------------|--------------------|------|----------|------------------|
| `LTX_NORMALIZE_AUDIO_NOISE=0` (new default) | 588        | 10687 | -34.92  | ~1000-1600       |
| `LTX_NORMALIZE_AUDIO_NOISE=1` (legacy)      | 608        | 11775 | -34.63  | ~800-1400        |
| Historical-bug 10s clip (April)             | 1137       | 9899  | —       | mumble level     |

Both modes produce healthy 20-second audio with proper dialog levels, no
"5× quieter from second 0" pattern.  The per-second RMS trajectories show
normal silence floors (40-80 RMS) and proper speech bursts (>1000 RMS)
throughout the clip in both modes — no inverse-duration amplitude scaling.

**Why the bug is gone:** several fixes landed *after* the workaround that
appear to have removed the original root cause without anyone re-testing:
- Denoise mask broadcasting `(B,T) → (B,T,1)` (commit `3ce9ab7`, same commit
  as the workaround — but the broadcast fix likely resolved the latent
  amplitude scaling issue on its own).
- Audio VAE decoder corrections (causal axis drop, PixelNorm, weight paths)
  bringing audio VAE to 0.999 correlation with PyTorch.
- Vocoder LeakyReLU slope fix (0.1 → 0.01) and native MLX dilation.
- `av_ca_timestep_scale_multiplier` set to 1000 (was 1) — fixed audio-video
  cross-attention gate factor.

**Resolution in code** (commit landing May 21, 2026):
- Channelwise normalization gated behind `LTX_NORMALIZE_AUDIO_NOISE` env var.
- **Default: OFF.**  Now matches upstream Lightricks input distribution.
- Set `LTX_NORMALIZE_AUDIO_NOISE=1` to restore legacy MLX behavior for A/B.
- `env_flags` field in run-log sidecar records which mode produced each
  output, so post-hoc forensics can identify the regime.

**Why this matters for cross-modal output**, not just audio: the V2.3 AV
transformer has `a2v` cross-attention layers — changing the audio noise
init at step 0 propagates through to the video branch.  Observed effect
on a 20-second bakery prompt at seed=124: character appearance, ambient
prop layout, and speech tone all shifted coherently between the two
modes.  Default OFF puts us on the same attractor as a stock Lightricks
run with the same prompt+seed.

**DO NOT RE-ADD the unconditional normalization without re-validating
that the duration-amplitude symptom has returned.**  If you suspect it
has, regenerate a 20-second clip with `LTX_NORMALIZE_AUDIO_NOISE=0` and
compare per-second RMS against the table above before changing the
default.

### Temporal Degradation Pattern (RESOLVED — May 21, 2026)
Historical observation (April 2026):
- 5 second clip: nearly perfect voice (with 1000x gate fix)
- 10 second clip: overall 5x quieter + mumbles toward end
- 20 second clip: degradation from 7-8 seconds onward

Re-tested May 21, 2026 with a 20-second bakery generation
(`LTX_NORMALIZE_AUDIO_NOISE=0`, distilled two-stage, seed=124): no
amplitude degradation across the clip duration.  Dialog seconds 8-15
produce normal RMS bursts (~1000-1600), silence floors are clean
(~40-80 RMS).  Cumulative effect of the gate fix, denoise broadcast
fix, audio VAE/vocoder parity fixes, and timestep scale multiplier
fix appears to have resolved the original temporal degradation along
with the amplitude collapse.  See the Duration-Dependent Amplitude
Bug closure note above for the empirical table.

### Sequence-Start Audio Spike (OPEN — May 22, 2026)

**Symptom:** *Some* generated clips produce a loud burst of ~65 ms of
near-clipping audio at the very start, immediately followed by clean
silence until the first spoken word.  Audible as a sharp "click" or
"blat" at t=0 on playback.  **Not all clips exhibit this** — see the
ambient-onset counter-example below.  Reproduced on a 30-second
distilled two-stage AV smoke (576x320, dialog-heavy prompt, seed=42,
generate-audio).

**Decoded WAV characterization** (48 kHz stereo float, ALAC):

| Window (5 ms)   | RMS    | abs-peak | Notes                       |
|----------------:|-------:|---------:|-----------------------------|
| 0-5 ms          | 0.387  | 0.887    | Onset, full amplitude       |
| 5-65 ms         | 0.25-0.37 | up to 0.91 | Sustained loud content   |
| 65-70 ms        | 0.167  | 0.500    | Sharp decay                 |
| 70-95 ms        | 0.002-0.024 | <0.06 | Decay tail                |
| 95-250 ms       | <0.001 | <0.001   | Pure silence (intentional)  |
| 250+ ms         | ~0.20  | normal   | Speech starts               |

Global clip RMS is 0.120, so the first 5 ms is **3.2x global** and
sustains above 2x for ~50 ms.  Peak sample sits at t=60.2 ms with
amplitude -0.9102 — near-clipping.

**Latent characterization** (final_audio_latent, 25 fps_lat = 1
frame per 40 ms):

| lat_t | Time     | RMS   | max-abs | vs. median |
|------:|---------:|------:|--------:|-----------:|
| 0     | 0-40 ms  | 1.475 | 5.34    | 1.31×      |
| 1     | 40-80 ms | 2.042 | 3.88    | **1.81×**  |
| 2     | 80-120 ms| 1.787 | 3.42    | 1.59×      |
| 3+    | 120+ ms  | ≤0.93 | ≤2.98   | settled    |
| median| —        | 1.127 | —       | 1.00×      |

The first three latent frames carry 1.3-1.8× typical energy, peaking
at lat_t=1.  After lat_t=3 the latent energy is normal.

**The spike is already present in stage 1**, before the spatial
upscaler runs:

```
stage_1_audio_latent  lat_t=0  rms=1.346  max=6.75   <- spike present pre-upscaler
stage_2_audio_latent  lat_t=0  rms=1.475  max=5.34   <- refinement reshuffles
```

Stage 2 refinement slightly lowers the per-sample peak and slightly
raises the RMS — energy moves around within the frame but the spike
persists.  This rules out the spatial-upscaler refinement step as
the source.

**Counter-example (20-second ambient-onset clip, same model + seed
range):**

| Metric                       | Dialog-heavy (clicks) | Ambient-onset (no click) |
|------------------------------|----------------------:|-------------------------:|
| WAV global RMS               | 0.120                 | 0.018                    |
| WAV 0-50 ms RMS              | 0.310 (2.58x ↑)       | 0.0015 (0.08x ↓)         |
| final lat_t=0 RMS            | 1.475                 | **2.120**                |
| final lat_t=0 max-abs        | 5.34                  | **6.72**                 |
| stage_1 lat_t=0 max-abs      | 6.75                  | 7.00                     |

The ambient-onset clip's latent at t=0 is **more** elevated than the
dialog-heavy one's (RMS +44 %, max-abs +26 %), yet its WAV at t=0 is
**quieter than its own global RMS**.  So elevated lat_t=0 RMS alone
does not predict an audible click.

**Diagnosis (revised):** The *elevated lat_t=0* appears to be a
universal sequence-start sink — the transformer reliably encodes
something with high L2 norm into the first ~3 latent frames across
prompts and seeds.  That is consistent with classic attention-sink
behavior (no left-context, attention defaults to fixed sink
positions).  Whether that elevated latent becomes an *audible*
spike depends on the **spectral content** the model places there:

- *Dialog-heavy prompts* (where the model wants loud content at t=0):
  the model encodes loud transient onsets with broadband spectral
  content, which BWE (24 → 48 kHz bandwidth extension) faithfully
  synthesizes into a wide-spectrum click;
- *Ambient-onset prompts* (where the model wants quiet content at
  t=0): the model encodes quiet room tone — high latent RMS but
  DC-ish / low-frequency content — which decodes to faithful silence.

So the AV transformer's high-energy lat_t=0 is **not the bug** by
itself; the click is a *content-dependent* manifestation of it on
clips whose intended t=0 is loud.  This is **not** caused by:

- the VideoToolbox encode path (spike is in the latent, before any
  encoder runs);
- AVAssetWriter ALAC mux (ALAC is lossless);
- `LTX_NORMALIZE_AUDIO_NOISE` (off in the dialog-heavy test that
  exhibited the spike, also off in the ambient-onset test that
  didn't).

**What would actually predict a click?**  An open question.  Probably
the latent's *spectral / dynamic-range* content at t=0, not just its
RMS.  A useful follow-up would be to extract per-channel-of-8
statistics for lat_t=0 across clips that click vs. clips that don't,
and look for the discriminating feature.  The analyzer script's
verdict is therefore a *necessary but not sufficient* indicator —
the current 50 ms-RMS threshold catches the audible cases but says
nothing about why some quiet-onset clips have elevated latents too.

**Reproduction inputs:**
- Output: any `.{mp4,wav,npz}` sidecar bundle from a generate.py run
  with `--save-all-sidecars` (the run log JSON is preserved alongside).
- Generator: `--pipeline distilled --height 320 --width 576
  --duration 30 --generate-audio --fast-mode --save-all-sidecars`
  on a dialog-heavy prompt at any seed; the spike reproduces broadly,
  not just for one seed/prompt combo.
- Analysis: `scripts/analyze_audio_onset.py --run <stem>.mp4` (or
  `.wav` / `.npz`).  Prints WAV head profile (coarse 50 ms windows
  over the first 2 s, fine 5 ms windows over the first 120 ms),
  per-frame latent stats for every audio-latent key present, and a
  one-line VERDICT (spike / clean) using a configurable threshold.
  See the script's docstring for the full design notes.

**Possible mitigations** (in order of intrusiveness, none implemented).
The dialog-vs-ambient comparison narrows the design: clips that don't
click shouldn't be touched, so the production-grade fix is
**detect-then-trim** rather than unconditional surgery.

1. **Detect-then-trim** (recommended). After audio decode, run the
   same 50 ms-RMS-vs-global heuristic the analyzer script uses; if
   the first window exceeds the threshold, drop the leading 100-150
   ms of WAV before mux.  Clean clips pass through unchanged.  Trim
   length 100-150 ms is sized to clear the click decay tail (~95 ms
   on the diagnosed clip) plus a margin; it sits inside the
   intentional silence the model places before the first spoken
   word (95-250 ms on the same clip), so the cushion costs nothing.
   Natural CLI: `--audio-onset-trim {auto, off, N_ms}`, default
   `auto`.  Must apply in both the new VT encode helper and the
   ffmpeg encode helper, and to the sidecar WAV writer.
2. **Linear / cosine fade-in** over 100-150 ms with the same
   detect gate.  Same protection, slightly softer if the spike ever
   overlaps real audio (the diagnosed clip has clean silence between
   the burst and the first word, so trim and fade are indistinguishable
   there — fade matters only if future clips have legitimate content
   right at t=0).
3. **Zero the first 2-3 audio-latent frames** before vocoder decode.
   Universal, no gating needed: the elevated lat_t=0 appears on both
   clicking and non-clicking clips, and the non-clicking clip already
   decodes those high-RMS latent frames to silence, so dropping them
   should be ~imperceptible there.  Caveat: the vocoder has temporal
   context spread, so neighbor latent frames still feed forward into
   the vocoder's left edge — a fully clean t=0 isn't guaranteed.
   Useful as an A/B against (1) to confirm whether the residual
   energy comes from the latent or from the vocoder's left-edge
   behavior on its own.
4. ~~**Notch / subtract a stable spectrum.**~~  **Demoted.**  The
   dialog-vs-ambient counter-example shows the spike is content-
   dependent — there's no single stable spectrum across clips to
   notch.  Not worth pursuing.

Whichever path is taken, the audio sidecar (`--save-audio-sidecar`)
and the muxed track should both get the cleaned version so post-hoc
listening doesn't rediscover the click.  Trim/fade should be applied
**after** any latent save (`--save-latents` keeps the raw latent so
the analyzer-script analysis stays reproducible).  The detect-then-
trim heuristic can reuse the analyzer's threshold logic so a single
constant defines both the diagnosis criterion and the mitigation
gate.

**AV sync safety** (do not get this wrong):

- The trim mitigation must **mute / zero-fill** the leading audio
  samples, not **drop** them.  Audio length must continue to match
  video length, otherwise every video frame after the trim point
  slips by the trim duration — 100 ms of drop = 2.4 video frames at
  24 fps, very audible on speech.  Equivalently: keep the audio
  track PTS at 0 and write zeros over the spike region.
- The fade-in mitigation preserves sample count and is sync-safe
  trivially.
- The latent-zero mitigation preserves vocoder output length (output
  length is determined by latent T, not by latent magnitude) and is
  sync-safe trivially.  Only caveat is vocoder temporal-spread bleed
  near the left edge — audible artifact possible but no sync drift.
- The detect gate is *also* a sync-safety feature: a clip with
  legitimate speech from t=0 would show normal speech RMS at the
  head (~0.2x global), not the click's ~2.5x global, so the trim
  wouldn't fire and lip sync at t=0 would be preserved.  Worth
  refining the heuristic before implementing — see "more robust
  shape detector" note below.

**More robust shape detector** (for future implementation): the
click signature is specifically *loud burst followed by silence*,
not just *loud burst*.  A two-window check —

```
window_0_rms > 2.0x global AND mean(windows in 100-250 ms) < 0.1x global
```

— would catch the diagnosed click signature but reject any future
clip where the loud t=0 is legitimate speech onset (which wouldn't
be followed by silence).  The current single-window threshold in
`scripts/analyze_audio_onset.py` is fine for diagnosis but the
mitigation gate should use the two-window variant.

### Potential Deep-Dive Areas
- Compare audio latents (pre-VAE-decode) between MLX and ComfyUI for same seed/prompt
- Layer-by-layer transformer output comparison (heavy instrumentation needed)
- Check if audio CFG/guidance scale is applied identically
- Verify MultiModalGuider computes modality_scale correctly for audio
- Check if audio self-attention RoPE positions match ComfyUI exactly

---

## Architecture Quick Reference

### Audio Decode Pipeline
```
Transformer output (B, T, 128) patchified
  → Unpatchify: (B, 8, T, 16)
  → Denormalize: per-channel stats (mean/std from checkpoint)
  → VAE Decoder: 2D convolutions, 3 upsample levels → (B, 2, T*4, 64) mel
  → Vocoder (BigVGAN v2): 108+ 1D convolutions, 5 upsample stages → waveform
  → BWE: mel recompute → second vocoder → resample → residual add
  → Output: (B, 2, samples) at 24kHz stereo
```

### Key Files
- Transformer: `LTX_2_MLX/model/transformer/model.py`, `transformer.py`
- Audio cross-attention: `transformer.py` lines 545-553
- Audio VAE decoder: `LTX_2_MLX/model/audio_vae/decoder.py`
- Vocoder: `LTX_2_MLX/model/audio_vae/vocoder.py`
- Audio patchifier: `LTX_2_MLX/components/patchifiers.py` (AudioPatchifier)
- Gemma text encoder: `LTX_2_MLX/model/text_encoder/gemma3.py`
- Connector: `LTX_2_MLX/model/text_encoder/connector.py`

### ComfyUI Reference Files
- AV model: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/av_model.py`
- Text encoder: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/text_encoders/lt.py`
- Connector: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/embeddings_connector.py`
- Audio VAE: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/vae/audio_vae.py`
- Vocoder: `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/comfy/ldm/lightricks/vocoders/vocoder.py`

### Test Tools Created
- `save_embeddings_hook.py` — Patches ComfyUI to export text embeddings
- `test_vocoder_from_mel.py` — Feeds MLX mel through PyTorch vocoder
- `scripts/analyze_audio_onset.py` — Detects + characterizes start-of-clip
  audio artifacts (e.g. the sequence-start spike).  Takes any run sidecar
  path (`.mp4` / `.wav` / `.npz`), prints WAV head RMS profiles at both
  coarse (50 ms) and fine (5 ms) resolutions, per-frame latent stats for
  every audio-latent key in the NPZ, and a VERDICT line.  `--strict`
  returns non-zero on spike detection (sweep / CI usable).
- Exported embeddings: `/Users/steveross/Documents/ComfyUI/exported_embeddings/`
