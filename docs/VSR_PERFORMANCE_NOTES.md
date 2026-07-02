# VSR Harness Performance Notes

Findings, gotchas, and methodology from the 2026-07 optimization campaign over the
`videotoolbox/` processors (deblockers, denoisers, restorers, and learned upscalers
driven by `scripts/vsr_harness.py`). Everything here was measured on an M1 Max
(64 GB) with MLX, fp16-first, MLX buffer cache capped at 1 GB. Companion docs:
`PERFORMANCE.md` (diffusion pipeline), `PARITY.md` (port validation standards).

The single most important lesson: **"compute-bound" must be established by
kernel-path analysis, not FLOP counting.** Four separate "this net is at the
hardware ceiling" verdicts were overturned during this campaign by looking at
which Metal kernel each op actually dispatches to, and why.

---

## 1. The MLX conv dispatch gates (read this first)

`mlx/backend/metal/conv.cpp` routes every `mx.conv2d` call through a decision
tree. Ops that miss a gate fall to a *general* kernel that is silently 2-4x
slower. The gates, as of the MLX version in use:

| path | conditions | character |
| --- | --- | --- |
| depthwise | groups: C_per_group==1, O_per_group==1, C==O, C%16==0, k<=7, stride<=2 | fixed 8x8x4 tile, scalar MACs, no MMA |
| grouped implicit GEMM | (C_per_group<=4 or %16==0) and (O_per_group<=16 or %16==0) | fine for small group counts |
| winograd | 3x3, stride 1, no dilation/flip, C%32==0, O%32==0, **C+O>=256**, input >= 4096 px | fastest for mid-size 3x3 convs |
| implicit GEMM (specialized) | (C<=4 or C%16==0) and (O<=16 or O%16==0) | the good default |
| implicit GEMM (general) | everything else | the 2-4x-slower fallback |

Practical rules:

- **Audit every conv's (C, O) against these gates when porting a net.** A `%16`
  miss is invisible in the code and costs 2-4x. Found and fixed in this codebase:
  STDF `offset_mask` (32->189, its FLOP-heaviest conv, 1.9x), STDF `in_conv`
  (7->32, 2.0x), FastDVDnet `inc0` (30 outputs/group on the explicit-grouped
  fallback, 4.1x), SPyNet's first conv (8->32 7x7, 2.5x), BasicVSR++
  `conv_offset.0` (196->64, 1.45x), RealBasicVSR backbone `main.0` (67->64, 1.55x).
- **Audit runtime concat widths, not just weight files.** The last two misses
  above only exist at runtime (`concat(cond, flow1, flow2)` = 196 channels); a
  weights-only shape sweep cannot see them.
- **The fix is zero-padding, and it is exact.** Pad weight columns/filters with
  zeros to the next %16 boundary (and append matching zero channels to the input,
  or slice junk output channels off). Zero weights contribute nothing; only the
  kernel changes. Most of these fixes measured bit-exact end to end.
- **Padding is not automatically a win.** FastDVDnet `inc3` (90->32) on the
  general path *beat* the specialized path at 96->32 (thin-N inefficiency), and
  padding toward the winograd gate costs real FLOPs (zero-padded weights still
  multiply) -- 64->64 at C+O=128 cannot be pushed to 256 profitably. Measure the
  exact shapes before and after; never assume.

## 2. Kernel-shape pathologies (and their exact-math fixes)

### Depthwise conv: `mx.conv2d(groups=C)` collapses at many-channel/small-spatial

Measured 83x over the memory-bandwidth floor at 1024ch / 60x108 (the NAFNet deep
stage). The dedicated depthwise kernel uses a fixed 8x8x4 tile with a
threadgroup-memory halo and scalar 9-tap MACs -- no MMA (depthwise has no
K-contraction to feed the matrix units). At large-spatial/few-channel scales it
is only 1-7x over floor (acceptable).

**Fix:** a manual 9-tap shift-and-add (`sum over (i,j) of
xp[:, i:i+H, j:j+W, :] * w[:, i, j, 0]`) is 8.2x faster at the pathological
scale, 1.0x elsewhere, so it is safe to apply unconditionally. See
`videotoolbox/nafnet/net.py:_depthwise3x3`. Whole-net NAFNet: 1.37-1.44x.
Output shift ~55 dB PSNR (fp32 summation-order compounding through 36 residual
blocks; the op itself matches conv2d to 1e-6).

### `mx.fast.layer_norm` / `rms_norm` are transformer-shaped

One threadgroup per normalized row, sized to the normalized axis. For a conv
channel-norm (small C = 32-512, many N*H*W rows) the threadgroup underfills
(~8 of 32 lanes at C=64) across ~100k tiny threadgroups: **2.2-2.5x slower**
than a hand-rolled `mx.mean`/`mx.rsqrt` reduction. The penalty is shape-bound,
not dtype-bound (the kernel accumulates in fp32 regardless). Reserve
`mx.fast.*norm` for transformer-width axes; keep manual reductions for NHWC
channel norms (see `videotoolbox/nafnet/net.py:_layernorm`).

### Winograd collapses at large spatial extents

The same 64->256 3x3 conv runs at 10.2 TF/s-effective at 480x854 but 3.3 TF/s at
960x1708. This is why BasicVSR++'s `upsample2` (conv at 2x resolution) dominates
its reconstruction tail. Two exact reformulations were tested and **rejected**:
2x2 spatial tiling with halos (no recovery; the collapse tracks total working
set, not dispatch size) and a subpixel-conv rewrite of shuffle-then-conv as four
phase convs at LR (0.98x: the phase decomposition inherently carries 1.78x the
FLOPs, exactly canceling the better GEMM rate). The tail is at its practical
floor.

### Dense blocks (RRDB / DenseNet-style): restack weights by input segment

Conv-over-concat equals the sum of per-segment convs, and in a dense block every
conv's x / x1 / x2 ... segments convolve the *same* tensors. Restacking the
weight slices by produced tensor (done once at load, bit-identical values) turns
five thin concat-fed convs into one fat conv per produced tensor: the x-stack
(64 -> 224) crosses the winograd gate, and the incremental concats (4 full
copies per block x 69 blocks) disappear. RRDBNet (bsrgan / x4plus / esrgan /
realesrnet / bsrnet / anime / x2plus): **1.54x**, output parity 65.5 dB. The
recombination sums must run in fp32: the split rounds each partial conv output
to fp16 where the original GEMM accumulated all of K in fp32. See
`videotoolbox/realesrgan/net.py:_restack_rdb_weights` and `_rdb`.

### Window attention is intrinsically expensive on M1

ESC-Real's 32x32-window attention (405 windows x 4 heads x 1024 tokens x head_dim
16 at 480p) costs ~230 ms per layer and ~60% of each block, and BOTH formulations
land there: `mx.fast.scaled_dot_product_attention` with a dense additive mask at
head_dim 16 takes a slow path (234 ms), and the manual two-batched-GEMMs + precise
softmax is 228 ms -- the batched (1620 x) 1024x16x1024 GEMMs are thin-K/launch
bound, ~5x over the traffic floor. There is no exact-math fix; nets built around
many-token window attention are simply a quality tier on this hardware (papers'
efficiency claims usually assume CUDA flex-attention). Channel attention (MDTA:
C x C over tokens, as in RealViformer) is cheap by contrast -- the matrices are
tiny and the cost lives in the qkv convs.

### Deformable conv: follow the input dtype

The DCNv2 path (`videotoolbox/deform_conv.py`) originally forced fp32: three
cast-copies plus a `Cin*K*K x N*oH*oW` fp32 columns buffer (~1.9 GB at
128ch/480p) written by the im2col kernel and re-read by an fp32 GEMM -- 2x the
necessary traffic for data that only ever had fp16 precision. Running the whole
path in fp16 (sampling reads, columns, GEMM with MLX's internal fp32
accumulation; tap positions stay float) is **3.8x on the op**; BasicVSR++
1.19x whole-net (58.1 dB vs the fp32 path), STDF 1.12x (78.3 dB).

Rejected follow-up: reorganizing the im2col kernel from thread-per-(channel,
pixel) to thread-per-(group,pixel) to share sampling positions across the
group's channels measured *slower* (46 vs 42 ms) -- the kernel is
write-scatter-bound, not position-math-bound. It still sits ~9x over its write
floor; a genuinely better kernel would need a different output layout, which
the downstream GEMM constrains.

## 3. Things that do NOT work at video resolutions

All measured, all worth not re-litigating:

- **Frame batching** (K frames per forward for stateless per-frame nets): 1.00x
  at 480p and 0.98x for NAFNet. These conv nets are compute-bound at >=360p; a
  single frame already saturates the GPU. Batching only pays below ~240p
  (1.37-1.39x at 90x160) -- real video never lives there. Param count does not
  determine dispatch-boundedness; frame size does.
- **Cross-stage fusion** (compiling consecutive stateless stages as one graph):
  1.01-1.04x when stages are balanced; 0.98x for the realistic
  fbcnn+fastdvd+nafnet chain (a temporal stage isolates the stateless ones, and
  the fusable pair was dominated by one net). Barrier removal is noise next to
  conv compute.
- **SPyNet pair batching** (all flow pairs on the batch axis): 1.12x at 256x448
  but 0.92x at 480x854; the per-pair eval also serves as the documented
  memory-spike guard. Left sequential.
- **conv_transpose rewrites**: a 2x2-stride-2 convT is exactly a 1x1 conv +
  pixel-shuffle, but the only shape that wins (fbcnn's 512->256, 1.58x) is a
  0.8 ms op. Not worth the code.

## 4. dtype policy

- **fp16 by default** where provably safe -- fastest conv/GEMM dtype on M1.
- **fp32 islands only where the math demands it:** NAFNet's whole body
  (SimpleGate multiplies channel halves; magnitudes square past fp16's 65504 and
  the harness once silently wrote NaN frames), channel LayerNorm reductions, the
  RDB recombination sums above. bf16 is NOT a good compromise here: bf16 conv is
  slower than fp32 conv on M1.
- **Do not upcast fp16 data "for safety"** on memory-heavy paths -- it doubles
  traffic and adds nothing (the deform_conv lesson). Upcasting is for
  *accumulation and reductions*, not for storage.
- **Any fp16 reduction spanning the full spatial extent overflows.** A global
  average pool (`mx.mean` over H*W: ESC's dynamic-kernel predictor) or an L2
  normalization over H*W tokens (RealViformer's q/k normalize) sums ~400k fp16
  values at 480p -- far past 65504. These pass every small-input parity gate and
  only blow up on real-resolution frames; give every such reduction an fp32
  island. The channel-axis reductions (LayerNorm over 32-512 channels) are safe.
- Watch for fp32-constant contamination: `mx.zeros`/`mx.full`/`mx.arange`
  default to fp32; on an fp16 path pass `dtype=x.dtype` explicitly.
- fp32 summation-order noise compounds through deep residual stacks: a 1e-6
  per-op reorder became 2.6e-2 max after 36 NAFNet blocks. Judge end-to-end
  deviations in PSNR against the 8-bit encode floor (~48 dB), not per-op.

## 5. Graph mechanics (compile / eval / caches)

- `mx.compile` every shape-stable forward once, in a module-level cache. Gains:
  1.3-1.4x for dispatch-bound graphs (many small ops), ~1.05x for compute-bound
  ones. Every net here follows the `make_forward` + bounded-cache pattern.
- Compile caches must be **bounded** (`videotoolbox/compile_cache.py`, FIFO cap
  16): entries close over the checkpoint, so an unbounded id(p)-keyed dict
  retains every checkpoint ever constructed in the process. Eviction is safe --
  a params dict can only be collected (and its id recycled) after its entries
  are gone.
- `mx.eval` exactly once per output frame, at the point the frame is produced.
  The per-step evals inside recurrent propagation loops are **load-bearing** --
  they bound the lazy graph and the DCN column transients (removing them once
  produced a 57 GB OOM). The redundant ones (re-evaluating already-materialized
  outputs at a second layer) are free to delete but worth ~1%.
- One eval is also a sync barrier: do not add them inside a forward.

## 6. Pipeline glue (measured attribution)

Per frame at 854x480, 60-frame runs: decode + passthrough pack + HEVC encode =
**7 ms**; with 4x output the emit path costs ~20 ms (dominated by the ~50 MB
fp16-RGBA upload -- unavoidable at 4x). MLX<->CVPixelBuffer conversion for
preprocessing adds ~7 ms. Conclusion: the glue layer is healthy; the nets
dominate wherever they should.

**Gotcha: never derive per-frame cost from short runs.** A 5-frame smoke test
implied 76 ms/frame of "glue" that was actually one-time warmup (compile traces,
session setup) amortized over too few frames. Harness startup is ~4-5 s
(imports + pyobjc lazy-attribute initialization); it only matters for short
clips. Use 60+ frames for wall-clock numbers.

## 7. Reference numbers (854x480 input, M1 Max, post-campaign)

Per-frame processing cost, fp16, compiled, capped cache. Preprocessors:

| processor | ms/frame | notes |
| --- | --- | --- |
| stdf deblock | 53 | 1.25x cumulative (compile + gate pads + fp16 deform) |
| fastdvdnet denoise (steady state) | ~53 | 1.14x (grouped-conv gate pad) |
| fbcnn deblock | ~230 | flat U-Net; audited, nothing actionable |
| nafnet gopro (fp32) | ~340 | 1.4x (manual depthwise) |

The full upscaler ladder (all parity-gated against independent torch
reimplementations and visually verified on real motion-blurred video):

| ms/frame | --spatial-mode / weights | character |
| --- | --- | --- |
| 19 | safmn light | fidelity, trained on compressed content; the fast default |
| 148 | realesrgan general (SRVGG) | light perceptual |
| ~754 | safmn real2x (2x output) | real-world perceptual, the HD -> 4K class tool |
| 756 | safmn real (SAFMN_L_Real_LSDIR) | real-world perceptual, per-frame |
| 817 | realviformer (streaming) | real-world perceptual + TEMPORAL consistency |
| 976 | realbasicvsr | temporal GAN (cleaning + BasicVSR) |
| 1391 | basicvsrpp | temporal fidelity; 70% propagation / 20% tail |
| ~2400 | realesrgan bsrgan (RRDBNet) | heavy perceptual (1.5x from dense restack) |
| ~3900 | esc gan / mse | window-attention quality tier; mse = fidelity twin |

RealViformer streams (causal recurrence, per-frame state, reset at cuts) --
temporal consistency at barely more than a per-frame net's cost. ESC's numbers
are attention-bound (section 2); its value is output quality plus the only
fidelity/perceptual twin pair above the SRVGG class.

Peak MLX memory, one pass at 854x480 (1 GB cache cap): stdf 0.8 GB, fastdvd
1.0, general 1.2, fbcnn 1.9, nafnet-fp32 2.3, bsrgan 4.0, realbasicvsr-5-window
6.4, basicvsrpp-5-window 7.2 GB. Scales roughly linearly with window length and
pixel count; 1080p BasicVSR++ projects to ~25-30 GB.

## 8. Methodology for future performance work

Ordered; stop at the first step that explains the time.

1. **Attribute wall-clock first.** 60+ frame run, passthrough config as the
   baseline, cProfile for the breakdown. Do not optimize a stage before knowing
   its share.
2. **Phase-profile the net.** Time its stages with per-phase evals (see the
   BasicVSR++ split above). Estimates from FLOP counts routinely miss by 3x.
3. **Micro-bench suspect ops against both rooflines.** FLOPs / peak-TF/s for
   compute; bytes / ~400 GB/s for bandwidth. An op far from both floors is on
   the wrong kernel.
4. **Kernel-path analysis.** Classify every conv (C, O, kernel, stride, groups)
   AND every runtime concat width against the dispatch gates in
   `mlx/backend/metal/conv.cpp`. Check `mx.fast.*` kernels' shape assumptions
   against the actual tensor shapes (threadgroup-per-row vs many-small-rows).
5. **Apply exact-math transformations only**, in this order of preference:
   gate padding (zero filters/columns; usually bit-exact), weight restacking
   (dense blocks; load-time reordering), manual formulations for pathological
   kernels (depthwise shift-add, hand-rolled channel norms), dtype-following
   (fp16 through memory-heavy paths, fp32 for reductions).
6. **Gate every change on parity.** Bit-exact when the transformation allows
   it; otherwise report max|d| and PSNR vs the previous path and accept only
   deviations far below the 8-bit encode floor. Full-net A/B with compiled
   forwards, capped cache, realistic resolution, then a harness smoke test.
7. **Record rejections with their mechanism.** A rejected idea without the
   measured "why" gets re-attempted. This document and the git history are the
   ledger; the failed ideas in section 3 cost as much to establish as the wins.
8. **When the well is dry, change the math, not the schedule.** After
   exact-math options are exhausted, further speed means lighter architectures
   (smaller trained variants such as nafnet width32; different nets such as the
   SAFMN family) -- i.e., accepting different output. Quantization is NOT a
   lever for these conv nets: MLX's quantized kernels are matmul/LLM-shaped and
   the nets are compute-bound in fp16, not bandwidth-bound.

## 9. Porting new models: selection, validation, drift traps

Lessons from porting the SAFMN family, ESC-Real, and RealViformer (and from
rejecting SMFANet). These cost real debugging time; follow them in order.

### Pick the checkpoint by its TRAINING DEGRADATION, not its name or awards

- A model is only as video-appropriate as the degradations it trained on. The
  Real-ESRGAN high-order pipeline (blur + noise + compression) transfers to real
  video; bicubic-benchmark training (DF2K/DIV2K classic SR) does not, and
  challenge checkpoints are tuned to the challenge's exact degradation and
  metrics. The AIM 2025 "SAFMN-L" (stills, synthetic degradation, no-reference
  perceptual metrics) hallucinates crusty texture over motion blur on real video
  -- verified to be the network, not the port -- while SAFMN_L_Real_LSDIR
  (Real-ESRGAN pipeline, blur included) is clean on the same frames.
- Watch for name collisions: three different checkpoints all named "SAFMN-L"
  exist (DF2K benchmark, AIM challenge, Real_LSDIR). Identify by tensor shapes
  and training provenance, not filename.
- If a project publishes ONLY bicubic-benchmark checkpoints (SMFANet), it has no
  video-appropriate model to port. Skip it.
- Release pages mix architectures: the ESC release ships the author's retrained
  comparison baselines (ATD/HiTSRF/SRFormer) and torch-only variants (FlashBias)
  alongside the real models.

### Validation protocol (all four gates, in order)

1. **Parity vs an independently written torch reimplementation.** Reference code
   is a written spec only -- never imported or executed, not even for parity.
   Reimplement the reference in torch with matching state_dict keys (declare
   vestigial checkpoint params as unused so strict=True verifies everything
   else), load the checkpoint with weights_only=True, and gate the MLX port
   against that. Expect 110-130 dB at fp32 on small inputs; test both the
   size-aligned and the padding code paths.
2. **fp16 on REAL frames at REAL resolution.** The fp32 small-input gate cannot
   see fp16 range problems; the full-spatial-reduction overflows (section 4)
   only appear here. Check `mx.all(mx.isfinite(...))` on actual video frames.
3. **Eyeball output frames on real video before shipping.** "Runs, finite, high
   parity on random input" catches neither a wrong checkpoint nor an
   out-of-distribution failure. Include a motion-blurred frame -- it is the
   degradation real video always has and synthetic training pipelines often
   lack, and it is what exposed the AIM checkpoint.
4. **Harness smoke end to end** (geometry, encoder, cut/reset behavior).

### Expect checkpoint-vs-source drift

Released weights routinely disagree with the repo's current code; the weights
win. Cases hit: ESC-Real's upsample tail is Upsample-first in the checkpoint
(params at Sequential indices 1/4/6/8) but conv-first in today's source;
RealViformer's checkpoint carries a vestigial attention parameter the reference
inference silently skips with strict=False; light_SAFMN++ ships ffn_scale 1.5
against the paper's 2.0. Derive EVERYTHING derivable from tensor shapes
(widths, block counts, expansion ratios, scale) and read the key layout before
writing the forward.

### Structure notes

- Causal/unidirectional recurrent nets (RealViformer) want a STREAMING driver:
  per-frame feed with carried state and reset() at cuts -- no window buffering,
  no trim. Bidirectional nets (BasicVSR++ class) need the windowed driver.
- U-Nets constrain input divisibility (2 downs -> /4; deepest pool -> /8;
  window attention -> /32 handled internally). Replicate-pad at the driver or
  net entry and crop the scaled output; document which the reference does --
  torch adaptive pooling at non-multiple sizes has different semantics than
  pad-then-uniform-pool (invisible in practice, but it moves parity numbers).
- Reuse the shared blocks: a checkpoint's SpyNet is usually BasicSR's or
  mmagic's -- semantically identical to `vsr_blocks.spynet_flow`; remap key
  names at load and the compiled + gate-padded implementation comes free.
- Fractional FFN expansion factors produce gate-missing widths: RealViformer's
  2.66 makes hidden dims 127/255/510, putting all 75 GDFN convs on the general
  path. The per-half zero-pad (project_in/dwconv rows become [first_h, zeros,
  second_h, zeros] so the gate-multiply chunk stays aligned; project_out gets
  zero columns) is bit-exact -- but measure the WHOLE net: the per-conv ~1.9x
  amounted to 1.02x end-to-end here (the compiled net is dominated elsewhere).
  Post-pass verdicts on the other two: SAFMN-real is healthy (its big CCM conv
  rides winograd at 15 TF/s-eff; all widths aligned); ESC's 13x13 partial conv
  runs at 2.9 TF/s-eff but output-padding it wider is a wash (measured), and
  the precise softmax costs nothing -- keep it.

## 10. Benchmarking gotchas checklist

- Cap the MLX buffer cache (`mx.set_cache_limit(1 GB)`) -- an uncapped cache
  contaminates both speed and peak-memory numbers.
- Fresh process per configuration for headline numbers; at minimum
  `mx.reset_peak_memory()` and separate compile caches between configs.
- Warm up before timing (compile traces retrace per input shape).
- 60+ frames for anything reported as per-frame cost (see section 6).
- Serial GPU work only -- concurrent MLX benchmark processes contend and can
  hang the GPU (M1 compute hangs are non-preemptible; recovery is a reboot).
  Write logs/sidecars to `$SHARED_TEMP_DIR` before running risky kernels.
- `mx.eval` the output inside the timed region, once.
- Isolated-op wins must be re-measured end to end: compile fusion, memory
  pressure, and phase overlap change the arithmetic (several 2x op wins landed
  as 1.1-1.2x whole-net; one 1.6x op win was a 0.3% whole-net no-op).
