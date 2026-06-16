# Test prompts and recipes

Canonical prompts + recipes for perf benchmarks and quality
regressions.  Every recipe includes `--save-all-sidecars` so a full
set of sidecars (latents .npz, text-conditioning .npz, run-log .json)
lands next to the .mp4 — useful for cosine-sim diffing latents,
re-running just stage 2 from saved stage-1 latents, replaying a run
with the exact same text conditioning, or auditing timing/metadata
without re-encoding.

## Layout

- **Prompts** below set themselves as shell variables (`BAKERY`,
  `KITTEN`).  Copy that block once at the top of your session.
- **Recipes** in the next section reference the variables, so the
  prompt text isn't repeated for every variant.
- Cite results by **prompt name + recipe name + seed** so other runs
  are comparable.

---

# Prompts

## `BAKERY` — 20 s two-speaker dialogue

Quiet neighborhood bakery, slow handheld camera, two speakers, sparse
ambient.  Exercises lip-sync + close dialogue audio.  Canonical seed
is `124`.  Reference duration is **20 s @ 24 fps = 481 frames**.

Set once per shell session:

```bash
BAKERY='Live-action footage shot on a consumer mirrorless camera inside a quiet neighborhood bakery in soft morning light. A flour-dusted wooden table fills the foreground. On the left, a young baker in a blue apron kneads dough slowly. On the right, an older baker places four golden croissants onto a metal tray. The camera begins close on the dough, slowly pulls back to reveal both bakers, then gently pans right toward the croissants. The young baker looks at the tray and says, "Four for the window, two for the regulars." The older baker smiles and replies, "Save the best one for Mrs. Ito." Their voices are close, dry, natural, and clearly recorded. The room is very quiet, with a soft low indoor hush. The dialogue is the main sound. The background stays minimal and smooth. The frame contains only the bakery, the bakers, the table, the tray, and warm morning light. Realistic lip movement, handheld camera with gentle natural sway, warm shadows, detailed textures.'
```

## `KITTEN` — 30 s monologue with three beats

Single speaker (kitten), static indoor framing, longer duration.
Stresses scaling at higher token counts.  Canonical seed is `42`.
Reference duration is **30 s @ 24 fps = 721 frames**.

Set once per shell session:

```bash
KITTEN='Live-action footage shot on a consumer camera, cozy indoor setting, shallow depth of field. A small marble black kitten with swirling black and dark grey fur, bright yellow-green eyes, tiny pink nose, and oversized ears sits on a living room couch. Warm afternoon light filters through nearby curtains. The kitten looks directly into the camera and says, "I have been awake for nineteen hours. Do I sleep? No. Do I rest? No. I sit here and I think about the vacuum cleaner. It was loud yesterday. It will be loud again. I don'\''t know when. That'\''s the thing. I never know when." The kitten glances to the left, then snaps back to the camera and says, "You probably think I'\''m being dramatic. The vacuum cleaner is literally called a vacuum. It creates a void. A void, in my home. And everyone just acts like that'\''s fine." The kitten licks its paw once, stares blankly for a moment, then says, "Anyway I'\''m going to knock something off the counter later. I haven'\''t decided what yet. I'\''m keeping my options open." Natural indoor ambience, soft room tone, faint ticking clock, clear kitten voice. Camera is handheld, mostly static with slight natural sway, warm soft shadows, slightly underexposed cozy indoor look.'
```

(The `'\''` sequences escape the single quotes inside the prompt.
Use bash, not zsh's `setopt RC_QUOTES`.)

---

# Recipes

All recipes:
- Run from the LTX-2-MLX repo root.
- Use `caffeinate -di` to prevent sleep.
- Use `--output-prefix <name>` so the run is saved to
  `<output_dir>/<name>_<timestamp>.mp4` automatically.  `<output_dir>`
  comes from `$DIFFUSERS_OUTPUT_DIR` / `$OUTPUT_DIR` env / `outputs/`
  fallback; export one of those to control where files land.
- Use `--save-all-sidecars` so every artifact lands alongside the mp4.
- Use `--fast-mode` (no intermediate evals — established default for
  perf runs).
- Pick the right `--duration` for the prompt's reference setting.
  `--fps` defaults to 24, so it's omitted.

## Bakery — full quality (`distilled` two-stage, 1024x576, ~25 min)

```bash
caffeinate -di python scripts/generate.py "$BAKERY" \
  --pipeline distilled \
  --height 576 --width 1024 --duration 20 --seed 124 \
  --generate-audio \
  --fast-mode \
  --save-all-sidecars \
  --output-prefix bakery
```

## Bakery — smoke (`one-stage`, 512x288, ~7-8 min)

Single-stage distilled at the half-bakery resolution.  No spatial
upscaler, no stage-2 refinement.  Same 8-step distilled schedule.
Fast for iterating on the prompt or doing seed sweeps.

```bash
caffeinate -di python scripts/generate.py "$BAKERY" \
  --pipeline one-stage \
  --height 288 --width 512 --duration 20 --seed 124 \
  --generate-audio \
  --fast-mode \
  --save-all-sidecars \
  --output-prefix bakery_smoke
```

## Kitten — full quality (`distilled` two-stage, 1024x576, ~45 min)

```bash
caffeinate -di python scripts/generate.py "$KITTEN" \
  --pipeline distilled \
  --height 576 --width 1024 --duration 30 --seed 42 \
  --generate-audio \
  --fast-mode \
  --save-all-sidecars \
  --output-prefix kitten
```

## Kitten — smoke (`one-stage`, 512x288, ~11-12 min)

```bash
caffeinate -di python scripts/generate.py "$KITTEN" \
  --pipeline one-stage \
  --height 288 --width 512 --duration 30 --seed 42 \
  --generate-audio \
  --fast-mode \
  --save-all-sidecars \
  --output-prefix kitten_smoke
```

---

# Recipe notes

- **`--pipeline distilled` requires `--height` and `--width` divisible
  by 64** (so the half-res stage 1 ends up divisible by 32).  Smallest
  valid 16:9 for this mode is **1024x576**.
- **`--pipeline one-stage` only needs dims divisible by 32**.  The
  next 16:9 step down is **512x288** (which is also what stage 1 of
  the full bakery runs at).  Lower than that, you have to drop the
  16:9 constraint.
- Both modes use the same distilled weights and the fixed 8-step
  distilled sigma schedule.  Two-stage adds a 3-step stage-2 refine
  after spatial upscale.
- With `cfg_scale=1.0` (default for distilled), the negative prompt
  encoding is skipped and the denoise loop runs the
  positive-only single-pass path (no second transformer call per
  step).  Should see `(distilled <mode> doesn't use negative)` in
  the prompt-encoding log line and `Running optimized single-pass
  inference` in the transformer-load log line.
