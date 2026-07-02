# SAFMN weights

Not bundled -- download + convert; the `.safetensors` are gitignored. Both checkpoints
live inside the SAFMN GitHub repo itself, so the download links are raw-file URLs.
These are `--safmn-weights <token>` tokens (default `real`). sha256 is of the source
`.pth`:

| token | download | sha256 |
| --- | --- | --- |
| real (SAFMN-L / Real_SAFMN++, perceptual 4x) | https://github.com/sunny2109/SAFMN/raw/main/AIM2025_EPSR/Real-Time-4K-Perceptual-SR-Challenge-2025-main/model_zoo/SAFMN-L.pth | `3a8874cfaa5b4ca6014267fb550408cb3c21b06dd650b39cf3d1941a7a917d31` |
| light (light_SAFMN++, fidelity 4x) | https://github.com/sunny2109/SAFMN/raw/main/AIS2024-RTSR/pretrained_model/light_safmnpp.pth | `a542c92072cb25adab1f9cc5209d4f4f4ca8549db084e6703d2e032357cd50a7` |

Convert -- the output name must match `net.py`'s `_VARIANTS`:

```bash
curl -L -O https://github.com/sunny2109/SAFMN/raw/main/AIS2024-RTSR/pretrained_model/light_safmnpp.pth
python scripts/pth_to_safetensors.py light_safmnpp.pth \
  -o LTX_2_MLX/videotoolbox/safmn/weights/light_safmnpp.safetensors
```

(`SAFMN-L.pth` -> `safmn_l_real.safetensors` the same way.) The converter picks the
`params_ema` nesting automatically; variant, width, block count, and scale are
auto-detected from the checkpoint at load.

Source: https://github.com/sunny2109/SAFMN
