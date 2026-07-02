# ESC-Real weights

Not bundled -- download + convert; the `.safetensors` are gitignored. These are
`--esc-weights <token>` tokens (default `gan`). sha256 is of the source `.pth`:

| token | download | sha256 |
| --- | --- | --- |
| gan (perceptual, Real-ESRGAN-style degradation) | https://github.com/dslisleedh/ESC/releases/download/1.0.0/ESC_Real_X4_GAN.pth | `d6932bd01e29c9335f5f678b0e9b31ede906a42244887ab8888b1d39a8667f2d` |
| mse (fidelity twin) | https://github.com/dslisleedh/ESC/releases/download/1.0.0/ESC_Real_X4_MSE.pth | `ba22ae75b5f77382b1046e68634f58a53fc7ff5ffa46714133bfba30811ad943` |

Convert -- the output name must match `net.py`'s `_VARIANTS`:

```bash
curl -L -O https://github.com/dslisleedh/ESC/releases/download/1.0.0/ESC_Real_X4_GAN.pth
python scripts/pth_to_safetensors.py ESC_Real_X4_GAN.pth \
  -o LTX_2_MLX/videotoolbox/esc/weights/esc_real_x4_gan.safetensors
```

The release's other checkpoints are not video-appropriate: DIV2K/DF2K/DFLIP/light/FP/XL
are bicubic-benchmark models, FB_R48 is a torch-only FlashBias attention variant, and
the ATD/HiTSRF/SRFormer files are other architectures (the author's retrained
comparison baselines).

Source: https://github.com/dslisleedh/ESC
