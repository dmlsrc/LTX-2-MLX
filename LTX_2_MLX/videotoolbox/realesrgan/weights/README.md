# Real-ESRGAN weights

The two **general** (SRVGG) models ARE bundled and ship in the repo:
`realesr_general_x4v3.safetensors` and its denoise companion
`realesr_general_wdn_x4v3.safetensors` (~4.6MB each -- the `--realesrgan-denoise` dial),
from Real-ESRGAN release v0.2.5.0.

The larger **RRDBNet x4** models (~64MB each) are NOT committed -- download + convert:

`--spatial-mode realesrgan --realesrgan-weights <token>`:

| token | checkpoint (.pth) | download |
| --- | --- | --- |
| realesrgan_x4plus | RealESRGAN_x4plus.pth | github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/ |
| realesrnet_x4plus | RealESRNet_x4plus.pth | github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ |
| bsrgan_x4 | BSRGAN.pth | github.com/cszn/KAIR/releases/download/v1.0/ |
| bsrnet_x4 | BSRNet.pth | github.com/cszn/KAIR/releases/download/v1.0/ |

Example:

```bash
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
python scripts/pth_to_safetensors.py RealESRGAN_x4plus.pth \
  -o LTX_2_MLX/videotoolbox/realesrgan/weights/realesrgan_x4plus.safetensors --strip-prefix ''
```

The converter auto-detects the checkpoint nesting (Real-ESRGAN uses `params_ema`); output
names must match `net.py`'s `_VARIANTS`. The two bundled general models came from
`realesr-general-x4v3.pth` / `realesr-general-wdn-x4v3.pth` (release v0.2.5.0).

Sources: https://github.com/xinntao/Real-ESRGAN , https://github.com/cszn/BSRGAN
