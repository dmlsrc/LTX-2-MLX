# Real-ESRGAN weights

Two general SRVGG models are bundled in the repo (`general` + its denoise companion,
~4.6MB each). The rest are not committed -- download + convert; the `.safetensors` are
gitignored. All of these are `--realesrgan-weights <token>` tokens (default `general`).
sha256 is of the source `.pth`:

| token | download | sha256 |
| --- | --- | --- |
| general (bundled) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| general-wdn (bundled; the --realesrgan-denoise companion) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |
| x4plus | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| realesrnet | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| bsrgan | https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth | `5d505a0766160921e0388d76e1ddf08cb114303990f9080432bf2b1c988b1c54` |
| bsrnet | https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth | `fa633d80ff4db5a546740ae8e3baebe925fc84d3c03e3e9002493acc5e88c3ec` |
| x2plus (2x output) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` |
| anime | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth | `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da` |
| animevideo | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth | `b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d` |
| esrgan | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth | `ff704c30ff560305e48ed1f4db895e525ab6bc81a46fafe80c0094a271c806d9` |

Convert -- the output name must match `net.py`'s `_VARIANTS` (e.g. x2plus ->
`realesrgan_x2plus.safetensors`, anime -> `realesrgan_x4plus_anime_6b.safetensors`,
animevideo -> `realesr_animevideov3.safetensors`, esrgan -> `esrgan_x4.safetensors`):

```bash
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
python scripts/pth_to_safetensors.py RealESRGAN_x4plus.pth \
  -o LTX_2_MLX/videotoolbox/realesrgan/weights/realesrgan_x4plus.safetensors --strip-prefix ''
```

Scale and block count auto-detect (x2plus outputs 2x, the rest 4x); the converter picks
the checkpoint nesting (RRDBNet uses `params_ema`, SRVGG uses `params`).

Source: https://github.com/xinntao/Real-ESRGAN , https://github.com/cszn/BSRGAN
