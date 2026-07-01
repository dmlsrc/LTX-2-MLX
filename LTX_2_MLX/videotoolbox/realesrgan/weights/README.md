# Real-ESRGAN weights

Two general SRVGG models are bundled in the repo (`realesr_general_x4v3` and its denoise
companion `realesr_general_wdn_x4v3`, ~4.6MB each). The rest are not committed -- download
+ convert; the `.safetensors` are gitignored.

Direct download and the sha256 of the source `.pth`:

| variant | download | sha256 |
| --- | --- | --- |
| realesrgan_x4plus | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| realesrnet_x4plus | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| bsrgan_x4 | https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth | `5d505a0766160921e0388d76e1ddf08cb114303990f9080432bf2b1c988b1c54` |
| bsrnet_x4 | https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth | `fa633d80ff4db5a546740ae8e3baebe925fc84d3c03e3e9002493acc5e88c3ec` |
| realesr_general_x4v3 (bundled) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| realesr_general_wdn_x4v3 (bundled) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |
| RealESRGAN_x2plus (2x) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` |
| RealESRGAN_x4plus_anime_6B | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth | `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da` |
| realesr-animevideov3 | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth | `b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d` |
| ESRGAN_SRx4_DF2KOST_official | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth | `ff704c30ff560305e48ed1f4db895e525ab6bc81a46fafe80c0094a271c806d9` |

The first six are `--realesrgan-weights` tokens; the last four (2x, anime, official ESRGAN)
are not built-in variants -- convert and pass the `.safetensors` path. Convert:

```bash
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
python scripts/pth_to_safetensors.py RealESRGAN_x4plus.pth \
  -o LTX_2_MLX/videotoolbox/realesrgan/weights/realesrgan_x4plus.safetensors --strip-prefix ''
```

The converter auto-detects the nesting (RRDBNet uses `params_ema`, SRVGG uses `params`);
output names for the token models must match `net.py`'s `_VARIANTS`.

Source: https://github.com/xinntao/Real-ESRGAN , https://github.com/cszn/BSRGAN
