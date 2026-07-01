# Real-ESRGAN weights

Two **general** SRVGG models are bundled in the repo: `realesr_general_x4v3` and its
denoise companion `realesr_general_wdn_x4v3` (~4.6MB each -- the `--realesrgan-denoise`
dial). The four **RRDBNet x4** models (~64MB) are NOT committed -- download + convert.

`--spatial-mode realesrgan --realesrgan-weights <token>`. sha256 is of the source `.pth`:

| token | download (.pth) | sha256 (.pth) |
| --- | --- | --- |
| realesrgan_x4plus | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` |
| realesrnet_x4plus | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| bsrgan_x4 | https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth | `5d505a0766160921e0388d76e1ddf08cb114303990f9080432bf2b1c988b1c54` |
| bsrnet_x4 | https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth | `fa633d80ff4db5a546740ae8e3baebe925fc84d3c03e3e9002493acc5e88c3ec` |
| realesr_general_x4v3 (bundled) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` |
| realesr_general_wdn_x4v3 (bundled) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` |

Convert (verify the download first: `shasum -a 256 RealESRGAN_x4plus.pth`):

```bash
curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
python scripts/pth_to_safetensors.py RealESRGAN_x4plus.pth \
  -o LTX_2_MLX/videotoolbox/realesrgan/weights/realesrgan_x4plus.safetensors --strip-prefix ''
```

The converter auto-detects the nesting (RRDBNet uses `params_ema`, SRVGG uses `params`)
and prints the keys; output names must match `net.py`'s `_VARIANTS`.

Sources: https://github.com/xinntao/Real-ESRGAN (model zoo: `docs/model_zoo.md`),
https://github.com/cszn/BSRGAN
