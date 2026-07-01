# FastDVDnet weights

`model.safetensors` and `model_clipped_noise.safetensors` (~9.5MB each) ARE bundled
(`--denoise fastdvd --fastdvd-variant model|clipped`). Provenance / re-download -- direct
download and the sha256 of the source `.pth`:

| variant | download | sha256 |
| --- | --- | --- |
| model | https://github.com/m-tassano/fastdvdnet/raw/master/model.pth | `9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17` |
| clipped | https://github.com/m-tassano/fastdvdnet/raw/master/model_clipped_noise.pth | `8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949` |

Convert:

```bash
curl -L -O https://github.com/m-tassano/fastdvdnet/raw/master/model.pth
python scripts/pth_to_safetensors.py model.pth \
  -o LTX_2_MLX/videotoolbox/fastdvdnet/weights/model.safetensors
```

The default `--strip-prefix 'module.'` handles the DataParallel prefix.

Source: Tassano et al., "FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow
Estimation" (CVPR 2020) -- https://github.com/m-tassano/fastdvdnet
