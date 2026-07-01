# FastDVDnet weights

`model.safetensors` and `model_clipped_noise.safetensors` (~9.5MB each) ARE bundled and
ship in the repo (`--denoise fastdvd`, `--fastdvd-variant model|clipped`). Provenance /
re-download:

Official repo: https://github.com/m-tassano/fastdvdnet -- `model.pth` (AWGN) and
`model_clipped_noise.pth` (clipped-AWGN) in the `models/` folder.

Convert:

```bash
python scripts/pth_to_safetensors.py model.pth \
  -o LTX_2_MLX/videotoolbox/fastdvdnet/weights/model.safetensors
python scripts/pth_to_safetensors.py model_clipped_noise.pth \
  -o LTX_2_MLX/videotoolbox/fastdvdnet/weights/model_clipped_noise.safetensors
```

The default `--strip-prefix 'module.'` handles the DataParallel prefix; the converter
prints the resulting keys.

Source: Tassano et al., "FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow
Estimation" (CVPR 2020) -- https://github.com/m-tassano/fastdvdnet
