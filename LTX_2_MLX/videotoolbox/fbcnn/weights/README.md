# FBCNN weights (not bundled)

The FBCNN checkpoints are ~287MB each and are deliberately **not** committed to this
repo. Download one and convert it to safetensors here; the `*.safetensors` files are
gitignored and never pushed.

## Color (RGB) -- the checkpoint `--deblock fbcnn` uses

```bash
curl -L -o fbcnn_color.pth \
  https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth

python scripts/pth_to_safetensors.py fbcnn_color.pth \
  -o LTX_2_MLX/videotoolbox/fbcnn/weights/fbcnn_color.safetensors --strip-prefix ''
```

The converter statically scans the pickle (nothing is executed), loads it with
`torch.load(weights_only=True)`, demotes any float64, and verifies the output loads in
MLX. After that, `--deblock fbcnn` finds the file automatically (this directory), or you
can point `$FBCNN_WEIGHTS` / `--deblock-weights` at any `fbcnn_color.safetensors`.

## Other variants

`fbcnn_gray.pth` and `fbcnn_gray_double.pth` (grayscale / double-JPEG) live at the same
release and convert the same way. They are listed in `net.py`'s `_VARIANTS`, but the RGB
`FbcnnDeblocker` only uses the color model, so they are not needed for video.

Source: Jiang et al., "Towards Flexible Blind JPEG Artifacts Removal" (ICCV 2021),
https://github.com/jiaxi-jiang/FBCNN
