# NAFNet weights (not bundled)

The NAFNet checkpoints are 68-464MB each and are **not** committed to this repo.
Download one from the NAFNet model zoo and convert it to safetensors here; the
`*.safetensors` files are gitignored and never pushed.

## Variants

`--nafnet <token>` selects the task + width; `--nafnet-weights` overrides the path.

| token   | checkpoint                | task                     |
| ------- | ------------------------- | ------------------------ |
| gopro   | NAFNet-GoPro-width64.pth  | motion deblur            |
| gopro32 | NAFNet-GoPro-width32.pth  | motion deblur (~4x fast) |
| sidd    | NAFNet-SIDD-width64.pth   | real-noise denoise       |
| sidd32  | NAFNet-SIDD-width32.pth   | denoise (~4x fast)       |
| reds    | NAFNet-REDS-width64.pth   | video restore            |

## Download + convert

Get the `.pth` from the NAFNet model zoo (Google Drive links in the README):
https://github.com/megvii-research/NAFNet#results-and-pre-trained-models

Then convert it -- the output name must match the token's file in `net.py`'s
`_VARIANTS` (`nafnet_<task>_width<NN>.safetensors`). Example for the `gopro` token:

```bash
python scripts/pth_to_safetensors.py NAFNet-GoPro-width64.pth \
  -o LTX_2_MLX/videotoolbox/nafnet/weights/nafnet_gopro_width64.safetensors --strip-prefix ''
```

The converter statically scans the pickle (nothing is executed), loads it with
`torch.load(weights_only=True)` -- basicsr checkpoints nest the weights under a `params`
key, which the converter unwraps automatically -- and verifies the output loads in MLX.
`--nafnet gopro` then finds it here, or point `$NAFNET_WEIGHTS` / `--nafnet-weights` at
any path.

Source: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022),
https://github.com/megvii-research/NAFNet
