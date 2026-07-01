# FBCNN weights (not bundled)

The FBCNN checkpoints (~287MB each) are not committed. Download + convert; the
`.safetensors` are gitignored. `--deblock fbcnn` uses the **color** model.

All at `https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/`; sha256 of the
source `.pth`:

| model | file | sha256 (.pth) |
| --- | --- | --- |
| color (RGB, used by --deblock fbcnn) | fbcnn_color.pth | `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8` |
| gray | fbcnn_gray.pth | `7beec25b883a6c3bd7d103429b78a2371f7e85c1f5261a6e2e5d0702d9e9f977` |
| gray_double | fbcnn_gray_double.pth | `4444b8b2393649a4acd5d8730410ec3106d15b6e262a49081e4e87b44cfa5cea` |

Convert (color -- the RGB model the driver uses):

```bash
curl -L -O https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth
python scripts/pth_to_safetensors.py fbcnn_color.pth \
  -o LTX_2_MLX/videotoolbox/fbcnn/weights/fbcnn_color.safetensors --strip-prefix ''
```

The converter statically scans the pickle (nothing executed), loads with
`weights_only=True`, and verifies the output loads in MLX. `--deblock fbcnn` then finds
it here, or point `$FBCNN_WEIGHTS` / `--deblock-weights` at any path. `gray` and
`gray_double` are listed in `net.py`'s `_VARIANTS` but the RGB driver only uses `color`.

Source: Jiang et al., "Towards Flexible Blind JPEG Artifacts Removal" (ICCV 2021) --
https://github.com/jiaxi-jiang/FBCNN
