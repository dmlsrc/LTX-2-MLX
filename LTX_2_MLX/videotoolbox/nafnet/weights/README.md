# NAFNet weights (not bundled)

The NAFNet checkpoints (68-464MB each) are not committed -- download + convert; the
`.safetensors` are gitignored. `--nafnet <token>`. Direct download and the sha256 of the
source `.pth`:

| variant | download | sha256 |
| --- | --- | --- |
| gopro | https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth | `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f` |
| gopro32 | https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth | `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c` |
| sidd | https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width64.pth | `cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445` |
| sidd32 | https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width32.pth | `89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4` |
| reds | https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth | `175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3` |

The URLs above are a HuggingFace mirror, byte-identical to the official checkpoints (sha256
verified against both). To download from the source instead, the originals are on Google
Drive via the NAFNet model zoo:
https://github.com/megvii-research/NAFNet#results-and-pre-trained-models

Convert (output name must match `net.py`'s `_VARIANTS`, `nafnet_<task>_width<NN>.safetensors`):

```bash
curl -L -O https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth
python scripts/pth_to_safetensors.py NAFNet-GoPro-width64.pth \
  -o LTX_2_MLX/videotoolbox/nafnet/weights/nafnet_gopro_width64.safetensors --strip-prefix ''
```

The converter statically scans the pickle (nothing executed); basicsr checkpoints nest
under a `params` key, unwrapped automatically.

Source: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022) --
https://github.com/megvii-research/NAFNet
