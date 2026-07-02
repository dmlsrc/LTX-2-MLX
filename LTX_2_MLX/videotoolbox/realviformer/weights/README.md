# RealViformer weights

Not bundled -- download + convert; the `.safetensors` are gitignored. The author
publishes a single pretrained model (Google Drive only; no direct URL exists):

| token | download | sha256 (source .pth) |
| --- | --- | --- |
| x4 | https://drive.google.com/drive/folders/1UzDfFSy5oELl7Z-umF_QhMQhUbUU378y (weights.pth) | `25d49a0128b1ecf218d8717b8a4041748b7fde25d8defb4ce82e53f120de9804` |

Convert -- the output name must match `net.py`'s `_VARIANTS`:

```bash
python scripts/pth_to_safetensors.py weights.pth \
  -o LTX_2_MLX/videotoolbox/realviformer/weights/realviformer_x4.safetensors
```

The checkpoint carries a vestigial `attn_merge.attn.masktemp` parameter that the
reference inference ignores (it loads strict=False); the loader here drops it.

Source: https://github.com/Yuehan717/RealViformer
