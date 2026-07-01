# BasicVSR++ weights (not bundled)

These checkpoints are not committed to the repo (too large). Download one from the
OpenMMLab model zoo and convert it here; `*.safetensors` are gitignored.

`--spatial-mode basicvsrpp --basicvsrpp-variant <token>` (default `vimeo90k_bd`):

| token | checkpoint (.pth) | for |
| --- | --- | --- |
| reds4 | basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth | REDS, 4x |
| vimeo90k_bd | basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth | Vimeo blur-down, 4x (default) |
| vimeo90k_bi | basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth | Vimeo bicubic, 4x |
| ntire_vsr | basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth | NTIRE'21 VSR (c128n25, sharpest) |

All live under `https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/`.
Example (the default):

```bash
BASE=https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus
curl -L -O $BASE/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth
python scripts/pth_to_safetensors.py \
  basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth \
  -o LTX_2_MLX/videotoolbox/basicvsrpp/weights/basicvsrpp_vimeo90k_bd.safetensors \
  --strip-prefix 'generator.'
```

Output names must match `net.py`'s `_VARIANTS` (`basicvsrpp_<token>.safetensors`). The
converter auto-detects the checkpoint nesting (`state_dict`) and prints the resulting
keys; the mmediting checkpoints prefix everything with `generator.`, hence
`--strip-prefix`. `ntire_vsr` is a single ~167MB file now (the old GitHub-limit sharding
was removed).

Source: https://github.com/open-mmlab/mmagic -- BasicVSR++ (Chan et al., CVPR 2022).
