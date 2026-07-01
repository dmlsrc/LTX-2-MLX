# BasicVSR++ weights (not bundled)

Too large for the repo -- download from the OpenMMLab model zoo and convert here;
`*.safetensors` are gitignored.

`--spatial-mode basicvsrpp --basicvsrpp-variant <token>` (default `vimeo90k_bd`). All
under `https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/`; sha256 is
of the source `.pth` (its first 8 hex also appear in the filename):

| token | file | sha256 (.pth) |
| --- | --- | --- |
| reds4 | basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth | `db622b2fd4caae0a4c63ab5e54f1cfef7a62a0f3b8ad101aba2eae068d928549` |
| vimeo90k_bd | basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth | `ab315ab1d83d09834d43c8ba17d019282a93f0cbf40bfd49be99dfcebf4c12eb` |
| vimeo90k_bi | basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth | `4ef437e27e7ed468853d9bad2e7a02e50fd4582c986a5cb4231054b0021b2e81` |
| ntire_vsr | basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth | `1ff352921112b84cacc79b23df502a2319d5714d6b26730248839a6e4074c285` |

Convert (the default):

```bash
BASE=https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus
curl -L -O $BASE/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth
python scripts/pth_to_safetensors.py \
  basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth \
  -o LTX_2_MLX/videotoolbox/basicvsrpp/weights/basicvsrpp_vimeo90k_bd.safetensors \
  --strip-prefix 'generator.'
```

Output names must match `net.py`'s `_VARIANTS` (`basicvsrpp_<token>.safetensors`). The
mmediting checkpoints nest under `state_dict` and prefix keys with `generator.`.
`ntire_vsr` is a single ~167MB file now (the GitHub-limit sharding was removed).

Source: https://github.com/open-mmlab/mmagic -- BasicVSR++ (Chan et al., CVPR 2022).
