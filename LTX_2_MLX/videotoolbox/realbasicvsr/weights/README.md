# RealBasicVSR weights

`realbasicvsr_x4.safetensors` (~24MB) IS bundled in the repo. Provenance / re-download.

Direct (OpenMMLab mirror -- byte-identical weights to the author's `RealBasicVSR_x4.pth`,
verified against our converted file):

- https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth
- sha256 (.pth): `52f77c2c835aaa3fe675b3959b2f85010a6c6f63f77f7e279394646e55a4e376`

The author's original is Google-Drive-only (https://github.com/ckkelvinchan/RealBasicVSR),
also mirrored at https://huggingface.co/akhaliq/RealBasicVSR_x4 -- same weights, different
container (so a different file hash).

Convert:

```bash
curl -L -O https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth
python scripts/pth_to_safetensors.py realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth \
  -o LTX_2_MLX/videotoolbox/realbasicvsr/weights/realbasicvsr_x4.safetensors --strip-prefix 'generator.'
```

Source: Chan et al., "Investigating Tradeoffs in Real-World Video Super-Resolution"
(CVPR 2022).
