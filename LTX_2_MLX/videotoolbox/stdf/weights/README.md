# STDF weights

`stdf_mfqev2_r3.safetensors` and `stdf_vimeo90k_r3.safetensors` (~1.4MB each) ARE bundled
and ship in the repo (`--deblock stdf --deblock-weights mfqev2|vimeo90k`). Provenance /
re-download -- both R3 (7-frame) checkpoints are inside the release archive:

https://github.com/ryanxingql/stdf-pytorch/releases/download/v1.0.0/exp.zip

The repo is archived and uses a custom CUDA deform op -- read `net_stdf.py` as a spec,
do NOT run its code. Unzip and convert the two `.pt` checkpoints:

```bash
curl -L -O https://github.com/ryanxingql/stdf-pytorch/releases/download/v1.0.0/exp.zip
unzip exp.zip
# each experiment dir holds a ckp_*.pt; convert the MFQEv2-R3 and Vimeo90K-R3 ones:
python scripts/pth_to_safetensors.py <mfqev2_r3_exp>/ckp_*.pt \
  -o LTX_2_MLX/videotoolbox/stdf/weights/stdf_mfqev2_r3.safetensors --strip-prefix ''
python scripts/pth_to_safetensors.py <vimeo90k_r3_exp>/ckp_*.pt \
  -o LTX_2_MLX/videotoolbox/stdf/weights/stdf_vimeo90k_r3.safetensors --strip-prefix ''
```

The converter statically scans the pickle (nothing executed) and prints the keys.

Source: Deng et al., "Spatio-Temporal Deformable Convolution for Compressed Video Quality
Enhancement" (AAAI 2020); archived impl: https://github.com/ryanxingql/stdf-pytorch
