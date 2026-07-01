# STDF weights

`stdf_mfqev2_r3.safetensors` and `stdf_vimeo90k_r3.safetensors` (~1.4MB each) ARE bundled
(`--deblock stdf --deblock-weights mfqev2|vimeo90k`). Both are `.pt` checkpoints inside one
release archive:

https://github.com/ryanxingql/stdf-pytorch/releases/download/v1.0.0/exp.zip

| variant | path in the archive | sha256 |
| --- | --- | --- |
| mfqev2 | exp/MFQEv2_R3_enlarge300x/ckp_290000.pt | `d28e0b30082ecbbaeb7b8436968079875c97d882101cc4a79114d33eca2e1ec7` |
| vimeo90k | exp/Vimeo90K_R3_enlarge300x/ckp_300000.pt | `03b20836c7ed38ad3fdce0cec0360523cfade02673bd9e7b8e236e6e6826d708` |

The repo is archived and uses a custom CUDA deform op -- read `net_stdf.py` as a spec, do
NOT run its code. Extract + convert:

```bash
curl -L -O https://github.com/ryanxingql/stdf-pytorch/releases/download/v1.0.0/exp.zip
unzip exp.zip
python scripts/pth_to_safetensors.py exp/MFQEv2_R3_enlarge300x/ckp_290000.pt \
  -o LTX_2_MLX/videotoolbox/stdf/weights/stdf_mfqev2_r3.safetensors --strip-prefix ''
python scripts/pth_to_safetensors.py exp/Vimeo90K_R3_enlarge300x/ckp_300000.pt \
  -o LTX_2_MLX/videotoolbox/stdf/weights/stdf_vimeo90k_r3.safetensors --strip-prefix ''
```

Source: Deng et al., "Spatio-Temporal Deformable Convolution for Compressed Video Quality
Enhancement" (AAAI 2020) -- https://github.com/ryanxingql/stdf-pytorch
