# RealBasicVSR weights

`realbasicvsr_x4.safetensors` (~24MB) IS bundled and ships in the repo. Provenance /
re-download:

- HuggingFace mirror (direct): https://huggingface.co/akhaliq/RealBasicVSR_x4 -> `RealBasicVSR_x4.pth`
- Official repo (Google Drive / Dropbox / OneDrive links in its README):
  https://github.com/ckkelvinchan/RealBasicVSR

Convert:

```bash
curl -L -O https://huggingface.co/akhaliq/RealBasicVSR_x4/resolve/main/RealBasicVSR_x4.pth
python scripts/pth_to_safetensors.py RealBasicVSR_x4.pth \
  -o LTX_2_MLX/videotoolbox/realbasicvsr/weights/realbasicvsr_x4.safetensors --strip-prefix 'generator.'
```

The converter auto-detects the nesting; adjust `--strip-prefix` if the printed keys do
not match `net.py`.

Source: Chan et al., "Investigating Tradeoffs in Real-World Video Super-Resolution"
(CVPR 2022) -- https://github.com/ckkelvinchan/RealBasicVSR
