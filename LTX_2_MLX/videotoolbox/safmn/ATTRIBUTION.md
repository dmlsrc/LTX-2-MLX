# Attribution

`videotoolbox/safmn/` is an independent MLX reimplementation of the SAFMN
super-resolution family, written from the published architectures as a spec -- no
upstream code is bundled. The model weights are the upstream project's,
redistributed under its license (see `weights/README.md` for the downloads).

## SAFMN

Long Sun, Jiangxin Dong, Jinshan Pan et al. -- "Spatially-Adaptive Feature
Modulation for Efficient Image Super-Resolution" (ICCV 2023).
https://github.com/sunny2109/SAFMN

The two ported variants are the project's challenge winners: `light_SAFMN++`
(1st place, fidelity track, AIS 2024 Real-Time 4K Super-Resolution of compressed
AVIF images) and `Real_SAFMN++` / SAFMN-L (1st place, AIM 2025 Efficient
Perceptual Super-Resolution).

Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0). Copyright the
SAFMN authors. You may not use these files except in compliance with the License;
the full text is at https://www.apache.org/licenses/LICENSE-2.0 and in the
upstream repository's `LICENSE`. Unless required by applicable law or agreed to
in writing, the software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND.
