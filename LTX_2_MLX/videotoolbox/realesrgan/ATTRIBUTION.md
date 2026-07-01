# Attribution

`videotoolbox/realesrgan/` is an independent MLX reimplementation of the RRDBNet / SRVGG
generators below, written from the published architectures as a spec -- no upstream code is
bundled. The model weights are the upstream projects', redistributed under their licenses
(see `weights/README.md` for the per-token downloads).

## Real-ESRGAN / ESRGAN

Covers the `general`, `general-wdn`, `x4plus`, `realesrnet`, `x2plus`, `anime`,
`animevideo`, and `esrgan` tokens.

Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan -- "Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data" (ICCV Workshops 2021).
https://github.com/xinntao/Real-ESRGAN

The `esrgan` token is the original ESRGAN generator (Xintao Wang, Ke Yu, Shixiang Wu, et
al. -- "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", ECCV Workshops
2018), redistributed via the Real-ESRGAN releases.

Licensed under the BSD 3-Clause License (SPDX: BSD-3-Clause):

    BSD 3-Clause License

    Copyright (c) 2021, Xintao Wang
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are
    permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of
       conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list
       of conditions and the following disclaimer in the documentation and/or other
       materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be
       used to endorse or promote products derived from this software without specific
       prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## BSRGAN

Covers the `bsrgan` and `bsrnet` tokens.

Kai Zhang, Jingyun Liang, Luc Van Gool, Radu Timofte -- "Designing a Practical Degradation
Model for Deep Blind Image Super-Resolution" (ICCV 2021).
https://github.com/cszn/BSRGAN

Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0). Copyright 2022 Kai Zhang.
You may not use these files except in compliance with the License; the full text is at
https://www.apache.org/licenses/LICENSE-2.0 and in the upstream repository's `LICENSE`.
Unless required by applicable law or agreed to in writing, the software is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
