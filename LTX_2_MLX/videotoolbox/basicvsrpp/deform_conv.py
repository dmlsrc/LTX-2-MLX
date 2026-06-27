"""Modulated deformable conv (DCNv2) in MLX, adapted from the mps-deform-conv
Metal kernel (deformable im2col) + mx.matmul for the GEMM. No torch.

deform_conv2d(input, offset, weight, bias, mask, ...) matches torchvision's
signature/layout (NCHW, offset (N, dg*2*K*K, oH, oW), mask (N, dg*K*K, oH, oW)).
Validated against a pure-MLX gather reference below.
"""
from __future__ import annotations

import mlx.core as mx

# bilinear_interpolate + the im2col body, lifted from mps-deform-conv's
# deformable_im2col kernel; dims are read from an int32 `params` buffer instead
# of `constant&` args, and buffers are referenced by MLX input/output name.
_HEADER = r"""
inline float bilin(device const float* in, int H, int W, float h, float w) {
    if (h <= -1.0f || float(H) <= h || w <= -1.0f || float(W) <= w) return 0.0f;
    int hl = int(floor(h)), wl = int(floor(w)), hh = hl + 1, wh = wl + 1;
    float lh = h - float(hl), lw = w - float(wl), mh = 1.0f - lh, mw = 1.0f - lw;
    float v1 = (hl >= 0 && wl >= 0)        ? in[hl * W + wl] : 0.0f;
    float v2 = (hl >= 0 && wh <= W - 1)    ? in[hl * W + wh] : 0.0f;
    float v3 = (hh <= H - 1 && wl >= 0)    ? in[hh * W + wl] : 0.0f;
    float v4 = (hh <= H - 1 && wh <= W - 1)? in[hh * W + wh] : 0.0f;
    return mh*mw*v1 + mh*lw*v2 + lh*mw*v3 + lh*lw*v4;
}
"""

_SRC = r"""
    uint gid = thread_position_in_grid.x;
    int H=params[0], W=params[1], KH=params[2], KW=params[3];
    int padH=params[4], padW=params[5], sH=params[6], sW=params[7];
    int dH=params[8], dW=params[9], batch=params[10], Cin=params[11];
    int n_grps=params[12], outH=params[13], outW=params[14], use_mask=params[15];
    int n = Cin * outH * outW * batch;
    if (int(gid) >= n) return;
    int index = int(gid);
    int out_x = index % outW;
    int out_y = (index / outW) % outH;
    int out_b = (index / (outW * outH)) % batch;
    int in_c  = index / (outW * outH * batch);
    int out_c = in_c * KH * KW;
    int grp   = in_c / (Cin / n_grps);
    int colstride = batch * outH * outW;
    int hw = outH * outW;
    int pix = out_y * outW + out_x;
    int inbase   = out_b * (Cin * H * W) + in_c * (H * W);
    int offbase  = (out_b * n_grps + grp) * 2 * KH * KW * hw;
    int maskbase = (out_b * n_grps + grp) * KH * KW * hw;
    for (int i = 0; i < KH; ++i) {
        for (int j = 0; j < KW; ++j) {
            int t = i * KW + j;
            float mv = use_mask ? mask[maskbase + t * hw + pix] : 1.0f;
            float oh = offset[offbase + (2 * t)     * hw + pix];
            float ow = offset[offbase + (2 * t + 1) * hw + pix];
            float y = float(out_y * sH - padH) + float(i * dH) + oh;
            float x = float(out_x * sW - padW) + float(j * dW) + ow;
            columns[(out_c + t) * colstride + out_b * hw + pix] =
                mv * bilin(input + inbase, H, W, y, x);
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="deform_im2col", input_names=["input", "offset", "mask", "params"],
            output_names=["columns"], header=_HEADER, source=_SRC,
        )
    return _KERNEL


def deform_conv2d(inp, offset, weight, bias=None, mask=None,
                  stride=1, padding=1, dilation=1, deform_groups=1):
    N, Cin, H, W = inp.shape
    Cout, _, KH, KW = weight.shape
    oH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    oW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    use_mask = 1 if mask is not None else 0
    params = mx.array([H, W, KH, KW, padding, padding, stride, stride,
                       dilation, dilation, N, Cin, deform_groups, oH, oW, use_mask],
                      dtype=mx.int32)
    inp_c = mx.contiguous(inp.astype(mx.float32))
    off_c = mx.contiguous(offset.astype(mx.float32))
    msk_c = mx.contiguous((mask if mask is not None else mx.zeros((N, 1, oH, oW))).astype(mx.float32))
    (cols,) = _kernel()(
        inputs=[inp_c, off_c, msk_c, params],
        grid=(Cin * oH * oW * N, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(Cin * KH * KW, N * oH * oW)], output_dtypes=[mx.float32],
    )
    out = (weight.reshape(Cout, Cin * KH * KW).astype(mx.float32) @ cols)  # (Cout, N*oH*oW)
    out = out.reshape(Cout, N, oH, oW).transpose(1, 0, 2, 3)
    if bias is not None:
        out = out + bias.reshape(1, Cout, 1, 1)
    return out


# --- pure-MLX reference (gather-based im2col), for validation only ----------
def _bilin_sample(img, sy, sx):
    """img (N,C,H,W) sampled at (sy,sx) (N,oH,oW) -> (N,C,oH,oW), 0 outside."""
    N, C, H, W = img.shape
    oH, oW = sy.shape[1], sy.shape[2]
    y0 = mx.floor(sy)
    x0 = mx.floor(sx)
    ly = (sy - y0)[:, None]
    lx = (sx - x0)[:, None]
    y0i = y0.astype(mx.int32)
    x0i = x0.astype(mx.int32)
    flat = img.reshape(N, C, H * W)

    def g(yi, xi):
        valid = ((yi >= 0) & (yi <= H - 1) & (xi >= 0) & (xi <= W - 1)).astype(mx.float32)
        idx = (mx.clip(yi, 0, H - 1) * W + mx.clip(xi, 0, W - 1)).reshape(N, 1, oH * oW)
        v = mx.take_along_axis(flat, mx.broadcast_to(idx, (N, C, oH * oW)), axis=2)
        return v.reshape(N, C, oH, oW) * valid[:, None]

    v1 = g(y0i, x0i)
    v2 = g(y0i, x0i + 1)
    v3 = g(y0i + 1, x0i)
    v4 = g(y0i + 1, x0i + 1)
    return (1 - ly) * (1 - lx) * v1 + (1 - ly) * lx * v2 + ly * (1 - lx) * v3 + ly * lx * v4


def ref_deform_conv2d(inp, offset, weight, bias=None, mask=None,
                      stride=1, padding=1, dilation=1, deform_groups=1):
    N, Cin, H, W = inp.shape
    Cout, _, KH, KW = weight.shape
    oH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    oW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    oy = (mx.arange(oH, dtype=mx.float32) * stride - padding).reshape(1, oH, 1)
    ox = (mx.arange(oW, dtype=mx.float32) * stride - padding).reshape(1, 1, oW)
    cpg = Cin // deform_groups
    groups = []
    for grp in range(deform_groups):
        taps = []
        gch = inp[:, grp * cpg:(grp + 1) * cpg]
        for i in range(KH):
            for j in range(KW):
                t = i * KW + j
                oh = offset[:, grp * 2 * KH * KW + 2 * t]
                ow = offset[:, grp * 2 * KH * KW + 2 * t + 1]
                samp = _bilin_sample(gch, oy + i * dilation + oh, ox + j * dilation + ow)
                if mask is not None:
                    samp = samp * mask[:, grp * KH * KW + t][:, None]
                taps.append(samp)
        groups.append(mx.stack(taps, axis=2))           # (N, cpg, KH*KW, oH, oW)
    col = mx.concatenate(groups, axis=1).reshape(N, Cin * KH * KW, oH, oW)
    cols = col.transpose(1, 0, 2, 3).reshape(Cin * KH * KW, N * oH * oW)
    out = (weight.reshape(Cout, Cin * KH * KW) @ cols).reshape(Cout, N, oH, oW).transpose(1, 0, 2, 3)
    if bias is not None:
        out = out + bias.reshape(1, Cout, 1, 1)
    return out


if __name__ == "__main__":
    mx.random.seed(0)
    # BasicVSR++-shaped-ish small case: deform_groups, mask, 3x3 pad1
    N, Cin, Cout, H, W, K, dg = 1, 8, 6, 12, 14, 3, 2
    inp = mx.random.normal((N, Cin, H, W))
    weight = mx.random.normal((Cout, Cin, K, K)) * 0.1
    bias = mx.random.normal((Cout,))
    offset = mx.random.normal((N, dg * 2 * K * K, H, W)) * 2.0   # real offsets
    mask = mx.sigmoid(mx.random.normal((N, dg * K * K, H, W)))
    a = deform_conv2d(inp, offset, weight, bias, mask, deform_groups=dg)
    b = ref_deform_conv2d(inp, offset, weight, bias, mask, deform_groups=dg)
    mx.eval(a, b)
    print(f"kernel vs pure-MLX ref: max|diff|={float(mx.max(mx.abs(a - b))):.2e}  shapes {a.shape}")
    # also without mask (plain deformable conv, DCNv1)
    a2 = deform_conv2d(inp, offset, weight, None, None, deform_groups=dg)
    b2 = ref_deform_conv2d(inp, offset, weight, None, None, deform_groups=dg)
    mx.eval(a2, b2)
    print(f"no-mask (DCNv1):        max|diff|={float(mx.max(mx.abs(a2 - b2))):.2e}")
