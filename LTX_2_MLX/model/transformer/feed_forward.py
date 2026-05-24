"""Feed-forward networks for LTX-2 Transformer."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.kernels import silu_mul


def gelu_approx(x: mx.array) -> mx.array:
    """
    GELU activation with tanh approximation.

    This is the fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return nn.gelu_approx(x)


def _empty_linear() -> nn.Linear:
    """Create an update-ready Linear without allocating full random weights."""
    linear = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(linear)
    linear.weight = mx.zeros((0, 0), dtype=mx.float32)
    linear.bias = mx.zeros((0,), dtype=mx.float32)
    return linear


class GELUApprox(nn.Module):
    """Linear layer followed by GELU (tanh approximation)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    Architecture: Linear -> GELU -> Linear
    """

    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        """
        Initialize feed-forward network.

        Args:
            dim: Input dimension.
            dim_out: Output dimension.
            mult: Multiplier for hidden dimension.
        """
        super().__init__()
        inner_dim = int(dim * mult)

        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.project_in = GELUApprox(dim, inner_dim)
        self.project_out = nn.Linear(inner_dim, dim_out)
        self._project_in_weight_t = None
        self._project_out_weight_t = None

    def __call__(self, x: mx.array) -> mx.array:
        # When the cached pre-transposed weight has a different dtype than the
        # input (e.g. cache was built with `--video-ff-dtype float16` so the
        # weight is FP16 while the residual stream stays BF16), do the cast
        # at the FF boundary so the interior — including the huge inner_dim
        # hidden activation — lives in the weight dtype.  Output is cast back
        # at exit so the residual stream is unchanged.
        target_dtype = self._target_compute_dtype()
        if target_dtype is None or x.dtype == target_dtype:
            x = self._project_in(x)
            x = self._project_out(x)
            return x
        orig_dtype = x.dtype
        x = x.astype(target_dtype)
        x = self._project_in(x)
        x = self._project_out(x)
        return x.astype(orig_dtype)

    def _target_compute_dtype(self) -> Optional[mx.Dtype]:
        """The dtype of the cached pre-transposed weights, or None when both
        slots are empty (i.e. fall back to `nn.Linear` BF16 path)."""
        if self._project_in_weight_t is not None:
            return self._project_in_weight_t.dtype
        if self._project_out_weight_t is not None:
            return self._project_out_weight_t.dtype
        return None

    def _project_in_linear(self, x: mx.array) -> mx.array:
        """Run project_in linear, optionally using a pre-transposed contiguous weight."""
        if self._project_in_weight_t is None:
            return self.project_in.proj(x)

        bias = self.project_in.proj.get("bias")
        if bias is not None:
            return mx.addmm(bias, x, self._project_in_weight_t)
        return x @ self._project_in_weight_t

    def _project_in(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self._project_in_linear(x))

    def _project_out(self, x: mx.array) -> mx.array:
        """Run project_out, optionally using a pre-transposed contiguous weight."""
        if self._project_out_weight_t is None:
            return self.project_out(x)

        bias = self.project_out.get("bias")
        if bias is not None:
            return mx.addmm(bias, x, self._project_out_weight_t)
        return x @ self._project_out_weight_t

    def pretranspose_project_in(self) -> list[mx.array]:
        """Cache a contiguous ``weight.T`` for project_in same-math experiments."""
        if not isinstance(self.project_in.proj, nn.Linear):
            raise ValueError("project_in pretranspose only supports nn.Linear")
        if self._project_in_weight_t is not None:
            arrays = [self._project_in_weight_t]
            bias = self.project_in.proj.get("bias")
            if bias is not None:
                arrays.append(bias)
            return arrays
        if "weight" not in self.project_in.proj:
            raise ValueError("project_in weight is unavailable for pretranspose")

        self._project_in_weight_t = mx.contiguous(self.project_in.proj.weight.T)
        arrays = [self._project_in_weight_t]
        bias = self.project_in.proj.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def pretranspose_project_out(self) -> list[mx.array]:
        """Cache a contiguous ``weight.T`` for project_out same-math experiments."""
        if not isinstance(self.project_out, nn.Linear):
            raise ValueError("project_out pretranspose only supports nn.Linear")
        if self._project_out_weight_t is not None:
            arrays = [self._project_out_weight_t]
            bias = self.project_out.get("bias")
            if bias is not None:
                arrays.append(bias)
            return arrays
        if "weight" not in self.project_out:
            raise ValueError("project_out weight is unavailable for pretranspose")

        self._project_out_weight_t = mx.contiguous(self.project_out.weight.T)
        arrays = [self._project_out_weight_t]
        bias = self.project_out.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def drop_layout_sources(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> None:
        """Drop original arrays replaced by materialized layout transforms."""
        for target, layout in layout_specs:
            if target == "project_in" and layout == "pretranspose":
                if self._project_in_weight_t is not None and "weight" in self.project_in.proj:
                    del self.project_in.proj.weight
            elif target == "project_out" and layout == "pretranspose":
                if self._project_out_weight_t is not None and "weight" in self.project_out:
                    del self.project_out.weight
            else:
                raise ValueError(f"Unsupported FF layout spec: {target}:{layout}")

    def restore_linear_projections(
        self,
        targets: tuple[str, ...] = ("project_in", "project_out"),
    ) -> None:
        """Restore projection modules after in-place quantization experiments."""
        if "project_in" in targets:
            self._project_in_weight_t = None
            if not isinstance(self.project_in.proj, nn.Linear):
                self.project_in.proj = _empty_linear()
        if "project_out" in targets:
            self._project_out_weight_t = None
            if not isinstance(self.project_out, nn.Linear):
                self.project_out = _empty_linear()

    def _quantized_linear_arrays(self, linear: nn.QuantizedLinear) -> list[mx.array]:
        arrays = [linear.weight, linear.scales]
        biases = linear.get("biases")
        if biases is not None:
            arrays.append(biases)
        bias = linear.get("bias")
        if bias is not None:
            arrays.append(bias)
        return arrays

    def quantize_project_in(
        self,
        mode: str = "mxfp8",
        group_size: int | None = None,
        bits: int | None = None,
    ) -> list[mx.array]:
        """Replace the input projection with an MLX quantized linear layer."""
        self._project_in_weight_t = None
        self.project_in.proj = nn.QuantizedLinear.from_linear(
            self.project_in.proj,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

        return self._quantized_linear_arrays(self.project_in.proj)

    def quantize_project_out(
        self,
        mode: str = "mxfp8",
        group_size: int | None = None,
        bits: int | None = None,
    ) -> list[mx.array]:
        """Replace the output projection with an MLX quantized linear layer."""
        self._project_out_weight_t = None
        self.project_out = nn.QuantizedLinear.from_linear(
            self.project_out,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

        return self._quantized_linear_arrays(self.project_out)

    def quantize_projections(
        self,
        targets: tuple[str, ...] = ("project_out",),
        mode: str = "mxfp8",
        group_size: int | None = None,
        bits: int | None = None,
    ) -> list[mx.array]:
        """Replace selected feed-forward projections with quantized linear layers."""
        arrays: list[mx.array] = []
        for target in targets:
            if target == "project_in":
                arrays.extend(self.quantize_project_in(
                    mode=mode,
                    group_size=group_size,
                    bits=bits,
                ))
            elif target == "project_out":
                arrays.extend(self.quantize_project_out(
                    mode=mode,
                    group_size=group_size,
                    bits=bits,
                ))
            else:
                raise ValueError(f"Unsupported FF quantization target: {target}")
        return arrays

    def apply_layouts(
        self,
        layout_specs: tuple[tuple[str, str], ...],
    ) -> list[mx.array]:
        """Apply selected same-math layout transforms to feed-forward projections."""
        arrays: list[mx.array] = []
        for target, layout in layout_specs:
            if target == "project_in" and layout == "pretranspose":
                arrays.extend(self.pretranspose_project_in())
            elif target == "project_out" and layout == "pretranspose":
                arrays.extend(self.pretranspose_project_out())
            else:
                raise ValueError(f"Unsupported FF layout spec: {target}:{layout}")
        return arrays

    def profile(self, x: mx.array, prefix: str, mark_profile) -> mx.array:
        """Forward pass with forced-eval timing checkpoints for diagnostics."""
        x = self._project_in_linear(x)
        mark_profile(f"{prefix} project_in", x)
        x = nn.gelu_approx(x)
        mark_profile(f"{prefix} gelu", x)
        x = self._project_out(x)
        mark_profile(f"{prefix} project_out", x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (alternative to standard FFN).

    Architecture: x -> Linear_gate * SiLU(Linear_up) -> Linear_down
    """

    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)

        self.w_up = nn.Linear(dim, inner_dim, bias=False)
        self.w_gate = nn.Linear(dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, dim_out, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Use fused silu_mul kernel for efficiency
        return self.w_down(silu_mul(self.w_gate(x), self.w_up(x)))
