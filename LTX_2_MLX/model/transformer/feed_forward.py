"""Feed-forward networks for LTX-2 Transformer."""

import mlx.core as mx
import mlx.nn as nn

from LTX_2_MLX.kernels import silu_mul


def gelu_approx(x: mx.array) -> mx.array:
    """
    GELU activation with tanh approximation.

    This is the fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return nn.gelu_approx(x)


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

        self.project_in = GELUApprox(dim, inner_dim)
        self.project_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.project_in(x)
        x = self.project_out(x)
        return x

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

    def profile(self, x: mx.array, prefix: str, mark_profile) -> mx.array:
        """Forward pass with forced-eval timing checkpoints for diagnostics."""
        x = self.project_in.proj(x)
        mark_profile(f"{prefix} project_in", x)
        x = nn.gelu_approx(x)
        mark_profile(f"{prefix} gelu", x)
        x = self.project_out(x)
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
