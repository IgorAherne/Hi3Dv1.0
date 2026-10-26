"""Feed-forward building block for the sparse transformer."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain the sparse feed-forward path only.
# Modified for Spellcast3D: the GELU in place on the hidden layer (the same values, one hidden layer less).

from __future__ import annotations

import torch
from torch import nn

from ... import accel
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU


class SparseFeedForwardNet(nn.Module):
    """Two-layer sparse MLP with tanh-approximated GELU."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        """
        Args:
            channels: Input and output channels.
            mlp_ratio: Hidden width as a multiple of ``channels``.
        """
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            SparseLinear(channels, hidden_channels),
            SparseGELU(approximate="tanh"),
            SparseLinear(hidden_channels, channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Transform sparse features with the MLP."""
        mlp = self.mlp
        if (
            accel.on("gelu_inplace")
            and not torch.is_grad_enabled()
            and isinstance(mlp, nn.Sequential)
            and len(mlp) == 3
            and type(mlp[1]) is SparseGELU
        ):
            h = mlp[0](x)
            torch.ops.aten.gelu_(h.feats, approximate=mlp[1].approximate)
            return mlp[2](h)
        return mlp(x)
