"""Spatial rotary position embedding shared by the sparse attention blocks."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain long-grid frequency scaling only.
# Modified for Spellcast3D: the phases computable once per coordinate set (``phases``) and passed in; the
# frequencies recomputable after a build on the meta device (``reset_freqs``).

from __future__ import annotations

import math

import torch
from torch import nn

from sparc3d.nn import accel


class RotaryPositionEmbedder(nn.Module):
    """Spatial rotary embedding with frequency scaling for long coordinate grids."""

    def __init__(self, hidden_size: int, in_channels: int = 3):
        """
        Args:
            hidden_size: Head dimension the rotation is applied over; must be even.
            in_channels: Spatial coordinate axes sharing that dimension.

        Raises:
            ValueError: ``hidden_size`` is odd or too small to split across axes.
        """
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be divisible by 2")
        if hidden_size // in_channels // 2 == 0:
            raise ValueError("hidden_size is too small for spatial RoPE")
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.register_buffer("freqs", self._base_freqs(), persistent=False)

    def _base_freqs(self) -> torch.Tensor:
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        return 1.0 / (10000**freqs)

    def reset_freqs(self, device: torch.device | str | None = None) -> None:
        """Recompute the frequencies (a non-persistent buffer) as at construction, on the host, and put them on
        ``device`` (default: where the module lives; the host for a module on the meta device)."""
        if device is None:
            device = self.freqs.device
        device = torch.device(device)
        self.freqs = self._base_freqs().to("cpu" if device.type == "meta" else device)

    @staticmethod
    def _update_freqs(
        freqs: torch.Tensor, train_length: int, infer_length: int
    ) -> torch.Tensor:
        wavelengths = freqs * train_length / (2 * math.pi)
        threshold = 24
        ratio = (wavelengths - 1) / (threshold - 1)
        ratio = ratio + (1 - ratio) * train_length / infer_length
        ratio = torch.where(wavelengths > threshold, 1.0, ratio)
        ratio = torch.where(wavelengths < 1, train_length / infer_length, ratio)
        return freqs * ratio

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        max_length = int(indices.max().item())
        if max_length < 64:
            freqs = self.freqs
        else:
            freqs = self._update_freqs(
                self.freqs, train_length=64, infer_length=(64 + max_length) // 2
            )
        phases = torch.outer(indices, freqs.to(indices.device))
        return torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def _rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        if accel.on("rope_inplace") and not torch.is_grad_enabled():
            rotated = x_complex.mul_(phases.unsqueeze(1))     # the same product, in place
        else:
            rotated = x_complex * phases.unsqueeze(1)
        return torch.view_as_real(rotated).reshape(*rotated.shape[:-1], -1).to(x.dtype)

    def phases(self, indices: torch.Tensor) -> torch.Tensor:
        """The complex rotations for ``indices`` (spatial positions ending in three
        coordinates), as ``forward`` applies them; they depend on the positions only."""
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        missing = self.hidden_size // 2 - phases.shape[-1]
        if missing > 0:
            padding = torch.polar(
                torch.ones(*phases.shape[:-1], missing, device=phases.device),
                torch.zeros(*phases.shape[:-1], missing, device=phases.device),
            )
            phases = torch.cat((phases, padding), dim=-1)
        return phases

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indices: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys at ``indices``.

        Args:
            q: Queries ending in ``[H, D]``.
            k: Keys with the same shape as ``q``.
            indices: Spatial positions ending in three coordinates.
            phases: ``phases(indices)``, when already computed.

        Returns:
            Rotated queries and keys.
        """
        if phases is None:
            phases = self.phases(indices)
        return (
            self._rotary_embedding(q, phases),
            self._rotary_embedding(k, phases),
        )
