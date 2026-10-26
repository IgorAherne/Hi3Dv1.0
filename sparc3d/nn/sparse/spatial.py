"""Paired average downsampling and cached nearest-neighbor upsampling."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for SPARC3D to retain only paired downsample and upsample operations.
# Modified for Spellcast3D: the downsample's mapping kept in the spatial cache (the coordinates of a pass are
# fixed), the upsample without re-checking the layout it recorded.

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from sparc3d.nn import accel

from .basic import SparseTensor


def _factor_tuple(factor: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    values = tuple(factor) if isinstance(factor, Sequence) else (factor,) * dimensions
    if len(values) != dimensions or any(value <= 0 for value in values):
        raise ValueError("factor must provide one positive value per spatial dimension")
    return values


class SparseDownsample(nn.Module):
    """Average features that collapse onto the same downsampled coordinate."""

    def __init__(self, factor: int | Sequence[int]):
        """
        Args:
            factor: Per-axis downsampling factor, or one factor for every axis.
        """
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Downsample ``input`` and register the inverse mapping."""
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        key = f"downsample_{factor}"
        mapping = input.get_spatial_cache(key) if accel.on("downsample") else None
        if mapping is None:
            new_coords, inverse, layout = self._mapping(input, factor, dimensions)
        else:
            new_coords, inverse, layout = mapping
        if accel.ENABLED and accel.DETERMINISTIC:
            new_feats = self._serial_mean(input, inverse, new_coords.shape[0], factor)
        else:
            new_feats = torch.scatter_reduce(
                torch.zeros(
                    new_coords.shape[0],
                    input.feats.shape[1],
                    device=input.device,
                    dtype=input.dtype,
                ),
                dim=0,
                index=inverse[:, None].expand(-1, input.feats.shape[1]),
                src=input.feats,
                reduce="mean",
            )
        output = SparseTensor(
            new_feats,
            new_coords,
            shape=input.shape,
            layout=layout,
            scale=tuple(scale * value for scale, value in zip(input.scale, factor)),
            spatial_cache=input._spatial_cache,
            trusted=layout is not None,
        )
        if accel.on("downsample") and mapping is None:
            input.register_spatial_cache(key, (new_coords, inverse, output.layout))
        output.register_spatial_cache(f"upsample_{factor}_coords", input.coords)
        output.register_spatial_cache(f"upsample_{factor}_layout", input.layout)
        output.register_spatial_cache(f"upsample_{factor}_idx", inverse)
        return output

    @staticmethod
    def _serial_mean(
        input: SparseTensor, inverse: torch.Tensor, outputs: int, factor: tuple[int, ...]
    ) -> torch.Tensor:
        """``scatter_reduce(zeros, inverse, feats, "mean")`` with its sums in one fixed order.

        The CUDA scatter adds each row into its output with atomics, in whatever order the
        threads arrive, rounding after each add (fp16 here): its last bits change from run
        to run. This adds each output's rows in the order of the input rows, rounding after
        each add as a serial scatter does, then divides by the count plus one (the mean
        counts the zero it starts from): one of the original's own results, every time.
        """
        key = f"downsample_slots_{factor}"
        slots = input.get_spatial_cache(key)
        if slots is None:
            rows = inverse.shape[0]
            order = torch.argsort(inverse, stable=True)
            grouped = inverse[order]
            counts = torch.bincount(inverse, minlength=outputs)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(rows, device=inverse.device) - starts[grouped]
            width = int(counts.max()) if rows else 1
            slots = torch.full((outputs, width), rows, dtype=torch.long, device=inverse.device)
            slots[grouped, rank] = order
            divisor = (counts + 1).to(input.dtype)[:, None]
            slots = (slots, divisor)
            input.register_spatial_cache(key, slots)
        slots, divisor = slots
        padded = torch.cat((input.feats, input.feats.new_zeros((1, input.feats.shape[1]))))
        total = padded[slots[:, 0]] + 0.0      # from +0, as the scatter starts: -0 + 0 is +0
        for column in range(1, slots.shape[1]):
            total = total + padded[slots[:, column]]
        return total / divisor

    @staticmethod
    def _mapping(
        input: SparseTensor, factor: tuple[int, ...], dimensions: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[slice] | None]:
        """Downsampled coordinates, each input row's output row, and (unknown yet) the layout."""
        coordinate_columns = list(input.coords.unbind(dim=-1))
        for axis, value in enumerate(factor):
            coordinate_columns[axis + 1] = coordinate_columns[axis + 1] // value

        extents = [
            int(coordinate_columns[axis + 1].max().item()) + 1
            for axis in range(dimensions)
        ]
        offsets = torch.cumprod(torch.tensor(extents[::-1]), 0).tolist()[::-1] + [1]
        codes = sum(
            coordinate * offset
            for coordinate, offset in zip(coordinate_columns, offsets)
        )
        unique_codes, inverse = codes.unique(return_inverse=True)
        new_coords = torch.stack(
            [unique_codes // offsets[0]]
            + [
                (unique_codes // offsets[axis + 1]) % extents[axis]
                for axis in range(dimensions)
            ],
            dim=-1,
        )
        return new_coords, inverse, None


class SparseUpsample(nn.Module):
    """Restore coordinates from a paired downsample cache."""

    def __init__(self, factor: int | Sequence[int]):
        """
        Args:
            factor: Per-axis factor, matching the paired :class:`SparseDownsample`.
        """
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Upsample ``input`` through the recorded inverse mapping.

        Raises:
            ValueError: ``input`` is not at a scale this factor could have produced,
                or no paired :class:`SparseDownsample` registered a cache for it.
        """
        coords, layout, inverse, scale = self.mapping(input)
        return SparseTensor(
            input.feats[inverse],
            coords,
            shape=input.shape,
            layout=layout,
            scale=scale,
            spatial_cache=input._spatial_cache,
            trusted=True,
        )

    def mapping(
        self, input: SparseTensor
    ) -> tuple[torch.Tensor, list[slice], torch.Tensor, tuple[int, ...]]:
        """The upsampled coordinates, their layout, each output row's input row, and the scale."""
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        if any(scale % value for scale, value in zip(input.scale, factor)):
            raise ValueError(
                f"cannot upsample by {factor} from scale {input.scale}; "
                "SparseUpsample must mirror its paired SparseDownsample"
            )
        coords = input.get_spatial_cache(f"upsample_{factor}_coords")
        layout = input.get_spatial_cache(f"upsample_{factor}_layout")
        inverse = input.get_spatial_cache(f"upsample_{factor}_idx")
        if coords is None or layout is None or inverse is None:
            raise ValueError(
                "upsample cache not found; SparseUpsample must follow its "
                "paired SparseDownsample"
            )
        return (
            coords,
            layout,
            inverse,
            tuple(scale // value for scale, value in zip(input.scale, factor)),
        )
