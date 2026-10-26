"""Piece-streamed decode of :class:`PTv3MeshDecoder`, for cards with little memory.

Spellcast3D addition (not part of Hi3D v1.0).

The decoder's GroupNorms take one statistic over every point of a level, so a level cannot be cut
into independent tiles. Here the levels above the latent grid are kept sorted by (x, y, z) and swept
piece by piece (:func:`plan_pieces`): runs of whole x layers, and a layer whose rows alone exceed
the budget (a flat face across x) in strips of y lines. A sweep applies one sparse 3x3x3
convolution to a piece plus a halo of one cell on each side, and gathers the moments the next
GroupNorm needs, in row order. Between sweeps a level's tensors live on the card or in page-locked
memory (copied on side streams); an upsampled level's input is not stored but recomputed from its
parents whenever a sweep needs it.

Every point goes through the reference decoder's own operations in the same dtypes. On one card it
comes out with the same bits whatever the pieces: spconv's implicit GEMM gives a row the same result
in any piece, a convolution takes the chunked-GEMM path exactly when the whole level would, and the
cuBLAS products are padded to heights where their rounding no longer depends on the height (heights
measured on an RTX 3060; another card's cuBLAS may change kernels at other heights). The one
deliberate difference: GroupNorm moments are summed in float64 over a canonical split of the level
(``blocks.exact_group_norm``), where the reference sums float32 atomics in no fixed order.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import itertools
import mmap
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch

from sparc3d.models.autoencoders.blocks import (MOMENT_CHUNK_ROWS, exact_group_norm, group_sums,
                                                moment_chunk_sums)
from sparc3d.ops import sparse_backend
from sparc3d.structures.point import Point

DECODED_CHANNELS = 40
"""Columns of a decoded row: the 8 sign logits, then the 32 activated attributes."""


# ----------------------------------------------------------------------------------------------
# Levels, pieces and stores


Box = tuple[int, int, int, int]
"""``(x0, x1, y0, y1)``: the points of the layers ``[x0, x1)`` whose y lies in ``[y0, y1)``."""


@dataclass
class Level:
    """One resolution of the decode: its points sorted by (x, y, z), and where each x layer starts."""

    coords: torch.Tensor
    """``int32 [N, 3]`` grid coordinates on the card, sorted by (x, y, z)."""
    starts: list[int]
    """``starts[x]`` is the first row of layer ``x``; ``len(starts) == resolution + 1``."""
    resolution: int

    def __post_init__(self):
        self._lines: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.starts[-1]

    def line_starts(self) -> np.ndarray:
        """``int64 [R * R + 1]``: entry ``x * R + y`` is the first row of the line (x, y) (made when first
        needed: one pass over the coordinates and a host copy)."""
        if self._lines is None:
            r = self.resolution
            counts = torch.bincount(self.coords[:, 0].long() * r + self.coords[:, 1].long(), minlength=r * r)
            self._lines = np.concatenate([[0], np.cumsum(counts.cpu().numpy())])
        return self._lines

    def clamp(self, box: Box) -> Box:
        r = self.resolution
        x0, x1, y0, y1 = (min(max(v, 0), r) for v in box)
        return x0, max(x1, x0), y0, max(y1, y0)

    def ranges(self, box: Box) -> list[tuple[int, int]]:
        """The rows of ``box`` (clamped to the grid) as non-empty row ranges in order, adjacent ones joined."""
        x0, x1, y0, y1 = self.clamp(box)
        r = self.resolution
        if y0 == 0 and y1 == r:
            a, b = self.starts[x0], self.starts[x1]
            return [(a, b)] if b > a else []
        lines = self.line_starts()
        out: list[tuple[int, int]] = []
        for x in range(x0, x1):
            a, b = int(lines[x * r + y0]), int(lines[x * r + y1])
            if b <= a:
                continue
            if out and out[-1][1] == a:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return out

    def count(self, box: Box) -> int:
        return sum(b - a for a, b in self.ranges(box))

    @staticmethod
    def build(coords: torch.Tensor, resolution: int) -> "Level":
        """A level over coordinates already sorted by (x, y, z) (one host sync, for the layer starts)."""
        counts = torch.bincount(coords[:, 0].long(), minlength=resolution)
        starts = [0] + torch.cumsum(counts, 0).tolist()
        return Level(coords, starts, resolution)


def lattice_key(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """``(x * R + y) * R + z`` in int64: increasing in (x, y, z) order."""
    c = coords.long()
    return (c[:, 0] * resolution + c[:, 1]) * resolution + c[:, 2]


@dataclass(frozen=True)
class Piece:
    """A box of a level that a sweep makes in one go, from the rows of its window (the box grown by the
    sweep's halo; the grid's edges clamp it, so a piece spanning every y line keeps whole layers)."""

    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def box(self) -> Box:
        return self.x0, self.x1, self.y0, self.y1

    def window(self, below: int, above: int) -> Box:
        return self.x0 - below, self.x1 + above, self.y0 - below, self.y1 + above


FORCE_STRIP_LINES: int | None = None
"""For checks only: every layer is cut into strips of at most this many y lines (the result must not
change by a bit)."""


def plan_pieces(level: Level, max_rows: int, below: int = 1, above: int = 1, box: Box | None = None) -> list[Piece]:
    """Pieces covering every point of ``box`` (default: the whole level) in row order, the window of each
    (``below`` / ``above`` cells of halo) holding at most ``max_rows`` rows: runs of whole layers, and a
    layer whose window alone holds more in strips of y lines (a line beyond that is a strip of its own).
    No piece starts on an empty layer or line. In a box cut in y every piece is one layer. A piece's own
    rows are always one range."""
    r = level.resolution
    xa, xb, ya, yb = level.clamp(box if box is not None else (0, r, 0, r))
    runs = ya == 0 and yb == r and FORCE_STRIP_LINES is None
    pieces: list[Piece] = []
    x0 = xa
    while x0 < xb:
        if level.count((x0, x0 + 1, ya, yb)) == 0:
            x0 += 1
            continue
        x1 = x0 + 1
        if runs:
            while x1 < xb and level.count((x0 - below, x1 + 1 + above, 0, r)) <= max_rows:
                x1 += 1
        piece = Piece(x0, x1, ya, yb)
        if FORCE_STRIP_LINES is None and level.count(piece.window(below, above)) <= max_rows:
            pieces.append(piece)
        else:
            pieces += _strips(level, x0, ya, yb, max_rows, below, above)
        x0 = x1
    return pieces


def _strips(level: Level, x: int, ya: int, yb: int, max_rows: int, below: int, above: int) -> list[Piece]:
    """Layer ``x``'s lines ``[ya, yb)`` in strips whose windows hold at most ``max_rows`` rows."""
    r = level.resolution
    lines = level.line_starts()
    own = lines[x * r:x * r + r + 1]
    window = np.sum([lines[k * r:k * r + r + 1] for k in range(max(x - below, 0), min(x + 1 + above, r))], axis=0)
    pieces = []
    y0 = ya
    while y0 < yb:
        y0 = int(np.searchsorted(own, own[y0], side="right")) - 1        # the next line holding points
        if y0 >= yb:
            break
        limit = window[max(y0 - below, 0)] + max_rows
        if window[r] <= limit:
            y1 = yb
        else:
            y1 = min(yb, int(np.searchsorted(window, limit, side="right")) - 1 - above)
        y1 = max(y1, y0 + 1)
        if FORCE_STRIP_LINES is not None:
            y1 = min(y1, y0 + FORCE_STRIP_LINES)
        pieces.append(Piece(x, x + 1, y0, y1))
        y0 = y1
    return pieces


def offset_in(ranges: Sequence[tuple[int, int]], row: int) -> int:
    """Where ``row`` sits in the concatenation of ``ranges``."""
    at = 0
    for a, b in ranges:
        if a <= row < b:
            return at + row - a
        at += b - a
    raise ValueError(f"row {row} is not in {list(ranges)}")


def take(tensor: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """The rows of ``ranges``, concatenated (a view when they are one range)."""
    if len(ranges) == 1:
        a, b = ranges[0]
        return tensor[a:b]
    return torch.cat([tensor[a:b] for a, b in ranges]) if ranges else tensor[:0]


def piece_rows(level: Level, piece: Piece, below: int = 1, above: int = 1):
    """``(window ranges, (o0, o1) the piece's own rows, where they start in the window)``."""
    window = level.ranges(piece.window(below, above))
    (own,) = level.ranges(piece.box)
    return window, own, offset_in(window, own[0])


class Streams:
    """The side streams page-locked stores copy on, one per direction (reads overlap writes)."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.h2d = torch.cuda.Stream(self.device) if self.cuda else None
        self.d2h = torch.cuda.Stream(self.device) if self.cuda else None

    def barrier(self) -> None:
        """Later reads and compute see every write queued so far (no host sync)."""
        if self.cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.d2h)
            self.h2d.wait_stream(self.d2h)


class Pending:
    """Rows on their way to the card, in parts; :meth:`get` makes the compute stream wait for them and
    joins the parts."""

    def __init__(self, parts: list[torch.Tensor], event: torch.cuda.Event | None = None):
        self.parts, self.event = parts, event

    def get(self) -> torch.Tensor:
        if self.event is not None:
            compute = torch.cuda.current_stream(self.parts[0].device)
            compute.wait_event(self.event)
            for part in self.parts:
                part.record_stream(compute)
            self.event = None
        return self.parts[0] if len(self.parts) == 1 else torch.cat(self.parts)


_REGISTER_MIN_BYTES = 64 << 20
"""Host blocks at least this big are mapped from the OS and registered with the driver; smaller ones come
from PyTorch's pinned allocator (which rounds a block up to a power of two, a waste only on big ones)."""


@functools.lru_cache(maxsize=None)
def _cuda_runtime():
    """PyTorch's own CUDA runtime library (the one ``torch.cuda.cudart()`` calls), through ctypes, or None."""
    folder = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.name == "nt":
        names = sorted(glob.glob(os.path.join(folder, "cudart64_*.dll")))
    else:
        major = (torch.version.cuda or "12").split(".")[0]
        names = [f"libcudart.so.{major}"] + sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "cuda_runtime", "lib", "libcudart.so.*")))
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def clear_cuda_error() -> None:
    """Reset the CUDA runtime's last error after a failed call whose failure is handled here: left set,
    the next kernel-launch check anywhere would raise it (``torch.cuda.cudart()`` has no
    ``cudaGetLastError``)."""
    runtime = _cuda_runtime()
    if runtime is not None:
        runtime.cudaGetLastError()


def page_locked(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, mmap.mmap | None, str | None]:
    """A host tensor for a store: ``(tensor, the anonymous mapping under it when it is a big one, why it
    is not page-locked or None)``.

    Big blocks come straight from the OS: PyTorch's CPU allocator was seen to keep freed 1.5 GB tensors
    (neither given back nor reused), so a decode per pass would grow the process by its stores each time.
    The mapping must be unregistered and closed once no tensor uses it (:meth:`Store.free`). When memory
    cannot be page-locked (the page file, the driver), the tensor is plain pageable memory: its copies
    then run synchronously, which the streams' ordering allows."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    if nbytes < _REGISTER_MIN_BYTES:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True), None, None
        except RuntimeError as error:
            clear_cuda_error()
            return torch.empty(shape, dtype=dtype), None, f"pinned allocation failed ({error})"
    try:
        mapping = mmap.mmap(-1, nbytes)
    except OSError as error:
        return torch.empty(shape, dtype=dtype), None, f"no {nbytes >> 20} MiB mapping ({error})"
    tensor = torch.frombuffer(mapping, dtype=dtype).view(shape)
    code = int(torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), nbytes, 0))
    if code != 0:
        clear_cuda_error()
        return tensor, mapping, f"cudaHostRegister of {nbytes >> 20} MiB failed (CUDA error {code})"
    return tensor, mapping, None


class Store:
    """The rows of one level tensor, on the card or in (page-locked when possible) host memory."""

    def __init__(self, data: torch.Tensor, streams: Streams, device: torch.device, mapping: mmap.mmap | None = None,
                 registered: bool = False):
        self.data, self.streams, self.device, self.mapping = data, streams, torch.device(device), mapping
        self.registered = registered
        self.on_card = data.device.type == self.device.type
        self.pageable: str | None = None
        """Why a host store is not page-locked (None when it is)."""

    @staticmethod
    def empty(rows: int, channels: int, dtype: torch.dtype, on_card: bool, streams: Streams,
              device: torch.device) -> "Store":
        if on_card:
            return Store(torch.empty((rows, channels), dtype=dtype, device=device), streams, device)
        data, mapping, pageable = page_locked((rows, channels), dtype)
        store = Store(data, streams, device, mapping, registered=mapping is not None and pageable is None)
        store.pageable = pageable
        return store

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> Pending:
        """The rows of ``ranges``, concatenated, on the card (views until :meth:`Pending.get` when the store
        is on the card)."""
        if self.on_card:
            return Pending([self.data[a:b] for a, b in ranges] or [self.data[:0]])
        h2d = self.streams.h2d
        with torch.cuda.stream(h2d):
            rows = sum(b - a for a, b in ranges)
            out = torch.empty((rows,) + tuple(self.data.shape[1:]), dtype=self.data.dtype, device=self.device)
            at = 0
            for a, b in ranges:
                out[at:at + b - a].copy_(self.data[a:b], non_blocking=True)
                at += b - a
            event = torch.cuda.Event()
            event.record(h2d)
        return Pending([out], event)

    def write(self, a: int, values: torch.Tensor) -> None:
        """Rows ``[a, a + len(values))`` from a card tensor (queued; :meth:`Streams.barrier` orders reads)."""
        if self.on_card:
            self.data[a:a + len(values)].copy_(values)
            return
        d2h = self.streams.d2h
        d2h.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(d2h):
            self.data[a:a + len(values)].copy_(values, non_blocking=True)
        values.record_stream(d2h)

    def free(self) -> None:
        """Drop the rows; a mapped host block is unregistered and given back once its copies are done."""
        if self.mapping is not None:
            self.streams.h2d.synchronize()
            self.streams.d2h.synchronize()
            if self.registered:
                self.registered = False
                code = int(torch.cuda.cudart().cudaHostUnregister(self.data.data_ptr()))
                if code != 0:
                    clear_cuda_error()
                    warnings.warn(f"cudaHostUnregister failed (CUDA error {code}); the block is given back "
                                  "to the OS all the same", stacklevel=2)
            self.data = None
            mapping, self.mapping = self.mapping, None
            try:
                mapping.close()
            except BufferError:       # a view is still alive somewhere: the mapping goes with it
                pass
        self.data = None

    def __del__(self):
        if getattr(self, "mapping", None) is not None:
            try:
                self.free()
            except Exception:
                pass


def prefetched(pieces: Sequence, fetch: Callable) -> Iterator:
    """Yield ``(piece, fetched)``; the next piece's copies are queued before the current one is used."""
    if not pieces:
        return
    pending = fetch(pieces[0])
    for k, piece in enumerate(pieces):
        upcoming = fetch(pieces[k + 1]) if k + 1 < len(pieces) else None
        yield piece, pending
        pending = upcoming


# ----------------------------------------------------------------------------------------------
# The reference decoder's operations, on rows


class Moments:
    """Per-channel float64 sums of one level tensor, gathered slab by slab as its rows are made (in
    order): over the same chunks of :data:`MOMENT_CHUNK_ROWS` rows as ``moment_sums`` over the whole
    level, a chunk cut by a slab boundary being carried over, so the sums do not depend on the slabs."""

    def __init__(self, channels: int, device: torch.device):
        self.sums = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sums_sq = torch.zeros_like(self.sums)
        self.rows = 0
        self.carry: torch.Tensor | None = None

    def _chunk(self, chunk: torch.Tensor) -> None:
        sums, sums_sq = moment_chunk_sums(chunk)
        self.sums += sums
        self.sums_sq += sums_sq

    def add(self, feat: torch.Tensor) -> None:
        """The next rows of the level."""
        self.rows += feat.shape[0]
        if self.carry is not None:
            take = MOMENT_CHUNK_ROWS - self.carry.shape[0]
            self.carry = torch.cat([self.carry, feat[:take]])
            feat = feat[take:]
            if self.carry.shape[0] < MOMENT_CHUNK_ROWS:
                return
            self._chunk(self.carry)
            self.carry = None
        whole = feat.shape[0] - feat.shape[0] % MOMENT_CHUNK_ROWS
        for start in range(0, whole, MOMENT_CHUNK_ROWS):
            self._chunk(feat[start:start + MOMENT_CHUNK_ROWS])
        if whole < feat.shape[0]:
            self.carry = feat[whole:].clone()

    def _close(self) -> None:
        if self.carry is not None:
            self._chunk(self.carry)
            self.carry = None

    def stats(self, norm) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, inv_std)`` ``[1, G]`` in float32, by PointGroupNorm's formula for one batch element."""
        self._close()
        groups, eps = norm.group, norm.eps
        with torch.autocast("cuda", enabled=False):
            sums, sums_sq = group_sums(self.sums, groups), group_sums(self.sums_sq, groups)
            counts = torch.tensor([self.rows], device=sums.device).float().view(1, 1) * norm.channels_per_group
            mean = sums.div_(counts + eps)
            var = sums_sq.div_(counts + eps).sub_(mean.square())
            inv_std = torch.rsqrt(var + eps)
        return mean, inv_std


def group_norm(feat: torch.Tensor, norm, stats: tuple[torch.Tensor, torch.Tensor],
               out_dtype: torch.dtype) -> torch.Tensor:
    """PointGroupNorm.forward on rows with the level's statistics: float32 inside, ``out_dtype`` out
    (the same operations in the same order; in place on the float32 copy when there is one)."""
    mean, inv_std = stats
    rows, channels = feat.shape
    groups = norm.group
    with torch.autocast("cuda", enabled=False):
        x = feat.float()
        grouped = x.view(rows, groups, channels // groups)
        if x.data_ptr() == feat.data_ptr():
            grouped = grouped.sub(mean.view(1, groups, 1))
        else:
            grouped.sub_(mean.view(1, groups, 1))
        y = grouped.mul_(inv_std.view(1, groups, 1)).view(rows, channels)
        del x, grouped
        y.mul_(norm.weights).add_(norm.bias)
    return y if out_dtype == torch.float32 else y.to(out_dtype)


def swish(x: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(x)`` (the product commutes, so the bits are the reference Swish's)."""
    return torch.sigmoid(x).mul_(x)


def conv_takes_large_path(conv, level_rows: int, feat_dtype: torch.dtype) -> bool:
    """Whether SubMConv3d.forward takes its chunked-GEMM path for the whole level."""
    element = max(torch.empty((), dtype=feat_dtype).element_size(), conv.weight.element_size())
    return level_rows * conv.in_channels * element >= sparse_backend._SPCONV_INT32_BYTE_LIMIT


MIN_GEMM_ROWS = 4096
"""cuBLAS takes other kernels, with other rounding, for products of fewer rows. Measured on the RTX 3060:
every row of a product under 2048 rows differs from the same row in a 250 000-row product; the float16
Linears of the decoder agree from 4096 rows up. Shorter products are padded with zero rows."""

LARGE_MIN_GEMM_ROWS = 65_536
"""The same for the float32 products of the chunked-GEMM convolution path, whose shapes reach agreement
later: 4096 rows for 128->128 and 256->256, 8192 for 128->40, 16 384 for 128->8, 256->128, 512->256
and 512->512, 65 536 for 256->8 and 512->8 (RTX 3060)."""

_LARGE_CHUNK_ROWS = 250_000
"""SubMConv3d._large_forward's chunk of rows."""


def tall(fn: Callable, x: torch.Tensor, min_rows: int = MIN_GEMM_ROWS) -> torch.Tensor:
    """``fn(x)`` for a row-wise product, computed over at least ``min_rows`` rows."""
    rows = x.shape[0]
    if rows >= min_rows:
        return fn(x)
    padded = x.new_zeros((min_rows,) + tuple(x.shape[1:]))
    padded[:rows] = x
    return fn(padded)[:rows]


def large_conv(conv, tensor) -> torch.Tensor:
    """SubMConv3d._large_forward's arithmetic (centre product, then each offset pair and its mirror, then the
    bias) over a slab, its products padded to :data:`LARGE_MIN_GEMM_ROWS` rows. Returns the features."""
    from math import prod

    from spconv.pytorch import ops

    spconv = sparse_backend._spconv()
    _, pairs, pair_counts = ops.get_indice_pairs(
        tensor.indices, tensor.batch_size, tensor.spatial_shape, spconv.ConvAlgo.Native, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.output_padding, True, False)
    features = tensor.features
    kernel_volume = prod(conv.kernel_size)
    weights = conv.weight.reshape(conv.out_channels, kernel_volume, conv.in_channels)
    center = kernel_volume // 2
    output = torch.empty((features.shape[0], conv.out_channels), dtype=features.dtype, device=features.device)
    chunk = _LARGE_CHUNK_ROWS
    center_weight = weights[:, center, :].T
    for start in range(0, len(features), chunk):
        output[start:start + chunk] = tall(lambda x: x @ center_weight, features[start:start + chunk],
                                           LARGE_MIN_GEMM_ROWS)

    def accumulate(source, target, weight):
        # A submanifold pair map is one to one per offset: index_add_ adds to every target once per
        # call, so no two atomics meet and the sums keep the loop's order in every run.
        weight_t = weight.T
        for start in range(0, len(source), chunk):
            contribution = tall(lambda x: x @ weight_t, features[source[start:start + chunk]], LARGE_MIN_GEMM_ROWS)
            output.index_add_(0, target[start:start + chunk], contribution)

    counts = pair_counts.tolist()
    for offset in range(center):
        source = pairs[0, offset, :counts[offset]].long()
        target = pairs[1, offset, :counts[offset]].long()
        accumulate(source, target, weights[:, offset, :])
        accumulate(target, source, weights[:, kernel_volume - 1 - offset, :])
    if conv.bias is not None:
        output.add_(conv.bias)
    return output


def sparse_conv(conv, feats: torch.Tensor, coords: torch.Tensor, resolution: int, large: bool) -> torch.Tensor:
    """SubMConv3d.forward over the rows of a slab and its halo, on the path the whole level takes."""
    spconv = sparse_backend._spconv()
    feats_dtype = feats.dtype
    with torch.autocast("cuda", enabled=False):
        if feats_dtype != conv.weight.dtype:
            feats = feats.to(conv.weight.dtype)
        indices = torch.cat([coords.new_zeros((coords.shape[0], 1)), coords], dim=1).contiguous()
        tensor = spconv.SparseConvTensor(feats.contiguous(), indices, [resolution] * 3, 1)
        del feats
        if large:
            out = large_conv(conv, tensor)
        else:
            out = spconv.SubMConv3d.forward(conv, tensor).features
        del tensor
        if out.dtype != feats_dtype:
            out = out.to(feats_dtype)
    return out


class StoreSource:
    """A stored level tensor as a block's input."""

    def __init__(self, store: Store):
        self.store = store

    @property
    def dtype(self) -> torch.dtype:
        return self.store.dtype

    def fetch(self, level: Level, box: Box):
        return self.store.fetch(level.ranges(box))

    def get(self, fetched) -> torch.Tensor:
        return fetched.get()


def sorted_children(parts, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The children of every piece of an upsample, ``(coords, src)`` in (x, y, z) order. ``parts``: per
    piece, in order, ``(layer if a strip else None, coords, src)`` sorted within the piece; a layer's
    strips interleave (each holds children in both of the layer's child layers), so they are sorted
    together."""
    coords, src = [], []
    for layer, group in itertools.groupby(parts, key=lambda part: part[0]):
        group = list(group)
        if layer is None or len(group) == 1:
            coords += [part[1] for part in group]
            src += [part[2] for part in group]
            continue
        c, s = torch.cat([part[1] for part in group]), torch.cat([part[2] for part in group])
        order = torch.argsort(lattice_key(c, resolution))
        coords.append(c[order])
        src.append(s[order])
        del c, s, order
    return torch.cat(coords), torch.cat(src)


PROJ_CHUNK_BYTES = 96 << 20
"""Upsample projections are computed this many output bytes at a time."""


class ChildSource:
    """An upsampled level's input, recomputed from the parents: ``proj(parent)`` viewed as one row per
    child, the kept children picked by ``src`` (parent row * 8 + child index)."""

    def __init__(self, parent: Level, store: Store, src: torch.Tensor, proj, channels: int):
        self.parent, self.store, self.src, self.proj, self.channels = parent, store, src, proj, channels
        self.dtype: torch.dtype | None = None
        """The projection's output dtype, known once a row was made."""

    def fetch(self, level: Level, box: Box):
        """Queue the copies of the parents of ``box`` (a box of the child level)."""
        xa, xb, ya, yb = level.clamp(box)
        parent_box = (xa // 2, (xb + 1) // 2, ya // 2, (yb + 1) // 2)
        parents = self.parent.ranges(parent_box)
        return level, (xa, xb, ya, yb), parent_box, parents, self.store.fetch(parents)

    def get(self, fetched) -> torch.Tensor:
        """The rows of the box, in order. The projection runs over pieces of the parents (whole lines,
        whose children are whole lines of the child level), :data:`PROJ_CHUNK_BYTES` of output at a time."""
        level, (xa, xb, ya, yb), parent_box, parents, pending = fetched
        rows = pending.get()
        children = level.ranges((xa, xb, ya, yb))
        out = None
        channels = self.channels
        for unit in plan_pieces(self.parent, max(1, PROJ_CHUNK_BYTES // (16 * channels)), 0, 0, parent_box):
            ((s0, s1),) = self.parent.ranges(unit.box)
            at = offset_in(parents, s0)
            x = tall(self.proj, rows[at:at + s1 - s0]).view(-1, channels)
            if out is None:
                self.dtype = x.dtype
                out = torch.empty((sum(b - a for a, b in children), channels), dtype=x.dtype, device=rows.device)
            box = (max(xa, 2 * unit.x0), min(xb, 2 * unit.x1), max(ya, 2 * unit.y0), min(yb, 2 * unit.y1))
            for c0, c1 in level.ranges(box):
                at = offset_in(children, c0)
                out[at:at + c1 - c0] = x[self.src[c0:c1].long() - s0 * 8]
            del x
        if out is None:
            out = torch.empty((0, channels), dtype=self.dtype or rows.dtype, device=rows.device)
        return out


# ----------------------------------------------------------------------------------------------
# The decode


@dataclass
class DecodedField:
    """The decoder's output in (x, y, z) order: per point 8 sign logits and 32 activated attributes."""

    level: Level
    store: Store
    matrix: torch.Tensor

    @property
    def n(self) -> int:
        return self.level.n

    def to_point(self) -> Point:
        """The whole field as the reference decode's Point (on the card), for comparisons."""
        rows = self.store.fetch([(0, self.n)]).get()
        point = Point({"feat": rows[:, 8:], "sdf_sign": rows[:, :8], "grid_coord": self.level.coords.long(),
                       "resolution": self.level.resolution, "matrix": self.matrix,
                       "batch": torch.zeros(self.n, dtype=torch.long, device=rows.device)})
        start, layouts = 0, {}
        for key in ("sdf", "deform"):
            entry = Point.ALL_LAYOUTS[key]
            layouts[key] = {**entry, "range": (start, start + entry["size"])}
            start += entry["size"]
        point["layouts"] = layouts
        return point

    def free(self) -> None:
        self.store.free()


class SlabPlan:
    """How a decode is cut: rows per piece (halo included) and where stores live.

    The fixed plan: ``max_rows`` for every sweep, every store on the card. :class:`BudgetPlan` (in
    ``sparc3d.pipelines.streamed_decode``) sizes them from a memory budget instead.
    """

    release_above = 0
    """Cached blocks are given back before a sweep or a store once PyTorch reserves more than this many
    bytes (a fresh block per store and sweep keeps the reserve near what is allocated)."""

    def __init__(self, max_rows: int = 1 << 20, stores_on_card: bool = True):
        self.max_rows, self.stores_on_card = max_rows, stores_on_card

    def slab_rows(self, kind: str, index: int, channels_in: int, host_row_bytes: int = 0) -> int:
        """Rows per piece for a sweep (``kind``: resnet1, resnet2, upsample, project, conv_out or extract)
        whose input has ``channels_in`` channels; ``host_row_bytes`` per row arrive from page-locked stores."""
        return self.max_rows

    def store_on_card(self, name: str, index: int, nbytes: int, kind: str, channels_in: int) -> bool:
        """Whether a store goes on the card, made for a sweep of ``kind`` over ``channels_in`` channels."""
        return self.stores_on_card


class CardPeaks:
    """The card's peaks over a stretch of work, without resetting PyTorch's counters (the caller may be
    measuring with them). PyTorch's reserve only shrinks when the cache is emptied, so its largest value
    sampled before every :meth:`empty_cache` and at the end is the stretch's peak; when the stretch set a
    new process peak, the counters themselves give it (the allocated peak exactly, else the largest
    allocation sampled: a lower bound)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.counters = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
        self.alloc = self.reserved = 0
        self.sample()

    def sample(self) -> None:
        self.alloc = max(self.alloc, torch.cuda.memory_allocated(self.device))
        self.reserved = max(self.reserved, torch.cuda.memory_reserved(self.device))

    def empty_cache(self) -> None:
        self.sample()
        torch.cuda.empty_cache()

    def result(self) -> tuple[int, int]:
        """``(allocated, reserved)`` peak bytes since this was made."""
        self.sample()
        alloc, reserved = torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)
        return (alloc if alloc > self.counters[0] else self.alloc,
                reserved if reserved > self.counters[1] else self.reserved)


class StreamedDecoder:
    """Decode latents piece by piece (see the module docstring).

    ``trace``: time every sweep and record its card peaks, which synchronizes the card after each sweep
    and resets PyTorch's process-wide peak counters (for profiling only). Otherwise the card's peaks over
    the decode come from ``peaks``.
    """

    def __init__(self, decoder, plan: SlabPlan, device: torch.device, trace: bool = False,
                 peaks: CardPeaks | None = None):
        self.decoder, self.plan, self.device, self.trace = decoder, plan, device, trace
        self.peaks = peaks or CardPeaks(device)
        self.streams = Streams(device)
        self.sweeps: list[dict] = []
        self.host_bytes = self.host_peak = 0
        self.peak_alloc = self.peak_reserved = 0.0
        self.pageable_warned = False

    def _store(self, name: str, index: int, rows: int, channels: int, dtype: torch.dtype, kind: str,
               channels_in: int) -> Store:
        nbytes = rows * channels * torch.empty((), dtype=dtype).element_size()
        self._release()               # a long-lived block of its own, not one split out of cached ones
        on_card = bool(self.plan.store_on_card(name, index, nbytes, kind, channels_in))
        if not on_card:
            self.host_bytes += nbytes
            self.host_peak = max(self.host_peak, self.host_bytes)
        store = Store.empty(rows, channels, dtype, on_card, self.streams, self.device)
        if store.pageable and not self.pageable_warned:
            self.pageable_warned = True
            print(f"[sparc3d] streamed decode: host memory could not be page-locked ({store.pageable}); "
                  "stores off the card use pageable memory, their copies run synchronously", file=sys.stderr,
                  flush=True)
        return store

    def _release(self) -> None:
        if torch.cuda.memory_reserved(self.device) > self.plan.release_above:
            self.peaks.empty_cache()

    def _free(self, store: Store) -> None:
        if not store.on_card and store.data is not None:
            self.host_bytes -= store.nbytes
        store.free()

    def _begin(self, record: dict) -> None:
        """Start a sweep's record (traced: its clock, and the card's peak counters reset)."""
        if self.trace:
            torch.cuda.synchronize(self.device)
            record["t0"] = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sweeps.append(record)

    def _pieces(self, kind: str, index: int, level: Level, channels_in: int, inputs: Sequence[Store],
                halo: int = 1) -> list[Piece]:
        """The sweep's pieces, and its record. ``inputs``: the stores it reads; those in page-locked memory
        cost two pieces of copies on the card (current and next)."""
        self._release()               # the previous sweep's cached blocks would fragment this one's
        host = sum(2 * store.data.shape[1] * store.data.element_size() for store in inputs if not store.on_card)
        rows = self.plan.slab_rows(kind, index, channels_in, host)
        pieces = plan_pieces(level, rows, halo, halo)
        self._begin({"sweep": kind, "level": index, "rows": level.n, "slab_rows": rows, "slabs": len(pieces),
                     "strips": sum((p.y0, p.y1) != (0, level.resolution) for p in pieces),
                     "host_row_bytes": host,
                     "widest": max((level.count(p.window(halo, halo)) for p in pieces), default=0),
                     "alloc_before_mib": torch.cuda.memory_allocated(self.device) / 2**20})
        return pieces

    def _end_sweep(self) -> None:
        self.peaks.sample()
        if not self.trace:
            return
        record = self.sweeps[-1]
        torch.cuda.synchronize(self.device)
        record["seconds"] = round(time.perf_counter() - record.pop("t0"), 3)
        record["peak_alloc_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        self.peak_alloc = max(self.peak_alloc, record["peak_alloc_mib"])
        self.peak_reserved = max(self.peak_reserved, record["peak_reserved_mib"])

    # -- the latent level: the reference modules on every point

    def _latent_level(self, coords: torch.Tensor, feats: torch.Tensor, resolution: int):
        grid_coord = coords[:, 1:].int()
        point = Point({"grid_coord": grid_coord, "feat": feats.float(), "sdf": None, "deform": None,
                       "resolution": resolution,
                       "offset": torch.tensor([grid_coord.shape[0]], dtype=torch.int64, device=grid_coord.device)})
        point.initialization()
        decoder = self.decoder
        with exact_group_norm():
            point.sparsify()
            point = decoder.z_to_f(point)
            point.serialization(order=decoder.order)
            point = decoder.mid(point)
            stage = decoder.dec._modules[f"dec{decoder.num_stages - 1}"]
            for name, block in stage._modules.items():
                if name != "up":
                    point = block(point)
        order = torch.argsort(lattice_key(point.grid_coord, resolution))
        level = Level.build(point.grid_coord[order].int().contiguous(), resolution)
        feat, matrix = point.feat[order], point.matrix
        del point, order
        return level, feat, matrix

    # -- sweeps

    def _resnet(self, index: int, level: Level, block, source, source_moments: Moments,
                in_place: Store | None) -> tuple[Store, Moments]:
        """A SparseResnetBlock as two sweeps; the output goes to ``in_place`` when given (its input)."""
        modules = list(block.block._modules.values())
        norm1, conv1, norm2, conv2 = modules[0], modules[2], modules[3], modules[5]
        dtype = source.dtype
        channels = conv2.out_channels
        moments = Moments(channels, self.device)
        if level.n > block.skip_points:
            warnings.warn(f"SparseResnetBlock skipped: {level.n} points exceeds the skip_points budget of "
                          f"{block.skip_points}. The block is a no-op, so the output is the input.", stacklevel=2)
            out = in_place or self._store("resnet_out", index, level.n, channels, dtype, "resnet2",
                                          conv1.in_channels)
            pieces = self._pieces("resnet2", index, level, conv1.in_channels, [source.store], halo=0)
            for piece, fetched in prefetched(pieces, lambda p: source.fetch(level, p.box)):
                x = source.get(fetched)
                del fetched
                out.write(level.ranges(piece.box)[0][0], x)
                moments.add(x)
                del x
            self.streams.barrier()
            self._end_sweep()
            return out, moments

        # conv1 over GroupNorm + Swish of the input
        hidden = self._store("hidden", index, level.n, conv1.out_channels, dtype, "resnet1", conv1.in_channels)
        hidden_moments = Moments(conv1.out_channels, self.device)
        stats = source_moments.stats(norm1)
        large = conv_takes_large_path(conv1, level.n, dtype)
        pieces = self._pieces("resnet1", index, level, conv1.in_channels, [source.store])
        for piece, fetched in prefetched(pieces, lambda p: source.fetch(level, p.window(1, 1))):
            window, (o0, o1), at = piece_rows(level, piece)
            y = swish(group_norm(source.get(fetched), norm1, stats, dtype))
            del fetched
            h = sparse_conv(conv1, y, take(level.coords, window), level.resolution, large)[at:at + o1 - o0]
            del y
            hidden.write(o0, h)
            hidden_moments.add(h)
            del h
        self.streams.barrier()
        self._end_sweep()

        # conv2 over GroupNorm + Swish of the hidden rows, plus the shortcut
        out = in_place or self._store("resnet_out", index, level.n, channels, dtype, "resnet2", conv1.in_channels)
        stats = hidden_moments.stats(norm2)
        large = conv_takes_large_path(conv2, level.n, dtype)
        pieces = self._pieces("resnet2", index, level, conv1.in_channels, [hidden, source.store])

        def fetch(p):
            return hidden.fetch(level.ranges(p.window(1, 1))), source.fetch(level, p.box)

        for piece, (fetched_h, fetched_x) in prefetched(pieces, fetch):
            window, (o0, o1), at = piece_rows(level, piece)
            y = swish(group_norm(fetched_h.get(), norm2, stats, dtype))
            del fetched_h
            c = sparse_conv(conv2, y, take(level.coords, window), level.resolution, large)[at:at + o1 - o0]
            del y
            x = source.get(fetched_x)
            del fetched_x
            shortcut = tall(block.res_conv, x) if block.res_conv is not None else x
            del x
            r = shortcut + c
            del shortcut, c
            out.write(o0, r)
            moments.add(r)
            del r
        self.streams.barrier()
        self._end_sweep()
        self._free(hidden)
        return out, moments

    def _upsample(self, index: int, level: Level, up, store: Store, moments: Moments):
        """Upsample's occupancy head over every piece, the kept children sorted by (x, y, z); then the
        moments of their input, recomputed piece by piece.

        Returns the child level, its input (recomputed from ``store``) and that input's moments.
        """
        unpool = up.upsample
        norm, conv = unpool.cls_head.fc._modules["0"], unpool.cls_head.fc._modules["2"]
        channels = unpool.proj.in_features
        stats = moments.stats(norm)
        large = conv_takes_large_path(conv, level.n, torch.float32)
        child_resolution = level.resolution * 2
        offsets = unpool.inv_grid_coord.to(torch.int32)
        parts = []
        pieces = self._pieces("upsample", index, level, channels, [store])
        for piece, fetched in prefetched(pieces, lambda p: store.fetch(level.ranges(p.window(1, 1)))):
            window, (o0, o1), at = piece_rows(level, piece)
            with torch.autocast("cuda", enabled=False):
                y = swish(group_norm(fetched.get(), norm, stats, torch.float32))
                del fetched
                logits = sparse_conv(conv, y, take(level.coords, window), level.resolution, large)[at:at + o1 - o0]
                del y
            local = torch.nonzero(logits.flatten() > 0).squeeze(1)
            del logits
            child = level.coords[local // 8 + o0] * 2 + offsets[local % 8]
            order = torch.argsort(lattice_key(child, child_resolution))
            strip = piece.x0 if (piece.y0, piece.y1) != (0, level.resolution) else None
            parts.append((strip, child[order], (local[order] + o0 * 8).to(torch.int32)))
            del child, order, local
        self._end_sweep()
        coords, src = sorted_children(parts, child_resolution)
        del parts
        if coords.shape[0] == 0:
            raise RuntimeError("the decode's occupancy is empty: no surface to extract")
        child_level = Level.build(coords, child_resolution)
        source = ChildSource(level, store, src, unpool.proj, channels)

        child_moments = Moments(channels, self.device)
        pieces = self._pieces("project", index, child_level, channels, [store], halo=0)
        for piece, fetched in prefetched(pieces, lambda p: source.fetch(child_level, p.box)):
            x = source.get(fetched)
            del fetched
            child_moments.add(x)
            del x
        self._end_sweep()
        return child_level, source, child_moments

    def _conv_out(self, index: int, level: Level, conv_out, store: Store, moments: Moments) -> Store:
        norm, conv = conv_out.conv._modules["0"], conv_out.conv._modules["2"]
        stats = moments.stats(norm)
        large = conv_takes_large_path(conv, level.n, torch.float32)
        decoded = self._store("decoded", index, level.n, DECODED_CHANNELS, torch.float32, "conv_out",
                              conv.in_channels)
        pieces = self._pieces("conv_out", index, level, conv.in_channels, [store])
        for piece, fetched in prefetched(pieces, lambda p: store.fetch(level.ranges(p.window(1, 1)))):
            window, (o0, o1), at = piece_rows(level, piece)
            with torch.autocast("cuda", enabled=False):
                y = swish(group_norm(fetched.get(), norm, stats, torch.float32))
                del fetched
                out = sparse_conv(conv, y, take(level.coords, window), level.resolution, large)[at:at + o1 - o0]
                del y
                out = torch.cat([out[:, :8], torch.tanh(out[:, 8:])], dim=1)
            decoded.write(o0, out)
            del out
        self.streams.barrier()
        self._end_sweep()
        return decoded

    # -- the whole decode

    def __call__(self, coords: torch.Tensor, feats: torch.Tensor, resolution: int) -> DecodedField:
        """Decode latents (``coords`` ``[N, 4]`` as batch, x, y, z; ``feats`` denormalised ``[N, 32]``)."""
        decoder = self.decoder
        with torch.inference_mode(), torch.autocast("cuda"):
            self._release()           # what earlier work left cached would sit under the latent level
            self._begin({"sweep": "latent", "level": 0, "rows": int(coords.shape[0]),
                         "alloc_before_mib": torch.cuda.memory_allocated(self.device) / 2**20})
            level, feat, matrix = self._latent_level(coords, feats, resolution)
            self._end_sweep()
            store = Store(feat, self.streams, self.device)
            moments = Moments(feat.shape[1], self.device)
            moments.add(feat)
            del feat
            stage = decoder.dec._modules[f"dec{decoder.num_stages - 1}"]
            level, source, source_moments = self._upsample(0, level, stage.up, store, moments)
            index = 0
            for s in reversed(range(decoder.num_stages - 1)):
                index += 1
                parent_store = store
                stage = decoder.dec._modules[f"dec{s}"]
                blocks = [m for name, m in stage._modules.items() if name != "up"]
                current, current_moments = source, source_moments
                for k, block in enumerate(blocks):
                    in_place = None
                    if isinstance(current, StoreSource) and block.res_conv is None:
                        in_place = current.store
                    store, current_moments = self._resnet(index, level, block, current, current_moments, in_place)
                    if k == 0:
                        self._free(parent_store)
                        source = None
                    current = StoreSource(store)
                moments = current_moments
                del current, current_moments
                if "up" in stage._modules:
                    level, source, source_moments = self._upsample(index, level, stage.up, store, moments)
            decoded = self._conv_out(index, level, decoder.conv_out, store, moments)
            self._free(store)
        return DecodedField(level, decoded, matrix)
