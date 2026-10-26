"""Slab-streamed decode of :class:`PTv3MeshDecoder`, for cards with little memory.

Spellcast3D addition (not part of Hi3D v1.0).

The decoder's GroupNorms take one statistic over every point of a level, so a level cannot be cut
into independent tiles. Here the levels above the latent grid are kept sorted by (x, y, z) and swept
in slabs of whole x layers: a sweep applies one sparse 3x3x3 convolution to a slab plus one halo
layer on each side, and gathers the moments the next GroupNorm needs. Between sweeps a level's
tensors live on the card or in page-locked memory (copied on side streams); an upsampled level's
input is not stored but recomputed from its parents whenever a sweep needs it.

Every point goes through the reference decoder's own operations in the same dtypes, and comes out
with the same bits whatever the slabs: spconv's implicit GEMM gives a row the same result in any
slab, a convolution takes the chunked-GEMM path exactly when the whole level would, and the cuBLAS
products are padded to heights where their rounding no longer depends on the height. The one
deliberate difference: GroupNorm moments are summed in float64 over a canonical split of the level
(``blocks.exact_group_norm``), where the reference sums float32 atomics in no fixed order.
"""

from __future__ import annotations

import bisect
import mmap
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch

from sparc3d.models.autoencoders.blocks import (MOMENT_CHUNK_ROWS, exact_group_norm, group_sums,
                                                moment_chunk_sums)
from sparc3d.ops import sparse_backend
from sparc3d.structures.point import Point

DECODED_CHANNELS = 40
"""Columns of a decoded row: the 8 sign logits, then the 32 activated attributes."""


# ----------------------------------------------------------------------------------------------
# Levels, slabs and stores


@dataclass
class Level:
    """One resolution of the decode: its points sorted by (x, y, z), and where each x layer starts."""

    coords: torch.Tensor
    """``int32 [N, 3]`` grid coordinates on the card, sorted by (x, y, z)."""
    starts: list[int]
    """``starts[x]`` is the first row of layer ``x``; ``len(starts) == resolution + 1``."""
    resolution: int

    @property
    def n(self) -> int:
        return self.starts[-1]

    def rows(self, x0: int, x1: int) -> tuple[int, int]:
        """Rows of the layers ``[x0, x1)``, clamped to the grid."""
        x0 = min(max(x0, 0), self.resolution)
        x1 = min(max(x1, x0), self.resolution)
        return self.starts[x0], self.starts[x1]

    def layer_of(self, row: int) -> int:
        return bisect.bisect_right(self.starts, row) - 1

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


def plan_slabs(level: Level, max_rows: int, halo: int = 1) -> list[tuple[int, int]]:
    """Layer ranges ``[x0, x1)`` covering every point, each holding at most ``max_rows`` rows with its
    ``halo`` layers on both sides (a single layer bigger than that is a slab of its own)."""
    starts, size = level.starts, level.resolution
    occupied = [x for x in range(size) if starts[x + 1] > starts[x]]
    if not occupied:
        return []
    x0, last = occupied[0], occupied[-1] + 1
    slabs = []
    while x0 < last:
        x1 = x0 + 1
        while x1 < last and starts[min(x1 + 1 + halo, size)] - starts[max(x0 - halo, 0)] <= max_rows:
            x1 += 1
        slabs.append((x0, x1))
        x0 = x1
    return slabs


class Streams:
    """The side streams page-locked stores copy on, one per direction (reads overlap writes)."""

    def __init__(self, device: torch.device):
        self.cuda = torch.device(device).type == "cuda"
        self.h2d = torch.cuda.Stream(device) if self.cuda else None
        self.d2h = torch.cuda.Stream(device) if self.cuda else None

    def barrier(self) -> None:
        """Later reads and compute see every write queued so far (no host sync)."""
        if self.cuda:
            torch.cuda.current_stream().wait_stream(self.d2h)
            self.h2d.wait_stream(self.d2h)


class Pending:
    """Rows on their way to the card; :meth:`get` makes the compute stream wait for them."""

    def __init__(self, tensor: torch.Tensor, event: torch.cuda.Event | None = None):
        self.tensor, self.event = tensor, event

    def get(self) -> torch.Tensor:
        if self.event is not None:
            compute = torch.cuda.current_stream()
            compute.wait_event(self.event)
            self.tensor.record_stream(compute)
            self.event = None
        return self.tensor


_REGISTER_MIN_BYTES = 64 << 20
"""Host blocks at least this big are mapped from the OS and registered with the driver; smaller ones come
from PyTorch's pinned allocator (which rounds a block up to a power of two, a waste only on big ones)."""


def page_locked(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, mmap.mmap | None]:
    """A page-locked host tensor, and the anonymous mapping under it when it is a big one.

    Big blocks come straight from the OS: PyTorch's CPU allocator was seen to keep freed 1.5 GB tensors
    (neither given back nor reused), so a decode per pass would grow the process by its stores each time.
    The mapping must be unregistered and closed once no tensor uses it (:meth:`Store.free`)."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    if nbytes < _REGISTER_MIN_BYTES:
        return torch.empty(shape, dtype=dtype, pin_memory=True), None
    mapping = mmap.mmap(-1, nbytes)
    tensor = torch.frombuffer(mapping, dtype=dtype).view(shape)
    if int(torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), nbytes, 0)) != 0:
        del tensor
        mapping.close()
        raise RuntimeError(f"could not page-lock {nbytes >> 20} MiB of host memory")
    return tensor, mapping


class Store:
    """The rows of one level tensor, on the card or in page-locked host memory."""

    def __init__(self, data: torch.Tensor, streams: Streams, device: torch.device, mapping: mmap.mmap | None = None):
        self.data, self.streams, self.device, self.mapping = data, streams, torch.device(device), mapping
        self.on_card = data.device.type == self.device.type

    @staticmethod
    def empty(rows: int, channels: int, dtype: torch.dtype, on_card: bool, streams: Streams,
              device: torch.device) -> "Store":
        if on_card:
            return Store(torch.empty((rows, channels), dtype=dtype, device=device), streams, device)
        data, mapping = page_locked((rows, channels), dtype)
        return Store(data, streams, device, mapping)

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def fetch(self, a: int, b: int) -> Pending:
        """Rows ``[a, b)`` on the card (a view when the store is on the card)."""
        if self.on_card:
            return Pending(self.data[a:b])
        h2d = self.streams.h2d
        with torch.cuda.stream(h2d):
            out = torch.empty((b - a,) + tuple(self.data.shape[1:]), dtype=self.data.dtype, device=self.device)
            out.copy_(self.data[a:b], non_blocking=True)
            event = torch.cuda.Event()
            event.record(h2d)
        return Pending(out, event)

    def write(self, a: int, values: torch.Tensor) -> None:
        """Rows ``[a, a + len(values))`` from a card tensor (queued; :meth:`Streams.barrier` orders reads)."""
        if self.on_card:
            self.data[a:a + len(values)].copy_(values)
            return
        d2h = self.streams.d2h
        d2h.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(d2h):
            self.data[a:a + len(values)].copy_(values, non_blocking=True)
        values.record_stream(d2h)

    def free(self) -> None:
        """Drop the rows; a mapped host block is unregistered and given back once its copies are done."""
        if self.mapping is not None:
            self.streams.h2d.synchronize()
            self.streams.d2h.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self.data.data_ptr())
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


def prefetched(slabs: Sequence, fetch: Callable) -> Iterator:
    """Yield ``(slab, fetched)``; the next slab's copies are queued before the current one is used."""
    if not slabs:
        return
    pending = fetch(slabs[0])
    for k, slab in enumerate(slabs):
        upcoming = fetch(slabs[k + 1]) if k + 1 < len(slabs) else None
        yield slab, pending
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

    def fetch(self, level: Level, a: int, b: int):
        return self.store.fetch(a, b)

    def get(self, fetched) -> torch.Tensor:
        return fetched.get()


PROJ_CHUNK_BYTES = 96 << 20
"""Upsample projections are computed this many output bytes at a time (whole parent layers)."""


def layer_chunks(level: Level, x0: int, x1: int, max_rows: int) -> Iterator[tuple[int, int]]:
    """Runs of whole layers ``[l0, l1)`` within ``[x0, x1)`` of at most ``max_rows`` rows (or one layer)."""
    x = x0
    while x < x1:
        y = x + 1
        while y < x1 and level.starts[y + 1] - level.starts[x] <= max_rows:
            y += 1
        yield x, y
        x = y


def project_children(proj, parents: torch.Tensor, parent: Level, layers: tuple[int, int], first_row: int,
                     child: Level, src: torch.Tensor, a: int, b: int, channels: int, dtype) -> torch.Tensor:
    """Rows ``[a, b)`` of the child level: ``proj(parent row)`` split into 8 children, the kept ones picked
    by ``src`` (parent row * 8 + child index). ``parents`` holds the parent rows of ``layers`` from row
    ``first_row``; the projection runs over a few parent layers at a time, whose children are the rows of
    twice those layers."""
    out = torch.empty((b - a, channels), dtype=dtype, device=parents.device)
    chunk = max(1, PROJ_CHUNK_BYTES // (16 * channels))
    for l0, l1 in layer_chunks(parent, layers[0], layers[1], chunk):
        q0, q1 = parent.rows(l0, l1)
        c0, c1 = child.rows(2 * l0, 2 * l1)
        c0, c1 = max(c0, a), min(c1, b)
        if c1 <= c0:
            continue
        x = tall(proj, parents[q0 - first_row:q1 - first_row])
        out[c0 - a:c1 - a] = x.view(-1, channels)[src[c0:c1].long() - q0 * 8]
        del x
    return out


class ChildSource:
    """An upsampled level's input, recomputed from the parents: ``proj(parent)`` viewed as one row per
    child, the kept children picked by ``src`` (parent row * 8 + child index)."""

    def __init__(self, parent: Level, store: Store, src: torch.Tensor, proj, channels: int, dtype: torch.dtype):
        self.parent, self.store, self.src, self.proj, self.channels = parent, store, src, proj, channels
        self.dtype = dtype

    def fetch(self, level: Level, a: int, b: int):
        if b <= a:
            return level, a, b, (0, 0), 0, Pending(self.store.data[:0])
        layers = level.layer_of(a) // 2, level.layer_of(b - 1) // 2 + 1
        pa, pb = self.parent.rows(*layers)
        return level, a, b, layers, pa, self.store.fetch(pa, pb)

    def get(self, fetched) -> torch.Tensor:
        level, a, b, layers, pa, pending = fetched
        return project_children(self.proj, pending.get(), self.parent, layers, pa, level, self.src, a, b,
                                self.channels, self.dtype)


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
        rows = self.store.fetch(0, self.n).get()
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
    """How a decode is cut: rows per slab (halo layers included) and where stores live.

    The fixed plan: ``max_rows`` for every sweep, every store on the card. :class:`BudgetPlan` (in
    ``sparc3d.pipelines.streamed_decode``) sizes them from a memory budget instead.
    """

    release_above = 0
    """Cached blocks are given back before a sweep or a store once PyTorch reserves more than this many
    bytes (a fresh block per store and sweep keeps the reserve near what is allocated)."""

    def __init__(self, max_rows: int = 1 << 20, stores_on_card: bool = True):
        self.max_rows, self.stores_on_card = max_rows, stores_on_card

    def slab_rows(self, kind: str, index: int, channels_in: int, host_row_bytes: int = 0) -> int:
        """Rows per slab for a sweep (``kind``: resnet1, resnet2, upsample, conv_out or extract) whose input
        has ``channels_in`` channels; ``host_row_bytes`` per row arrive from page-locked stores."""
        return self.max_rows

    def store_on_card(self, name: str, index: int, nbytes: int, kind: str, channels_in: int) -> bool:
        """Whether a store goes on the card, made for a sweep of ``kind`` over ``channels_in`` channels."""
        return self.stores_on_card


class StreamedDecoder:
    """Decode latents slab by slab (see the module docstring)."""

    def __init__(self, decoder, plan: SlabPlan, device: torch.device):
        self.decoder, self.plan, self.device = decoder, plan, device
        self.streams = Streams(device)
        self.sweeps: list[dict] = []
        self.host_bytes = self.host_peak = 0
        self.peak_alloc = self.peak_reserved = 0.0

    def _store(self, name: str, index: int, rows: int, channels: int, dtype: torch.dtype, kind: str,
               channels_in: int) -> Store:
        nbytes = rows * channels * torch.empty((), dtype=dtype).element_size()
        self._release()               # a long-lived block of its own, not one split out of cached ones
        on_card = bool(self.plan.store_on_card(name, index, nbytes, kind, channels_in))
        if not on_card:
            self.host_bytes += nbytes
            self.host_peak = max(self.host_peak, self.host_bytes)
        store = Store.empty(rows, channels, dtype, on_card, self.streams, self.device)
        store.name = name
        return store

    def _release(self) -> None:
        if torch.cuda.memory_reserved(self.device) > self.plan.release_above:
            torch.cuda.empty_cache()

    def _free(self, store: Store) -> None:
        if not store.on_card and store.data is not None:
            self.host_bytes -= store.nbytes
        store.free()

    def _slabs(self, kind: str, index: int, level: Level, channels_in: int, inputs: Sequence[Store]):
        """The sweep's slabs, and its record (the card's peak is reset for it). ``inputs``: the stores it
        reads; those in page-locked memory cost two slabs of copies on the card (current and next)."""
        self._release()               # the previous sweep's cached blocks would fragment this one's
        host = sum(2 * store.data.shape[1] * store.data.element_size() for store in inputs if not store.on_card)
        rows = self.plan.slab_rows(kind, index, channels_in, host)
        slabs = plan_slabs(level, rows)
        record = {"sweep": kind, "level": index, "rows": level.n, "slab_rows": rows, "slabs": len(slabs),
                  "host_row_bytes": host, "t0": time.perf_counter(),
                  "widest": max((level.rows(a - 1, b + 1)[1] - level.rows(a - 1, b + 1)[0] for a, b in slabs),
                                default=0),
                  "alloc_before_mib": torch.cuda.memory_allocated(self.device) / 2**20}
        torch.cuda.reset_peak_memory_stats(self.device)
        self.sweeps.append(record)
        return slabs

    def _end_sweep(self) -> None:
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
            slabs = self._slabs("resnet2", index, level, conv1.in_channels, [source.store])
            for (x0, x1), fetched in prefetched(slabs, lambda s: source.fetch(level, *level.rows(*s))):
                x = source.get(fetched)
                out.write(level.rows(x0, x1)[0], x)
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
        slabs = self._slabs("resnet1", index, level, conv1.in_channels, [source.store])
        for (x0, x1), fetched in prefetched(slabs, lambda s: source.fetch(level, *level.rows(s[0] - 1, s[1] + 1))):
            h0, h1 = level.rows(x0 - 1, x1 + 1)
            o0, o1 = level.rows(x0, x1)
            y = swish(group_norm(source.get(fetched), norm1, stats, dtype))
            del fetched
            h = sparse_conv(conv1, y, level.coords[h0:h1], level.resolution, large)[o0 - h0:o1 - h0]
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
        slabs = self._slabs("resnet2", index, level, conv1.in_channels, [hidden, source.store])

        def fetch(s):
            return hidden.fetch(*level.rows(s[0] - 1, s[1] + 1)), source.fetch(level, *level.rows(*s))

        for (x0, x1), (fetched_h, fetched_x) in prefetched(slabs, fetch):
            h0, h1 = level.rows(x0 - 1, x1 + 1)
            o0, o1 = level.rows(x0, x1)
            y = swish(group_norm(fetched_h.get(), norm2, stats, dtype))
            del fetched_h
            c = sparse_conv(conv2, y, level.coords[h0:h1], level.resolution, large)[o0 - h0:o1 - h0]
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
        """Upsample's occupancy head over every slab; the kept children sorted by (x, y, z).

        Returns the child level, its input (recomputed from ``store``) and that input's moments.
        """
        unpool = up.upsample
        norm, conv = unpool.cls_head.fc._modules["0"], unpool.cls_head.fc._modules["2"]
        channels = unpool.proj.in_features
        stats = moments.stats(norm)
        large = conv_takes_large_path(conv, level.n, torch.float32)
        child_resolution = level.resolution * 2
        offsets = unpool.inv_grid_coord.to(torch.int32)
        child_moments = Moments(channels, self.device)
        coords_parts, src_parts = [], []
        dtype = None
        slabs = self._slabs("upsample", index, level, channels, [store])
        for (x0, x1), fetched in prefetched(slabs, lambda s: store.fetch(*level.rows(s[0] - 1, s[1] + 1))):
            h0, h1 = level.rows(x0 - 1, x1 + 1)
            o0, o1 = level.rows(x0, x1)
            z = fetched.get()
            del fetched
            with torch.autocast("cuda", enabled=False):
                y = swish(group_norm(z, norm, stats, torch.float32))
                logits = sparse_conv(conv, y, level.coords[h0:h1], level.resolution, large)[o0 - h0:o1 - h0]
                del y
            local = torch.nonzero(logits.flatten() > 0).squeeze(1)
            del logits
            child = level.coords[local // 8 + o0] * 2 + offsets[local % 8]
            order = torch.argsort(lattice_key(child, child_resolution))
            local = local[order]
            coords_parts.append(child[order])
            src_parts.append((local + o0 * 8).to(torch.int32))
            del child, order, local
            chunk = max(1, PROJ_CHUNK_BYTES // (16 * channels))
            for l0, l1 in layer_chunks(level, x0, x1, chunk):
                q0, q1 = level.rows(l0, l1)
                x = tall(unpool.proj, z[q0 - h0:q1 - h0])
                dtype = x.dtype
                picked = src_parts[-1]
                picked = picked[(picked >= q0 * 8) & (picked < q1 * 8)].long() - q0 * 8
                child_moments.add(x.view(-1, channels)[picked])
                del x, picked
            del z
        child_level = Level.build(torch.cat(coords_parts), child_resolution)
        src = torch.cat(src_parts)
        del coords_parts, src_parts
        self._end_sweep()
        return child_level, ChildSource(level, store, src, unpool.proj, channels, dtype), child_moments

    def _conv_out(self, index: int, level: Level, conv_out, store: Store, moments: Moments) -> Store:
        norm, conv = conv_out.conv._modules["0"], conv_out.conv._modules["2"]
        stats = moments.stats(norm)
        large = conv_takes_large_path(conv, level.n, torch.float32)
        decoded = self._store("decoded", index, level.n, DECODED_CHANNELS, torch.float32, "conv_out",
                              conv.in_channels)
        slabs = self._slabs("conv_out", index, level, conv.in_channels, [store])
        for (x0, x1), fetched in prefetched(slabs, lambda s: store.fetch(*level.rows(s[0] - 1, s[1] + 1))):
            h0, h1 = level.rows(x0 - 1, x1 + 1)
            o0, o1 = level.rows(x0, x1)
            with torch.autocast("cuda", enabled=False):
                y = swish(group_norm(fetched.get(), norm, stats, torch.float32))
                del fetched
                out = sparse_conv(conv, y, level.coords[h0:h1], level.resolution, large)[o0 - h0:o1 - h0]
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
            torch.cuda.reset_peak_memory_stats(self.device)
            self.sweeps.append({"sweep": "latent", "level": 0, "rows": int(coords.shape[0]), "t0": time.perf_counter(),
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
