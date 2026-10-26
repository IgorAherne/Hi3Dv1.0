"""One refinement pass's VAE decode and mesh extraction within a memory budget.

Spellcast3D addition (not part of Hi3D v1.0). :func:`decode_and_extract` runs the slab-streamed decode
(:mod:`sparc3d.models.autoencoders.streamed`) and extraction (:mod:`sparc3d.sparcubes.streamed`):
the same mesh as ``Sparc3DPipeline._extract_mesh(Sparc3DPipeline._decode_latents(...))``, with the
card's memory held under ``budget_mib``. :func:`memory_model` says what a decode will need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from sparc3d.models.autoencoders.streamed import SlabPlan, StreamedDecoder
from sparc3d.sparcubes.streamed import extract_streamed

MIB = 1 << 20

ROW_BYTES = {
    "resnet1": {128: 2_300, 256: 2_500, 512: 4_800},
    "resnet2": {128: 2_300, 256: 1_700, 512: 3_200},
    "upsample": {128: 1_400, 256: 2_200, 512: 4_600},
    "project": {128: 700, 256: 1_400, 512: 2_800},
    "conv_out": {128: 1_500},
    "extract": {40: 2_100},
}
"""Card bytes a sweep holds at its peak per row of its widest slab (halo layers included), by sweep and
input channels: measured on the RTX 3060 (knight, pass 3, 400 000-row slabs), rounded up."""
COST_SAFETY = 1.15
MARGIN_MIB = 600
"""Kept free under the budget: the caching allocator's split blocks (measured up to ~450 MiB above the
allocated peak) and the moments' temporaries."""
MIN_SLAB_ROWS = 65_536
"""Smaller slabs cost more halo and launch overhead than they save."""
RESERVE_ROWS = 1 << 18
"""A store goes on the card only if a slab this tall of its sweep still fits beside it."""
PIECE_BYTES = 64
"""Card bytes per decoded point the extraction keeps until its merge (dual vertices and quads, with keys)."""


def row_bytes(kind: str, channels_in: int) -> int:
    """Measured card bytes per slab row of a sweep (the next wider measurement when not measured)."""
    table = ROW_BYTES[kind]
    wider = [c for c in sorted(table) if c >= channels_in]
    if wider:
        return table[wider[0]]
    widest = max(table)
    return table[widest] * channels_in // widest


class BudgetPlan(SlabPlan):
    """Slab sizes and store places decided as the decode goes: what is allocated now, plus what the
    sweep needs per slab row, stays under the budget."""

    def __init__(self, budget_mib: float, device: torch.device, margin_mib: float = MARGIN_MIB):
        super().__init__()
        self.budget = int(budget_mib * MIB)
        self.margin = int(margin_mib * MIB)
        self.device = device
        self.release_above = int(0.6 * self.budget)

    def free_bytes(self) -> int:
        return self.budget - self.margin - torch.cuda.memory_allocated(self.device)

    def slab_rows(self, kind: str, index: int, channels_in: int, host_row_bytes: int = 0,
                  reserve_bytes: int = 0) -> int:
        per_row = row_bytes(kind, channels_in) * COST_SAFETY + host_row_bytes
        return max(MIN_SLAB_ROWS, int((self.free_bytes() - reserve_bytes) // per_row))

    def store_on_card(self, name: str, index: int, nbytes: int, kind: str, channels_in: int) -> bool:
        reserve = RESERVE_ROWS * (row_bytes(kind, channels_in) * COST_SAFETY)
        return self.free_bytes() - nbytes >= reserve


@dataclass
class StreamedMesh:
    """A pass's extracted mesh and what making it took."""

    vertices: torch.Tensor
    """``[V, 3]`` float32, in the decode's unit frame (as the reference's ``_extract_mesh``)."""
    faces: torch.Tensor
    """``[F, 3]`` int64 (the reference's own faces are int32 unless holes were filled; its trimesh is int64)."""
    decoded_points: int
    matrix: torch.Tensor | None = None
    """The decode's grid-to-unit transform ``[4, 4]`` (the latent level's ``Point.matrix``), on the CPU."""
    seconds: dict = field(default_factory=dict)
    peak_alloc_mib: float = 0.0
    peak_reserved_mib: float = 0.0
    host_mib: float = 0.0
    """Host memory the decode's stores used at most (those that did not fit on the card; page-locked unless
    the system refused it)."""
    sweeps: list = field(default_factory=list)


def decode_and_extract(vae, latents_coords: torch.Tensor, latents_feats: torch.Tensor, structure: int,
                       budget_mib: float = 5000, device: torch.device | str = "cuda",
                       to_cpu: bool = True) -> StreamedMesh:
    """Decode one pass's latents and extract its mesh, the card's memory kept under ``budget_mib``.

    Args:
        vae: The :class:`SparconvVAE` (its weights on ``device``; they count against the budget).
        latents_coords: ``int [N, 4]`` (batch 0, x, y, z) on the structure grid.
        latents_feats: ``float [N, 32]``, already denormalised (``feats * std + mean``).
        structure: The structure resolution (64, 128 or 192).
        budget_mib: What the card may hold for this, weights included (allocated bytes; the caching
            allocator's reserve stays within a few hundred MiB of it).
        device: The card.
        to_cpu: Return the mesh on the CPU (else it stays on the card).

    Returns:
        :class:`StreamedMesh`; vertices and faces equal those of the reference's ``_extract_mesh``.

    Raises:
        RuntimeError: The isosurface is empty.
    """
    device = torch.device(device)
    plan = BudgetPlan(budget_mib, device)
    decoder = StreamedDecoder(vae.decoder, plan, device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    field_ = decoder(latents_coords.to(device), latents_feats.to(device), structure)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    trace: list = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        host = 0 if field_.store.on_card else 2 * 40 * 4
        rows = plan.slab_rows("extract", 3, 40, host, reserve_bytes=PIECE_BYTES * field_.n)
        vertices, faces = extract_streamed(field_, max_rows=rows, trace=trace, release=True)
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise RuntimeError("mesh extraction produced an empty isosurface")
        faces = faces.long()
        if to_cpu:
            vertices, faces = vertices.cpu(), faces.cpu()
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    extract_peak = max((r["peak_alloc_mib"] for r in trace), default=0.0)
    extract_reserved = max((r["peak_reserved_mib"] for r in trace), default=0.0)
    sweeps = decoder.sweeps + [{"sweep": "extract", "level": 3, "slab_rows": rows, **r} for r in trace]
    return StreamedMesh(vertices, faces, field_.n, field_.matrix.detach().double().cpu(),
                        seconds={"decode": round(t1 - t0, 3), "extract": round(t2 - t1, 3)},
                        peak_alloc_mib=max(decoder.peak_alloc, extract_peak),
                        peak_reserved_mib=max(decoder.peak_reserved, extract_reserved),
                        host_mib=decoder.host_peak / MIB, sweeps=sweeps)


WEIGHTS_MIB = 376
"""The VAE's float32 weights on the card."""
POINTS_PER_LATENT = 72
"""Decoded points per latent (measured 68-77 on the five reference props, every pass)."""
LATENT_STAGE_KIB = (12.0, 25.6)
"""The latent level (reference modules, every point at once): KiB per latent allocated / reserved."""
SMALL_DECODE_MIB = (800, 2.9 / 1024)
"""PyTorch's reserve when the decode is well under the budget: MiB + MiB per decoded point (cached
blocks are given back only above 60 % of the budget, so the reserve runs ahead of the allocations)."""
TAIL_BYTES_PER_POINT = 170
"""The extraction's merge and hole filling (not slabbed): card bytes per decoded point at their peak
(1537 MiB allocated with the weights for 7.4 M points)."""
SECONDS_PER_MPOINT = {"decode": (1.6, 2.1), "extract": (2.1, 3.4)}
"""RTX 3060, per million decoded points: (spconv path, chunked-GEMM path) / (fewest, most holes)."""
BUDGET_OVERSHOOT_MIB = 100
"""PyTorch's reserve went at most this far over the budget (5090 MiB at 5000; the allocated peak stays
under budget - MARGIN_MIB)."""


def memory_model(latents: int, budget_mib: float = 5000, decoded_points: int | None = None) -> dict:
    """What :func:`decode_and_extract` will need, for planning: card and page-locked MiB, seconds.

    The planner's own decisions at the decoded level (the one that matters), with the measured costs.
    ``decoded_points`` defaults to ``POINTS_PER_LATENT * latents``.

    Returns:
        ``card_mib``: PyTorch's reserve at its peak; ``floor_mib``: the smallest budget that works;
        ``host_mib``: page-locked RAM for the stores that do not fit on the card; ``decode_s`` and
        ``extract_s``: (low, high) on the RTX 3060.
    """
    n3 = int(decoded_points if decoded_points is not None else POINTS_PER_LATENT * latents)
    mib = 1 / MIB
    latent_alloc = WEIGHTS_MIB + LATENT_STAGE_KIB[0] * latents / 1024
    latent_reserved = WEIGHTS_MIB + LATENT_STAGE_KIB[1] * latents / 1024
    tail = WEIGHTS_MIB + TAIL_BYTES_PER_POINT * n3 * mib
    reserve = {kind: RESERVE_ROWS * row_bytes(kind, channels) * COST_SAFETY * mib
               for kind, channels in (("resnet1", 128), ("resnet2", 128), ("conv_out", 128))}
    # Level 3: coordinates and children map, the parents' output, then two 128-channel float16 stores
    # (hidden, output) and, at the end, the decoded rows.
    persistent = WEIGHTS_MIB + n3 * 16 * mib + n3 / 4 * 268 * mib + 32
    free = budget_mib - MARGIN_MIB - persistent
    store = n3 * 256 * mib
    host = 0.0
    hidden_on_card = free - store >= reserve["resnet1"]
    if hidden_on_card:
        free -= store
    else:
        host += store
    out_on_card = free - store >= reserve["resnet2"]
    if not out_on_card:
        host += store
    decoded = n3 * 160 * mib
    free_end = budget_mib - MARGIN_MIB - WEIGHTS_MIB - n3 * 12 * mib - (store if out_on_card else 0)
    if free_end - decoded < reserve["conv_out"]:
        host += decoded
    floor = max(latent_reserved, tail + 300, persistent + MIN_SLAB_ROWS * row_bytes("resnet1", 128) * 2 * mib)
    small = SMALL_DECODE_MIB[0] + SMALL_DECODE_MIB[1] * n3
    large = n3 * 128 * 4 >= 2**31 - 1
    return {
        "decoded_points": n3,
        "card_mib": round(max(latent_reserved, min(budget_mib + BUDGET_OVERSHOOT_MIB, small))),
        "floor_mib": round(floor),
        "host_mib": round(host),
        "latent_stage_mib": (round(latent_alloc), round(latent_reserved)),
        "decode_s": round(SECONDS_PER_MPOINT["decode"][1 if large else 0] * n3 / 1e6, 1),
        "extract_s": tuple(round(k * n3 / 1e6, 1) for k in SECONDS_PER_MPOINT["extract"]),
    }
