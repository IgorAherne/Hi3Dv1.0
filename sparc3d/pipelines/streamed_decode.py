"""One refinement pass's VAE decode and mesh extraction within a memory budget.

Spellcast3D addition (not part of Hi3D v1.0). :func:`decode_and_extract` runs the piece-streamed decode
(:mod:`sparc3d.models.autoencoders.streamed`) and extraction (:mod:`sparc3d.sparcubes.streamed`):
the mesh of ``Sparc3DPipeline._extract_mesh(Sparc3DPipeline._decode_latents(...))`` up to the
reference's own run-to-run variation (deterministic here), with the card's memory held under
``budget_mib``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from sparc3d.models.autoencoders.streamed import CardPeaks, SlabPlan, StreamedDecoder
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
"""Card bytes a sweep holds at its peak per row of its widest piece (halo included), by sweep and input
channels: measured on the RTX 3060 (knight, pass 3, 400 000-row pieces), rounded up."""
COST_SAFETY = 1.15
MARGIN_MIB = 600
"""Kept free under the budget: the caching allocator's split blocks (measured up to ~450 MiB above the
allocated peak) and the moments' temporaries."""
MIN_SLAB_ROWS = 65_536
"""Smaller pieces cost more halo and launch overhead than they save."""
RESERVE_ROWS = 1 << 18
"""A store goes on the card only if a piece this tall of its sweep still fits beside it."""
PIECE_BYTES = 64
"""Card bytes per decoded point the extraction keeps until its merge (dual vertices and quads, with keys)."""
LATENT_BYTES = 12 << 10
"""Card bytes per latent the latent level holds (the reference modules on every latent at once)."""
TAIL_BYTES = 170 * 77
"""Card bytes per latent the extraction's merge and hole filling hold (not streamed): 170 per decoded
point, at most 77 decoded points per latent (68-77 measured)."""


def row_bytes(kind: str, channels_in: int) -> int:
    """Measured card bytes per piece row of a sweep (the next wider measurement when not measured)."""
    table = ROW_BYTES[kind]
    wider = [c for c in sorted(table) if c >= channels_in]
    if wider:
        return table[wider[0]]
    widest = max(table)
    return table[widest] * channels_in // widest


class BudgetPlan(SlabPlan):
    """Piece sizes and store places decided as the decode goes: what is allocated now, plus what the
    sweep needs per piece row, stays under the budget."""

    def __init__(self, budget_mib: float, device: torch.device, margin_mib: float = MARGIN_MIB):
        super().__init__()
        if torch.device(device).type != "cuda":
            raise ValueError(f"the streamed decode needs a CUDA device, not {device}")
        if not math.isfinite(budget_mib) or budget_mib <= 0:
            raise ValueError(f"the streamed decode needs a finite, positive budget in MiB, not {budget_mib}")
        self.budget = int(budget_mib * MIB)
        self.margin = int(margin_mib * MIB)
        self.device = device
        self.release_above = int(0.6 * self.budget)

    def free_bytes(self) -> int:
        return self.budget - self.margin - torch.cuda.memory_allocated(self.device)

    def check_floor(self, latents: int) -> None:
        """Fail before any work when the budget cannot hold the decode's parts that are not streamed (the
        latent level, the extraction's tail) or its smallest pieces, beside what is allocated now."""
        piece = MIN_SLAB_ROWS * max(max(table.values()) for table in ROW_BYTES.values()) * COST_SAFETY
        need = max(LATENT_BYTES * latents, TAIL_BYTES * latents, piece)
        if self.free_bytes() < need:
            floor = (self.budget - self.free_bytes() + need) / MIB
            raise torch.OutOfMemoryError(
                f"out of memory: the streamed decode of {latents:,} latents needs a budget of at least "
                f"{floor:,.0f} MiB of the card (with {torch.cuda.memory_allocated(self.device) / MIB:,.0f} MiB "
                f"already allocated), not {self.budget / MIB:,.0f} MiB")

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
    """The card's allocated peak during the call: exact when traced or when the call set the process's
    peak (as after the caller resets PyTorch's peak counters), else a lower bound."""
    peak_reserved_mib: float = 0.0
    """PyTorch's reserve at its peak during the call (see :class:`CardPeaks`)."""
    host_mib: float = 0.0
    """Host memory the decode's stores used at most (those that did not fit on the card; page-locked unless
    the system refused it)."""
    sweeps: list = field(default_factory=list)


def decode_and_extract(vae, latents_coords: torch.Tensor, latents_feats: torch.Tensor, structure: int,
                       budget_mib: float = 5000, device: torch.device | str = "cuda",
                       to_cpu: bool = True, trace: bool = False) -> StreamedMesh:
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
        trace: Time every sweep and phase and record its card peaks (profiling only: the card is
            synchronized after every sweep and PyTorch's process-wide peak counters are reset).

    Returns:
        :class:`StreamedMesh`; vertices and faces those of the reference's ``_extract_mesh`` up to its own
        run-to-run variation (the same in every run here).

    Raises:
        ValueError: ``device`` is not a CUDA device, or ``budget_mib`` is not a finite positive number.
        torch.OutOfMemoryError: ``budget_mib`` is below what the decode needs at the least (before any work).
        RuntimeError: The isosurface is empty.
    """
    device = torch.device(device)
    plan = BudgetPlan(budget_mib, device)
    plan.check_floor(int(latents_coords.shape[0]))
    peaks = CardPeaks(device)
    decoder = StreamedDecoder(vae.decoder, plan, device, trace=trace, peaks=peaks)
    t0 = time.perf_counter()
    field_ = decoder(latents_coords.to(device), latents_feats.to(device), structure)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    records: list | None = [] if trace else None
    if trace:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        host = 0 if field_.store.on_card else 2 * 40 * 4
        rows = plan.slab_rows("extract", 3, 40, host, reserve_bytes=PIECE_BYTES * field_.n)
        vertices, faces = extract_streamed(field_, max_rows=rows, trace=records, release=True, peaks=peaks)
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise RuntimeError("mesh extraction produced an empty isosurface")
        faces = faces.long()
        if to_cpu:
            vertices, faces = vertices.cpu(), faces.cpu()
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    if trace:
        peak_alloc = max([decoder.peak_alloc] + [r["peak_alloc_mib"] for r in records])
        peak_reserved = max([decoder.peak_reserved] + [r["peak_reserved_mib"] for r in records])
    else:
        peak_alloc, peak_reserved = (value / MIB for value in peaks.result())
    sweeps = decoder.sweeps + [{"sweep": "extract", "level": 3, "slab_rows": rows, **r} for r in records or []]
    return StreamedMesh(vertices, faces, field_.n, field_.matrix.detach().double().cpu(),
                        seconds={"decode": round(t1 - t0, 3), "extract": round(t2 - t1, 3)},
                        peak_alloc_mib=peak_alloc, peak_reserved_mib=peak_reserved,
                        host_mib=decoder.host_peak / MIB, sweeps=sweeps)
