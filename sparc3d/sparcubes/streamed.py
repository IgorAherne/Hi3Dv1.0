"""Slab-streamed mesh extraction from a decoded field, for cards with little memory.

Spellcast3D addition (not part of Hi3D v1.0).

:func:`point_to_mesh` + :func:`fill_mesh_holes` over slabs of whole x layers. A slab reads its own
cube layers ``[x0, x1)`` plus three layers below and two above, which is what makes every corner
average, sign, ambiguity resolution and dual vertex it keeps the same as over the whole field. It
keeps the dual vertices of its own cubes and the quads of the surface edges whose lowest x lies in
the slab. The pieces are put in the reference's order by keys (dual vertices by (vertex count of the
cube, cube, index in the cube); quads flipped first, then by their edge's corner indices), so the
mesh is the reference's, vertex for vertex and face for face (bit for bit under PyTorch's deterministic
algorithms; by default the corner and dual-vertex sums of both are float32 atomics, whose last bit
varies from run to run).
"""

from __future__ import annotations

import numpy as np
import torch

from sparc3d.sparcubes.extract import _extractor
from sparc3d.sparcubes.fill_hole import _boundary_loops, _ear_clip
from sparc3d.sparcubes.sparse_cube import CUBE_CORNERS, cubes_to_verts
from sparc3d.structures.point import Point

HALO_BELOW = 3
HALO_ABOVE = 2
"""Cube layers a slab reads beyond its own (see the module docstring)."""

_QUAD_SPLIT = (0, 1, 3, 3, 1, 2)
"""FlexiCubes' fixed quad diagonal (``quad_split_2``)."""


def _lattice_key(coords: torch.Tensor, size: int) -> torch.Tensor:
    c = coords.long()
    return (c[:, 0] * size + c[:, 1]) * size + c[:, 2]


def _voxel_grid(coords: torch.Tensor, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`construct_voxel_grid` with one int64 key per corner instead of a row-wise unique: the same
    sorted corners and the same cube-to-corner indices, in less time and memory."""
    size = resolution + 1
    corners = (coords.unsqueeze(1) + CUBE_CORNERS.to(coords).unsqueeze(0)).reshape(-1, 3)
    keys, inverse = torch.unique(_lattice_key(corners, size), return_inverse=True)
    del corners
    verts = torch.stack([keys // (size * size), (keys // size) % size, keys % size], dim=1)
    return verts, inverse.reshape(-1, 8)


def _slab_pieces(rows: torch.Tensor, grid: torch.Tensor, resolution: int, matrix: torch.Tensor,
                 x0: int, x1: int) -> tuple | None:
    """point_to_mesh over one slab's rows (already in (x, y, z) order); the slab's own dual vertices
    and quads with their ordering keys, or None when it holds no surface."""
    fc = _extractor(rows.device)
    feat = rows[:, 8:]
    sdf = feat[:, 0:8].reshape(-1, 8, 1)
    deform = feat[:, 8:32].reshape(-1, 8, 3) * 0.5
    vert_feat = torch.cat([sdf, deform], dim=-1)
    sdf_sign = torch.sigmoid(rows[:, :8]).unsqueeze(2)
    vert_feat = torch.cat([sdf_sign, vert_feat], dim=2)
    del sdf, deform, sdf_sign
    vert_coord, cubes = _voxel_grid(grid, resolution)
    vert_feat = cubes_to_verts(vert_coord.shape[0], cubes, vert_feat)
    sdf_sign, vert_feat = vert_feat[:, 0:1], vert_feat[:, 1:]
    sdf, deform = vert_feat[..., 0:1], vert_feat[..., 1:4]
    positions = Point.grid_to_coord(vert_coord + deform, matrix)
    sdf = torch.where(sdf_sign >= 0.5, 1, -1) * sdf.abs()
    del vert_feat, sdf_sign, deform

    with torch.autocast("cuda", enabled=False):
        positions, sdf = positions.to(torch.float32), sdf.to(torch.float32)
        surf_cubes, occ_fx8 = fc._identify_surf_cubes(sdf, cubes)
        if not bool(surf_cubes.any()):
            return None
        case_ids = fc._get_case_id_sparse(occ_fx8, surf_cubes, resolution, grid)
        surf_edges, idx_map, edge_counts, surf_edges_mask = fc._identify_surf_edges(sdf, cubes, surf_cubes)
        vd, vd_idx_map = fc._compute_vd(positions, surf_edges, sdf, case_ids, idx_map)
    del positions, occ_fx8

    # Each dual vertex's key, in the order _compute_vd numbers them: by vertex count of the cube,
    # then cube, then index within the cube.
    device = rows.device
    surf_index = torch.nonzero(surf_cubes).squeeze(1)
    cube_key = _lattice_key(grid[surf_index], resolution)
    cube_space = resolution ** 3
    num_vd = torch.index_select(fc.num_vd_table, 0, case_ids)
    vd_cube, vd_key = [], []
    for num in torch.unique(num_vd).tolist():
        group = torch.nonzero(num_vd == num).squeeze(1)
        cube = group.repeat_interleave(num)
        j = torch.arange(num, device=device).repeat(group.shape[0])
        vd_cube.append(cube)
        vd_key.append(((num - 1) * cube_space + cube_key[cube]) * 4 + j)
    vd_cube, vd_key = torch.cat(vd_cube), torch.cat(vd_key)
    own_vd = (grid[surf_index[vd_cube], 0] >= x0) & (grid[surf_index[vd_cube], 0] < x1)

    # Quads as _triangulate builds them, with the edge each comes from.
    group_mask = (edge_counts == 4) & surf_edges_mask
    group = idx_map.reshape(-1)[group_mask]
    vd_idx = vd_idx_map[group_mask]
    edge_indices, indices = torch.sort(group, stable=True)
    quad_vd = vd_idx[indices].reshape(-1, 4)
    quad_edge = edge_indices.reshape(-1, 4)[:, 0]
    ends = surf_edges[quad_edge].long()
    s_edges = sdf[ends.reshape(-1)].reshape(-1, 2)
    flip = s_edges[:, 0] > 0
    p0, p1 = vert_coord[ends[:, 0]], vert_coord[ends[:, 1]]
    low = torch.minimum(p0[:, 0], p1[:, 0])
    own_quad = (low >= x0) & (low < x1)
    step = p1 - p0
    direction = torch.where(step[:, 0] != 0, 2, torch.where(step[:, 2] != 0, 1, 0))
    corner_space = (resolution + 1) ** 3
    face_key = (~flip).long() * (3 * corner_space) + _lattice_key(p0, resolution + 1) * 3 + direction
    quad_vd = torch.where(flip.unsqueeze(1), quad_vd[:, [0, 1, 3, 2]], quad_vd[:, [2, 3, 1, 0]])
    quad_keys = vd_key[quad_vd[own_quad]]
    return vd[own_vd], vd_key[own_vd], quad_keys, face_key[own_quad]


def extract_streamed(field, max_rows: int = 1 << 20, length_thresh: int = 10000,
                     trace: list | None = None, release: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference ``fill_mesh_holes(point_to_mesh(decoded))``, slab by slab.

    Args:
        field: A :class:`~sparc3d.models.autoencoders.streamed.DecodedField`.
        max_rows: Decoded rows per slab, halo layers included.
        length_thresh: Longest boundary loop filled, as in :func:`fill_mesh_holes`.
        trace: When given, per-phase card peaks are appended to it.
        release: Free the field's rows once every slab is read (before the merge and hole filling).

    Returns:
        ``(vertices [V, 3] float32, faces [F, 3])`` on the card, in the decode's unit frame.
    """
    from sparc3d.models.autoencoders.streamed import plan_slabs, prefetched

    level, store, matrix = field.level, field.store, field.matrix
    resolution = level.resolution
    slabs = plan_slabs(level, max_rows, halo=HALO_BELOW)
    vds, vd_keys, quad_keys, face_keys = [], [], [], []
    with torch.inference_mode():
        for (x0, x1), fetched in prefetched(
                slabs, lambda s: store.fetch(*level.rows(s[0] - HALO_BELOW, s[1] + HALO_ABOVE))):
            r0, r1 = level.rows(x0 - HALO_BELOW, x1 + HALO_ABOVE)
            pieces = _slab_pieces(fetched.get(), level.coords[r0:r1].long(), resolution, matrix, x0, x1)
            if pieces is not None:
                for out, piece in zip((vds, vd_keys, quad_keys, face_keys), pieces):
                    out.append(piece)
            del pieces, fetched
        if release:
            field.free()
        _mark(trace, "extract_slabs")
        torch.cuda.empty_cache()
        if not vds:
            device = level.coords.device
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.int32, device=device)

        vd_keys = torch.cat(vd_keys)
        order = torch.argsort(vd_keys)
        vd_keys = vd_keys[order]
        vertices = torch.cat(vds)[order]
        del vds, order
        quad_keys = torch.cat(quad_keys)
        quads = torch.searchsorted(vd_keys, quad_keys)
        if not bool((vd_keys[quads.clamp(max=vd_keys.shape[0] - 1)] == quad_keys).all()):
            raise RuntimeError("a quad refers to a dual vertex no slab kept")
        quads = quads.int()
        del quad_keys, vd_keys
        face_order = torch.argsort(torch.cat(face_keys))
        del face_keys
        faces = quads[face_order][:, list(_QUAD_SPLIT)].reshape(-1, 3)
        del quads, face_order
        _mark(trace, "extract_merge")
        torch.cuda.empty_cache()
        faces = fill_mesh_holes_lean(vertices, faces, length_thresh)
        _mark(trace, "fill_holes")
    return vertices, faces


_EDGE_PARTS = 4
"""Edge keys are sorted a quarter at a time (by key modulo 4) when looking for edges used once."""


def fill_mesh_holes_lean(vertices: torch.Tensor, faces: torch.Tensor, length_thresh: int = 10000) -> torch.Tensor:
    """:func:`fill_mesh_holes`'s faces, finding the boundary half-edges with one int64 key per undirected
    edge, sorted part by part, instead of a row-wise unique over all edge pairs at once (the same edges in
    the same order, in a fraction of the memory)."""
    num_vertices = vertices.shape[0]
    corners = faces.unbind(dim=1)
    pairs = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]

    def undirected(a, b):
        return torch.minimum(a, b).long() * num_vertices + torch.maximum(a, b).long()

    once = []
    for part in range(_EDGE_PARTS):
        keys = []
        for a, b in pairs:
            key = undirected(a, b)
            keys.append(key[key % _EDGE_PARTS == part])
            del key
        keys = torch.sort(torch.cat(keys)).values
        single = torch.ones_like(keys, dtype=torch.bool)
        single[1:] &= keys[1:] != keys[:-1]
        single[:-1] &= keys[:-1] != keys[1:]
        once.append(keys[single])
        del keys, single
    once = torch.sort(torch.cat(once)).values
    boundary = []
    for a, b in pairs:
        key = undirected(a, b)
        pos = torch.searchsorted(once, key).clamp_(max=max(once.shape[0] - 1, 0))
        hit = once[pos] == key if once.shape[0] else torch.zeros_like(key, dtype=torch.bool)
        del key, pos
        boundary.append(torch.stack([a, b], dim=1)[hit])
        del hit
    boundary = torch.cat(boundary)

    loops = _boundary_loops(boundary.cpu().numpy().astype(int))
    new_faces = [_ear_clip(vertices, loop) for loop in loops if 3 <= len(loop) <= length_thresh]
    if new_faces:
        appended = torch.from_numpy(np.vstack(new_faces)).long().to(vertices.device)
        faces = torch.cat([faces, appended], dim=0)
    return faces


def _mark(trace: list | None, what: str) -> None:
    """The card's peaks since the previous mark (the counters restart here)."""
    if trace is not None:
        trace.append({"stage": what, "alloc_after_mib": round(torch.cuda.memory_allocated() / 2**20),
                      "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)})
        torch.cuda.reset_peak_memory_stats()
