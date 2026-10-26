"""Piece-streamed mesh extraction from a decoded field, for cards with little memory.

Spellcast3D addition (not part of Hi3D v1.0).

:func:`point_to_mesh` + :func:`fill_mesh_holes` piece by piece (the decode's pieces: runs of whole x
layers, a big layer in strips of y lines). A piece reads its own cubes plus three cells below and two
above in x (and in y when cut in y), which is what makes every corner average, sign, ambiguity
resolution and dual vertex it keeps the same as over the whole field. It keeps the dual vertices of
its own cubes and the quads of the surface edges whose lowest corner lies in the piece. The pieces
are put in the reference's order by keys (dual vertices by (vertex count of the cube, cube, index in
the cube); quads flipped first, then by their edge's corner indices), so the mesh is the reference's,
vertex for vertex and face for face, up to the reference's own run-to-run variation: its corner and
dual-vertex sums are float32 atomics, whose last bit varies from run to run; here they are taken in a
fixed order, so the same field gives the same mesh in every run and whatever the pieces.
"""

from __future__ import annotations

import numpy as np
import torch

from sparc3d.models.autoencoders.streamed import lattice_key, plan_pieces, prefetched, take
from sparc3d.sparcubes.extract import _extractor
from sparc3d.sparcubes.fill_hole import _boundary_loops, _ear_clip
from sparc3d.sparcubes.sparse_cube import CUBE_CORNERS
from sparc3d.structures.point import Point

HALO_BELOW = 3
HALO_ABOVE = 2
"""Cells a piece reads beyond its own, in x and (when cut) in y (see the module docstring)."""

_QUAD_SPLIT = (0, 1, 3, 3, 1, 2)
"""FlexiCubes' fixed quad diagonal (``quad_split_2``)."""


def _voxel_grid(coords: torch.Tensor, resolution: int,
                values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`sparse_cube2verts`: the sorted cube corners, the cubes' corner indices and the mean of the
    ``values [V, 8, M]`` the cubes give each corner. One int64 key per corner instead of a row-wise
    unique (the same corners and indices, in less time and memory), and each corner's sum taken in a
    fixed order (its cubes in row order, one after the other) instead of float32 atomics, so the same
    bits in every run and in every piece holding all of the corner's cubes."""
    size = resolution + 1
    corners = (coords.unsqueeze(1) + CUBE_CORNERS.to(coords).unsqueeze(0)).reshape(-1, 3)
    keys, order = torch.sort(lattice_key(corners, size), stable=True)
    del corners
    new = torch.ones_like(keys, dtype=torch.bool)
    new[1:] = keys[1:] != keys[:-1]
    first = torch.nonzero(new).squeeze(1)
    keys = keys[first]
    vertex = torch.cumsum(new, 0) - 1
    del new
    cubes = torch.empty_like(vertex)
    cubes[order] = vertex
    del vertex
    counts = torch.diff(first, append=first.new_tensor([order.shape[0]]))
    flat = values.reshape(order.shape[0], -1)
    sums = flat[order[first]]
    last = order.shape[0] - 1
    for k in range(1, 8):
        more = (counts > k).unsqueeze(1)
        sums = torch.where(more, sums + flat[order[(first + k).clamp_(max=last)]], sums)
    mean = sums / counts.unsqueeze(1).to(sums.dtype)
    verts = torch.stack([keys // (size * size), (keys // size) % size, keys % size], dim=1)
    return verts, cubes.view(-1, 8), mean


def _dual_vertices(fc, positions: torch.Tensor, surf_edges: torch.Tensor, sdf: torch.Tensor,
                   case_ids: torch.Tensor, idx_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``FlexiCubes._compute_vd``, each dual vertex's edge crossings summed in a fixed order (they are
    consecutive: added one after the other) instead of float32 atomics."""
    device = positions.device
    surf_edges_x = positions[surf_edges.reshape(-1)].reshape(-1, 2, 3)
    surf_edges_s = sdf[surf_edges.reshape(-1)].reshape(-1, 2, 1)
    idx_map = idx_map.reshape(-1, 12)
    num_vd = fc.num_vd_table[case_ids]
    edge_group, edge_group_to_vd, edge_group_to_cube = [], [], []
    total = 0
    group_size = fc.dmc_table.shape[-1]
    for num in torch.unique(num_vd).tolist():
        cur_cubes = num_vd == num
        cube_ids = torch.nonzero(cur_cubes).squeeze(1)
        curr_num_vd = cube_ids.shape[0] * num
        curr_edge_group = fc.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * group_size)
        curr_edge_group_to_vd = (torch.arange(curr_num_vd, device=device).unsqueeze(-1)
                                 .repeat(1, group_size) + total).reshape_as(curr_edge_group)
        total += curr_num_vd
        curr_edge_group_to_cube = cube_ids.unsqueeze(-1).repeat(1, num * group_size)
        mask = curr_edge_group != -1
        edge_group.append(curr_edge_group[mask])
        edge_group_to_vd.append(curr_edge_group_to_vd[mask])
        edge_group_to_cube.append(curr_edge_group_to_cube[mask])
    edge_group = torch.cat(edge_group)
    edge_group_to_vd = torch.cat(edge_group_to_vd)
    edge_group_to_cube = torch.cat(edge_group_to_cube)

    index = edge_group_to_cube * 12 + edge_group
    idx_group = idx_map.reshape(-1)[index]
    x_group = surf_edges_x[idx_group.reshape(-1)].reshape(-1, 2, 3)
    s_group = surf_edges_s[idx_group.reshape(-1)].reshape(-1, 2, 1)
    crossings = fc._linear_interp(s_group.float(), x_group)
    del x_group, s_group, idx_group

    counts = torch.bincount(edge_group_to_vd, minlength=total)
    first = torch.cumsum(counts, 0) - counts
    sums = torch.zeros((total, 3), dtype=torch.float, device=device)
    last = crossings.shape[0] - 1
    for k in range(group_size):
        more = (counts > k).unsqueeze(1)
        sums = torch.where(more, sums + crossings[(first + k).clamp_(max=last)], sums)
    vd = sums / counts.unsqueeze(1).float()

    v_idx = torch.arange(total, dtype=torch.int64, device=device)
    vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.int64, device=device)
    vd_idx_map = vd_idx_map.reshape(-1).scatter(dim=0, index=index, src=v_idx[edge_group_to_vd])
    return vd, vd_idx_map


def _piece_parts(rows: torch.Tensor, grid: torch.Tensor, resolution: int, matrix: torch.Tensor,
                 box: tuple[int, int, int, int]) -> tuple | None:
    """point_to_mesh over one piece's window rows (in (x, y, z) order); the dual vertices of the cubes
    in ``box`` = (x0, x1, y0, y1) and the quads of the edges whose lowest corner lies in it, with their
    ordering keys, or None when it holds no surface."""
    x0, x1, y0, y1 = box
    fc = _extractor(rows.device)
    feat = rows[:, 8:]
    sdf = feat[:, 0:8].reshape(-1, 8, 1)
    deform = feat[:, 8:32].reshape(-1, 8, 3) * 0.5
    vert_feat = torch.cat([sdf, deform], dim=-1)
    sdf_sign = torch.sigmoid(rows[:, :8]).unsqueeze(2)
    vert_feat = torch.cat([sdf_sign, vert_feat], dim=2)
    del sdf, deform, sdf_sign
    vert_coord, cubes, vert_feat = _voxel_grid(grid, resolution, vert_feat)
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
        vd, vd_idx_map = _dual_vertices(fc, positions, surf_edges, sdf, case_ids, idx_map)
    del positions, occ_fx8

    # Each dual vertex's key, in the order _dual_vertices numbers them: by vertex count of the cube,
    # then cube, then index within the cube.
    device = rows.device
    surf_index = torch.nonzero(surf_cubes).squeeze(1)
    cube_key = lattice_key(grid[surf_index], resolution)
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
    cube = grid[surf_index[vd_cube]]
    own_vd = (cube[:, 0] >= x0) & (cube[:, 0] < x1) & (cube[:, 1] >= y0) & (cube[:, 1] < y1)
    del cube

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
    low = torch.minimum(p0, p1)
    own_quad = (low[:, 0] >= x0) & (low[:, 0] < x1) & (low[:, 1] >= y0) & (low[:, 1] < y1)
    del low
    step = p1 - p0
    direction = torch.where(step[:, 0] != 0, 2, torch.where(step[:, 2] != 0, 1, 0))
    corner_space = (resolution + 1) ** 3
    face_key = (~flip).long() * (3 * corner_space) + lattice_key(p0, resolution + 1) * 3 + direction
    quad_vd = torch.where(flip.unsqueeze(1), quad_vd[:, [0, 1, 3, 2]], quad_vd[:, [2, 3, 1, 0]])
    quad_keys = vd_key[quad_vd[own_quad]]
    return vd[own_vd], vd_key[own_vd], quad_keys, face_key[own_quad]


def extract_streamed(field, max_rows: int = 1 << 20, length_thresh: int = 10000, trace: list | None = None,
                     release: bool = False, peaks=None) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference ``fill_mesh_holes(point_to_mesh(decoded))``, piece by piece.

    Args:
        field: A :class:`~sparc3d.models.autoencoders.streamed.DecodedField`.
        max_rows: Decoded rows per piece, halo included.
        length_thresh: Longest boundary loop filled, as in :func:`fill_mesh_holes`.
        trace: When given, per-phase card peaks are appended to it (PyTorch's peak counters are reset
            after each phase).
        release: Free the field's rows once every piece is read (before the merge and hole filling).
        peaks: A :class:`~sparc3d.models.autoencoders.streamed.CardPeaks` that empties the cache.

    Returns:
        ``(vertices [V, 3] float32, faces [F, 3])`` on the card, in the decode's unit frame.
    """
    level, store, matrix = field.level, field.store, field.matrix
    resolution = level.resolution
    device = level.coords.device
    empty_cache = peaks.empty_cache if peaks is not None else torch.cuda.empty_cache
    pieces = plan_pieces(level, max_rows, HALO_BELOW, HALO_ABOVE)
    vds, vd_keys, quad_keys, face_keys = [], [], [], []
    with torch.inference_mode():
        for piece, fetched in prefetched(
                pieces, lambda p: store.fetch(level.ranges(p.window(HALO_BELOW, HALO_ABOVE)))):
            window = level.ranges(piece.window(HALO_BELOW, HALO_ABOVE))
            kept = _piece_parts(fetched.get(), take(level.coords, window).long(), resolution, matrix, piece.box)
            if kept is not None:
                for out, part in zip((vds, vd_keys, quad_keys, face_keys), kept):
                    out.append(part)
            del kept, fetched
        if release:
            field.free()
        _mark(trace, "extract_slabs", device)
        empty_cache()
        if not vds:
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.int32, device=device)

        vd_keys = torch.cat(vd_keys)
        order = torch.argsort(vd_keys)
        vd_keys = vd_keys[order]
        vertices = torch.cat(vds)[order]
        del vds, order
        quad_keys = torch.cat(quad_keys)
        quads = torch.searchsorted(vd_keys, quad_keys)
        if not bool((vd_keys[quads.clamp(max=vd_keys.shape[0] - 1)] == quad_keys).all()):
            raise RuntimeError("a quad refers to a dual vertex no piece kept")
        quads = quads.int()
        del quad_keys, vd_keys
        face_order = torch.argsort(torch.cat(face_keys))
        del face_keys
        faces = quads[face_order][:, list(_QUAD_SPLIT)].reshape(-1, 3)
        del quads, face_order
        _mark(trace, "extract_merge", device)
        empty_cache()
        faces = fill_mesh_holes_lean(vertices, faces, length_thresh)
        _mark(trace, "fill_holes", device)
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

    loops = [loop for loop in _boundary_loops(boundary.cpu().numpy().astype(int)) if 3 <= len(loop) <= length_thresh]
    if not loops:
        return faces
    points = vertices.cpu()           # one copy for every loop: _ear_clip then picks its points on the host
    new_faces = [_ear_clip(points, loop) for loop in loops]
    del points
    appended = torch.from_numpy(np.vstack(new_faces)).long().to(vertices.device)
    return torch.cat([faces, appended], dim=0)


def _mark(trace: list | None, what: str, device: torch.device) -> None:
    """The card's peaks since the previous mark (the counters restart here)."""
    if trace is not None:
        trace.append({"stage": what, "alloc_after_mib": round(torch.cuda.memory_allocated(device) / 2**20),
                      "peak_alloc_mib": round(torch.cuda.max_memory_allocated(device) / 2**20),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20)})
        torch.cuda.reset_peak_memory_stats(device)
