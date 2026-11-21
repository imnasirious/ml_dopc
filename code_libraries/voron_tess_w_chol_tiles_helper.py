import numpy as np
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

# ---------- PBC tiling helper ----------

def _tile_points_pbc_chol_tiles(coms: np.ndarray, Lx: float, Ly: float):
    """
    Given coms (N×2), returns:
      pts    (9N×2) tiled points in a 3×3 grid
      ids    (9N,)  original index [0..N-1] for each copy
    """
    N = len(coms)
    shifts = np.array([
        ( 0.0,  0.0),
        (+Lx,   0.0),
        (-Lx,   0.0),
        ( 0.0, +Ly),
        ( 0.0, -Ly),
        (+Lx, +Ly),
        (+Lx, -Ly),
        (-Lx, +Ly),
        (-Lx, -Ly),
    ])
    pts = (shifts[:, None, :] + coms[None, :, :]).reshape(-1, 2)
    ids = np.tile(np.arange(N, dtype=int), 9)
    return pts, ids


# ---------- Simple polygon clip to [0, Lx] × [0, Ly] ----------

def _clip_polygon_to_box_chol_tiles(polygon: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Clip a polygon (M×2) to the box [0,Lx]×[0,Ly].
    Returns a (K×2) array; may be empty if polygon lies completely outside.
    """
    poly = [tuple(p) for p in polygon]

    def clip(poly, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_in = inside(prev)
        for curr in poly:
            curr_in = inside(curr)
            if prev_in and curr_in:
                output.append(curr)
            elif prev_in and not curr_in:
                output.append(intersect(prev, curr))
            elif (not prev_in) and curr_in:
                output.append(intersect(prev, curr))
                output.append(curr)
            prev, prev_in = curr, curr_in
        return output

    # x >= 0
    poly = clip(
        poly,
        inside=lambda p: p[0] >= 0.0,
        intersect=lambda p1, p2: (
            0.0,
            p1[1] + (p2[1] - p1[1]) * (0.0 - p1[0]) / (p2[0] - p1[0])
            if p2[0] != p1[0] else p1[1],
        ),
    )

    # x <= Lx
    poly = clip(
        poly,
        inside=lambda p: p[0] <= Lx,
        intersect=lambda p1, p2: (
            Lx,
            p1[1] + (p2[1] - p1[1]) * (Lx - p1[0]) / (p2[0] - p1[0])
            if p2[0] != p1[0] else p1[1],
        ),
    )

    # y >= 0
    poly = clip(
        poly,
        inside=lambda p: p[1] >= 0.0,
        intersect=lambda p1, p2: (
            p1[0] + (p2[0] - p1[0]) * (0.0 - p1[1]) / (p2[1] - p1[1])
            if p2[1] != p1[1] else p1[0],
            0.0,
        ),
    )

    # y <= Ly
    poly = clip(
        poly,
        inside=lambda p: p[1] <= Ly,
        intersect=lambda p1, p2: (
            p1[0] + (p2[0] - p1[0]) * (Ly - p1[1]) / (p2[1] - p1[1])
            if p2[1] != p1[1] else p1[0],
            Ly,
        ),
    )

    return np.array(poly)


# ---------- Fully periodic 2D Voronoi on xy-plane ----------

def _compute_periodic_voronoi_chol_tiles(coms: np.ndarray, Lx: float, Ly: float):
    """
    coms : (N×2) positions in the primary box [0,Lx]×[0,Ly].
    Returns:
      regions : dict { i -> (M_i×2) polygon for central-copy site i }
    """
    N = coms.shape[0]
    pts, ids = _tile_points_pbc_chol_tiles(coms, Lx, Ly)
    vor = Voronoi(pts)

    regions = {}

    # Only central copies are indices 0..N-1
    for j in range(N):
        region_idx = vor.point_region[j]
        vert_ids = vor.regions[region_idx]
        if not vert_ids or any(v < 0 for v in vert_ids):
            # infinite or empty region; skip
            continue

        poly = np.vstack([vor.vertices[v] for v in vert_ids])
        poly = _clip_polygon_to_box_chol_tiles(poly, Lx, Ly)
        if len(poly) == 0:
            continue

        regions[j] = poly

    return regions
