from matplotlib.colors import Normalize
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
import numpy as np

from code_libraries.bilayer_operations import Bilayer
from code_libraries.acyl_chain_preprocessing import *
from code_libraries.miscellaneous_functions import *
#from code_libraries.predict_acyl_chains_NN import *


def _tile_points_pbc(coms: np.ndarray, Lx: float, Ly: float):
    """
    Given coms (N×2), returns:
      pts   (9N×2) tiled points in a 3×3 grid
      ids   (9N,)  original index [0..N-1] for each copy
      shifts (9×2) the translation vectors used
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
    # (9,1,2) + (1,N,2) -> (9,N,2) -> (9N,2)
    pts = (shifts[:, None, :] + coms[None, :, :]).reshape(-1, 2)
    ids = np.tile(np.arange(N, dtype=int), 9)
    return pts, ids, shifts


# ---- Polygon clipping to [0,Lx]×[0,Ly] (Sutherland–Hodgman) ----

def _clip_polygon_to_box(polygon: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
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
            if p2[0] != p1[0] else p1[1]
        ),
    )

    # x <= Lx
    poly = clip(
        poly,
        inside=lambda p: p[0] <= Lx,
        intersect=lambda p1, p2: (
            Lx,
            p1[1] + (p2[1] - p1[1]) * (Lx - p1[0]) / (p2[0] - p1[0])
            if p2[0] != p1[0] else p1[1]
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


# ---- Fully periodic 2D Voronoi on the xy-plane ----

def _compute_periodic_voronoi(coms: np.ndarray, Lx: float, Ly: float):
    """
    coms: (N×2) positions in the primary box [0,Lx]×[0,Ly]
    Returns:
      regions   : dict { i -> (M_i×2) clipped polygon for lipid i }
      neighbors : list of neighbor indices for each lipid
    """
    N = coms.shape[0]
    pts, ids, shifts = _tile_points_pbc(coms, Lx, Ly)
    vor = Voronoi(pts)

    regions = {}

    # only central copies (shift = (0,0)) are indices 0..N-1
    for j in range(N):
        region_i = vor.point_region[j]
        vert_ids = vor.regions[region_i]
        if not vert_ids or any(v < 0 for v in vert_ids):
            # infinite or empty region; skip (should be rare after tiling)
            continue

        poly = np.vstack([vor.vertices[v] for v in vert_ids])
        poly = _clip_polygon_to_box(poly, Lx, Ly)
        if len(poly) == 0:
            continue

        regions[j] = poly

    # Neighbor list for central points only
    neighbors = [[] for _ in range(N)]
    for i1, i2 in vor.ridge_points:
        id1, id2 = ids[i1], ids[i2]
        if id1 == id2:
            continue
        # Only connect when *both* copies belong to the central tile
        if i1 < N and i2 < N:
            if id2 not in neighbors[id1]:
                neighbors[id1].append(id2)
            if id1 not in neighbors[id2]:
                neighbors[id2].append(id1)

    return regions, neighbors



###Voronoi Tesselation with Chol overlay
def plot_predicted_average_voronoi_periodic(
        self,
        title="Periodic Voronoi Lo probability",
        leaflet="both",
        figure_size=(12, 6),
        cmap="viridis"
):
        """
        Plot Voronoi tessellation of average acyl-chain COM positions
        under 2D periodic boundary conditions, colored by mean Lo probability.

        Cholesterol COMs (time-averaged) are overlaid as red markers,
        but are NOT used as Voronoi sites.

        leaflet: "lower", "upper", or "both"
        """

    # 1) Get average DOPC COMs and average probabilities per lipid
        coord_av_lower, prob_av_lower, coord_av_upper, prob_av_upper = \
        self.predict_average_probability()

        # 2) Obtain approximate box lengths Lx, Ly (nm) using Bilayer.box_dimensions
        bilayer = Bilayer(
            self.sys_name,
            self.sys_type,
            self.start_frame_real,
            self.stop_frame_real,
        )
        box_dims = bilayer.box_dimensions()  # shape (n_frames, 3) in nm
        Lx, Ly = box_dims.mean(axis=0)[:2]

        # 3) Prepare figure / axes
        if leaflet == "both":
            fig, axes = plt.subplots(1, 2, figsize=figure_size)
            ax_lower, ax_upper = axes
        elif leaflet == "lower":
            fig, ax_lower = plt.subplots(1, 1, figsize=figure_size)
            ax_upper = None
        elif leaflet == "upper":
            fig, ax_upper = plt.subplots(1, 1, figsize=figure_size)
            ax_lower = None
        else:
            raise ValueError('leaflet must be "lower", "upper", or "both"')

        fig.suptitle(title, fontsize=16)
        norm = Normalize(vmin=0.0, vmax=1.0)

        # 4) Get time-averaged cholesterol COMs (if any)
        chol_lower_av_2d = None
        chol_upper_av_2d = None
        try:
            chol_lower, chol_upper = self.processed_chol_coms
            if chol_lower is not None:
                # chol_lower: (n_frames_red, n_chol, 3)
                chol_lower_av = np.mean(chol_lower, axis=0)  # (n_chol, 3)
                chol_lower_av_2d = chol_lower_av[:, :2]  # (n_chol, 2)
            if chol_upper is not None:
                chol_upper_av = np.mean(chol_upper, axis=0)
                chol_upper_av_2d = chol_upper_av[:, :2]
        except Exception:
            # In pure DOPC systems or if anything goes wrong, just skip CHOL overlay
            chol_lower_av_2d = None
            chol_upper_av_2d = None

        # ---- helper to plot one leaflet ----
        def _plot_leaflet(ax, coords3d, probs, chol_coords2d, label):
            # Use xy-plane for tessellation
            coms_2d = coords3d[:, :2]  # (N×2)

            # Periodic Voronoi for DOPC sites only
            regions, neighbors = _compute_periodic_voronoi(coms_2d, Lx, Ly)

            polys = []
            colors = []
            for idx, poly in regions.items():
                polys.append(poly)
                colors.append(probs[idx])

            poly_coll = PolyCollection(
                polys,
                array=np.array(colors),
                cmap=cmap,
                norm=norm,
                edgecolors="k",
                linewidths=0.2,
            )
            ax.add_collection(poly_coll)

            # Overlay DOPC COMs (small black dots)
            ax.scatter(coms_2d[:, 0], coms_2d[:, 1],
                       s=5, color="k", alpha=0.6)

            # Overlay CHOL COMs if provided
            if chol_coords2d is not None:
                ax.scatter(
                    chol_coords2d[:, 0], chol_coords2d[:, 1],
                    s=20, color="red", marker="x", alpha=0.9,
                    label="CHOL COM"
                )

            ax.set_xlim(0.0, Lx)
            ax.set_ylim(0.0, Ly)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x / nm")
            ax.set_ylabel("y / nm")
            ax.set_title(label)

            # Put legend only if chol is present
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8, frameon=False)

            return poly_coll

        mappable = None

        # Lower leaflet
        if ax_lower is not None:
            mappable = _plot_leaflet(
                ax_lower, coord_av_lower, prob_av_lower,
                chol_lower_av_2d,
                "Lower leaflet"
            )

        # Upper leaflet
        if ax_upper is not None:
            mappable = _plot_leaflet(
                ax_upper, coord_av_upper, prob_av_upper,
                chol_upper_av_2d,
                "Upper leaflet"
            )

        # Shared colorbar
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.8)
            cbar.set_label("Lo probability")

        #plt.tight_layout()
        plt.show()
###End of Voronoi Tesselation with chol overlay
####################### ---- *********************** ---- #######################



