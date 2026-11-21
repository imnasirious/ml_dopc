"""
classified_analysis.py
----------------------
Fast (numpy) versions of the calculations used by the analysis notebooks after classification
(order-parameter profiles, chain-CHOL RDFs, headgroup-CHOL RDFs).

Every function here reproduces an existing library function, only without Python loops over frames and
molecules, so 4000 frames run in minutes instead of hours:

    order_parameters        <-> acyl_chain_preprocessing.DOPC_op_H_av      (per lipid, per frame)
    acyl_chain_coms         <-> counts_rdfs_4.acyl_chain_one_com
    chol_coms               <-> counts_rdfs_4.chol_com_all
    pbc_distances_xy        <-> counts_rdfs_4.compute_distances_PBC_chol_to_acyl_all_ts  /  counts_rdfs_3...acyl_to_chol...
    running_mean            <-> counts_rdfs_4.convolve_pbc_list, op_by_chain.convolve_ops, convolve_positions
    rdf_selected            <-> counts_rdfs_*.RDF_over_all_frames_diff_species (after selecting Lo / Ld)

The notebooks check each one against the original on a few frames before using it.

Array conventions: leaflet positions (frames, lipids, atoms, 3) in nm; per-molecule arrays (frames, molecules, ...).
"""

import os
import numpy as np


# --------------------------------------------------------------------------------------------- order parameters
# atom indices (within one DOPC) of each chain, same lists as DOPC_op_H_av
SN1_ATOMS = np.array([41, 42, 43] + list(range(91, 134)))
SN2_ATOMS = np.array([32, 33, 34] + list(range(44, 87)))

# for C1 ... C16: position (inside the 46-atom chain list) of the carbon and of its H atoms
_C_H = [(0, (1, 2)), (3, (4, 5)), (6, (7, 8)), (9, (10, 11)), (12, (13, 14)), (15, (16, 17)), (18, (19, 20)),
        (21, (22,)), (23, (24,)),
        (25, (26, 27)), (28, (29, 30)), (31, (32, 33)), (34, (35, 36)), (37, (38, 39)), (40, (41, 42)), (43, (44, 45))]


def order_parameters(leaflet, chain, chunk=250):
    """S_CH of C1..C16 for every frame and lipid: (frames, lipids, 16). Same definition as DOPC_op_H_av
    (bilayer normal = z, -(3 cos^2 - 1)/2, averaged over the H atoms of each carbon)."""
    atoms = SN1_ATOMS if chain == "sn1" else SN2_ATOMS
    n_f, n_l = leaflet.shape[:2]
    out = np.empty((n_f, n_l, 16))
    for s in range(0, n_f, chunk):
        a = np.asarray(leaflet[s:s + chunk][:, :, atoms, :], dtype=float)
        for k, (ci, hs) in enumerate(_C_H):
            vals = []
            for hi in hs:
                v = a[:, :, ci, :] - a[:, :, hi, :]
                cos2 = (v[..., 2] ** 2) / np.sum(v ** 2, axis=-1)
                vals.append(-(3 * cos2 - 1) / 2)
            out[s:s + chunk, :, k] = np.mean(vals, axis=0)
    return out


# --------------------------------------------------------------------------------------------- centres of mass
_ACYL_MASSES = np.array([12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1.,
                         12., 1., 12., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1.,
                         12., 1., 1., 12., 1., 1.])

_CHOL_MASSES = np.array([
    12.011, 1.008, 15.999, 1.008, 12.011, 1.008, 1.008, 12.011, 12.011, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008,
    12.011, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 12.011, 12.011, 1.008, 1.008, 1.008,
    12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 12.011, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008,
    1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008,
    1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008])


def acyl_chain_coms(leaflet, chain):
    """COM of one acyl chain (same 46 atoms and masses as acyl_chain_one_com): (frames, lipids, 3)."""
    atoms = SN1_ATOMS if chain == "sn1" else SN2_ATOMS
    w = _ACYL_MASSES / _ACYL_MASSES.sum()
    return np.einsum("flai,a->fli", leaflet[:, :, atoms, :], w)


def chol_coms(chol_leaflet):
    """COM of every cholesterol (same masses as chol_com_all): (frames, chol, 3)."""
    w = _CHOL_MASSES / _CHOL_MASSES.sum()
    return np.einsum("fcai,a->fci", chol_leaflet, w)


# --------------------------------------------------------------------------------------------- time averaging
def running_mean(arr, n_point):
    """n-point running average along axis 0 (time), 'valid' mode: length frames - n + 1.
    Same as np.convolve(x, ones(n)/n, 'valid') for every other index."""
    if n_point in (None, 1, "raw"):
        return np.asarray(arr)
    a = np.asarray(arr, dtype=float)
    cs = np.cumsum(a, axis=0)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), cs], axis=0)
    return (cs[n_point:] - cs[:-n_point]) / n_point


def unwrap_xy(pos, box_xy):
    """Remove periodic jumps in x, y along time (frames, molecules, 3), so positions can be time-averaged."""
    p = np.array(pos, dtype=float)
    d = np.diff(p[..., :2], axis=0)
    d -= box_xy * np.round(d / box_xy)
    p[1:, ..., :2] = p[0, ..., :2] + np.cumsum(d, axis=0)
    return p


# --------------------------------------------------------------------------------------------- distances and RDFs
def pbc_distances_xy(ref, tgt, box_xy, dtype=np.float32):
    """Minimum-image xy distances for every frame: (frames, n_ref, n_tgt)."""
    n_f = ref.shape[0]
    out = np.empty((n_f, ref.shape[1], tgt.shape[1]), dtype=dtype)
    for s in range(0, n_f, 200):
        d = ref[s:s + 200, :, None, :2] - tgt[s:s + 200, None, :, :2]
        d -= box_xy * np.round(d / box_xy)
        out[s:s + 200] = np.sqrt(np.sum(d ** 2, axis=-1))
    return out


def rdf_bin_edges(dr, cutoff):
    """Shell edges exactly as RDF_calculator_onelist_diff_species builds them."""
    n_bins = len(np.arange(0, cutoff, dr))
    edges, r = [0.0], 0.0
    for _ in range(n_bins):
        r += dr
        edges.append(r)
    return np.array(edges)


def rdf_selected(dist, dr, cutoff, box_xy, ref_mask=None, tgt_mask=None):
    """
    2D g(r) between the selected reference and target molecules, averaged over references in each frame and then
    over frames; identical to RDF_over_all_frames_diff_species applied to the Lo/Ld-selected distance arrays.

    dist     : (frames, n_ref, n_tgt) distances
    ref_mask : (frames, n_ref) bool, references to keep in each frame (None = all)
    tgt_mask : (frames, n_tgt) bool, targets to keep in each frame (None = all)
    Frames with no reference or no target are skipped. Density = selected targets / box area.
    """
    edges = rdf_bin_edges(dr, cutoff)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    area = box_xy[0] * box_xy[1]
    g_sum = np.zeros(len(shell))
    n_used = 0
    for f in range(dist.shape[0]):
        d = dist[f]
        if ref_mask is not None:
            d = d[ref_mask[f]]
        if tgt_mask is not None:
            d = d[:, tgt_mask[f]]
        n_ref, n_tgt = d.shape
        if n_ref == 0 or n_tgt == 0:
            continue
        flat = np.sort(d.ravel())
        # strictly inside (inner, outer), as in the original shell counting
        counts = np.searchsorted(flat, edges[1:], side="left") - np.searchsorted(flat, edges[:-1], side="right")
        g_sum += counts / n_ref / (shell * n_tgt / area)
        n_used += 1
    return g_sum / n_used if n_used else np.full(len(shell), np.nan)


# --------------------------------------------------------------------------------------------- positions
def load_leaflets(sys_name, start_frame, stop_frame, sys_type="sym"):
    """(dopc_lower, dopc_upper, chol_lower, chol_upper) from position_npys, without opening the trajectory."""
    from code_libraries.bilayer_operations import Bilayer
    bil = Bilayer(sys_name, sys_type, start_frame, stop_frame)
    bil.cleave_leaflets()
    low, up = bil.lower_leaflet, bil.upper_leaflet
    if isinstance(low, tuple):
        return low[0], up[0], low[1], up[1]
    return low, up, None, None


def cached(path, compute):
    """Load an .npy cache if it exists, otherwise compute, save and return it."""
    if os.path.exists(path):
        return np.load(path)
    arr = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return arr
