# counts_rdfs_fast.py
"""
Vectorized drop-in replacements for:
    - counts_rdfs.py
    - counts_rdfs_dopc_dopc.py

Key speedup: all pairwise distances per frame are computed with numpy
broadcasting instead of nested Python for-loops. The RDF binning also
uses np.histogram instead of looping over shells.

Public API (matches the original function names):

    # diff-species  (e.g. CHOL → DOPC)
    compute_distances_PBC_chol_to_acyl_all_ts(A, B, box)
    RDF_over_all_frames_diff_species(dist_list, dr, cutoff, box)

    # same-species  (e.g. DOPC P → DOPC P)
    compute_distances_PBC_dopc_to_dopc_all_ts(A, box)
    RDF_over_all_frames_same_species(dist_list, dr, cutoff, box)

    # two-sets with optional intramolecular exclusion (e.g. DOPC P → DOPC N)
    compute_distances_PBC_two_sets_all_ts(A, B, box, exclude_paired)
    RDF_over_all_frames_diff_species(...)   ← same function, reused

Input shapes
------------
    A, B  : (n_frames, n_molecules, 3)  numpy arrays
    box   : (2,)  mean xy box dimensions  [same units as coordinates]
"""

import numpy as np


# ============================================================
# LOW-LEVEL: vectorized pairwise 2D-PBC distance for one frame
# ============================================================

def _pairwise_pbc_2d(A, B, box):
    """
    All pairwise 2D minimum-image distances between rows of A and B.

    A   : (NA, 3) or (NA, 2)
    B   : (NB, 3) or (NB, 2)
    box : (2,)

    Returns : (NA, NB)  unsorted distance matrix
    """
    a = A[:, :2]                                    # (NA, 2)
    b = B[:, :2]                                    # (NB, 2)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (NA, NB, 2)
    diff -= box * np.round(diff / box)              # minimum image
    return np.sqrt((diff ** 2).sum(axis=-1))        # (NA, NB)


# ============================================================
# DISTANCE CALCULATORS
# ============================================================

def compute_distances_PBC_chol_to_acyl_all_ts(chol_array, acyl_array, box_dim_vector):
    """
    Diff-species distances: CHOL → DOPC (or any A → B).

    chol_array  : (n_frames, n_chol, 3)
    acyl_array  : (n_frames, n_acyl, 3)
    box_dim_vector : (2,)

    Returns : list of (n_chol, n_acyl) sorted distance matrices, one per frame.
              Frames where n_acyl == 0 are skipped.
    """
    box = np.asarray(box_dim_vector)[:2]
    n_frames = chol_array.shape[0]
    out = []

    for t in range(n_frames):
        A = chol_array[t]   # (n_chol, 3)
        B = acyl_array[t]   # (n_acyl, 3)

        if B.shape[0] == 0:
            continue

        dist = _pairwise_pbc_2d(A, B, box)     # (n_chol, n_acyl)
        dist = np.sort(dist, axis=1)            # sort along acyl axis
        out.append(dist)

    return out


def compute_distances_PBC_dopc_to_dopc_all_ts(dopc_array, box_dim_vector):
    """
    Same-species distances: DOPC → DOPC (self excluded).

    dopc_array     : (n_frames, n_dopc, 3)
    box_dim_vector : (2,)

    Returns : list of (n_dopc, n_dopc-1) sorted distance matrices, one per frame.
              Frames with fewer than 2 lipids are skipped.
    """
    box = np.asarray(box_dim_vector)[:2]
    n_frames = dopc_array.shape[0]
    out = []

    for t in range(n_frames):
        A = dopc_array[t]   # (n_dopc, 3)
        n = A.shape[0]

        if n < 2:
            continue

        dist = _pairwise_pbc_2d(A, A, box)     # (n, n)  — diagonal = 0 (self)

        # remove self-distance diagonal → each row becomes (n-1,) sorted distances
        mask = ~np.eye(n, dtype=bool)
        dist_no_self = dist[mask].reshape(n, n - 1)
        dist_no_self = np.sort(dist_no_self, axis=1)
        out.append(dist_no_self)

    return out


def compute_distances_PBC_two_sets_all_ts(A_all, B_all, box_dim_vector, exclude_paired=False):
    """
    Two-set distances: A → B (diff species, e.g. DOPC-P → DOPC-N).

    A_all, B_all   : (n_frames, n_mol, 3)
    box_dim_vector : (2,)
    exclude_paired : if True, exclude A[i]–B[i] pairs (intramolecular)

    Returns : list of (NA, NB or NB-1) sorted distance matrices, one per frame.
    """
    box = np.asarray(box_dim_vector)[:2]

    if A_all.shape[0] != B_all.shape[0]:
        raise ValueError("A_all and B_all must have the same number of frames")

    n_frames = A_all.shape[0]
    out = []

    for t in range(n_frames):
        A = A_all[t]    # (NA, 3)
        B = B_all[t]    # (NB, 3)

        if A.shape[0] == 0 or B.shape[0] == 0:
            continue

        dist = _pairwise_pbc_2d(A, B, box)     # (NA, NB)

        if exclude_paired:
            # remove intramolecular A[i]-B[i] diagonal entries
            NA, NB = dist.shape
            n_pairs = min(NA, NB)
            # build mask: True = keep
            mask = np.ones((NA, NB), dtype=bool)
            mask[np.arange(n_pairs), np.arange(n_pairs)] = False

            # each row i loses one element → (NA, NB-1) when NA==NB
            NB_eff = NB - 1
            dist_excl = np.empty((NA, NB_eff), dtype=float)
            for i in range(NA):
                dist_excl[i] = np.sort(dist[i][mask[i]])
            out.append(dist_excl)
        else:
            out.append(np.sort(dist, axis=1))

    return out


# ============================================================
# RDF CALCULATORS
# ============================================================

def _rdf_one_frame_diff_species(dist_matrix, dr, cutoff, box):
    """
    RDF for one frame, diff-species.

    dist_matrix : (N_ref, N_target) sorted distances
    Returns     : (n_bins,) rdf array
    """
    bins = np.arange(0, cutoff + dr, dr)
    n_bins = len(bins) - 1
    n_ref = dist_matrix.shape[0]
    n_target = dist_matrix.shape[1]
    area = box[0] * box[1]
    density = n_target / area           # global 2D density of target species

    # flatten all distances, histogram them
    all_dists = dist_matrix.flatten()
    counts, _ = np.histogram(all_dists, bins=bins)

    # shell area for each bin
    r_inner = bins[:-1]
    r_outer = bins[1:]
    shell_area = np.pi * (r_outer ** 2 - r_inner ** 2)

    # average over reference atoms, normalise
    rdf = counts / (n_ref * shell_area * density)

    return rdf


def _rdf_one_frame_same_species(dist_matrix, dr, cutoff, box):
    """
    RDF for one frame, same-species (self already excluded).

    dist_matrix : (N, N-1) sorted distances
    Returns     : (n_bins,) rdf array
    """
    bins = np.arange(0, cutoff + dr, dr)
    n_bins = len(bins) - 1
    n = dist_matrix.shape[0]            # number of reference atoms
    n_other = dist_matrix.shape[1]      # N - 1
    area = box[0] * box[1]
    density = n_other / area            # (N-1)/A  — same as original

    all_dists = dist_matrix.flatten()
    counts, _ = np.histogram(all_dists, bins=bins)

    r_inner = bins[:-1]
    r_outer = bins[1:]
    shell_area = np.pi * (r_outer ** 2 - r_inner ** 2)

    rdf = counts / (n * shell_area * density)

    return rdf


def RDF_over_all_frames_diff_species(dist_list, dr, cutoff_distance, box_dim_vector):
    """
    Time-averaged RDF, diff-species (also used for two-set P→N).

    dist_list      : list of per-frame distance matrices
    Returns        : (n_bins,) numpy array
    """
    box = np.asarray(box_dim_vector)[:2]
    rdfs = []

    for dist_matrix in dist_list:
        if dist_matrix.shape[0] == 0 or dist_matrix.shape[1] == 0:
            continue
        rdfs.append(_rdf_one_frame_diff_species(dist_matrix, dr, cutoff_distance, box))

    return np.mean(np.vstack(rdfs), axis=0)


def RDF_over_all_frames_same_species(dist_list, dr, cutoff_distance, box_dim_vector):
    """
    Time-averaged RDF, same-species.

    dist_list      : list of per-frame (N, N-1) distance matrices
    Returns        : (n_bins,) numpy array
    """
    box = np.asarray(box_dim_vector)[:2]
    rdfs = []

    for dist_matrix in dist_list:
        if dist_matrix.shape[0] == 0:
            continue
        rdfs.append(_rdf_one_frame_same_species(dist_matrix, dr, cutoff_distance, box))

    return np.mean(np.vstack(rdfs), axis=0)