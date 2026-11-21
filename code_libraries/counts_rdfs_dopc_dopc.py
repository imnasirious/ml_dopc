# counts_rdf_dopc_dopc.py
import numpy as np

###############################################################################
# DOPC–DOPC (same species) RDF / neighbor counting utilities (2D PBC in xy)
#
# Analogous to counts_rdfs.py (which is for different species, e.g., CHOL→DOPC),
# but for SAME-SPECIES distances (e.g., DOPC→DOPC).
#
# It:
#   1) Computes 2D minimum-image distances in the xy plane under PBC
#   2) Excludes self-distances (i == j)
#   3) Produces per-reference sorted distance lists (N x (N-1)) per frame
#   4) Provides RDF calculators and a neighbor-count-within-radius helper
#
# Input convention:
#   - coord arrays per frame should be (N, 3) or (N, 2)
#   - time series can be:
#       a) numpy array shape (T, N, 3) OR
#       b) list of arrays, each shape (N_t, 3) where N_t can vary by frame
#
# Units:
#   - Box dims and coordinates must be in the SAME units (nm or Å).
###############################################################################


def radial_distance_pbc_2d(target_vector, reference_vector, box_dim_vector):
    """2D minimum-image distance (xy only) and displacement vector."""
    target_vector = np.asarray(target_vector)[:2]
    reference_vector = np.asarray(reference_vector)[:2]
    box_dim_vector = np.asarray(box_dim_vector)[:2]

    rltv_vec = target_vector - reference_vector
    rltv_vec -= box_dim_vector * np.round(rltv_vec / box_dim_vector)
    return np.linalg.norm(rltv_vec), rltv_vec


def compute_distances_PBC_dopc_to_dopc_one_ts(dopc_array_one_ts, box_dim_vector):
    """
    For one timestep:
      dopc_array_one_ts: (N,3) or (N,2)
    Returns:
      all_pbc_distances: (N, N-1) array where row i is sorted distances from i to all j!=i.
    """
    dopc_array_one_ts = np.asarray(dopc_array_one_ts)
    box_dim_vector = np.asarray(box_dim_vector)[:2]

    n = dopc_array_one_ts.shape[0]
    if n < 2:
        return np.empty((0, 0))

    all_pbc_distances = np.empty((n, n - 1), dtype=float)

    for i in range(n):
        ref = dopc_array_one_ts[i]
        dists = np.empty(n - 1, dtype=float)

        k = 0
        for j in range(n):
            if j == i:
                continue
            tgt = dopc_array_one_ts[j]
            dists[k] = radial_distance_pbc_2d(tgt, ref, box_dim_vector)[0]
            k += 1

        all_pbc_distances[i] = np.sort(dists)

    return all_pbc_distances


def compute_distances_PBC_dopc_to_dopc_all_ts(dopc_array, box_dim_vector):
    """
    For all timesteps:
      dopc_array: (T,N,3)/(T,N,2) OR list of arrays [ (N_t,3), (N_t,3), ... ]
    Returns:
      list of (N_t, N_t-1) arrays, skipping frames with N_t < 2
    """
    box_dim_vector = np.asarray(box_dim_vector)[:2]

    # Normalize input to iterable of frames
    if isinstance(dopc_array, np.ndarray):
        frames = [dopc_array[t] for t in range(dopc_array.shape[0])]
    else:
        frames = list(dopc_array)

    all_pbc_distances_in_time = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.shape[0] < 2:
            continue
        all_pbc_distances_in_time.append(
            compute_distances_PBC_dopc_to_dopc_one_ts(frame, box_dim_vector)
        )

    return all_pbc_distances_in_time


def count_pairs_within_radius_same_species(dopc_array, cutoff, box_dim_vector):
    """
    Counts ordered pairs (i->j, i!=j) within cutoff for each frame.
    This matches the per-reference distance-list representation (N x (N-1)).

    Returns:
      counted_all_ts: (n_frames_used,) array of counts per frame.
    """
    pbc_distances_in_time = compute_distances_PBC_dopc_to_dopc_all_ts(dopc_array, box_dim_vector)
    total_ts = len(pbc_distances_in_time)
    counted_all_ts = np.empty(total_ts, dtype=float)

    for ts in range(total_ts):
        dist_mat = pbc_distances_in_time[ts]  # (N, N-1)
        counted_all_ts[ts] = np.sum(dist_mat <= cutoff)

    return counted_all_ts


def RDF_calculator_onelist_same_species(sorted_distances_one_atom, dr, cutoff_distance, box_dim_vector):
    """
    RDF for a single reference atom i using distances to all OTHER atoms (N-1 distances).
    Uses 2D shell area normalization and global 'other-particle' density.

    sorted_distances_one_atom: shape (N-1,)
    """
    radial_distances = np.arange(0, cutoff_distance, dr)

    outer_radius = 0.0
    inner_radius = 0.0

    n_other = sorted_distances_one_atom.shape[0]
    area = box_dim_vector[0] * box_dim_vector[1]
    density_global = n_other / area  # (N-1)/A

    hit_list = []
    for _ in radial_distances:
        outer_radius += dr
        shell_area = np.pi * (outer_radius ** 2 - inner_radius ** 2)

        # count hits in (inner, outer)
        hits = np.sum((sorted_distances_one_atom > inner_radius) &
                      (sorted_distances_one_atom < outer_radius))

        hit_list.append(hits / (shell_area * density_global))
        inner_radius = outer_radius

    return radial_distances, hit_list


def RDF_over_all_atoms_one_frame_same_species(pbc_sorted_distances, dr, cutoff_distance, box_dim_vector):
    """
    Average RDF over all reference atoms in a single frame.
    pbc_sorted_distances: (N, N-1)
    Returns:
      list of rdf values (length = cutoff_distance/dr)
    """
    unweighted_rdfs = []
    for i in range(len(pbc_sorted_distances)):
        one_atom_rdf = RDF_calculator_onelist_same_species(
            pbc_sorted_distances[i], dr, cutoff_distance, box_dim_vector
        )[1]
        unweighted_rdfs.append(np.array(one_atom_rdf))

    # average over atoms
    return np.mean(np.vstack(unweighted_rdfs), axis=0).tolist()


def RDF_over_all_frames_same_species(vec_timesteps, dr, cutoff_distance, box_dim_vector):
    """
    Time-averaged RDF across frames.
    vec_timesteps: list of (N_t, N_t-1) distance matrices (from compute_distances_*_all_ts)
    Returns:
      list of rdf values (length = cutoff_distance/dr)
    """
    time_unaveraged_rdfs = []
    for i in range(len(vec_timesteps)):
        rdf_one_timestep = RDF_over_all_atoms_one_frame_same_species(
            vec_timesteps[i], dr, cutoff_distance, box_dim_vector
        )
        time_unaveraged_rdfs.append(np.array(rdf_one_timestep))

    return np.mean(np.vstack(time_unaveraged_rdfs), axis=0).tolist()

def compute_distances_PBC_two_sets_one_ts(A_one_ts, B_one_ts, box_dim_vector, exclude_paired=False):
    """
    A_one_ts: (NA,3) or (NA,2)   reference set (e.g., P)
    B_one_ts: (NB,3) or (NB,2)   target set (e.g., N)
    exclude_paired: if True, exclude A[i]–B[i] distances for i < min(NA, NB)
                    (useful when A and B are same lipids in same order)

    Returns:
        all_pbc_distances: (NA, NB') array where each row i is sorted distances from A[i] to all allowed B[j]
                           If exclude_paired=True, NB' = NB-1 (when NA==NB and aligned); otherwise NB' = NB
    """
    A_one_ts = np.asarray(A_one_ts)
    B_one_ts = np.asarray(B_one_ts)
    box_dim_vector = np.asarray(box_dim_vector)[:2]

    NA = A_one_ts.shape[0]
    NB = B_one_ts.shape[0]
    if NA == 0 or NB == 0:
        return np.empty((0, 0))

    if exclude_paired:
        # If sets are aligned (same lipids), we drop one element per row when possible
        NB_eff = NB - 1 if NB >= 2 else 0
        all_pbc_distances = np.empty((NA, NB_eff), dtype=float)

        for i in range(NA):
            ref = A_one_ts[i]
            dists = []
            for j in range(NB):
                if i < min(NA, NB) and j == i:
                    continue  # exclude intramolecular pair
                tgt = B_one_ts[j]
                dists.append(radial_distance_pbc_2d(tgt, ref, box_dim_vector)[0])
            all_pbc_distances[i] = np.sort(np.asarray(dists, float))

        return all_pbc_distances

    else:
        all_pbc_distances = np.empty((NA, NB), dtype=float)
        for i in range(NA):
            ref = A_one_ts[i]
            dists = np.empty(NB, dtype=float)
            for j in range(NB):
                tgt = B_one_ts[j]
                dists[j] = radial_distance_pbc_2d(tgt, ref, box_dim_vector)[0]
            all_pbc_distances[i] = np.sort(dists)
        return all_pbc_distances


def compute_distances_PBC_two_sets_all_ts(A_all_ts, B_all_ts, box_dim_vector, exclude_paired=False):
    """
    Accepts (T, N, 3) arrays or lists of per-frame arrays.
    Returns list of per-frame distance matrices.
    """
    box_dim_vector = np.asarray(box_dim_vector)[:2]

    if isinstance(A_all_ts, np.ndarray):
        framesA = [A_all_ts[t] for t in range(A_all_ts.shape[0])]
    else:
        framesA = list(A_all_ts)

    if isinstance(B_all_ts, np.ndarray):
        framesB = [B_all_ts[t] for t in range(B_all_ts.shape[0])]
    else:
        framesB = list(B_all_ts)

    if len(framesA) != len(framesB):
        raise ValueError("A_all_ts and B_all_ts must have the same number of frames")

    out = []
    for A_one_ts, B_one_ts in zip(framesA, framesB):
        A_one_ts = np.asarray(A_one_ts)
        B_one_ts = np.asarray(B_one_ts)
        if A_one_ts.shape[0] == 0 or B_one_ts.shape[0] == 0:
            continue
        out.append(compute_distances_PBC_two_sets_one_ts(A_one_ts, B_one_ts, box_dim_vector, exclude_paired=exclude_paired))
    return out