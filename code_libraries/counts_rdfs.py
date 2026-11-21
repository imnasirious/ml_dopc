import numpy as np
import MDAnalysis as mda
import pandas as pd
import os
import matplotlib.pyplot as plt


def acyl_chains_com(array_molecules):  ###specify upper or lower leaflet
    
    sn1_masses = np.array([12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,
        1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1., 12.,  1., 12.,
        1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,
        1., 12.,  1.,  1., 12.,  1.,  1.])

    sn2_masses = np.array([12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,
            1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1., 12.,  1., 12.,
            1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,
            1., 12.,  1.,  1., 12.,  1.,  1.])
    
    sn1_indices = [ 41,  42,  43,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
       101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
       114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
       127, 128, 129, 130, 131, 132, 133]
    
    sn2_indices = [32, 33, 34, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
       58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
       75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
    
    
    sn1_sn2_indices = sn1_indices + sn2_indices
    sn1_sn2_masses = list(sn1_masses) + list(sn2_masses)
    total_sn1_sn2_mass = np.sum(sn1_sn2_masses)
    
    array_molecules = np.swapaxes(array_molecules , 0 , 1)
    n_red = array_molecules.shape[1]
    n_mol = array_molecules.shape[0]
    
    all_coms = np.empty([n_mol, n_red , 3])
    
    all_acyl_chains = array_molecules[: , : , sn1_sn2_indices , :]
    
    #print(all_acyl_chains.shape)
    
    for lipid in range(n_mol):
        
        for frame in range(n_red):
            
            current_acyl_current_frame = all_acyl_chains[lipid , frame]
            current_acyl_current_frame_mass_prod_sum = 0
            
            for atom_index in range(len(sn1_sn2_indices)):
                
                mass_prod = current_acyl_current_frame[atom_index] * sn1_sn2_masses[atom_index]
                current_acyl_current_frame_mass_prod_sum+=mass_prod
                
            all_coms[lipid][frame] = current_acyl_current_frame_mass_prod_sum/total_sn1_sn2_mass
            
    return np.swapaxes(all_coms , 0 , 1)


def chol_com_all(chol_positions):
    chol_masses = np.array([
        12.011,  1.008, 15.999,  1.008, 12.011,  1.008,  1.008, 12.011,
        12.011,  1.008, 12.011,  1.008,  1.008, 12.011,  1.008, 12.011,
        1.008, 12.011,  1.008,  1.008, 12.011,  1.008,  1.008, 12.011,
        1.008, 12.011, 12.011,  1.008,  1.008,  1.008, 12.011,  1.008,
        1.008, 12.011,  1.008,  1.008, 12.011,  1.008, 12.011, 12.011,
        1.008,  1.008,  1.008, 12.011,  1.008,  1.008, 12.011,  1.008,
        1.008, 12.011,  1.008, 12.011,  1.008,  1.008,  1.008, 12.011,
        1.008,  1.008, 12.011,  1.008,  1.008, 12.011,  1.008,  1.008,
        12.011,  1.008, 12.011,  1.008,  1.008,  1.008, 12.011,  1.008,
        1.008,  1.008
    ])

    no_timesteps, no_mols, chol_mol_length, _ = chol_positions.shape

    # Preallocate output array
    all_chol_coms = np.empty((no_timesteps, no_mols, 3))

    for i in range(no_timesteps):
        for j in range(no_mols):
            # Get positions of a single molecule at a single timestep
            chol_pos_one_mol_one_ts = chol_positions[i, j]

            # Compute the weighted sum of positions
            weighted_positions = chol_pos_one_mol_one_ts * chol_masses[:, np.newaxis]
            com = weighted_positions.sum(axis=0) / chol_masses.sum()

            # Store the center of mass
            all_chol_coms[i, j] = com

    return np.swapaxes(all_chol_coms , 0 , 1)



def predictions_cutoff(coord_array, pred_array):

    label_1_all = []
    label_0_all = []

    for f in range(len(coord_array)):

        coord_frame = coord_array[f]
        pred_frame = pred_array[f]

        label_1 = coord_frame[pred_frame == True]
        label_0 = coord_frame[pred_frame == False]

        label_1_all.append(label_1)
        label_0_all.append(label_0)

    return label_1_all, label_0_all


def radial_distance_pbc_2d(target_vector, reference_vector, box_dim_vector):

    # take x,y only
    target_vector = target_vector[:2]
    reference_vector = reference_vector[:2]
    box = box_dim_vector[:2]

    # minimum-image displacement
    rltv_vec = target_vector - reference_vector
    rltv_vec -= box * np.round(rltv_vec / box)

    return np.linalg.norm(rltv_vec), rltv_vec
    
    
def compute_distances_PBC_chol_to_acyl_one_ts(chol_array_one_ts , acyl_array_one_ts , box_dim_vector):
    
    no_chol = chol_array_one_ts.shape[0]
    no_acyl = acyl_array_one_ts.shape[0]
    
    all_pbc_distances = np.empty([no_chol , no_acyl])
    
    for chol_com_index in range(no_chol):
        
        current_chol_com = chol_array_one_ts[chol_com_index]
        chol_distances_one_com = np.empty(no_acyl)
        
        for acyl_com_index in range(no_acyl):
            
            current_acyl_com = acyl_array_one_ts[acyl_com_index]
            
            current_chol_pbc_distance = radial_distance_pbc_2d(current_chol_com, current_acyl_com , box_dim_vector )[0]
            
            chol_distances_one_com[acyl_com_index] = current_chol_pbc_distance
            
        all_pbc_distances[chol_com_index] = np.sort(chol_distances_one_com)
        
        
    return all_pbc_distances


def compute_distances_PBC_chol_to_acyl_all_ts(chol_array, acyl_array, box_dim_vector):

    no_ts = len(chol_array)
    all_pbc_distances_in_time = []

    for ts in range(no_ts):

        # Skip frames with zero valid acyl COMs
        if acyl_array[ts].shape[0] == 0:
            continue

        one_ts_pbc_distances = compute_distances_PBC_chol_to_acyl_one_ts(
            chol_array[ts],
            acyl_array[ts],
            box_dim_vector
        )

        all_pbc_distances_in_time.append(one_ts_pbc_distances)

    return all_pbc_distances_in_time




def count_types_within_radius(chol_array , acyl_array , cutoff , box_dim_vector):
    
    pbc_distances_in_time = compute_distances_PBC_chol_to_acyl_all_ts(chol_array , acyl_array , box_dim_vector)
    total_ts = len(pbc_distances_in_time)
    
    counted_all_ts = np.empty(total_ts)
    
    for ts in range(total_ts):
        
        counted_one_ts = 0
        
        for chol_atom_distance_list in pbc_distances_in_time[ts]:
            
            for one_distance in chol_atom_distance_list:
                
                if one_distance <= cutoff:
                    
                    counted_one_ts +=1
                    
        counted_all_ts[ts] = counted_one_ts
        
    return counted_all_ts   



def RDF_calculator_onelist_diff_species(sorted_distances_one_atom , dr , cutoff_distance , box_dim_vector):
    
    radial_distances = np.arange(0 , cutoff_distance , dr)
    
    outer_radius = 0
    inner_radius = 0
    
    density_global = sorted_distances_one_atom.shape[0]/(box_dim_vector[0]*box_dim_vector[1])
    
    hit_list = []
    
    for i in range(len(radial_distances)):
        
        outer_radius += dr
        shell_volume = np.pi  * (outer_radius**2 - inner_radius**2)
        hits = 0
        
        for j in range(len(sorted_distances_one_atom)):
            
            if (sorted_distances_one_atom[j] > inner_radius) and sorted_distances_one_atom[j] < outer_radius:
                
                hits += 1
        
        #hit_list.append(hits/(shell_volume/dopc_density_global))
        hit_list.append(hits/(shell_volume*density_global))
        
        inner_radius = outer_radius
    
    return radial_distances, hit_list




def RDF_over_all_atoms_one_frame_diff_species(pbc_sorted_distances  , dr , cutoff_distance, box_dim_vector):
    
    unweighted_rdfs = []
    
    for i in range(len(pbc_sorted_distances)):
        
        one_atom_rdf = RDF_calculator_onelist_diff_species(pbc_sorted_distances[i]  , dr , cutoff_distance, box_dim_vector)[1]
        unweighted_rdfs.append(np.array(one_atom_rdf))
    
    weighted_rdfs = []
    
    for j in range(len(unweighted_rdfs[0])):
        
        one_sum = 0
        
        for k in range(len(unweighted_rdfs)):
            
            one_sum += unweighted_rdfs[k][j]
            
        one_sum = one_sum / len(unweighted_rdfs)
        
        weighted_rdfs.append(one_sum)
        
    return weighted_rdfs


def RDF_over_all_frames_diff_species(vec_timesteps , dr , cutoff_distance, box_dim_vector):
    
    time_unaveraged_RDFS = []
    
    for i in range(len(vec_timesteps)):
        
        rdf_one_timestep = RDF_over_all_atoms_one_frame_diff_species(vec_timesteps[i], dr , cutoff_distance, box_dim_vector)
        time_unaveraged_RDFS.append(rdf_one_timestep)
        
    time_averaged_RDF = []
    
    for j in range(len(time_unaveraged_RDFS[0])):
        
        one_sum = 0
        
        for k in range(len(time_unaveraged_RDFS)):
            
            one_sum += time_unaveraged_RDFS[k][j]
        
        one_sum = one_sum / len(time_unaveraged_RDFS)
        
        time_averaged_RDF.append(one_sum)
        
    
        
    return time_averaged_RDF



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










                
                
    



