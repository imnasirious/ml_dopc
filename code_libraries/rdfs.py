import numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt

def radial_distance(reference_vector , target_vector): #returns vector between two points its distance
    rltv_vec_x = reference_vector[0] - target_vector[0]
    rltv_vec_y = reference_vector[1] - target_vector[1]
    #rltv_vec_z = reference_vector[2] - target_vector[2]
    distance_squared = rltv_vec_x**2 + rltv_vec_y**2
    return np.array([rltv_vec_x , rltv_vec_y]) ,np.sqrt(distance_squared)# , target_vector


def compute_distances_with_PBCS(atom_list , xbox_dim , ybox_dim):
    
    list_of_distances = [] ##will store PBC corrected distances
    
    total_atoms = len(atom_list)
    
    for i in range(total_atoms): 
        
        distances_from_reference = []  ##list to store distances from ith reference atom in outer loop
        reference_atom = atom_list[i]
        
        
        for j in range(total_atoms): 
            
            target_atom = atom_list[j]
            
            vector_and_distance = radial_distance(reference_atom , target_atom)
            ##this variable stores the radial distance/relative vector information for atoms
            ##i and j
            
            #the PBC calculation as described above is implemented here
            
            #first check to see if ALL the vector compoonents joining two atoms are less than L/2
            
            if (abs(vector_and_distance[0][0]) <= 0.5*xbox_dim) and (abs(vector_and_distance[0][1]) <= 0.5*ybox_dim):
                
                distance = vector_and_distance[1]
                distances_from_reference.append(distance)
            
            ##if not, then correct accordingly for each x,y and z coordinate
            
            else:
                
                if abs(vector_and_distance[0][0]) > 0.5*xbox_dim:
                    
                    vector_and_distance[0][0] = abs(vector_and_distance[0][0]) - xbox_dim
                    
                else:
                    
                    vector_and_distance[0][0] += 0
                    
                if abs(vector_and_distance[0][1]) > 0.5*ybox_dim:
                    
                    vector_and_distance[0][1] = abs(vector_and_distance[0][1]) - ybox_dim
                    
                else:
                    
                    vector_and_distance[0][1] += 0
                    
                    
                
                #when(and if) coordinates need to fixed, use a recalculated distance with updated
                #information
                
                recalculated_distance = np.sqrt(vector_and_distance[0][0]**2 + vector_and_distance[0][1]**2)
                distances_from_reference.append(recalculated_distance)
        
        list_of_distances.append(distances_from_reference)
    
    return np.array(list_of_distances)


def RDF_calculator_onelist(sorted_distances_one_atom , dr , cutoff_distance , density_global):
    
    radial_distances = np.arange(0 , cutoff_distance , dr)
    
    outer_radius = dr
    inner_radius = dr
    
    
    hit_list = []
    
    for i in range(len(radial_distances)):
        
        outer_radius += dr
        shell_volume = np.pi * (outer_radius**2 - inner_radius**2)
        hits = 0
        
        for j in range(len(sorted_distances_one_atom)):
            
            if (sorted_distances_one_atom[j] > inner_radius) and sorted_distances_one_atom[j] <= outer_radius:
                
                hits += 1
        
        #hit_list.append(hits/(shell_volume/dopc_density_global))
        hit_list.append(hits/(shell_volume*density_global))
        
        inner_radius = outer_radius
    
    return radial_distances, hit_list
            

    
 #function takes in a whole list of all sorted atom vectors, i.e some_distances.
#this corresponds to one time step of data
    
def RDF_over_all_atoms_one_frame(sorted_distances , dr , cutoff_distance, density_global):
    
    unweighted_rdfs = []
    
    for i in range(len(sorted_distances)):
        
        one_atom_rdf = RDF_calculator_onelist(sorted_distances[i] , dr , cutoff_distance, density_global)[1]
        unweighted_rdfs.append(np.array(one_atom_rdf))
    
    weighted_rdfs = []
    
    for j in range(len(unweighted_rdfs[0])):
        
        one_sum = 0
        
        for k in range(len(unweighted_rdfs)):
            
            one_sum += unweighted_rdfs[k][j]
            
        one_sum = one_sum / len(unweighted_rdfs)
        
        weighted_rdfs.append(one_sum)
        
    return weighted_rdfs


def RDF_over_all_frames(vec_timesteps , dr , cutoff_distance, density_global):
    
    time_unaveraged_RDFS = []
    
    for i in range(len(vec_timesteps)):
        
        rdf_one_timestep = RDF_over_all_atoms_one_frame(vec_timesteps[i] , dr , cutoff_distance, density_global)
        time_unaveraged_RDFS.append(rdf_one_timestep)
        
    time_averaged_RDF = []
    
    for j in range(len(time_unaveraged_RDFS[0])):
        
        one_sum = 0
        
        for k in range(len(time_unaveraged_RDFS)):
            
            one_sum += time_unaveraged_RDFS[k][j]
        
        one_sum = one_sum / len(time_unaveraged_RDFS)
        
        time_averaged_RDF.append(one_sum)
        
    
        
    return time_averaged_RDF
             


def compute_distances_PBC_diff_species(ref1 , ref2 , xbox_dim , ybox_dim):
    
    list_of_distances = []
    
    atoms_ref1 = len(ref1)
    atoms_ref2 = len(ref2)
    
    for i in range(atoms_ref1):
        
        distances_from_reference = []
        reference_atom = ref1[i]
        
        for j in range(atoms_ref2):
            
            target_atom = ref2[j]
            vector_and_distance = radial_distance(reference_atom , target_atom)
            
            if (abs(vector_and_distance[0][0]) <= 0.5*xbox_dim) and (abs(vector_and_distance[0][1]) <= 0.5*ybox_dim):
                
                distance = vector_and_distance[1]
                distances_from_reference.append(distance)
            
            ##if not, then correct accordingly for each x,y and z coordinate
            
            else:
                
                if abs(vector_and_distance[0][0]) > 0.5*xbox_dim:
                    
                    vector_and_distance[0][0] = abs(vector_and_distance[0][0]) - xbox_dim
                    
                else:
                    
                    vector_and_distance[0][0] += 0
                    
                if abs(vector_and_distance[0][1]) > 0.5*ybox_dim:
                    
                    vector_and_distance[0][1] = abs(vector_and_distance[0][1]) - ybox_dim
                    
                else:
                    
                    vector_and_distance[0][1] += 0
                    
                
                #when(and if) coordinates need to fixed, use a recalculated distance with updated
                #information
                
                recalculated_distance = np.sqrt(vector_and_distance[0][0]**2 + vector_and_distance[0][1]**2)
                distances_from_reference.append(recalculated_distance)
        
        list_of_distances.append(np.sort(distances_from_reference))
        
    return np.array(list_of_distances)
    
    
def RDF_calculator_onelist_diff_species(sorted_distances_one_atom , dr , cutoff_distance , density_global):
    
    radial_distances = np.arange(0 , cutoff_distance , dr)
    
    outer_radius = 0
    inner_radius = 0
    
    
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
            

    
 #function takes in a whole list of all sorted atom vectors, i.e some_distances.
#this corresponds to one time step of data
    
def RDF_over_all_atoms_one_frame_diff_species(sorted_distances  , dr , cutoff_distance, density_global):
    
    unweighted_rdfs = []
    
    for i in range(len(sorted_distances)):
        
        one_atom_rdf = RDF_calculator_onelist(sorted_distances[i]  , dr , cutoff_distance, density_global)[1]
        unweighted_rdfs.append(np.array(one_atom_rdf))
    
    weighted_rdfs = []
    
    for j in range(len(unweighted_rdfs[0])):
        
        one_sum = 0
        
        for k in range(len(unweighted_rdfs)):
            
            one_sum += unweighted_rdfs[k][j]
            
        one_sum = one_sum / len(unweighted_rdfs)
        
        weighted_rdfs.append(one_sum)
        
    return weighted_rdfs


def RDF_over_all_frames_diff_species(vec_timesteps , dr , cutoff_distance, density_global):
    
    time_unaveraged_RDFS = []
    
    for i in range(len(vec_timesteps)):
        
        rdf_one_timestep = RDF_over_all_atoms_one_frame(vec_timesteps[i], dr , cutoff_distance, density_global)
        time_unaveraged_RDFS.append(rdf_one_timestep)
        
    time_averaged_RDF = []
    
    for j in range(len(time_unaveraged_RDFS[0])):
        
        one_sum = 0
        
        for k in range(len(time_unaveraged_RDFS)):
            
            one_sum += time_unaveraged_RDFS[k][j]
        
        one_sum = one_sum / len(time_unaveraged_RDFS)
        
        time_averaged_RDF.append(one_sum)
        
    
        
    return time_averaged_RDF


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

    return all_chol_coms


             
    
   