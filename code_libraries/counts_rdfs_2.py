import numpy as np
import MDAnalysis as mda
import pandas as pd
import os
import matplotlib.pyplot as plt


def acyl_chain_one_com(array_molecules , chain_type):  
    
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
    
    
    sn1_total_mass = np.sum(sn1_masses)
    sn2_total_mass = np.sum(sn2_masses)
    
    array_molecules = np.swapaxes(array_molecules , 0 , 1)
    n_red = array_molecules.shape[1]
    n_mol = array_molecules.shape[0]
    
    all_coms = np.empty([n_mol, n_red , 3])

    if chain_type == "sn1":
        
        all_acyl_chains = array_molecules[: , : , sn1_indices , :]
        
        for lipid in range(n_mol):
        
            for frame in range(n_red):
                
                current_acyl_current_frame = all_acyl_chains[lipid , frame]
                current_acyl_current_frame_mass_prod_sum = 0
                
                for atom_index in range(len(sn1_indices)):
                    
                    mass_prod = current_acyl_current_frame[atom_index] * sn1_masses[atom_index]
                    current_acyl_current_frame_mass_prod_sum+=mass_prod
                    
                all_coms[lipid][frame] = current_acyl_current_frame_mass_prod_sum/sn1_total_mass        

    else:
        
        all_acyl_chains = array_molecules[: , : , sn2_indices , :]
        
        for lipid in range(n_mol):
        
            for frame in range(n_red):
                
                current_acyl_current_frame = all_acyl_chains[lipid , frame]
                current_acyl_current_frame_mass_prod_sum = 0
                
                for atom_index in range(len(sn2_indices)):
                    
                    mass_prod = current_acyl_current_frame[atom_index] * sn2_masses[atom_index]
                    current_acyl_current_frame_mass_prod_sum+=mass_prod
                    
                all_coms[lipid][frame] = current_acyl_current_frame_mass_prod_sum/sn2_total_mass      
        
            
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

    return all_chol_coms



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


def convolve_pbc_list(pbc_list , n_point):

    no_ts = pbc_list.shape[0]
    no_dopc = pbc_list.shape[1]
    no_chol = pbc_list.shape[2]

    pbc_list = np.swapaxes(pbc_list , 0 , 1)

    filt_array = np.ones(n_point) / n_point
    resulting_size = no_ts - (n_point - 1)

    convolved_pbcs = np.empty((no_dopc,  no_chol , resulting_size))

    for dopc_mol in range(no_dopc):

        one_dopc = pbc_list[dopc_mol]
        convolved_chol_for_one_dopc = np.empty([no_chol , resulting_size])

        for chol_mol in range(no_chol):

            one_chol = one_dopc[: , chol_mol]
            convolved_list = np.convolve(one_chol , filt_array , mode = "valid")
            
            convolved_chol_for_one_dopc[chol_mol] = convolved_list

        convolved_pbcs[dopc_mol] = convolved_chol_for_one_dopc

        

    return np.swapaxes(np.swapaxes(convolved_pbcs , 0 , 2) , 1 , 2)


def radial_distance_pbc_2d(target_vector, reference_vector, box_dim_vector):

    # take x,y only
    target_vector = target_vector[:2]
    reference_vector = reference_vector[:2]
    box = box_dim_vector[:2]

    # minimum-image displacement
    rltv_vec = target_vector - reference_vector
    rltv_vec -= box * np.round(rltv_vec / box)

    return np.linalg.norm(rltv_vec), rltv_vec


def compute_distances_PBC_acyl_to_chol_one_ts(acyl_array_one_ts, chol_array_one_ts , box_dim_vector):
    
    no_chol = chol_array_one_ts.shape[0]
    no_acyl = acyl_array_one_ts.shape[0]
    
    all_pbc_distances = np.empty([no_acyl , no_chol])
    
    for acyl_com_index in range(no_acyl):
        
        current_acyl_com = acyl_array_one_ts[acyl_com_index]
        acyl_distances_one_com = np.empty(no_chol)
        
        for chol_com_index in range(no_chol):
            
            current_chol_com = chol_array_one_ts[chol_com_index]
            
            current_acyl_chol_pbc_distance = radial_distance_pbc_2d(current_acyl_com , current_chol_com , box_dim_vector )[0]
            
            acyl_distances_one_com[chol_com_index] = current_acyl_chol_pbc_distance
            
        all_pbc_distances[acyl_com_index] = np.sort(acyl_distances_one_com)
        
        
    return all_pbc_distances


def compute_distances_PBC_acyl_to_chol_all_ts(acyl_array, chol_array , box_dim_vector):

    no_ts = len(acyl_array)
    all_pbc_distances_in_time = []

    for ts in range(no_ts):

        one_ts_pbc_distances = compute_distances_PBC_acyl_to_chol_one_ts(acyl_array[ts],chol_array[ts],
                                                                         box_dim_vector)

        all_pbc_distances_in_time.append(one_ts_pbc_distances)

    return np.array(all_pbc_distances_in_time)


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

        if len(vec_timesteps[i]) == 0:

            continue

        else:
        
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


def predictions_cutoff_pbc(pbc_array, pred_array):

    label_1_all = []
    label_0_all = []

    for f in range(len(pbc_array)):

        pbc_frame = pbc_array[f]
        pred_frame = pred_array[f]

        label_1 = pbc_frame[pred_frame == True]
        label_0 = pbc_frame[pred_frame == False]

        label_1_all.append(label_1)
        label_0_all.append(label_0)

    return label_1_all, label_0_all


# def RDF_integral_and_all(vec_timesteps , dr , cutoff_distance, box_dim_vector , no_blocks):
    
#     time_unaveraged_RDFS = []
#     cutoff_distances = np.arange(0 , radius , cutoff_distance)


    
#     for i in range(len(vec_timesteps)):

#         if len(vec_timesteps[i]) == 0:

#             continue

#         else:
        
#             rdf_one_timestep = RDF_over_all_atoms_one_frame_diff_species(vec_timesteps[i], dr , cutoff_distance, box_dim_vector)
#             time_unaveraged_RDFS.append(rdf_one_timestep)

#     integrals_all_frames = []

#     for rdf_frame in time_unaveraged_RDFS:   # your per-frame RDFs
#         g = np.array(rdf_frame)
#         integral = np.trapezoid(g * cutoff_distances , dx = dr)
#         # N = coordination_from_rdf(g, r, rho_y, dr)
#         # coordination_per_frame.append(N)
#         integrals_all_frames.append(integral)
    
#     blocks = np.array_split(integrals_all_frames, no_blocks)

#     block_means = np.array([b.mean() for b in blocks])
#     mean_value = block_means.mean()
#     stderr = block_means.std(ddof=1) / np.sqrt(no_blocks)

#     return mean_value , stderr

def RDF_integral_and_all(vec_timesteps, dr, cutoff_distance, box_dim_vector, no_blocks):

    time_unaveraged_RDFS = []

    for i in range(len(vec_timesteps)):
        if len(vec_timesteps[i]) == 0:
            continue
        
        rdf_one_timestep = RDF_over_all_atoms_one_frame_diff_species(
            vec_timesteps[i], dr, cutoff_distance, box_dim_vector
        )
        time_unaveraged_RDFS.append(rdf_one_timestep)

    integrals_all_frames = []

    for rdf_frame in time_unaveraged_RDFS:
        g = np.array(rdf_frame)
        r = np.arange(0, cutoff_distance, dr)[:len(g)]

        integral = np.trapezoid(g * r, dx=dr)
        integrals_all_frames.append(integral)

    integrals_all_frames = np.array(integrals_all_frames)

    return integrals_all_frames

    # blocks = np.array_split(integrals_all_frames, no_blocks)
    # block_means = np.array([b.mean() for b in blocks])

    # mean_value = block_means.mean()
    # stderr = block_means.std(ddof=1)/np.sqrt(no_blocks)

    # return mean_value, stderr

        
    









                
                
    



