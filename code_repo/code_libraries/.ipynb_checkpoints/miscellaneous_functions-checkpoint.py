import numpy as np
import MDAnalysis as mda
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def acyl_chains_com(array_molecules):  
    
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


def com_ft_last(mol_coms, n_point):

    no_mol = mol_coms.shape[1]
    total_ts_per_lipid = mol_coms.shape[0]

    mol_coms = np.swapaxes(mol_coms, 0, 1)

    filt_array = np.ones(n_point) / n_point
    resulting_size = total_ts_per_lipid - (n_point - 1)

    convolved_ft_last = np.empty((no_mol, resulting_size, 3))

    for c in range(no_mol):

        one_com = mol_coms[c]

        # convolve x, y, z separately
        convolved_x = np.convolve(one_com[:, 0], filt_array, mode="valid")
        convolved_y = np.convolve(one_com[:, 1], filt_array, mode="valid")
        convolved_z = np.convolve(one_com[:, 2], filt_array, mode="valid")

        convolved_ft_last[c] = np.column_stack([convolved_x, convolved_y, convolved_z])

    return np.swapaxes(convolved_ft_last , 0 , 1)


def com_savgol(mol_coms, n_point):

    # savgol constraints

    if n_point == 3:
        
        polyorder = 2

    else:

        polyorder = 3
    
    if n_point <= polyorder:
        raise ValueError("n_point must be > polyorder (3).")
    
    if n_point % 2 == 0:
        raise ValueError("n_point must be odd for Savitzky-Golay.")

    # number of lipids and timesteps
    total_ts_per_lipid = mol_coms.shape[0]
    no_mol = mol_coms.shape[1]

    # swap to shape (lipid, time, 3)
    mol_coms = np.swapaxes(mol_coms, 0, 1)

    # output array has same size as input (Savitzky-Golay preserves length)
    filtered_coms = np.empty((no_mol, total_ts_per_lipid, 3))

    for c in range(no_mol):

        one_com = mol_coms[c]   # shape (time, 3)

        # Filter each coordinate with SG
        filtered_x = savgol_filter(one_com[:, 0], window_length=n_point,
                                   polyorder=polyorder)
        filtered_y = savgol_filter(one_com[:, 1], window_length=n_point,
                                   polyorder=polyorder)
        filtered_z = savgol_filter(one_com[:, 2], window_length=n_point,
                                   polyorder=polyorder)

        # combine into (time, 3)
        filtered_coms[c] = np.column_stack([filtered_x, filtered_y, filtered_z])

    return np.swapaxes(filtered_coms , 0 , 1)



	
	






	