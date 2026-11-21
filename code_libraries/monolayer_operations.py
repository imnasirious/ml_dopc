import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import os
import scipy
from code_libraries.bilayer_operations import Bilayer

class Monolayer: ###takes in an input array of positions and isolates one whole leaflet of 
                ###phospolipids
    
    gro_file_suffix = ".gro"
    xtc_file_suffix = ".xtc"

    main_directory = os.getcwd()
    gmx_trajectory_directory = os.path.join(main_directory , "gro_xtc/")
    position_directory = os.path.join(main_directory , "position_npys/")
    
    def __init__(self, sys_type, sys_name, upper_or_lower, start_frame, stop_frame):
        
        self.sys_type = sys_type
        self.sys_name = sys_name
        self.upper_or_lower = upper_or_lower
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        
        # Instantiate Bilayer
        bilayer_instance = Bilayer(sys_name, sys_type, start_frame, stop_frame)
        cleaved_leaflets = bilayer_instance.cleave_leaflets()
        
        # Check whether we need the upper or lower leaflet
        if upper_or_lower == "upper":
            self.leaflet = cleaved_leaflets[0]  # Upper leaflet
        elif upper_or_lower == "lower":
            self.leaflet = cleaved_leaflets[1]  # Lower leaflet
        else:
            raise ValueError("upper_or_lower must be either 'upper' or 'lower'.")
           
    def acyl_chains(self , which_chain="both" , carbon_only=False):

        dopc_sn1_indices = [42 , 92 , 95 , 98 , 101 , 104 , 107 , 110 , 112 , 114 , 
                           117 , 120 , 123 , 126 , 129 , 132 , 135]
        
        dopc_sn2_indices = [33 , 45 , 48 , 51 , 54 , 57 , 60 , 63 , 65 , 67 , 70,
                           73 , 76 , 79 , 82 , 85 , 88]

        dopc_positions = self.leaflet[0]
        num_frames = dopc_positions.shape[0]
        num_molecules = dopc_positions.shape[1]

        sn1_c1 = [42 , 43 , 45]
        sn1_c2 = [92 , 93 , 94]
        sn1_c3 = [95 , 96 , 97]
        sn1_c4  = [98 , 99 , 100]
        sn1_c5 = [101 , 102 , 103]
        sn1_c6 = [104 , 105 , 106]
        sn1_c7 = [107 , 108 , 109]
        sn1_c8 = [110 , 111]
        sn1_c9 = [112 , 113]
        sn1_c10 = [114 , 115 , 116]
        sn1_c11 = [117 , 118 , 119]
        sn1_c12 = [120 , 121 , 122]
        sn1_c13 = [123 , 124 , 125]
        sn1_c14 = [126 , 127 , 128]
        sn1_c15 = [129 , 130 , 131]
        sn1_c16 = [132 , 133 , 134]
        sn1_c17  = [135 , 136 , 137 , 138]

        all_sn1_indices = np.concatenate([sn1_c1 , sn1_c2 , sn1_c3 , sn1_c4,
                                    sn1_c5 , sn1_c6 , sn1_c7 , sn1_c8,
                                    sn1_c9 , sn1_c10 , sn1_c11 , sn1_c12,
                                    sn1_c13 , sn1_c14 , sn1_c15 , sn1_c16 , sn1_c17])
        all_sn1_indices = all_sn1_indices - 1

        sn2_c1 = [33 , 34 , 35]
        sn2_c2 = [45 , 46 , 47]
        sn2_c3 = [48 , 49 , 50]
        sn2_c4 = [51 , 52 , 53]
        sn2_c5 = [54 , 55 , 56]
        sn2_c6 = [57 , 58 , 59]
        sn2_c7 = [60 , 61 , 62]
        sn2_c8 = [63 , 64]
        sn2_c9 = [65 , 66]
        sn2_c10 = [67 , 68 , 69]
        sn2_c11 = [70 , 71 , 72]
        sn2_c12 = [73 , 74 , 75]
        sn2_c13 = [76 , 77 , 78]
        sn2_c14 = [79 , 80 , 81]
        sn2_c15 = [82 , 83 , 84]
        sn2_c16 = [85 , 86 , 87]
        sn2_c17 = [88 , 89 , 90 , 91]

        all_sn2_indices = np.concatenate([sn2_c1 , sn2_c2 , sn2_c3 , sn2_c4,
                                    sn2_c5 , sn2_c6 , sn2_c7 , sn2_c8,
                                    sn2_c9 , sn2_c10 , sn2_c11 , sn2_c12,
                                    sn2_c13 , sn2_c14 , sn2_c15 , sn2_c16 , sn2_c17])

        all_sn2_indices = all_sn2_indices - 1


        sn1_chains = np.empty([num_frames ,num_molecules,
                              50 , 3])
        sn2_chains = np.empty([num_frames ,num_molecules,
                              50 , 3])


        for ts in range(num_frames):

            sn1_chains_one_frame = np.empty([num_molecules,50 , 3])
            sn2_chains_one_frame = np.empty([num_molecules,50 , 3])

            dopc_positions_current_ts = dopc_positions[ts]

            for mol in range(num_molecules):

                one_mol_all_sn1 = dopc_positions_current_ts[mol][all_sn1_indices , :]
                sn1_chains_one_frame[mol] = one_mol_all_sn1
                one_mol_all_sn2 = dopc_positions_current_ts[mol][all_sn2_indices , :]
                sn2_chains_one_frame[mol] = one_mol_all_sn2

            sn1_chains[ts] = sn1_chains_one_frame
            sn2_chains[ts] = sn2_chains_one_frame
                
        return sn1_chains , sn2_chains
                    
                    
