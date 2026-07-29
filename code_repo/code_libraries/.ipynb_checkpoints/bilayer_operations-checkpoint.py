import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import os

class Bilayer:

    #os.chdir("../")
    gro_file_suffix = ".gro"
    xtc_file_suffix = ".xtc"

    main_directory = "../"
    heavy_data_directory = "heavy_data_NO_GIT/"
    gro_xtc_directory = os.path.join(main_directory , heavy_data_directory , "gro_xtc/")
    position_directory = os.path.join(main_directory , heavy_data_directory , "position_npys/")
    
    def __init__(self , sys_name , sys_type , start_frame , stop_frame):
        
        self.sys_name = sys_name
        self.sys_type = sys_type
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.universe = 0
        self.lower_leaflet = 0
        self.upper_leaflet = 0
        
    def create_universe(self):
        
        sys_type = str(self.sys_type)
        sys_name = str(self.sys_name)
        
        containing_directory = os.path.join(self.main_directory, self.heavy_data_directory , self.gro_xtc_directory)
        #xtc_file = self.main_directory/sys_type/sys_name+self.xtc_file_suffix
        
        
        filepath_xtc = os.path.join(containing_directory , sys_type, sys_name+self.xtc_file_suffix)
        filepath_gro = os.path.join(containing_directory , sys_type , sys_name+self.gro_file_suffix)
        
        self.universe = mda.Universe(filepath_gro , filepath_xtc)
    
        return mda.Universe(filepath_gro , filepath_xtc)
        
    def select_residue(self , res_string):
        
        universe = self.create_universe()
        
        residue_name = str("resname") + str(" ") + str(res_string)
        
        return universe.select_atoms(residue_name) , universe
    

            
    def get_positions(self , res_string):
        
        residue , universe = self.select_residue(res_string)
        
        start_frame , stop_frame = self.start_frame , self.stop_frame
        
        desired_frames = (stop_frame - start_frame)

        no_of_atoms = len(residue)

        residue_positions = np.empty([desired_frames , no_of_atoms , 3 ])


        trajectory = universe.trajectory

        for i , ts in enumerate(trajectory[start_frame:stop_frame]):
            
            residue_step = residue.positions
            residue_positions[i] = residue_step/10
            
        return residue_positions
    
    def save_residue_positions(self , res_string):
        
        sys_type = str(self.sys_type)
        sys_name = str(self.sys_name)
        
        all_positions = self.get_positions(res_string)
        containing_directory = os.path.join(self.position_directory , self.sys_type)
        
        filename = self.sys_type + "_" + self.sys_name + "_" + res_string + "_positions.npy"
        filepath = os.path.join(containing_directory , filename)
        
        np.save(filepath , all_positions)
        
    def cleave_leaflets(self):
        
        positions_directory = self.position_directory
  
        DOPC_filename = self.sys_type + "_" + self.sys_name + "_" + "DOPC_positions.npy"
        CHOL_filename = self.sys_type + "_" + self.sys_name + "_" + "CHL1_positions.npy"
        DSPC_filename = self.sys_type + "_" + self.sys_name + "_" + "DSPC_positions.npy"
        ###for non DOPC/CHOL lipids, change names accordingly
        
        DOPC_directory = os.path.join(positions_directory , self.sys_type,  DOPC_filename)
        CHOL_directory = os.path.join(positions_directory ,self.sys_type, CHOL_filename)
        DSPC_directory = os.path.join(positions_directory ,self.sys_type, DSPC_filename) #change accordingly
        
        DOPC_positions = np.load(DOPC_directory)
        #CHOL_positions = np.load(CHOL_directory)
        
        DOPC_positions = DOPC_positions[self.start_frame:self.stop_frame]
        #CHOL_positions = CHOL_positions[self.start_frame:self.stop_frame]
        
        DOPC_reshape_by_mol = DOPC_positions.reshape(len(DOPC_positions) , -1 , 138 , 3)
        #CHOL_reshape_by_mol = CHOL_positions.reshape(len(CHOL_positions) , -1 , 74 , 3)
        DOPC_molecule_count = DOPC_reshape_by_mol.shape[1]
        #CHOL_molecule_count = CHOL_reshape_by_mol.shape[1]
        
        if self.sys_type=="sym":
            
            upper_dopc = DOPC_reshape_by_mol[: , :DOPC_molecule_count//2 , : , :]
            lower_dopc = DOPC_reshape_by_mol[: , DOPC_molecule_count//2: , :]
            
            #upper_chol = CHOL_reshape_by_mol[: , :CHOL_molecule_count//2 , : , :]
            #lower_chol = CHOL_reshape_by_mol[: , CHOL_molecule_count//2: , :]
        
            try: #change accordingly
                chol_positions = np.load(CHOL_directory)
                chol_positions = chol_positions[self.start_frame:self.stop_frame]
                CHOL_reshape_by_mol = chol_positions.reshape(len(chol_positions) , -1 , 74 , 3)
                CHOL_molecule_count = CHOL_reshape_by_mol.shape[1]
                
                upper_chol = CHOL_reshape_by_mol[: , :CHOL_molecule_count//2 , : , :]
                lower_chol = CHOL_reshape_by_mol[: , CHOL_molecule_count//2: , :]
                
                #DSPC_positions = np.load(DSPC_directory)
                ##DSPC_positions = DSPC_positions[self.start_frame:self.stop_frame]
                #DSPC_reshape_by_mol = DSPC_positions.reshape(len(DSPC_positions) , -1 , 142 , 3)
                #DSPC_molecule_count = DSPC_reshape_by_mol.shape[1]
                #upper_dspc = DSPC_reshape_by_mol[: , :DSPC_molecule_count//2 , : , :]
                #lower_dspc = DSPC_reshape_by_mol[: , DSPC_molecule_count//2: , :]
                
                self.upper_leaflet = upper_dopc , upper_chol
                self.lower_leaflet = lower_dopc ,lower_chol

                #return  (upper_dopc , upper_chol) , (lower_dopc ,lower_chol)
            
            except FileNotFoundError:
                
                self.upper_leaflet = upper_dopc
                self.lower_leaflet = lower_dopc

                #return  upper_dopc , lower_dopc
            
        else:  #change accordingly
            
            DSPC_positions = np.load(DSPC_directory)
            DSPC_positions = DSPC_positions[self.start_frame:self.stop_frame]
            DSPC_reshape_by_mol = DSPC_positions.reshape(len(DSPC_positions) , -1 , 142 , 3)
            DSPC_molecule_count = DSPC_reshape_by_mol.shape[1]
            
            upper_dspc = DSPC_reshape_by_mol
            
            upper_chol = CHOL_reshape_by_mol[: , :28 , : , :]
            lower_chol = CHOL_reshape_by_mol[: , 28: , : , :]
            
            lower_dopc = DOPC_reshape_by_mol[: , :-5 , : , : ]
            upper_dopc = DOPC_reshape_by_mol[: , -5: , : , :]
            
            return (upper_dopc , upper_chol , upper_dspc) , (lower_dopc , lower_chol)
        
        
    def box_dimensions(self):

        start_frame = self.start_frame
        stop_frame = self.stop_frame

        universe = self.create_universe()
        desired_frames = (stop_frame - start_frame)

        box_dim_all = np.empty([desired_frames , 3])

        trajectory = universe.trajectory

        for i, ts in enumerate(trajectory[start_frame: stop_frame]):

            one_step_dimensions = universe.dimensions[:3]/10 ##convert to nm from angstroms
            box_dim_all[i] = one_step_dimensions

        return box_dim_all
    
    ####THE NEXT TWO FUNCTIONS ARE USEFUL ONLY FOR SYMMETRIC BILAYERS OR JUST
    ####TO GET A ROUGH ORDER OF MAGNITUDE ESTIMATES FOR ASYMMETRIC ONES;
    ####THEY ARE NOT RELIABLE FOR ASYMMETRIC SYSTEMS
    
    def total_bilayer_height(self , average=False):
        
        universe = self.universe
        water_mol_total = len(universe.select_atoms("resname TIP3"))//3
        water_total_volume = 0.029*water_mol_total #in cubic nanometers
        
        start_frame = self.start_frame
        stop_frame = self.stop_frame
        box_dim_all = self.box_dimensions()
        
        vol_all = np.prod(box_dim_all , axis = 1)
        area_all = np.prod(box_dim_all[: , :2] , axis = 1)
        
        bilayer_volume_all_steps = vol_all - water_total_volume
        bilayer_height = bilayer_volume_all_steps/area_all
        
        if average == True:
            
            bilayer_height = np.mean(bilayer_height)
        
        return bilayer_height
    
    def area_per_lipid(self , average = False):
        
        universe = self.universe
        
        start_frame = self.start_frame
        stop_frame = self.stop_frame
        box_dim_all = self.box_dimensions()
        box_dim_av = np.mean(box_dim_all , axis = 0)
        
        box_dim_xy = box_dim_all[: , :2]
        box_area = np.prod(box_dim_xy , axis = 1)
        
        if type(self.lower_leaflet) == tuple and type(self.upper_leaflet) == tuple:
            
            no_lipids_lower = self.lower_leaflet[0].shape[1] + self.lower_leaflet[1].shape[1]
            #no_lipids_upper - self.upper_leaflet[0].shape + self.upper_leaflet[1].shape
            
        else:
            
            no_lipids_lower = self.lower_leaflet.shape[1]
            #no_lipids_upper = self.upper_leaflet.shape
            
        apl_nm = box_area/no_lipids_lower
        
        return apl_nm
        
#         try:
            
#             no_chol_mol_
#             no_dopc_chol_mol = no_dopc_mol + no_chol_mol
            
#             apl = box_area/no_dopc_chol_mol
            
#         except ValueError:
            
#             apl = box_area/no_dopc_mol
            
#         return apl
        

    
    
        
        
       
