pimport numpy as np
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
import os
from code_libraries.bilayer_operations import Bilayer
from code_libraries.acyl_chain_preprocessing import *
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


class clf_trj_first:
    
    main_dir = "../"
    dataframe_dir = os.path.join(main_dir, "heavy_data_NO_GIT/dataframes")
    model_scaler_dir = os.path.join(main_dir, "heavy_data_NO_GIT/models_scalers")
    
    sn1_indices = [41, 42, 43]
    sn1_indices.extend(i for i in range(91, 134))
    sn1_indices_carbon_only = np.array([
        42, 92, 95, 98, 101, 104, 107, 110, 112, 114,
        117, 120, 123, 126, 129, 132
    ])

    sn2_indices = [32, 33, 34]
    sn2_indices.extend(i for i in range(44, 91))
    sn2_indices_carbon_only = np.array([
        33, 45, 48, 51, 54, 57, 60, 63, 65, 67,
        70, 73, 76, 79, 82, 85
    ])
    
    sn1_sn2 = sn1_indices + sn2_indices
    
    no_chol = "No cholesterol; pure DOPC system given"
    
    processed_arrays = 0
    processed_dfs = 0
    processed_scaled_dfs = 0
    
    lo_prob_cutoff = 0.5
    labeled_predictions_prob_false = 0
    labeled_predictions_prob_true = 0

    def __init__(self, sys_name, sys_type, n_red, start_frame_real, stop_frame_real, model_type):

        self.sys_name = sys_name
        self.sys_type = sys_type
        self.n_red = int(n_red)
        self.start_frame_real = int(start_frame_real)
        self.stop_frame_real = int(stop_frame_real)
        self.model_type = model_type

        if self.model_type == "svc":
            self.prefix = "svc"
        else:
            self.prefix = "nn"

        if self.n_red == 990:
            self.scaler_name = f"{self.prefix}_11pt_scaler_trj_first.sav"
            self.model_name = f"{self.prefix}_11pt_model_trj_first.sav"

        elif self.n_red == 970:
            self.scaler_name = f"{self.prefix}_31pt_scaler_trj_first.sav"
            self.model_name = f"{self.prefix}_31pt_model_trj_first.sav"

        elif self.n_red == 950:
            self.scaler_name = f"{self.prefix}_51pt_scaler_trj_first.sav"
            self.model_name = f"{self.prefix}_51pt_model_trj_first.sav"

        else:
            self.scaler_name = f"{self.prefix}_101pt_scaler_trj_first.sav"
            self.model_name = f"{self.prefix}_101pt_model_trj_first.sav"

    def bilayer_all_data(self):

        bilayer_init = Bilayer(
            self.sys_name,
            self.sys_type,
            self.start_frame_real,
            self.stop_frame_real
        )

        bilayer_init.create_universe()
        bilayer_init.cleave_leaflets()

        upper_leaflet_all = bilayer_init.upper_leaflet
        lower_leaflet_all = bilayer_init.lower_leaflet

        return lower_leaflet_all, upper_leaflet_all

    def bilayer_all_dopc_data(self):

        lower_leaflet, upper_leaflet = self.bilayer_all_data()

        if type(lower_leaflet) != tuple and type(upper_leaflet) != tuple:
            return lower_leaflet, upper_leaflet
        else:
            return lower_leaflet[0], upper_leaflet[0]

    def bilayer_all_chol_data(self):

        no_chol = "No cholesterol; pure DOPC system given"

        lower_leaflet, upper_leaflet = self.bilayer_all_data()

        if type(lower_leaflet) != tuple and type(upper_leaflet) != tuple:
            return no_chol
        else:
            return lower_leaflet[1], upper_leaflet[1]

    def cholesterol_convolve(self):
        chol_lower, chol_upper = self.bilayer_all_chol_data()
        chol_lower_convolved = convolve_trajectory_valid(chol_lower, self.stop_frame_real, self.n_red)
        chol_lower_convolved = np.swapaxes(chol_lower_convolved, 0, 1)

        chol_upper_convolved = convolve_trajectory_valid(chol_upper, self.stop_frame_real, self.n_red)
        chol_upper_convolved = np.swapaxes(chol_upper_convolved, 0, 1)

        return chol_lower_convolved, chol_upper_convolved

    def process_all_input(self):

        lower_dopc, upper_dopc = self.bilayer_all_dopc_data()

        dopc_lower_all_convolved = convolve_trajectory_valid(lower_dopc, self.stop_frame_real, self.n_red)
        dopc_lower_all_convolved = np.swapaxes(dopc_lower_all_convolved, 0, 1)

        dopc_upper_all_convolved = convolve_trajectory_valid(upper_dopc, self.stop_frame_real, self.n_red)
        dopc_upper_all_convolved = np.swapaxes(dopc_upper_all_convolved, 0, 1)

        dopc_lower_sn1_op_height = op_height_df(dopc_lower_all_convolved, "sn1", self.n_red, is_pure=True, heights=True)
        dopc_lower_sn2_op_height = op_height_df(dopc_lower_all_convolved, "sn2", self.n_red, is_pure=True, heights=True)

        dopc_upper_sn1_op_height = op_height_df(dopc_upper_all_convolved, "sn1", self.n_red, is_pure=True, heights=True)
        dopc_upper_sn2_op_height = op_height_df(dopc_upper_all_convolved, "sn2", self.n_red, is_pure=True, heights=True)

        lower_sn1_sn2_combined = combine_sn1_sn2(dopc_lower_sn1_op_height, dopc_lower_sn2_op_height)
        upper_sn1_sn2_combined = combine_sn1_sn2(dopc_upper_sn1_op_height, dopc_upper_sn2_op_height)

        self.processed_arrays = dopc_lower_all_convolved, dopc_upper_all_convolved
        self.processed_dfs = (
            lower_sn1_sn2_combined.iloc[:, :-1],
            upper_sn1_sn2_combined.iloc[:, :-1]
        )

        scaler_path = os.path.join(self.model_scaler_dir, self.scaler_name)
        scaler = pickle.load(open(scaler_path, 'rb'))

        unscaled_lower, unscaled_upper = self.processed_dfs
        scaled_lower = scaler.transform(unscaled_lower)
        scaled_upper = scaler.transform(unscaled_upper)

        self.processed_scaled_dfs = scaled_lower, scaled_upper

    def classify(self, probability=False):

        model_path = os.path.join(self.model_scaler_dir, self.model_name)
        model = pickle.load(open(model_path, 'rb'))

        scaled_lower, scaled_upper = self.processed_scaled_dfs

        if not probability:

            predictions_lower = model.predict(scaled_lower)
            predictions_lower = predictions_lower.reshape(self.n_red, -1)

            predictions_upper = model.predict(scaled_upper)
            predictions_upper = predictions_upper.reshape(self.n_red, -1)

            self.labeled_predictions_prob_false = predictions_lower, predictions_upper

        else:

            predictions_lower = model.predict_proba(scaled_lower)
            predictions_lower = predictions_lower.reshape(self.n_red, -1, 2)

            predictions_upper = model.predict_proba(scaled_upper)
            predictions_upper = predictions_upper.reshape(self.n_red, -1, 2)

            self.labeled_predictions_prob_true = predictions_lower, predictions_upper

    
    
    def reset_lo_prob_cutoff(self , new_lo_cutoff):
        
        self.lo_prob_cutoff = new_lo_cutoff
    
    
    def predict_frame(self , reduced_frame):
        
        lower_leaflet_array , upper_leaflet_array = self.processed_arrays

        lower_leaflet_array_red_frame = lower_leaflet_array[reduced_frame]
        upper_leaflet_array_red_frame = upper_leaflet_array[reduced_frame]
        
        if self.labeled_predictions_prob_false != 0:
        
            lower_predictions , upper_predictions = self.labeled_predictions_prob_false

            lower_prediction_red_frame = lower_predictions[reduced_frame]
            upper_prediction_red_frame = upper_predictions[reduced_frame]

            label_lower_0 = lower_leaflet_array_red_frame[lower_prediction_red_frame==0]
            label_lower_1 = lower_leaflet_array_red_frame[lower_prediction_red_frame==1]

            label_upper_0 = upper_leaflet_array_red_frame[upper_prediction_red_frame==0]
            label_upper_1 = upper_leaflet_array_red_frame[upper_prediction_red_frame==1]
            
        else:
            
            lower_predictions , upper_predictions = self.labeled_predictions_prob_true
        
            lower_leaflet_array_red_frame = lower_leaflet_array[reduced_frame]
            upper_leaflet_array_red_frame = upper_leaflet_array[reduced_frame]

            lower_prediction_red_frame = lower_predictions[reduced_frame]
            upper_prediction_red_frame = upper_predictions[reduced_frame]

            label_lower_1 = lower_leaflet_array_red_frame[lower_prediction_red_frame[: , 1] >= self.lo_prob_cutoff]
            label_lower_0 = lower_leaflet_array_red_frame[lower_prediction_red_frame[: , 1] < self.lo_prob_cutoff]

            label_upper_1 = upper_leaflet_array_red_frame[upper_prediction_red_frame[: , 1] >= self.lo_prob_cutoff]
            label_upper_0 = upper_leaflet_array_red_frame[upper_prediction_red_frame[: , 1] < self.lo_prob_cutoff]
            
            
            
        return label_lower_0 , label_lower_1 , label_upper_0 , label_upper_1
        
        
        
    def lo_ld_proportions_in_reduced_time(self):
        
        lower_ratios_all = []
        upper_ratios_all = []
        
        if self.labeled_predictions_prob_false != 0:
            
            lower_predictions , upper_predictions = self.labeled_predictions_prob_false
            
        else:
            
            lower_predictions , upper_predictions = self.labeled_predictions_prob_true
            
            
        for t in range(self.n_red):
            
            pred_one_t = self.predict_frame(t)

            lower_pred_one_t_ld = pred_one_t[0].shape[0]
            lower_pred_one_t_lo = pred_one_t[1].shape[0]
            lo_ld_prop_lower_one_t = lower_pred_one_t_lo/(lower_pred_one_t_ld +lower_pred_one_t_lo)

            upper_pred_one_t_ld = pred_one_t[2].shape[0]
            upper_pred_one_t_lo = pred_one_t[3].shape[0]
            lo_ld_prop_upper_one_t = upper_pred_one_t_lo/(upper_pred_one_t_ld + upper_pred_one_t_lo)

            lower_ratios_all.append(lo_ld_prop_lower_one_t)
            upper_ratios_all.append(lo_ld_prop_upper_one_t)
        
            
        return lower_ratios_all , upper_ratios_all
    
    def predict_average_probability(self):
            
            lower_leaflet_array , upper_leaflet_array = self.processed_arrays
            predictions_lower, predictions_upper = self.labeled_predictions_prob_true
            
            lower_leaflet_array = np.swapaxes(lower_leaflet_array , 0 , 1)
            upper_leaflet_array = np.swapaxes(upper_leaflet_array , 0 , 1)
                
            predictions_lower = predictions_lower[: , : , 1] 
            predictions_upper = predictions_upper[: , : , 1] 
            
            predictions_lower = np.swapaxes(predictions_lower , 0 , 1)
            predictions_upper = np.swapaxes(predictions_upper , 0 , 1)
            

            av_coords_all_lipids_lower = np.empty([lower_leaflet_array.shape[0] , lower_leaflet_array.shape[2] , lower_leaflet_array.shape[3]])
            av_probs_all_lipids_lower = np.empty(predictions_lower.shape[0])

            for lipid in range(predictions_lower.shape[0]):

                current_lipid_positions = lower_leaflet_array[lipid]
                current_lipid_probs = predictions_lower[lipid]

                average_coords_current_lipid = np.mean(current_lipid_positions , axis = 0)
                av_probs_current_lipid = np.mean(current_lipid_probs , axis = 0)

                av_coords_all_lipids_lower[lipid] = average_coords_current_lipid
                av_probs_all_lipids_lower[lipid] = av_probs_current_lipid
            

            av_coords_all_lipids_upper = np.empty([upper_leaflet_array.shape[0] , upper_leaflet_array.shape[2] , upper_leaflet_array.shape[3]])
            av_probs_all_lipids_upper = np.empty(predictions_upper.shape[0])

            for lipid in range(predictions_upper.shape[0]):

                current_lipid_positions = upper_leaflet_array[lipid]
                current_lipid_probs = predictions_upper[lipid]

                average_coords_current_lipid = np.mean(current_lipid_positions , axis = 0)
                av_probs_current_lipid = np.mean(current_lipid_probs , axis = 0)

                av_coords_all_lipids_upper[lipid] = average_coords_current_lipid
                av_probs_all_lipids_upper[lipid] = av_probs_current_lipid
                
            return av_coords_all_lipids_lower , av_probs_all_lipids_lower , av_coords_all_lipids_upper , av_probs_all_lipids_upper
            
    
    
    def plot_one_frame_labels(self ,reduced_frame, title,  plot_type = "2d",figure_size = [12 , 8]):
        
        lower_0 , lower_1 , upper_0 , upper_1 = self.predict_frame(reduced_frame)
        
        if plot_type == "2d":
        
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figure_size)

            fig.suptitle(str(title)+ ", reduced frame:" + str(reduced_frame) , fontsize=16)

            axes[0].scatter(lower_0[: , self.sn1_sn2 , 0] , lower_0[: , self.sn1_sn2 , 1],
                           color = "blue")

            axes[0].scatter(lower_1[: , self.sn1_sn2 , 0] , lower_1[: , self.sn1_sn2 , 1],
                           color = "green")

            #axes[0].scatter(chol_lower_red_frame[: , : , 0] , chol_lower_red_frame[: , : , 1],
                          # color = "red")

            axes[0].set_xlabel("x/nm")
            axes[0].set_ylabel("y/nm")
            axes[0].set_title("Lower leaflet(CHOL in red) ")

            axes[1].scatter(upper_0[: , self.sn1_sn2 , 0] , upper_0[: , self.sn1_sn2 , 1],
                           color = "blue")

            axes[1].scatter(upper_1[: , self.sn1_sn2 , 0] , upper_1[: , self.sn1_sn2 , 1],
                           color = "green")

            #axes[1].scatter(chol_upper_red_frame[: , : , 0] , chol_upper_red_frame[: , : , 1],
                           #color = "red")

            axes[1].set_xlabel("x/nm")
            axes[1].set_ylabel("y/nm")
            axes[1].set_title("Lower leaflet(CHOL in red) ")

            try:
                chol_lower , chol_upper = self.bilayer_all_chol_data()
                chol_lower_convolved = convolve_trajectory_valid(chol_lower , self.stop_frame_real, self.n_red)
                chol_lower_convolved = np.swapaxes(chol_lower_convolved , 0 , 1)
                chol_upper_convolved = convolve_trajectory_valid(chol_upper , self.stop_frame_real, self.n_red)
                chol_upper_convolved = np.swapaxes(chol_upper_convolved , 0 , 1)
                chol_lower_red_frame = chol_lower_convolved[reduced_frame]
                chol_upper_red_frame = chol_upper_convolved[reduced_frame]

                axes[0].scatter(chol_lower_red_frame[: , : , 0] , chol_lower_red_frame[: , : , 1],
                               color = "red")

                axes[0].set_title("Lower leaflet(CHOL in red) ")



                axes[1].scatter(chol_upper_red_frame[: , : , 0] , chol_upper_red_frame[: , : , 1],
                               color = "red")

                axes[1].set_title("Lower leaflet(CHOL in red) ")



                plt.show()

            except ValueError:

                axes[0].set_title("Lower leaflet")
                axes[1].set_title("Lower leaflet")

                plt.show()
                
        else:
            
            fig = plt.figure(figsize=figure_size)
            plot = fig.add_subplot(111, projection='3d')
            lower_0_x = lower_0[: , self.sn1_sn2 , 0]
            lower_0_y = lower_0[: , self.sn1_sn2 , 1]
            lower_0_z = lower_0[: , self.sn1_sn2 , 2]
            lower_1_x = lower_1[: , self.sn1_sn2 , 0]
            lower_1_y = lower_1[: , self.sn1_sn2 , 1]
            lower_1_z = lower_1[: , self.sn1_sn2 , 2]

            #cog_x_lower = np.mean(coord_av_lower_x , axis = 0)
            #cog_y_lower = np.mean(coord_av_lower_y , axis = 0)
            #cog_z_lower = np.mean(coord_av_lower_z , axis = 0)
            lower = plot.scatter(lower_0_x, lower_0_y, lower_0_z, marker='o', s=2, alpha=0.9,
                      color = "blue")

            lower = plot.scatter(lower_1_x, lower_1_y, lower_1_z, marker='o', s=2, alpha=0.9,
                      color = "green")

            upper_0_x = upper_0[: , self.sn1_sn2 , 0]
            upper_0_y = upper_0[: , self.sn1_sn2 , 1]
            upper_0_z = upper_0[: , self.sn1_sn2 , 2]
            upper_1_x = upper_1[: , self.sn1_sn2 , 0]
            upper_1_y = upper_1[: , self.sn1_sn2 , 1]
            upper_1_z = upper_1[: , self.sn1_sn2 , 2]

            #cog_x_upper = np.mean(coord_av_upper_x , axis = 0)
            #cog_y_upper = np.mean(coord_av_upper_y , axis = 0)
            #cog_z_upper = np.mean(coord_av_upper_z , axis = 0)

            upper = plot.scatter(upper_0_x, upper_0_y, upper_0_z, marker='o', s=2, alpha=0.9,
                      color = "blue")

            upper = plot.scatter(upper_1_x, upper_1_y, upper_1_z, marker='o', s=2, alpha=0.9,
                      color = "green")

            plot.set_xlabel('X')
            plot.set_ylabel('Y')
            plot.set_zlabel('Z')
            # plot.set_xlim(0,)
            # plot.set_ylim(0,)
            #plot.set_zlim(0,)
            plot.set_title(str(title))
    #         cbar = fig.colorbar(lower)
    #         cbar.set_label("Lo probability")

            try:
                chol_lower , chol_upper = self.bilayer_all_chol_data()
                chol_lower_convolved = convolve_trajectory_valid(chol_lower , self.stop_frame_real, self.n_red)
                chol_lower_convolved = np.swapaxes(chol_lower_convolved , 0 , 1)
                chol_upper_convolved = convolve_trajectory_valid(chol_upper , self.stop_frame_real, self.n_red)
                chol_upper_convolved = np.swapaxes(chol_upper_convolved , 0 , 1)
                chol_lower_red_frame = chol_lower_convolved[reduced_frame]
                chol_upper_red_frame = chol_upper_convolved[reduced_frame]

                chol_lower_x = chol_lower_red_frame[: , : , 0]
                chol_lower_y = chol_lower_red_frame[: , : , 1]
                chol_lower_z = chol_lower_red_frame[: , : , 2]
                chol_upper_x = chol_upper_red_frame[: , : , 0]
                chol_upper_y = chol_upper_red_frame[: , : , 1]
                chol_upper_z = chol_upper_red_frame[: , : , 2]


                lower.scatter(chol_lower_x , chol_lower_y, chol_lower_z, s = 4, color = "red")

                upper.scatter(chol_upper_x , chol_upper_y , chol_upper_z , s= 4 , color = "red")


                plt.show()

            except ValueError:

                plt.show()
        
            
            
    def plot_predicted_average_all_frames(self , title , plot_type="2d",figure_size = [12,8]):


        coord_av_lower , prob_av_lower , coord_av_upper, prob_av_upper = self.predict_average_probability()

        coord_av_lower_x = coord_av_lower[: , : , 0]
        coord_av_lower_y = coord_av_lower[: , : , 1]
        coord_av_lower_z = coord_av_lower[: , : , 2]

        coord_av_upper_x = coord_av_upper[: , : , 0]
        coord_av_upper_y = coord_av_upper[: , : , 1]
        coord_av_upper_z = coord_av_upper[: , : , 2]

        colors_lower = [i for i in prob_av_lower for j in range(0,coord_av_lower.shape[1])]
        colors_upper = [i for i in prob_av_upper for j in range(0,coord_av_upper.shape[1])]

        if plot_type != "2d":

            fig = plt.figure(figsize=[12,8])
            plot = fig.add_subplot(111, projection='3d')
            cog_x_lower = np.mean(coord_av_lower_x , axis = 0)
            cog_y_lower = np.mean(coord_av_lower_y , axis = 0)
            cog_z_lower = np.mean(coord_av_lower_z , axis = 0)
            lower = plot.scatter(coord_av_lower_x, coord_av_lower_y, coord_av_lower_z, marker='o', s=2, alpha=0.9,
                      c = colors_lower , vmin = 0 , vmax = 1)

            cog_x_upper = np.mean(coord_av_upper_x , axis = 0)
            cog_y_upper = np.mean(coord_av_upper_y , axis = 0)
            cog_z_upper = np.mean(coord_av_upper_z , axis = 0)
            upper = plot.scatter(coord_av_upper_x, coord_av_upper_y, coord_av_upper_z, marker='o', s=2, alpha=0.9,
                      c = colors_upper , vmin = 0 , vmax = 1)

            try:
                chol_lower , chol_upper = self.bilayer_all_chol_data()
                chol_lower_convolved = convolve_trajectory_valid(chol_lower , self.stop_frame_real, self.n_red)
                chol_upper_convolved = convolve_trajectory_valid(chol_upper , self.stop_frame_real, self.n_red)

                chol_lower_av = np.mean(chol_lower_convolved , axis = 1)
                chol_upper_av = np.mean(chol_upper_convolved , axis = 1)

                chol_lower_x = chol_lower_av[: , : , 0]
                chol_lower_y = chol_lower_av[: , : , 1]
                chol_lower_z = chol_lower_av[: , : , 2]
                
                chol_upper_x = chol_upper_av[: , : , 0]
                chol_upper_y = chol_upper_av[: , : , 1]
                chol_upper_z = chol_upper_av[: , : , 2]

                lower = plot.scatter(chol_lower_x , chol_lower_y , chol_lower_z , marker = "o" , s= 4 , alpha = 0.9)
                upper = plot.scatter(chol_upper_x , chol_upper_y , chol_upper_z , marker = "o" , s= 4 , alpha = 0.9)

                plot.set_xlabel('X')
                plot.set_ylabel('Y')
                plot.set_zlabel('Z')
                # plot.set_xlim(0,)
                # plot.set_ylim(0,)
                #plot.set_zlim(0,)
                plot.set_title(str(title))
                cbar = fig.colorbar(lower , )
                cbar.set_label("Lo probability")

            except ValueError:

                plot.set_xlabel('X')
                plot.set_ylabel('Y')
                plot.set_zlabel('Z')
                # plot.set_xlim(0,)
                # plot.set_ylim(0,)
                #plot.set_zlim(0,)
                plot.set_title(str(title))
                cbar = fig.colorbar(lower , )
                cbar.set_label("Lo probability")

                plt.show()

        else:

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figure_size)

            fig.suptitle(str(title) , fontsize=16)

            lower = axes[0].scatter(coord_av_lower_x, coord_av_lower_y , c = colors_lower , vmin = 0 , vmax = 1 , cmap = "viridis")

            #axes[0].scatter(chol_lower_red_frame[: , : , 0] , chol_lower_red_frame[: , : , 1],
                          # color = "red")

            axes[0].set_xlabel("x/nm")
            axes[0].set_ylabel("y/nm")
            axes[0].set_title("Lower leaflet(CHOL in red) ")
            plt.colorbar(lower , label= "Lo probability")

            upper = axes[1].scatter(coord_av_upper_x, coord_av_upper_y , c = colors_upper , vmin = 0 , vmax = 1, cmap = "viridis")

            #axes[1].scatter(chol_upper_red_frame[: , : , 0] , chol_upper_red_frame[: , : , 1],
                           #color = "red")

            axes[1].set_xlabel("x/nm")
            axes[1].set_ylabel("y/nm")
            axes[1].set_title("Upper leaflet(CHOL in red) ")
            plt.colorbar(upper , cmap = "viridis" , label= "Lo probability")

            try:
                chol_lower , chol_upper = self.bilayer_all_chol_data()
                chol_lower_convolved = convolve_trajectory_valid(chol_lower , self.stop_frame_real, self.n_red)
                chol_upper_convolved = convolve_trajectory_valid(chol_upper , self.stop_frame_real, self.n_red)

                chol_lower_av = np.mean(chol_lower_convolved , axis = 1)
                chol_upper_av = np.mean(chol_upper_convolved , axis = 1)

                chol_lower_x = chol_lower_av[: , : , 0]
                chol_lower_y = chol_lower_av[: , : , 1]

                chol_upper_x = chol_upper_av[: , : , 0]
                chol_upper_y = chol_upper_av[: , : , 1]

                axes[0].scatter(chol_lower_x , chol_lower_y,
                               color = "red")

                axes[0].set_title("Lower leaflet(CHOL in red) ")



                axes[1].scatter(chol_upper_x , chol_upper_y,
                               color = "red")


                axes[1].set_title("Upper leaflet(CHOL in red) ")
            


            except ValueError:
                

                axes[0].set_title("Lower leaflet")
                axes[1].set_title("Upper leaflet")
                
                
                plt.show()
        
        
        
        
        
        
        
        