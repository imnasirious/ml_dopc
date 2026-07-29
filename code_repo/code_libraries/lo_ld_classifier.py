import numpy as np
import MDAnalysis as mda
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection

from code_libraries.acyl_chain_preprocessing import *
from code_libraries.miscellaneous_functions import *
from voron_tess_w_chol_overlay_helper import _compute_periodic_voronoi
#from voron_tess_w_chol_overlay_helper import *
from voron_tess_w_chol_tiles_helper import _compute_periodic_voronoi_chol_tiles
#from voron_tess_w_chol_tiles_helper import *


class clf_lo_ld:
    main_dir = "../../"
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

    sav_gol_poly_order = 3

    processed_acyl_coms = 0
    processed_chol_coms = 0
    processed_dopc_nitros = 0
    processed_dopc_phosphos = 0

    
    processed_dfs = 0
    processed_scaled_dfs = 0

    lo_prob_cutoff = 0.5
    prediction_probs = 0
    prediction_labels = 0

    def __init__(self, sys_name, sys_type, red_type, model_type, n_point, start_frame_real, stop_frame_real):

        ##red_type is either "ft_last" or "sav_gol"
        ##model_type only for probablistic outputs(NN, XGBoost, Logistic regression .predict_proba() etc.)
        ##n_points is 7, 9 , 11 , 31 , etc.

        self.sys_name = sys_name
        self.sys_type = sys_type
        self.red_type = red_type
        self.n_point = n_point
        self.start_frame_real = int(start_frame_real)
        self.stop_frame_real = int(stop_frame_real)
        self.model_type = model_type

        if self.red_type == "sav_gol" and self.n_point == 3:

            self.sav_gol_poly_order = 2

        if self.red_type != "raw":

            self.scaler_name = str(self.model_type) + "_" + str(self.n_point) + "pt_scaler_" + str(
                self.red_type) + ".sav"
            self.model_name = str(self.model_type) + "_" + str(self.n_point) + "pt_model_" + str(self.red_type) + ".sav"

        else:

            self.scaler_name = str(self.model_type) + "_raw_scaler.sav"
            self.model_name = str(self.model_type) + "_raw_model.sav"

        self.n_red = self.stop_frame_real - (self.n_point - 1)

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

    def chol_com(self):

        chol_pos_lower, chol_pos_upper = self.bilayer_all_chol_data()
        chol_com_lower = chol_com_all(chol_pos_lower)
        chol_com_upper = chol_com_all(chol_pos_upper)

        return chol_com_lower, chol_com_upper

    def chol_com_av(self):

        chol_com_lower, chol_com_upper = self.chol_com()

        if self.red_type == "ft_last":

            chol_av_lower = com_ft_last(chol_com_lower, self.n_point)
            chol_av_upper = com_ft_last(chol_com_upper, self.n_point)

        elif self.red_type == "sav_gol":

            chol_av_lower = com_savgol(chol_com_lower, self.n_point)
            chol_av_upper = com_savgol(chol_com_upper, self.n_point)

        else:

            chol_av_lower, chol_av_upper = chol_com_lower, chol_com_upper

        return chol_av_lower, chol_av_upper

    def process_all_input(self):

        try:
            self.processed_chol_coms = self.chol_com_av()

        except ValueError:

            self.processed_chol_coms = None, None

        lower_dopc, upper_dopc = self.bilayer_all_dopc_data()

        dopc_lower_coms = acyl_chains_com(lower_dopc)
        dopc_upper_coms = acyl_chains_com(upper_dopc)

        dopc_lower_sn1_df = op_height_df(lower_dopc, "sn1", self.stop_frame_real, is_pure=True, heights=True)
        dopc_upper_sn1_df = op_height_df(upper_dopc, "sn1", self.stop_frame_real, is_pure=True, heights=True)
        dopc_lower_sn2_df = op_height_df(lower_dopc, "sn2", self.stop_frame_real, is_pure=True, heights=True)
        dopc_upper_sn2_df = op_height_df(upper_dopc, "sn2", self.stop_frame_real, is_pure=True, heights=True)

        if self.red_type == "ft_last":

            dopc_lower_sn1_df = convolve_features_valid(dopc_lower_sn1_df, self.n_red, self.stop_frame_real,
                                                        label_type="pure")
            dopc_upper_sn1_df = convolve_features_valid(dopc_upper_sn1_df, self.n_red, self.stop_frame_real,
                                                        label_type="pure")
            dopc_lower_sn2_df = convolve_features_valid(dopc_lower_sn2_df, self.n_red, self.stop_frame_real,
                                                        label_type="pure")
            dopc_upper_sn2_df = convolve_features_valid(dopc_upper_sn2_df, self.n_red, self.stop_frame_real,
                                                        label_type="pure")

            dopc_lower_coms = com_ft_last(dopc_lower_coms, self.n_point)
            dopc_upper_coms = com_ft_last(dopc_upper_coms, self.n_point)

        elif self.red_type == "sav_gol":

            dopc_lower_sn1_df = sav_gol_features(dopc_lower_sn1_df, self.n_point, self.sav_gol_poly_order,
                                                 self.stop_frame_real, label_type="pure")
            dopc_upper_sn1_df = sav_gol_features(dopc_upper_sn1_df, self.n_point, self.sav_gol_poly_order,
                                                 self.stop_frame_real, label_type="pure")
            dopc_lower_sn2_df = sav_gol_features(dopc_lower_sn2_df, self.n_point, self.sav_gol_poly_order,
                                                 self.stop_frame_real, label_type="pure")
            dopc_upper_sn2_df = sav_gol_features(dopc_upper_sn2_df, self.n_point, self.sav_gol_poly_order,
                                                 self.stop_frame_real, label_type="pure")

            dopc_lower_coms = com_savgol(dopc_lower_coms, self.n_point)
            dopc_upper_coms = com_savgol(dopc_upper_coms, self.n_point)

        else:

            pass

        lower_sn1_sn2 = combine_sn1_sn2(dopc_lower_sn1_df, dopc_lower_sn2_df)
        upper_sn1_sn2 = combine_sn1_sn2(dopc_upper_sn1_df, dopc_upper_sn2_df)

        lower_sn1_sn2 = lower_sn1_sn2.iloc[:, :-1]
        upper_sn1_sn2 = upper_sn1_sn2.iloc[:, :-1]

        self.processed_dfs = (lower_sn1_sn2, upper_sn1_sn2)
        self.processed_acyl_coms = (dopc_lower_coms, dopc_upper_coms)

        scaler_path = os.path.join(self.model_scaler_dir, self.scaler_name)
        scaler = pickle.load(open(scaler_path, 'rb'))

        unscaled_lower, unscaled_upper = self.processed_dfs
        scaled_lower = scaler.transform(unscaled_lower)
        scaled_upper = scaler.transform(unscaled_upper)

        self.processed_scaled_dfs = scaled_lower, scaled_upper

    def reset_lo_prob_cutoff(self, new_lo_cutoff):

        self.lo_prob_cutoff = new_lo_cutoff

    def classify(self, uses_predict_proba=False):

        model_path = os.path.join(self.model_scaler_dir, self.model_name)
        model = pickle.load(open(model_path, 'rb'))

        scaled_lower, scaled_upper = self.processed_scaled_dfs

        if uses_predict_proba == False:

            predictions_lower = model.predict(scaled_lower)
            predictions_upper = model.predict(scaled_upper)

            if self.red_type == "ft_last":
                predictions_lower = predictions_lower.reshape(self.n_red, -1)
                predictions_upper = predictions_upper.reshape(self.n_red, -1)

            else:
                predictions_lower = predictions_lower.reshape(self.stop_frame_real, -1)
                predictions_upper = predictions_upper.reshape(self.stop_frame_real, -1)
        else:

            predictions_lower = model.predict_proba(scaled_lower)
            predictions_upper = model.predict_proba(scaled_upper)

            if self.red_type == "ft_last":

                predictions_lower = predictions_lower.reshape(self.n_red, -1, 2)
                predictions_upper = predictions_upper.reshape(self.n_red, -1, 2)

            else:
                predictions_lower = predictions_lower.reshape(self.stop_frame_real, -1, 2)
                predictions_upper = predictions_upper.reshape(self.stop_frame_real, -1, 2)

            predictions_lower = predictions_lower[:, :, 1]
            predictions_upper = predictions_upper[:, :, 1]

        self.prediction_probs = predictions_lower, predictions_upper
        self.prediction_labels = predictions_lower > self.lo_prob_cutoff, predictions_upper > self.lo_prob_cutoff

    def predict_frame(self, frame):

        lower_leaflet_array, upper_leaflet_array = self.processed_acyl_coms
        lower_predictions, upper_predictions = self.prediction_labels

        lower_leaflet_array_frame = lower_leaflet_array[frame]
        upper_leaflet_array_frame = upper_leaflet_array[frame]

        lower_prediction_frame = lower_predictions[frame]
        upper_prediction_frame = upper_predictions[frame]

        label_lower_1 = lower_leaflet_array_frame[lower_prediction_frame == True]
        label_lower_0 = lower_leaflet_array_frame[lower_prediction_frame == False]

        label_upper_1 = upper_leaflet_array_frame[upper_prediction_frame == True]
        label_upper_0 = upper_leaflet_array_frame[upper_prediction_frame == False]

        return label_lower_0, label_lower_1, label_upper_0, label_upper_1

    def lo_ld_proportions(self):

        lower_ratios_all = []
        upper_ratios_all = []

        if self.red_type == "ft_last":

            frames = self.n_red

        else:

            frames = self.stop_frame_real

        for t in range(frames):
            pred_one_t = self.predict_frame(t)

            lower_pred_one_t_ld = pred_one_t[0].shape[0]
            lower_pred_one_t_lo = pred_one_t[1].shape[0]
            lo_ld_prop_lower_one_t = lower_pred_one_t_lo / (lower_pred_one_t_ld + lower_pred_one_t_lo)

            upper_pred_one_t_ld = pred_one_t[2].shape[0]
            upper_pred_one_t_lo = pred_one_t[3].shape[0]
            lo_ld_prop_upper_one_t = upper_pred_one_t_lo / (upper_pred_one_t_ld + upper_pred_one_t_lo)

            lower_ratios_all.append(lo_ld_prop_lower_one_t)
            upper_ratios_all.append(lo_ld_prop_upper_one_t)

        return np.array(lower_ratios_all), np.array(upper_ratios_all)

    def predict_average_probability(self):

        lower_leaflet_array, upper_leaflet_array = self.processed_acyl_coms
        predictions_lower, predictions_upper = self.prediction_probs

        lower_leaflet_array = np.swapaxes(lower_leaflet_array, 0, 1)  # (no_lipids, timesteps, 3)
        upper_leaflet_array = np.swapaxes(upper_leaflet_array, 0, 1)

        predictions_lower = np.swapaxes(predictions_lower, 0, 1)  # (no_lipids, timesteps)
        predictions_upper = np.swapaxes(predictions_upper, 0, 1)

        no_lipids = lower_leaflet_array.shape[0]

        av_coords_all_lipids_lower = np.empty((no_lipids, 3))
        av_probs_all_lipids_lower = np.empty(no_lipids)
        av_coords_all_lipids_upper = np.empty((no_lipids, 3))
        av_probs_all_lipids_upper = np.empty(no_lipids)

        for lipid in range(no_lipids):
            av_coords_all_lipids_lower[lipid] = np.mean(lower_leaflet_array[lipid], axis=0)
            av_probs_all_lipids_lower[lipid] = np.mean(predictions_lower[lipid])

            av_coords_all_lipids_upper[lipid] = np.mean(upper_leaflet_array[lipid], axis=0)
            av_probs_all_lipids_upper[lipid] = np.mean(predictions_upper[lipid])

        return av_coords_all_lipids_lower, av_probs_all_lipids_lower, av_coords_all_lipids_upper, av_probs_all_lipids_upper


    def plot_predicted_average_all_frames_simple(self, title, plot_type="2d", figure_size=[12, 8]):

        coord_av_lower, prob_av_lower, coord_av_upper, prob_av_upper = self.predict_average_probability()

        coord_av_lower_x = coord_av_lower[:, 0]
        coord_av_lower_y = coord_av_lower[:, 1]
        coord_av_lower_z = coord_av_lower[:, 2]

        coord_av_upper_x = coord_av_upper[:, 0]
        coord_av_upper_y = coord_av_upper[:, 1]
        coord_av_upper_z = coord_av_upper[:, 2]

        # colors_lower = [i for i in prob_av_lower for j in range(0,coord_av_lower.shape[0])]
        # colors_upper = [i for i in prob_av_upper for j in range(0,coord_av_upper.shape[0])]

        colors_lower = prob_av_lower
        colors_upper = prob_av_upper

        if plot_type != "2d":

            fig = plt.figure(figsize=[12, 8])
            plot = fig.add_subplot(111, projection='3d')
            cog_x_lower = np.mean(coord_av_lower_x, axis=0)
            cog_y_lower = np.mean(coord_av_lower_y, axis=0)
            cog_z_lower = np.mean(coord_av_lower_z, axis=0)
            lower = plot.scatter(coord_av_lower_x, coord_av_lower_y, coord_av_lower_z, marker='o', s=2, alpha=0.9,
                                 c=colors_lower, vmin=0, vmax=1)

            cog_x_upper = np.mean(coord_av_upper_x, axis=0)
            cog_y_upper = np.mean(coord_av_upper_y, axis=0)
            cog_z_upper = np.mean(coord_av_upper_z, axis=0)
            upper = plot.scatter(coord_av_upper_x, coord_av_upper_y, coord_av_upper_z, marker='o', s=2, alpha=0.9,
                                 c=colors_upper, vmin=0, vmax=1)

            try:
                chol_lower, chol_upper = self.processed_chol_coms

                chol_lower_av = np.mean(chol_lower, axis=0)
                chol_upper_av = np.mean(chol_upper, axis=0)

                chol_lower_x = chol_lower_av[:, 0]
                chol_lower_y = chol_lower_av[:, 1]
                chol_lower_z = chol_lower_av[:, 2]

                chol_upper_x = chol_upper_av[:, 0]
                chol_upper_y = chol_upper_av[:, 1]
                chol_upper_z = chol_upper_av[:, 2]

                lower = plot.scatter(chol_lower_x, chol_lower_y, chol_lower_z, marker="o", s=4, alpha=0.9)
                upper = plot.scatter(chol_upper_x, chol_upper_y, chol_upper_z, marker="o", s=4, alpha=0.9)

                plot.set_xlabel('X')
                plot.set_ylabel('Y')
                plot.set_zlabel('Z')
                # plot.set_xlim(0,)
                # plot.set_ylim(0,)
                # plot.set_zlim(0,)
                plot.set_title(str(title))
                cbar = fig.colorbar(lower, )
                cbar.set_label("Lo probability")

            except ValueError:

                plot.set_xlabel('X')
                plot.set_ylabel('Y')
                plot.set_zlabel('Z')
                # plot.set_xlim(0,)
                # plot.set_ylim(0,)
                # plot.set_zlim(0,)
                plot.set_title(str(title))
                cbar = fig.colorbar(lower, )
                cbar.set_label("Lo probability")

                plt.show()

        else:

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figure_size)

            fig.suptitle(str(title), fontsize=16)

            lower = axes[0].scatter(coord_av_lower_x, coord_av_lower_y, c=colors_lower, vmin=0, vmax=1, cmap="viridis")

            # axes[0].scatter(chol_lower_red_frame[: , : , 0] , chol_lower_red_frame[: , : , 1],
            # color = "red")

            axes[0].set_xlabel("x/nm")
            axes[0].set_ylabel("y/nm")
            axes[0].set_title("Lower leaflet(CHOL in red) ")
            plt.colorbar(lower, label="Lo probability")

            upper = axes[1].scatter(coord_av_upper_x, coord_av_upper_y, c=colors_upper, vmin=0, vmax=1, cmap="viridis")

            # axes[1].scatter(chol_upper_red_frame[: , : , 0] , chol_upper_red_frame[: , : , 1],
            # color = "red")

            axes[1].set_xlabel("x/nm")
            axes[1].set_ylabel("y/nm")
            axes[1].set_title("Upper leaflet(CHOL in red) ")
            plt.colorbar(upper, cmap="viridis", label="Lo probability")

            try:
                chol_lower, chol_upper = self.processed_chol_coms

                chol_lower_av = np.mean(chol_lower, axis=0)
                chol_upper_av = np.mean(chol_upper, axis=0)

                chol_lower_x = chol_lower_av[:, 0]
                chol_lower_y = chol_lower_av[:, 1]
                chol_lower_z = chol_lower_av[:, 2]

                chol_upper_x = chol_upper_av[:, 0]
                chol_upper_y = chol_upper_av[:, 1]
                chol_upper_z = chol_upper_av[:, 2]

                axes[0].scatter(chol_lower_x, chol_lower_y,
                                color="red")

                axes[0].set_title("Lower leaflet(CHOL in red) ")

                axes[1].scatter(chol_upper_x, chol_upper_y,
                                color="red")

                axes[1].set_title("Upper leaflet(CHOL in red) ")



            except ValueError:

                axes[0].set_title("Lower leaflet")
                axes[1].set_title("Upper leaflet")

                plt.show()

####################### ---- *********************** ---- #######################
####################### ---- * voronoi tesselation * ---- #######################

###Voronoi Tesselation with Chol overlay
    def plot_predicted_average_voronoi_periodic(
            self,
            title="Periodic Voronoi Lo probability",
            leaflet="both",
            figure_size=(12, 6),
            cmap="viridis"
    ):
        """
        Plot Voronoi tessellation of average acyl-chain COM positions
        under 2D periodic boundary conditions, colored by mean Lo probability.

        Cholesterol COMs (time-averaged) are overlaid as red markers,
        but are NOT used as Voronoi sites.

        leaflet: "lower", "upper", or "both"
        """

        # 1) Get average DOPC COMs and average probabilities per lipid
        coord_av_lower, prob_av_lower, coord_av_upper, prob_av_upper = \
            self.predict_average_probability()

        # 2) Obtain approximate box lengths Lx, Ly (nm) using Bilayer.box_dimensions
        bilayer = Bilayer(
            self.sys_name,
            self.sys_type,
            self.start_frame_real,
            self.stop_frame_real,
        )
        box_dims = bilayer.box_dimensions()  # shape (n_frames, 3) in nm
        Lx, Ly = box_dims.mean(axis=0)[:2]

        # 3) Prepare figure / axes
        if leaflet == "both":
            fig, axes = plt.subplots(1, 2, figsize=figure_size)
            ax_lower, ax_upper = axes
        elif leaflet == "lower":
            fig, ax_lower = plt.subplots(1, 1, figsize=figure_size)
            ax_upper = None
        elif leaflet == "upper":
            fig, ax_upper = plt.subplots(1, 1, figsize=figure_size)
            ax_lower = None
        else:
            raise ValueError('leaflet must be "lower", "upper", or "both"')

        fig.suptitle(title, fontsize=16)
        norm = Normalize(vmin=0.0, vmax=1.0)

        # 4) Get time-averaged cholesterol COMs (if any)
        chol_lower_av_2d = None
        chol_upper_av_2d = None
        try:
            chol_lower, chol_upper = self.processed_chol_coms
            if chol_lower is not None:
                # chol_lower: (n_frames_red, n_chol, 3)
                chol_lower_av = np.mean(chol_lower, axis=0)  # (n_chol, 3)
                chol_lower_av_2d = chol_lower_av[:, :2]  # (n_chol, 2)
            if chol_upper is not None:
                chol_upper_av = np.mean(chol_upper, axis=0)
                chol_upper_av_2d = chol_upper_av[:, :2]
        except Exception:
            # In pure DOPC systems or if anything goes wrong, just skip CHOL overlay
            chol_lower_av_2d = None
            chol_upper_av_2d = None

        # ---- helper to plot one leaflet ----
        def _plot_leaflet(ax, coords3d, probs, chol_coords2d, label):
            # Use xy-plane for tessellation
            coms_2d = coords3d[:, :2]  # (N×2)

            # Periodic Voronoi for DOPC sites only
            regions, neighbors = _compute_periodic_voronoi(coms_2d, Lx, Ly)

            polys = []
            colors = []
            for idx, poly in regions.items():
                polys.append(poly)
                colors.append(probs[idx])

            poly_coll = PolyCollection(
                polys,
                array=np.array(colors),
                cmap=cmap,
                norm=norm,
                edgecolors="k",
                linewidths=0.2,
            )
            ax.add_collection(poly_coll)

            # Overlay DOPC COMs (small black dots)
            ax.scatter(coms_2d[:, 0], coms_2d[:, 1],
                       s=5, color="k", alpha=0.6)

            # Overlay CHOL COMs if provided
            if chol_coords2d is not None:
                ax.scatter(
                    chol_coords2d[:, 0], chol_coords2d[:, 1],
                    s=20, color="red", marker="x", alpha=0.9,
                    label="CHOL COM"
                )

            ax.set_xlim(0.0, Lx)
            ax.set_ylim(0.0, Ly)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x / nm")
            ax.set_ylabel("y / nm")
            ax.set_title(label)

            # Put legend only if chol is present
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8, frameon=False)

            return poly_coll

        mappable = None

        # Lower leaflet
        if ax_lower is not None:
            mappable = _plot_leaflet(
                ax_lower, coord_av_lower, prob_av_lower,
                chol_lower_av_2d,
                "Lower leaflet"
            )

        # Upper leaflet
        if ax_upper is not None:
            mappable = _plot_leaflet(
                ax_upper, coord_av_upper, prob_av_upper,
                chol_upper_av_2d,
                "Upper leaflet"
            )

        # Shared colorbar
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.8)
            cbar.set_label("Lo probability")

        #plt.tight_layout()
        plt.show()

######## ---- *** end of voronoi tesseleation with chol overla *** ---- ##########

### Voronoi Tesselation with Chol tiles

    def plot_predicted_average_voronoi_periodic_with_chol(
            self,
            title="Periodic Voronoi Lo probability (with CHOL)",
            leaflet="both",
            figure_size=(12, 6),
            cmap="viridis",
    ):
        """
        Periodic Voronoi tessellation of average acyl-chain COMs,
        including cholesterol COMs as Voronoi sites.

        - DOPC cells are colored by mean Lo probability per lipid.
        - CHOL cells are drawn with distinct red edges.
        leaflet: "lower", "upper", or "both".
        """

        # 1) Average DOPC COMs + mean prob per lipid (what you already use)
        coord_av_lower, prob_av_lower, coord_av_upper, prob_av_upper = \
            self.predict_average_probability()

        # 2) Get CHOL COMs (time-reduced), then average over time
        chol_lower, chol_upper = self.processed_chol_coms  # from process_all_input()

        if chol_lower is not None:
            chol_lower_av = np.mean(chol_lower, axis=0)  # (N_chol_lower, 3)
        else:
            chol_lower_av = None

        if chol_upper is not None:
            chol_upper_av = np.mean(chol_upper, axis=0)  # (N_chol_upper, 3)
        else:
            chol_upper_av = None

        # 3) Get box dimensions from Bilayer (in nm)
        bilayer = Bilayer(
            self.sys_name,
            self.sys_type,
            self.start_frame_real,
            self.stop_frame_real,
        )
        box_dims = bilayer.box_dimensions()  # (n_frames, 3)
        Lx, Ly = box_dims.mean(axis=0)[:2]

        # 4) Figure & axes
        if leaflet == "both":
            fig, axes = plt.subplots(1, 2, figsize=figure_size)
            ax_lower, ax_upper = axes
        elif leaflet == "lower":
            fig, ax_lower = plt.subplots(1, 1, figsize=figure_size)
            ax_upper = None
        elif leaflet == "upper":
            fig, ax_upper = plt.subplots(1, 1, figsize=figure_size)
            ax_lower = None
        else:
            raise ValueError('leaflet must be "lower", "upper", or "both"')

        fig.suptitle(title, fontsize=16)
        norm = Normalize(vmin=0.0, vmax=1.0)

        # ---------- helper to draw one leaflet ----------

        def _plot_leaflet(ax, coord_av, prob_av, chol_av, label):
            """
            coord_av : (N_lipids, 3) average DOPC COMs
            prob_av  : (N_lipids,) mean Lo probability per lipid
            chol_av  : (N_chol, 3) average CHOL COMs or None
            """

            # DOPC COMs in xy-plane
            dopc_xy = coord_av[:, :2]
            n_lipids = dopc_xy.shape[0]

            # Combine with CHOL COMs (if present)
            if chol_av is not None and chol_av.size > 0:
                chol_xy = chol_av[:, :2]
                n_chol = chol_xy.shape[0]
                coms_all = np.vstack([dopc_xy, chol_xy])  # (N_lipids+N_chol, 2)
            else:
                chol_xy = None
                n_chol = 0
                coms_all = dopc_xy
            N_total = coms_all.shape[0]

            # Build periodic Voronoi on combined sites
            regions = _compute_periodic_voronoi_chol_tiles(coms_all, Lx, Ly)

            # Separate polygons for DOPC vs CHOL
            polys_dopc = []
            colors_dopc = []
            polys_chol = []

            for idx, poly in regions.items():
                if idx < n_lipids:
                    polys_dopc.append(poly)
                    colors_dopc.append(prob_av[idx])
                else:
                    polys_chol.append(poly)

            # PolyCollection for DOPC (colored by Lo probability)
            lipid_coll = None
            if polys_dopc:
                lipid_coll = PolyCollection(
                    polys_dopc,
                    array=np.array(colors_dopc),
                    cmap=cmap,
                    norm=norm,
                    edgecolors="k",
                    linewidths=0.2,
                )
                ax.add_collection(lipid_coll)

            # PolyCollection for CHOL (no colorbar mapping; red-edged cells)
            if polys_chol:
                chol_coll = PolyCollection(
                    polys_chol,
                    facecolors="red",
                    edgecolors="darkred",
                    linewidths=0.6,
                    alpha=0.35,  # Transparency so DOPC map still visible
                )
                ax.add_collection(chol_coll)

            # Scatter points: DOPC = black, CHOL = red
            ax.scatter(dopc_xy[:, 0], dopc_xy[:, 1],
                       s=5, color="k", alpha=0.6, label="DOPC COM")

            if chol_xy is not None and n_chol > 0:
                ax.scatter(chol_xy[:, 0], chol_xy[:, 1],
                           s=10, color="red", alpha=0.8, label="CHOL COM")

            ax.set_xlim(0.0, Lx)
            ax.set_ylim(0.0, Ly)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x / nm")
            ax.set_ylabel("y / nm")
            ax.set_title(label)
            ax.legend(loc="upper right", fontsize=8)

            return lipid_coll

        # ---------- lower leaflet ----------

        mappable = None
        if ax_lower is not None:
            mappable = _plot_leaflet(
                ax_lower,
                coord_av_lower,
                prob_av_lower,
                chol_lower_av,
                "Lower leaflet",
            )

        # ---------- upper leaflet ----------

        if ax_upper is not None:
            mappable = _plot_leaflet(
                ax_upper,
                coord_av_upper,
                prob_av_upper,
                chol_upper_av,
                "Upper leaflet",
            )

        # Shared colorbar for Lo probability (DOPC only)
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.8)
            cbar.set_label("Lo probability (DOPC)")


        plt.show()

####################### ---- *********************** ---- #######################




####################### ---- *********************** ---- #######################

