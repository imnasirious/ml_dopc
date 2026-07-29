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

from code_libraries.bilayer_operations import Bilayer
from code_libraries.acyl_chain_preprocessing import *
from code_libraries.miscellaneous_functions import *
from lo_ld_classifier import *



save_dir = "../voronoi_plots"
os.makedirs(save_dir, exist_ok=True)

start_frame , stop_frame = 0 , 1000
sys_type = "sym"
sys_name_85 = "dopc_85"
sys_name_80 = "dopc_80"
sys_name_75 = "dopc_75"
sys_name_70 = "dopc_70"
sys_name_65 = "dopc_65"

#dopc_85 = clf_lo_ld(sys_name_85 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
#dopc_80 = clf_lo_ld(sys_name_80 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
#dopc_75 = clf_lo_ld(sys_name_75 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
dopc_70 = clf_lo_ld(sys_name_70 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
#dopc_65 = clf_lo_ld(sys_name_65 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)

dopc_70.process_all_input()

# For smooth colors use the true probabilities:
dopc_70.classify(uses_predict_proba=True)

# Periodic Voronoi for both leaflets
dopc_70.plot_predicted_average_voronoi_periodic(
    title="DOPC 70 – periodic Voronoi with CHOL Overlay",
    leaflet="both",
    figure_size=(14, 6),
)
plt.show()

dopc_70.plot_predicted_average_voronoi_periodic_with_chol(
    title="DOPC 70 – periodic Voronoi with CHOL Tiles",
    leaflet="both",
    figure_size=(14, 6),
)
plt.show()