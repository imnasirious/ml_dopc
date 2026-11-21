import numpy as np
import pandas as pd


##ROW ORDER CONVENTION FOR EVERY DATAFRAME IN THIS FILE: lipid-major
##row = lipid * frames + frame  (rows 0 .. frames-1 are lipid 0 over time, next frames rows are lipid 1, ...)
##input leaflet shape: (no_frames , no_lipids , no_atoms_per_lipid , 3)


def convolve_features_valid(full_df, n_to_reduce_to, total_ts_per_lipid, label_type):

    ##full_df must be lipid-major (as returned by op_height_df), label in last column
    ##output is lipid-major too: row = lipid * n_to_reduce_to + reduced_frame

    df_features = full_df.iloc[:, :-1]
    df_feature_column_names = df_features.columns
    original_df_total_size = df_features.shape[0]
    no_features = df_features.shape[1]

    # number of lipids
    no_lipids = original_df_total_size // total_ts_per_lipid

    assert original_df_total_size % total_ts_per_lipid == 0, \
        f"{original_df_total_size} rows is not a whole number of lipids x {total_ts_per_lipid} frames"
    assert 1 <= n_to_reduce_to <= total_ts_per_lipid, \
        f"n_to_reduce_to = {n_to_reduce_to} must be between 1 and total_ts_per_lipid = {total_ts_per_lipid}"

    # kernel length for uniform running average
    filt_argument = total_ts_per_lipid - (n_to_reduce_to - 1)
    filt_array = np.ones(filt_argument) / filt_argument

    # This will hold: [lipid][time_reduced][feature]
    convolved = np.zeros((no_lipids, n_to_reduce_to, no_features))

    # Loop over lipids
    for L in range(no_lipids):
        start = L * total_ts_per_lipid
        end   = start + total_ts_per_lipid

        lipid_block = df_features.iloc[start:end, :]   # (T, F)

        # For each feature, convolve down the time dimension
        for f in range(no_features):
            series = lipid_block.iloc[:, f].to_numpy()
            smoothed = np.convolve(series, filt_array, mode="valid")
            convolved[L, :, f] = smoothed  # length n_to_reduce_to

    # Flatten [lipid][time][feature] → rows × features
    out_matrix = convolved.reshape(no_lipids * n_to_reduce_to, no_features)

    out_df = pd.DataFrame(out_matrix, columns=df_feature_column_names)

    # Add labels
    label_value = 0 if label_type == "pure" else 1
    out_df["label"] = label_value

    return out_df


#########BLOCK AVERAGE##################

##non-overlapping blocks of n_block consecutive frames of each lipid:
##block m of lipid i = mean of that lipid's frames m*n_block ... (m+1)*n_block - 1
##number of blocks = total_ts_per_lipid // n_block; trailing frames that don't fill a whole block are dropped
##(n_block = 2, 4, 5, 10, 20, ... all divide 4000 exactly, so nothing is dropped for those)

def block_average_feature_df(full_df , total_ts_per_lipid , n_block , is_pure):

    ##full_df must be lipid-major (as returned by op_height_df), label in last column
    ##output is lipid-major too: row = lipid * no_blocks + block

    df_features = full_df.iloc[: , :-1]
    no_lipids = df_features.shape[0] // total_ts_per_lipid

    assert df_features.shape[0] == no_lipids * total_ts_per_lipid, \
        f"{df_features.shape[0]} rows is not a whole number of lipids x {total_ts_per_lipid} frames"
    assert 1 <= n_block <= total_ts_per_lipid, \
        f"n_block = {n_block} must be between 1 and total_ts_per_lipid = {total_ts_per_lipid}"

    no_blocks = total_ts_per_lipid // n_block

    features = df_features.to_numpy().reshape([no_lipids , total_ts_per_lipid , -1])     # (lipid, frame, feature)
    features = features[: , :no_blocks * n_block]
    blocked = features.reshape([no_lipids , no_blocks , n_block , -1]).mean(axis = 2)    # (lipid, block, feature)

    out_df = pd.DataFrame(blocked.reshape([no_lipids * no_blocks , -1]) , columns = df_features.columns)

    if is_pure == True:

        out_df["label"] = 0

    else:

        out_df["label"] = 1

    return out_df






##########ORDER PARAMETER CALCULATORS##########


def get_vector(coord1 , coord2):
    return coord1 - coord2

def get_cos_angle(coord1 , coord2 , z):  ##z input is a 3d numpy array
    z_norm = np.linalg.norm(z)
    vector_norm = np.linalg.norm(get_vector(coord1 , coord2))
    dot_prod = np.dot(get_vector(coord1 , coord2) , z)
    cos_angle = dot_prod / (z_norm * vector_norm)
    angle_radians = np.arccos(cos_angle)
    #angle_degrees = angle_radians * 180/np.pi
    angle_degrees = np.rad2deg(angle_radians)

    return cos_angle

    #return cos_angle, angle_radians , angle_degrees


def get_order_parameter(coord1 , coord2, z):
    cos_squared = get_cos_angle(coord1 , coord2 , z) **2
    return -1 * ((3*cos_squared - 1)/ 2)



def DOPC_op_H_av(lipid , chain_type , z_axis = np.array([0 , 0 , 1])):

    sn1_indices = np.array([ 41,  42,  43,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
       101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
       114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
       127, 128, 129, 130, 131, 132, 133])

    sn2_indices = np.array([32, 33, 34, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
       58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
       75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])

    #order_parameters = np.empty(16)

    if chain_type == "sn1":

        acyl_atoms = lipid[sn1_indices , :]

    else:

        acyl_atoms = lipid[sn2_indices , :]

    c1 , h11 , h12 = acyl_atoms[0] , acyl_atoms[1] , acyl_atoms[2]
    c2,  h21 , h22 = acyl_atoms[3] , acyl_atoms[4] , acyl_atoms[5]
    c3 , h31 , h32 = acyl_atoms[6] , acyl_atoms[7] , acyl_atoms[8]
    c4 , h41 , h42 = acyl_atoms[9] , acyl_atoms[10] , acyl_atoms[11]
    c5 , h51 , h52 = acyl_atoms[12] , acyl_atoms[13] , acyl_atoms[14]
    c6 , h61 , h62 = acyl_atoms[15] , acyl_atoms[16] , acyl_atoms[17]
    c7 , h71 , h72 = acyl_atoms[18] , acyl_atoms[19] , acyl_atoms[20]
    c8 , h8 = acyl_atoms[21] , acyl_atoms[22]
    c9 , h9 = acyl_atoms[23] , acyl_atoms[24]
    c10 , h101 , h102 = acyl_atoms[25] ,acyl_atoms[26] , acyl_atoms[27]
    c11 , h111 , h112 = acyl_atoms[28] , acyl_atoms[29] , acyl_atoms[30]
    c12 , h121 , h122 = acyl_atoms[31] , acyl_atoms[32] , acyl_atoms[33]
    c13 , h131 , h132 = acyl_atoms[34] , acyl_atoms[35] , acyl_atoms[36]
    c14 , h141 , h142 = acyl_atoms[37] , acyl_atoms[38] , acyl_atoms[39]
    c15 , h151 , h152 = acyl_atoms[40] , acyl_atoms[41] , acyl_atoms[42]
    c16 , h161 , h162 = acyl_atoms[43] , acyl_atoms[44] , acyl_atoms[45]

    c1_order = 0.5*(get_order_parameter(c1 , h11 , z_axis) + get_order_parameter(c1 , h12 , z_axis))
    c2_order = 0.5*(get_order_parameter(c2 , h21 , z_axis) + get_order_parameter(c2 , h22 , z_axis))
    c3_order = 0.5*(get_order_parameter(c3 , h31 , z_axis) + get_order_parameter(c3 , h32 , z_axis))
    c4_order = 0.5*(get_order_parameter(c4 , h41 , z_axis) + get_order_parameter(c4 , h42 , z_axis))
    c5_order = 0.5*(get_order_parameter(c5 , h51 , z_axis) + get_order_parameter(c5 , h52 , z_axis))
    c6_order = 0.5*(get_order_parameter(c6 , h61 , z_axis) + get_order_parameter(c6 , h62 , z_axis))
    c7_order = 0.5*(get_order_parameter(c7 , h71 , z_axis) + get_order_parameter(c7 , h72 , z_axis))
    c8_order = get_order_parameter(c8 , h8 , z_axis)
    c9_order = get_order_parameter(c9 , h9 , z_axis)
    c10_order = 0.5*(get_order_parameter(c10 , h101 , z_axis) + get_order_parameter(c10 , h102 , z_axis))
    c11_order = 0.5*(get_order_parameter(c11 , h111 , z_axis) + get_order_parameter(c11 , h112 , z_axis))
    c12_order = 0.5*(get_order_parameter(c12 , h121 , z_axis) + get_order_parameter(c12 , h122 , z_axis))
    c13_order = 0.5*(get_order_parameter(c13 , h131 , z_axis) + get_order_parameter(c13 , h132 , z_axis))
    c14_order = 0.5*(get_order_parameter(c14 , h141 , z_axis) + get_order_parameter(c14 , h142 , z_axis))
    c15_order = 0.5*(get_order_parameter(c15 , h151 , z_axis) + get_order_parameter(c15 , h152 , z_axis))
    c16_order = 0.5*(get_order_parameter(c16 , h161 , z_axis) + get_order_parameter(c16 , h162 , z_axis))


    orders = [c1_order , c2_order , c3_order , c4_order , c5_order , c6_order , c7_order,
             c8_order , c9_order , c10_order , c11_order , c12_order , c13_order,
             c14_order , c15_order , c16_order]

    return np.array(orders)


##########CHAIN HEIGHT CALCULATORS##########


def com_one_acyl_chain(chain_type , chain_array):

    sn1_masses = np.array([12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,
        1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1., 12.,  1., 12.,
        1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,
        1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1.,  1.])

    sn2_masses = np.array([12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,
            1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1., 12.,  1., 12.,
            1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,
            1., 12.,  1.,  1., 12.,  1.,  1., 12.,  1.,  1.,  1.])

    if chain_type == "sn1":

        mass_prod_sum = 0

        for atom_index in range(len(chain_array)):

            mass_prod = chain_array[atom_index]*sn1_masses[atom_index]
            mass_prod_sum += mass_prod

        return mass_prod_sum/np.sum(sn1_masses)

    else:

        mass_prod_sum = 0

        for atom_index in range(len(chain_array)):

            mass_prod = chain_array[atom_index]*sn2_masses[atom_index]
            mass_prod_sum += mass_prod

        return mass_prod_sum/np.sum(sn2_masses)


def chain_height_df(leaflet , frames, chain ,iqr_filter = False):

    ##output is lipid-major: row = lipid * frames + frame

    sn1_indices = [41, 42 , 43]
    sn1_indices.extend(i for i in range(91 ,138))

    sn2_indices = [32, 33 , 34]
    sn2_indices.extend(i for i in range(44 ,91))


    all_z_heights_all_lipids = []
    all_radius_xy_all_lipids = []
    no_lipids = leaflet.shape[1]


    if chain == "sn1":

        leaflet = leaflet[: , : , sn1_indices , :]
    else:
        leaflet = leaflet[: , : , sn2_indices , :]

    no_atoms_per_chain =leaflet.shape[2]

    for lipid in range(no_lipids):

        all_z_heights_one_lipid = []
        all_radius_xy_one_lipid = []

        for frame in range(frames):

            one_lipid = leaflet[frame , lipid]

            lipid_com = com_one_acyl_chain(chain, one_lipid.reshape(-1 , 3))
            lipid_shifted = one_lipid.reshape(-1 , 3) - lipid_com
            lipid_of = lipid_shifted.reshape([no_atoms_per_chain , 3])

            cstart, cend = lipid_of[0] , lipid_of[-4]
            height_z = np.linalg.norm(cstart[2] - cend[2])
            radius_xy = np.linalg.norm(cstart[:-1] - cend[:-1])
            all_z_heights_one_lipid.append(height_z)
            all_radius_xy_one_lipid.append(radius_xy)

        all_z_heights_all_lipids.append(all_z_heights_one_lipid)
        all_radius_xy_all_lipids.append(all_radius_xy_one_lipid)

    all_z_height_series = pd.Series(np.array(all_z_heights_all_lipids).reshape(-1,))
    all_radius_xy_series = pd.Series(np.array(all_radius_xy_all_lipids).reshape(-1,))
    all_z_height_df = pd.DataFrame(all_z_height_series)
    all_radius_xy_df = pd.DataFrame(all_radius_xy_series)
    full_df = pd.concat([all_z_height_df , all_radius_xy_df] , axis = 1)
    column_names = ["z_height" , "xy_radius"]
    full_df.columns = column_names

    if iqr_filter == False:

        return full_df

    else:

        return None

#         first_q = all_height_series.quantile(0.25)
#         inter_q = all_height_series.quantile(0.75) - first_q
#         lower_whisker = float(first_q - 1.5 * inter_q)

#         all_height_df = all_height_df[all_height_df.iloc[: , 0] > lower_whisker]

#         return all_height_df


##########ORDER PARAMETER AND CHAIN HEIGHT/RADIUS CONSTRUCTOR##########



def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

def op_height_df(leaflet, chain_type , frames, is_pure , heights = True):

    ##output is lipid-major: row = lipid * frames + frame
    ##columns: C1 ... C16 , xy_radius , z_height , label

    assert chain_type in ("sn1" , "sn2"), f"chain_type must be 'sn1' or 'sn2', got {chain_type}"
    assert frames <= leaflet.shape[0], f"frames = {frames} exceeds the {leaflet.shape[0]} frames in leaflet"

    no_lipids = leaflet.shape[1]
    all_c1_to_c16_ops_all_lipids = np.empty([no_lipids , frames , 16])

    for lipid in range(no_lipids):

        c1_to_c16_ops_one_lipid_all_frames = np.empty([frames , 16])

        for f in range(frames):

            one_lipid = leaflet[f , lipid]
            c1_to_c16_ops_one_frame = DOPC_op_H_av(one_lipid , chain_type)
            c1_to_c16_ops_one_lipid_all_frames[f] = c1_to_c16_ops_one_frame


        all_c1_to_c16_ops_all_lipids[lipid] = c1_to_c16_ops_one_lipid_all_frames

    c1_to_c16_ops_df = pd.DataFrame(all_c1_to_c16_ops_all_lipids.reshape([-1 , 16]))

    if is_pure == True:

        labels = [0 for i in range(len(c1_to_c16_ops_df))]

    else:

        labels = [1 for i in range(len(c1_to_c16_ops_df))]

    final_df = pd.concat([c1_to_c16_ops_df , pd.Series(labels)] , axis = 1)

    column_names = [str("C"+ str(i+1)) for i in range(16)]
    column_names.extend(["label"])

    final_df.columns = column_names

    if heights == True:

        height_df = chain_height_df(leaflet , frames , chain_type)
        df_label_last_two_rev = pd.concat([final_df , height_df] , axis = 1)
        final_df = swap_columns(df_label_last_two_rev , "label" , "xy_radius")



    return final_df


def combine_sn1_sn2(sn1_df , sn2_df):

    ##sn1_df and sn2_df must come from op_height_df on the same leaflet, same frames (same lipid-major row order)

    assert sn1_df.shape[0] == sn2_df.shape[0], "sn1_df and sn2_df have different numbers of rows"

    labels = pd.DataFrame(sn1_df.iloc[: , -1])

    sn1_dfs_c1_c16 = sn1_df.iloc[: , :16]
    sn2_dfs_c1_c16 = sn2_df.iloc[: , :16]

    sn1_sn2_c1_c16 = pd.concat([sn1_dfs_c1_c16 , sn2_dfs_c1_c16] , axis = 1 )

    sn1_dfs_xy_z = sn1_df[["xy_radius" , "z_height"]]
    sn2_dfs_xy_z = sn2_df[["xy_radius" , "z_height"]]

    sn1_sn2_xy_z = (sn1_dfs_xy_z + sn2_dfs_xy_z)/2

    sn1_sn2_features = pd.concat([sn1_sn2_c1_c16 , sn1_sn2_xy_z] , axis = 1)

    sn1_col_names = ["C" + str(i+1) + "_sn1" for i in range(0  , 16)]
    sn2_col_names = ["C" + str(i+1) + "_sn2" for i in range(0  , 16)]
    xy_z_col_names = ["mean_xy_rad" , "mean_z_height"]

    new_feat_col_names = sn1_col_names + sn2_col_names + xy_z_col_names

    sn1_sn2_features.columns = new_feat_col_names

    sn1_sn2_df = pd.concat([sn1_sn2_features , labels] , axis = 1)

    return sn1_sn2_df



        
        
        
# def sav_gol_features(full_df, window_size, poly_order, total_ts_per_lipid, label_type):
    
#     df_features = full_df.iloc[:, :-1]
#     df_labels   = full_df.iloc[:, -1]
    
#     df_feature_column_names = df_features.columns
    
#     no_lipids = int(full_df.shape[0] / total_ts_per_lipid)
    
#     original_df_total_size = df_features.shape[0]
#     no_features = df_features.shape[1]
    
#     # container for all features
#     filtered_features_all = []
    
#     for feature in range(no_features):
        
#         filtered_feature_set = []
        
#         for i in range(0, original_df_total_size, total_ts_per_lipid):
            
#             lipid_i = df_features.iloc[i:i+total_ts_per_lipid, feature].to_numpy()

#             filtered_lipid_i = savgol_filter(
#                 lipid_i,
#                 window_length=window_size,
#                 polyorder=poly_order,
#                 mode='interp'    
#             )
            
#             filtered_feature_set.append(filtered_lipid_i)
        
#         filtered_features_all.append(filtered_feature_set)
    
#     filtered_features_all = np.array(filtered_features_all)
    
#     filtered_features_all = filtered_features_all.reshape(no_features, -1)
#     filtered_features_all = np.swapaxes(filtered_features_all, 0, 1)
    
#     filtered_features_all_df = pd.DataFrame(filtered_features_all,
#                                             columns=df_feature_column_names)
    
#     if label_type == "pure":
#         labels = pd.Series([0] * len(filtered_features_all_df), name="label")
#     else:
#         labels = pd.Series([1] * len(filtered_features_all_df), name="label")
    
#     filtered_features_all_df = pd.concat(
#         [filtered_features_all_df, labels],
#         axis=1
#     )
    
#     return filtered_features_all_df
        


#########BLOCK AVERAGE################## 

####ONLY USE FOR ARRAYS OF EVEN LENGTH, AND N_REDUCE_TO should be exactly divisible by array
###length

# def block_average(array , reduced_size): ####best to just use on even sized arrays with len(array)&reduced_size=0

#     #if reduced_size > len(array)//2

#     array_len = len(array)

#     window_size = array_len//reduced_size
#     remainder = array_len%reduced_size

#     final_array = np.empty(reduced_size)
#     index_counter = 0

#     if remainder == 0:

#         for i in range(0 , array_len , window_size):

#             array_subselection = array[i : i+window_size]
#             subselection_average = np.mean(array_subselection)
#             final_array[index_counter] = subselection_average
#             index_counter += 1

#     else:

#         if remainder == 1:

#             reduced_size +=1

#         remaining_array = array[-reduced_size:]
#         remaining_array_av = np.mean(remaining_array)
#         final_array[-1] = remaining_array_av

#         front_array = array[:-reduced_size]

#         for i in range(0 , len(front_array) , window_size):

#             front_array_subselection = front_array[i: i + window_size]
#             front_array_sub_av = np.mean(front_array_subselection)
#             final_array[index_counter] = front_array_sub_av
#             index_counter +=1
#             #print(front_array_subselection , remaining_array)
            
#     return final_array

    

# def block_av_trajectory(leaflet_array , frames , n_to_reduce_to):

#     if leaflet_array.shape[0] == frames:
        
#         leaflet_array = np.swapaxes(leaflet_array , 0 , 1)
        
#     no_lipids = leaflet_array.shape[0]
#     atoms_per_lipid = leaflet_array.shape[2]

#     block_averaged_traj = np.empty([no_lipids , atoms_per_lipid, n_to_reduce_to, 3])

#     for lipid_i in range(no_lipids):
        
#         current_lipid_all_time_all_coords = leaflet_array[lipid_i , : , : , :]
#         current_lipid_all_conv_coords = np.empty([atoms_per_lipid, n_to_reduce_to , 3])
        
#         for atom in range(atoms_per_lipid):
            
#             current_lipid_atom_x_all_time = current_lipid_all_time_all_coords[: , atom , 0]
#             current_lipid_atom_y_all_time = current_lipid_all_time_all_coords[: , atom , 1]
#             current_lipid_atom_z_all_time = current_lipid_all_time_all_coords[: , atom , 2]
            
#             block_av_x = block_average(current_lipid_atom_x_all_time , n_to_reduce_to)
#             block_av_y = block_average(current_lipid_atom_y_all_time , n_to_reduce_to)
#             block_av_z = block_average(current_lipid_atom_z_all_time , n_to_reduce_to)
            
#             block_av_xyz_one_atom = np.array([block_av_x , block_av_y, block_av_z])
#             block_av_xyz_one_atom = np.swapaxes(block_av_xyz_one_atom , 0 , 1)
#             current_lipid_all_conv_coords[atom] = block_av_xyz_one_atom
        
#         block_averaged_traj[lipid_i] = current_lipid_all_conv_coords
        
#     #return block_averaged_traj
#     block_averaged_traj = np.swapaxes(block_averaged_traj , 2 , 1)
#     return np.swapaxes(block_averaged_traj , 0 , 1)


# def block_average_feature_df(df , frames, n_to_reduce_to , is_pure = False):

#     df = df.iloc[: , :-1]
#     feature_names = list(df.columns)

#     no_lipids = len(df) // frames
#     no_features = df.shape[1]
#     no_rows = df.shape[0]

#     final_array = np.empty([no_lipids , no_features, n_to_reduce_to])

#     index_counter = 0

#     for lipid_i in range(0 , len(df) , frames):

#         lipid_i_all_ts = df.iloc[lipid_i:lipid_i+frames , :]

#         for feature in range(no_features):

#             lipid_i_all_ts_feature = np.array(lipid_i_all_ts.iloc[: , feature])

#             lipid_i_block_av_feature = block_average(lipid_i_all_ts_feature , n_to_reduce_to)
#             final_array[index_counter , feature] = lipid_i_block_av_feature

#         index_counter += 1


#     final_array = final_array.reshape(-1 , no_features)

#     feature_df = pd.DataFrame(final_array , columns = feature_names)

#     if is_pure == False:

#         labels = pd.Series([0 for i in range(len(feature_df))])

#     else:

#         labels = pd.Series([1 for i in range(len(feature_df))])

#     final_df = pd.concat([feature_df , labels] , axis = 1)

#     feature_names.append("label")

#     final_df.columns = feature_names

#     return final_df 
        
        
