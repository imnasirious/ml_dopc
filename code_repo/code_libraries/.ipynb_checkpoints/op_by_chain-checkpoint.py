import numpy as np




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



def convolve_ops(ops_array, n_point):

    n_frames, n_lipids, n_carbons = ops_array.shape
    n_reduced = n_frames - n_point + 1
    kernel = np.ones(n_point) / n_point

    convolved = np.empty([n_reduced, n_lipids, n_carbons])

    for j in range(n_lipids):
        for c in range(n_carbons):
            convolved[:, j, c] = np.convolve(ops_array[:, j, c], kernel, mode="valid")

    return convolved


def split_ops_by_label(ops_array, pred_labels):

    lo_ops = []
    ld_ops = []

    for f in range(ops_array.shape[0]):
        frame_ops   = ops_array[f]        # (n_lipids, 16)
        frame_preds = pred_labels[f]      # (n_lipids,) boolean

        lo_ops.append(frame_ops[frame_preds == True])
        ld_ops.append(frame_ops[frame_preds == False])

    return lo_ops, ld_ops



def mean_op_profile(lo_upper, lo_lower, ld_upper, ld_lower):

    # concatenate upper and lower across all frames
    lo_all = np.vstack([frame for frame in lo_upper + lo_lower if len(frame) > 0])
    ld_all = np.vstack([frame for frame in ld_upper + ld_lower if len(frame) > 0])

    lo_mean = lo_all.mean(axis=0)   # (16,)
    lo_std  = lo_all.std(axis=0)
    ld_mean = ld_all.mean(axis=0)
    ld_std  = ld_all.std(axis=0)

    return lo_mean, lo_std, ld_mean, ld_std