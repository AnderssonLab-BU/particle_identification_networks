import os

import numpy as np
import pandas as pd


def find_readout_noise(save_dir, sigma_all, heatmap_ids_all, edge, desired_emitter):
    """
    This is to help readout noise in a given subregion
    Args: be careful about the coordinates in python o -→ x
                                                     ↓ y
        i (int): the detected emitter locates at the i-th column (x axis).
        j (int): the detected emitter locates at the j-th row (y axis).
        edge : the cropped img is in an edge-by-edge pixlated region.
    """

    # compute the left-top point (aj, ai) = (j*8-4, i*8-4)
    for t, id in enumerate(heatmap_ids_all):
        print("t = %d" % t)
        # compute the top-left corner of the cropping region which is 16-by-16 pixelated.
        ai = id[0] * 8 - 4  # top
        aj = id[1] * 8 - 4  # left
        # review the ground truth heatmap calculation for details.
        pix_id = ai * 512 + aj 
        # calculate position of current emitter
        rec_id = []
        for i in range(edge):
            add_item = i * 512
            for j in range(edge):
                rec_id.append(pix_id + j + add_item)
        rec_sigma = [sigma_all[int(k)] for k in rec_id]
        df = pd.DataFrame(rec_sigma)
        df.to_csv(save_dir + "readout_spot%d_t%04d.csv" %
                  (desired_emitter, t), index=False, header=False)


if __name__ == '__main__':
    trial = 1
    SBR = 5
    method = "PIN2"
    for desired_emitter in range(20):
        dir_path = '../extraction/'
        heatmap_ids_all = np.loadtxt(dir_path + "SBR%d_%s_Trial%d/heatmap_ids/heatmap_ids_all_spot%d.csv" %
                                     (SBR, method, trial, desired_emitter), delimiter=",", skiprows=0)
        save_dir = dir_path + "SBR%d_%s_Trial%d/readout_noise/spot%d/" % (
            SBR, method, trial, desired_emitter)
        os.makedirs(save_dir, exist_ok=True)
        sigma_all = np.loadtxt('../sigma_all_p512.csv', delimiter=",", skiprows=0)
        find_readout_noise(save_dir=save_dir, sigma_all=sigma_all, heatmap_ids_all=heatmap_ids_all,
                           edge=16, desired_emitter=desired_emitter)
