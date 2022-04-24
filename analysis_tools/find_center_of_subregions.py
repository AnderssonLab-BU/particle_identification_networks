import json
import os
from os import stat
from re import T

import cv2
import matplotlib.pyplot as pyplt
import numpy as np
import pandas as pd
import torch
from detection.bwdataset import BornWolfDataset
from detection.config import get_default_detection_config

from detection.pin import ParticleIdentificationNetwork
from detection.pinmk2 import ParticleIdentificationNetworkMK2
from detection.pinmk3 import ParticleIdentificationNetworkMK3


def find_all_centers(save_dir, heatmap_ids_all, edge, desired_emitter):
    """
    This is to help extract center position and readout noise in a given subregion
    Args: be careful about the coordinates in python o -→ x
                                                     ↓ y
        i (int): the detected emitter locates at the i-th column (x axis).
        j (int): the detected emitter locates at the j-th row (y axis).
        edge : the cropped img is in an edge-by-edge pixlated region.
    """
    delta_x = 0.1
    delta_y = 0.1
    xcenter_unit = np.load(
        '../data/xcenter_unit_p512.npy', allow_pickle=True)
    ycenter_unit = np.load(
        '../data/ycenter_unit_p512.npy', allow_pickle=True)
    x_center = delta_x * xcenter_unit
    y_center = delta_y * ycenter_unit

    # compute the left-top point (aj, ai) = (j*8-4, i*8-4)
    for t, id in enumerate(heatmap_ids_all):
        print("center position at t = %d" % t)
        # compute the top-left corner of the cropping region which is 16-by-16 pixelated.
        ai = id[0] * 8 - 4  # top
        aj = id[1] * 8 - 4  # left
        # review the ground truth heatmap calculation for details.
        pix_id = ai * 512 + aj 
        # calculate x center position of current emitter
        rec_id = []
        for i in range(edge):
            add_item = i * 512
            for j in range(edge):
                rec_id.append(pix_id + j + add_item)
        # Attention that the original setting of (x_center, y_center) and that of the calculated center positions are different.
        rec_x_center = [x_center[0, int(k)] for k in rec_id]
        rec_y_center = [y_center[0, int(k)] for k in rec_id]
        dfx = pd.DataFrame(rec_x_center)
        dfy = pd.DataFrame(rec_y_center)
        dfx.to_csv(save_dir + "x_center_spot%d_t%04d.csv" %
                   (desired_emitter, t), index=False, header=False)
        dfy.to_csv(save_dir + "y_center_spot%d_t%04d.csv" %
                   (desired_emitter, t), index=False, header=False)


if __name__ == '__main__':
    trial = 1
    SBR = 5
    method = "PIN2"
    for desired_emitter in range(20):
        dir_path = '../extraction/'
        heatmap_ids_all = np.loadtxt(dir_path + "SBR%d_%s_Trial%d/heatmap_ids/heatmap_ids_all_spot%d.csv" %
                                    (SBR, method, trial, desired_emitter), delimiter=",", skiprows=0)
        save_dir = dir_path + "SBR%d_%s_Trial%d/centers/spot%d/" % (SBR, method, trial, desired_emitter)
        os.makedirs(save_dir, exist_ok=True)
        find_all_centers(save_dir=save_dir, heatmap_ids_all=heatmap_ids_all,
                        edge=16, desired_emitter=desired_emitter)
