import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from absl import flags
from utils import get_logger

_logger = get_logger('Image Simulation with single spot.')


def image_simulation_single_spot(num_spots, background, peak_intensity, D, Trial, num_images, spot_label):
    """
    Assume particle is following browinian motion model in 3D with Double-Helix PSF.
    :param num_spots: number of fluorescent particles appearing in the image
    :param background: background noise
    :param peak_intensity: peak signal level, i.e., "G"
    :param D: diffusing at the same speed of D in three dimensions
    :param Trial: simulate the Trial-th dataset 
    :param num_images: number of images for a single datasest
    :param  spot_label: the current fluorescent particle of interest is the (spot_label)-th one.
    :return: simulated images of (Trial)-th dataset
    """

    dir_path = os.getcwd()
    dir_name = dir_path + '/brownian_Trial' + str(Trial) + 'N10G' + str(peak_intensity) + 'Image' + str(
        num_images) + 'D' + str(D) + '_dh_' + str(num_spots) + 'spots/'
    os.makedirs(dir_name, exist_ok=True)
    SBR = int(peak_intensity / background)
    _logger.info('SBR' + str(SBR) + ', Trial ' +
                 str(Trial) + ': create directory ... ')

    num_pixels = 512 * 512
    r = 0.3  # radium is 300 nm
    delta_x = 0.1
    delta_y = 0.1
    xcenter_unit = np.load('../data/xcenter_unit_p512.npy', allow_pickle=True)
    ycenter_unit = np.load('../data/ycenter_unit_p512.npy', allow_pickle=True)
    x_center = delta_x * xcenter_unit
    y_center = delta_y * ycenter_unit

    # scaled intensity for table lookup
    # attention that we half the signal intensity so as to mimic the low-light conditions.
    scaled_intensity = float(peak_intensity)
    lambda_intensity = np.zeros((num_pixels, num_images))
    start_time = datetime.now()

    # # # # # # # # # # # # # # # # # # # # enter imaging process # # # # # # # # # # # # # # # #  # # # # # # # # # # #
    # load interpolation for table lookup
    fn_info = np.load(
        '../data/fn_lambda_base_dh_PIN_normalization.npy', allow_pickle=True)
    fn = fn_info.item()

    _logger.info('Spot #' + str(spot_label))
    # load ground truth with subsamples
    read_path = './brownian_traj_gt/brownian_3D_trajectories_Trial' + str(Trial) + '_Image' + str(
        num_images) + 'Spots' + str(num_spots) + '/ground_truth_all_spot_' + str(spot_label) + '.csv'
    ground_truth_all = np.loadtxt(read_path, delimiter=",", skiprows=0)
    x_ground_truth_all = ground_truth_all[0, :num_images]
    y_ground_truth_all = ground_truth_all[1, :num_images]
    z_ground_truth_all = ground_truth_all[2, :num_images]

    # convert to double-helix
    theta = -100.0 * z_ground_truth_all / 180 * math.pi
    x1_all = x_ground_truth_all + r * np.cos(theta)
    y1_all = y_ground_truth_all + r * np.sin(theta)
    x2_all = x_ground_truth_all + r * np.cos(theta + np.pi)
    y2_all = y_ground_truth_all + r * np.sin(theta + np.pi)

    # set storage room
    Info = np.zeros((num_pixels, num_images))

    # begin image forming process with sub-sample accumulations
    for k in range(num_images):
        _logger.info('    Generate Image #' + str(k))
        for j in range(num_pixels):
            # single particle has effect on 16-by-16 subregion only, and the sensor position is (0, 0).
            if (x_ground_truth_all[k]+x_center[0, j])**2 + (y_ground_truth_all[k]+y_center[0, j])**2 <= 128*delta_x**2:
                loc_x1 = x1_all[k] + x_center[0, j]
                loc_y1 = y1_all[k] + y_center[0, j]
                loc_x2 = x2_all[k] + x_center[0, j]
                loc_y2 = y2_all[k] + y_center[0, j]
                # # using table lookup
                tmp_lambda = fn(np.array([loc_x1, loc_y1])) + \
                    fn(np.array([loc_x2, loc_y2]))
                Info[j, k] = tmp_lambda[0]
        lambda_intensity[:, k] = scaled_intensity * Info[:, k]
    _logger.info('Complete simulating spot #' + str(spot_label))

    pd.DataFrame(lambda_intensity).to_csv(dir_name + 'lambda_intensity_Spot_' +
                                          str(spot_label) + '.csv', header=False, index=False)
    end_time = datetime.now()
    _logger.info('Runtime for the imaging process with single spot is {}'.format(
        end_time - start_time))


if __name__ == '__main__':
    num_spots = 20
    for spot in range(num_spots):
        image_simulation_single_spot(num_spots=num_spots, background=10,
                                     peak_intensity=100, D=0.01, Trial=0, num_images=100, spot_label=spot)
