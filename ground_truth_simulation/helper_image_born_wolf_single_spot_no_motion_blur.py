import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from absl import flags
from utils import get_logger

_logger = get_logger('Image Simulation with single spot.')

flags.DEFINE_integer('signal', 10, 'peak signal intensity level.')
flags.DEFINE_integer('Trial', 0, 'Trial-th dataset.')
flags.DEFINE_integer('spot_label', 0, 'the $spot_label$-th single spot.')


def image_simulation_born_wolf_single_spot(background, peak_intensity, D, Trial, num_images, num_spots, spot_label):
    """
    Assume particle is following browinian motion model with Born-Wolf PSF.
    :param num_spots: number of fluorescent particles appearing in the image
    :param background: background noise
    :param peak_intensity: peak signal level, i.e., "G"
    :param D: diffusing at the same speed of D in three dimensions
    :param Trial: simulate the Trial-th dataset 
    :param num_images: number of images for a single datasest
    :param  spot_label: the current fluorescent particle of interest is the (spot_label)-th one.
    :return: simulated images of (Trial)-th dataset
    """
    cwd = os.getcwd()
    dir_name = cwd + '/brownian_Trial' + str(Trial) + 'N10G' + str(peak_intensity) + 'Image' + str(
        num_images) + 'D' + str(D) + '_born_wolf/'
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
    scaled_intensity = float(peak_intensity)
    lambda_intensity = np.zeros((num_pixels, num_images))
    start_time = datetime.now()

    # # # # # # # # # # # # # # # # # # # # enter imaging process # # # # # # # # # # # # # # # #  # # # # # # # # # # #
    # load interpolation for table lookup
    fn_info = np.load('../data/fn_lambda_base_born_wolf_PIN_normalization.npy', allow_pickle=True)
    fn = fn_info.item()

    _logger.info('Spot #' + str(spot_label))
    # load ground truth with subsamples
    read_path = './brownian_traj_gt/brownian_3D_trajectories_Trial' + str(Trial) + '_Image' + str(
        num_images) + 'Spots' + str(num_spots) + '/ground_truth_all_spot_' + str(spot_label) + '.csv'
    ground_truth_all = np.loadtxt(read_path, delimiter=",", skiprows=0)
    x_ground_truth_all = ground_truth_all[0, :num_images]
    y_ground_truth_all = ground_truth_all[1, :num_images]

    # set storage room
    Info = np.zeros((num_pixels, num_images))

    # begin image forming process with sub-sample accumulations
    for k in range(num_images):
        _logger.info('    Generate Image #' + str(k))
        for j in range(num_pixels):
            # single particle has effect on 16-by-16 subregion only, and the sensor position is (0, 0).
            if (x_ground_truth_all[k]+x_center[0, j])**2 + (y_ground_truth_all[k]+y_center[0, j])**2 <= 128*delta_x**2:
                loc_x = x_ground_truth_all[k] + x_center[0, j]
                loc_y = y_ground_truth_all[k] + y_center[0, j]
                # # using table lookup
                tmp_lambda = fn(np.array([loc_x, loc_y]))
                Info[j, k] = tmp_lambda[0]
        lambda_intensity[:, k] = scaled_intensity * Info[:, k]
    _logger.info('Complete simulating spot #' + str(spot_label))

    pd.DataFrame(lambda_intensity).to_csv(dir_name + 'lambda_intensity_Spot_' + str(spot_label) + '.csv',
                                          header=False, index=False)
    end_time = datetime.now()
    _logger.info('Runtime for the imaging process with single spot is {}'.format(
        end_time - start_time))


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    for spot in range(200):
        image_simulation_born_wolf_single_spot(background=10, peak_intensity=flags.FLAGS.signal, D=0.01, Trial=flags.FLAGS.Trial,
                                               num_images=100, num_spots=200, spot_label=spot)
