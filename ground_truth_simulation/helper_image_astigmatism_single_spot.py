import os
import sys
from datetime import datetime
import math
from scipy import integrate
import numpy as np
import pandas as pd
from absl import flags
from utils import get_logger

_logger = get_logger('Image Simulation with single spot.')


def image_simulation_astigmatism_single_spot(num_spots, background, peak_intensity, D, Trial, num_images, spot_label):
    """
    Assume particle is following browinian motion model with Astigmatism PSF.
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
        num_images) + 'D' + str(D) + '_astig_' + str(num_spots) + 'spots/'
    os.makedirs(dir_name, exist_ok=True)
    SBR = int(peak_intensity / background)
    _logger.info('SBR' + str(SBR) + ', Trial ' + str(Trial) + ': create directory ... ')

    num_pixels = 512 * 512
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
    _logger.info('Spot #' + str(spot_label))
    # load ground truth with subsamples
    read_path = './brownian_traj_gt/brownian_3D_trajectories_Trial' + str(Trial) + '_Image' + str(num_images) + 'Spots' + str(num_spots) + '/ground_truth_all_spot_' + str(spot_label) + '.csv'
    ground_truth_all = np.loadtxt(read_path, delimiter=",", skiprows=0)
    x_ground_truth_all = ground_truth_all[0, :num_images]
    y_ground_truth_all = ground_truth_all[1, :num_images]
    z_ground_truth_all = ground_truth_all[2, :num_images]

    # set storage room
    Info = np.zeros((num_pixels, num_images))

    # begin image forming process with sub-sample accumulations
    for k in range(num_images):
        _logger.info('    Generate Image #' + str(k))
        loc_z = z_ground_truth_all[k]
        for j in range(num_pixels):
            # single particle has effect on 16-by-16 subregion only, and the sensor position is (0, 0).
            if (x_ground_truth_all[k]+x_center[0, j])**2 + (y_ground_truth_all[k]+y_center[0, j])**2 <= 128*delta_x**2:
                loc_x = x_ground_truth_all[k] + x_center[0, j]
                loc_y = y_ground_truth_all[k] + y_center[0, j]
                tmp_lambda = lookup_helper(loc_x, loc_y, loc_z)
                Info[j, k] = tmp_lambda
        lambda_intensity[:, k] = scaled_intensity * Info[:, k]
    _logger.info('Complete simulating spot #' + str(spot_label))

    pd.DataFrame(lambda_intensity).to_csv(dir_name + 'lambda_intensity_Spot_' + str(spot_label) + '.csv',
                                          header=False, index=False)
    end_time = datetime.now()
    _logger.info('Runtime for the imaging process with all single spots is {}'.format(
        end_time - start_time))


def lookup_helper(loc_x, loc_y, z):
    # for the jth pixel at time t, compute local positions.
    # loc_x = global_x + x_center[0, j] - sensor_position[t, 0]
    # loc_y = global_y + y_center[0, j] - sensor_position[t, 1]

    delta_x = 0.1
    delta_y = 0.1
    boundaries = 0.5 * np.array([-delta_x, delta_x, -delta_y, delta_y])

    def psf_tmp(t1, t2): return psf_astigmatism(loc_x - t1, loc_y - t2, z)
    tmp_lambda = integrate.dblquad(psf_tmp, boundaries[0], boundaries[1], lambda t1: boundaries[2],
                                   lambda t1: boundaries[3])
    lambda_base = tmp_lambda[0]
    return lambda_base


def psf_astigmatism(x, y, z):
    # Basic parameter settings
    # the parameters (Ax, Bx, dx, Ay, By, dy) are refered from Papers
    # < Smith, Carlas S., et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." Nature methods, 2010.>
    Ax = -0.0708
    Bx = -0.073
    cx = 0.389
    dx = 0.531
    Ay = 0.164
    By = 0.0417
    cy = -cx
    dy = dx
    sigma0x = 1.08
    sigma0y = 1.01
    pix_width = 0.1
    sigmax = psf_width_astigmatism(pix_width, sigma0x, Ax, Bx, cx, dx, z)
    sigmay = psf_width_astigmatism(pix_width, sigma0y, Ay, By, cy, dy, z)
    norm_val = 104.9166

    # Compute the astigmatism PSF
    res = norm_val * math.exp(-x**2/(2*sigmax**2) - y**2/(2*sigmay**2))
    return res


def psf_width_astigmatism(pix_width, sigma0, A, B, c, d, z):
    # compute the PSF width
    sigma = sigma0 * np.sqrt(1.0+((z-c)/d)**2 + A *
                             ((z-c)/d)**3 + B*((z-c)/d)**4)
    return pix_width * sigma


if __name__ == '__main__':
    num_spots = 20
    for spot in range(num_spots):
        image_simulation_astigmatism_single_spot(num_spots=num_spots, background=10, peak_intensity=100, D=0.01, Trial=0,
                                                 num_images=100, spot_label=spot)
