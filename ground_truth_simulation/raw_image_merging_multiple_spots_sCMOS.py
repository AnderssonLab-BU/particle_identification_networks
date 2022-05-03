import os
from datetime import datetime

import cv2
import numpy as np
from detection.utils import get_logger

_logger = get_logger('Image Simulation.')


def image_simulation_merging_multiple_spots(background, peak_intensity, D, Trial, num_images, num_spots):
    """
    Assume particle is following brownian motion.
    :param background: background noise
    :param peak_intensity: peak signal level, i.e., "G"
    :param D: diffusing at the same speed of D in all dimensions
    :param Trial: simulate the Trial-th dataset
    :param num_images: simulate number of images for every single dataset
    :param num_spots: number of fluorescent particles appearing in an image
    :return: simulated images of (i)th dataset, where i=trial_label
    """
    dir_path = os.getcwd()
    dir_name = dir_path + '/brownian_Trial' + str(Trial) + 'N10G' + str(peak_intensity) + 'Image' + str(
        num_images) + 'D' + str(D) + '_astig_' + str(num_spots) + 'spots/'
    _IMAGE_SHAPE = [512, 512]
    num_pixels = 512 * 512
    sigma_all = np.loadtxt('../data/sigma_all_p512.csv',
                           delimiter=",", skiprows=0)
    lambda_intensity = np.zeros((num_pixels, num_images))
    photon_observation = np.zeros((num_pixels, num_images))

    start_time = datetime.now()
    for i in range(num_spots):
        tmp_lambda_intensity = np.loadtxt(
            dir_name + 'lambda_intensity_Spot_' + str(i) + '.csv', delimiter=",", skiprows=0)
        lambda_intensity = lambda_intensity + tmp_lambda_intensity

    dir_save = dir_path + \
        '/%dspots_img_N10G%dTrial%d' % (num_spots, peak_intensity, Trial) + '/'
    os.makedirs(dir_save, exist_ok=True)
    for k in range(num_images):
        res = np.abs(np.random.normal(0, sigma_all))
        photon_observation[:, k] = np.random.poisson(
            lambda_intensity[:, k] + background) + res
        img = np.reshape(photon_observation[:, k], _IMAGE_SHAPE)
        path_out = os.path.join(dir_save, "%04d.jpg" % k)
        if np.max(img) > 255:
            _logger.info("Photon number is larger than 255 at %.2f. %s" %
                         (np.max(img), path_out))
        cv2.imwrite(path_out, img)
    end_time = datetime.now()
    _logger.info("Runtime for merging all spots in an image is {}".format(
        end_time - start_time))


if __name__ == '__main__':
    image_simulation_merging_multiple_spots(
        background=10, peak_intensity=100, D=0.01, Trial=0, num_images=100, num_spots=20)
