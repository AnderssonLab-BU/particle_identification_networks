import os
import random

import numpy as np
import pandas as pd
from absl import flags
from detection.utils import get_logger

from initial_loc_sampling import generate_points_with_min_distance

_logger = get_logger("Simulate ground truth trajectory.")

flags.DEFINE_integer("Trial", 0, "Trial-th dataset")
flags.DEFINE_integer("spot_label", 0, "Spot label")

_pix_width = 0.1


def brownian_traj_3d(Trial, num_images, num_spots):
    """
    This is to simulate trajectories in brownian environment of 3D
    :param Trial: label the ground truth of the "Trial-th" dataset
    :param num_images: number of images contained in a single datasest
    :param num_spots: number of fluorophores in an image
    :return: ground truth in x, y, and z position.
    """
    cwd = os.getcwd()
    dir_save = cwd + "brownian_3D_trajectories_Trial" + \
        str(Trial) + "_Image" + str(num_images) + "Spots" + str(num_spots)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    # basic parameter settings.
    numSub = 1  # no motion blur is considered to save computation time
    D = 0.01  # unit - [\mu m^2/s], assume Dx=Dy=Dz=D
    delta_t = 0.1  # image period
    dt = delta_t / numSub  # shutter period
    sigma_D = np.sqrt(2.0 * D * dt)

    # allocate space for x and y position [x;y;z]
    ground_truth_all = np.zeros((3, num_images * numSub))
    # initial positions
    coords = generate_points_with_min_distance(
        num_spots=num_spots, shape=(400, 400), min_dist=2, Trial=Trial)
    x0_all = coords[:, 0]*_pix_width
    y0_all = coords[:, 1]*_pix_width
    np.random.seed(Trial)
    z0_all = np.random.uniform(-0.5, 0.5, num_spots)

    for spot_label in range(num_spots):
        _logger.info("Trial #" + str(Trial) + ": spot #" + str(spot_label))
        ground_truth_all[0, 0] = x0_all[spot_label]
        ground_truth_all[1, 0] = y0_all[spot_label]
        ground_truth_all[2, 0] = z0_all[spot_label]
        # simulate the remaining trajectories with Rejection Sampling Algorithm
        for t in range(1, num_images * numSub):
            ground_truth_all[0, t] = ground_truth_all[0,
                                                      t - 1] + random.gauss(0, sigma_D)
            ground_truth_all[1, t] = ground_truth_all[1,
                                                      t - 1] + random.gauss(0, sigma_D)
            ground_truth_all[2, t] = ground_truth_all[2,
                                                      t - 1] + random.gauss(0, sigma_D)
        pd.DataFrame(ground_truth_all).to_csv(dir_save + "/ground_truth_all_spot_" + str(spot_label) + ".csv", header=False,
                                              index=False)


if __name__ == '__main__':
    for Trial in range(100):
        brownian_traj_3d(Trial=Trial, num_images=100, num_spots=20)
