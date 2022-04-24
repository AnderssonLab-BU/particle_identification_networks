import math
from datetime import datetime
import numpy as np
import scipy.integrate
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from utils import get_logger

_logger = get_logger("Interpolation for Born-Wolf PSF used in CNN.")


def lookup_helper(loc_x, loc_y):
    # for the jth pixel at time t, compute local positions.
    # loc_x = global_x + x_center[0, j] - sensor_position[t, 0]
    # loc_y = global_y + y_center[0, j] - sensor_position[t, 1]

    delta_x = 0.1
    delta_y = 0.1
    boundaries = 0.5 * np.array([-delta_x, delta_x, -delta_y, delta_y])
    wave_length = 0.54
    NA = 1.2
    sigmaxy = np.sqrt(2)*wave_length/(2*math.pi*NA)
    norm_val = 108.320251

    def psf_tmp(t1, t2): return psf_born_wolf(loc_x - t1, loc_y - t2, sigmaxy, norm_val)
    tmp_lambda = scipy.integrate.dblquad(psf_tmp, boundaries[0], boundaries[1], lambda t1: boundaries[2],
                                         lambda t1: boundaries[3])
    lambda_base = tmp_lambda[0]
    return lambda_base


def psf_born_wolf(x, y, sigmaxy, norm_val):
    # Note that peak_intensity is not included here
    res = norm_val * math.exp(-(x ** 2 + y ** 2) / (2 * sigmaxy ** 2))
    return res


if __name__ == '__main__':
    # ---------------------- Create interpolation for the ease of 2d Born-Wolf PSF calculation  ---------------------- # 
    start_p = -55.0
    end_p = 55
    num_p = 2201  
    loc_x_all = np.linspace(start_p, end_p, num_p)
    loc_y_all = np.linspace(start_p, end_p, num_p)
    V = np.zeros((len(loc_x_all), len(loc_y_all)))
    t1 = datetime.now()
    num_cores = 4  

    for i in range(len(loc_x_all)):
        _logger.info(str(i) + 'th x position')
        V[i, :] = Parallel(n_jobs=num_cores)(delayed(lookup_helper)(
            loc_x=loc_x_all[i], loc_y=loc_y_all[j]) for j in range(num_p))

    _logger.info('Begin the interpolation ... ')
    fn_lambda_base = RegularGridInterpolator((loc_x_all, loc_y_all), V)
    test_point = np.array([-51, 47])
    real_output = lookup_helper(test_point[0], test_point[1])
    _logger.info('Real lambda base is ' + str(real_output))
    approx_output = fn_lambda_base(test_point)
    _logger.info("Approx. lambda base is " + str(approx_output))
    _logger.info('Difference btw real and approx. value is' +
                 str(abs(real_output - approx_output)))
    save_path = "./"
    np.save(save_path + 'fn_lambda_base_born_wolf_CNN_normalization.npy', fn_lambda_base)
    t2 = datetime.now()
    _logger.info('Finish with runtime ' + str(t2 - t1))

    