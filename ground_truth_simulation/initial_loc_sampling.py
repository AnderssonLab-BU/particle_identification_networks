# https://stackoverflow.com/questions/27499139/how-can-i-set-a-minimum-distance-constraint-for-generating-points-with-numpy-ran
import matplotlib.pyplot as plt
import numpy as np


def generate_points_with_min_distance(num_spots, shape, min_dist, Trial):
    # compute grid shape based on number of points
    n = num_spots-1
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(-int(shape[1]/2), int(shape[1]/2), num_x, dtype=np.float32)
    y = np.linspace(-int(shape[1]/2), int(shape[1]/2), num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))
    # perturb points
    max_movement = (init_dist - min_dist)/2
    # ensure SCC generates different trajectories
    np.random.seed(Trial)
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise

    return coords


if __name__ == '__main__':
    pix_width = 0.1
    coords = generate_points_with_min_distance(
        num_spots=20, shape=(400, 400), min_dist=10, Trial=7)
    plt.scatter(coords[:, 0], coords[:, 1], s=10)
    plt.show()
    plt.savefig("tmp7_init_locs.png")
    print(coords[:, 0]*pix_width)
    print(coords[:, 1]*pix_width)
