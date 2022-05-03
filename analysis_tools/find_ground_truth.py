import json
import os

import numpy as np
import pandas as pd


def read_gt(file_path, trial):
    with open(file_path) as json_file:
        trajectories = json.load(json_file)
    # lookup in json
    particles_in_trial = trajectories[str(trial)]
    return particles_in_trial


def find_ground_truth_of_a_spot(save_path, ground_truth_file_path, trial, num_spots, desired_emitter, heatmap_ids_all):
    particles_in_trial = read_gt(ground_truth_file_path, trial)
    # we can find out the ground truth according to the initial time "loc_time = 0".
    loc_time = 0
    pix_width = 0.1
    heatmap_id = heatmap_ids_all[loc_time]
    i_hm = heatmap_id[0]
    j_hm = heatmap_id[1]
    est_x = (512/2 - j_hm * 8) * pix_width
    est_y = (i_hm * 8 - 512/2) * pix_width
    print("Est.(x, y) = (%f, %f))" % (est_x, est_y))
    dist_min = 2  # unit - um
    focus_spot_id = -1
    for particle in range(num_spots):
        cur_gt = np.array(particles_in_trial[str(particle)])
        gt_x, gt_y, _ = cur_gt[:, loc_time]
        if abs(gt_x - est_x) <= dist_min and abs(gt_y - est_y) <= dist_min:
            dist_min = max(abs(gt_x - est_x), abs(gt_y - est_y))
            focus_spot_id = particle
    if focus_spot_id == -1:
        print("Lost tracking!")
        return

    # 3 x 100, rows denote x, y, z positions.
    selected_gt = np.array(particles_in_trial[str(focus_spot_id)])
    save_path = save_path + "gt"
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame(selected_gt)
    df.to_csv(save_path + "/gt_trial%d_spot%d.csv" %
              (trial, desired_emitter), index=False, header=False)
    print("GT.(x, y) = (%f, %f))" %
          (selected_gt[0][loc_time], selected_gt[1][loc_time]))

    return focus_spot_id, selected_gt


if __name__ == "__main__":
    ground_truth_file_path = "../data/SPT_BROWN/3d_traj_brownian.json"
    trial = 1
    SBR = 5
    method = "PIN2"
    num_spots = 20
    for desired_emitter in range(20):
        print('Emitter # %d' % desired_emitter)
        read_path = "../extraction/SBR%d_%s_Trial%d/heatmap_ids/heatmap_ids_all_spot%d.csv" % (
            SBR, method, trial, desired_emitter)
        heatmap_ids_all = np.loadtxt(read_path, delimiter=",", skiprows=0)
        save_path = "../extraction/SBR%d_%s_Trial%d/" % (
            SBR, method, trial)
        find_ground_truth_of_a_spot(save_path, ground_truth_file_path, trial, num_spots, desired_emitter, heatmap_ids_all)
