import os

import numpy as np
import pandas as pd
import torch
from detection.config import get_default_detection_config
from detection.pin import ParticleIdentificationNetwork
from detection.pinmk2 import ParticleIdentificationNetworkMK2
from detection.pinmk3 import ParticleIdentificationNetworkMK3
from detection.real_world_dataset import RealWorldDataset

def find_heatmap_loc_of_an_emitter(cfg, method, desired_emitter, threshold, heatmap_ids_all, index):
    heatmap_edge = 150
    if method == "PIN1":
        model = ParticleIdentificationNetwork().cuda(0)
        ckpt = torch.load("../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK1-TRAIN_12_08.20:20:53.519117/checkpoints/CKPT-E10-S20000.pth",
                          map_location=lambda storage, loc: storage.cpu())
    if method == "PIN2":
        model = ParticleIdentificationNetworkMK2().cuda(0)
        ckpt = torch.load("../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK2-TRAIN_12_08.23:17:05.870921/checkpoints/CKPT-E5-S9000.pth",
                          map_location=lambda storage, loc: storage.cpu())
    if method == "PIN3":
        model = ParticleIdentificationNetworkMK3().cuda(0)
        ckpt = torch.load("../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK3-TRAIN_12_09.12:19:33.976674/checkpoints/CKPT-E1-S4000.pth",
                          map_location=lambda storage, loc: storage.cpu())

    restore_kv = {key.replace("module.", ""): ckpt["state_dict"][key] for key in
                  ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)
    dataset = RealWorldDataset(cfg.REAL_WORLD.DATA)
    dp = dataset.__getitem__(index)

    with torch.no_grad():
        if method == "PIN1" or method == "PIN2":
            output = model(dp["img"].unsqueeze(0).cuda(0))
        elif method == "PIN3":
            output = model.make_prediction(dp["img"].unsqueeze(0).cuda(0))
            output = output.reshape([1, heatmap_edge, heatmap_edge])

    prob_matrix = output[0].cpu().numpy()
    detect_matrix = np.where(prob_matrix > threshold, 1, 0)

    # =============== find the heatmap position of a fixed emitter as a function of time t ===============
    # we are interested in observing the "desired emitter"
    cur_emitter = 0
    # record the heatmap position of the "desired emitter" at initial time t = 0.
    if index == 0:
        print("Locate at the emitter at t = %d." % index)
        for i in range(heatmap_edge):
            for j in range(heatmap_edge):
                if detect_matrix[i][j] == 1:
                    if cur_emitter == desired_emitter:
                        heatmap_ids_all.append([i, j])
                    cur_emitter += 1
                if cur_emitter == desired_emitter + 1:
                    break
            if cur_emitter == desired_emitter + 1:
                break
    else:
        # now to find emitter at time t = 1, 2, .... 99.
        print("Locate at the emitter at t = %d." % index)
        min_distance = 3
        nearest_i = -1
        nearest_j = -1
        for i in range(heatmap_edge):
            for j in range(heatmap_edge):
                # we find the same emitter by nearest neighbor method,
                # in this case, the distance of the same emitter at two continuous time is initialized to 3 heatmap pixels.
                prev_i, prev_j = heatmap_ids_all[-1]
                if abs(i-prev_i) < min_distance and abs(j-prev_j) < min_distance:
                    min_distance = max(abs(i-prev_i), abs(j-prev_j))
                    nearest_i = i
                    nearest_j = j
        if nearest_i == -1 or nearest_j == -1:
            print("Lost Tracking!")
        else:
            heatmap_ids_all.append([nearest_i, nearest_j])
    return heatmap_ids_all


if __name__ == '__main__':
    SBR_level = "medium"
    method = "PIN2"
    threshold_val = 0.5
    num_images = 100
    num_spots = 1
    cfg = get_default_detection_config()
    if SBR_level == "high":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_high_SBR_jpgs/"
    elif SBR_level == "medium":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_medium_SBR_jpgs/"
    elif SBR_level == "low":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_low_SBR_jpgs/"

    for desired_emitter in range(num_spots):
        print('Start %s SBR level, %s for the emitter # %d.' %(SBR_level, method, desired_emitter))
        save_dir = "../extraction/real_world_%sSBR_%s/heatmap_ids/" % (SBR_level, method)
        os.makedirs(save_dir, exist_ok=True)

        heatmap_ids_all = []
        for i in range(num_images):
            heatmap_ids_all = find_heatmap_loc_of_an_emitter(cfg = cfg, method=method, desired_emitter=desired_emitter, threshold=threshold_val, heatmap_ids_all=heatmap_ids_all, index=i)
        # we only save the results of which length >= 50 timesteps.
        if len(heatmap_ids_all) >= 50:
            df = pd.DataFrame(heatmap_ids_all)
            df.to_csv(save_dir + "heatmap_ids_all_spot%d.csv" %
                      desired_emitter, index=False, header=False)
            # np.save(save_dir + "heatmap_ids_all_spot%d.npy" % desired_emitter, heatmap_ids_all)
            print("Finished!")
        else:
            print("The output is abandoned: recorded timesteps < 50!")
