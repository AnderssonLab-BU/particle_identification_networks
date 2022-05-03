import os

import matplotlib.pyplot as pyplt
import numpy as np
from detection.bwdataset import BornWolfDataset
from detection.config import get_default_detection_config


def crop_image(save_dir, heatmap_id, desired_emitter, num_spots, G):
    from PIL import Image
    cfg = get_default_detection_config()
    cfg.EVAL.DATA.G_VALUES = [G]
    if num_spots != 20:
        cfg.EVAL.DATA.NUM_SPOTS = num_spots
        cfg.EVAL.DATA.GT_JSON = "../data/SPT_BROWN/3d_traj_brownian_%dspots.json" % (
            num_spots)
    dataset = BornWolfDataset(cfg.EVAL.DATA)
    for t, id in enumerate(heatmap_id):
        dp = dataset.__getitem__(t)
        raw_img = Image.open(dp["path"])
        i = id[0]
        j = id[1]
        # Assume the edge of cropped image is 16.
        # left_top = (j*8-4, i*8-4) # (left, top)
        # right_bottom = (j*8+12, i*8+12) # (right, bottom)
        left = j*8-4
        top = i*8-4
        right = j*8+12
        bottom = i*8+12
        cropped_img = raw_img.crop((left, top, right, bottom))
        pyplt.imshow(cropped_img, cmap="gray")
        cropped_img.save(save_dir + "img%04d.jpg" % t)


if __name__ == "__main__":
    SBR = 5
    method = "PIN2"
    threshold_val = 0.4
    trial = 1
    dir_path = '../extraction/'
    for desired_emitter in range(20):
        heatmap_ids_all = np.loadtxt(dir_path + "SBR%d_%s_Trial%d/heatmap_ids/heatmap_ids_all_spot%d.csv" %
                                     (SBR, method, trial, desired_emitter), delimiter=",", skiprows=0)
        save_dir = dir_path + \
            "SBR%d_%s_Trial%d/cropped_imgs/spot%d/" % (
                SBR, method, trial, desired_emitter)
        os.makedirs(save_dir, exist_ok=True)
        crop_image(save_dir=save_dir, heatmap_id=heatmap_ids_all,
                   desired_emitter=desired_emitter, num_spots=20, G=int(10 * SBR))
    print('Finished!')
