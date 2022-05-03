import os

import matplotlib.pyplot as pyplt
import numpy as np
from detection.config import get_default_detection_config
from detection.real_world_dataset import RealWorldDataset


def crop_image(save_dir, cfg, heatmap_id):
    from PIL import Image
    dataset = RealWorldDataset(cfg.REAL_WORLD.DATA)
    for t, id in enumerate(heatmap_id):
        dp = dataset.__getitem__(t)
        raw_img = Image.open(dp["path"])
        i = id[0]
        j = id[1]
        left = j*8-4
        top = i*8-4
        right = j*8+12
        bottom = i*8+12
        cropped_img = raw_img.crop((left, top, right, bottom))
        pyplt.imshow(cropped_img, cmap="gray")
        cropped_img.save(save_dir + "img%04d.jpg" % t)


if __name__ == "__main__":
    SBR_level = "low"
    method = "PIN2"
    num_emitters = 1
    dir_path = '../extraction/'
    cfg = get_default_detection_config()
    if SBR_level == "high":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_high_SBR_jpgs/"
    elif SBR_level == "medium":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_medium_SBR_jpgs/"
    elif SBR_level == "low":
        cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_low_SBR_jpgs/"

    for desired_emitter in range(num_emitters):
        heatmap_ids_all = np.loadtxt(dir_path + "real_world_%sSBR_%s/heatmap_ids/heatmap_ids_all_spot%d.csv" %
                                     (SBR_level, method, desired_emitter), delimiter=",", skiprows=0)
        save_dir = dir_path + \
            "real_world_%sSBR_%s/cropped_imgs/spot%d/" % (
                SBR_level, method, desired_emitter)
        os.makedirs(save_dir, exist_ok=True)
        crop_image(save_dir=save_dir, cfg=cfg, heatmap_id=heatmap_ids_all)
    print('Finished!')
