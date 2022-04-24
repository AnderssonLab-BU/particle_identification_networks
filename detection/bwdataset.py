import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_logger

_DELTA_X = 0.1
_DELTA_Y = 0.1


class BornWolfDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()
        self._logger = get_logger("born_wolf_dataset")
        self.data_cfg = data_cfg.clone()
        with open(self.data_cfg.GT_JSON) as f:
            trajectories = json.load(f)
        self.images = list()
        for trial in range(self.data_cfg.TRIAL_START, self.data_cfg.TRIAL_END):
            particles_in_trial = trajectories[str(trial)]
            for image in range(self.data_cfg.IMAGE_RANGE):
                gt_locs = []
                for p in particles_in_trial:
                    particle = np.array(particles_in_trial[p])
                    gt_loc = particle[:, image]
                    gt_locs.append(gt_loc)
                for g in self.data_cfg.G_VALUES:
                    path_to_image = os.path.join(
                        self.data_cfg.HOME, "%dspots_img_N10G%d_born_wolf" % (self.data_cfg.NUM_SPOTS, g),  "%dspots_img_N10G%dTrial%d" % (self.data_cfg.NUM_SPOTS, g, trial), "%04d.jpg" % image)
                    self.images.append(
                        {"path": path_to_image, "gt_locs": gt_locs})
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dp = self.images[index].copy()
        frame = cv2.imread(dp["path"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            self._logger.warning("Frame is None, PATH: %s" % dp["path"])
            dp["img"] = torch.zeros([512, 512])
        else:
            dp["img"] = torch.from_numpy(frame).to(dtype=torch.float32)
        if len(dp["img"].shape) != 2:
            self._logger.warning("Image not in grayscale.")
        w, h = dp["img"].shape
        dp["normed_img"] = dp["img"] / torch.max(dp["img"])
        gt_heatmap = np.zeros(
            shape=[w//self.data_cfg.GT_STRIDE, h//self.data_cfg.GT_STRIDE])
        for gt_loc in dp["gt_locs"]:
            x = w/2-gt_loc[0]/_DELTA_X
            y = h/2+gt_loc[1]/_DELTA_Y
            x = x//self.data_cfg.GT_STRIDE
            y = y//self.data_cfg.GT_STRIDE
            gt_heatmap[int(y), int(x)] = 1.0
        dp["gt_heatmap"] = torch.from_numpy(gt_heatmap).to(dtype=torch.float32)
        return dp
