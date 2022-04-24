import json
import os
import random
from skimage import io
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_logger


class RealWorldDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()
        self._logger = get_logger("real_world_dataset")
        self.data_cfg = data_cfg.clone()
        self.images = list()
        for image in range(self.data_cfg.IMAGE_RANGE):
            path_to_image = os.path.join(
                self.data_cfg.HOME, "1200by1200_born_wolf_real_world_jpgs/real_world_%04d.jpg" % image)
            self.images.append({"path": path_to_image})
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dp = self.images[index].copy()
        frame = io.imread(dp["path"])
        if frame is None:
            self._logger.warning("Frame is None, PATH: %s" % dp["path"])
            dp["img"] = torch.zeros([1200, 1200])
        else:
            dp["img"] = torch.from_numpy(frame).to(dtype=torch.float32)
        if len(dp["img"].shape) != 2:
            self._logger.warning("Image not in grayscale.")
        return dp
