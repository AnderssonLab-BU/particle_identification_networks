import unittest
import os
import torch
import matplotlib.pyplot as pyplt
import numpy as np
from detection.config import get_default_detection_config
from detection.real_world_dataset import RealWorldDataset
from detection.pin import ParticleIdentificationNetwork
from detection.pinmk2 import ParticleIdentificationNetworkMK2
from detection.pinmk3 import ParticleIdentificationNetworkMK3


def detect_real_world(num_images, cfg, trained_weights, threshold_prob):
    # set color RGB
    box_color = [255, 160, 122]  # orange [255, 160, 122]; cyan [0,255,255]
    # Line thickness of 0.2 px
    thickness = 1

    if cfg.MODEL == "MK1":
        model = ParticleIdentificationNetwork()
        method = "PIN1"
    elif cfg.MODEL == "MK2":
        model = ParticleIdentificationNetworkMK2()
        method = "PIN2"
    elif cfg.MODEL == "MK3":
        model = ParticleIdentificationNetworkMK3()
        method = "PIN3"
    model = model.cuda(0)

    save_path = "./results_" + method + "/"
    os.makedirs(save_path, exist_ok=True)

    # trained_weights
    ckpt = torch.load(
        trained_weights, map_location=lambda storage, loc: storage.cpu())
    restore_kv = {key.replace(
        "module.", ""): ckpt["state_dict"][key] for key in ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)

    dataset = RealWorldDataset(cfg.REAL_WORLD.DATA)
    for img_index in range(num_images):
        print("Image No.%04d" % img_index)
        dp = dataset.__getitem__(index=img_index)
        with torch.no_grad():
            output = model(dp["img"].unsqueeze(0).cuda(0))
        if cfg.MODEL == "MK3":
            final_output = (output['feat2'] +
                            output['feat3']+output['feat4'])/3.0
        else:
            final_output = output

        prob_matrix = final_output[0].cpu().numpy()
        # # ====== plot heatmap of probability ====== # #
        pyplt.imshow(prob_matrix)
        cbar = pyplt.colorbar()
        cbar.set_label('probability', rotation=90)
        ax = pyplt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        pyplt.margins(0, 0)
        pyplt.savefig(save_path +
                      cfg.MODEL + "_bw_predict_prob_heatmap_img%d.jpg" % img_index,  bbox_inches='tight', pad_inches=0)
        pyplt.clf()

        # # ====== plot detectoin map with specific threshold ====== # #
        detect_matrix = np.where(prob_matrix >= threshold_prob, 1, 0)
        pyplt.imshow(detect_matrix)
        ax = pyplt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        pyplt.margins(0, 0)
        pyplt.savefig(save_path + cfg.MODEL + "_bw_predict_map_thresh%.3f_img%d.jpg" %
                      (threshold_prob, img_index),  bbox_inches='tight', pad_inches=0)
        pyplt.clf()

        # # ====== plot heatmap of original img ====== # #
        pyplt.imshow(dp["img"], cmap='gray')
        cbar = pyplt.colorbar()
        cbar.set_label('photon counts', rotation=90)
        ax = pyplt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        pyplt.margins(0, 0)
        pyplt.savefig(save_path + "bw_origin_img%d.jpg" % img_index,
                      bbox_inches='tight', pad_inches=0)
        pyplt.clf()

        # ======= plot the original image with predicting boxes =======
        import cv2
        raw_img = cv2.imread(dp["path"])

        for i in range(150):
            for j in range(150):
                if detect_matrix[i][j] == 1:
                    start_point = (j*8-6, i*8-6)
                    end_point = (j*8+14, i*8+14)
                    cv2.rectangle(raw_img, start_point,
                                  end_point, box_color, thickness)

        # Showing the converted image
        pyplt.imshow(raw_img)
        ax = pyplt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        pyplt.savefig(save_path + method + "_img%d_with_pred_boxes.jpg" %
                      img_index, bbox_inches='tight')
        pyplt.clf()


if __name__ == '__main__':
    cfg = get_default_detection_config()
    cfg.PSF = "Born-Wolf"
    cfg.REAL_WORLD.DATA.HOME = "../real_world_jpgs/real_world_high_SBR_jpgs/"
    threshold_prob = 0.3
    all_methods = ["MK1"] 
    num_images = 1

    for method in all_methods:
        cfg.MODEL = method
        if cfg.MODEL == "MK1":
            trained_weights = "../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK1-TRAIN_12_08.20:20:53.519117/checkpoints/CKPT-E10-S20000.pth"
        elif cfg.MODEL == "MK2":
            trained_weights = "../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK2-TRAIN_12_08.23:17:05.870921/checkpoints/CKPT-E5-S9000.pth"
        elif cfg.MODEL == "MK3":
            trained_weights = "../experiments/cnn_spt/BCE/Born-Wolf-PIN-MK3-TRAIN_12_09.12:19:33.976674/checkpoints/CKPT-E1-S4000.pth"

        detect_real_world(num_images=num_images, cfg=cfg, trained_weights=trained_weights, threshold_prob=threshold_prob)
