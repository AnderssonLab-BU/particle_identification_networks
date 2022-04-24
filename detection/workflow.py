import gc
import math
import os
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import (TqdmToLogger, get_logger, prepare_training_checkpoint_directory)

from detection.config import get_default_detection_config
from detection.dhdataset import DoubleHelixDataset
from detection.bwdataset import BornWolfDataset
from detection.astigdataset import AstigmatismDataset
from detection.pin import ParticleIdentificationNetwork
from detection.pinmk2 import ParticleIdentificationNetworkMK2
from detection.pinmk3 import ParticleIdentificationNetworkMK3

torch.multiprocessing.set_sharing_strategy('file_system')

flags.DEFINE_integer("num_machines", 1, "Number of machines.")
flags.DEFINE_integer("local_machine", 0,
                     "Master node is 0, worker nodes starts from 1."
                     "Max should be num_machines - 1.")

flags.DEFINE_integer("num_gpus", 1, "Number of GPUs per machines.")
flags.DEFINE_string("config_file", "experiments/default_detection.yaml",
                    "Default Configuration File.")

flags.DEFINE_string("master_ip", "127.0.0.1",
                    "Master node IP for initialization.")
flags.DEFINE_integer("master_port", 12000,
                     "Master node port for initialization.")

FLAGS = flags.FLAGS


def train_model_on_dataset(rank, train_cfg):
    _logger = get_logger("training", save_to_dir="./")
    dist_rank = rank + train_cfg.LOCAL_MACHINE * train_cfg.NUM_GPU_PER_MACHINE
    dist.init_process_group(backend="nccl", rank=dist_rank,
                            world_size=train_cfg.WORLD_SIZE,
                            init_method=train_cfg.INIT_METHOD)
    if train_cfg.PSF == "Double-Helix":
        dataset = DoubleHelixDataset(train_cfg.TRAIN.DATA)
        valid_dataset = DoubleHelixDataset(train_cfg.TRAIN.VALID_DATA)
    elif train_cfg.PSF == "Born-Wolf":
        dataset = BornWolfDataset(train_cfg.TRAIN.DATA)
        valid_dataset = BornWolfDataset(train_cfg.TRAIN.VALID_DATA)
    elif train_cfg.PSF == "Astigmatism":
        dataset = AstigmatismDataset(train_cfg.TRAIN.DATA)
        valid_dataset = AstigmatismDataset(train_cfg.TRAIN.VALID_DATA)

    if train_cfg.MODEL == "MK1":
        model = ParticleIdentificationNetwork()
    elif train_cfg.MODEL == "MK2":
        model = ParticleIdentificationNetworkMK2()
    elif train_cfg.MODEL == "MK3":
        model = ParticleIdentificationNetworkMK3()
    model = model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank],
                                    output_device=rank,
                                    broadcast_buffers=train_cfg.WORLD_SIZE > 1)
    model.train()

    train_sampler = DistributedSampler(dataset)
    valid_sampler = DistributedSampler(valid_dataset)
    dataloader = DataLoader(dataset, batch_size=train_cfg.TRAIN.BATCH_SIZE,
                            num_workers=train_cfg.TRAIN.NUM_WORKERS,
                            sampler=train_sampler)
    validloader = DataLoader(valid_dataset, batch_size=train_cfg.TRAIN.BATCH_SIZE,
                             num_workers=train_cfg.TRAIN.NUM_WORKERS,
                             sampler=valid_sampler)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=train_cfg.TRAIN.LR.BASE_LR)
    lr_scheduler = StepLR(optimizer,
                          step_size=train_cfg.TRAIN.LR.STEP_SIZE,
                          gamma=train_cfg.TRAIN.LR.WEIGHT_DECAY)
    global_step = 0
    if train_cfg.TRAIN.CONTINUE_FROM != "":
        ckpt = torch.load(train_cfg.TRAIN.CONTINUE_FROM,
                          map_location=lambda storage, loc: storage.cpu())
        model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        train_cfg.TRAIN.START_EPOCH = ckpt["epoch"] + 1

    best_average_precision = 0.0
    for epoch in range(train_cfg.TRAIN.START_EPOCH, train_cfg.TRAIN.EPOCH):
        pbar = tqdm(total=len(dataloader), leave=False,
                    desc="Training Epoch %d" % epoch,
                    file=TqdmToLogger(),
                    mininterval=1, maxinterval=100, )
        for data in dataloader:
            optimizer.zero_grad()
            loss = model.module.compute_loss(data, rank)
            if train_cfg.MODEL == "MK3":
                total_loss = loss["feat2_loss"] + \
                    loss["feat3_loss"] + loss["feat4_loss"]
            else:
                total_loss = loss
            if (global_step < 1000
                or not (math.isnan(total_loss.data.item())
                        or math.isinf(total_loss.data.item())
                        or total_loss.data.item() > train_cfg.TRAIN.LOSS_CLIP_VALUE)):
                total_loss.backward()
                optimizer.step()
            pbar.update()
            if global_step % train_cfg.TRAIN.PRINT_FREQ == 0:
                gc.collect()
                if rank == 0:
                    _logger.info("EPOCH\t%d; STEP\t%d; TRAIN LOSS\t%.4f" % (
                        epoch, global_step, total_loss.data.item()))
                    if train_cfg.MODEL == "MK3":
                        _logger.info("feat2_loss\t%.4f; feat3_loss\t%.4f; feat4_loss\t%.4f" % (
                            loss["feat2_loss"].data.item(), loss["feat3_loss"].data.item(), loss["feat4_loss"].data.item()))

            if global_step % train_cfg.TRAIN.EVAL_FREQ == 0 and global_step != 0:  
                # add validation into training process
                model.eval()
                all_targets = []
                all_predictions = []
                pbar = tqdm(total=len(validloader), leave=False,
                            desc="Validation Epoch %d" % epoch,
                            file=TqdmToLogger(),
                            mininterval=1, maxinterval=100, )
                for val_data in validloader:
                    reshaped_gt_map = val_data["gt_heatmap"].cuda(
                        rank).reshape([-1])
                    target = [torch.zeros_like(reshaped_gt_map)
                              for _ in range(train_cfg.NUM_GPU_PER_MACHINE)]
                    prediction = [torch.zeros_like(reshaped_gt_map)
                                  for _ in range(train_cfg.NUM_GPU_PER_MACHINE)]
                    dist.all_gather(target, reshaped_gt_map)
                    with torch.no_grad():
                        if train_cfg.MODEL == "MK3":
                            pred = model.module.make_prediction(
                                val_data["img"].cuda(rank))
                        else:
                            pred = model.forward(
                                val_data["img"].cuda(rank)).reshape([-1])
                        dist.all_gather(prediction, pred)
                    all_targets += target
                    all_predictions += prediction
                    pbar.update()
                pbar.close()
                # compute validation average precision
                _logger.info("Computing and gathering all heatmaps.")
                all_targets = torch.cat(all_targets).cpu().numpy()
                all_predictions = torch.cat(all_predictions).cpu().numpy()
                average_precision = average_precision_score(all_targets, all_predictions)

                # if AP increases, then save the training weights
                if average_precision >= best_average_precision:
                    _logger.info("BETTER AP FOUND: %.4f" % average_precision)
                    best_average_precision = average_precision
                    if rank == 0:  
                        checkpoint_file = os.path.join(train_cfg.LOG_DIR,
                                                       train_cfg.EXPR_NAME, "checkpoints",
                                                       "CKPT-E%d-S%d.pth" % (
                                                           epoch, global_step))
                        torch.save({"epoch": epoch, "global_step": global_step-1,
                                    "state_dict": model.state_dict(),
                                    "optimizer": optimizer.state_dict()}, checkpoint_file)
                        _logger.info("CHECKPOINT SAVED AT: %s" %
                                     checkpoint_file)
                model.train()
            global_step += 1
        pbar.close()
        lr_scheduler.step()

    dist.destroy_process_group()
    _logger.info("FINISHED.")


def eval_model_on_dataset(eval_cfg):
    _logger = get_logger("evaluation")
    if eval_cfg.PSF == "Double-Helix":
        dataset = DoubleHelixDataset(eval_cfg.EVAL.DATA)
    elif eval_cfg.PSF == "Born-Wolf":
        dataset = BornWolfDataset(eval_cfg.EVAL.DATA)
    elif eval_cfg.PSF == "Astigmatism":
        dataset = AstigmatismDataset(eval_cfg.EVAL.DATA)

    dataloader = DataLoader(dataset, batch_size=eval_cfg.EVAL.BATCH_SIZE,
                            num_workers=eval_cfg.EVAL.NUM_WORKERS)
    if eval_cfg.MODEL == "MK1":
        model = ParticleIdentificationNetwork()
    elif eval_cfg.MODEL == "MK2":
        model = ParticleIdentificationNetworkMK2()
    elif eval_cfg.MODEL == "MK3":
        model = ParticleIdentificationNetworkMK3()
    model = model.cuda()
    model.eval()
    _logger.info("Loading model PIN weight from %s." %
                 eval_cfg.EVAL.RESTORE_FROM)
    ckpt = torch.load(eval_cfg.EVAL.RESTORE_FROM,
                      map_location=lambda storage, loc: storage.cpu())
    restore_kv = {key.replace(
        "module.", ""): ckpt["state_dict"][key] for key in ckpt["state_dict"].keys()}
    model.load_state_dict(restore_kv, strict=True)
    _logger.info("Model weight loaded.")
    pbar = tqdm(total=len(dataloader), leave=False,
                desc="Evaluation",
                file=TqdmToLogger(),
                mininterval=1, maxinterval=100, )
    target = []
    prediction = []
    rec = dict()
    for dp in dataloader:
        # convert tensor to numpy array
        target.append(dp["gt_heatmap"].numpy().reshape([-1]))
        with torch.no_grad():
            if eval_cfg.MODEL == "MK3":
                pred = model.make_prediction(
                    dp["img"].cuda()).cpu().numpy()
            else:
                pred = model.forward(
                    dp["img"].cuda()).reshape([-1]).cpu().numpy()
            prediction.append(pred)
        pbar.update()
    pbar.close()
    target = np.concatenate(target, axis=0)
    prediction = np.concatenate(prediction, axis=0)
    precision, recall, _ = precision_recall_curve(target, prediction)
    average_precision = average_precision_score(target, prediction)
    _logger.info("EVAL AP FOUND: %.5f" % average_precision)

    SBR = int(eval_cfg.EVAL.DATA.G_VALUES[0]/10)
    rec["SBR"] = SBR
    rec["AP"] = float("{0:0.5f}".format(average_precision))
    rec["target"] = target.tolist()
    rec["prediction"] = prediction.tolist()

    with open('SBR' + str(SBR) + '_average_precision.json', 'w') as outfile:
        json.dump(rec, outfile)
    _logger.info("FINISH SBR = " + str(SBR) + ".")
    return average_precision


if __name__ == "__main__":
    FLAGS(sys.argv)
    cfg = get_default_detection_config()
    cfg.merge_from_file(FLAGS.config_file)
    cfg.NUM_GPU_PER_MACHINE = FLAGS.num_gpus
    cfg.NUM_MACHINES = FLAGS.num_machines
    cfg.LOCAL_MACHINE = FLAGS.local_machine
    cfg.WORLD_SIZE = FLAGS.num_machines * FLAGS.num_gpus
    cfg.EXPR_NAME = cfg.EXPR_NAME + "_" + datetime.now().strftime(
        "%m_%d.%H:%M:%S.%f")
    cfg.INIT_METHOD = "tcp://%s:%d" % (FLAGS.master_ip, FLAGS.master_port)
    if cfg.TYPE == "TRAIN":
        prepare_training_checkpoint_directory(cfg)
        mp.spawn(train_model_on_dataset, args=(cfg,),
                 nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
    elif cfg.TYPE == "EVAL":
        all_G = [10]
        rec_ap = []
        for signal in all_G:
            cfg.EVAL.DATA.G_VALUES = [signal]
            ap = eval_model_on_dataset(cfg)
            rec_ap.append(ap)
        print(rec_ap)
