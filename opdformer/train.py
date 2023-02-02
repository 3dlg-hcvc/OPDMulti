# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OPDFormer Training Script.

This script is a simplified version of the training script in mask2former/train_net.py.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    add_maskformer2_config,
    register_motion_instances,
    add_motionnet_config,
    OPDTrainer
)
from time import time
import argparse
import sys
import os
import datetime

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
    cfg.merge_from_list(args.opts)
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    # Input format
    cfg.INPUT.FORMAT = args.input_format
    if args.input_format == "RGB":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[0:3]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[0:3]
    elif args.input_format == "depth":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[3:4]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[3:4]
    elif args.input_format == "RGBD":
        pass
    else:
        raise ValueError("Invalid input format")

    cfg.MODEL.MODELATTRPATH = args.model_attr_path

    # Options for OPDFormer-V1/V2/V3
    cfg.MODEL.MOTIONNET.VOTING = args.voting
    if (not cfg.MODEL.MOTIONNET.TYPE == "BMOC_V1" and not cfg.MODEL.MOTIONNET.TYPE == "BMOC_V2" and not cfg.MODEL.MOTIONNET.TYPE == "BMOC_V3") and (args.voting != "none"):
        raise ValueError("Voting Option is only for BMOC_V1 or BMOC_V2 or BMOC_V3")

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="opdformer-V3")
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Train OPDFormer")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/swin/opd_v6_synthetic.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"output_temp/{datetime.datetime.now().isoformat()}",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--data-path",
        default=f"/localhome/xsa55/Xiaohao/OPD/dataset/MotionDataset_h5_6.11",
        # default=f"/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/backup/SmallDataset_h5_6.11",
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--input-format",
        default="RGBD",
        choices=["RGB", "RGBD", "depth"],
        help="input format (RGB, RGBD, or depth)",
    )
    parser.add_argument(
        "--model_attr_path",
        required=False,
        default="/localhome/xsa55/Xiaohao/OPD/dataset/OPD/dataset/MotionDataset_h5_6.11/urdf-attr.json",
        help="indicating the path to the statistics on the diagonal length",
    )
    # Parameters for the OPDFormer-V1, V2, V3
    parser.add_argument(
        "--voting",
        default="none",
        choices=["none", "median", "mean", "geo-median"],
        help="if not None, use voting strategy for the extrinsic parameters when evalaution",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    # Parameters for distributed training
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    return parser


def register_datasets(data_path, cfg):
    dataset_keys = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
    for dataset_key in dataset_keys:
        json = f"{data_path}/annotations/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json, imgs)

def main(args):
    cfg = setup_cfg(args)
    register_datasets(args.data_path, cfg)
    
    trainer = OPDTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    start = time()

    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    stop = time()
    print(str(stop - start) + " seconds")

