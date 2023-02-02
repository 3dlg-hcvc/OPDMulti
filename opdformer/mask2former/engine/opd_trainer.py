from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    DatasetEvaluator,
    print_csv_format,
    inference_context,
)
from detectron2.utils import comm
from typing import Any, Dict, List, Set
from collections import OrderedDict
import os
import torch
import copy
import itertools
import logging
from mask2former.data import MotionDatasetMapper
from mask2former.evaluation import (
    MotionEvaluator,
    motion_inference_on_dataset,
)
import logging
from contextlib import ExitStack
import json
from detectron2.utils.comm import get_world_size
from torch import nn
import time


class OPDTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "OPD":

            # MotionNet: Evaluation for motion data (Including traditional coco evaluation)
            if "TYPE_MATCH" in cfg.MODEL:
                type_match = cfg.MODEL.TYPE_MATCH
            else:
                type_match = False

            if "PART_CAT" in cfg.MODEL:
                part_cat = cfg.MODEL.PART_CAT
            else:
                part_cat = False

            if "MICRO_AVG" in cfg.MODEL:
                micro_avg = cfg.MODEL.MICRO_AVG
            else:
                micro_avg = False

            if "AxisThres" in cfg.MODEL:
                AxisThres = cfg.MODEL.AxisThres
            else:
                AxisThres = 10

            if "OriginThres" in cfg.MODEL:
                OriginThres = cfg.MODEL.OriginThres
            else:
                OriginThres = 0.25

            # This is used for OPD
            evaluator_list.append(MotionEvaluator(
                dataset_name,
                cfg,
                True,
                output_folder,
                motionnet_type=cfg.MODEL.MOTIONNET.TYPE,
                MODELATTRPATH=cfg.MODEL.MODELATTRPATH,
                TYPE_MATCH=type_match,
                PART_CAT=part_cat,
                MICRO_AVG=micro_avg,
                AxisThres=AxisThres,
                OriginThres=OriginThres,
            ))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MotionDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = MotionDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * \
                        cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    # Modify this function to support evaluating on the exsited inference file
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if "INFERENCE_FILE" in cfg:
                inference_file = cfg.INFERENCE_FILE
            else:
                inference_file = None
            if "FILTER_FILE" in cfg:
                filter_file = cfg.FILTER_FILE
            else:
                filter_file = None
            if "FILTER_TYPE" in cfg:
                filter_type = cfg.FILTER_TYPE
            else:
                filter_type = None
            if "TYPE_IDX" in cfg:
                type_idx = cfg.TYPE_IDX
            else:
                type_idx = None
            results_i = motion_inference_on_dataset(
                model, data_loader, evaluator, inference_file, filter_file, filter_type, type_idx
            )
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
