# Copyright (c) Facebook, Inc. and its affiliates.
import pdb
from typing import Tuple

import torch
from torch import device, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.tranform import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, geometric_median
from .modeling.criterion import convert_to_filled_tensor

import numpy as np

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # OPD
        motionnet_type,
        voting,
        gtdet,
        inference_matcher,
        gtextrinsic,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        
        # OPD
        self.motionnet_type = motionnet_type
        self.voting = voting
        self.gtdet = gtdet
        self.inference_matcher = inference_matcher
        self.gtextrinsic = gtextrinsic

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # OPD
        mtype_weight = cfg.MODEL.MASK_FORMER.MTYPE_WEIGHT
        morigin_weight = cfg.MODEL.MASK_FORMER.MORIGIN_WEIGHT
        maxis_weight = cfg.MODEL.MASK_FORMER.MAXIS_WEIGHT
        extrinsic_weight = cfg.MODEL.MASK_FORMER.EXTRINSIC_WEIGHT

        motionnet_type = cfg.MODEL.MOTIONNET.TYPE

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        if "GTDET" in cfg.MODEL:
            gtdet = cfg.MODEL.GTDET
        else:
            gtdet = False

        if "GTEXTRINSIC" in cfg.MODEL:
            gtextrinsic = cfg.MODEL.GTEXTRINSIC
        else:
            gtextrinsic = None

        if gtdet or gtextrinsic:
            # This inference matcher is used for GT ablation when inferencing
            inference_matcher = matcher
        else:
            inference_matcher = None


        # OPD
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_mtype": mtype_weight, "loss_morigin": morigin_weight, "loss_maxis": maxis_weight}
        if motionnet_type == "BMOC_V1" or motionnet_type == "BMOC_V2" or motionnet_type == "BMOC_V3" or motionnet_type == "BMOC_V4" or motionnet_type == "BMOC_V5" or motionnet_type == "BMOC_V6":
            weight_dict["loss_extrinsic"] = extrinsic_weight

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # OPD
        if motionnet_type == "BMOC_V0":
            weight_dict["loss_extrinsic"] = extrinsic_weight

        # OPD
        losses = ["labels", "masks", "mtypes", "morigins", "maxises", "extrinsics"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            motionnet_type=motionnet_type,
        )

        # OPD
        if "VOTING" in cfg.MODEL.MOTIONNET:
            voting = cfg.MODEL.MOTIONNET.VOTING
        else:
            voting = None

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # OPD
            "motionnet_type": motionnet_type,
            "voting": voting,
            "gtdet": gtdet,
            "inference_matcher": inference_matcher,
            "gtextrinsic": gtextrinsic,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # Load the targets if it's training or it's in the groundtruth ablation study
        if self.training or self.gtdet or self.gtextrinsic:
            # get the grpundtruth
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    print(f"Warning: {k} is not in loss")
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # OPD
            mask_mtype_results = outputs["pred_mtypes"]
            mask_morigin_results = outputs["pred_morigins"]
            mask_maxis_results = outputs["pred_maxises"]
            if "BMOC" in self.motionnet_type:
                mask_extrinsic_results  = outputs["pred_extrinsics"]

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            if self.gtdet or self.gtextrinsic:
                if self.gtdet:
                    # Make other predictions be bad, so that they will not consider when evaluating
                    mask_pred_results[:, :, :, :] = -30
                    mask_cls_results[:, :, :3] = 0
                    mask_cls_results[:, :, 3] = 15 # weight for softmax
                # Initialize the predicted class and predicted mask to the default value
                if targets[0]["masks"].shape[0] != 0:
                    outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
                    # Retrieve the matching between the outputs of the last layer and the targets
                    indices = self.inference_matcher(outputs_without_aux, targets)
                    def _get_src_permutation_idx(indices):
                        # permute predictions following indices
                        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
                        src_idx = torch.cat([src for (src, _) in indices])
                        return batch_idx, src_idx

                    def _get_tgt_permutation_idx(indices):
                        # permute targets following indices
                        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
                        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
                        return batch_idx, tgt_idx
                    
                    src_idx = _get_src_permutation_idx(indices)
                    tgt_idx = _get_tgt_permutation_idx(indices)
                    if self.gtdet:
                        mask_pred_results[src_idx] = targets[0]["masks"].unsqueeze(0)[tgt_idx].float() * 30
                        mask_pred_results[mask_pred_results == 0] = -30
                        mask_cls_results[src_idx] = F.one_hot(targets[0]["labels"][tgt_idx[1]], num_classes=self.sem_seg_head.num_classes+1).float() * 15
                    if self.gtextrinsic:
                        if self.motionnet_type == "BMOC_V6":
                            gt_extrinsic_raw = targets[0]["gt_extrinsic"][0]
                            gt_extrinsic = torch.cat(
                                [
                                    gt_extrinsic_raw[0:3],
                                    gt_extrinsic_raw[4:7],
                                    gt_extrinsic_raw[8:11],
                                    gt_extrinsic_raw[12:15],
                                ],
                                0,
                            )
                            mask_extrinsic_results[0] = gt_extrinsic
                        else:
                            raise ValueError("Not Implemented")

            del outputs

            if "BMOC" in self.motionnet_type:
                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size, mask_mtype_result, mask_morigin_result, mask_maxis_result, mask_extrinsic_result in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, mask_mtype_results, mask_morigin_results, mask_maxis_results, mask_extrinsic_results
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)
                        # OPD
                        mask_mtype_result = mask_mtype_result.to(mask_pred_result)
                        mask_morigin_result = mask_morigin_result.to(mask_pred_result)
                        mask_maxis_result = mask_maxis_result.to(mask_pred_result)
                        mask_extrinsic_result = mask_extrinsic_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_mtype_result, mask_morigin_result, mask_maxis_result, mask_extrinsic_result)
                        processed_results[-1]["instances"] = instance_r
            else:
                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size, mask_mtype_result, mask_morigin_result, mask_maxis_result in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, mask_mtype_results, mask_morigin_results, mask_maxis_results
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)
                        # OPD
                        mask_mtype_result = mask_mtype_result.to(mask_pred_result)
                        mask_morigin_result = mask_morigin_result.to(mask_pred_result)
                        mask_maxis_result = mask_maxis_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_mtype_result, mask_morigin_result, mask_maxis_result, None)
                        processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            if hasattr(targets_per_image, "gt_masks"):
                # pad gt
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            else:
                padded_masks = torch.tensor([])
            if "BMOC" in self.motionnet_type:
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        # OPD
                        "gt_motion_valids": targets_per_image.gt_motion_valids,
                        "gt_types": targets_per_image.gt_types,
                        "gt_origins": targets_per_image.gt_origins,
                        "gt_axises": targets_per_image.gt_axises,
                        "gt_extrinsic": targets_per_image.gt_extrinsic,
                        "gt_extrinsic_quaternion": targets_per_image.gt_extrinsic_quaternion,
                        "gt_extrinsic_6d": targets_per_image.gt_extrinsic_6d,
                    }
                )
            else:
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        # OPD
                        "gt_motion_valids": targets_per_image.gt_motion_valids,
                        "gt_types": targets_per_image.gt_types,
                        "gt_origins": targets_per_image.gt_origins,
                        "gt_axises": targets_per_image.gt_axises,
                    }
                )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    # Voting algorithms for inference
    def votingProcess(self, x, voting):
        device = x.device
        if voting == "median":
            final = torch.median(x, axis=0)[0]
        elif voting == "mean":
            final = torch.mean(x, axis=0)
        elif voting == "geo-median":
            x = x.detach().cpu().numpy()
            final = geometric_median(x)
            final = torch.from_numpy(final).to(device)
        return final
    
    def convert_to_valid_extrinsic(self, mask_extrinsic, dim=0):
        if dim == 0:
            translation = mask_extrinsic[9:12]
            rotation_mat = quaternion_to_matrix(matrix_to_quaternion(torch.transpose(mask_extrinsic[:9].reshape(3, 3), 0, 1)))
            rotation_vector = torch.flatten(rotation_mat.transpose(0, 1))
            final_mask_extrinsic = torch.cat((rotation_vector, translation))
        elif dim == 1:
            translation = mask_extrinsic[:, 9:12]
            rotation_mat = quaternion_to_matrix(matrix_to_quaternion(torch.transpose(mask_extrinsic[:, :9].reshape(-1, 3, 3), 1, 2)))
            rotation_vector = torch.flatten(rotation_mat.transpose(1, 2), start_dim=1)
            final_mask_extrinsic = torch.cat((rotation_vector, translation), dim=1)
        return final_mask_extrinsic


    def instance_inference(self, mask_cls, mask_pred, mask_mtype, mask_morigin, mask_maxis, mask_extrinsic):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # OPD
        mask_mtype = mask_mtype[topk_indices]
        pred_probs = F.softmax(mask_mtype, dim=1)
        mask_mtype = torch.argmax(pred_probs, 1).float()

        mask_morigin = mask_morigin[topk_indices]
        mask_maxis = mask_maxis[topk_indices]

        if self.motionnet_type == "BMOC_V1":
            mask_extrinsic = mask_extrinsic[topk_indices]
            mask_extrinsic = self.convert_to_valid_extrinsic(mask_extrinsic, dim=1)
            if self.voting != "none":
                final_translation = torch.median(mask_extrinsic[:, 9:12], axis=0)[0]
                quaternions = matrix_to_quaternion(torch.transpose(mask_extrinsic[:, :9].reshape(-1, 3, 3), 1, 2))
                final_quaternion = self.votingProcess(quaternions, self.voting)
                final_rotation = quaternion_to_matrix(final_quaternion)
                final_rotation_vector = torch.flatten(final_rotation.transpose(0, 1))
                mask_extrinsic = torch.cat((final_rotation_vector, final_translation))
        elif self.motionnet_type == "BMOC_V2":
            mask_extrinsic = mask_extrinsic[topk_indices]
            if self.voting != "none":
                final_translation = torch.median(mask_extrinsic[:, 4:7], axis=0)[0]
                final_quaternion = self.votingProcess(mask_extrinsic[:, :4], self.voting)
                final_rotation = quaternion_to_matrix(final_quaternion)
                final_rotation_vector = torch.flatten(final_rotation.transpose(0, 1))
                mask_extrinsic = torch.cat((final_rotation_vector, final_translation))
            elif self.voting == "none":
                translations = mask_extrinsic[:, 4:7]
                quaternions = mask_extrinsic[:, :4]
                rotation_vector = torch.flatten(quaternion_to_matrix(quaternions).transpose(1, 2), 1)
                mask_extrinsic = torch.cat((rotation_vector, translations), 1)
        elif self.motionnet_type == "BMOC_V3":
            mask_extrinsic = mask_extrinsic[topk_indices]
            if self.voting != "none":
                final_translation = torch.median(mask_extrinsic[:, 6:9], axis=0)[0]
                final_6d = self.votingProcess(mask_extrinsic[:, :6], self.voting)
                final_rotation = rotation_6d_to_matrix(final_6d)
                final_rotation_vector = torch.flatten(final_rotation.transpose(0, 1))
                mask_extrinsic = torch.cat((final_rotation_vector, final_translation))
            elif self.voting == "none":
                translations = mask_extrinsic[:, 6:9]
                rotation_6ds = mask_extrinsic[:, :6]
                rotation_vector = torch.flatten(rotation_6d_to_matrix(rotation_6ds).transpose(1, 2), 1)
                mask_extrinsic = torch.cat((rotation_vector, translations), 1)
        elif self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5":
            translation = mask_extrinsic[4:7]
            quaternion = mask_extrinsic[:4]
            rotation_vector = torch.flatten(quaternion_to_matrix(quaternion).transpose(0, 1))
            mask_extrinsic = torch.cat((rotation_vector, translation))
        elif self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMOC_V6":
            mask_extrinsic = self.convert_to_valid_extrinsic(mask_extrinsic, dim=0)

        if "BMOC" in self.motionnet_type:
            # Use the predicted extrinsic matrix to convert the predicted morigin and maxis back to camera coordinate
            maxis_end = mask_morigin + mask_maxis
            mextrinsic_c2w = torch.eye(4, device=mask_morigin.device).repeat(
                        mask_morigin.shape[0], 1, 1
                    )

            if self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMOC_V4"  or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6" or (self.motionnet_type == "BMOC_V1" and self.voting != "none") or (self.motionnet_type == "BMOC_V2" and self.voting != "none")  or (self.motionnet_type == "BMOC_V3" and self.voting != "none"):
                mextrinsic_c2w[:, 0:3, 0:4] = torch.transpose(
                            mask_extrinsic.reshape(4, 3).repeat(mask_morigin.shape[0], 1, 1), 1, 2
                        )
            elif self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2" or self.motionnet_type == "BMOC_V3":
                mextrinsic_c2w[:, 0:3, 0:4] = torch.transpose(
                            mask_extrinsic.reshape(-1, 4, 3), 1, 2
                        )
            mextrinsic_w2c = torch.inverse(mextrinsic_c2w)
            mask_morigin = (
                        torch.matmul(
                            mextrinsic_w2c[:, :3, :3], mask_morigin.unsqueeze(2)
                        ).squeeze(2)
                        + mextrinsic_w2c[:, :3, 3]
                    )
            end_in_cam = (
                        torch.matmul(
                            mextrinsic_w2c[:, :3, :3], maxis_end.unsqueeze(2)
                        ).squeeze(2)
                        + mextrinsic_w2c[:, :3, 3]
                    )
            mask_maxis = end_in_cam - mask_morigin

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        # OPD
        result.mtype = mask_mtype
        result.morigin = mask_morigin
        result.maxis = mask_maxis
        if self.motionnet_type == "BMOC_V0"  or self.motionnet_type == "BMOC_V4"  or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6" or (self.motionnet_type == "BMOC_V1" and self.voting != "none") or (self.motionnet_type == "BMOC_V2" and self.voting != "none") or (self.motionnet_type == "BMOC_V3" and self.voting != "none"):
            result.mextrinsic = mask_extrinsic.repeat(mask_morigin.shape[0], 1)
        elif self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2" or self.motionnet_type == "BMOC_V3":
            result.mextrinsic = mask_extrinsic
        return result
