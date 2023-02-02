# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils.tranform import matrix_to_quaternion, quaternion_to_matrix

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def convert_to_filled_tensor(tensor_list):
    max_size = _max_by_axis([list(tensor.shape) for tensor in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    filled_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    for old, new in zip(tensor_list, filled_tensor):
        new[:old.shape[0]] = old
    return filled_tensor

def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
    ::
                      | 0.5 * x ** 2 / beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:
    ::
                    | 0.5 * x ** 2                  if abs(x) < beta
         huber(x) = |
                    | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, motionnet_type):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # OPD
        self.motionnet_type = motionnet_type

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    # OPD
    def loss_mtypes(self, outputs, targets, indices, num_masks):
        assert "pred_mtypes" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
        src_mtypes = outputs["pred_mtypes"][src_idx][target_motion_valid]
        target_mtypes = convert_to_filled_tensor([t["gt_types"] for t in targets])[tgt_idx][target_motion_valid]

        if src_mtypes.shape[0] == 0:
            return {"loss_mtype": 0.0 * src_mtypes.sum()}

        loss_mtype = F.cross_entropy(src_mtypes, target_mtypes.long(), reduction="sum") / num_masks
        losses = {"loss_mtype": loss_mtype}
        return losses

    def loss_morigins(self, outputs, targets, indices, num_masks):
        assert "pred_morigins" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
        # Only calculate origin loss for the rotation axis
        target_mtypes = convert_to_filled_tensor([t["gt_types"] for t in targets])[tgt_idx][target_motion_valid]
        rot_inds = (
                (target_mtypes == 0).nonzero().unbind(1)[0]
            )
        src_morigins = outputs["pred_morigins"][src_idx][target_motion_valid][rot_inds]
        target_morigins = convert_to_filled_tensor([t["gt_origins"] for t in targets])[tgt_idx][target_motion_valid][rot_inds]

        if src_morigins.shape[0] == 0:
            return {"loss_morigin": 0.0 * src_morigins.sum()}

        loss_morigin = smooth_l1_loss(src_morigins, target_morigins, 1.0, reduction="sum") / num_masks
        losses = {"loss_morigin": loss_morigin}
        return losses

    def loss_maxises(self, outputs, targets, indices, num_masks):
        assert "pred_maxises" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
        src_maxises = outputs["pred_maxises"][src_idx][target_motion_valid]
        target_maxises = convert_to_filled_tensor([t["gt_axises"] for t in targets])[tgt_idx][target_motion_valid]

        if src_maxises.shape[0] == 0:
            return {"loss_maxis": 0.0 * src_maxises.sum()}

        loss_maxis = smooth_l1_loss(src_maxises, target_maxises, 1.0, reduction="sum") / num_masks
        losses = {"loss_maxis": loss_maxis}
        return losses

    def loss_extrinsics(self, outputs, targets, indices, num_masks):
        assert "pred_extrinsics" in outputs
        if self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMOC_V6":
            target_motion_valid = torch.tensor([t["gt_motion_valids"][0] for t in targets], device=outputs["pred_extrinsics"].device)
            src_extrinsics = outputs["pred_extrinsics"][target_motion_valid]
            target_extrinsics_full = [t["gt_extrinsic"][0] for t in targets]
            target_extrinsics = convert_to_filled_tensor([torch.cat(
                                [
                                    extrinsic[0:3],
                                    extrinsic[4:7],
                                    extrinsic[8:11],
                                    extrinsic[12:15],
                                ],
                                0,
                            ) for extrinsic in target_extrinsics_full])[target_motion_valid]
            if src_extrinsics.shape[0] == 0:
                return {"loss_extrinsic": 0.0 * src_extrinsics.sum()}

            # Much proper to make sure each valid image gives the same contribution to the loss
            # Therefore, here use the number of images to average
            loss_extrinsic = smooth_l1_loss(src_extrinsics, target_extrinsics, 1.0, reduction="sum") / outputs["pred_extrinsics"].shape[0]
        elif self.motionnet_type == "BMOC_V1":
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)

            target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
            src_extrinsics = outputs["pred_extrinsics"][src_idx][target_motion_valid]
            target_extrinsics_full = []
            for t in targets:
                extrinsics = t["gt_extrinsic"]
                target_extrinsics_full.append(torch.cat(
                                [
                                    extrinsics[:, 0:3],
                                    extrinsics[:, 4:7],
                                    extrinsics[:, 8:11],
                                    extrinsics[:, 12:15],
                                ],
                                1,
                            ))

            target_extrinsics = convert_to_filled_tensor(target_extrinsics_full)[tgt_idx][target_motion_valid]
            if src_extrinsics.shape[0] == 0:
                return {"loss_extrinsic": 0.0 * src_extrinsics.sum()}

            # Much proper to make sure each valid image gives the same contribution to the loss
            # Therefore, here use the number of images to average
            loss_extrinsic = smooth_l1_loss(src_extrinsics, target_extrinsics, 1.0, reduction="sum") / num_masks
        elif self.motionnet_type == "BMOC_V2":
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)

            target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
            src_extrinsics = outputs["pred_extrinsics"][src_idx][target_motion_valid]
            target_extrinsics = convert_to_filled_tensor([t["gt_extrinsic_quaternion"] for t in targets])[tgt_idx][target_motion_valid]

            if src_extrinsics.shape[0] == 0:
                return {"loss_extrinsic": 0.0 * src_extrinsics.sum()}

            # Much proper to make sure each valid image gives the same contribution to the loss
            # Therefore, here use the number of images to average
            loss_extrinsic = smooth_l1_loss(src_extrinsics, target_extrinsics, 1.0, reduction="sum") / num_masks
        elif self.motionnet_type == "BMOC_V3":
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)

            target_motion_valid = convert_to_filled_tensor([t["gt_motion_valids"] for t in targets])[tgt_idx]
            src_extrinsics = outputs["pred_extrinsics"][src_idx][target_motion_valid]
            target_extrinsics = convert_to_filled_tensor([t["gt_extrinsic_6d"] for t in targets])[tgt_idx][target_motion_valid]

            if src_extrinsics.shape[0] == 0:
                return {"loss_extrinsic": 0.0 * src_extrinsics.sum()}

            # Much proper to make sure each valid image gives the same contribution to the loss
            # Therefore, here use the number of images to average
            loss_extrinsic = smooth_l1_loss(src_extrinsics, target_extrinsics, 1.0, reduction="sum") / num_masks
        elif self.motionnet_type == "BMOC_V4"  or self.motionnet_type == "BMOC_V5":
            target_motion_valid = torch.tensor([t["gt_motion_valids"][0] for t in targets], device=outputs["pred_extrinsics"].device)
            src_extrinsics = outputs["pred_extrinsics"][target_motion_valid]
            target_extrinsics = convert_to_filled_tensor([t["gt_extrinsic_quaternion"][0] for t in targets])[target_motion_valid]

            if src_extrinsics.shape[0] == 0:
                return {"loss_extrinsic": 0.0 * src_extrinsics.sum()}

            # Much proper to make sure each valid image gives the same contribution to the loss
            # Therefore, here use the number of images to average
            loss_extrinsic = smooth_l1_loss(src_extrinsics, target_extrinsics, 1.0, reduction="sum") / outputs["pred_extrinsics"].shape[0]
        
        return {"loss_extrinsic": loss_extrinsic}
            

    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            # OPD
            "mtypes": self.loss_mtypes,
            "morigins": self.loss_morigins,
            "maxises": self.loss_maxises,
            "extrinsics": self.loss_extrinsics,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "extrinsics" and self.motionnet_type == "BMCC":
                    continue
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "extrinsics" and (self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMCC"):
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
