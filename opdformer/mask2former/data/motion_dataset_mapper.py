from detectron2.data import *
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

import copy
import logging
import math
import numpy as np
import json
from PIL import Image
import os
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util
import h5py

from ..utils.tranform import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d

MOTION_TYPE = {"rotation": 0, "translation": 1}

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = mask_util.frPyObjects(polygons, height, width)
        mask = mask_util.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

# MotionNet Version: based on DatasetMapper
class MotionDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        # is_train: bool,
        # *,
        # augmentations: List[Union[T.Augmentation, T.Transform]],
        # image_format: str,
        # use_instance_mask: bool = False,
        # use_keypoint: bool = False,
        # instance_mask_format: str = "polygon",
        # keypoint_hflip_indices: Optional[np.ndarray] = None,
        # precomputed_proposal_topk: Optional[int] = None,
        # recompute_boxes: bool = False,
        annotations_to_instances=None,
        network_type=None,
        use_gt=False,
        MODELATTRPATH=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.annotations_to_instances = annotations_to_instances
        self.network_type = network_type
        self.use_gt = use_gt

        self.images = None
        self.filenames = None
        self.filenames_map = {}

        self.depth_images = None
        self.depth_filenames = None
        self.depth_filenames_map = {}

        self.object_poses = None
        self.MODELATTRPATH = MODELATTRPATH

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = []

        if is_train:
            augs.append(T.RandomFlip(0.5))
            augs.append(T.RandomBrightness(0.5, 1.5))
            augs.append(T.RandomContrast(0.5, 1.5))
        recompute_boxes = False
       
        def f(
            annos, shape, mask_format, motion_valid, extrinsic_matrix, model_name, network_type, object_poses
        ):
            return bm_annotations_to_instances(
                annos,
                shape,
                mask_format=mask_format,
                motion_valid=motion_valid,
                extrinsic_matrix=extrinsic_matrix,
                model_name=model_name,
                network_type=network_type,
                object_poses=object_poses
            )

        annotations_to_instances = f

        if "GTDET" in cfg.MODEL:
            gtdet = cfg.MODEL.GTDET
        else:
            gtdet = False

        if "GTEXTRINSIC" in cfg.MODEL:
            gtextrinsic = cfg.MODEL.GTEXTRINSIC
        else:
            gtextrinsic = False

        use_gt = gtdet or gtextrinsic

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "annotations_to_instances": annotations_to_instances,
            "network_type": cfg.MODEL.MOTIONNET.TYPE,
            "use_gt": use_gt,
            "MODELATTRPATH": cfg.MODEL.MODELATTRPATH,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def load_depth_h5(self, base_dir):
        # Load the dataset at the first time
        if self.depth_images == None:
            depth_h5file = h5py.File(f'{base_dir}/depth.h5')
            self.depth_images = depth_h5file['depth_images']
            self.depth_filenames = depth_h5file['depth_filenames']
            num_images = self.depth_filenames.shape[0]
            for i in range(num_images):
                self.depth_filenames_map[self.depth_filenames[i].decode('utf-8')] = i

    def load_h5(self, base_dir, dir):
        if self.images == None:
            h5file = h5py.File(f'{base_dir}/{dir}.h5')
            self.images = h5file[f'{dir}_images']
            self.filenames = h5file[f'{dir}_filenames']
            num_images = self.filenames.shape[0]
            for i in range(num_images):
                self.filenames_map[self.filenames[i].decode('utf-8')] = i

    def load_object_pose(self, path):
        if self.object_poses == None:
            model_attr_file = open(self.MODELATTRPATH)
            model_attr = json.load(model_attr_file)
            model_attr_file.close()
            
            self.object_poses = {}
            for model in model_attr:
                self.object_poses[model] = model_attr[model]["object_pose"]
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # load images from the h5 file
        # Get the Dataset path
        base_dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[0]
        dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[-1]
        file_name = os.path.split(dataset_dict["file_name"])[-1]
        
        if self.image_format == "depth":
            self.load_depth_h5(base_dir)
            image = self.depth_images[self.depth_filenames_map[dataset_dict["depth_file_name"]]]
        elif self.image_format == "RGB":
            self.load_h5(base_dir, dir)
            image = self.images[self.filenames_map[file_name]]
        elif self.image_format == "RGBD":
            self.load_depth_h5(base_dir)
            self.load_h5(base_dir, dir)
            depth_image = self.depth_images[self.depth_filenames_map[dataset_dict["depth_file_name"]]]
            RGB_image = self.images[self.filenames_map[file_name]]
            image = np.concatenate([RGB_image, depth_image], axis=2)

        if self.network_type == "BMOC_V1":
            # Need to read the gt object poses for each part
            self.load_object_pose(self.MODELATTRPATH)

        model_name = dataset_dict["file_name"].split("/")[-1].split("-")[0]

        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        #### MotionNet: apply transform on image and mask/bbx; if there is flip operation, then set motion_valid to false
        motion_valid = False
        if self.is_train:
            # Judge if there is no random flip, then motion annotations are valid
            if isinstance(transforms[0], T.NoOpTransform):
                motion_valid = True
        else:
            # When inferencing, all motions are valid; currently inferencing code doesn't use the attribute
            motion_valid = True
        #####

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        #### MotionNet
        # Only use gt in world coordinate for BMOC. For other cases, no need for extra transformation
        if "extrinsic" in dataset_dict["camera"] and "BMOC" in self.network_type:
            # extrinsic_matrix = np.array(dataset_dict["camera"]["extrinsic"]["matrix"])
            extrinsic_matrix = np.array(dataset_dict["camera"]["extrinsic"])
        else:
            extrinsic_matrix = None

        # All other annotations are in camera coordinate, assume the camera intrinsic parameters are fixed (Just used in final visualization)
        dataset_dict.pop("camera")
        dataset_dict.pop("depth_file_name")
        # dataset_dict.pop("label")
        ####

        if not self.is_train and self.use_gt == False:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            #### MotionNet
            # Convert the annotations into tensor
            # Add motion valid to instance to indicate if the instance will be used for motion loss
            instances = self.annotations_to_instances(
                annos,
                image_shape,
                mask_format=self.instance_mask_format,
                motion_valid=motion_valid,
                extrinsic_matrix=extrinsic_matrix,
                model_name=model_name,
                network_type=self.network_type,
                object_poses=self.object_poses
            )
            ####

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)

            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances

        return dataset_dict


# MotionNet: add motion type, motion origin. motion axis
# For motion valid = False, extrinsic matrix and motion parameters may not be the true gt
# This is to better train the detection and segmentation
def bm_annotations_to_instances(
    annos, image_size, mask_format="polygon", motion_valid=True, extrinsic_matrix=None, model_name=None, network_type=None, object_poses=None
):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [
        BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
        for obj in annos
    ]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    target.gt_model_name = torch.tensor(
            [int(model_name)] * len(annos), dtype=torch.int32
        )

    # Add extra annotations: gt_origins, gt_axises and gt_types
    types = [MOTION_TYPE[obj["motion"]["type"]] for obj in annos]
    types = torch.tensor(types, dtype=torch.float32)
    target.gt_types = types

    if extrinsic_matrix is not None:
        if network_type != "BMOC_V1":
            # We use this version for our scene_pose baseline (We store the scene_pose in the matrix)
            extrinsic_matrix = torch.tensor(
                [extrinsic_matrix] * len(annos), dtype=torch.float32
            )
        else:
            # We use this version for our separate object poses for each part (We store the gt object poses for each part into this variable)
            object_keys = [obj["object_key"] for obj in annos]
            extrinsic_matrix = torch.tensor(np.array([object_poses[object_key] for object_key in object_keys]), dtype=torch.float32)
        
        # Store the whole scene pose or the separate object poses (each part has one annotation)
        # For single scene pose, all parts have the same thing; for the separate object poses, each part may have different stuff
        target.gt_extrinsic = extrinsic_matrix

        if len(annos) == 0:
            transformation = None
            target.gt_extrinsic_quaternion = torch.tensor([])
            target.gt_extrinsic_6d = torch.tensor([])
        else:
            transformation = torch.transpose(extrinsic_matrix.reshape(extrinsic_matrix.size(0), 4, 4), 1, 2)

            gt_translations = extrinsic_matrix[:, 12:15]
            gt_quaternions = matrix_to_quaternion(torch.transpose(torch.cat(
                            [
                                extrinsic_matrix[:, 0:3],
                                extrinsic_matrix[:, 4:7],
                                extrinsic_matrix[:, 8:11],
                            ],
                            1,
                        ).reshape(-1, 3, 3), 1, 2))
            target.gt_extrinsic_quaternion = torch.cat((gt_quaternions, gt_translations), 1)

            gt_6d_rotations = matrix_to_rotation_6d(torch.transpose(torch.cat(
                            [
                                extrinsic_matrix[:, 0:3],
                                extrinsic_matrix[:, 4:7],
                                extrinsic_matrix[:, 8:11],
                            ],
                            1,
                        ).reshape(-1, 3, 3), 1, 2))
            target.gt_extrinsic_6d = torch.cat((gt_6d_rotations, gt_translations), 1)

    else:
        transformation = None

    origins_cam = [obj["motion"]["origin"] for obj in annos]
    if transformation is not None:
        temp_origins_cam = torch.tensor(np.array([origin + [1] for origin in origins_cam]), dtype=torch.float32)
        origins_world = torch.einsum("kij,kj->ki", transformation, temp_origins_cam)
        origins = origins_world[:, 0:3]
    else:
        origins = torch.tensor(origins_cam, dtype=torch.float32)
    target.gt_origins = origins

    axes_cam = [obj["motion"]["axis"] for obj in annos]
    if transformation is not None:
        axes_end_cam = list(np.asarray(axes_cam) + np.asarray(origins_cam))
        temp_axes_end_cam = torch.tensor(np.array([list(axis) + [1] for axis in axes_end_cam]), dtype=torch.float32)
        axes_end_world = torch.einsum("kij,kj->ki", transformation, temp_axes_end_cam)
        axes = axes_end_world[:, 0:3] - origins
    else:
        axes = torch.tensor(axes_cam, dtype=torch.float32)
    target.gt_axises = axes

    motion_valids = [motion_valid] * len(annos)
    target.gt_motion_valids = torch.tensor(motion_valids)

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            # TODO check type and provide better error
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert (
                        segm.ndim == 2
                    ), "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target
