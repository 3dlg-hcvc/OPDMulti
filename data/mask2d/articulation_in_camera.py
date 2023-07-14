import os
import time
import json
import logging
import numpy as np

from multiscan.utils import io
from core import Mask2DCore, CameraParameter
import pdb

log = logging.getLogger(__name__)

class ArticulationInCamera(Mask2DCore):
    def __init__(self):
        super().__init__()
        self.annotation = {}
        self.additional_transformation = np.eye(4)

    def set_cameras(self, cameras: list):
        self.cameras = cameras

    def set_additional_transformation(self, transformation):
        self.additional_transformation = transformation

    def load_annotations(self, annotations_path):
        self.annotation = io.read_json(annotations_path)

    def transform_obbs(self, transformation):
        objects = self.annotation['objects']
        new_obbs = []
        for obj in objects:
            # pdb.set_trace()
            obb = obj.get('obb')
            if obb is None:
                if obj['label'] != 'remove':
                    log.warning(obj['label'] + ' has no obb')
                continue
            new_obb = {}
            centroid = transformation[:3,:3].dot(obb['centroid']) + transformation[:3,3]
            axes_lengths = obb['axesLengths']
            normalized_axes = transformation[:3,:3].dot(np.reshape(obb['normalizedAxes'], (3,3), order='F'))
            min_bound = transformation[:3,:3].dot(obb['min']) + transformation[:3,3]
            max_bound = transformation[:3,:3].dot(obb['max']) + transformation[:3,3]
            front = transformation[:3,:3].dot(obb['front'])
            up = transformation[:3,:3].dot(obb['up'])
            new_obb['objectId'] = obj['objectId']
            new_obb['centroid'] = centroid.tolist()
            new_obb['axesLengths'] = axes_lengths
            new_obb['normalizedAxes'] = normalized_axes.flatten(order='F').tolist()
            new_obb['min'] = min_bound.tolist()
            new_obb['max'] = max_bound.tolist()
            new_obb['front'] = front.tolist()
            new_obb['up'] = up.tolist()
            new_obbs.append(new_obb)
        return new_obbs

    def transform_articulations(self, transformation):
        parts = self.annotation['parts']
        new_articulations = []
        for part in parts:
            part_id = part['partId']
            articulations = part.get('articulations', [])
            for articulation in articulations:
                origin = articulation['origin']
                axis = articulation['axis']

                new_articulation = articulation.copy()
                new_articulation['origin'] = (transformation[:3, 3] + np.dot(transformation[:3, :3], origin)).tolist()
                new_articulation['axis'] = np.dot(transformation[:3, :3], axis).tolist()
                new_articulation.update({'partId': part_id})

                new_articulations.append(new_articulation)
        return new_articulations

    def export_annotations(self, output_dir):
        for camera in self.cameras:
            intrinsics = camera.intrinsics
            extrinsics = camera.extrinsics
            transformation = np.diag([1, -1, -1, 1]).dot(extrinsics.dot(self.additional_transformation))
            new_articulations = {
                'intrinsics': intrinsics.flatten(order='F').tolist(),
                'extrinsics': np.linalg.inv(transformation).flatten(order='F').tolist(),
                'obbs': self.transform_obbs(transformation),
                'articulations': self.transform_articulations(transformation)
            }
            io.write_json(new_articulations, os.path.join(output_dir, camera.name + '.json'))