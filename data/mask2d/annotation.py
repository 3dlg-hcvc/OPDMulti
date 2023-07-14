import os
from re import T
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import cv2
from plyfile import PlyData
import trimesh
import utils

import hydra
from omegaconf import DictConfig

from multiscan.utils import io, memory_limit, NoDaemonPool
import pdb
from dataclasses import dataclass
from time import time
import logging

"""
Modify annotation to get a final version following the previous format
read annotation from both processed articulation annotation and original annotation
read projected mask
Add diameter of diagonal length
Add camera intrinsic and extrinsic
"""


@dataclass
class MeshInfo:
    mesh: trimesh.Trimesh
    object_ids: np.ndarray
    part_ids: np.ndarray
    obj_verts_id: np.ndarray
    part_verts_id: np.ndarray


def get_bound(ply_path):
    plydata = PlyData.read(ply_path)
    # pdb.set_trace()
    x = np.asarray(plydata['vertex']['x'])
    y = np.asarray(plydata['vertex']['y'])
    z = np.asarray(plydata['vertex']['z'])
    # nx = np.asarray(plydata['vertex']['nx'])
    # ny = np.asarray(plydata['vertex']['ny'])
    # nz = np.asarray(plydata['vertex']['nz'])
    vertices = np.column_stack((x, y, z))

    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])

    min_bound = [x_min, y_min, z_min]
    max_bound = [x_max, y_max, z_max]

    diameter = np.linalg.norm(np.asarray(min_bound) - np.asarray(max_bound))

    return diameter

def update_articulation(mask, scan_id, frame_id, annotations, obj_part_info):
    articulation = annotations["articulations"]
    obbs = annotations["obbs"]
    part_channel = mask[:, :, 1]
    part_ids = np.unique(part_channel)
    new_arti_anno = []
    
    obb_obj_idx = {}
    for idx, obb in enumerate(obbs):
        obb_obj_idx.update({obb["objectId"]: idx})
    # pdb.set_trace()

    
    for index, part in enumerate(articulation):
        if part["partId"] in part_ids:
            objectID = int(obj_part_info.loc[obj_part_info["partID"] == part["partId"]]["objectID"].values)
            # obb_idx = obb_obj_idx[objectID]
            # object_pose = get_pose_from_obb(obbs[obb_idx])

            idx = np.where(part_channel == part["partId"])
            xmin = min(idx[0])
            ymin = min(idx[1])
            xmax = max(idx[0])
            ymax = max(idx[1])

            width = xmax - xmin
            height = ymax - ymin

            bbox = np.asarray([xmin, ymin, width, height]).astype(np.float64)
            articulation[index]["bbox"] = bbox.tolist()
            part_pixel_num = len(idx[0])
            articulation[index]["pixel_num"] = np.float64(part_pixel_num)
            # articulation[index]["object_pose"] = object_pose.flatten(order='F').tolist()
            articulation[index]["object_key"] = f"{scan_id}_{frame_id}_{objectID}"

            new_arti_anno.append(articulation[index])

    return new_arti_anno


def get_frame_id(cfg, scan_id):
    file_path = f"{cfg.input_dir}/{scan_id}/annotation/*"
    paths = glob.glob(file_path)
    frame_ids = [tmp.split('/')[-1].split(".")[0] for tmp in paths]
    return frame_ids


def get_obj_part_info(origin_annotation):
    obj_part_info = pd.DataFrame({"objectID": [], "partID": [], "object_label": [
    ], "part_label": [], "mobility_type": []})
    # pdb.set_trace()
    for object in origin_annotation["objects"]:
        obj_id = object["objectId"]
        obj_label = object["label"]
        mobility_type = object["mobilityType"]
        for part_id in object["partIds"]:
            part_label = origin_annotation["parts"][part_id-1]["label"]
            tmp = pd.DataFrame({"objectID": [obj_id], "partID": [part_id], "object_label": [obj_label],
                                "part_label": [part_label], "mobility_type": [mobility_type]})
            obj_part_info = obj_part_info.append(tmp, ignore_index=True)
            # pdb.set_trace()
        # pdb.set_trace()

    return obj_part_info


# TODO: Add mask information and object vertex information

def get_mesh_info(ply_path, alignment):
    plydata = PlyData.read(ply_path)
    x = np.asarray(plydata['vertex']['x'])
    y = np.asarray(plydata['vertex']['y'])
    z = np.asarray(plydata['vertex']['z'])
    vertices = np.column_stack((x, y, z))
    # vertex_normals = np.column_stack((nx, ny, nz))
    triangles = np.vstack(plydata['face'].data['vertex_indices'])
    alignment = np.reshape(
        alignment['coordinate_transform'], (4, 4), order='F')
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles, process=False)
    mesh.apply_transform(alignment)
    object_ids = plydata['face'].data['objectId']
    part_ids = plydata['face'].data['partId']

    # get objectID for each vertex
    obj_verts_id = np.zeros(len(vertices))
    for idx, triangle in enumerate(triangles):
        obj_verts_id[triangle[0]] = obj_verts_id[triangle[1]
                                                 ] = obj_verts_id[triangle[2]] = object_ids[idx]

    # get partID for each vertex
    part_verts_id = np.zeros(len(vertices))
    for idx, triangle in enumerate(triangles):
        part_verts_id[triangle[0]] = part_verts_id[triangle[1]
                                                   ] = part_verts_id[triangle[2]] = part_ids[idx]

    mesh_info = MeshInfo(mesh, object_ids, part_ids,
                         obj_verts_id, part_verts_id)

    return mesh_info


def get_obj_frame_info(cfg, mesh_info, annotations, mask):
    # read plydata and project vertices to image plane
    intrinsics = np.reshape(annotations["intrinsics"], (3, 3), order='F')
    extrinsics = np.reshape(annotations["extrinsics"], (4, 4), order='F')
    mesh = mesh_info.mesh.copy()
    mesh = mesh.apply_transform(extrinsics)
    vertices = mesh.vertices
    # pdb.set_trace()
    v_pixels = np.dot(vertices, intrinsics.T)
    v_pixels /= v_pixels[:, -1].reshape(-1, 1)

    # count vertex in the frame for different object
    obj_frame_info = []
    pixel_x_in_frame = (0 < v_pixels[:, 0]) & (
        v_pixels[:, 0] < cfg.image_width)
    pixel_y_in_frame = (0 < v_pixels[:, 1]) & (
        v_pixels[:, 1] < cfg.image_height)
    obj_in_frame = pixel_x_in_frame & pixel_y_in_frame

    obj_ids = np.unique(mesh_info.object_ids)
    for obj_id in obj_ids:
        if obj_id != 0:
            verts_idx = np.where(mesh_info.obj_verts_id == obj_id)
            valid_obj_verts_num = np.sum(obj_in_frame[verts_idx])
            pixel_idx = np.where(mask[:, :, 0] == obj_id)
            tmp = {
                "object_id": int(obj_id),
                "pixel_count": np.float64(len(pixel_idx[0])),
                "total_vertex_count": np.float64(len(verts_idx[0])),
                "vertex_count": np.float64(valid_obj_verts_num),
            }
            # pdb.set_trace()
            obj_frame_info.append(tmp)
    # pdb.set_trace()
    return obj_frame_info


def get_part_frame_info(cfg, mesh_info, annotations, mask):
    # read plydata and project vertices to image plane
    intrinsics = np.reshape(annotations["intrinsics"], (3, 3), order='F')
    extrinsics = np.reshape(annotations["extrinsics"], (4, 4), order='F')
    mesh = mesh_info.mesh.copy()
    mesh = mesh.apply_transform(extrinsics)
    vertices = mesh.vertices
    # pdb.set_trace()
    v_pixels = np.dot(vertices, intrinsics.T)
    v_pixels /= v_pixels[:, -1].reshape(-1, 1)

    # count vertex in the frame for different object
    part_frame_info = []
    pixel_x_in_frame = (0 < v_pixels[:, 0]) & (
        v_pixels[:, 0] < cfg.image_width)
    pixel_y_in_frame = (0 < v_pixels[:, 1]) & (
        v_pixels[:, 1] < cfg.image_height)
    part_in_frame = pixel_x_in_frame & pixel_y_in_frame

    part_ids = np.unique(mesh_info.part_ids)
    for part_id in part_ids:
        if part_id != 0:
            verts_idx = np.where(mesh_info.part_verts_id == part_id)
            valid_part_verts_num = np.sum(part_in_frame[verts_idx])
            pixel_idx = np.where(mask[:, :, 1] == part_id)
            tmp = {
                "part_id": int(part_id),
                "pixel_count": np.float64(len(pixel_idx[0])),
                "total_vertex_count": np.float64(len(verts_idx[0])),
                "vertex_count": np.float64(valid_part_verts_num),
            }
            # pdb.set_trace()
            part_frame_info.append(tmp)

    return part_frame_info


def get_pose_from_obb(obb):
    if obb.get('front') is None:
        return np.eye(4)
    front = np.asarray(obb['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(obb['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)
    orientation = np.eye(4)
    orientation[:3, :3] = np.stack([right, up, front], axis=0)
    translation = np.eye(4)
    translation[:3, 3] = -np.asarray(obb['centroid'])
    return np.dot(orientation, translation)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):
    log = logging.getLogger(__name__)
    total_duration = 0

    scan_ids = io.get_folder_list(cfg.input_dir, join_path=False)
    for scan_id in tqdm(scan_ids):
        start = time()
        frame_ids = get_frame_id(cfg, scan_id)
        origin_annotation_path = f'{cfg.origin_data_dir}/{scan_id}/{scan_id}.annotations.json'
        alignment_path = f'{cfg.origin_data_dir}/{scan_id}/{scan_id}.align.json'
        ply_path = f'/project/3dlg-hcvc/multiscan/anonymous_data/{scan_id}/{scan_id}.ply'
        origin_annotation = io.read_json(origin_annotation_path)
        alignment = io.read_json(alignment_path)
        obj_part_info = get_obj_part_info(origin_annotation)
        mesh_info = get_mesh_info(ply_path, alignment)
        diagonal = get_bound(ply_path)

        # calculate scan diameter

        for frame_id in frame_ids:
            articulation_path = f"{cfg.input_dir}/{scan_id}/annotation/{frame_id}.json"
            mask_path = f"{cfg.input_dir}/{scan_id}/mask/{frame_id}.png"
            annotations = io.read_json(articulation_path)
            # articulation = annotations['articulations']
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            new_articulation = update_articulation(mask, scan_id, frame_id, annotations, obj_part_info)
            obj_frame_info = get_obj_frame_info(
                cfg, mesh_info, annotations, mask)
            part_frame_info = get_part_frame_info(
                cfg, mesh_info, annotations, mask)
            # pdb.set_trace()
            # TODO: add scan obb from yongsen side
            annotation = {
                "intrinsics": annotations["intrinsics"],
                "extrinsics": annotations["extrinsics"],
                "diagonal": diagonal
            }
            annotation["articulation"] = new_articulation
            frame_info = {
                "object_info": obj_frame_info,
                "part_info": part_frame_info
            }
            anno_output_dir = cfg.output_annotation_update_dir.format(
                scanId=scan_id)
            frame_info_output_dir = cfg.output_frame_info_dir.format(
                scanId=scan_id)
            # pdb.set_trace()
            io.ensure_dir_exists(anno_output_dir)
            io.ensure_dir_exists(frame_info_output_dir)
            annotation_path = f"{anno_output_dir}/{frame_id}.json"
            frame_info_path = f"{frame_info_output_dir}/{frame_id}.json"
            io.write_json(annotation, annotation_path)
            io.write_json(frame_info, frame_info_path)

        log.info(
            f'Duration update scan {scan_id} annotation: {utils.duration_in_hours(time() - start)}')
        total_duration += time() - start

    log.info(
        f'Average duration update scan annotation: {utils.duration_in_hours(total_duration / len(scan_ids))}')

    return


if __name__ == '__main__':
    main()
