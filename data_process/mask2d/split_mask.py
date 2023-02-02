import glob
import os
import json
import multiprocessing
from time import time
import pdb
import hydra
from omegaconf import DictConfig
from multiscan.utils import io
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial

def get_frame_id(cfg, scan_id):
    file_path = f"{cfg.input_dir}/{scan_id}/depth/*"
    paths = glob.glob(file_path)
    frame_ids = [tmp.split('/')[-1].split(".")[0] for tmp in paths]
    return frame_ids


def split_mask(mask, articulations, obj_part_info):
    masks = []
    cat_ids = ["drawer", "door", "lid", "cover", "covert", "lid_cover"]
    part_channel = mask[:, :, 1]
    for articulation in articulations:
        part_id = articulation["partId"]
        part_info = obj_part_info.loc[obj_part_info["partID"] == part_id]
        cat_id = part_info["part_label"].values[0].split(".")[0]
        if cat_id in cat_ids:
            idx = np.where(part_channel == part_id)
            tmp_mask = np.zeros(part_channel.shape)
            tmp_mask[idx] = 255
            tmp = {
                "part_id": part_id,
                "mask": tmp_mask.astype(np.uint8)
            }
            masks.append(tmp)
    return masks


def update_annotation(annotation, obj_part_info):
    cat_ids = ["drawer", "door", "lid", "cover", "covert","lid_cover"]
    new_annotation = {
        "intrinsics": annotation["intrinsics"],
        "extrinsics": annotation["extrinsics"],
        "diagonal": annotation["diagonal"]
    }
    arti = []
    for articulation in annotation["articulation"]:
        part_info = obj_part_info.loc[obj_part_info["partID"]
                                      == articulation["partId"]]
        cat_id = part_info["part_label"].values[0].split(".")[0]
        if cat_id in cat_ids:
            if cat_id in ["cover", "lid_cover", "covert", "lid"]:
                cat_id = "lid"
            articulation["part_label"] = cat_id
            arti.append(articulation)

    new_annotation["articulation"] = arti

    return new_annotation


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

    return obj_part_info


def copy_file(input_path, output_path, frame_id, scan_id, obj_part_info):
    try:
        file_path = glob.glob(f"{input_path}/{frame_id}.*")[0]
    except:
        return
    if input_path.split("/")[-1] == "mask":
        output_dir = f"{output_path}/mask"
        io.ensure_dir_exists(output_dir)
        annotation = io.read_json(
            f"{input_path[:-5]}/annotation_update/{frame_id}.json")
        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if annotation == None:
            masks = []
        else:
            masks = split_mask(mask, annotation["articulation"], obj_part_info)
        if len(masks) == 0:
            mask = np.zeros(mask[:, :, 1].shape)
            cv2.imwrite(f"{output_dir}/{scan_id}_{frame_id}_0.png", mask)
        else:
            for tmp_mask in masks:
                part_id = tmp_mask["part_id"]
                # pdb.set_trace()
                Image.fromarray(tmp_mask["mask"]).save(
                    f"{output_dir}/{scan_id}_{frame_id}_{part_id}.png")
    if input_path.split("/")[-1] == "rgb":
        output_dir = f"{output_path}/rgb"
        io.ensure_dir_exists(output_dir)
        os.system(f"cp {file_path} {output_dir}/{scan_id}_{frame_id}.png")
    if input_path.split("/")[-1] == "depth":
        output_dir = f"{output_path}/depth"
        io.ensure_dir_exists(output_dir)
        os.system(f"cp {file_path} {output_dir}/{scan_id}_{frame_id}_d.png")
    if input_path.split("/")[-1] == "annotation_update":
        output_dir = f"{output_path}/annotation"
        io.ensure_dir_exists(output_dir)
        annotation = io.read_json(file_path)
        annotation = update_annotation(annotation, obj_part_info)
        io.write_json(annotation, f"{output_dir}/{scan_id}_{frame_id}.json")


def process(scan_id, cfg):
    frame_ids = get_frame_id(cfg, scan_id)
    mask_path = f"{cfg.input_dir}/{scan_id}/mask"
    depth_path = f"{cfg.input_dir}/{scan_id}/depth"
    annotation_path = f"{cfg.input_dir}/{scan_id}/annotation_update"
    rgb_path = f"{cfg.input_dir}/{scan_id}/rgb"
    origin_annotation_path = f'{cfg.origin_data_dir}/{scan_id}/{scan_id}.annotations.json'
    origin_annotation = io.read_json(origin_annotation_path)
    obj_part_info = get_obj_part_info(origin_annotation)
    
    io.ensure_dir_exists(cfg.output_initial_data_dir)
    for frame_id in frame_ids:
        copy_file(rgb_path, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(depth_path, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(annotation_path, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(mask_path, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):
    scan_ids = io.get_folder_list(cfg.input_dir, join_path=False)
    debug = False
    if debug:
        scan_id = "scene_00021_00"
        process(scan_id, cfg)
    process_map(partial(process, cfg = cfg), scan_ids, chunksize=1)


if __name__ == "__main__":
    main()
