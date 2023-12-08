import os
import glob
import cv2
import multiprocessing

import hydra
import numpy as np
from omegaconf import DictConfig

from time import time
from functools import partial
import io
from tqdm.contrib.concurrent import process_map

from utils import *


def copy_file(input_path, output_path, frame_id, scan_id, obj_part_info):
    try:
        file_path = glob.glob(f"{input_path}/{frame_id}.*")[0]
    except:
        return
    if input_path.split("/")[-1] == "mask":
        output_dir = f"{output_path}/mask"
        ensure_dir_exists(output_dir)
        annotation = read_json(
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
                Image.fromarray(tmp_mask["mask"]).save(
                    f"{output_dir}/{scan_id}_{frame_id}_{part_id}.png")
    if input_path.split("/")[-1] == "rgb":
        output_dir = f"{output_path}/rgb"
        ensure_dir_exists(output_dir)
        os.system(f"cp {file_path} {output_dir}/{scan_id}_{frame_id}.png")
    if input_path.split("/")[-1] == "depth":
        output_dir = f"{output_path}/depth"
        ensure_dir_exists(output_dir)
        os.system(f"cp {file_path} {output_dir}/{scan_id}_{frame_id}_d.png")
    if input_path.split("/")[-1] == "annotation_update":
        output_dir = f"{output_path}/annotation"
        ensure_dir_exists(output_dir)
        annotation = read_json(file_path)
        annotation = update_annotation(annotation, obj_part_info)
        write_json(annotation, f"{output_dir}/{scan_id}_{frame_id}.json")


def process(scan_id, cfg, threshold):
    frame_ids = get_frame_id(cfg, scan_id)
    origin_annotation_path = f'{cfg.origin_data_dir}/{scan_id}/{scan_id}.annotations.json'
    ply_path = f'/project/3dlg-hcvc/multiscan/anonymous_data2/{scan_id}/{scan_id}.ply'
    origin_annotation = read_json(origin_annotation_path)
    obj_part_info = get_obj_part_info(origin_annotation)
    diagonal = get_bound(ply_path)

    mask_dir = f"{cfg.input_dir}/{scan_id}/mask"
    depth_dir = f"{cfg.input_dir}/{scan_id}/depth"
    annotation_dir = f"{cfg.input_dir}/{scan_id}/annotation_update"
    rgb_dir = f"{cfg.input_dir}/{scan_id}/rgb"

    ensure_dir_exists(cfg.output_initial_data_dir)

    for frame_id in frame_ids:
        articulation_path = f"{cfg.input_dir}/{scan_id}/annotation/{frame_id}.json"
        mask_path = f"{cfg.input_dir}/{scan_id}/mask/{frame_id}.png"
        annotations = read_json(articulation_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        new_articulation = update_articulation(
            mask, threshold, scan_id, frame_id, annotations, obj_part_info)
        annotation = {
            "intrinsics": annotations["intrinsics"],
            "extrinsics": annotations["extrinsics"],
            "diagonal": diagonal
        }
        annotation["articulation"] = new_articulation
        anno_output_dir = cfg.annotation_update_dir.format(
            scanId=scan_id)
        ensure_dir_exists(anno_output_dir)
        annotation_path = f"{anno_output_dir}/{frame_id}.json"
        write_json(annotation, annotation_path)

        copy_file(mask_dir, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(depth_dir, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(annotation_dir, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)
        copy_file(rgb_dir, cfg.output_initial_data_dir,
                  frame_id, scan_id, obj_part_info)


def split_files(scan_id_dir, data_dir, output_dir):
    TESTIDSPATH = f'{scan_id_dir}/test_scanids.json'
    VALIDIDPATH = f'{scan_id_dir}/val_scanids.json'
    TRAINIDPATH = f'{scan_id_dir}/train_scanids.json'
    RAWDATAPATH = data_dir
    OUTPUTDATAPATH = output_dir

    start = time()
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    train_ids_file = open(TRAINIDPATH)
    train_ids = json.load(train_ids_file)
    train_ids_file.close()

    dir_names = ['rgb/', 'mask/', 'depth/', 'annotation/']

    output_dir = OUTPUTDATAPATH + "/all/"
    for dir_name in dir_names:
        existDir(output_dir + 'train/' + dir_name)
        existDir(output_dir + 'valid/' + dir_name)
        existDir(output_dir + 'test/' + dir_name)

    anno_paths = glob.glob(RAWDATAPATH + '/annotation/*')
    types = ["annotation", "rgb", "depth", "mask"]

    process_map(partial(split, types=types, test_ids=test_ids, valid_ids=valid_ids, train_ids=train_ids, output_dir=output_dir), anno_paths, chunksize=1)

    stop = time()
    print("Time to split the files into different splits" +str(stop - start) + " seconds")


def process_to_h5(input_dir, output_dir):
    start = time()
    pool = multiprocessing.Pool(processes=16)

    # Create the annotations dir
    existDir(f'{output_dir}/annotations')
    # Move the annotations
    file_paths = glob.glob(f'{input_dir}/annotations/*')
    for file_path in file_paths:
        print('Copying the coco annotations')
        file_name = file_path.split('/')[-1]
        new_file_path = f'{output_dir}/annotations/{file_name}'
        os.system(f'cp {file_path} {new_file_path}')

    # Convert the images in to h5 for train/test/valid/depth dir
    splits = ['train', 'test', 'valid', 'depth']
    for split in splits:
        print(f'Processing {split}')
        pool.apply_async(convert_h5, (split, input_dir, output_dir))
    
    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Start preprocess the extracted multiscan data to update annotatoin and split the part mask to single files")
    threshold = cfg.threshold * cfg.image_width * cfg.image_height
    
    scan_ids = get_folder_list(cfg.input_dir, join_path=False)

    # debug=True

    # if debug == True:
    #     scan_id = "scene_00076_01"
    #     process(scan_id, cfg, threshold)
    process_map(partial(process, threshold=threshold, cfg = cfg), scan_ids, chunksize=1)

    print("Start to post-process the data and change naming format")
    ensure_dir_exists(cfg.data_statistics_dir)
    motion_anno_path = cfg.output_initial_data_dir + "/annotation"
    real_attr_path = cfg.data_statistics_dir + "/real-attr.json"
    real_name_map_path = cfg.data_statistics_dir + "/real_name.json"

    data_dir = cfg.output_initial_data_dir
    datasets = ["all"]
    scan_split_dir = cfg.data_statistics_dir

    motion_real_diagonal(motion_anno_path, real_attr_path, real_name_map_path)
    motion_real_statistics(data_dir, real_name_map_path, datasets, scan_split_dir)

    print("Split files into different splits: train/val/test")
    split_output_dir = cfg.output_split_data_dir
    split_files(scan_split_dir, data_dir, split_output_dir)

    print("Convert the data to COCO format")
    convert_coco(split_output_dir)

    print("Save the processed data into MotionDataset")
    final_process(split_output_dir, cfg.output_dir_final)

    print("Conver the data into h5 format")
    process_to_h5(cfg.output_dir_final, cfg.output_dir_final_h5)


    return


if __name__ == "__main__":
    main()