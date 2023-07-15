import glob
import os
import json
import multiprocessing
from time import time
import pdb
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

def get_parser():
    parser = argparse.ArgumentParser(description="Motion_real_diagonal")
    parser.add_argument(
        "--scan_id_dir",
        default=f"../mask2d/output/data_statistics/scan_list",
        metavar="DIR",
        help="directory of the train/val/test renamed scan id",
    )
    parser.add_argument(
        "--data_dir",
        default=f"../mask2d/output/opdmulti_V3_processed/",
        metavar="DIR",
        help="directory of the processed dataset",
    )
    parser.add_argument(
        "--output_dir",
        default=f"../mask2d/output/opdmulti_V3_output_split/",
        metavar="DIR",
        help="output directory of the splited processed dataset",
    )

    return parser

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def process(model_path, train_test, output_dir):
    # dir_names = ['origin', 'mask', 'depth', 'origin_annotation']

    # for dir_name in dir_names:
    #     file_paths = glob.glob(f'{model_path}/{dir_name}/*')
    #     for file_path in file_paths:
    #         # pdb.set_trace()
    #         file_name = file_path.split('/')[-1]
    #         process_path = OUTPUTDATAPATH + train_test + '/' + dir_name + '/' + file_name
    #         os.system(f'cp {file_path} {process_path}'

    file_path = model_path
    process_path = output_dir + train_test + '/' + \
        file_path.split('/')[-2] + '/' + file_path.split('/')[-1]
    # print("=========================")
    # pdb.set_trace()
    os.system(f'cp {file_path} {process_path}')

def save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir):
    if model_id in test_ids:
        process(model_path, 'test', output_dir)
        # pool.apply_async(process, (model_path, 'test', output_dir))
    elif model_id in valid_ids:
        process(model_path, 'valid', output_dir)
        # pool.apply_async(process, (model_path, 'valid', output_dir))
    elif model_id in train_ids:
        process(model_path, 'train', output_dir)
        # pool.apply_async(process, (model_path, 'train', output_dir))


def split(anno_path, types, test_ids, valid_ids, train_ids):
    model_id = anno_path.split('/')[-1].split('.')[0].split("_")[0]
    anno_file = open(anno_path)
    anno = json.load(anno_file)
    anno_file.close()

    output_dir = OUTPUTDATAPATH + "all/"
    for t in types:
            temp_path = anno_path.replace("annotation", t)
            tmp = temp_path.split(".")[0]
            paths = glob.glob(f"{tmp}.*") + glob.glob(f"{tmp}_*")
            for model_path in paths:
                save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir)

    if anno["articulation"] == []:
        output_dir = OUTPUTDATAPATH + "no_gt/"
        for t in types:
            temp_path = anno_path.replace("annotation", t)
            tmp = temp_path.split(".")[0]
            paths = glob.glob(f"{tmp}.*") + glob.glob(f"{tmp}_*")
            for model_path in paths:
                save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir)
    elif len(anno["articulation"]) > 0:
        multi_obj = False
        key = anno["articulation"][0]["object_key"]
        if len(anno["articulation"]) > 1:
            for annotation in anno["articulation"][1:]:
                if annotation["object_key"] != key:
                    multi_obj = True
                    break
        if multi_obj:
            output_dir = OUTPUTDATAPATH + "multi_obj/"
            for t in types:
                temp_path = anno_path.replace("annotation", t)
                tmp = temp_path.split(".")[0]
                paths = glob.glob(f"{tmp}.*") + glob.glob(f"{tmp}_*")
                for model_path in paths:
                    save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir)
        else:
            output_dir = OUTPUTDATAPATH + "single_obj/"
            for t in types:
                temp_path = anno_path.replace("annotation", t)
                tmp = temp_path.split(".")[0]
                paths = glob.glob(f"{tmp}.*") + glob.glob(f"{tmp}_*")
                for model_path in paths:
                    save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir)


if __name__ == "__main__":
    args = get_parser().parse_args()

    TESTIDSPATH = f'{args.scan_id_dir}/test_scanids.json'
    VALIDIDPATH = f'{args.scan_id_dir}/val_scanids.json'
    TRAINIDPATH = f'{args.scan_id_dir}/train_scanids.json'
    RAWDATAPATH = args.data_dir
    OUTPUTDATAPATH = args.output_dir

    start = time()
    pool = multiprocessing.Pool(processes=16)

    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()
    # pdb.set_trace()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    train_ids_file = open(TRAINIDPATH)
    train_ids = json.load(train_ids_file)
    train_ids_file.close()

    dir_names = ['rgb/', 'mask/', 'depth/', 'annotation/']

    output_dir = OUTPUTDATAPATH + "all/"
    for dir_name in dir_names:
        existDir(output_dir + 'train/' + dir_name)
        existDir(output_dir + 'valid/' + dir_name)
        existDir(output_dir + 'test/' + dir_name)


    model_paths = glob.glob(RAWDATAPATH + '*/*')
    anno_paths = glob.glob(RAWDATAPATH + 'annotation/*')
    types = ["annotation", "rgb", "depth", "mask"]

    process_map(partial(split, types=types, test_ids=test_ids, valid_ids=valid_ids, train_ids=train_ids), anno_paths, chunksize=1)

    stop = time()
    print(str(stop - start) + " seconds")
