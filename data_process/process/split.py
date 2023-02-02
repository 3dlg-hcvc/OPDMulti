import glob
import os
import json
import multiprocessing
from time import time
import pdb

# Split the raw dataset into train/valid/test set based on the splitted model ids
TESTIDSPATH = '/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/test_scan.json'
VALIDIDPATH = '/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/val_scan.json'
RAWDATAPATH = '/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/opdmulti_V0/'
OUTPUTDATAPATH = '/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/opdmulti_V0_output/'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def process(model_path, train_test):
    # dir_names = ['origin', 'mask', 'depth', 'origin_annotation']
    
    # for dir_name in dir_names:
    #     file_paths = glob.glob(f'{model_path}/{dir_name}/*')
    #     for file_path in file_paths:
    #         # pdb.set_trace()
    #         file_name = file_path.split('/')[-1]
    #         process_path = OUTPUTDATAPATH + train_test + '/' + dir_name + '/' + file_name
    #         os.system(f'cp {file_path} {process_path}')
    file_path = model_path
    process_path = OUTPUTDATAPATH + train_test + '/' + file_path.split('/')[-2] + '/' + file_path.split('/')[-1] 
    os.system(f'cp {file_path} {process_path}')

# def save_file(paths):
#     for model_path in paths:
#         model_id = model_path.split('/')output_all_preprocess
#             # process(model_path, 'test')
#             pool.apply_async(process, (model_path, 'test',))
#         elif model_id in valid_ids:
#             # process(model_path, 'valid')
#             pool.apply_async(process, (model_path, 'valid',))
#         else:
#             # process(model_path, 'train')
#             pool.apply_async(process, (model_path, 'train',))

if __name__ == "__main__":
    start = time()
    pool = multiprocessing.Pool(processes=16)

    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()
    # pdb.set_trace()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    dir_names = ['rgb/', 'mask/', 'depth/', 'annotation/']

    for dir_name in dir_names:
        existDir(OUTPUTDATAPATH + 'train/' + dir_name)
        existDir(OUTPUTDATAPATH + 'valid/' + dir_name)
        existDir(OUTPUTDATAPATH + 'test/' + dir_name)

    # mask_paths = glob.glob(RAWDATAPATH + 'mask/*')
    # origin_paths = glob.glob(RAWDATAPATH + 'origin/*')
    # annotation_paths = glob.glob(RAWDATAPATH + 'origin_annotation/*')
    # depth_paths = glob.glob(RAWDATAPATH + 'depth/*')
    model_paths = glob.glob(RAWDATAPATH + '*/*')
    for model_path in model_paths:
        model_id = model_path.split('/')[-1].split('-')[0]
        # pdb.set_trace()
       
        if model_id in test_ids:
            # process(model_path, 'test')
            pool.apply_async(process, (model_path, 'test',))
            # pool.apply_async(process, (model_path, 'valid',))
            # pool.apply_async(process, (model_path, 'train',))
        elif model_id in valid_ids:
            # process(model_path, 'valid')
            pool.apply_async(process, (model_path, 'valid',))
        else:
            # process(model_path, 'train')
            pool.apply_async(process, (model_path, 'train',))

            

    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")

