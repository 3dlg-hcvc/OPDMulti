import os
import glob
import json
import numpy as np
import pdb
from tqdm import tqdm

data_dir = {"all": "/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/opdmulti_V0"}
# data_dir = {"all": "/Users/sun_xh/multiopd/scripts/back_project/output"}

name_map_path = "/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/data_statistics/real_name_V0.json"
# name_map_path = "/Users/sun_xh/multiopd/scripts/multiscan_process/preprocess/real_name.json"

TESTIDSPATH = '/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/test_scan.json'
VALIDIDPATH = '/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/val_scan.json'


# datasets = ['train', 'val', 'test']
datasets = ['all']

if __name__ == "__main__":

    with open(name_map_path) as f:
        name_map = json.load(f)

    total_stat = {}

    test_scan_list = "/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/test_scan.json"
    val_scan_list = "/localhome/xsa55/Xiaohao/multiopd/data/MultiScan_dataset/scan_list/val_scan.json"

    test_ids_file = open(test_scan_list)
    test_scans = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(val_scan_list)
    val_scans = json.load(valid_ids_file)
    valid_ids_file.close()

    test_ids = []
    val_ids = []
    for dataset in datasets:

        current_object = {}
        current_object_number = 0
        current_scan_number = 0
        current_image_number = 0
        dirs = glob.glob(f"{data_dir[dataset]}")
        # pdb.set_trace()
        for dir in dirs:
            # print(f"working on {dir}")
            model_name = dir.split('/')[-1]
            current_object[model_name] = glob.glob(f"{dir}/rgb/*")
            current_object_number += 1
            current_image_number += len(current_object[model_name])
            scan_ids = []
            problem_scan_ids = []
            for image in tqdm(current_object[model_name]):
                image_name = (image.split('/')[-1]).split('.')[0]
                scan_id = (image.split('/')[-1]).split('.')[0][:50]
                # pdb.set_trace()
                if not os.path.isfile(f"{dir}/annotation/{image_name}.json"):
                    print(f"No annotation: {image}")
                    import pdb
                    pdb.set_trace()
                    # os.system(f"rm -rf {image}")
                    # os.system(f"rm -rf {dir}/depth/{image_name}_d.png")
                    # os.system(f"rm -rf {dir}/mask/{image_name}_*.png")
                    # if scan_id not in problem_scan_ids:
                    #     problem_scan_ids.append(scan_id)
                    # else:
                    #     print("Multiple images for the problem scan")
                    # continue
                if scan_id not in scan_ids:
                    current_scan_number += 1
                    scan_ids.append(scan_id)
                # Read the motion number
                # pdb.set_trace()
                with open(f"{dir}/annotation/{image_name}.json") as f:
                    anno = json.load(f)
                # Make it consistent with 2DMotion dataset
                extrinsic_matrix = np.linalg.inv(np.reshape(anno["extrinsics"], (4, 4), order="F")).flatten(order="F")
                anno["extrinsics"] = list(extrinsic_matrix)
                with open(f"{dir}/annotation/{image_name}.json", 'w') as f:
                    json.dump(anno, f)
                
                motion_number = len(anno["articulation"])
                motion_ids = [anno["partId"] for anno in anno["articulation"]]
                mask_paths = glob.glob(f"{dir}/mask/{image_name}_*")
                if not motion_number == len(mask_paths):
                    print(f"Not consistent mask and motion {image}")
                # Rename the RGB
                model_name = image_name.rsplit('-', 1)[0] + '_1'
                # pdb.set_trace()
                new_image_name = name_map[model_name] + '-' + image_name.rsplit('_', 1)[1]
                os.system(f"mv {dir}/rgb/{image_name}.png {dir}/rgb/{new_image_name}.png")
                # Rename the depth
                os.system(f"mv {dir}/depth/{image_name}_d.png {dir}/depth/{new_image_name}_d.png")
                # Rename the annotation
                os.system(f"mv {dir}/annotation/{image_name}.json {dir}/annotation/{new_image_name}.json")
                # Rename all the masks
                for mask_path in mask_paths:
                    mask_name = (mask_path.split('/')[-1]).split('.')[0]
                    if int(mask_name.rsplit('_', 1)[1]) not in motion_ids and int(mask_name.rsplit('_', 1)[1]) != 0:
                        import pdb
                        pdb.set_trace()
                    new_mask_name = f"{new_image_name}_{mask_name.rsplit('_', 1)[1]}"
                    os.system(f"mv {dir}/mask/{mask_name}.png {dir}/mask/{new_mask_name}.png")

                # pdb.set_trace()
                
                if scan_id[:14] in test_scans:
                    test_ids.append(new_image_name.split('-')[0])
                elif scan_id[:14] in val_scans:
                    val_ids.append(new_image_name.split('-')[0])

        total_stat[dataset] = current_object

        print(f"{dataset} Set -> Object Number {current_object_number}, Scan Number {current_scan_number}, Image Number {current_image_number}, Avg Images Per Object {current_image_number/current_object_number}, Avg Images Per Scan {current_image_number/current_scan_number}")
    
    val_ids_file = open(VALIDIDPATH, 'w')
    json.dump(val_ids, val_ids_file)
    val_ids_file.close()
    
    test_ids_file = open(TESTIDSPATH, 'w')
    json.dump(test_ids, test_ids_file)
    test_ids_file.close()
