# Run this code, then statistics to process the MotionREAL data

import json
import glob
import pdb
import statistics
from multiscan.utils import io

object_info_path = "/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/opdmulti_V3_processed/annotation"
real_attr_path = "/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/data_statistics/real-attr-V3.json"
real_name_map_path = "/localhome/xsa55/Xiaohao/multiopd/scripts/mask2d/output/data_statistics/real_name_V3.json"

# not need scene diameter
if __name__ == "__main__":
    datasets = ["."]

    diagonal_dict = {}
    scan_index = 0
    scan_idx = 0
    real_name_map = {}
    tmp_scanid = []
    for dataset in datasets:
        anno_paths = glob.glob(f"{object_info_path}/{dataset}/*.json")
        # pdb.set_trace()
        for anno_path in anno_paths:
            anno_file = open(anno_path)
            try:
                anno = json.load(anno_file)
            except:
                pdb.set_trace()
            anno_file.close()
            # pdb.set_trace()
            if not len(anno.keys()) == 1:
                # pdb.set_trace()
                print(f"Something wrong: {anno_path}")
            if list(anno.keys())[0] in diagonal_dict.keys():
                print(f"Something wrong 2: {anno_path} {diagonal_dict[list(anno.keys())[0]]}")
            # diagonal_dict[str(scan_index)] = {"diameter": anno[list(anno.keys())[0]]["diameter"], "min_bound": anno[list(anno.keys())[0]]["min_bound"], "max_bound": anno[list(anno.keys())[0]]["max_bound"]}
            diagonal_dict[str(scan_index)] = {"diameter": anno["diagonal"]}
            # diagonal_dict[list(anno.keys())[0]] = anno_path
            # real_name_map[list(anno.keys())[0].rsplit('_', 1)[0] + '_1'] = str(scan_index)
            scan_id = anno_path.split("/")[-1].split(".")[0].rsplit("_",1)[0]
            frame_id = anno_path.split("/")[-1].split(".")[0].rsplit("_",1)[1]
            if scan_id not in tmp_scanid:
                real_name_map[scan_id] = str(scan_idx)
                scan_idx += 1
                tmp_scanid.append(scan_id)
            # pdb.set_trace()
            diagonal_dict[str(scan_id)+"-"+str(frame_id)] = {"diameter": anno["diagonal"]}
            scan_index += 1
    
    diagonal_file = open(real_attr_path, 'w')
    json.dump(diagonal_dict, diagonal_file)
    diagonal_file.close()

    map_file = open(real_name_map_path, 'w')
    json.dump(real_name_map, map_file)
    map_file.close()