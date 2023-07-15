# Downloaded data organization
The downloaded data organized as follows:
```PowerShell
OPDMulti
├── MotionDataset_h5
│   ├── annotations
│   │   ├── MotionNet_test.json
│   │   ├── MotionNet_train.json
│   │   ├── MotionNet_valid.json
│   ├── depth.h5
│   ├── test.h5
│   ├── train.h5
│   ├── valid.h5
├── obj_info.json
```
The depth.h5 file is a compressed .h5 file that stores all the depth images. The {train/valid/test}.h5 file is a compressed .h5 file containing all the RGB images from the respective train, validation, or test set.
The MotionNet_{train/test/valid}.json file includes annotations for each frame in the COCO format. The formatting is detailed as follows:
```json
{
    "annotations":[
        {
            "area": "product of the width and height of the bounding box",
            "bbox":["Bounding box of the openable part"],
            "category_id": "category type ID",
            "height": "height of the image",
            "id": "annotation ID",
            "image_id": "image ID",
            "iscrowd": "iscrowd or not",
            "motion":{
                "bbox": "Bounding box of the openable part",
                "axis": "aligned axis of the joint in object local common coordinate frame",
                "isClosed": "bool type, whether the part is close or not",
                "origin": "aligned origin of the joint in object local common coordinate frame",
                "pixel_num": "number of pixels of the openable part (this is for further frame filter procedure)",  
                "partId": "part ID",
                "part_label": "part label",
                "rangeMin": "min range of the motion",
                "rangeMax": "max range of the motion", 
                "state": "current state (degree/length) of the corresponding part",
                "type": "articulation type rotation or translation",
            },
            "object_key":"object key for the corresponding object information in obj_info.json file, {scan_id}_{frame_id}_{object_id}",
            "segmentation":["segmentation annotation of the openable part"],
            "width": "width of the image",
        },
        "..."
    ],
    "categories":[
        {
            "id": "category id",
            "name": "name of the openable part category",
            "supercategory": "Container_parts",
        },
        "..."
    ],
    "images":[
        {
            "camera":{
                "extrinsic":["16x1 vector of column major camera extrinsics (camera extrinsic)"],
                "intrinsic":["9x1 vector of column major camera intrinsics"],
            },
            "coco_url":"",
            "date_captured":"",
            "depth_file_name":"name of the depth image {scan_id}-{frame_id}-d.png",
            "file_name":"name of the rgb image {scan_id}-{frame_id}.png",
            "flickr_url":"",
            "height": "height of the image, 192",
            "id": "image ID",
            "license": "license ID",
            "width": "width of the image, 256"
        },
        "..."
    ],
    "info":["highlevel information of the dataset"],
    "licenses":["..."]
}
```
The obj_info.json file contains the information of each openable object. The format is as follows:
```json
"{scan_id}_{frame_id}_{object_id}": {
    "object_pose": ["16x1 vector of plained object pose matrix"],
    "diagonal": "diagonal length of the object",
    "min_bound": ["minimum bound coordinate"],
    "max_bound": ["maximum bound coordinate"]
  },
```

# Data processing procedure

The OPDMulti dataset is created from [MultiScan](https://github.com/smartscenes/multiscan.git) dataset. To download the raw MultiScan dataset, you can visit their repository.

## Additional environment setup
This procedure mainly followed the Python environment setup in the MultiScan project. [link](https://3dlg-hcvc.github.io/multiscan/read-the-docs/server/index.html)

You can install the packages needed for the MultiScan project in the opdmulti environment. After installing the neccessary packages, there are a few more packages needed:
```sh
conda install pandas
conda install psutil
pip install pyglet==1.5.27
pip install plyfile
```

## Process procedure
First, we preprocess the MultiScan dataset to 2D image with corresponding annotations: 
```sh
cd data_process/mask2d
python mask2d.py input_dir=<PATH_TO_MULTISCAN_DATA> output_dir=<OUTPUT_DIR>
```
After the above procedure, we further process the processed MultiScan dataset to have the annotations only with the openable object and parts.
```sh
python annotation_update.py input_dir=<PATH_TO_PROCESSED_DATA> output_dir=<PATH_TO_PROCESSED_DATA>
python split_mask.py input_dir=<PATH_TO_PROCESSED_DATA>
```
Then, we can get the motion annotation (.json file) for the openable part in each frame. The annotation format is as follows:
```json
{
    "intrinsic": {
        "matrix": "9x1 vector of column major camera intrinsics"
    },
    "extrinsic": {
        "matrix": "16x1 vector of column major camera extrinsics (camera extrinsic)"
    },
    "diagonal": "Diagonal length of the scene",
    "articulation": [
        {
            "bbox": "Bounding box of the openable part",
            "axis": "aligned axis of the joint in object local common coordinate frame",
            "isClosed": "bool type, whether the part is close or not",
            "origin": "aligned origin of the joint in object local common coordinate frame",
            "pixel_num": "number of pixels of the openable part (this is for further frame filter procedure)",  
            "partId": "part ID",
            "part_label": "part label",
            "rangeMin": "min range of the motion",
            "rangeMax": "max range of the motion", 
            "state": "current state (degree/length) of the corresponding part",
            "type": "articulation type rotation or translation"        
        }
        "..."
    ]
}
```


Get the name mapping to rename the scans into consistent format and get the diagonal of each scan
```sh
cd ../process
python motion_real_diagonal.py
```
Change the annotation into 2DMotion format, get the dataset split
```sh
python motion_real_statistics.py
```
Split the processed dataset to train/val/test set
```sh
python split.py
```
Convert the dataset to COCO format that detectron2 needs, and convert to h5 format.
```sh
python convert_coco.py
python final_dataset.py
python convert_h5.py
```
After the above data processing procedure, the data directory will be organized as follows:
```PowerShell
MotionDataset
├── annotations
│   ├── MotionNet_train.json
│   ├── MotionNet_val.json
│   ├── MotionNet_test.json
├── depth
│   ├── {new_scan_id}_{frame_id}-d.png
├── train
│   ├── {new_scan_id}_{frame_id}.png
├── test
│   ├── {new_scan_id}_{frame_id}.png
├── valid
│   ├── {new_scan_id}_{frame_id}.png
```
