# Data processing procedure

The OPDMulti dataset is created from [MultiScan](https://github.com/smartscenes/multiscan.git) dataset.To download the raw MultiScan dataset, you can visit their repository.

## Additional enviornment setup
This procedure mainly followed the python environment setup in MultiScan project. [link](https://3dlg-hcvc.github.io/multiscan/read-the-docs/server/index.html)

You can install the packages needed for MultiScan project in the opdmulti environment. After install the neccessary packages, there are a few more packages needed:
```sh
conda install pandas
conda install psutil
pip install pyglet==1.5.27
pip install plyfile
```

## Process procedure
First we preprocess the MultiScan dataset to 2D image with corresponding annotations: 
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
* In the above procedures, there are some directories in the code need to be modified manually depends on how you save the processed the dataset.

After the above data processing procedure, the data directory will be organized as follows:
```shell
MotionDataset
├── annotations
│   ├── MotionNet_train.json
│   ├── MotionNet_val.json
│   ├── MotionNet_test.json
├── depth
│   ├── {scene_id}_{frame_id}-d.png
├── train
│   ├── {scene_id}_{frame_id}.png
├── test
│   ├── {scene_id}_{frame_id}.png
├── valid
│   ├── {scene_id}_{frame_id}.png
```
