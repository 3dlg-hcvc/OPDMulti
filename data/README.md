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
```javascript
{
    "annotations":[
        {
            "area":11223, // product of the width and height of the bounding box
            "bbox":[
                64.0,
                0.0,
                82.0,
                159.0
            ], // Bounding box of the openable part,
            "category_id":2, // category type ID
            "height":192, // height of the image
            "id":1, //annotation ID
            "image_id":301, // image ID
            "motion":{
                "axis":[
                    0.025904115699919135,
                    -0.9962978954426519,
                    0.08197207293729558
                ], // aligned axis of the joint in object local common coordinate frame
                "bbox":[
                    0.0,
                    64.0,
                    158.0,
                    81.0
                ], // Bounding box of the openable part,
                "isClosed":false, // bool type, whether the part is close or not
                "origin":[
                    0.08532447757098727,
                    -0.21926973532126864,
                    1.0033603450033186
                ], // aligned origin of the joint in object local common coordinate frame
                "partId":64, //part ID
                "part_label":"door", // part label
                "pixel_num":11223.0, //number of pixels of the openable part
                "rangeMax":1.5882496193148399, // max range of the motion
                "rangeMin":0.0, // min range of the motion
                "state":1.5882496193148399, // current state (degree/length) of the corresponding part
                "type":"rotation", // articulation type rotation or translation
            },
            "object_key":"scene_00022_01_6720_38", //object key for the corresponding object information in obj_info.json file, {scan_id}_{frame_id}_{object_id}
            "segmentation":[
                [
                    90.0,
                    158.5,
                    76.5,
                    158.0,
                    82.0,
                    154.5,
                    68.0,
                    157.5,
                    63.5,
                    ...    
                ]
            ], // segmentation annotation of the openable part
            "width":256, // width of the image
        },
        "..."
    ],
    "categories":[
        {
            "id":1, // category id
            "name":"drawer", // name of the openable part category
            "supercategory":"Container_parts", // supercategory name of the openable part
        },
        "..."
    ],
    "images":[
        {
            "camera":{
                "extrinsic":[
                    0.6855309133070147,
                    -0.727778169346729,
                    -0.019662642160811438,
                    0.0,
                    -0.651918925356933,
                    -0.6016042139238198,
                    -0.4615997947215913,
                    0.0,
                    0.3241130992861542,
                    0.3292593623576632,
                    -0.8868704966825542,
                    0.0,
                    -1.2662087105240858,
                    0.8717800943911488,
                    -0.06219740833738439,
                    1.0
                ], // 16x1 vector of column major camera extrinsics (camera extrinsic
                "intrinsic":[
                    213.77478841145833,
                    0.0,
                    0.0,
                    0.0,
                    213.77478841145833,
                    0.0,
                    125.90768229166666,
                    96.78001302083334,
                    1.0
                ] // 9x1 vector of column major camera intrinsics
            },
            "depth_file_name":"157-8820_d.png", //name of the depth image {scan_id}-{frame_id}-d.png
            "file_name":"157-8820.png", //name of the rgb image {scan_id}-{frame_id}.png
            "height":192,// height of the image
            "id":1, // image ID
            "width":256, // width of the image
        },
        "..."
    ],
}
```
The obj_info.json file contains the information of each openable object. The format is as follows:
```javascript
"scene_00000_00_5340_18": { // "{scan_id}_{frame_id}_{object_id}"
    "object_pose": [
      0.9450423736594992,
      -0.05121394778485824,
      -0.3229118192784162,
      0.0,
      -0.17907651349002185,
      -0.9074135599260141,
      -0.3801739377275279,
      0.0,
      -0.2735443552776718,
      0.4171064166548963,
      -0.8667154862139299,
      0.0,
      0.1552664914067867,
      -0.47938126636805617,
      0.9964610850667033,
      1.0
    ], // 16x1 vector of plained object pose matrix"
    "diagonal": 1.7046188091507766, // diagonal length of the object
    "min_bound": [
      0.9386587973543685,
      0.1965621361842551,
      0.8723619817984625
    ], // minimum bound coordinate
    "max_bound": [
      -0.7165200254461667,
      -0.08946254516454864,
      1.0447304952034777
    ] // maximum bound coordinate
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
Then, we can get the motion annotation (.json file) for the openable part in each frame.

Next, get the name mapping to rename the scans into consistent format and get the diagonal of each scan
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
Convert the dataset to COCO format that detectron2 needs
```sh
python convert_coco.py
python final_dataset.py
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
Finally, convert the images to `.h5` format, we can get the final dataset structured follow the [organization](#downloaded-data-organization) at the begining.
```
python convert_h5.py
```