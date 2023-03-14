# OPDMulti: Openable Part Detection for Multiple Objects
[Xiaohao Sun*](https://ca.linkedin.com/in/xiaohao-sun-237537195?trk=public_profile_browsemap), [Hanxiao Jiang*](https://jianghanxiao.github.io/), [Manolis Savva](https://msavva.github.io/), [Angel Xuan Chang](http://angelxuanchang.github.io/)

## Overview
This repository contains the implementation of **OPDFormer** based methods for the new proposed **OPDMulti** task and corresponding dataset. The code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [OPD](https://github.com/3dlg-hcvc/OPD.git)

<p align="center"><img src="fig/teaser.png" width="100%"></p>


[arXiv]()&nbsp; [Website]()


## Content

- [Setup](#Setup)
- [Dataset](#Dataset)
- [Pretrained Models](#Pretrained-Models)
- [Training](#Training)
- [Evaluation](#Eveluation)
- [Visualization](#Visualization)



## Setup
The implementation has been tested on Ubuntu 20.04, with Python 3.7, PyTorch 1.10.1, CUDA 11.1.1 and CUDNN 8.2.0.

* Clone the repository (to be modified)
```sh
git clone git@github.com:Sun-XH/OPD-Multi.git
```

* Setup python environment to train the model
```sh
conda create -n opdmulti python=3.7 
conda activate opdmulti
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -U opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git
pip install setuptools==59.5.0
pip install -r requirements.txt

cd opdformer/mask2former/modeling/pixel_decoder/ops
python setup.py build install
```

## Dataset
Download our `[OPDMulti]` [dataset](https://aspis.cmpt.sfu.ca/projects/opdmulti/OPDMulti.zip) to **./dataset** folder and extract the content. \
We also provide the code about how to process the [MultiScan](https://github.com/smartscenes/multiscan.git) dataset to OPDMulti dataset. So, you can process your own dataset through this procedure to get the customized dataset for OPDMulti task. Details can be found in [data_process](data_process).

## Pretrained-Models
You can download our pretrained [models](https://aspis.cmpt.sfu.ca/projects/opdmulti/models.zip) to **./models** folder.

The folder contains the pretrained models trained with OPDMulti dataset **RGB** input. 
There will be pretrained models for following OPDFormer variants:

`[OPDFormer-C]` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `[OPDFormer-O]` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `[OPDFormer-P]` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

## Training
To train from the scratch, you can use below commands. The output will include evaluation results on the val set.

```sh
cd opdformer
python train.py \
--config-file <MODEL_CONFIG> \
--output-dir <OUTPUT_DIR> \
--data-path <PATH_TO_DATASET> \
--input-format <RGB/depth/RGBD> \
--model_attr_path <PATH_TO_ATTR> 
```
* Model:
    * OPDFormer-C: 
        * --config-file `/opdfomer/configs/opd_cc_real.yaml`
    * OPDFormer-O:
        * --config-file `/opdfomer/configs/opd_o_real.yaml`
    * OPDFormer-O-W:
        * --config-file `/opdfomer/configs/opd_op_real.yaml`
    * OPDFormer-O-P:
        * --config-file `/opdfomer/configs/opd_ow_real.yaml`
* Dataset:
    * --data-path `OPDMulti/MotionDataset_h5`
    * --model_attr_path: ` OPDMulti/obj_info.json `
* Using pretrained model on OPDReal dataset: add the following command in the trianing command: 
    
    `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>`
## Evaluation
Evaluate with pretrained model, or your own trained model on val set

```sh
python evaluate_on_log.py \
--config-file <MODEL_CONFIG> \
--output-dir <OUTPUT_DIR> \
--data-path <PATH_TO_DATASET> \
--input-format <RGB/depth/RGBD> \
--model_attr_path <PATH_TO_ATTR> \
--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>
```

* Model needs the same options as above
* Evaluate on test set: add things to `--opts DATASETS.TEST "('MotionNet_test',)"` (The complete version will be `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL> DATASETS.TEST "('MotionNet_test',)"`)
* Use inference result file instead of pretrained model: --inference-file `<PATH_TO_INFERENCE_FILE>`, this will directly evaluate using the results without inferencing again

## Visualization (To be done)