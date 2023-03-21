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

* Clone the repository
```sh
git clone git@github.com:3dlg-hcvc/OPDMulti-Release.git
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
Download our `[OPDMulti]` [dataset](https://aspis.cmpt.sfu.ca/projects/opdmulti/OPDMulti.zip) (7.7G) to **./dataset** folder and extract the content. \
We also provide the code about how to process the [MultiScan](https://github.com/smartscenes/multiscan.git) dataset to OPDMulti dataset. So, you can process your own dataset through this procedure to get the customized dataset for OPDMulti task. Details can be found in [data_process](data_process).

If you want to try our model on OPDSynth and OPDReal datasets, you can find the data in original [OPD](https://github.com/3dlg-hcvc/OPD.git) repository.

## Training
To train from the scratch, you can use the below commands. The output will include evaluation results on the val set.

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
        * --config-file `/opdfomer/configs/opd_c_real.yaml`
    * OPDFormer-O:
        * --config-file `/opdfomer/configs/opd_o_real.yaml`
    * OPDFormer-P:
        * --config-file `/opdfomer/configs/opd_p_real.yaml`
* Dataset:
    * --data-path `OPDMulti/MotionDataset_h5`
    * --model_attr_path: ` OPDMulti/obj_info.json `
* Using pretrained model on OPDReal dataset (we use the pretrained model on OPDReal dataset to train the model on OPDMulti dataset in our paper): add the following command in the training command: 
    
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

## Pretrained-Models

You can download our pretrained models (OPDReal and OPDMulti) for different input format from the following table.
For people who want to just test/evaluate our model, you can download the pretrained models on OPDMulti dataset.
For those who want to train by themselves, you can download the pretrained OPDReal models to train your own model on OPDMulti.

### OPDMulti
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model Name</th>
<th valign="bottom">Input</th>
<th valign="bottom">PDet</th>
<th valign="bottom">+M</th>
<th valign="bottom">+MA</th>
<th valign="bottom">+MAO</th>
<th valign="bottom">Model</th>
<th valign="bottom">Pretrained OPDReal model (used for training)</th>
<!-- TABLE BODY -->
<!-- ROW: OPDFormer with RGB input -->
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-C</a></td>
<td align="center">RGB</td>
<td align="center">31.9</td>
<td align="center">30.5</td>
<td align="center">14.1</td>
<td align="center">12.9</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_c_rgb.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_c_rgb.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">RGB</td>
<td align="center">30.4</td>
<td align="center">28.8</td>
<td align="center">5.1</td>
<td align="center">1.6</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_o_rgb.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_o_rgb.pth">pretrained OPDReal model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-P</a></td>
<td align="center">RGB</td>
<td align="center">34.1</td>
<td align="center">32.8</td>
<td align="center">20.1</td>
<td align="center">16.0</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_p_rgb.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_p_rgb.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
<!-- ROW: OPDFormer with depth input -->
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-C</a></td>
<td align="center">depth</td>
<td align="center">22.1</td>
<td align="center">19.9</td>
<td align="center">11.4</td>
<td align="center">10.2</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_c_depth.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_c_depth.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">depth</td>
<td align="center">24.9</td>
<td align="center">22.6</td>
<td align="center">5.8</td>
<td align="center">1.9</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_o_depth.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_o_depth.pth">pretrained OPDReal model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-P</a></td>
<td align="center">depth</td>
<td align="center">23.0</td>
<td align="center">20.8</td>
<td align="center">16.1</td>
<td align="center">13.9</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_p_depth.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_p_depth.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
<!-- ROW: OPDFormer with RGBD input -->
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-C</a></td>
<td align="center">RGBD</td>
<td align="center">25.3</td>
<td align="center">23.6</td>
<td align="center">14.2</td>
<td align="center">13.5</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_c_rgbd.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_c_rgbd.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">RGBD</td>
<td align="center">24.1</td>
<td align="center">22.0</td>
<td align="center">6.6</td>
<td align="center">2.6</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_o_rgbd.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_o_rgbd.pth">pretrained OPDReal model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-P</a></td>
<td align="center">RGBD</td>
<td align="center">28.6</td>
<td align="center">26.5</td>
<td align="center">18.7</td>
<td align="center">17.2</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_p_rgbd.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_p_rgbd.pth">pretrained OPDReal model</a>(169M)</td>
</tr>
</tbody></table>



## Visualization (To be done)
