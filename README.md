# OPDMulti: Openable Part Detection for Multiple Objects
[Xiaohao Sun*](https://sun-xh.github.io/), [Hanxiao Jiang*](https://jianghanxiao.github.io/), [Manolis Savva](https://msavva.github.io/), [Angel Xuan Chang](http://angelxuanchang.github.io/)

## Overview
This repository contains the implementation of **OPDFormer** based methods for the new proposed **OPDMulti** task and corresponding dataset. The code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [OPD](https://github.com/3dlg-hcvc/OPD.git). And the **OPDFormer** models were built on [Mask2Former](https://github.com/facebookresearch/Mask2Former).

<p align="center"><img src="fig/teaser.png" width="100%"></p>


[arXiv](https://arxiv.org/abs/2303.14087)&nbsp; [Website](https://3dlg-hcvc.github.io/OPDMulti/)&nbsp;
[Demo](https://huggingface.co/spaces/3dlg-hcvc/opdmulti-demo)


## Content

- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Visualization](#visualization)



## Setup
The implementation has been tested on Ubuntu 20.04, with Python 3.7, PyTorch 1.10.1, CUDA 11.1.1 and CUDNN 8.2.0.

* Clone the repository
```sh
git clone git@github.com:3dlg-hcvc/OPDMulti.git
```

* Setup python environment to train the model
<!-- conda create -n opdmulti python=3.7 
conda activate opdmulti
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -U opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git
pip install setuptools==59.5.0 -->
```sh
conda create -n opdmulti python=3.7 
conda activate opdmulti

pip install -r requirements.txt

cd opdformer/mask2former/modeling/pixel_decoder/ops
python setup.py build install
```

## Dataset
Download our [OPDMulti](https://docs.google.com/forms/d/e/1FAIpQLSeG1Jafcy9P_OFBJ8WffYt6WJsJszXPqKIgQz0tGTYYuhm4SA/viewform?vc=0&c=0&w=1&flr=0) dataset (7.1G) and extract it inside `./dataset/` folder. Make sure the data is in [this](https://github.com/3dlg-hcvc/OPDMulti/blob/master/data/README.md#downloaded-data-organization) format.  You can follow [these](https://github.com/3dlg-hcvc/OPDMulti/blob/master/data/README.md#data-processing-procedure) steps if you want to convert your data to OPDMulti dataset. To try our model on OPDSynth and OPDReal datasets, download the data from [OPD](https://github.com/3dlg-hcvc/OPD#dataset) repository.



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
* `<MODEL_CONFIG>`: the config file path for different model variants can be found in the table [OPDMulti](#opdmulti) "Model Name" column.
    
* Dataset:
    * --data-path `OPDMulti/MotionDataset_h5`
    * --model_attr_path: ` OPDMulti/obj_info.json `
* You can add the following command to use the model weights, pretrained on OPDReal dataset. We finetune this model on OPDMulti dataset:

  `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>`

## Evaluation
To evaluate, use the following command:

```sh
python evaluate_on_log.py \
--config-file <MODEL_CONFIG> \
--output-dir <OUTPUT_DIR> \
--data-path <PATH_TO_DATASET> \
--input-format <RGB/depth/RGBD> \
--model_attr_path <PATH_TO_ATTR> \
--opts MODEL.WEIGHTS <PPRETRAINED_MODEL>
```

* Evaluate on test set: `--opts MODEL.WEIGHTS <PPRETRAINED_MODEL> DATASETS.TEST "('MotionNet_test',)"`.
* To evaluate directly on pre-saved inference file, pass the file path as an argument `--inference-file <PATH_TO_INFERENCE_FILE>`.

## Pretrained-Models

You can download our pretrained model weights (on both OPDReal and OPDMulti) for different input format (RGB, RGB-D, depth) from the following table.


For model evaluation, download pretrained weights from the OPDMulti column. To finetune with custom data, use pretrained weights from OPDReal column, which are also utilized in OPDMulti results.

### How to read the table
The "Model Name" column contains a link to the config file. "PSeg" is the part segmentation score, "+M" adds motion type prediction, 
"+MA" includes axis prediction, and "+MAO" further incorporates origin prediction. 

To train/evaluate the different model variants, change the ``` --config-file /path/to/config/name.yaml``` in the training/evaluation command. 

### OPDMulti
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model Name</th>
<th valign="bottom">Input</th>
<th valign="bottom">PSeg</th>
<th valign="bottom">+M</th>
<th valign="bottom">+MA</th>
<th valign="bottom">+MAO</th>
<th valign="bottom">OPDMulti Model</th>
<th valign="bottom">OPDReal Model</th>
<!-- TABLE BODY -->
<!-- ROW: OPDFormer with RGB input -->
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-C</a></td>
<td align="center">RGB</td>
<td align="center">29.1</td>
<td align="center">28.0</td>
<td align="center">13.5</td>
<td align="center">12.3</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_c_rgb.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_c_rgb.pth">model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">RGB</td>
<td align="center">27.8</td>
<td align="center">26.3</td>
<td align="center">5.0</td>
<td align="center">1.5</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_o_rgb.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_o_rgb.pth">model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-P</a></td>
<td align="center">RGB</td>
<td align="center">31.4</td>
<td align="center">30.4</td>
<td align="center">18.9</td>
<td align="center">15.1</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGB/opdmulti_opdformer_p_rgb.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGB/opdreal_opdformer_p_rgb.pth">model</a>(169M)</td>
</tr>
<!-- ROW: OPDFormer with depth input -->
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-C</a></td>
<td align="center">depth</td>
<td align="center">20.9</td>
<td align="center">18.9</td>
<td align="center">11.4</td>
<td align="center">10.1</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_c_depth.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_c_depth.pth">model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">depth</td>
<td align="center">23.4</td>
<td align="center">21.5</td>
<td align="center">5.9</td>
<td align="center">1.9</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_o_depth.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_o_depth.pth">model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-P</a></td>
<td align="center">depth</td>
<td align="center">21.7</td>
<td align="center">19.8</td>
<td align="center">15.4</td>
<td align="center">13.5</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/depth/opdmulti_opdformer_p_depth.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/depth/opdreal_opdformer_p_depth.pth">model</a>(169M)</td>
</tr>
<!-- ROW: OPDFormer with RGBD input -->
<tr><td align="left"><a href="opdformer/configs/opd_c_real.yaml">OPDFormer-C</a></td>
<td align="center">RGBD</td>
<td align="center">24.2</td>
<td align="center">22.7</td>
<td align="center">14.1</td>
<td align="center">13.4</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_c_rgbd.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_c_rgbd.pth">model</a>(169M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_o_real.yaml">OPDFormer-O</a></td>
<td align="center">RGBD</td>
<td align="center">23.1</td>
<td align="center">21.2</td>
<td align="center">6.7</td>
<td align="center">2.6</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_o_rgbd.pth">model</a>(175M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_o_rgbd.pth">model</a>(175M)</td>
</tr>
<tr><td align="left"><a href="opdformer/configs/opd_p_real.yaml">OPDFormer-P</a></td>
<td align="center">RGBD</td>
<td align="center">27.4</td>
<td align="center">25.5</td>
<td align="center">18.1</td>
<td align="center">16.7</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdmulti/RGBD/opdmulti_opdformer_p_rgbd.pth">model</a>(169M)</td>
<td align="center"><a href="https://aspis.cmpt.sfu.ca/projects/opdmulti/models/opdreal/RGBD/opdreal_opdformer_p_rgbd.pth">model</a>(169M)</td>
</tr>
</tbody></table>



## Visualization
The visualization code is based on [OPD](https://github.com/3dlg-hcvc/OPD.git) repository. We only support visualization based on raw dataset format ([download link](https://docs.google.com/forms/d/e/1FAIpQLSeG1Jafcy9P_OFBJ8WffYt6WJsJszXPqKIgQz0tGTYYuhm4SA/viewform?vc=0&c=0&w=1&flr=0) (4.9G)).

And the visualization uses the inference file, which can be obtained after the evaluation.
* Visualize the GT with 1000 random images in val set 
  ```sh
  cd opdformer
  python render_gt.py \
  --output-dir vis_output \
  --data-path <PATH_TO_DATASET> \
  --valid-image <IMAGE_LIST_FILE> \
  --is-real
  ```
* Visualize the PREDICTION with 1000 random images in val set
  ```sh
  cd opdformer
  python render_pred.py \
  --output-dir vis_output \
  --data-path <PATH_TO_DATASET> \
  --model_attr_path <PATH_TO_ATTR> \
  --valid-image <IMAGE_LIST_FILE> \
  --inference-file <PATH_TO_INFERENCE_FILE> \
  --score-threshold 0.8 \
  --update-all \
  --is-real
  ```
  * --data-path `dataset/MotionDataset`
  * --valid_image `dataset/MotionDataset/valid_1000.json`

## Citation
If you find this code useful, please consider citing:
```bibtex
@article{sun2023opdmulti,
  title={OPDMulti: Openable Part Detection for Multiple Objects},
  author={Sun, Xiaohao and Jiang, Hanxiao and Savva, Manolis and Chang, Angel Xuan},
  journal={arXiv preprint arXiv:2303.14087},
  year={2023}
}

@article{mao2022multiscan,
  title={MultiScan: Scalable RGBD scanning for 3D environments with articulated objects},
  author={Mao, Yongsen and Zhang, Yiming and Jiang, Hanxiao and Chang, Angel and Savva, Manolis},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9058--9071},
  year={2022}
}

@inproceedings{jiang2022opd,
  title={OPD: Single-view 3D openable part detection},
  author={Jiang, Hanxiao and Mao, Yongsen and Savva, Manolis and Chang, Angel X},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXIX},
  pages={410--426},
  year={2022},
  organization={Springer}
}

@inproceedings{cheng2022masked,
  title={Masked-attention mask transformer for universal image segmentation},
  author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander G and Kirillov, Alexander and Girdhar, Rohit},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1290--1299},
  year={2022}
}
```

## Acknowledgement
This work was funded in part by a Canada CIFAR AI Chair, a Canada Research Chair and
NSERC Discovery Grant, and enabled in part by support from WestGrid and Compute Canada. We thank Yongsen Mao for helping us with the data processing procedure. We also thank Jiayi Liu, Sonia Raychaudhuri, Ning Wang, Yiming Zhang for feedback on paper drafts.