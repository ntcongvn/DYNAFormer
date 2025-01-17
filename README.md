# DYNAFormer
DynAMFormer: Enhancing Transformer Segmentation with Dynamic Anchor Mask for Medical Imaging

##  Introduction

This repository contains the PyTorch implementation of DYNAFormer, Enhancing Transformer Segmentation with Dynamic Anchor Mask for Medical Imaging.
DYNAFormer use an anchor mask-guided mechanism that enhances segmentation by using each positional query as an anchor mask, instead of traditional anchor boxes in methods like DINO and MaskDINO, enabling adaptive focus on relevant areas with pixel-level precision rather than coarse bounding box regions. This approach allows more precise, adaptive feature learning through dynamic refinement across decoder layers.

![model](figures/DynAMFormer_Overview.jpg)

##  Install dependencies

Dependent libraries
* torch
* torchvision 
* opencv
* ninja
* fvcore
* iopath
* antlr4-python3-runtime==4.9.2

Install detectron2 and DYNAFormer

```bask
# Under your working directory
# Install Detectron2
cd ./detectron2
!python setup.py build develop
cd ..

#Install requirements for DYNAFormer
cd ./dynaformer
!pip install -r requirements.txt
cd ..

cd ./dynaformer/dynaformer/modeling/pixel_decoder/ops
!sh make.sh
cd ..
```

##  Polyp instance segmentation dataset (PolypDB_INS)
In this work, Introduce PolypDB_INS, a dataset of 4,403 images with 4,918 polyps, gathered from multiple sources and annotated for Sessile and Pedunculated polyp instances, which supports the development of polyp instance segmentation tasks. Besides, PolypDB_INS is also adapted for polyp semantic segmentation by converting instance segmentation masks into binary masks. These masks identify regions containing polyps without distinguishing between specific types, enhancing the dataset’s applicability to broader segmentation tasks. The dataset is avaiable at [download link](<https://drive.google.com/file/d/1olTs9hZA4o81vfrYO32oZVuGzvTVNIQ_/view?usp=sharing>)
| Subset | Images | Prop. | Polyps | Sessile | Pedunculated |
|--------|--------|-------|--------|---------|--------------|
| All    | 4,403  | 100%  | 4,918  | 4,566   | 352          |
| Train  | 2,642  | 60%   | 2,926  | 2,711   | 215          |
| Val    | 880    | 20%   | 989    | 916     | 73           |
| Test   | 881    | 20%   | 1,003  | 939     | 64           |


##  Usage

####  1. Training

```bash
!python "./dynaformer/train_net.py" --config-file "$config_file" --num-gpus 1 --resume DATASETS.TRAIN '("polypdb_ins_train",)' DATASETS.TEST '("polypdb_ins_val",)' DATALOADER.NUM_WORKERS 8  SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0001 SOLVER.STEPS "(19800,26400)" SOLVER.MAX_ITER 29700 SOLVER.CHECKPOINT_PERIOD 330 TEST.EVAL_PERIOD 330 TEST.EVAL_START_ITER 0 OUTPUT_DIR "$output_dir"
```
* $config_file: the path to the config file, polyp instance segmentation(./dynaformer/configs/polypdb_ins/instance-segmentation/dynaformer_R50_bs8_90ep.yaml).
* $output_dir: specify the path to save the checkpoint during the training process.

####  2. Inference

```bash
!python "./dynaformer/train_net.py" --config-file "$config_file" --num-gpus 1 --eval-only  DATASETS.TEST '("polypdb_ins_test",)' DATALOADER.NUM_WORKERS 1  SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS "$checkpoint" OUTPUT_DIR "$output_dir"
```
* $config_file: the path to the config file, polyp instance segmentation(./dynaformer/configs/polypdb_ins/instance-segmentation/dynaformer_R50_bs8_90ep.yaml).
* $output_dir: specify the path to save the results during the evaluating process.
* $checkpoint: path to the trained model's checkpoint.

##  Acknowledgement

Part of the code was adpated from [Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation](<https://github.com/IDEA-Research/MaskDINO>)

```bash
@INPROCEEDINGS {10204168,
author = { Li, Feng and Zhang, Hao and Xu, Huaizhe and Liu, Shilong and Zhang, Lei and Ni, Lionel M. and Shum, Heung-Yeung },
booktitle = { 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) },
title = {{ Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation }},
year = {2023},
volume = {},
ISSN = {},
pages = {3041-3050},
keywords = {Training;Visualization;Semantic segmentation;Machine vision;Semantics;Noise reduction;Object detection},
doi = {10.1109/CVPR52729.2023.00297},
url = {https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.00297},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Jun
}
```

