# OPA: Learning Object-level Point Augmentor for Semi-supervised 3D Object Detection.

## Introduction

This repository is the code release for our BMVC 2022 [paper](https://reurl.cc/3Y0bZ9).

In this repository, we provide an implementation (with Pytorch) based on [VoteNet](https://github.com/facebookresearch/votenet), [SESS](https://github.com/Na-Z/sess), [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch) and [PointAugment](https://github.com/liruihui/PointAugment) with some modification, as well as the training and evaluation scripts on ScanNet.

## Installation
This repo is tested under the following environment:
- Python 3.7.6
- NumPy 1.18.5
- pytorch 1.10.1, cuda 11.3
- tensorflow 2.9.1
- Pointnet2 from [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- OpenPCDet from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

You can follow the steps below to install the above dependencies:
```
# Create and activate virtualenv
conda create -n OPA python=3.7.6
conda activate OPA

# Install NumPy
pip install numpy==1.18.5

# Install PyTorch according to your CUDA version.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install TensorFlow (for TensorBoard).
# We tested this repo with TensorFlow 2.9.1.
pip install tensorflow

# Compile the CUDA code for PointNet++, which is used in the backbone network.
# If you have any probelm about this part, you can refer to [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch#building-only-the-cuda-kernels)
cd pointnet2
python setup.py install

# Compile the CUDA code for general 3D IoU calculation in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet):
cd OpenPCDet
python setup.py develop

# Install dependencies:
pip install -r requirements.txt
```

## Dataset
### ScanNet
Please download the ScanNet data following the [README](https://github.com/nomiaro/OPA/blob/main/scannet/README.md) in scannet folder.

## Pre-training
Pre-train with script.
```
sh run_pretrain.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <BATCH_SIZE> <BOX_NUM> <LAMBDA> <WARM_UP>
```
For example:
```
sh run_pretrain.sh 0 results/pretrain scannet scannetv2_train_0.1.txt 4 3 0.1 0
```

## Training
Train with script.
```
sh run_train.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <PRETRAINED_DETECOR_CKPT> <PRETRAINED_AUGMENTOR_CKPT> <BATCH_SIZE> <BOX_NUM> <BOX_NUM_UNLABELED>
```
For example:
```
sh run_train.sh 0 results/train scannet scannetv2_train_0.1.txt results/pretrain/best_checkpoint_sum.tar results/pretrain/aug_best_checkpoint_sum.tar 2,4 3 3
```

## Evaluation
Evaluate with script.
```
sh run_eval.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT>
```
For example:
```
sh run_eval.sh 0 results/eval scannet scannetv2_train_0.1.txt results/train/best_checkpoint_sum.tar
```
If you want to evaluate with IoU optimization, please run:
```
sh run_eval_opt.sh <GPU_ID> <LOG_DIR> <DATASET> <LABELED_LIST> <CKPT> <OPT_RATE>
```
The number of steps (of optimization) is by default 10.

## Acknowledgements
Our implementation uses code from the following repositories:
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [SESS: Self-Ensembling Semi-Supervised 3D Object Detection](https://github.com/Na-Z/sess)
- [3DIoUMatch](https://github.com/THU17cyz/3DIoUMatch)
- [PointAugment: an Auto-Augmentation Framework for Point Cloud Classification](https://github.com/liruihui/PointAugment)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
