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
- Pointnet2 from https://github.com/erikwijmans/Pointnet2_PyTorch
- OpenPCDet from https://github.com/open-mmlab/OpenPCDet
