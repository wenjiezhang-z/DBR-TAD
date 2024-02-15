# Diffusion Action Distiller for Temporal Action Detection

## Overview

This repository contains the code for Diffusion Action Distiller for Temporal Action Detection [paper](https://www.google.com). This code is an extension of the SoccerNet Action Spotting 2023 [championship algorithm](https://github.com/SoccerNet/sn-spotting).

## Installation
+ Recommended Environment: python 3.8.8, Cuda11.6, PyTorch 1.12.1(The PyTorch version should be at least >= 1.11.)
+ Install dependencies: `pip install  -r requirements.txt`
+ Install NMS: `cd ./libs/utils; python setup.py install --user; cd ../..`

## Data
### THUMOS14
+ I3D feature: from ActionFormer repository ([thumos_i3d](https://github.com/happyharrycn/actionformer_release/tree/main)).
+ VideoMAE feature: from InternVideo repository ([thumos_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)).
  
### ActivityNet
+ I3D feature: from TAGS repository ([anet_i3d](https://github.com/sauradip/tags)).
+ TSP feature: from ActionFormer repository ([anet_tsp](https://github.com/happyharrycn/actionformer_release/tree/main)).
+ VideoMAE feature: from InternVideo repository ([anet_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)).

### Epic-Kitchen
+ SlowFast feature: from ActionFormer repository ([epic_kitchen](https://github.com/happyharrycn/actionformer_release/tree/main)).

## Model
