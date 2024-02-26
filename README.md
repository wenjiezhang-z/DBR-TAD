# DB<sup>2</sup>R: Diffusion-Based Boundary Refinement for Temporal Action Detection

## Overview

![Overview](./assets/Overview.png)

## Installation
+ Recommended Environment: python 3.8.8, Cuda11.6, PyTorch 1.12.1(The PyTorch version should be at least >= 1.11.)
+ Install dependencies: `pip install  -r requirements.txt`
+ Install NMS: `cd ./libs/utils; python setup.py install --user; cd ../..`

## Data
**Download Features and Annotations**
| Dataset  | Feature Encoder| Link |
|:-----------:|:-----------:|:------------:|
| THUMOS14 | I3D | [thumos_i3d](https://github.com/happyharrycn/actionformer_release/tree/main)|
| THUMOS14 | VideoMAE | [thumos_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)|
| ActivityNet | I3D | [anet_i3d](https://github.com/sauradip/tags)|
| ActivityNet | TSP | [anet_tsp](https://github.com/happyharrycn/actionformer_release/tree/main)|
| ActivityNet | VideoMAE | [anet_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)|
| Epic-Kitchen | SlowFast | [epic_kitchen](https://github.com/happyharrycn/actionformer_release/tree/main)|

**Unpack Features and Annotations**
+ Unpack the file under `./data`
+ The folder structure should look like
```
DAD-TAD/
  ├── data
  │   ├── anet_1.3
  │   │   ├── annotations
  │   │   ├── i3d_features
  │   │   ├── tsp_features
  │   │   └── anet_mae_hugek700
  │   ├── epic_kitchens
  │   │   ├── annotations
  │   │   ├── features
  │   └── thumos
  │       ├── annotations
  │       ├── i3d_features
  │       ├── th14_mae_g_16_4
  ├── libs
  ├── tools
  └── ...
```

<!--
### THUMOS14
+ I3D feature: from ActionFormer repository ([thumos_i3d](https://github.com/happyharrycn/actionformer_release/tree/main)).
+ VideoMAE feature: from InternVideo repository ([thumos_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)).
  
### ActivityNet
+ I3D feature: from TAGS repository ([anet_i3d](https://github.com/sauradip/tags)).
+ TSP feature: from ActionFormer repository ([anet_tsp](https://github.com/happyharrycn/actionformer_release/tree/main)).
+ VideoMAE feature: from InternVideo repository ([anet_videomae](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization)).

### Epic-Kitchen
+ SlowFast feature: from ActionFormer repository ([epic_kitchen](https://github.com/happyharrycn/actionformer_release/tree/main)).
-->

## Training and Evaluation
+ We have provided a script list that allows you to replicate our results with just a single click. Further details can be found in `./tools/run_all_exps.sh`.
+ [Optional] Monitor the training using TensorBoard
```
tensorboard --logdir=./ckpt/thumos_i3d_reproduce/logs
```

## Trained Models
We provide pre-trained models for each dataset, which you can download from XXX.
