#!/bin/bash

# THUMOS14 with I3D features
python ./train_eval.py ./configs/thumos_i3d.yaml --output final

# THUMOS14 with VideoMAEv2 features
python ./train_eval.py ./configs/thumos_videomaev2.yaml --output final

# ActivityNet 1.3 with I3D features
python ./train_eval.py ./configs/anet_i3d.yaml --output final

# ActivityNet 1.3 with TSP features
python ./train_eval.py ./configs/anet_tsp.yaml --output final

# ActivityNet 1.3 with VideoMAEv2 features
python ./train_eval.py ./configs/anet_videomaev2.yaml --output final

# EPIC-Kitchens verb with slowfast features
python ./train_eval.py ./configs/epic_slowfast_verb.yaml --output final

# EPIC-Kitchens noun with slowfast features
python ./train_eval.py ./configs/epic_slowfast_noun.yaml --output final
