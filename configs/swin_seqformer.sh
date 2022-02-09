#!/usr/bin/env bash

set -x

python3 -u main.py \
    --dataset_file jointcoco \
    --epochs 12 \
    --lr 2e-4 \
    --lr_drop 4 10\
    --batch_size 1 \
    --num_workers 1 \
    --coco_path /home/omkarthawakar/datasets/coco \
    --ytvis_path /home/omkarthawakar/datasets/ytvis-2019 \
    --num_queries 300 \
    --num_frames 5 \
    --with_box_refine \
    --masks \
    --rel_coord \
    --backbone swin_l_p4w12 \
    --pretrain_weights weights/swinL_weight.pth \
    --output_dir swin_joint \

