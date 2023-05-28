#!/bin/bash

python evaluate/main_image_space_loss.py \
  --n_azimuth 10 \
  --mesh_name bunny \
  --expname bunny_20k_10db \
  --image_dir ../render_output/bunny \
  --gt_image_dir ../render_output/gt_image/bunny \
  --thresh 0.2
