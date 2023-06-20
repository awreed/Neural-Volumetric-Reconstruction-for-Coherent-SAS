#!/bin/bash

python main_image_space_loss.py \
  --n_azimuth 10 \
  --mesh_name buddha \
  --expname buddha_20k_20db \
  --image_dir ../render_output/buddha \
  --gt_image_dir ../render_output/gt_image/buddha \
  --thresh 0.2
