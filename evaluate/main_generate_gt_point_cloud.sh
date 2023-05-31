#!/bin/bash

python evaluate/main_generate_gt_point_cloud.py \
  --mesh_file ../data/gt_meshes/budda.obj \
  --gt_n_points 20000 \
  --gt_n_points_volume 50000