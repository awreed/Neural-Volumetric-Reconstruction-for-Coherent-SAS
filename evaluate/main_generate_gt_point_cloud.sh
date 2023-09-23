#!/bin/bash

python main_generate_gt_point_cloud.py \
  --mesh_file $1 \
  --gt_n_points 20000 \
  --gt_n_points_volume 1000 `# paper uses 50000`