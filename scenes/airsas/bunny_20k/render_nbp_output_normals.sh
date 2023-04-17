#!/bin/bash

python ../../../render_utils.py \
  --scene ./example_outputs/npb_output/bunny_20k_release/numpy/comp_albedo25000.npy \
  --normals ./example_outputs/npb_output/bunny_20k_release/numpy/normal25000.npy \
  --output_dir ./example_outputs/renders \
  --video_name bunny_20k_release_iter_25k_normals.avi \
  --image_name bunny_20k_release_iter_25k_normals.png \
  --render_x 800 \
  --render_y 800 \
  --thresh 0.2 \
  --x 150 \
  --y 150 \
  --z 120 \
  --azimuth 270 \
  --elevation 30 \
  --distance 200 \
  --make_video \
  --elevation_width 10 \
  --azimuth_min 90 \
  --azimuth_max 450 \
  --azimuth_step 10 \
  --drc .1 \
