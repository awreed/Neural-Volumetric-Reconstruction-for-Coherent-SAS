#!/bin/bash

python ../../../render_utils.py \
  --scene /home/albert/SINR_Stuff/code_release/scenes/airsas/arma_20k/npb_output/arma_20k_release_2/numpy/comp_albedo20000.npy \
  --output_dir ./example_outputs/renders \
  --video_name arma_20k_release_iter_20k.avi \
  --image_name arma_20k_release_iter_20k.png \
  --render_x 800 \
  --render_y 800 \
  --thresh 0.2 \
  --x 150 \
  --y 150 \
  --z 120 \
  --azimuth 230 \
  --elevation 10 \
  --distance 200 \
  --make_video \
  --elevation_width 10 \
  --azimuth_min 90 \
  --azimuth_max 450 \
  --azimuth_step 10 \
  --drc .1 \
