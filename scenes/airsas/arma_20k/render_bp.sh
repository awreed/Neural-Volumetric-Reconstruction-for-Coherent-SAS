#!/bin/bash

python ../../../render_utils.py \
  --scene ./bp/scene.npy \
  --output_dir ./example_outputs/renders \
  --video_name bp.avi \
  --image_name bp.png \
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