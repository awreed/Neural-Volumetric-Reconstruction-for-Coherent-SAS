#!/bin/bash

python evaluate/main_render.py \
  --n_azimuth 10 \
  --mesh_name buddha \
  --expname buddha_20k_20db \
  --render_output_dir ../render_output \
  --recon_mesh_dir ../reconstructed_mesh \
  --thresh 0.2 \
  --elevation 0.1 \
  --distance 0.3 \
  --fov 60