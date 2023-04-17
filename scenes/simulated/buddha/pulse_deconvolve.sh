#!/bin/bash

python /home/awreed/SINR3D/inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./deconvolve_config.json \
  --system_data ./system_data_20db.pik \
  --output_dir ./deconvolved_measurements \
  --learning_rate 1e-4 \
  --num_trans_per_inr 360 \
  --number_iterations 1000 \
  --info_every 999 \
  --sparsity 1e-1 \
  --phase_loss 1e-4 \
  --norm_kernel \
  --clear_output_directory
  #--phase_loss 1e-5
  #--clear_output_directory \