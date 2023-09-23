#!/bin/bash

python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./deconvolve_config.json \
  --system_data $1 \
  --output_dir ./deconvolved_measurements \
  --learning_rate 1e-4 \
  --num_trans_per_inr 360 \
  --number_iterations 1000 \
  --info_every 999 \
  --sparsity 1e-4 \
  --phase_loss 0. \
  --clear_output_directory
    #--norm_kernel \
  #--phase_loss 1e-5
  #--clear_output_directory \