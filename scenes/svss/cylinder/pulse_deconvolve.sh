#!/bin/bash

python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./pulse_deconvolve.json \
  --system_data $1 \
  --output_dir $2 \
  --learning_rate 1e-3 \
  --num_trans_per_inr 600 \
  --number_iterations 1000 \
  --sparsity 2e-2 \
  --info_every 999 \
  --compare_with_mf \
  --subtract_dc \
  --phase_loss 3e-5 \
  --linear_plot \
  --drc_gt 1.0 \
  --drc_weights 0.6 \
  --clear_output_directory \