#!/bin/bash

python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./pulse_deconvolve.json \
  --system_data $1 \
  --output_dir $2 \
  --learning_rate 1e-3 \
  --num_trans_per_inr 2292 \
  --number_iterations 5000 \
  --sparsity 2e-1 \
  --info_every 999 \
  --compare_with_mf \
  --subtract_dc \
  --phase_loss 3e-5 \
  --linear_plot \
  --drc_gt 1.0 \
  --drc_weights 0.5 \
  --clear_output_directory \