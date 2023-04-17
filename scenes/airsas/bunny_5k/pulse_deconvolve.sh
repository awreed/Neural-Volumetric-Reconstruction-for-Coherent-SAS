#!/bin/bash

python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./pulse_deconvolve.json \
  --system_data ./system_data.pik \
  --clear_output_dir \
  --output_dir ./deconvolved_measurements \
  --learning_rate 1e-3 \
  --num_trans_per_inr 360 \
  --number_iterations 1000 \
  --info_every 999 \
  --sparsity 1e-2 \
  --load_wfm ../../../data/wfm/5khz_bw_lfm.npy \
  --phase_loss 1e-4 \

  #--tv_loss 1e-1 \
  #--phase_loss 0. \
  #--num_radial 1000 \
# 50-60 9e-4, 60-70 7e-4