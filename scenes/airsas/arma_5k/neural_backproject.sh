#!/bin/bash

python ../../../inr_reconstruction/reconstruct_scene.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements \
  --system_data ./system_data.pik \
  --output_dir ./npb_output \
  --plot_thresh 2. \
  --learning_rate 1e-4 \
  --num_epochs 26000 \
  --num_rays 5000 \
  --info_every 25 \
  --scene_every 5000 \
  --accum_grad 5 \
  --scale_factor 3e1 \
  --max_weights 200 \
  --use_up_to 120 \
  --sampling_distribution_uniformity 1.0 \
  --lambertian_ratio 0. \
  --occlusion \
  --occlusion_scale 1e3 \
  --num_layers 4 \
  --num_neurons 128 \
  --reg_start 500 \
  --thresh .15 \
  --smooth_loss 5e1 \
  --smooth_delta 1.0 \
  --sparsity 1e1 \
  --point_at_center \
  --transmit_from_tx \
  --normalize_scene_dims \
  --expname arma_20k_release \
  --no_reload \
  --beamwidth 30 \
  --phase_loss 1e-1 \


