#!/bin/bash

python /home/awreed/SINR3D/inr_reconstruction/reconstruct_scene.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements \
  --system_data ./system_data_20db.pik \
  --output_dir ./nbp_output \
  --plot_thresh 2. \
  --learning_rate 1e-4 \
  --num_epochs 100001 \
  --num_rays 5000 \
  --info_every 25 \
  --scene_every 5000 \
  --accum_grad 5 \
  --scale_factor 1e1 \
  --max_weights 110 \
  --use_up_to 150 \
  --sampling_distribution_uniformity 1.0 \
  --lambertion_ratio 0. \
  --occlusion \
  --occlusion_scale 5e2 \
  --num_layers 4 \
  --num_neurons 128 \
  --point_at_center \
  --reg_start 500 \
  --transmit_from_tx \
  --smooth_delta 1.0 \
  --sparsity 1e0 \
  --thresh .02 \
  --expname sim_buddha_20db_20k \
  --normalize_scene_dims \
  --no_reload \
  --importance_sampling_rays 500
  #--beamwidth 30 \
  #--smooth_loss 1e3 \
  #--importance_sampling_rays 500 \
  #--no_network \
  #--skip_normals
  #--beamwidth 30
  #--normalize_scene_dims \
  #--beamwidth 30 \
  #--real_only
  #--k_normal_per_ray 1
  #--importance_sampling_rays 500 \
  #--tv_loss 1e3
  #--norm_weights \
  #--phase_loss 1e0
  #--skip_normals
  #--k_normal_per_ray 1
  #--skip_normals \
  #--norm_weights
  #--ray_trace_return \
  #--beamwidth 30 \
  #--use_mf_wfms
  #--norm_weights
