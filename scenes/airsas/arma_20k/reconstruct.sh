#!/bin/bash

echo $1
echo $2

echo "Starting reconstructing using backprojection (the conventional method)"
python ../../../airsas/reconstruct_from_system_file.py \
  --orig_system_file $1 \
  --output_dir ./reconstructed_scenes  \
  --use_up_to 120 \
  --interpolation_factor 5  \
  --gpu  \


echo "Starting pulse deconvolution (reconstruction step 1/2 of our method)"
python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./pulse_deconvolve.json \
  --system_data $1 \
  --clear_output_dir \
  --output_dir ./deconvolved_measurements \
  --learning_rate 1e-3  \
  --num_trans_per_inr 360  \
  --number_iterations 1000 \
  --info_every 999 \
  --sparsity 1e-1 \
  --load_wfm ../../../data/wfm/20khz_bw_lfm.npy \
  --phase_loss 1e-4

echo "Starting neural backprojection (reconstruction step 2/2 of our method)"
python ../../../inr_reconstruction/reconstruct_scene.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements \
  --system_data $1 \
  --output_dir ./nbp_output \
  --plot_thresh 2. \
  --learning_rate 1e-4 \
  --num_epochs 21000 \
  --num_rays 5000 \
  --info_every 25 \
  --scene_every 1000 \
  --accum_grad 5 \
  --scale_factor 3e1 \
  --max_weights 110 \
  --use_up_to 120 \
  --sampling_distribution_uniformity 1.0 \
  --lambertian_ratio 0. \
  --occlusion \
  --occlusion_scale 5e2 \
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
  --expname $2 \
  --beamwidth 30 \
  --phase_loss 1e-1  \
  --max_voxels 15000

echo "Sampling network for scene exporting scene *.mat file"
python ../../../inr_reconstruction/upsample_network_with_input_args.py \
  --exp_name $2 \
  --experiment_dir ./ \
  --inr_config ./nbp_config.json \
  --output_scene_file_name final_upsampled_scene \
  --output_dir_name reconstructed_scenes \
  --system_data $1 \
  --normalize_scene_dims \
  --sf 2 \
  --max_voxels 15000