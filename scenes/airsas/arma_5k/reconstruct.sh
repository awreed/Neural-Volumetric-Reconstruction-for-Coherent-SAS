#!/bin/bash

echo "Starting reconstructing using backprojection (the conventional method)"
python ../../../airsas/reconstruct_from_system_file.py \
  --orig_system_file $1 \
  --output_dir ./reconstructed_scenes `#Save directory for backprojection` \
  --use_up_to 120 `#Use 120 vertical measurements` \
  --interpolation_factor 5 `#Upsample factor for nearest neighbor backprojection interpolation` \
  --gpu `#Run on GPU if possible` \

echo "Starting pulse deconvolution (reconstruction step 1/2 of our method)"

python ../../../inr_reconstruction/deconvolve_measurements.py \
  --inr_config ./pulse_deconvolve.json \
  --system_data $1 \
  --clear_output_dir \
  --output_dir ./deconvolved_measurements \
  --learning_rate 1e-3 \
  --num_trans_per_inr 360 \
  --number_iterations 1000 \
  --info_every 999 \
  --sparsity 1e-1 \
  --load_wfm ../../../data/wfm/5khz_bw_lfm.npy \
  --phase_loss 1e-4 \

echo "Starting neural backprojection (reconstruction step 2/2 of our method)"

python ../../../inr_reconstruction/reconstruct_scene.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements `#folder containing the deconvolved measurements (previous step)`\
  --system_data $1 `#.pik file containing the system data`\
  --output_dir ./nbp_output `#save directory`\
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
  --expname $2 `# the experiment name`\
  --no_reload \
  --beamwidth 30 \
  --phase_loss 1e-1 \

echo "Sampling network for scene exporting scene *.mat file"
python ../../../inr_reconstruction/upsample_network_with_input_args.py \
  --exp_name $2 `# experiment name` \
  --experiment_dir ./ `# experiment directory` \
  --inr_config ./nbp_config.json `# json file containing network configuration` \
  --output_scene_file_name final_upsampled_scene \
  --output_dir_name reconstructed_scenes