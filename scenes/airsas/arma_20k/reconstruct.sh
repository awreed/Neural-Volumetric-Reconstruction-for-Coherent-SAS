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
  --learning_rate 1e-3 `#network learning rate` \
  --num_trans_per_inr 360 `#number of measurements to fit with single network` \
  --number_iterations 1000 `#number of iterations to train each single network`\
  --info_every 999 `#how often to print`\
  --sparsity 1e-1 `#strength of sparsity regularization`\
  --load_wfm ../../../data/wfm/20khz_bw_lfm.npy `#path to the sonar transmit waveform`\
  --phase_loss 1e-4 `#strength of phase regularization`

echo "Starting neural backprojection (reconstruction step 2/2 of our method)"
python ../../../inr_reconstruction/reconstruct_scene.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements `#folder containing the deconvolved measurements (previous step)`\
  --system_data $1 `#.pik file containing the system data`\
  --output_dir ./nbp_output `#save directory`\
  --plot_thresh 2. `# linear plot threshold for matplotlib plots`\
  --learning_rate 1e-4 `# learning rate for the neural network`\
  --num_epochs 21000 `# num iterations to train network`\
  --num_rays 5000 `# number of rays to transmit `\
  --info_every 25 `# how often to print losses`\
  --scene_every 1000 `# how often to save the scene`\
  --accum_grad 5 `# accumulate the gradients across this many measurements before updating network`\
  --scale_factor 3e1 `# scale the measurements by this much before optimization`\
  --max_weights 200 `# number of ellipsoids to sample in the scene`\
  --use_up_to 120 `# number of vertical measurements to use`\
  --sampling_distribution_uniformity 1.0 `# setting this closer to 0. makes the waveform distribution more uniform which influences the range importance sampling`\
  --lambertian_ratio 0. `# 0 for total lambertian assumption. 1 for no lambertian assumption`\
  --occlusion `# model occlusion`\
  --occlusion_scale 5e2 `# scaling factor for occlusion cumulative product`\
  --num_layers 4 `# number of layers in neural network`\
  --num_neurons 128 `# number of layers`\
  --reg_start 500 `# don't start regularization until this iteration `\
  --thresh .15 `# don't sample time bins with amplitude less than this value`\
  --smooth_loss 5e1 `# smoothing regularization`\
  --smooth_delta 1.0 `# spatial gradient step size`\
  --sparsity 1e1 `# sparisty regularization`\
  --point_at_center `# point the transmitter and mic at the scene center`\
  --transmit_from_tx `# transmit rays from the transmitter (instead of the the phase center)`\
  --normalize_scene_dims `# normalize the scene dimensions before training the network `\
  --expname $2 `# the experiment name`\
  --beamwidth 30 `# beamwidth of the transmitter`\
  --phase_loss 1e-1 `# smoothness reg on the phase`

echo "Sampling network for scene exporting scene *.mat file"
python ../../../inr_reconstruction/upsample_network_with_input_args.py \
  --exp_name $2 `# experiment name` \
  --experiment_dir ./ `# experiment directory` \
  --inr_config ./nbp_config.json `# json file containing network configuration` \
  --output_scene_file_name final_upsampled_scene \
  --output_dir_name reconstructed_scenes