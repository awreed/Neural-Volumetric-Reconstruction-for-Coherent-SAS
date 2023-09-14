#!/bin/bash

python ../../../airsas/reconstruct_airsas_scene.py \
  --data_folder $1 `#Folder containing AirSAS data of a scene` \
  --background_folder $2 `#Folder containing AirSAS data of a background (can remove if you don't have this)` \
  --output_dir $3 `#Directory to save outputs` \
  --x_min -0.125 `#Min x coordinate (meters)`\
  --x_max 0.125 `#Max x coordinate (meters)`\
  --y_min -0.125 `#Min y coordinate (meters)`\
  --y_max 0.125 `#Max y coordinate (meters)`\
  --z_min 0.00 `#Min z coordinate (meters)`\
  --z_max 0.2 `#Max z coordinate (meters)`\
  --num_x 150 `#Number of x voxels`\
  --num_y 150 `#Number of y voxels`\
  --num_z 120 `#Number of z voxels`\
  --interpolation_factor 5 `#Upsample factor for beamforming of y voxels`\
  --generate_inverse_config `#This command creates the system_data.pik`\
  --gpu `#Run on GPU`\
  --save3D `#Save matplotlib plots`\
  --read_only_wfm 1 `#0 for the 20khz scene, 1 for the 5 khz scene`\
  --load_wfm ../../../data/wfm/5khz_bw_lfm.npy `#Path to a numpy file of the transmit waveform`\