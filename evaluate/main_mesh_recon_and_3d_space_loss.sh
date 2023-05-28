#!/bin/bash

python evaluate/main_mesh_recon_and_3d_space_loss.py \
  --scene_inr_result_directory ../experiments/figure_10/scenes/scene_inr_result \
  --output_dir ../reconstructed_mesh \
  --system_data_path ../experiments/figure_10/scenes/configs/bunny_20k/10db/system_data_5k.pik \
  --expname bunny_20k_10db \
  --mesh_name bunny \
  --comp_albedo_path ../experiments/figure_10/scenes/scene_inr_result/bunny_20k_10db/comp_albedo100000.npy \
  --csv_file_name bunny \
  --gt_mesh_dir ../airsas_data/gt_meshes \
  --thresh 0.2
