#!/bin/bash

python main_mesh_recon_and_3d_space_loss.py \
  --scene_inr_result_directory ../experiments/scene_inr_result \
  --output_dir ../reconstructed_mesh \
  --system_data_path ../scenes/simulated/buddha/system_data_20db.pik \
  --expname buddha_20k_20db \
  --mesh_name buddha \
  --comp_albedo_path ../scenes/simulated/buddha/numpy/comp_albedo100000.npy \
  --csv_file_name buddha \
  --gt_mesh_dir ../data/gt_meshes \
  --thresh 0.2
