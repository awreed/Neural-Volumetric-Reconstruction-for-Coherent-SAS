#!/bin/bash

python ../../../airsas/reconstruct_from_system_file.py \
  --orig_system_file /data/sjayasur/awreed/system_data_arma_20k.pik \
  --output_dir ./bp \
  --use_up_to 120 \
  --interpolation_factor 5 \
  --gpu \




