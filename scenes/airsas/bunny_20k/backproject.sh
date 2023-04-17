#!/bin/bash

python ../../../airsas/reconstruct_from_system_file.py \
  --orig_system_file ./system_data.pik \
  --output_dir ./example_outputs/bp \
  --use_up_to 120 \
  --interpolation_factor 5 \
  --gpu \




