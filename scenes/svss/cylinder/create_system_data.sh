#!/bin/bash

python ../../../svss/create_serdp_system_data.py \
  --output_dir $2 \
  --root_path $1/asasinOutput/allElementAndMotions/COINv2_Imagery \
  --sound_speed_table $1/navData/soundSpeedTable.csv \
  --track_id '2019 1106 163841' \
  --image_number 1 \
  --clear_output_dir \
  --r 100 \
  --gpu \
  --save_images \
  --tx_list 0,1,2,3,4 \
  --min_ping 0 \
  --max_ping 200 \
  --min_depth .9 \
  --max_depth 2.0 \
  --x_min 0.25 \
  --x_max 1.75 \
  --y_min 0.3 \
  --y_max 0.8 \
  --drc_mf 0.6