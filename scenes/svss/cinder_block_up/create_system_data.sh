#!/bin/bash

python ../../serdp/create_serdp_system_data.py \
  --output_dir $2 \
  --root_path $1/asasinOutput/allElementAndMotions/COINv2_Imagery \
  --sound_speed_table $1/navData/soundSpeedTable.csv \
  --track_id '2019 1106 163841' \
  --clear_output_dir \
  --min_ping 10 \
  --max_ping 190 \
  --rx_min 1 \
  --rx_max 80 \
  --y_min 0.5 \
  --y_max 0.1 \
  --x_min 0.5 \
  --x_max 1.3 \
  --x_step .01 \
  --y_step .01 \
  --z_step .01 \
  --min_depth 0.85 \
  --max_depth 1.5 \
  --r 100 \
  --gpu \
  --save_images \
  --tx_list 0,1,2 \
  --drc_mf 0.6