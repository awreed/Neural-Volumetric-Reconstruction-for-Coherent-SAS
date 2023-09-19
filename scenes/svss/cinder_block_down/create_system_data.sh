#!/bin/bash

python ../../serdp/create_serdp_system_data.py \
  --output_dir $2 \
  --root_path $1/asasinOutput/allElementAndMotions/COINv2_Imagery \
  --sound_speed_table $1/navData/soundSpeedTable.csv \
  --track_id '2019 0614 154036' \
  --min_ping 570 \
  --max_ping 760 \
  --rx_min 0 \
  --rx_max 80 \
  --x_min 5.23 \
  --x_max 6.41 \
  --y_min .5 \
  --y_max .3 \
  --min_depth .95 \
  --max_depth 1.7 \
  --image_number 2 \
  --clear_output_dir \
  --r 100 \
  --gpu \
  --save_images \
  --tx_list 0,1,2,3,4 \