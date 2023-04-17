#!/bin/bash

python /home/awreed/SINR3D/airsas/reconstruct_from_numpy.py \
  --input_config /data/sjayasur/awreed/system_data.pik \
  --output_config ./system_data_20db.pik \
  --output_dir ./bp \
  --gpu \
  --bin_upsample 20 \
  --wfm_bw 20 \
  --signal_snr 20 \
  --wfm_part_1 /data/sjayasur/awreed/airsas_data/sim_buddha/data_full.npy \
  #--fiveK \
  #--correction_term

