#!/bin/bash

python ../../../airsas/reconstruct_from_numpy.py \
  --input_config $1 \
  --output_config ./system_data_20db.pik \
  --output_dir ./bp \
  --gpu \
  --bin_upsample 20 `# upsample factor`\
  --wfm_bw 20 `# bandwidth of the transmit waveform`\
  --signal_snr 20 `#signal snr with respect to noise`\
  --wfm_part_1 $2 \
  #--fiveK \
  #--correction_term

