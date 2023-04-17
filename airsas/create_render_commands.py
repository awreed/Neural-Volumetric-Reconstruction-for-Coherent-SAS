import argparse
import pickle
import constants as c
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")
    parser.add_argument('--inverse_config', required=True,
                        help='Configuaration pickle containing AirSAS config')
    parser.add_argument('--xml_name', required=True,
                        help="Name of renderer .xml file (path on the docker)")
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save code output')
    parser.add_argument('--samples', required=True,
                        default=10000)
    parser.add_argument('--bin_upsample', required=True, type=int,
                        default=20)

    args = parser.parse_args()

    with open(args.inverse_config, 'rb') as handle:
        system_data = pickle.load(handle)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tx_coords = system_data[c.TX_COORDS]
    rx_coords = system_data[c.RX_COORDS]
    temp = np.mean(system_data[c.TEMPS])
    speed_of_sound = 331.4 + 0.6 * temp
    fs = system_data[c.SYS_PARAMS][c.FS]

    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]

    t_res = speed_of_sound/system_data[c.SYS_PARAMS][c.FS]

    t_res = t_res / args.bin_upsample

    assert tx_coords.shape == rx_coords.shape

    output_file = os.path.join(args.output_dir, 'simulate_airsas.sh')

    with open(output_file, 'w') as f:
        f.write('#!/bin/bash\n\n')

        for i in range(tx_coords.shape[0]):
            f.write(''.join(['mitsuba ', args.xml_name, ' -o ', str(i), ' -D ', 'samples=', str(args.samples),
                             " -D decomposition=transient", " -D ", "tMin=", str(wfm_crop_settings[c.MIN_DIST]),
                             " -D tMax=", str(wfm_crop_settings[c.MAX_DIST]),
                             " -D tRes=", str(t_res), " -D modulation=None", ' -D lambda=0', ' -D phase=0',
                             " -D tx_x=", str(tx_coords[i, 0]), " -D tx_y=", str(tx_coords[i, 1]),
                             " -D tx_z=", str(tx_coords[i, 2]), " -D rx_x=", str(rx_coords[i, 0]),
                             " -D rx_y=", str(rx_coords[i, 1]), " -D rx_z=", str(rx_coords[i, 2]), "\n\n"]))

    exr2mat_script = os.path.join(args.output_dir, 'exr2mat_script.sh')

    with open(exr2mat_script, 'w') as f:
        f.write('#!/bin/bash\n\n')

        for i in range(tx_coords.shape[0]):
            f.write(''.join(['python ../exr2mat.py ', str(i) + '.exr ', str(i), '\n\n']))


    # Need to redefine the number of min and max samples




