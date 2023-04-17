import argparse
from utils import process_folder, load_img_and_preprocess
import os
import numpy as np
from geometry import create_voxels
import constants as c
from sas_utils import gen_real_lfm, crop_wfm, kernel_from_waveform, delay_waveforms, correct_group_delay, no_rc_kernel_from_waveform
import matplotlib.pyplot as plt
import torch
from beamformer import backproject_measurements
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")
    parser.add_argument('--from_data_folder', required=False, default=None,
                        help='Setup AirSAS simulation using geometry and parameters found in'
                             'an AirSAS data folder')
    parser.add_argument('--from_config', required=False, default=None,
                        help='Setup AirSAS simulation using geometry and parameters using'
                             'specificiations found in a .json file')
    parser.add_argument('--gt_img', required=True,
                        help='Path to ground truth scattering image')
    parser.add_argument('--use_wfm_cache', required=False, default=False, action='store_true',
                        help='Attempt to load cached flights (code will run faster)')
    parser.add_argument('--use_coords_cache', required=False, default=False, action='store_true',
                        help='Attempt to load cached coordinates (code will run faster)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save code output')
    parser.add_argument('--use_measured_wfm', required=False, default=False, action='store_true',
                        help='Try to use measured waveform instead of analytic')
    parser.add_argument('--x_min', type=float, required=False, default=-0.2,
                        help='x min bound for scene in (m)')
    parser.add_argument('--x_max', type=float, required=False, default=0.2,
                        help='x max for scene in (m)')
    parser.add_argument('--y_min', type=float, required=False, default=-0.2,
                        help='y min for scene in (m)')
    parser.add_argument('--y_max', type=float, required=False, default=0.2,
                        help='y max for scene in (m)')
    parser.add_argument('--num_x', type=int, required=False, default=100,
                        help='Number of voxels in the x direction')
    parser.add_argument('--num_y', type=int, required=False, default=100,
                        help='Number of voxels in the y direction')
    parser.add_argument('--generate_inverse_config', required=False, action='store_true',
                        help='Generate config file that can be used in inr_reconstruction scripts')
    parser.add_argument('--interpolation_factor', type=int, required=False, default=100)
    parser.add_argument('--incoherent', action='store_true', required=False, help='Whether to beamform incoherently')
    #parser.add_argument('--group_delay', type=float, required=False, default=None,
    #                    help="Overrwrite the AirSAS measured group delay to something else for testing purposes")

    args = parser.parse_args()

    assert torch.cuda.is_available()
    dev = 'cuda:0'

    if args.from_data_folder is not None:
        assert args.from_config is None, "Cannot provide both a data folder and config folder"
    elif args.from_config is not None:
        assert args.from_data_folder is None, "Cannot provide both a data folder and config folder"
    else:
        raise IOError("Must provide either data folder or .json file")

    if args.from_config is not None:
        raise OSError("Not supported yet.")

    airsas_data = process_folder(args.from_data_folder, args.use_wfm_cache, args.use_coords_cache)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Use the average temperature
    temp = np.mean(airsas_data[c.TEMPS])
    speed_of_sound = 331.4 + 0.6 * temp

    geometry = create_voxels(args.x_min, args.x_max,
                             args.y_min, args.y_max,
                             0., 0.,
                             args.num_x, args.num_y, 1)


    if args.use_measured_wfm:
        wfm = airsas_data[c.WFM]
    else:
        wfm = gen_real_lfm(airsas_data[c.SYS_PARAMS][c.FS],
                           airsas_data[c.WFM_PARAMS][c.F_START],
                           airsas_data[c.WFM_PARAMS][c.F_STOP],
                           airsas_data[c.WFM_PARAMS][c.T_DUR],
                           window=True,
                           win_ratio=airsas_data[c.WFM_PARAMS][c.WIN_RATIO])
        # Overwrite with analytical waveform
        airsas_data[c.WFM] = wfm

    wfm_length = airsas_data[c.WFM_PARAMS][c.T_DUR] * airsas_data[c.SYS_PARAMS][c.FS]
    wfm_crop_settings = crop_wfm(airsas_data[c.TX_COORDS],
                                 airsas_data[c.RX_COORDS],
                                 geometry[c.CORNERS],
                                 wfm_length,
                                 airsas_data[c.SYS_PARAMS][c.FS],
                                 speed_of_sound)

    gt_img = load_img_and_preprocess(args.gt_img, size=(args.num_x, args.num_y))

    # Save so we can compute performance metrics in downstream tasks
    airsas_data[c.GT_IMG] = gt_img

    plt.figure()
    plt.imshow(gt_img)
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'gt_img.png'))

    gt_img = torch.from_numpy(gt_img).to(dev)

    kernel = kernel_from_waveform(airsas_data[c.WFM], wfm_crop_settings[c.NUM_SAMPLES])
    kernel = kernel.to(gt_img.device)

    kernel_no_rc = no_rc_kernel_from_waveform(airsas_data[c.WFM], wfm_crop_settings[c.NUM_SAMPLES])
    kernel_no_rc = kernel_no_rc.to(gt_img.device)

    # Overwrite the airsas group delay (probably to zero)
    #if args.group_delay is not None:
    #    airsas_data[c.SYS_PARAMS][c.GROUP_DELAY] = args.group_delay

    print("Delaying waveforms")
    data, data_rc = delay_waveforms(tx_pos=torch.from_numpy(airsas_data[c.TX_COORDS]).to(dev),
                              rx_pos=torch.from_numpy(airsas_data[c.RX_COORDS]).to(dev),
                              weights=gt_img.reshape(-1),
                              voxels=torch.from_numpy(geometry[c.VOXELS]).to(dev),
                              kernel=kernel,
                              kernel_no_rc=kernel_no_rc,
                              min_dist=wfm_crop_settings[c.MIN_DIST],
                              group_delay=0.,
                              fs=airsas_data[c.SYS_PARAMS][c.FS],
                              speed_of_sound=speed_of_sound)


    if args.generate_inverse_config:
        airsas_data[c.WFM_DATA] = data.detach().cpu().numpy()
        airsas_data[c.WFM_RC] = data_rc.detach().cpu().numpy()
        airsas_data[c.GEOMETRY] = geometry
        airsas_data[c.WFM_CROP_SETTINGS] = wfm_crop_settings
        with open(os.path.join(args.output_dir, 'system_data.pik'), 'wb') as handle:
            print("Saving system data to ", os.path.join(args.output_dir, 'system_data.pik'))
            pickle.dump(airsas_data, handle)

    if args.incoherent:
        data_rc = data_rc.abs()

    scene = backproject_measurements(
                              airsas_data[c.TX_COORDS],
                              airsas_data[c.RX_COORDS],
                              geometry[c.VOXELS],
                              data_rc.detach().cpu().numpy(),
                              speed_of_sound,
                              min_dist=wfm_crop_settings[c.MIN_DIST],
                              group_delay=0.,
                              r=args.interpolation_factor,
                              fs=airsas_data[c.SYS_PARAMS][c.FS],
                              basebanded=False)

    #scene = scene.detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, c.BF_FILE), scene)

    scene = scene.reshape((args.num_x, args.num_y))

    plt.figure()
    plt.imshow(np.abs(scene))
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'bf_2d_center_xy_abs' + '.png'))

    plt.figure()
    plt.imshow(np.real(scene))
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'bf_2d_center_xy_real' + '.png'))

    plt.figure()
    plt.imshow(np.imag(scene))
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, 'bf_2d_center_imag' + '.png'))
