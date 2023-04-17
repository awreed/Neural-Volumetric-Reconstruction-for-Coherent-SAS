import argparse
import os
from tqdm import tqdm
import constants as c
from geometry import create_voxels, plot_geometry
from sas_utils import remove_room, gen_real_lfm, match_filter_all, crop_wfm, correct_group_delay, view_fft, baseband_signal, modulate_signal
import matplotlib.pyplot as plt
from beamformer import backproject_all_airsas_measurements
import numpy as np
import pickle
from utils import process_folder
import torch
import scipy
import scipy.io
import scipy.signal
from argument_io import directory_cleaup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")

    parser.add_argument('--data_folder', required=True,
                        help='Input data folder containing AirSAS data')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save code output')
    parser.add_argument('--clear_output_directory', action='store_true', default=False)
    parser.add_argument('--use_wfm_cache', required=False, default=False, action='store_true',
                        help='Attempt to load cached flights (code will run faster)')
    parser.add_argument('--use_coords_cache', required=False, default=False, action='store_true',
                        help='Attempt to load cached coordinates (code will run faster)')
    parser.add_argument('--use_bf_cache', required=False, default=False, action='store_true',
                        help='Attempt to load beamformed scene from cache (if only wanting to plot)')
    parser.add_argument('--plot_geometry', required=False, default=False, action='store_true',
                        help='Plot the TX/RX and scene geometry')
    parser.add_argument('--use_measured_wfm', required=False, default=False, action='store_true',
                        help='Try to use measured waveform instead of analytic')
    parser.add_argument('--generate_inverse_config', required=False, action='store_true',
                        help='Generate config file that can be used in inr_reconstruction scripts')
    parser.add_argument('--x_min', type=float, required=False, default=-0.2,
                        help='x min bound for scene in (m)')
    parser.add_argument('--x_max', type=float, required=False, default=0.2,
                        help='x max for scene in (m)')
    parser.add_argument('--y_min', type=float, required=False, default=-0.2,
                        help='y min for scene in (m)')
    parser.add_argument('--y_max', type=float, required=False, default=0.2,
                        help='y max for scene in (m)')
    parser.add_argument('--z_min', type=float, required=False, default=0.0,
                        help='z min for scene in (m)')
    parser.add_argument('--z_max', type=float, required=False, default=0.5,
                        help='z max for scene in (m)')
    parser.add_argument('--num_x', type=int, required=False, default=100,
                        help='Number of voxels in the x direction')
    parser.add_argument('--num_y', type=int, required=False, default=100,
                        help='Number of voxels in the y direction')
    parser.add_argument('--num_z', type=int, required=False, default=100,
                        help='Number of voxels in the z direction')
    parser.add_argument('--bf_up_to', type=int, required=False, default=None,
                        help='Only beamform a subset of transducers specified by index')
    parser.add_argument('--interpolation_factor', type=int, required=False, default=100)
    parser.add_argument('--background_folder', type=str, required=False, default=None,
                        help="AirSAS folder of background so we can perform background subtraction")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Attempt to use GPU")
    parser.add_argument('--voxels_within', type=float, default=None, required=False,
                        help="use voxels within beamwidth")
    parser.add_argument('--incoherent', action='store_true', required=False, help='Whether to beamform incoherently')
    parser.add_argument('--save3D', action='store_true', default=False, help="Whether to store 3D plots")
    parser.add_argument('--remove_room_per_k', type=int, default=None)
    parser.add_argument('--filter_path', required=False, default=None)
    parser.add_argument('--read_only_wfm', required=False, default=None, type=int, help="If multiple waveforms"
                                                                                        "were used to scan the scene"
                                                                                        "and we want to only use one")
    parser.add_argument('--bg_read_only_wfm', required=False, default=None, type=int, help="read background waveform index")
    parser.add_argument('--load_wfm', required=False, default=None, help="Load measured waveform")
    parser.add_argument('--baseband', required=False, action='store_true', default=False, help="Whether to baseband data. ")
    parser.add_argument('--geometry_only', required=False, action='store_true', default=False,
                        help="Whether to only save scene geometry ")

    args = parser.parse_args()
    device = 'cpu'

    if args.read_only_wfm is not None:
        assert args.bg_read_only_wfm is None

    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:

            print("Did not find GPU --- using CPU")

    if args.num_z == 0:
        args.num_z = 1

    airsas_data = process_folder(args.data_folder, args.use_wfm_cache, args.use_coords_cache, args.read_only_wfm)

    temp = np.mean(airsas_data[c.TEMPS])
    airsas_data[c.SOUND_SPEED] = 331.4 + 0.6 * temp

    read_only_wfm = args.read_only_wfm
    if args.bg_read_only_wfm is not None:
        read_only_wfm = args.bg_read_only_wfm

    if args.background_folder is not None:
        print("Subtracting background...")
        bg_data = process_folder(args.background_folder, False, False, read_only_wfm)
        airsas_data[c.WFM_DATA] = airsas_data[c.WFM_DATA] - bg_data[c.WFM_DATA][:airsas_data[c.WFM_DATA].shape[0]]
        del bg_data

    # Use the average temperature
    temp = np.mean(airsas_data[c.TEMPS])
    speed_of_sound = 331.4 + 0.6 * temp

    print("Speed of sound", speed_of_sound)

    directory_cleaup(args.output_dir, args.clear_output_directory)

    for i in tqdm(range(airsas_data[c.WFM_DATA].shape[0]), desc="Accounting for group delay..."):
        airsas_data[c.WFM_DATA][i, :] = correct_group_delay(wfm=airsas_data[c.WFM_DATA][i, :],
                                                            gd=airsas_data[c.SYS_PARAMS][c.GROUP_DELAY],
                                                            fs=airsas_data[c.SYS_PARAMS][c.FS])

    if args.remove_room_per_k is not None:
        assert airsas_data[c.WFM_DATA].shape[0] % args.remove_room_per_k == 0, "Must be able to divide k into the " \
                                                                               "number of waveforms (" + \
                                                                            str(airsas_data[c.WFM_DATA].shape[0]) + ")"

        for i in range(0, airsas_data[c.WFM_DATA].shape[0], args.remove_room_per_k):
            print("i", i, "stop", i + args.remove_room_per_k)
            airsas_data[c.WFM_DATA][i:i+args.remove_room_per_k] = \
                airsas_data[c.WFM_DATA][i:i+args.remove_room_per_k] - \
                np.mean(airsas_data[c.WFM_DATA][i:i+args.remove_room_per_k], 0, keepdims=True)
    else:
        print("Not subtracting room...")

    if args.filter_path is not None:
        print("Filtering using coefficients in ", args.filter_path)
        view_fft(airsas_data[c.WFM_DATA][0, :], airsas_data[c.SYS_PARAMS][c.FS])
        b = scipy.io.loadmat(args.filter_path)['b'].squeeze()
        airsas_data[c.WFM_DATA] = scipy.signal.lfilter(b, 1, airsas_data[c.WFM_DATA])
        view_fft(airsas_data[c.WFM_DATA][0, :], airsas_data[c.SYS_PARAMS][c.FS])
        # account for filter delay
        airsas_data[c.WFM_DATA] = np.roll(airsas_data[c.WFM_DATA], -(b.shape[0]-1)//2, 1)

    geometry = create_voxels(args.x_min, args.x_max,
                             args.y_min, args.y_max,
                             args.z_min, args.z_max,
                             args.num_x, args.num_y, args.num_z)

    if args.plot_geometry:
        plot_geometry(airsas_data[c.TX_COORDS], geometry[c.CORNERS],
                      args.output_dir)

    if args.load_wfm is not None:
        assert args.use_measured_wfm is False

    if args.load_wfm is not None:
        print("Loading waveform")
        wfm = np.load(args.load_wfm)[0:100]
        airsas_data[c.WFM] = wfm.squeeze()
    elif args.use_measured_wfm:
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

    print("Computing match filtered waveforms from scratch...")
    data_rc = match_filter_all(airsas_data[c.WFM_DATA], wfm)
    fc = (airsas_data[c.WFM_PARAMS][c.F_START] + airsas_data[c.WFM_PARAMS][c.F_STOP]) / 2

    airsas_data[c.SYS_PARAMS][c.FC] = fc

    if args.baseband:
        print("Basebanding RC...")

        data_rc = baseband_signal(data_rc, fs=airsas_data[c.SYS_PARAMS][c.FS], fc=fc)

        # baseband the data
        airsas_data[c.WFM_DATA] = baseband_signal(airsas_data[c.WFM_DATA], fs=airsas_data[c.SYS_PARAMS][c.FS], fc=fc)

        # baseband the waveform
        airsas_data[c.WFM] = baseband_signal(airsas_data[c.WFM], fs=airsas_data[c.SYS_PARAMS][c.FS], fc=fc)


    wfm_length = airsas_data[c.WFM_PARAMS][c.T_DUR]*airsas_data[c.SYS_PARAMS][c.FS]
    wfm_crop_settings = crop_wfm(airsas_data[c.TX_COORDS],
                                 airsas_data[c.RX_COORDS],
                                 geometry[c.CORNERS],
                                 wfm_length,
                                 airsas_data[c.SYS_PARAMS][c.FS],
                                 speed_of_sound)
    if args.geometry_only:
        assert args.generate_inverse_config is True

    if args.generate_inverse_config:
        airsas_data[c.WFM_RC] = data_rc
        airsas_data[c.GEOMETRY] = geometry
        airsas_data[c.WFM_CROP_SETTINGS] = wfm_crop_settings

        if args.geometry_only:
            airsas_data[c.WFM_RC] = None
            airsas_data[c.WFM] = None
            airsas_data[c.WFM_DATA] = None

        with open(os.path.join(args.output_dir, 'system_data.pik'), 'wb') as handle:
            print("Saving system data to ", os.path.join(args.output_dir, 'system_data.pik'))
            pickle.dump(airsas_data, handle)

        if args.geometry_only:
            exit(0)

    # Crop waveforms down
    _, before = data_rc.shape
    data_rc = \
        data_rc[:,
        wfm_crop_settings[c.MIN_SAMPLE]:wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]
    print("Wfms cropped from", before, "to", data_rc.shape[1])

    if args.bf_up_to is not None:
        airsas_data[c.TX_COORDS] = airsas_data[c.TX_COORDS][:args.bf_up_to, :]
        airsas_data[c.RX_COORDS] = airsas_data[c.RX_COORDS][:args.bf_up_to, :]
        airsas_data[c.WFM_DATA] = airsas_data[c.WFM_DATA][:args.bf_up_to, :]
        data_rc = data_rc[:args.bf_up_to, :]

    if args.use_bf_cache and os.path.exists(os.path.join(args.data_folder, c.BF_FILE)):
        print("Loading backprojected scene from ", os.path.join(args.data_folder, c.BF_FILE))
        scene = np.load(os.path.join(args.data_folder, c.BF_FILE))
    else:
        print("Backprojecting scene from scratch...")

        if args.incoherent:
            data_rc = np.abs(data_rc)

        scene = backproject_all_airsas_measurements(
                                                    tx_pos=airsas_data[c.TX_COORDS],
                                                    rx_pos=airsas_data[c.RX_COORDS],
                                                    voxels=geometry[c.VOXELS],
                                                    measurements=data_rc,
                                                    speed_of_sound=speed_of_sound,
                                                    min_dist=wfm_crop_settings[c.MIN_DIST],
                                                    group_delay=0.,
                                                    r=args.interpolation_factor,
                                                    fs=airsas_data[c.SYS_PARAMS][c.FS],
                                                    basebanded=False,
                                                    fc=fc,
                                                    device=device)

        np.save(os.path.join(args.output_dir, c.NUMPY, c.BF_FILE), scene)

    try:
        scene = scene.reshape((args.num_y, args.num_x, args.num_z))
    except ValueError:
        print("Size mismatch between cached beamformed scene and specified num_x, num_y, num_z. Try"
              " removing the --use_bf_cache flag and running again.")
        exit(0)

    for slice in range(args.num_z):
        # hello
        plt.figure()
        plt.imshow(np.abs(scene)[..., slice])
        plt.colorbar()
        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_xy_abs_' + str(slice) + '.png'))

        plt.figure()
        plt.imshow(np.real(scene)[..., slice])
        plt.colorbar()
        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_xy_real_' + str(slice) + '.png'))

        plt.figure()
        plt.imshow(np.imag(scene)[..., slice])
        plt.colorbar()
        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_imag_' + str(slice) + '.png'))

    if args.save3D:
        mag = np.abs(scene)
        mag = mag.ravel()

        u = mag.mean()
        var = mag.std()
        vals = np.arange(0., 10, 0.5)
        thresh_vals = u + vals*var

        for thresh_count in tqdm(range(0, len(thresh_vals)), desc="Saving 3D plots"):
            mag[mag[:] < thresh_vals[thresh_count]] = None

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()
            im = ax.scatter(geometry[c.VOXELS][:, 0],
                       geometry[c.VOXELS][:, 1],
                       geometry[c.VOXELS][:, 2],
                       c=mag, alpha=0.5)
            ax.set_xlim3d((geometry[c.CORNERS][:, 0].min(), geometry[c.CORNERS][:, 0].max()))
            ax.set_ylim3d((geometry[c.CORNERS][:, 1].min(), geometry[c.CORNERS][:, 1].max()))
            ax.set_zlim3d((geometry[c.CORNERS][:, 2].min(), geometry[c.CORNERS][:, 2].max()))
            plt.grid(True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.colorbar(im)
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_3d_thresh_' + str(thresh_count) + '.png'))

        mag = np.log10(np.abs(scene))
        mag = mag.ravel()

        u = mag.mean()
        var = mag.std()
        vals = np.arange(0., 10, 0.5)
        thresh_vals = u + vals * var

        for thresh_count in tqdm(range(0, len(thresh_vals)), desc="Saving 3D plots"):
            mag[mag[:] < thresh_vals[thresh_count]] = None

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()
            im = ax.scatter(geometry[c.VOXELS][:, 0],
                       geometry[c.VOXELS][:, 1],
                       geometry[c.VOXELS][:, 2],
                       c=mag, alpha=0.5)
            ax.set_xlim3d((geometry[c.CORNERS][:, 0].min(), geometry[c.CORNERS][:, 0].max()))
            ax.set_ylim3d((geometry[c.CORNERS][:, 1].min(), geometry[c.CORNERS][:, 1].max()))
            ax.set_zlim3d((geometry[c.CORNERS][:, 2].min(), geometry[c.CORNERS][:, 2].max()))
            plt.grid(True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.colorbar(im)
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_3d_log_thresh_' + str(thresh_count) + '.png'))

