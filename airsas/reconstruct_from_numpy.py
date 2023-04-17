import argparse
import pickle
import constants as c
import os
from utils import resample_cylindrical
import glob
import scipy.io
import matplotlib.pyplot as plt
from sas_utils import kernel_from_waveform, gen_real_lfm, crop_wfm, no_rc_kernel_from_waveform, match_filter_all
import numpy as np
from beamformer import backproject_all_airsas_measurements
import torch
from sas_utils import interpfft
from tqdm import tqdm

def create_voxels(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z):
    scene_corners = np.array(([x_min, y_min, z_min],
                              [x_min, y_max, z_min],
                              [x_min, y_max, z_max],
                              [x_min, y_min, z_max],
                              [x_max, y_min, z_min],
                              [x_max, y_max, z_min],
                              [x_max, y_max, z_max],
                              [x_max, y_min, z_max]))

    x_vect = np.linspace(x_min, x_max, num_x, endpoint=True)
    y_vect = np.linspace(y_min, y_max, num_y, endpoint=True)
    z_vect = np.linspace(z_min, z_max, num_z, endpoint=True)

    (x, y, z) = np.meshgrid(x_vect, y_vect, z_vect)

    voxels = np.hstack((np.reshape(x, (np.size(x), 1)),
                        np.reshape(y, (np.size(y), 1)),
                        np.reshape(z, (np.size(z), 1))
                       ))

    return {
        c.CORNERS: scene_corners,
        c.VOXELS: voxels,
        c.NUM_X: num_x,
        c.NUM_Y: num_y,
        c.NUM_Z: num_z
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")

    parser.add_argument('--input_config', required=True,
                        help='Configuaration pickle containing AirSAS config')
    parser.add_argument('--output_config', required=True,
                        help='Output for modified config file containing simulated waveforms')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Attempt to use GPU")
    parser.add_argument('--voxels_within', type=float, default=None, required=False,
                        help="use voxels within beamwidth")
    parser.add_argument('--old_cube_geo', action='store_true', default=False)
    parser.add_argument('--object', required=False, help='cube', default=None)
    parser.add_argument('--correction_term', required=False, default=False, action='store_true',
                        help="Start from batch")
    parser.add_argument('--bin_upsample', required=True, type=int,
                        default=20)
    parser.add_argument('--signal_snr', required=True,
                        default=20)
    parser.add_argument('--wfm_part_1', required=True)
    parser.add_argument('--wfm_part_2', required=False, default=None)
    parser.add_argument('--wfm_bw', required=True, help="Waveform bandwidth (5, 10, or 20)")
    parser.add_argument('--resample_measurements', required=False, default=None,
                        type=str, help="Whether to resample cylindrical measurements")
    parser.add_argument('--pitch_levels', default=20, type=float, help="Pitch of helix. This is the number"
                                                                       "of vertical z steps to take for every "
                                                                       "rotation.")
    parser.add_argument('--skip_every_n', default=4, type=int, help="Skip every --skip_every_n measurements to "
                                                                    "for sparse view resampling. ")
    args = parser.parse_args()

    if args.resample_measurements is not None:
        assert args.resample_measurements in [c.HELIX, c.SPARSE]

    if args.object is not None:
        assert args.object in ['cube']

    assert args.wfm_bw in ['5', '10', '20']

    print("SIMULATING WITH BANDWIDTH", args.wfm_bw, "kHZ", "at", args.signal_snr, "dB!!")

    device = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print("Did not find GPU --- using CPU")

    with open(args.input_config, 'rb') as handle:
        system_data = pickle.load(handle)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(args.skip_every_n)

    #tx_coords = system_data[c.TX_COORDS]
    #rx_coords = system_data[c.RX_COORDS]

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(system_data[c.TX_COORDS][..., 0],
    #           system_data[c.TX_COORDS][..., 1],
    #           system_data[c.TX_COORDS][..., 2], label='TX', alpha=0.1)
    #plt.savefig(os.path.join(args.output_dir, 'debug_tx_before.png'))

    if args.resample_measurements is not None:
        print("Resampling measurements")
        resample_indeces = resample_cylindrical(system_data[c.TX_COORDS],
                                                resample_type=args.resample_measurements,
                                                pitch_levels=args.pitch_levels,
                                                skip_every_n=args.skip_every_n)

        print("INDECES_SHAPE", resample_indeces.shape)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(system_data[c.TX_COORDS][..., 0],
                   system_data[c.TX_COORDS][..., 1],
                   system_data[c.TX_COORDS][..., 2], label='TX Orig', alpha=0.01)
        ax.scatter(system_data[c.TX_COORDS][resample_indeces, 0],
                   system_data[c.TX_COORDS][resample_indeces, 1],
                   system_data[c.TX_COORDS][resample_indeces, 2], label='TX Resampled', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'resampled.png'))

        print("Original number of coordinates", system_data[c.TX_COORDS].shape)

        system_data[c.TX_COORDS] = system_data[c.TX_COORDS][resample_indeces, :]
        system_data[c.RX_COORDS] = system_data[c.RX_COORDS][resample_indeces, :]

        print("Resampled coordinates", system_data[c.TX_COORDS].shape)

    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]

    NUM_X = system_data[c.GEOMETRY][c.NUM_X]
    NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
    NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]
    voxels = system_data[c.GEOMETRY][c.VOXELS]
    corners = system_data[c.GEOMETRY][c.CORNERS]

    fs = system_data[c.SYS_PARAMS][c.FS]

    speed_of_sound = system_data[c.SOUND_SPEED]

    if args.wfm_bw == '5':
        wfm = gen_real_lfm(system_data[c.SYS_PARAMS][c.FS],
                           22500,
                           17500,
                           system_data[c.WFM_PARAMS][c.T_DUR],
                           window=True,
                           win_ratio=system_data[c.WFM_PARAMS][c.WIN_RATIO])
        system_data[c.WFM] = wfm
    elif args.wfm_bw == '10':
        wfm = gen_real_lfm(system_data[c.SYS_PARAMS][c.FS],
                           25000,
                           15000,
                           system_data[c.WFM_PARAMS][c.T_DUR],
                           window=True,
                           win_ratio=system_data[c.WFM_PARAMS][c.WIN_RATIO])
        system_data[c.WFM] = wfm
    elif args.wfm_bw == '20':
        wfm = gen_real_lfm(system_data[c.SYS_PARAMS][c.FS],
                           30000,
                           10000,
                           system_data[c.WFM_PARAMS][c.T_DUR],
                           window=True,
                           win_ratio=system_data[c.WFM_PARAMS][c.WIN_RATIO])
        system_data[c.WFM] = wfm
    else:
        raise IOError("wfm bandwidth not recognized")

    if args.object == 'cube':
        kernel = kernel_from_waveform(system_data[c.WFM], wfm_crop_settings[c.NUM_SAMPLES]).detach().cpu().numpy()
        kernel_no_rc = \
            no_rc_kernel_from_waveform(system_data[c.WFM], wfm_crop_settings[c.NUM_SAMPLES]).detach().cpu().numpy()
    else:
        tx_up = scipy.signal.resample(system_data[c.WFM], system_data[c.WFM].shape[0]*args.bin_upsample)

        kernel = kernel_from_waveform(tx_up, wfm_crop_settings[c.NUM_SAMPLES]*args.bin_upsample).detach().cpu().numpy()
        kernel_no_rc = \
            no_rc_kernel_from_waveform(tx_up, wfm_crop_settings[c.NUM_SAMPLES]*args.bin_upsample).detach().cpu().numpy()

    data_rc = []
    data_orig = []

    print("Loading waveforms")
    if args.object == 'cube':
        wfms1 = np.load('/data/sjayasur/awreed/airsas_data/cube_render_data/part_0_2.npy')
        wfms2 = np.load('/data/sjayasur/awreed/airsas_data/cube_render_data/part_1_2.npy')
        wfms = np.concatenate((wfms1, wfms2), axis=0)
        wfms = wfms / (2048**2)
    else:
        print("loading part 1")
        wfms = np.load(args.wfm_part_1)
        if args.wfm_part_2 is not None:
            print("loading part 2")
            wfms2 = np.load(args.wfm_part_2)
            wfms = np.concatenate((wfms, wfms2), axis=0)

        wfms = wfms / (2048 ** 2)

    print("Loaded waveforms with shape", wfms.shape)

    # If need to resize the rendered waveform
    # This is the old GEO
    if args.old_cube_geo:
        geo = create_voxels(-.1, .1, -.1, .1, 0, .3, 75, 75, 75)
        corners_old = geo[c.CORNERS]
        wfm_length = system_data[c.WFM_PARAMS][c.T_DUR] * system_data[c.SYS_PARAMS][c.FS]
        wfm_crop_settings_old = crop_wfm(system_data[c.TX_COORDS],
                                         system_data[c.RX_COORDS],
                                         corners_old,
                                         wfm_length,
                                         system_data[c.SYS_PARAMS][c.FS],
                                         speed_of_sound)

        wfms_old = np.zeros((wfms.shape[0], 1000))
        wfms_old[:, wfm_crop_settings_old['min_sample']:
                    wfm_crop_settings_old['min_sample']+wfm_crop_settings_old['num_samples']] = wfms
        wfms = wfms_old[:, wfm_crop_settings['min_sample']:wfm_crop_settings['min_sample']+wfm_crop_settings['num_samples']]

    if args.resample_measurements is not None:
        wfms = wfms[resample_indeces, :]
        print("Resampled waveforms to shape", wfms.shape)

    data_rc = []
    data_no_rc = []
    for count in tqdm(range(wfms.shape[0]), desc="Convolving transients with waveform"):
        if args.object == 'cube':
            wfm = wfms[count]
            wfm_convolve_rc = np.fft.ifft(np.fft.fft(wfm) * np.conj(kernel))
            wfm_convolve = np.fft.ifft(np.fft.fft(wfm) * kernel_no_rc).real
        else:
            wfm = wfms[count]
            correction_factor = system_data[c.SOUND_SPEED] * \
                                np.arange(0, wfm.shape[0], 1) / (system_data[c.SYS_PARAMS][c.FS] * args.bin_upsample)
            if args.correction_term:
                wfm = wfm * correction_factor
            wfm_convolve = np.fft.ifft(np.fft.fft(wfm) * kernel_no_rc).real

            wfm_convolve = wfm_convolve[::args.bin_upsample]

        data_no_rc.append(wfm_convolve)

    del wfms
    print("stacking...")
    #data_rc = np.stack((data_rc))
    data_no_rc = np.stack((data_no_rc))

    if args.signal_snr is not None:
        print("Adding noise to set signal SNR to", args.signal_snr, "dB")
        plt.figure()
        plt.plot(data_no_rc[0, :])
        plt.savefig(os.path.join(args.output_dir, 'before_noise.png'))

        data_avg_watts = np.mean(data_no_rc[data_no_rc > 1e-6]**2)
        data_avg_db = 10 * np.log10(data_avg_watts)
        noise_avg_db = data_avg_db - float(args.signal_snr)
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), data_no_rc.shape)
        data_no_rc = data_no_rc + noise

        plt.figure()
        plt.plot(data_no_rc[0, :])
        plt.savefig(os.path.join(args.output_dir, 'after_noise.png'))

    #system_data[c.WFM_RC] = data_rc
    system_data[c.WFM_DATA] = data_no_rc

    print("Computing match filtered waveforms from scratch...")
    data_rc_new = match_filter_all(system_data[c.WFM_DATA], system_data[c.WFM])

    system_data[c.WFM_RC] = data_rc_new

    # Overrwrite the system data with new wfms
    with open(args.output_config, 'wb') as handle:
        pickle.dump(system_data, handle)

    print("Backprojecting scene from scratch...")

    scene = backproject_all_airsas_measurements(tx_pos=system_data[c.TX_COORDS],
                                                rx_pos=system_data[c.RX_COORDS],
                                                voxels=voxels,
                                                measurements=data_rc_new,
                                                speed_of_sound=speed_of_sound,
                                                min_dist=wfm_crop_settings[c.MIN_DIST],
                                                group_delay=0.,
                                                r=5,
                                                fs=fs,
                                                basebanded=False,
                                                device=device)

    if torch.is_tensor(scene):
        scene = scene.detach().cpu().numpy()

    print(scene.shape)

    scene = np.reshape(scene, (int(NUM_X),
                               int(NUM_Y),
                               int(NUM_Z)))#

    np.save(os.path.join(args.output_dir, c.BF_FILE), scene)




