import argparse
import os
from tqdm import tqdm
import constants as c
from geometry import create_voxels, plot_geometry
from sas_utils import remove_room, gen_real_lfm, match_filter_all, crop_wfm, correct_group_delay
import matplotlib.pyplot as plt
from beamformer import backproject_all_airsas_measurements
import numpy as np
import pickle
from utils import resample_cylindrical
import torch
import scipy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")

    parser.add_argument('--orig_system_file', required=True,
                        help='Original system file (probably noiseless)')
    parser.add_argument('--new_system_file', required=False, default=None,
                        help='New system file containing noisy data')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save code output')
    parser.add_argument('--use_up_to', type=int, required=False, default=None,
                        help='Only beamform a subset of transducers specified by index')
    parser.add_argument('--interpolation_factor', type=int, required=False, default=100)
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Attempt to use GPU")
    parser.add_argument('--voxels_within', type=float, default=None, required=False,
                        help="use voxels within beamwidth")
    parser.add_argument('--incoherent', action='store_true', required=False, help='Whether to beamform incoherently')
    parser.add_argument('--save3D', action='store_true', default=False, help="Whether to store 3D plots")
    parser.add_argument('--signal_snr', type=float, help="Add noise to make signal desired SNR", default=None)
    parser.add_argument('--resample_measurements', required=False, default=None,
                        type=str, help="Whether to resample cylindrical measurements")
    parser.add_argument('--pitch_levels', default=20, type=float, help="Pitch of helix. This is the number"
                                                                       "of vertical z steps to take for every "
                                                                       "rotation.")
    parser.add_argument('--skip_every_n', default=4, type=int, help="Skip every --skip_every_n measurements to "
                                                                    "for sparse view resampling. ")

    args = parser.parse_args()
    device = 'cpu'

    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print("Did not find GPU --- using CPU")

    with open(args.orig_system_file, 'rb') as handle:
        system_data = pickle.load(handle)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print("Original number of coordinates", system_data[c.TX_COORDS].shape)
    if args.use_up_to is not None:
        max_index = 360*args.use_up_to
        system_data[c.TX_COORDS] = system_data[c.TX_COORDS][:max_index, :]
        system_data[c.RX_COORDS] = system_data[c.RX_COORDS][:max_index, :]
        system_data[c.WFM_DATA] = system_data[c.WFM_DATA][:max_index, :]

    print("Only using up to coordinates with shape", system_data[c.TX_COORDS].shape)

    if args.resample_measurements is not None:
        print("Resampling measurements")
        resample_indeces = resample_cylindrical(system_data[c.TX_COORDS],
                                                resample_type=args.resample_measurements,
                                                pitch_levels=args.pitch_levels,
                                                skip_every_n=args.skip_every_n)

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

        print("Coordinates before resampling", system_data[c.TX_COORDS].shape)

        system_data[c.TX_COORDS] = system_data[c.TX_COORDS][resample_indeces, :]
        system_data[c.RX_COORDS] = system_data[c.RX_COORDS][resample_indeces, :]
        system_data[c.WFM_DATA] = system_data[c.WFM_DATA][resample_indeces, :]

        print("Resampled coordinates", system_data[c.TX_COORDS].shape)

    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]

    NUM_X = system_data[c.GEOMETRY][c.NUM_X]
    NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
    NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]
    voxels = system_data[c.GEOMETRY][c.VOXELS]
    corners = system_data[c.GEOMETRY][c.CORNERS]

    fs = system_data[c.SYS_PARAMS][c.FS]

    speed_of_sound = system_data[c.SOUND_SPEED]

    #data = system_data[c.WFM_DATA]

    # If adding noise to signal
    if args.signal_snr is not None:
        print("Adding noise to set signal SNR to", args.signal_snr, "dB")
        plt.figure()
        plt.plot(system_data[c.WFM_DATA][0, :])
        plt.savefig(os.path.join(args.output_dir, 'before_noise.png'))

        data_avg_watts = np.mean(system_data[c.WFM_DATA][system_data[c.WFM_DATA] > 1e-6]**2)
        data_avg_db = 10 * np.log10(data_avg_watts)
        noise_avg_db = data_avg_db - args.signal_snr
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), system_data[c.WFM_DATA].shape)
        system_data[c.WFM_DATA] = system_data[c.WFM_DATA] + noise

        plt.figure()
        plt.plot(system_data[c.WFM_DATA][0, :])
        plt.savefig(os.path.join(args.output_dir, 'after_noise.png'))

    # The reconstruct_from_numpy() will set the appropriate waveform
    system_data[c.WFM_RC] = match_filter_all(system_data[c.WFM_DATA], system_data[c.WFM])

    if args.new_system_file is not None:
        # Write out system_data containing the new noisy signal
        with open(args.new_system_file, 'wb') as handle:
            pickle.dump(system_data, handle)

    # Crop waveforms down
    _, before = system_data[c.WFM_RC].shape
    system_data[c.WFM_RC] = \
        system_data[c.WFM_RC][:,
        wfm_crop_settings[c.MIN_SAMPLE]:wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]
    print("Wfms cropped from", before, "to", system_data[c.WFM_RC].shape[1])

    plt.figure()
    plt.plot(np.abs(system_data[c.WFM_RC][0, :]))
    plt.savefig(os.path.join(args.output_dir, 'mf_after_noise.png'))

    if args.incoherent:
        system_data[c.WFM_RC] = np.abs(system_data[c.WFM_RC])

    print("Backprojecting scene from scratch...")
    scene = backproject_all_airsas_measurements(tx_pos=system_data[c.TX_COORDS],
                                                rx_pos=system_data[c.RX_COORDS],
                                                voxels=voxels,
                                                measurements=system_data[c.WFM_RC],
                                                speed_of_sound=speed_of_sound,
                                                min_dist=wfm_crop_settings[c.MIN_DIST],
                                                group_delay=0.,
                                                r=5,
                                                fs=fs,
                                                basebanded=False,
                                                device=device)

    if torch.is_tensor(scene):
        scene = scene.detach().cpu().numpy()

    scene = scene.reshape((int(NUM_Y), int(NUM_X), int(NUM_Z)))

    np.save(os.path.join(args.output_dir, c.BF_FILE), scene)

    data = {
        'scene': scene,
    }


    scipy.io.savemat(os.path.join(args.output_dir, c.BF_FILE.split('.')[0] + '.mat'), data)

    for slice in [0]:
        # hello
        plt.figure()
        plt.imshow(np.abs(scene)[..., slice])
        plt.colorbar()
        plt.savefig(os.path.join(args.output_dir, 'bf_2d_center_xy_abs_' + str(slice) + '.png'))


