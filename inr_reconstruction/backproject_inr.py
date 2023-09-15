import argparse
import glob
import os
import constants as c
import pickle
import numpy as np
import commentjson as json
from beamformer import neural_backproject, backproject_all_airsas_measurements, backproject_all_svss_measurements
from argument_io import directory_cleaup
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sas_utils import hilbert_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backproject INR measurements")
    parser.add_argument('--clear_output_directory', required=False, default=False, action='store_true',
                        help="Whether to delete everything in output directory before running")
    parser.add_argument('--fit_folder', required=True,
                        help="Path to directory containing fit data")
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save output')
    parser.add_argument('--system_data', required=True,
                        help='Pickle file containing system data structure')
    parser.add_argument('--interpolation_factor', type=int, required=False, default=100)
    parser.add_argument('--use_phase', action='store_true', default=False)
    parser.add_argument('--sqrt_weights', action='store_true', default=False,
                         help="Whether to take sqrt of weights")
    parser.add_argument('--normalize_weights', action='store_true', default=False, help="Normalize weight batch by "
                                                                                        "their max. ")
    parser.add_argument('--save3D', action='store_true', default=False, help="Whether to store 3D plots")
    parser.add_argument('--bf_op', default=None, required=True, help="set to 'airsas' (without the '') for beamformer "
                                                      "to ignore beamwidth and 'svss' to account for it")
    parser.add_argument('--x_then_y', required=False, default=False, action='store_true',
                        help="Whether to reshape scene as Y, X, Z (False) or X, Y, Z (True)")
    parser.add_argument('--depth_slices', required=False, default=False, action='store_true',
                        help="Whether to save image slices")
    parser.add_argument('--weight_mag', required=False, default=False, action='store_true',
                        help="Use magnitude of weights")

    args = parser.parse_args()

    if args.bf_op is not None:
        assert args.bf_op in [c.AIRSAS_BF, c.SERDP_BF]

    assert torch.cuda.is_available()
    dev = 'cuda:0'

    directory_cleaup(args.output_dir, args.clear_output_directory)

    weight_paths = glob.glob(os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*'))

    assert len(weight_paths) > 0, "Failed to load any weight paths"

    with open(args.system_data, 'rb') as handle:
        system_data = pickle.load(handle)

    speed_of_sound = system_data[c.SOUND_SPEED]
    fc = system_data[c.SYS_PARAMS][c.FC]
    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]

    option = c.AIRSAS_BF
    if args.bf_op in c.SERDP_BF:
        assert system_data[c.SYS_PARAMS][c.TX_BW] is not None
        option = c.SERDP_BF

    if args.bf_op is None:
        print("Automatically detecting beamformer option based on whether TX BW is not None")
        if system_data[c.SYS_PARAMS][c.TX_BW] is not None:
            print("Detected a set bandwidth, beamformer will account for it.")
            option = 1

    print("Using option", option)

    with torch.no_grad():
        # backproject the scene
        if args.use_phase:
            total_scene = \
                torch.zeros((system_data[c.GEOMETRY][c.VOXELS].shape[0]), dtype=torch.complex64).to(dev)
        else:
            total_scene = torch.zeros((system_data[c.GEOMETRY][c.VOXELS].shape[0])).to(dev)

        # Iterate over all saved models
        for index in tqdm(range(len(weight_paths)), desc="Backprojecting INR"):
            # Beamforming using network to predict mag
            weight_path = weight_paths[index]
            split_weight_path = weight_path.split('_')
            start_index = int(split_weight_path[-2])
            stop_index = int(split_weight_path[-1].split('.')[0]) + 1

            tx_batch = torch.from_numpy(system_data[c.TX_COORDS][start_index:stop_index])
            rx_batch = torch.from_numpy(system_data[c.RX_COORDS][start_index:stop_index])
            wfm_batch = torch.from_numpy(system_data[c.WFM_RC][start_index:stop_index])

            weights = np.load(weight_path)

            if args.sqrt_weights:
                weights = np.sqrt(weights)

            if args.normalize_weights:
                weights = weights / weights.max()

            weights = torch.from_numpy(weights).to(dev).float()
            if args.use_phase:
                weights = hilbert_torch(weights.float())

            if args.weight_mag:
                weights = weights.abs()

            if option == c.AIRSAS_BF:
                total_scene = total_scene + backproject_all_airsas_measurements(
                                                                tx_pos=tx_batch,
                                                                rx_pos=rx_batch,
                                                                voxels=system_data[c.GEOMETRY][c.VOXELS],
                                                                measurements=weights,
                                                                speed_of_sound=speed_of_sound,
                                                                min_dist=wfm_crop_settings[c.MIN_DIST],
                                                                max_dist=wfm_crop_settings[c.MAX_DIST],
                                                                group_delay=0.,
                                                                r=args.interpolation_factor,
                                                                fs=system_data[c.SYS_PARAMS][c.FS],
                                                                basebanded=False,
                                                                device=dev,
                                                                fc=fc)

            elif option == c.SERDP_BF:
                total_scene = total_scene + backproject_all_svss_measurements(
                                                                tx_coords=tx_batch,
                                                                rx_coords=rx_batch,
                                                                measurements=weights,
                                                                min_dist=wfm_crop_settings[c.MIN_DIST],
                                                                max_dist=wfm_crop_settings[c.MAX_DIST],
                                                                speed_of_sound=system_data[c.SOUND_SPEED],
                                                                tx_vec=system_data[c.TX_VECS],
                                                                rx_vec=system_data[c.RX_VECS],
                                                                tx_bw=system_data[c.SYS_PARAMS][c.TX_BW],
                                                                rx_bw=system_data[c.SYS_PARAMS][c.RX_BW],
                                                                voxels=system_data[c.GEOMETRY][c.VOXELS],
                                                                fs=system_data[c.SYS_PARAMS][c.FS],
                                                                basebanded=False,
                                                                pca=False,
                                                                fc=fc,
                                                                same_tx_per_k_rx=system_data[c.SAME_TX_PER_K_RX],
                                                                r=args.interpolation_factor,
                                                                group_delay=0.,
                                                                device=dev)

            else:
                raise OSError("Unknown option.")

        total_scene = total_scene.squeeze().detach().cpu().numpy()
        np.save(os.path.join(args.output_dir, c.NUMPY, c.SCENE_RAVELED), total_scene)

    ##################
    # Visualization ##
    ##################

    try:
        if args.x_then_y:
            total_scene = total_scene.reshape(system_data[c.GEOMETRY][c.NUM_X],
                                              system_data[c.GEOMETRY][c.NUM_Y],
                                              system_data[c.GEOMETRY][c.NUM_Z])
            np.save(os.path.join(args.output_dir, c.NUMPY, c.DAS_INR_FILE), total_scene)
        else:
            total_scene = total_scene.reshape(system_data[c.GEOMETRY][c.NUM_Y],
                                              system_data[c.GEOMETRY][c.NUM_X],
                                              system_data[c.GEOMETRY][c.NUM_Z])
            np.save(os.path.join(args.output_dir, c.NUMPY, c.DAS_INR_FILE), total_scene)

        scene_abs = np.abs(total_scene)
        geo = system_data[c.GEOMETRY]

        print(scene_abs.shape)

        for slice in range(geo[c.NUM_Z]):
            # hello
            plt.figure()
            plt.imshow(np.abs(total_scene)[..., slice])
            plt.colorbar()
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_xy_abs_' + str(slice) + '.png'))

            plt.figure()
            plt.imshow(np.real(total_scene)[..., slice])
            plt.colorbar()
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_xy_real_' + str(slice) + '.png'))

            plt.figure()
            plt.imshow(np.imag(total_scene)[..., slice])
            plt.colorbar()
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_2d_center_imag_' + str(slice) + '.png'))

        if args.depth_slices:
            for i in range(0, geo[c.NUM_Z]):
                slice = str(i)
                fig = plt.figure()
                plt.imshow(scene_abs[..., i], cmap='jet')
                plt.title("Depth slice" + slice)
                plt.colorbar(label='Linear Mag')
                plt.xlabel('Cross Track')
                plt.ylabel('Along Track')
                plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'z' + str(i) + '.png'))
                plt.close(fig)
                plt.clf()

            for i in range(0, geo[c.NUM_X]):
                slice = str(i)
                fig = plt.figure()
                plt.imshow((scene_abs[i, ...]).T, cmap='jet')
                plt.title("Along Track Slice" + slice)
                plt.colorbar(label='Linear Mag')
                plt.xlabel('Cross Track')
                plt.ylabel('Depth')
                plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'x' + str(i) + '.png'))
                plt.close(fig)
                plt.clf()

            for i in range(0, geo[c.NUM_Y]):
                slice = str(i)
                fig = plt.figure()
                plt.imshow((scene_abs[:, i, :]).T, cmap='jet')
                plt.title("Cross Track Slice" + slice)
                plt.colorbar(label='Linear Mag')
                plt.xlabel('Along Track')
                plt.ylabel('Depth')
                plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'y' + str(i) + '.png'))
                plt.close(fig)
                plt.clf()


    except KeyError:
        print("Did not find number of Z, skipping 2d plot")

    if args.use_phase:
        mag = np.abs(total_scene)
    else:
        mag = total_scene

    if args.save3D:
        mag = mag.ravel()

        u = mag.mean()
        var = mag.std()
        vals = np.arange(0., 10, 0.1)
        thresh_vals = u + vals * var

        for thresh_count in tqdm(range(0, len(thresh_vals)), desc="Saving 3D plots"):
            mag[mag[:] < thresh_vals[thresh_count]] = None

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()
            im = ax.scatter(system_data[c.GEOMETRY][c.VOXELS][:, 0],
                            system_data[c.GEOMETRY][c.VOXELS][:, 1],
                            system_data[c.GEOMETRY][c.VOXELS][:, 2],
                            c=mag, alpha=0.5)
            ax.set_xlim3d((system_data[c.GEOMETRY][c.CORNERS][:, 0].min(),
                           system_data[c.GEOMETRY][c.CORNERS][:, 0].max()))
            ax.set_ylim3d((system_data[c.GEOMETRY][c.CORNERS][:, 1].min(),
                           system_data[c.GEOMETRY][c.CORNERS][:, 1].max()))
            ax.set_zlim3d((system_data[c.GEOMETRY][c.CORNERS][:, 2].min(),
                           system_data[c.GEOMETRY][c.CORNERS][:, 2].max()))
            plt.grid(True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.colorbar(im)
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_3d_thresh_' + str(thresh_count) + '.png'))
            plt.close('all')

        mag = np.log10(np.abs(total_scene))
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
            im = ax.scatter(system_data[c.GEOMETRY][c.VOXELS][:, 0],
                            system_data[c.GEOMETRY][c.VOXELS][:, 1],
                            system_data[c.GEOMETRY][c.VOXELS][:, 2],
                            c=mag, alpha=0.5)
            ax.set_xlim3d((system_data[c.GEOMETRY][c.CORNERS][:, 0].min(), system_data[c.GEOMETRY][c.CORNERS][:, 0].max()))
            ax.set_ylim3d((system_data[c.GEOMETRY][c.CORNERS][:, 1].min(), system_data[c.GEOMETRY][c.CORNERS][:, 1].max()))
            ax.set_zlim3d((system_data[c.GEOMETRY][c.CORNERS][:, 2].min(), system_data[c.GEOMETRY][c.CORNERS][:, 2].max()))
            plt.grid(True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            fig.colorbar(im)
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'bf_3d_log_thresh_' + str(thresh_count) + '.png'))
            plt.close('all')








