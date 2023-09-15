from system_parameters import SystemParameters
from array_position import ArrayPosition
from scene import DefineScene
from SERDPBeamformer import SERDPBeamformer
import numpy as np
import os
import argparse
from argument_io import directory_cleaup
import constants as c
from sas_utils import view_fft, crop_wfm_beamwidth, modulate_signal, match_filter_all, baseband_signal, hilbert_torch
from data_schemas import SASDataSchema, SysParams, WfmParams, Geometry
import math
from beamformer import backproject_all_svss_measurements
import torch
import pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Deconvolve SAS measurements using an INR")

parser.add_argument('--output_dir', required=True, help="Directory to save output")
parser.add_argument('--root_path', required=False, help="Path containing SERDP data")
parser.add_argument('--sound_speed_table', required=True, help="Path containing the sound speed table")
parser.add_argument('--clear_output_directory', required=False, default=False, action='store_true',
                        help="Whether to delete everything in output directory before running")
parser.add_argument('--save_images', required=False, default=False, action='store_true',
                        help="Whether to save beamformed images")
parser.add_argument('--gpu', required=False, default=False, action='store_true',
                        help="Whether to use GPU")
parser.add_argument('--do_mf', required=False, default=False, action='store_true',
                        help="Whether to mf by hand")
parser.add_argument('--find_ping', required=False, default=False, action='store_true',
                        help="Debug mode for finding target within track")
parser.add_argument('--min_ping', required=False, default=50, type=int,
                        help="Only use subset of the weights")
parser.add_argument('--max_ping', required=False, default=180, type=int,
                        help="Only use subset of the weights")
parser.add_argument('--rx_min', required=False, default=68, type=int,
                        help="Only use subset of the weights")
parser.add_argument('--rx_max', required=False, default=80, type=int,
                        help="Only use subset of the weights")
parser.add_argument('--y_min', required=False, default=None, type=float,
                        help="Vehicle left CT range")
parser.add_argument('--y_max', required=False, default=None, type=float,
                        help="Vehicle right CT range")
parser.add_argument('--x_min', required=False, default=None, type=float,
                        help="Min AT range")
parser.add_argument('--x_max', required=False, default=None, type=float,
                        help="Max AT range")
parser.add_argument('--drc_mf', required=False, default=1., type=float,
                        help="dynamic range compress waveforms")
parser.add_argument('--x_step', required=False, default=0.01, type=float,
                        help="X bin size")
parser.add_argument('--y_step', required=False, default=0.01, type=float,
                        help="Y bin size")
parser.add_argument('--z_step', required=False, default=0.01, type=float,
                        help="Z bin size")
parser.add_argument('--min_depth', required=False, default=.5, type=float,
                        help="Minimum measured depth")
parser.add_argument('--max_depth', required=False, default=2.0, type=float,
                        help="Maximum measured depth")
parser.add_argument('--r', required=False, default=100, type=int,
                        help="Upsample factor for beamformer")
parser.add_argument('--pca', required=False, default=False, action='store_true',
                        help="Whether to make PCA approximation when beamforming")
parser.add_argument('--track_id', required=False, type=str, help="ID of SERDP track", default='2019 1106 163841')
parser.add_argument('--image_number', required=False, type=int, help="SERDP image number in track.")
parser.add_argument('--tx_list', default=None, type=str, help="comma seperated (no spaces) list of tx "
                                                              "numbers to use 0,1,2,3,4) are all tx")
args = parser.parse_args()


tx_list = [0, 1, 2, 3, 4]
if "tx_list" in vars(args).keys() and args.tx_list is not None:
    tx_list = [int(s.strip()) for s in vars(args)["tx_list"].split(",")]


if tx_list is not None:
    assert min(tx_list) >= 0
    assert max(tx_list) <= 4
    print("Using TX IDs", tx_list)

assert args.rx_min != args.rx_max

device = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        print("Did not find GPU --- using CPU")

assert args.rx_min >= 0 and args.rx_min < 80, "RX index out of bounds"
assert args.rx_max > 0 and args.rx_max <= 80, "RX index out of bounds"

directory_cleaup(args.output_dir, args.clear_output_directory)

# create a save directory
save_dir = 'sas_scenes'

# Point this to a directory containing all elements and motions/coinv2_imagery
root_path = args.root_path
track_id = args.track_id
sound_speed_table_path = args.sound_speed_table
image_number = args.image_number
receiver_indeces = np.arange(args.rx_min, args.rx_max, 1)
print(receiver_indeces)
min_ping = args.min_ping
max_ping = args.max_ping

y_min = args.y_min
y_max = args.y_max
x_min = args.x_min
x_max = args.x_max

resolution = None
if args.x_step is not None and args.y_step is not None and args.z_step is not None:
    resolution=np.array([args.x_step, args.y_step, args.z_step])

print("Setup waveform and pick ping")
SP = SystemParameters(root_path, track_id, image_number, sound_speed_table_path)
# if find_ping, then we show the raw and match-filtered time series
ping_data, raw_data, mf_data, kernel = SP.process_waveforms(find_ping=args.find_ping, do_mf=args.do_mf)

print("Defining the Array position")
AP = ArrayPosition(root_path, track_id, image_number)
array_data = AP.define_array()

print('Defining the scene voxels')
scene = DefineScene(root_path, track_id, image_number)

print("Creating scene voxels")
# Pass in the defined array data
# Remove y_min, y_max, z_min, z_max and it will automatically set the scene dimensions based off of vehicle track
voxels, edges, corners, num_x, num_y, num_z, x_dim, y_dim, z_dim = scene.create_voxels_for_sas(array_data,
                                                                            min_depth=args.min_depth,
                                                                            max_depth=args.max_depth,
                                                                            ct_range_override=2.0, show_path=False,
                                                                            y_min=y_min, y_max=y_max,
                                                                            x_min=x_min, x_max=x_max,
                                                                            resolution=resolution)

tx_coords = []
rx_coords = []
tx_vectors = []
rx_vectors = []
wfms_raw = []
wfms_mf = []

print("Processing pings")
for ping_number, ping in enumerate(ping_data.pings):

    if ping_number < args.min_ping:
        continue
    if ping_number > args.max_ping:
        break

    print("Unrolling data", (ping_number - args.min_ping) /
          len(ping_data.pings[args.min_ping:args.max_ping]) * 100, "%")

    ping = ping_data.pings[ping_number]
    array_data = AP.snap_array_to_world(array_data, ping_number, show_array=False)

    #print(ping_number)
    #print(tx_list)

    # Check if using the transmitter ID
    if tx_list is not None:
        if (ping['tx_id'] - 1) not in tx_list:
            continue

    wfm_raw = ping['rx_raw'].T
    wfm_mf = ping['mf_raw'].T

    wfm_raw = wfm_raw[receiver_indeces, :]
    wfm_mf = wfm_mf[receiver_indeces, :]

    wfm_raw = modulate_signal(wfm_raw, fc=SP.Fc, fs=SP.Fs)
    wfm_mf = modulate_signal(wfm_mf, fc=SP.Fc, fs=SP.Fs, keep_quadrature=True)

    #if args.drc_mf is not None:
    wfm_mf.real = np.sign(wfm_mf.real) * np.abs(wfm_mf.real) ** (args.drc_mf / 1)
    wfm_mf.imag = np.sign(wfm_mf.imag) * np.abs(wfm_mf.imag) ** (args.drc_mf / 1)

    # Edge case when number of receivers is 1
    if wfm_raw.ndim == 1:
        wfm_raw = wfm_raw[None, :]
    if wfm_mf.ndim == 1:
        wfm_mf = wfm_mf[None, :]

    wfms_raw.append(wfm_raw)
    wfms_mf.append(wfm_mf)

    # Get the tx position and its direction vector
    tx_pos = array_data['tx_pos_trans'][ping['tx_id'] - 1]
    tx_vec = array_data['tx_vector']

    # rx position and vector
    rx_pos = array_data['rx_pos_trans']
    rx_vec = array_data['rx_vector']

    rx_pos = rx_pos[receiver_indeces, :]

    tx_pos = np.repeat(tx_pos[None, :], rx_pos.shape[0], axis=0)
    tx_vec = np.repeat(tx_vec[None, :], rx_pos.shape[0], axis=0)

    rx_vec = np.repeat(rx_vec[None, :], rx_pos.shape[0], axis=0)

    tx_coords.append(tx_pos)
    tx_vectors.append(tx_vec)

    rx_coords.append(rx_pos)
    rx_vectors.append(rx_vec)

tx_coords = np.concatenate(tx_coords, axis=0)
tx_vectors = np.concatenate(tx_vectors, axis=0)
rx_coords = np.concatenate(rx_coords, axis=0)
rx_vectors = np.concatenate(rx_vectors, axis=0)

wfms_raw = np.concatenate(wfms_raw, axis=0)
wfms_mf = np.concatenate(wfms_mf, axis=0)

###################################
### Creating the data for export###
###################################
serdp_data = SASDataSchema()

sys_params = SysParams()
sys_params[c.GROUP_DELAY] = 0.
sys_params[c.FS] = SP.Fs
sys_params[c.FC] = SP.Fc
sys_params[c.TX_BW] = math.radians(array_data['tx_bw'])
sys_params[c.RX_BW] = math.radians(array_data['rx_bw'])

serdp_data[c.SYS_PARAMS] = sys_params

serdp_data[c.TX_COORDS] = tx_coords
serdp_data[c.TX_VECS] = tx_vectors
serdp_data[c.RX_COORDS] = rx_coords
serdp_data[c.RX_VECS] = rx_vectors

# raw waveforms are on carrier
serdp_data[c.WFM_DATA] = wfms_raw

# mf waveforms are baseband
serdp_data[c.WFM_RC] = wfms_mf

geo = Geometry()
geo[c.CORNERS] = corners
geo[c.VOXELS] = voxels
geo[c.NUM_X] = num_x
geo[c.NUM_Y] = num_y
geo[c.NUM_Z] = num_z
geo[c.X_DIM] = x_dim
geo[c.Y_DIM] = y_dim
geo[c.Z_DIM] = z_dim

u = corners[0, :] - corners[1, :]
v = corners[0, :] - corners[2, :]
w = corners[0, :] - corners[3, :]

serdp_data[c.GEOMETRY] = geo

serdp_data[c.SOUND_SPEED] = SP.c

wfm_length = kernel.shape[0]

serdp_data[c.WFM] = kernel

wfm_crop_settings, valid_indeces = crop_wfm_beamwidth(
                            tx_coords=serdp_data[c.TX_COORDS],
                            rx_coords=serdp_data[c.RX_COORDS],
                            tx_vec=serdp_data[c.TX_VECS],
                            rx_vec=serdp_data[c.RX_VECS],
                            tx_bw=serdp_data[c.SYS_PARAMS][c.TX_BW],
                            rx_bw=serdp_data[c.SYS_PARAMS][c.RX_BW],
                            voxels=serdp_data[c.GEOMETRY][c.VOXELS],
                            wfm_length=wfm_length,
                            fs=serdp_data[c.SYS_PARAMS][c.FS],
                            speed_of_sound=serdp_data[c.SOUND_SPEED],
                            same_tx_per_k_rx=receiver_indeces.shape[0],
                            device=device)

serdp_data[c.TX_COORDS] = serdp_data[c.TX_COORDS][valid_indeces, :]
serdp_data[c.TX_VECS] = serdp_data[c.TX_VECS][valid_indeces, :]
serdp_data[c.RX_COORDS] = serdp_data[c.RX_COORDS][valid_indeces]
serdp_data[c.RX_VECS] = serdp_data[c.RX_VECS][valid_indeces]
print("To", serdp_data[c.TX_COORDS].shape)
# raw waveforms are on carrier
serdp_data[c.WFM_DATA] = serdp_data[c.WFM_DATA][valid_indeces, :]

# mf waveforms are baseband
serdp_data[c.WFM_RC] = serdp_data[c.WFM_RC][valid_indeces]


serdp_data[c.SAME_TX_PER_K_RX] = receiver_indeces.shape[0]

serdp_data[c.WFM_CROP_SETTINGS] = wfm_crop_settings

# Crop waveforms down
_, before = serdp_data[c.WFM_DATA].shape
serdp_data[c.WFM_DATA] = \
    serdp_data[c.WFM_DATA][:,
    wfm_crop_settings[c.MIN_SAMPLE]:wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]

# Crop waveforms down
_, before = serdp_data[c.WFM_RC].shape
serdp_data[c.WFM_RC] = \
    serdp_data[c.WFM_RC][:,
    wfm_crop_settings[c.MIN_SAMPLE]:wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]

with open(os.path.join(args.output_dir, 'system_data.pik'), 'wb') as handle:
    print("Saving system data to ", os.path.join(args.output_dir, 'system_data.pik'))
    pickle.dump(serdp_data, handle)

scene = backproject_all_svss_measurements(tx_coords=serdp_data[c.TX_COORDS],
                                            rx_coords=serdp_data[c.RX_COORDS],
                                            measurements=serdp_data[c.WFM_RC],
                                            min_dist=wfm_crop_settings[c.MIN_DIST],
                                            tx_vec=serdp_data[c.TX_VECS],
                                            rx_vec=serdp_data[c.RX_VECS],
                                            tx_bw=serdp_data[c.SYS_PARAMS][c.TX_BW],
                                            rx_bw=serdp_data[c.SYS_PARAMS][c.RX_BW],
                                            voxels=serdp_data[c.GEOMETRY][c.VOXELS],
                                            fs=serdp_data[c.SYS_PARAMS][c.FS],
                                            fc=serdp_data[c.SYS_PARAMS][c.FC],
                                            speed_of_sound=serdp_data[c.SOUND_SPEED],
                                            same_tx_per_k_rx=receiver_indeces.shape[0],
                                            basebanded=False,
                                            r=args.r,
                                            group_delay=sys_params[c.GROUP_DELAY],
                                            device=device,
                                            pca=args.pca)


complex_scene = np.reshape(scene, (geo[c.NUM_X], geo[c.NUM_Y], geo[c.NUM_Z]))

np.save(os.path.join(args.output_dir, c.NUMPY, c.BF_FILE), complex_scene)

if args.save_images:
    print("Saving images")
    scene_abs = np.abs(complex_scene)

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
