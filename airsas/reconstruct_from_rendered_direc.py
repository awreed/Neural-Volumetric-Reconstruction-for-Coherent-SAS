import argparse
import pickle
import constants as c
import os
import glob
import scipy.io
import matplotlib.pyplot as plt
from sas_utils import kernel_from_waveform
import numpy as np
from beamformer import backproject_all_airsas_measurements
import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruct AirSAS data")
    parser.add_argument('--render_directory', help="Directory containing mat files", required=True)
    parser.add_argument('--inverse_config', required=True,
                        help='Configuaration pickle containing AirSAS config')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Attempt to use GPU")
    parser.add_argument('--voxels_within', type=float, default=None, required=False,
                        help="use voxels within beamwidth")
    args = parser.parse_args()

    mat_files = glob.glob(os.path.join(args.render_directory, "*.mat"))
    mat_files = sorted(mat_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    device = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print("Did not find GPU --- using CPU")

    with open(args.inverse_config, 'rb') as handle:
        system_data = pickle.load(handle)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tx_coords = system_data[c.TX_COORDS]
    rx_coords = system_data[c.RX_COORDS]
    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]
    NUM_X = system_data[c.GEOMETRY][c.NUM_X]
    NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
    NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]
    voxels = system_data[c.GEOMETRY][c.VOXELS]
    fs = system_data[c.SYS_PARAMS][c.FS]

    sound_speed = system_data[c.SOUND_SPEED]

    kernel = kernel_from_waveform(system_data[c.WFM], wfm_crop_settings[c.NUM_SAMPLES]).detach().cpu().numpy()

    data_rc = []
    data_orig = []

    for count, mat in tqdm(enumerate(mat_files), desc="Convolving with kernel"):
    #    plt.clf()
        wfm_render = scipy.io.loadmat(mat)['I'][..., ::3].squeeze()

        wfm_render = np.sqrt(wfm_render) / np.sqrt(np.sum(np.sqrt(wfm_render)**2) + 1e-7)

        wfm_convolve = np.fft.ifft(np.fft.fft(wfm_render) * np.conj(kernel))
        #wfm_convolve = wfm_convolve/np.sqrt(np.sum(np.abs(wfm_convolve)**2) + 1e-7)
        #wfm_render = wfm_render/wfm_render.max()
        #wfm_convolve = np.abs(wfm_convolve) / np.abs(wfm_convolve).max()

        #plt.figure()
        #plt.plot(wfm_render, label="Rendered wfm at angle 0")
        #plt.plot(wfm_convolve, label="rendered wfm * transmit wfm")
        #plt.legend()
        #plt.show()

        data_rc.append(wfm_convolve)
        data_orig.append(wfm_render)

    data_rc = np.stack((data_rc))
    data_orig = np.stack((data_orig))

    #print(data_rc.shape, data_orig.shape)

    #plt.figure(figsize=(40, 8))
    #plt.subplot(2, 1, 1)
    #plt.imshow(data_orig.T)
    #plt.subplot(2, 1, 2)
    #plt.imshow(np.abs(data_rc).T)
    #plt.tight_layout()
    #plt.show()

    #plt.figure()
    #plt.imshow(np.abs(data_rc))
    #plt.show()

    #plt.figure()
    #plt.plot(np.abs(data_rc[0, :]), label='0')
    #plt.plot(np.abs(data_rc[45, :]), label='45')
    #plt.plot(np.abs(data_rc[90, :]), label='90')
    #plt.plot(np.abs(data_rc[135, :]), label='135')
    #plt.plot(np.abs(data_rc[180, :]), label='180')
    #plt.plot(np.abs(data_rc[225, :]), label='225')
    #plt.plot(np.abs(data_rc[270, :]), label='270')
    #plt.plot(np.abs(data_rc[315, :]), label='315')
    #plt.plot(np.abs(data_rc[359, :]), label='359')
    #plt.legend()
    #plt.xlabel('Samples')
    #plt.ylabel('Amplitude')
    #plt.show()

    system_data[c.WFM_RC] = data_rc

    # Overrwrite the system data with new wfms
    with open(args.inverse_config, 'wb') as handle:
        system_data = pickle.dump(system_data, handle)

    print("Backprojecting scene from scratch...")


    scene = backproject_all_airsas_measurements_gpu(torch.from_numpy(tx_coords).to(device),
                                                    torch.from_numpy(rx_coords).to(device),
                                                    torch.from_numpy(voxels).to(device),
                                                    data_rc,
                                                    sound_speed,
                                                    min_dist=wfm_crop_settings[c.MIN_DIST],
                                                    group_delay=0.,
                                                    r=1,
                                                    fs=fs,
                                                    basebanded=False,
                                                    voxels_within=args.voxels_within)

    scene = scene.detach().cpu().numpy()

    np.save(os.path.join(args.output_dir, c.BF_FILE), scene)

    scene = np.reshape(scene, (NUM_X,
                               NUM_Y,
                               NUM_Z))

    plt.figure()
    plt.imshow(np.real(scene[..., 0]))
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.imag(scene[..., 0]))
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.abs(scene[..., 0]))
    plt.colorbar()
    plt.show()




