import numpy as np
from sas_utils import interpfft, hilbert_torch
from tqdm import tqdm
import torch
from sampling import find_voxels_within_fov
import time
"""Generic backprojection function"""
def backproject_measurements(tx, rx, voxels, measurements, speed_of_sound, min_dist=0., gd=0.,
                             r=100, fs=100000, basebanded=False, fc=None, max_dist=None):
    assert tx.shape[0] == rx.shape[0]
    if measurements.ndim > 1:
        assert measurements.shape[0] == rx.shape[0]

    assert voxels.ndim == 2, "Voxels should be shape [num_voxels, 3]"
    assert voxels.shape[-1] == 3, "Voxels should be shape [num_voxels, 3]"

    if tx.ndim == 1:
        assert tx.shape[0] == 3, "tx should be shape [3] or [num_tx, 3]"
        tx = tx[None, :]

    if rx.ndim == 1:
        assert rx.shape[0] == 3, "tx should be shape [3] or [num_tx, 3]"
        rx = rx[None, :]

    if basebanded:
        assert fc is not None, "Provide center frequency if data is basebanded"

    gd = gd * r
    if max_dist is not None:
        orig_length = measurements.shape[-1]

    measurements_up = interpfft(measurements, r).to(measurements.device)

    d1 = torch.sqrt(torch.sum((voxels - tx) ** 2, dim=1))
    d2 = torch.sqrt(torch.sum((voxels - rx) ** 2, dim=1))

    if max_dist is None:
        tof = ((d1 + d2 - min_dist) / speed_of_sound) * fs * r + gd
    else:
        tof = d1 + d2 + (gd / fs) * speed_of_sound
        # Normalize by max_dist if we have it
        tof = (tof - min_dist) / (max_dist - min_dist)
        tof = (tof * orig_length * r).long()
        #print(tof.min().item(), tof.max().item())
        #print(measurements_up.shape)

    if basebanded:
        return measurements_up[tof.long()] * \
                torch.exp(1j * 2 * np.pi * fc * (d1 + d2) / speed_of_sound)
    else:
        return measurements_up[tof.long()]

"""Method for backrpojecting all AirSAS measurements"""
def backproject_all_airsas_measurements(tx_pos, rx_pos, voxels, measurements, speed_of_sound, min_dist=0.,
                                        group_delay=0., r=100, fs=100000, basebanded=False, fc=None, device='cpu',
                                        max_dist=None):
    assert tx_pos.shape[0] == rx_pos.shape[0]
    assert measurements.shape[0] == rx_pos.shape[0]

    orig_numpy = False

    if not torch.is_tensor(tx_pos):
        tx_pos = torch.from_numpy(tx_pos)

    if not torch.is_tensor(rx_pos):
        rx_pos = torch.from_numpy(rx_pos)

    if not torch.is_tensor(measurements):
        orig_numpy = True
        measurements = torch.from_numpy(measurements)

    if not torch.is_tensor(voxels):
        voxels = torch.from_numpy(voxels)

    tx_pos = tx_pos.to(device)
    rx_pos = rx_pos.to(device)
    measurements = measurements.to(device)
    voxels = voxels.to(device)

    if basebanded:
        assert fc is not None, "Provide center frequency if data is basebanded"

    scene = torch.zeros(voxels.shape[0], dtype=measurements.dtype).to(voxels.device)

    for i in tqdm(range(tx_pos.shape[0]), desc='Beamforming...'):

        scene = scene + backproject_measurements(tx=tx_pos[i, :],
                                                 rx=rx_pos[i, :],
                                                 measurements=measurements[i, :],
                                                 voxels=voxels,
                                                 speed_of_sound=speed_of_sound,
                                                 min_dist=min_dist,
                                                 gd=group_delay,
                                                 r=r,
                                                 fs=fs,
                                                 basebanded=basebanded,
                                                 fc=fc,
                                                 max_dist=max_dist)


    if orig_numpy:
        scene = scene.detach().cpu().numpy()

    return scene

"""Method for backprojecting SVSS measurements. The main difference between this and the AirSAS function is
that this one must account for TX/RX beamwidths."""
def backproject_all_svss_measurements(tx_coords, rx_coords, measurements, min_dist,
                                      speed_of_sound, tx_vec, rx_vec, tx_bw, rx_bw, voxels, fs, basebanded,
                                      fc, same_tx_per_k_rx=1, r=100, group_delay=0., device='cpu', pca=False,
                                      max_dist=None):

    assert tx_coords.shape[0] == measurements.shape[0]
    assert rx_coords.shape[0] == tx_coords.shape[0]

    orig_numpy = False

    if not torch.is_tensor(tx_coords):
        tx_coords = torch.from_numpy(tx_coords)

    if not torch.is_tensor(rx_coords):
        rx_coords = torch.from_numpy(rx_coords)

    if not torch.is_tensor(measurements):
        orig_numpy = True
        measurements = torch.from_numpy(measurements)

    if not torch.is_tensor(tx_vec):
        tx_vec = torch.from_numpy(tx_vec)

    if not torch.is_tensor(rx_vec):
        rx_vec = torch.from_numpy(rx_vec)

    if not torch.is_tensor(voxels):
        voxels = torch.from_numpy(voxels)

    tx_coords = tx_coords.to(device)
    rx_coords = rx_coords.to(device)
    measurements = measurements.to(device)
    tx_vec = tx_vec.to(device)
    rx_vec = rx_vec.to(device)
    voxels = voxels.to(device)

    assert voxels.ndim == 2
    assert voxels.shape[-1] == 3

    scene = torch.zeros(voxels.shape[0], dtype=measurements.dtype).to(device)
    count = 0

    # Find all the valid distances from tx/rx to voxels (voxels within both FOV)
    for tx, rx, tx_v, rx_v in tqdm(zip(tx_coords, rx_coords, tx_vec, rx_vec),
                                   desc="Beamforming"):
        # Only update tx when it changes
        if count % same_tx_per_k_rx == 0:
            in_tx_fov_indeces, in_tx_fov_voxels = find_voxels_within_fov(trans_pos=tx,
                                                                         tx_vec=tx_v,
                                                                         origin=torch.tensor([0, 0, -1.0]).to(tx.device),
                                                                         voxels=voxels,
                                                                         bw=tx_bw,
                                                                         device=device)
            scene_subset = torch.zeros(in_tx_fov_voxels.shape[0], dtype=measurements.dtype).to(device)

        in_both_fov_indeces, in_both_fov_voxels = find_voxels_within_fov(trans_pos=rx,
                                                                         tx_vec=rx_v,
                                                                         origin=torch.tensor([0, 0, -1.0]).to(tx.device),
                                                                         voxels=in_tx_fov_voxels,
                                                                         bw=rx_bw,
                                                                         device=device)
        # phase center approximation
        if pca:
            tx = (tx + rx) / 2
            rx = (tx + rx) / 2

        # Add contributions to scene within both FOV
        scene_subset[in_both_fov_indeces] = scene_subset[in_both_fov_indeces] + \
                                            backproject_measurements(
                                                    tx=tx,
                                                    rx=rx,
                                                    measurements=measurements[count, :],
                                                    voxels=in_both_fov_voxels,
                                                    speed_of_sound=speed_of_sound,
                                                    min_dist=min_dist,
                                                    gd=group_delay,
                                                    r=r,
                                                    fs=fs,
                                                    basebanded=basebanded,
                                                    fc=fc,
                                                    max_dist=max_dist)
        # Add contributions to total scene
        scene[in_tx_fov_indeces] = scene[in_tx_fov_indeces] + scene_subset

        count = count + 1

    if orig_numpy:
        scene = scene.detach().cpu().numpy()

    return scene

def neural_backproject(tx_pos, rx_pos, voxels, speed_of_sound, min_dist, max_dist, weights,
                       group_delay=0.,
                       r=1, fs=100000, use_phase=False, fc=None, wfm_batch=None):
    assert tx_pos.shape[0] == rx_pos.shape[0]
    if use_phase:
        assert fc is not None

    scene = torch.zeros(voxels.shape[0], dtype=weights.dtype).to(weights.device)

    for i in tqdm(range(tx_pos.shape[0]), desc='Beamforming...'):

        orig_length = len(weights[i, :])
        if use_phase:
            weight = hilbert_torch(weights[i, :])
        else:
            weight = weights[i, :]

        weight_up = interpfft(weight, r).to(weight.device)

        d1 = torch.sqrt(torch.sum((voxels - tx_pos[i, :][None, ...]) ** 2, dim=1))
        d2 = torch.sqrt(torch.sum((voxels - rx_pos[i, :][None, ...]) ** 2, dim=1))

        # Compute time of flight
        tof = d1 + d2 + (group_delay / fs) * speed_of_sound
        # Normalize
        tof = (tof - min_dist) / (max_dist - min_dist)

        tof_weights = (tof * orig_length * r).long()

        scene = scene + weight_up[tof_weights]

    return scene
