import scipy.signal
import numpy as np
import torch
from tqdm import tqdm
import math
import constants as c
import matplotlib.pyplot as plt
import scipy.fftpack
from data_schemas import WfmCropSettings
from sampling import find_voxels_within_fov


def find_indeces_within_scene(x, corners):
    return torch.where((x[..., 0] >= corners[..., 0].min()) &
                       (x[..., 0] <= corners[..., 0].max()) &
                       (x[..., 1] >= corners[..., 1].min()) &
                       (x[..., 1] <= corners[..., 1].max()) &
                       (x[..., 2] >= corners[..., 2].min()) &
                       (x[..., 2] <= corners[..., 2].max()))[0].long()


def figure_to_tensorboard(writer, fig, fig_name, global_step):
    fig.canvas.draw()
    # Convert the figure to numpy array, read the pixel values and reshape the array
    fig_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    fig_img = fig_img / 255.0
    writer.add_image(fig_name, fig_img.transpose(2, 0, 1), global_step)


def view_fft(y, fs, N=None, path=None):
    assert y.ndim == 1
    if N is None:
        N = y.shape[0]

    T = 1 / fs
    xf = np.linspace(0, 1 / (2. * T), N // 2)
    yf = scipy.fftpack.fft(y)

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title("Freq Domain")
    plt.xlabel('Hz')
    plt.subplot(2, 1, 2)
    plt.plot(y)
    plt.title("Time Domain")
    plt.xlabel('Samples')
    plt.tight_layout()
    plt.show()
    if path is not None:
        plt.savefig(path)


def matplotlib_render(mag, thresh, x_voxels, y_voxels, z_voxels, x_corners, y_corners, z_corners, save_path):
    mag = np.abs(mag)
    mag = mag.ravel()

    u = mag.mean()
    var = mag.std()
    mag[mag[:] < (u + thresh * var)] = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    im = ax.scatter(x_voxels,
                    y_voxels,
                    z_voxels,
                    c=mag, alpha=0.5)
    ax.set_xlim3d(
        (x_corners.min(), x_corners.max()))
    ax.set_ylim3d(
        (y_corners.min(), y_corners.max()))
    ax.set_zlim3d(
        (z_corners.min(), z_corners.max()))
    plt.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(im)
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def comp_mag(x):
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-5)


# ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
def finite_difference_normal(x, model, epsilon=1e-3):
    bound = 1e2
    # x: [N, 3]
    dx_pos = torch.relu(model(
        (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)))[..., 0]).clamp(-bound, bound)
    dx_neg = torch.relu(model(
        (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)))[..., 0]).clamp(-bound, bound)
    dy_pos = torch.relu(model(
        (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)))[..., 0]).clamp(-bound, bound)
    dy_neg = torch.relu(model(
        (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)))[..., 0]).clamp(-bound, bound)
    dz_pos = torch.relu(model(
        (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)))[..., 0]).clamp(-bound, bound)
    dz_neg = torch.relu(model(
        (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)))[..., 0]).clamp(-bound, bound)

    normal = torch.stack([
        .5 * (dx_pos - dx_neg) / epsilon,
        .5 * (dy_pos - dy_neg) / epsilon,
        .5 * (dz_pos - dz_neg) / epsilon
    ], dim=-1)

    return -normal


"""Hilbert transform in pytorch"""


def hilbert_torch(x):
    N = x.shape[-1]

    # add extra dimension that will be removed later.
    if x.ndim == 1:
        x = x[None, :]

    # Take forward fourier transform
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros_like(x)

    if N % 2 == 0:
        h[:, 0] = h[:, N // 2] = 1
        h[:, 1:N // 2] = 2
    else:
        h[:, 0] = 1
        h[:, 1:(N + 1) // 2] = 2

    # Take inverse Fourier transform
    x_hilbert = torch.fft.ifft(Xf * h.to(Xf.device), dim=-1).squeeze()

    return x_hilbert


"""Method to remove the mean amplitude and phase response from time series"""


def remove_room(ts):
    # ts.shape = [nangles, n_samples]
    ang = np.angle(ts)
    cm = np.mean(ts, 0)
    # cm.shape = [n_samples]
    dang = np.angle(np.exp(1j * (np.angle(cm[None, ...]) - np.angle(ts))))
    # dang.shape = [n_angles, n_samples]
    beta = 1. / (1 + np.abs(dang) ** 2)
    alpha = 1. / (1 + (np.abs(cm)[None, ...] - np.abs(ts)) ** 2)
    rm = np.abs(cm)[None, ...] * np.exp(1j * ang)
    nts = ts - alpha * beta * rm

    return nts


"""Method to create LFM waveform
Fs: sample rate
n_samples: number of samples in padded lfm
f_start: LFM start frequency
f_stop: LFM stop frequency
t_dur: LFM duration
window: Option to apply Tukey Window
win_ratio: Tukey window ratio
"""


def gen_real_lfm(Fs, f_start, f_stop, t_dur, window=True, win_ratio=0.1, phase=0):
    times = np.linspace(0, t_dur - 1 / Fs, num=int((t_dur) * Fs))
    LFM = scipy.signal.chirp(times, f_start, t_dur, f_stop, phi=phase)

    if window:
        tuk_win = scipy.signal.windows.tukey(len(LFM), win_ratio)
        LFM = tuk_win * LFM

    return LFM


def modulate_signal(x, fs, fc, keep_quadrature=False):
    if x.ndim == 1:
        x = x[None, :]

    modulate_vec = np.exp(1j * 2 * np.pi * fc * np.arange(0, x.shape[-1], 1) / fs)

    x_mod = x * modulate_vec[None, :]
    x_mod = x_mod.squeeze()

    if keep_quadrature:
        return 2 * x_mod
    else:
        return 2 * x_mod.real


# baseband along the last dimension of x
def baseband_signal(x, fs, fc):
    if x.ndim == 1:
        x = x[None, :]

    demodvect = np.exp(-1j * 2 * np.pi * fc * np.arange(0, x.shape[-1], 1) / fs)
    x_demod = x * demodvect[None, :]
    x_demod = x_demod.squeeze()

    # LPF
    b, a = scipy.signal.butter(5, fc * 2 / fs)
    x_demod = scipy.signal.filtfilt(b, a, x_demod)

    return x_demod


def match_filter_all(x, kernel):
    assert x.ndim == 2

    data_rc = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.complex128)

    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if not torch.is_tensor(kernel):
        kernel = torch.from_numpy(kernel)

    fft_kernel = torch.zeros((x.shape[1]), dtype=kernel.dtype)
    fft_kernel[:kernel.shape[0]] = kernel
    fft_kernel = torch.fft.fft(hilbert_torch(fft_kernel))

    # plt.figure()
    # plt.plot(np.abs(fft_kernel.detach().cpu().numpy()))
    # plt.savefig('./scene_data/das/fft_kernel_mf_all.png')
    # exit(0)

    for i in tqdm(range(x.shape[0]), desc='Match filtering'):
        data_rc[i, ...] = replica_correlate_torch(x[i, ...], fft_kernel)

    return data_rc.detach().cpu().numpy()


def replica_correlate_torch(x, kernel):
    assert not x.dtype == torch.complex, "x should be real"
    # Forward fourier transform of received waveform
    x_hil = hilbert_torch(x)
    x_fft = torch.fft.fft(x_hil)

    # Definition of cross-correlation
    x_rc = torch.fft.ifft(x_fft * torch.conj(kernel))

    return x_rc


"""Interpolation using fft"""


def interpfft(x, r):
    nx = len(x)
    X = torch.fft.fft(x)

    Xint = torch.zeros(int(len(X) * r), dtype=X.dtype)
    nxint = len(Xint)

    if len(x) % 2 == 0:
        Xint[0:nx // 2] = X[0:nx // 2]
        Xint[nx // 2] = X[nx // 2] / 2
        Xint[nxint - nx // 2] = X[nx // 2] / 2
        Xint[nxint - nx // 2 + 1:] = X[nx // 2 + 1:]
    else:
        Xint[0:math.floor(nx / 2) + 1] = X[:math.floor(nx / 2) + 1]
        Xint[nxint - math.floor(nx / 2):] = X[math.floor(nx / 2) + 1:]

    xint = torch.fft.ifft(Xint) * r

    if torch.is_complex(x):
        return xint
    else:
        return xint.real


"""
Same as crop_wfm except accounts for beamwidth of TX/RX. Finite beamwidth means voxels may not be within FOV
"""


def crop_wfm_beamwidth(tx_coords, rx_coords, tx_vec, rx_vec, tx_bw, rx_bw, voxels, wfm_length, fs, speed_of_sound,
                       same_tx_per_k_rx=1, pad=.05, device='cpu'):
    all_dists_min = []
    all_dists_max = []

    count = 0
    valid_indeces = []
    # Find all the valid distances from tx/rx to voxels (voxels within both FOV)
    for tx, rx, tx_v, rx_v in tqdm(zip(tx_coords, rx_coords, tx_vec, rx_vec),
                                   desc="Cropping waveforms"):
        # Only update tx when it changes
        if count % same_tx_per_k_rx == 0:
            _, in_tx_fov_voxels = find_voxels_within_fov(trans_pos=tx,
                                                         tx_vec=tx_v,
                                                         origin=torch.tensor([0., 0., -1.]),
                                                         voxels=voxels,
                                                         bw=tx_bw,
                                                         device=device)

        _, in_both_fov_voxels = find_voxels_within_fov(trans_pos=rx,
                                                       tx_vec=rx_v,
                                                       origin=torch.tensor([0., 0., -1.]),
                                                       voxels=in_tx_fov_voxels,
                                                       bw=rx_bw,
                                                       device=device)

        in_both_fov_voxels = in_tx_fov_voxels

        if in_both_fov_voxels.shape[0] > 0:
            valid_indeces.append(count)

        # print(in_tx_fov_voxels.shape)
        # np.save('/home/albert/tmp/both_fov_' + str(count) + '.npy', in_both_fov_voxels)

            d1 = np.sqrt(np.sum((tx[None, ...] - in_both_fov_voxels) ** 2, axis=-1))
            d2 = np.sqrt(np.sum((rx[None, ...] - in_both_fov_voxels) ** 2, axis=-1))

            tot_dist = d1+d2
            min_val = tot_dist.ravel().min()
            max_val = tot_dist.ravel().max()

            all_dists_min.append(min_val)
            all_dists_max.append(max_val)

        count = count + 1


    # Crop the waveforms based off of these distances
    all_dists_min = np.array(all_dists_min)
    all_dists_max = np.array(all_dists_max)
    # Pad by waveform length and some scalar offset
    min_dist = all_dists_min.min() - pad - (wfm_length / fs * speed_of_sound)
    max_dist = all_dists_max.max() + pad + (wfm_length / fs * speed_of_sound)

    assert max_dist > min_dist, "Sanity check failed"

    min_sample = math.floor(min_dist / speed_of_sound * fs)

    # Update the min dist based off the rounded down sample
    min_dist = min_sample / fs * speed_of_sound

    # 340 is a conservative sound speed to use.
    t_dur = (max_dist - min_dist) / speed_of_sound

    num_samples = math.ceil(t_dur * fs)

    # Update the max dist based off the rounded up sample
    max_dist = ((min_sample + num_samples) / fs) * speed_of_sound

    wfm_crop_settings = WfmCropSettings()
    wfm_crop_settings[c.MIN_SAMPLE] = min_sample
    wfm_crop_settings[c.MIN_DIST] = min_dist
    wfm_crop_settings[c.MAX_DIST] = max_dist
    wfm_crop_settings[c.NUM_SAMPLES] = num_samples

    return wfm_crop_settings, np.array(valid_indeces)


def crop_wfm(tx_coords, rx_coords, corners, wfm_length, fs, speed_of_sound, pad=.05):
    # A conservative estimate for speed of sound in water
    assert tx_coords.shape[0] == rx_coords.shape[0]
    # [1, num_tx, 3] - [num_corners, 1, 3] = [num_corners, num_tx, 3]
    d1 = np.sqrt(np.sum((tx_coords[None, ...] - corners[:, None, :]) ** 2, axis=-1))
    d2 = np.sqrt(np.sum((rx_coords[None, ...] - corners[:, None, :]) ** 2, axis=-1))

    # TODO Should really pad waveform to proper length with with zeros prior to deconvolution.
    min_dist = (d1 + d2).ravel().min() - pad - (min(wfm_length, 100) / fs * speed_of_sound)
    max_dist = (d1 + d2).ravel().max() + pad + (min(wfm_length, 100) / fs * speed_of_sound)

    assert max_dist > min_dist, "Sanity check failed"

    min_sample = math.floor(min_dist / speed_of_sound * fs)

    # Update the min dist based off the rounded down sample
    min_dist = min_sample / fs * speed_of_sound

    # 340 is a conservative sound speed to use.
    t_dur = (max_dist - min_dist) / speed_of_sound

    num_samples = math.ceil(t_dur * fs)

    # Update the max dist based off the rounded up sample
    max_dist = ((min_sample + num_samples) / fs) * speed_of_sound

    wfm_crop_settings = WfmCropSettings()
    wfm_crop_settings[c.MIN_SAMPLE] = min_sample
    wfm_crop_settings[c.MIN_DIST] = min_dist
    wfm_crop_settings[c.MAX_DIST] = max_dist
    wfm_crop_settings[c.NUM_SAMPLES] = num_samples

    return wfm_crop_settings


def radial_delay_wfms_fast(tsd, weights):
    # [batch_size, num_radial, 1] * [1, num_radial, num_samples] = [batch_size, num_radial, num_samples]
    tsd_scaled = weights[..., None] * tsd[None, ...]
    # tsd_scaled = weights[..., None] * tsd[None, ...]
    tsd_sum = torch.sum(tsd_scaled, 1)

    return tsd_sum


# Fixed SNR Wiener filter
def wiener_deconvolution(signal_fft, kernel_fft, lambd):
    if signal_fft.ndim == 2:
        if kernel_fft.ndim == 1:
            kernel_fft = kernel_fft[None, :]
    deconvolved = torch.real(torch.fft.ifft(signal_fft * torch.conj(kernel_fft) /
                                            (kernel_fft * torch.conj(kernel_fft) + lambd ** 2)))
    return deconvolved


# Source https://github.com/ashawkey/stable-dreamfusion/blob/5c8b53f8e8fc041e98bd7d3d210bdd62e7d6fae2/nerf/utils.py#L39
def safe_normalize(x, eps=1e-4):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def no_rc_kernel_from_waveform(wfm, num_samples):
    assert wfm.ndim == 1
    sig = np.zeros((num_samples), dtype=wfm.dtype)
    sig[:wfm.shape[0]] = wfm
    sig = torch.from_numpy(sig)
    if wfm.dtype == complex:
        kernel = torch.fft.fft(sig)
        # kernel = torch.fft.fft(hilbert_torch(sig))
    else:
        kernel = torch.fft.fft(hilbert_torch(sig))
    # kernel_cc = kernel * torch.conj(kernel)
    return kernel


def kernel_from_waveform(wfm, num_samples):
    assert wfm.ndim == 1
    sig = np.zeros((num_samples))
    sig[:wfm.shape[0]] = wfm
    sig = torch.from_numpy(sig)
    kernel = torch.fft.fft(hilbert_torch(sig))
    kernel_cc = kernel * torch.conj(kernel)
    return kernel_cc


def correct_group_delay(wfm, gd, fs):
    assert np.isreal(wfm).all()
    num_samples = wfm.shape[-1]
    df = fs / num_samples
    f_ind = np.linspace(0, int(num_samples - 1), num=int(num_samples),
                        dtype=np.float64)
    f = f_ind * df
    f[f > (fs / 2)] -= fs
    w = (2 * math.pi * f)

    tau = gd / fs

    phase = np.array([tau * w])

    complex_phase = np.zeros_like(phase) + 1j * phase

    pr = np.exp(complex_phase)

    wfm_correct_ifft = np.fft.fft(wfm) * pr

    wfm_correct = np.fft.ifft(wfm_correct_ifft, axis=1).real

    return wfm_correct


def precompute_time_series(dists, min_dist, kernel, speed_of_sound, fs, num_samples):
    df = fs / num_samples
    f_ind = torch.linspace(0, int(num_samples - 1), steps=int(num_samples),
                           dtype=torch.float64)
    f = f_ind * df
    f[f > (fs / 2)] -= fs
    w = (2 * math.pi * f).to(dists.device)

    tau = ((dists) - min_dist) / speed_of_sound

    phase = tau[:, None] * w[None, :]

    complex_phase = torch.complex(real=torch.zeros_like(phase).to(phase.device),
                                  imag=-1 * phase)

    pr = torch.exp(complex_phase)

    tsd_fft = kernel[None, :] * pr  # * torch.exp(1j * 2 * np.pi * tau).to(pr.device)
    tsd = torch.fft.ifft(tsd_fft, dim=1)

    return tsd


def delay_waveforms(tx_pos, rx_pos, weights, voxels, kernel, kernel_no_rc, min_dist, group_delay, fs, speed_of_sound):
    assert tx_pos.shape[0] == rx_pos.shape[0]
    assert kernel.ndim == 1
    num_samples = kernel.shape[0]

    df = fs / (num_samples)
    f_ind = torch.linspace(0, int(num_samples - 1), steps=int(num_samples),
                           dtype=torch.float64)
    f = f_ind * df
    f[f > (fs / 2)] -= fs
    w = (2 * math.pi * f).to(weights.device)

    data_rc = torch.zeros((tx_pos.shape[0], num_samples), dtype=torch.complex128)
    data = torch.zeros((tx_pos.shape[0], num_samples), dtype=torch.float64)

    for i in tqdm(range(tx_pos.shape[0]), desc='Simulating waveforms...'):
        d1 = torch.sqrt(torch.sum((voxels - tx_pos[i, :][None, ...]) ** 2, dim=1))
        d2 = torch.sqrt(torch.sum((voxels - rx_pos[i, :][None, ...]) ** 2, dim=1))

        tau = ((d1 + d2 + (group_delay / fs) * speed_of_sound) - min_dist) / speed_of_sound

        phase = tau[:, None] * w[None, :]

        complex_phase = torch.complex(real=torch.zeros_like(phase).to(phase.device),
                                      imag=-1 * phase)

        pr = torch.exp(complex_phase)

        tsd_fft = kernel[None, :] * pr
        tsd = torch.fft.ifft(tsd_fft, dim=1)
        tsd_scaled = weights[:, None] * tsd
        tsd_sum = torch.sum(tsd_scaled, 0)
        data_rc[i, :] = tsd_sum

        tsd_fft = kernel_no_rc[None, :] * pr
        tsd = torch.fft.ifft(tsd_fft, dim=1).real
        tsd_scaled = weights[:, None] * tsd
        tsd_sum = torch.sum(tsd_scaled, 0)
        data[i, :] = tsd_sum

    return data, data_rc


def range_normalize(x):
    return (x - x.min()) / (x.max() - x.min())
