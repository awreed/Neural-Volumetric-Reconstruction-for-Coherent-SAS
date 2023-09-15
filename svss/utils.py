import numpy as np
import matplotlib.pyplot as plt
import scipy

def demodulate(x, fc, fs):
    assert x.ndim == 1
    assert np.isreal(x).all()
    # Get the analytic signal
    x = scipy.signal.hilbert(x)
    # Divide out the carrier
    mod_vec = np.exp(-1j * 2 * np.pi * fc * np.arange(0, len(x), 1)/fs)
    # Return the complex envelope
    return x * mod_vec

def modulate(x, fc, fs):
    assert x.ndim == 1
    assert np.iscomplex(x).any()
    # define the carrier
    mod_vec = np.exp(1j * 2 * np.pi * fc * np.arange(0, len(x), 1)/fs)
    # multiply the carrier by the complex envelope and take real part
    return np.real(x * mod_vec)

def match_filter(x, kernel):
    x_fft = np.fft.fft(x)
    x_rc = np.fft.ifft(x_fft * np.conj(kernel))

    return x_rc

def drc(img, med, des_med):
    fp = (des_med - med*des_med)/(med - med*des_med)
    return (img*fp)/(fp*img-img + 1)

def skew_symm_cp(x):
    assert len(x) == 3
    assert x.ndim == 1
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def normalize_vector(x):
    assert x.ndim == 2
    assert x.shape[1] == 3

    return x/np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)[..., None]


def plot_mips(scene):
    pass


def log_img(x):
    assert x.dtype == np.float64
    return 20*np.log10(x + 1e-8)


def plot_scene_geometry(tx_pos, rx_pos, edges, selected=None):
    tx_pos = np.reshape(tx_pos, (-1, 3))
    rx_pos = np.reshape(rx_pos, (-1, 3))
    edges = np.reshape(edges, (-1, 3))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(tx_pos[:, 0], tx_pos[:, 1], tx_pos[:, 2], c='b', label='TX', marker='x')
    ax.scatter(rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2], c='g', label='RX')
    ax.scatter(edges[:, 0], edges[:, 1], edges[:, 2], c='r', label='voxels', marker='s')

    if selected is not None:
        idx = np.random.randint(len(selected), size=2000)
        subset = selected[idx, :]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c='m', label='fov_voxels', marker='^', alpha=0.2)

    min_x, max_x = min(np.concatenate((tx_pos[:, 0], rx_pos[:, 0], edges[..., 0]))), \
                   max(np.concatenate((tx_pos[:, 0], rx_pos[:, 0], edges[..., 0])))
    range_x = np.abs(max_x - min_x)
    y_mean = np.mean(np.concatenate((tx_pos[:, 1], rx_pos[:, 1], edges[..., 1])))
    z_mean = np.mean(np.concatenate((tx_pos[:, 2], rx_pos[:, 2], edges[..., 2])))
    ax.set_ylim([y_mean - (range_x / 2), y_mean + (range_x / 2)])
    ax.set_zlim([z_mean - (range_x / 2), z_mean + (range_x / 2)])
    ax.set_aspect('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()