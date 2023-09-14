import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os


def save_3d_matplotlib_scene(scene, output_dir, output_name, elev=2, num_angles=4, thresh=2, downsample_factor=None):

    angles = np.linspace(0, 360, num_angles)
    mag = np.abs(scene)

    if downsample_factor is not None:
        df = math.floor(downsample_factor)

        mag = mag[::df, ::df, ::df]

    num_x, num_y, num_z = mag.shape
    mag = mag.ravel()

    u = mag.mean()
    var = mag.std()
    thresh = u + thresh*var

    mag[mag[:] < thresh] = None

    x = np.linspace(-1, 1, num_x)
    y = np.linspace(-1, 1, num_y)
    z = np.linspace(-1, 1, num_z)
    voxels = np.stack(np.meshgrid(x, y, z, indexing='ij'))
    voxels = np.transpose(voxels, (1, 2, 3, 0))
    voxels = np.reshape(voxels, (-1, 3))

    for i, angle in enumerate(angles):
        print("Saving plot " + str(i) + '/' + str(len(angles)-1))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()

        im = ax.scatter(voxels[:, 0],
               voxels[:, 1],
               voxels[:, 2],
               c=mag, alpha=0.5)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        ax.view_init(elev, angle, 0)

        plt.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(im)
        plt.savefig(os.path.join(output_dir, output_name + '_view_' + str(i) + '.png'))

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def inr_fit_sampling(batch_size, dists_norm, rand_trans, trans_batch, dev):
    # Create a vector of indececs that encode the transducer number
    indeces = torch.ones((batch_size, dists_norm.shape[0])).to(dev)
    # [15, 1000]


    trans_vector = torch.from_numpy(np.asarray(rand_trans)).to(dev)
    trans_vector = trans_vector - min(trans_batch)
    trans_vector = trans_vector / len(trans_batch)

    # [batch_size, num_radial]
    indeces = indeces * trans_vector[:, None]


    dists_repeat = torch.ones((batch_size, dists_norm.shape[0])).to(dev)
    dists_repeat = dists_repeat * dists_norm[None, :]

    samples = torch.cat((dists_repeat[..., None], indeces[..., None]), dim=2)
    samples = samples.reshape(-1, 2)

    return samples



# compute vectors between tx and center
# compute vectors to the corners of the scene.
# Compute the theta and phi values
# option to compute beamwidth using scene corners
#


def aggressive_crop_weights(tx_coords, rx_coords, corners, old_min_dist, old_max_dist, num_radial):
    # [1, num_tx, 3] - [num_corners, 1, 3] = [num_corners, num_tx, 3]
    d1 = torch.sqrt(torch.sum((tx_coords[None, ...] - corners[:, None, :]) ** 2, dim=-1))
    d2 = torch.sqrt(torch.sum((rx_coords[None, ...] - corners[:, None, :]) ** 2, dim=-1))

    min_dist = (d1 + d2).view(-1).min()
    max_dist = (d1 + d2).view(-1).max()

    new_min_sample = math.floor(((min_dist - old_min_dist)/(old_max_dist - old_min_dist))*num_radial)
    new_max_sample = math.ceil(((max_dist - old_min_dist)/(old_max_dist - old_min_dist))*num_radial)

    return new_min_sample, new_max_sample


# Credit to DongJT1996 for grid sample 3D with backward capability
# https://github.com/pytorch/pytorch/issues/34704
def custom_grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():
        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2,
                           (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2,
                           (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2,
                           (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2,
                           (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2,
                           (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2,
                           (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2,
                           (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2,
                           (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

