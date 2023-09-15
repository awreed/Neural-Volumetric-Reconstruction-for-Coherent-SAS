import numpy as np
import constants as c
import matplotlib.pyplot as plt
import os
from data_schemas import Geometry

def create_voxels(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z, permute_xy=False):
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

    x_dim = np.abs(x_max - x_min)
    y_dim = np.abs(y_max - y_min)
    z_dim = np.abs(z_max - z_min)

    if permute_xy:
        (x, y, z) = np.meshgrid(y_vect, x_vect, z_vect)
    else:
        (x, y, z) = np.meshgrid(x_vect, y_vect, z_vect)

    voxels = np.hstack((np.reshape(x, (np.size(x), 1)),
                        np.reshape(y, (np.size(y), 1)),
                        np.reshape(z, (np.size(z), 1))
                       ))

    geo = Geometry()
    geo[c.CORNERS] = scene_corners
    geo[c.VOXELS] = voxels
    geo[c.NUM_X] = num_x
    geo[c.NUM_Y] = num_y
    geo[c.NUM_Z] = num_z
    geo[c.X_DIM] = x_dim
    geo[c.Y_DIM] = y_dim
    geo[c.Z_DIM] = z_dim

    return geo

def plot_geometry(tx_pos, corners, save_path):
    pad = 0.1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.scatter(tx_pos[:, 0], tx_pos[:, 1], tx_pos[:, 2], alpha=.1, label='Transducers')
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], alpha=1.0, label='Corners')
    ax.set_xlim3d((tx_pos[:, 0].min() - pad, tx_pos[:, 0].max() + pad))
    ax.set_ylim3d((tx_pos[:, 1].min() - pad, tx_pos[:, 1].max() + pad))
    ax.set_zlim3d((tx_pos[:, 2].min() - pad, tx_pos[:, 2].max() + pad))
    plt.grid(True)
    plt.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(tx_pos[:, 2].max() + pad, 0)
    plt.draw()
    fig.savefig(os.path.join(save_path, "YZ_geo.png"))

    ax.view_init(tx_pos[:, 2].max() + pad, 90)
    plt.draw()
    fig.savefig(os.path.join(save_path, "XZ_geo.png"))

    ax.view_init(90, 90)
    plt.draw()
    fig.savefig(os.path.join(save_path, "XY_geo.png"))