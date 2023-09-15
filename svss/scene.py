import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import math

"""This class implements methods relating to the voxel scene that we backproject to"""
class DefineScene:
    def __init__(self, root_path, track_id, image_number):
        self.root_path = root_path
        self.image_number = image_number
        self.track_id = track_id
        self.science_file = 'SERDP - ' + self.track_id + ' - LF - down - ' + str(self.image_number) +'.science.hdf5'

        base_name = 'motion solution - LF - down - '
        ftype = '.hdf5'
        self.motion_path = base_name + str(self.image_number) + ftype

        with h5py.File(os.path.join(self.root_path, self.track_id, self.science_file), 'r') as f:
            self.step_size = np.array(f['ImagingGrid']['Step'][:])[0]

        with h5py.File(os.path.join(self.root_path, self.track_id, self.science_file), 'r') as f:
            self.offset = np.array(f['ImagingGrid']['Offset'][:])[0]

        self.xdim, self.ydim, self.zdim = None, None, None

    # useful for plotting purposes
    """This function finds the edges of the voxels for plotting purposes (plotting all voxels is slow)"""
    def find_voxel_edges(self, voxels):
        # Find the voxel edges
        voxels = np.reshape(voxels, (-1, 3))
        edge_indeces = np.where(
            ((voxels[..., 0] == min(voxels[..., 0])) &
             (voxels[..., 1] == max(voxels[..., 1]))) |

            ((voxels[..., 0] == max(voxels[..., 0])) &
             (voxels[..., 1] == max(voxels[..., 1]))) |

            ((voxels[..., 0] == min(voxels[..., 0])) &
             (voxels[..., 2] == max(voxels[..., 2]))) |

            ((voxels[..., 0] == max(voxels[..., 0])) &
             (voxels[..., 2] == max(voxels[..., 2]))) |

            ((voxels[..., 0] == min(voxels[..., 0])) &
             (voxels[..., 1] == min(voxels[..., 1]))) |

            ((voxels[..., 0] == max(voxels[..., 0])) &
             (voxels[..., 1] == min(voxels[..., 1]))) |

            ((voxels[..., 1] == min(voxels[..., 1])) &
             (voxels[..., 2] == min(voxels[..., 2]))) |

            ((voxels[..., 1] == min(voxels[..., 1])) &
             (voxels[..., 2] == max(voxels[..., 2]))) |

            ((voxels[..., 0] == max(voxels[..., 0])) &
             (voxels[..., 2] == min(voxels[..., 2]))) |

            ((voxels[..., 1] == max(voxels[..., 1])) &
             (voxels[..., 2] == min(voxels[..., 2]))) |

            ((voxels[..., 1] == max(voxels[..., 1])) &
             (voxels[..., 2] == max(voxels[..., 2]))) |

            ((voxels[..., 2] == min(voxels[..., 2])) &
             (voxels[..., 0] == min(voxels[..., 0])))
        )
        return voxels[edge_indeces]

    """This function creates the voxel world that we backproject to. It uses the array_data, depth, and motion file
    to automatically define the cross track and along track dimensions. You can override the automatic cross-track calculation
    by setting ct_range_override. Show path will show the path of the array through the world"""
    def create_voxels_for_sas(self, array_data, min_depth, max_depth, ct_range_override=None, show_path=False,
                              y_min=None, y_max=None, x_min=None, x_max=None, resolution=None):
        # Using the array shape and beamwidth to define the scene dimensions
        print("Creating SAS Voxels")
        with h5py.File(os.path.join(self.root_path, self.track_id, self.motion_path), 'r') as f:
            motion = np.array(f['Position'][:])

        if resolution is not None:
            self.step_size=resolution

        full_motion = [np.array([x[0], x[1], x[2]]) for x in motion]
        full_motion = np.array(full_motion)

        # First find the average coordinates of the array
        tx_pos = array_data['tx_pos']
        rx_pos = array_data['rx_pos']
        rx_mean = np.array([np.mean(rx_pos[:, 0]), np.mean(rx_pos[:, 1]), np.mean(rx_pos[:, 2])])
        tx_mean = np.array([np.mean(tx_pos[:, 0]), np.mean(tx_pos[:, 1]), np.mean(tx_pos[:, 2])])
        # mean of the entire transducer array
        phase_center_mean = np.array([np.mean(np.array([rx_mean[0], tx_mean[0]])),
                                      np.mean(np.array([rx_mean[1], tx_mean[1]])),
                                      np.mean(np.array([rx_mean[2], tx_mean[2]]))])
        full_motion = phase_center_mean + full_motion

        # Figure out what the cross track range should be
        tx_ct_range = np.abs(np.min(tx_pos[:, 1]) - np.max(tx_pos[:, 1]))
        tx_bw = array_data['tx_bw']
        rx_bw = array_data['rx_bw']
        # the length of beam projected to floor plus how far the tx sits off the center of the track gives one half ct range
        # multiply by 2 to get the full range
        ct_range = 2*(np.tan(math.radians(tx_bw))*max_depth + tx_ct_range/2)

        # override the above calculation since it can result in a really big scene.
        if ct_range_override is not None:
            ct_range = ct_range_override

        # Use the along track range, cross track range, and depth to create voxels relative to vehicle position
        min_x, max_x = min(full_motion[:, 0]), max(full_motion[:, 0])
        at_range = np.abs(max_x - min_x)

        y_mean = np.mean(full_motion[:, 1])
        z_mean = np.mean(full_motion[:, 2])

        print('Along track range', at_range, 'meters')
        print('Cross track mean', y_mean, 'meters')
        print('Depth mean', z_mean, 'meters')

        xmin = full_motion[0, 0]
        if x_min is not None:
            xmin = full_motion[0, 0] + x_min

        xmax = full_motion[-1, 0]
        if x_max is not None:
            xmax = full_motion[0, 0] + x_max

        ymin = y_mean-ct_range/2
        if y_min is not None:
            ymin = y_mean - y_min

        ymax = y_mean+ct_range/2
        if y_max is not None:
            ymax = y_mean + y_max

        zmin = z_mean+min_depth
        zmax = z_mean+max_depth

        print("Step sizes", self.step_size[0], self.step_size[1], self.step_size[2])

        x = np.arange(xmin, xmax, self.step_size[0])
        y = np.arange(ymin, ymax, self.step_size[1])
        z = np.arange(zmin, zmax, self.step_size[2])

        num_x, num_y, num_z = x.shape[0], y.shape[0], z.shape[0]

        scene_corners = np.array(([xmin, ymin, zmin],
                                  [xmin, ymax, zmin],
                                  [xmin, ymax, zmax],
                                  [xmin, ymin, zmax],
                                  [xmax, ymin, zmin],
                                  [xmax, ymax, zmin],
                                  [xmax, ymax, zmax],
                                  [xmax, ymin, zmax]))

        voxels = np.stack(np.meshgrid(x, y, z))
        voxels_r = np.transpose(voxels, (2, 1, 3, 0))
        edges = self.find_voxel_edges(voxels_r)

        if show_path:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(full_motion[:, 0], full_motion[:, 1], full_motion[:, 2], c='b', label='Vehicle position')
            min_x, max_x = min(full_motion[:, 0]), max(full_motion[:, 0])
            range_x = np.abs(max_x - min_x)

            ax.set_ylim([y_mean-(range_x/2), y_mean + (range_x/2)])
            ax.set_zlim([z_mean-(range_x/2), z_mean + (range_x/2)])
            ax.set_xlabel('Along Track (meters)')
            ax.set_ylabel('Cross Track (meters)')
            ax.set_zlabel('Depth (meters)')
            plt.legend()
            plt.show()

        x_dim = np.abs(xmax - xmin)
        y_dim = np.abs(ymax - ymin)
        z_dim = np.abs(zmax - zmin)

        self.xdim, self.ydim, self.zdim, _ = voxels_r.shape
        return np.reshape(voxels_r, (-1, 3)), edges, scene_corners, num_x, num_y, num_z, x_dim, y_dim, z_dim


    #def create_voxels_for_single_real_aperture_image(self, x_dim, y_dim, z_dim):
    #    x_size = x_dim*self.step_size[0]
    #    x = np.arange(self.offset[0], self.offset[0]+x_size, self.step_size[0])
    #    print(x.min(), x.max())

    #    y_size = y_dim * self.step_size[1]
    #    y = np.arange(self.offset[1], self.offset[1]+y_size, self.step_size[1])
    #    print(y.min(), y.max())

    #    z_size = z_dim * self.step_size[2]
    #    z = np.arange(self.offset[2], self.offset[2] + z_size, self.step_size[2])
    #    print(z.min(), z.max())

    #    voxels = np.stack(np.meshgrid(x, y, z))
    #    voxels_r = np.transpose(voxels, (2, 1, 3, 0))
    #    edges = self.find_voxel_edges(voxels)

    #    return voxels_r, edges
