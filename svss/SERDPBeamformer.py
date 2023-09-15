import numpy as np
import math
from utils import skew_symm_cp, plot_scene_geometry

# Full SAS reconstruction of SERDP track.
# Uses direction of TX/RX transducers to ensure we only reconstruct voxels that fall within the beamwidth of
# both TX and RX at each vehicle position

class SERDPBeamformer:
    def __init__(self, SP, AP, voxels, edges=None):
        self.SP = SP
        self.AP = AP
        self.voxels = voxels
        self.edges = edges
        # Complex scene
        self.scene = np.zeros((len(self.voxels)), dtype=np.complex64)
        # Count number of times we write to a voxel
        self.normalization_counts = np.zeros((len(self.voxels)), dtype=np.ushort)

        print("Reconstructing scene with", self.scene.shape, "voxels")

    """This function maps the tx/rx direction vector and all voxels onto the origin (0, 0, -1) so we can check if 
    voxels lie within a cone given by the TX/RX beamwidth"""
    def _find_voxels_within_fov(self, trans_pos, tx_vec, voxels, bw):
        assert trans_pos.ndim == 1

        origin = np.array([0, 0, -1.0])

        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        # rotate tx dir vector to origin and scene points
        v = np.cross(tx_vec, origin)
        c = np.dot(tx_vec, origin)
        v_x = skew_symm_cp(v)
        # this is the rotation matrix to map vector a onto vector b
        rot_mat = np.identity(3) + v_x + (v_x @ v_x) * (1 / (1 + c))

        shift_voxels = voxels - trans_pos[None, ...]

        origin_voxels = (rot_mat @ shift_voxels.T).T

        # now check if origin voxels fall within the cone given by the transducer beamwdith.
        in_fov_index = np.where(np.sqrt(origin_voxels[..., 0] ** 2 + origin_voxels[..., 1] ** 2) <=
                                   np.abs(origin_voxels[..., 2]) * np.tan(bw))

        in_fov_voxels = voxels[in_fov_index]

        return in_fov_index, in_fov_voxels

    """This function uses the beamwidth of RX/TX to backproject to voxels as the array flys over the scene."""
    def beamform(self, array_data, ping_data, show_geometry=False, show_array=False, receiver_indeces=None,
                 min_ping=0, max_ping=1e9):
        tx_bw = math.radians(array_data['tx_bw'])
        rx_bw = math.radians(array_data['rx_bw'])

        for ping_number, ping in enumerate(ping_data.pings):
            if ping_number < min_ping:
                continue
            if ping_number > max_ping:
                break

            print("Beamforming ping", ping_number + 1, "of", len(ping_data.pings))
            ping = ping_data.pings[ping_number]
            # wfm shape is [num_rx_channels, num_samples] after the transpose
            wfms = ping['mf_raw'].T
            assert wfms.shape[0] == self.SP.num_rx

            array_data = self.AP.snap_array_to_world(array_data, ping_number, show_array=show_array)

            # Get the tx position and its direction vector
            tx_pos = array_data['tx_pos_trans'][ping['tx_id'] - 1]
            tx_vec = array_data['tx_vector']

            # get the vector of rx_positions
            rx_pos = array_data['rx_pos_trans']
            rx_vec = array_data['rx_vector']

            # find the voxels within the fov of the transmitter.
            in_tx_fov_index, in_tx_fov_voxels = self._find_voxels_within_fov(tx_pos, tx_vec, self.voxels, tx_bw)

            scene_subset = np.zeros((len(in_tx_fov_voxels)), dtype=np.complex64)
            norm_subset = np.zeros((len(in_tx_fov_voxels)), dtype=np.ushort)

            # Distance from tx to all fov voxels # sum([1, 3] - [num_pixels, 3], axis=1) = [num_pixels]
            # Compute this outside the loop since loop needs a subset of this every iteration
            dist1_tot = np.sqrt(np.sum((tx_pos[None, :] - in_tx_fov_voxels) ** 2, axis=1))

            rx_indeces = np.arange(0, len(rx_pos))
            if receiver_indeces is not None:
                rx_indeces = receiver_indeces

            # For the set of voxels within the tx FOV, find the voxels within each RX fov
            for channel in rx_indeces:
                rx = rx_pos[channel]
                # of the voxels within the tx fov, find the voxels within the rx fov. Note that these indeces are with
                # respect to the new array in_tx_fov_voxels and this needs to be accounted for.
                in_rx_fov_index, in_rx_fov_voxels = self._find_voxels_within_fov(rx, rx_vec, in_tx_fov_voxels, rx_bw)

                # debugging purposes
                if show_geometry:
                    plot_scene_geometry(tx_pos, rx_pos, self.edges, selected=in_rx_fov_voxels)

                # only lookup time of flights to voxels within rx FOV
                dist1 = dist1_tot[in_rx_fov_index]
                # sum([1, 3] - [num_pixels, 3], axis=1) = [num_pixels]
                dist2 = np.sqrt(np.sum((rx[None, :] - in_rx_fov_voxels)**2, axis=1))

                tof = (dist1 + dist2)/self.SP.c * self.SP.Fs
                # nearest neighbor sampling
                tof = tof.astype(int)

                # account for phase term since data is basebanded
                phase_term = np.exp(1j*2*np.pi*self.SP.Fc*(dist1 + dist2)/self.SP.c)

                # Coherently combine measurements to our scene subset
                scene_subset[in_rx_fov_index] = scene_subset[in_rx_fov_index] + wfms[channel, tof]*phase_term

                # Keep track of how many times we write to a voxel for normalization purposes
                norm_subset[in_rx_fov_index] = norm_subset[in_rx_fov_index] + 1

            # Add the scene subset to the full scene
            self.scene[in_tx_fov_index] = self.scene[in_tx_fov_index] + scene_subset
            self.normalization_counts[in_tx_fov_index] = self.normalization_counts[in_tx_fov_index] + norm_subset

        return self.scene, self.normalization_counts
