import os
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import normalize_vector


"""This class defines the array geometry and implements functions for snapping the array to the world grid"""
class ArrayPosition:
    def __init__(self, root_path, track_id, image_number):
        self.root_path = root_path
        self.track_id = track_id
        self.xml_path = self.track_id + '.xml'
        self.image_number = image_number
        base_name = 'motion solution - LF - down - '
        ftype = '.hdf5'
        self.motion_path = base_name + str(self.image_number) + ftype

        self.rx_pos = None
        self.tx_pos = None

        # TODO read these from xml file
        self.rx_bw = 50
        self.tx_bw = 25

        with h5py.File(os.path.join(self.root_path, self.track_id, self.motion_path), 'r') as f:
            motion = np.array(f['Position'][:])

        full_motion = [np.array([x[0], x[1], x[2]]) for x in motion]
        self.full_motion = np.array(full_motion)
        # Account for roll, pitch yaw
        with h5py.File(os.path.join(self.root_path, self.track_id, self.motion_path), 'r') as f:
            roll = np.array(f['Attitude']['Roll'][:])

        self.roll = roll
        with h5py.File(os.path.join(self.root_path, self.track_id, self.motion_path), 'r') as f:
            pitch = np.array(f['Attitude']['Pitch'][:])

        self.pitch = pitch

        with h5py.File(os.path.join(self.root_path, self.track_id, self.motion_path), 'r') as f:
            yaw = np.array(f['Attitude']['Yaw'][:])

        self.yaw = yaw

    # Roll, pitch, yaw in radians
    """This function returns an attidue matrix given roll, pitch, and yaw in radians"""
    def attitude_matrix(self, roll, pitch, yaw):
        phi = roll
        theta = pitch
        gamma = yaw

        # https://www.wolframalpha.com/input?i=%28%7B%7B1%2C0%2C0%7D%2C%7B0%2Ccos+phi%2Csin+phi%7D%2C%7B0%2C-
        # sin+phi%2C+cos+phi%7D%7D*%7B%7Bcos+theta%2C0%2C-sin+theta%7D%2C%7B0%2C1%2C0%7D%2C%7Bsin+theta%2C0%2Ccos+
        # theta%7D%7D*%7B%7Bcos+psi%2C+sin+psi%2C0%7D%2C+%7B-sin+psi%2C+cos+psi%2C0%7D%2C+%7B0%2C0%2C1%7D%7D
        matrix = np.array([
            [np.cos(theta)*np.cos(gamma), np.cos(theta)*np.sin(gamma), -np.sin(theta)],

            [np.sin(theta)*np.cos(gamma)*np.sin(phi) - np.sin(gamma)*np.cos(phi),
             np.sin(theta)*np.sin(gamma)*np.sin(phi) + np.cos(gamma)*np.cos(phi), np.cos(theta)*np.sin(phi)],

            [np.sin(theta)*np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.sin(phi),
             np.sin(theta)*np.sin(gamma)*np.cos(phi) - np.cos(gamma)*np.sin(phi), np.cos(theta)*np.cos(phi)]
        ])

        return matrix

    """This function uses the motion file to snap the array to the voxels given the array geometry and ping number"""
    def snap_array_to_world(self, array, ping_number, show_array=False):
        normal = np.array([0, 0, -1])
        matrix = self.attitude_matrix(self.roll[ping_number],
                                      self.pitch[ping_number],
                                      self.yaw[ping_number])
        offset = self.full_motion[ping_number, :]

        # Transform the rx/tx coordinates with matrix about the origin
        # [3, 3] x [3, num_rx]
        array['rx_pos_trans'] = (matrix @ array['rx_pos'].T).T + offset
        # [3, 3] X [3, num_tx]
        array['tx_pos_trans'] = (matrix @ array['tx_pos'].T).T + offset

        dir_vec = matrix @ normal

        # unit norm direction vectors
        array['rx_vector'] = normalize_vector(dir_vec[None, ...]).squeeze()
        array['tx_vector'] = normalize_vector(dir_vec[None, ...]).squeeze()

        if show_array:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(array['tx_pos'][:, 0], array['tx_pos'][:, 1], array['tx_pos'][:, 2], c='b', label='TX')
            ax.scatter(array['rx_pos'][:, 0], array['rx_pos'][:, 1], array['rx_pos'][:, 2], c='g', label='RX')

            ax.quiver(array['tx_pos'][:, 0], array['tx_pos'][:, 1], array['tx_pos'][:, 2],
                      dir_vec[0] + array['tx_pos'][:, 0], dir_vec[1] + array['tx_pos'][:, 1],
                      dir_vec[2] + array['tx_pos'][:, 2], length=0.01, color='r')

            ax.quiver(array['rx_pos'][:, 0], array['rx_pos'][:, 1], array['rx_pos'][:, 2],
                      dir_vec[0] + array['rx_pos'][:, 0], dir_vec[1] + array['rx_pos'][:, 1],
                      dir_vec[2] + array['rx_pos'][:, 2], length=0.01, normalize=False)

            ax.set_xlabel('Along Track')
            ax.set_ylabel('Cross Track')
            ax.set_zlabel('Depth')
            plt.legend()
            plt.show()

        return array

    # Read TX/RX positions from xml file and snap to world grid
    """This function defines the array geometry using the specs given in the .xml file"""
    def define_array(self, show_array=False, single_real_aperture=False, show_path=False):
        with open(os.path.join(self.root_path, self.track_id, self.xml_path), 'r') as f:
            data = f.read()

        data = BeautifulSoup(data, "html.parser")

        # Read TX positions
        tx_struct = data.find_all('tx')
        tx_struct = str(tx_struct[0])

        x_pos = re.findall(r'(?<=<x>)(.*)(?=</x>)', tx_struct)
        x_pos = np.array([float(x) for x in x_pos])

        y_pos = re.findall(r'(?<=<y>)(.*)(?=</y>)', tx_struct)
        y_pos = np.array([float(y) for y in y_pos])

        z_pos = re.findall(r'(?<=<z>)(.*)(?=</z>)', tx_struct)
        # snap array to imaging grid by adjusting z offset
        z_pos = np.array([float(z) for z in z_pos])

        tx_coords = np.stack((x_pos, y_pos, z_pos), axis=1)

        # Read RX Positions
        rx_struct = data.find_all('rx')
        rx_struct = str(rx_struct[0])

        x_pos = re.findall(r'(?<=<x>)(.*)(?=</x>)', rx_struct)
        x_pos = np.array([float(x) for x in x_pos])

        y_pos = re.findall(r'(?<=<y>)(.*)(?=</y>)', rx_struct)
        y_pos = np.array([float(y) for y in y_pos])

        z_pos = re.findall(r'(?<=<z>)(.*)(?=</z>)', rx_struct)
        # snap array to imaging grid by adjusting z offset
        z_pos = np.array([float(z) for z in z_pos]) #+ z_offset

        rx_coords = np.stack((x_pos, y_pos, z_pos), axis=1)

        if show_array:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(tx_coords[:, 0], tx_coords[:, 1], tx_coords[:, 2], c='b', label='TX')
            ax.scatter(rx_coords[:, 0], rx_coords[:, 1], rx_coords[:, 2], c='g', label='RX')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend()
            plt.show()

        return {'tx_pos':tx_coords, 'rx_pos':rx_coords, 'rx_bw':self.rx_bw, 'tx_bw':self.tx_bw}







