import csv
import glob
import numpy as np
import os
from tqdm import tqdm
import constants as c
import cv2
from data_schemas import SASDataSchema, SysParams, WfmParams
import math

"""Assumes approximately collocated tx/rx"""
def resample_cylindrical(tx_coords,
                         resample_type=c.HELIX,
                         num_per_level=360,
                         pitch_levels=20,
                         skip_every_n=4):
    assert resample_type in [c.HELIX, c.SPARSE]

    if resample_type == c.HELIX:
        print("Helix sampling")
        num_tx_orig, _ = tx_coords.shape

        min_z = tx_coords[0, 2]
        max_z = tx_coords[-1, 2]

        radius = np.sqrt(tx_coords[0, 0]**2 + tx_coords[0, 1]**2)
        num_z = num_tx_orig / num_per_level - 1
        z_step = (max_z - min_z)/num_z

        # https://en.wikipedia.org/wiki/Helix
        b = (pitch_levels*z_step) / (2 * np.pi)
        # multiply b by 2*pi and divide into total height to get number of turns
        num_turns = math.ceil((max_z - min_z)/(b * 2 * np.pi))
        # t is given by number of turns and number of samples
        t = 2*np.pi*np.linspace(0, num_turns, int(num_per_level*(num_z+1)))

        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = b * t + min_z

        xyz = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)
        # Crop the coordinates that are out of range.
        xyz = xyz[xyz[:, 2] <= tx_coords[:, 2].max(), :]
        # Find the nearest measurement we have to helix coordinate
        all_dists = []
        helical_indeces = []
        for i in range(xyz.shape[0]):
            xyz_query = xyz[i, :]

            z = xyz_query[2]

            start_index = int(num_per_level * np.round((z - min_z)/z_step))

            candidate_coords = tx_coords[start_index:start_index+num_per_level, :]

            dists = np.sqrt(np.sum((xyz_query[None, :] - candidate_coords)**2, axis=-1))

            min_index = np.argmin(dists)
            min_dist = dists[min_index]

            all_dists.append(min_dist)

            tx_coord_min_index = start_index + min_index

            if tx_coord_min_index not in helical_indeces:
                helical_indeces.append(tx_coord_min_index)

        helical_indeces = np.array(helical_indeces).astype(np.long)

        #tx_coords_helical = tx_coords[helical_indeces, :]

        return helical_indeces
    elif c.SPARSE:
        print("Sparse sampling")
        all_indeces = np.arange(tx_coords.shape[0])

        print("Doing this thing")
        print(tx_coords.shape)
        print(all_indeces)

        sparse_indeces = all_indeces[::skip_every_n]

        print(sparse_indeces)

        return sparse_indeces

    else:
        raise OSError("Resample type on understood.")


def load_img_and_preprocess(path, size=(200, 200), rotate=False):
    assert len(size) == 2
    img = cv2.imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (size[0], size[1]))

    if rotate:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    return img


"""Helper for reading waveform parameters"""

def mat_str_2_float(s):
    s = s.split('E')
    base = float(s[0])

    pos=None
    if '+' in s[1]:
        pos=True
    elif '-' in s[1]:
        pos=False
    else:
        raise Exception('Could not parse waveform parameters from AirSAS. Check the parser logic.')
    if pos:
        exp = float(s[1].strip('+'))
        ans = base * 10 ** (exp)
    else:
        exp = float(s[1].strip('-'))
        ans = base * 10 ** (-exp)

    return ans



"""
Read in the flight data
"""


def read_flights(path, flights, flight_indeces):
    # count the number of samples
    num_samples = 0
    with open(flights[0]) as fp:
        reader = csv.reader(fp)
        for _ in reader:
            num_samples = num_samples + 1

    wfm_data = []

    # read the flights into a numpy array, hopefully not run out of memory
    for flight_num in tqdm(range(1, len(flights) + 1), desc='Reading flights'):
        if flight_num in flight_indeces:
            flight_file = "Flight-%06d.csv" % (flight_num)
            with open(os.path.join(path, flight_file)) as fp:
                reader = csv.reader(fp)
                single_wfm = []
                for sample, row in enumerate(reader):
                    single_wfm.append(float(row[0]))

                wfm_data.append(single_wfm)

    wfm_data = np.stack(np.stack(wfm_data, axis=0), axis=0)

    return wfm_data


"""
Read and store the system parameters
"""


def read_sys_params(system_parameters_file):
    tx_pos = np.zeros((3))
    rx_pos = np.zeros((3))
    center = np.zeros((3))
    group_delay = None
    fs = None
    set_tx = 0
    set_rx = 0
    set_center = 0

    with open(system_parameters_file) as fp:
        reader = csv.reader(fp)
        for row in reader:
            if row[0] == 'Speaker x':
                tx_pos[0] = float(row[1])
                set_tx = set_tx + 1
            if row[0] == 'Speaker y':
                tx_pos[1] = float(row[1])
                set_tx = set_tx + 1
            if row[0] == 'Speaker z':
                tx_pos[2] = float(row[1])
                set_tx = set_tx + 1
            if row[0] == 'Mic1 x':
                rx_pos[0] = float(row[1])
                set_rx = set_rx + 1
            if row[0] == 'Mic1 y':
                rx_pos[1] = float(row[1])
                set_rx = set_rx + 1
            if row[0] == 'Mic1 z':
                rx_pos[2] = (row[1])
                set_rx = set_rx + 1

            if row[0] == 'Group Delay':
                group_delay = float(row[1])

            if row[0] == 'Fs':
                fs = float(row[1])

            if row[0] == 'Center x':
                center[0] = float(row[1])
                set_center = set_center + 1
            if row[0] == 'Center y':
                center[1] = float(row[1])
                set_center = set_center + 1
            if row[0] == 'Center z':
                center[2] = float(row[1])
                set_center = set_center + 1

        assert group_delay is not None, "Failed to read group delay from " + system_parameters_file
        assert fs is not None, "Failed to read sampling rate from " + system_parameters_file
        assert set_tx == 3, "Failed to read tx position from " + system_parameters_file
        assert set_rx == 3, "Failed to read rx position from" + system_parameters_file
        assert set_center == 3, "Failed to read center position from " + system_parameters_file

    sys_params = SysParams()
    sys_params[c.TX_POS] = tx_pos
    sys_params[c.RX_POS] = rx_pos
    sys_params[c.CENTER] = center
    sys_params[c.GROUP_DELAY] = group_delay
    sys_params[c.FS] = fs

    return sys_params


"""
Read the coordinates.csv file and transform tx, rx position to world using flight angle
"""


def read_coords(coords_file, sys_params, read_only_wfm):
    # [rx/tx, flight_index, xyz[]
    tx_coords = []
    rx_coords = []
    temps = []

    flight_indeces = []

    with open(coords_file) as fp:
        reader = csv.reader(fp)
        array_index = 0
        for i, row in tqdm(enumerate(reader), desc='Reading coordinates'):
            if i > 0:
                flight_index = int(float(row[0]))
                angle = float(row[1])
                height = float(row[2])
                temp = float(row[3])

                if read_only_wfm is not None:
                    wfm_type = int(float(row[4]))
                    if read_only_wfm != wfm_type:
                        continue

                # Sanity check
                assert flight_index == i

                rot_mat = np.array(([np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.],
                                    [0., 0., 1.]))

                height_offset = np.array(([0., 0., height/1000.]))

                # Compute tx coordinates relative to the scene center
                tx_coords.append((rot_mat @ sys_params[c.TX_POS]) - sys_params[c.CENTER] + height_offset)
                rx_coords.append((rot_mat @ sys_params[c.RX_POS]) - sys_params[c.CENTER] + height_offset)
                temps.append(temp)
                flight_indeces.append(flight_index)

        tx_coords = np.stack(tx_coords, axis=0)
        rx_coords = np.stack(rx_coords, axis=0)
        temps = np.array(temps)

    return {
        c.TX_COORDS: tx_coords,
        c.RX_COORDS: rx_coords,
        c.TEMPS: temps,
    }, flight_indeces

"""
Extract info neccessary for beamforming AirSAS data
"""


def process_folder(airsas_folder, use_wfm_cache=True, use_coords_cache=True, read_only_wfm=None):
    # Coordinates.csv contains the coordinates for each flight (Flight #, Angle, Height, Temp, Waveform)
    # SysParams.csv contains system parameters. (Speaker xyz, Mic 1 xyz, Mic 2 xyz, Center xyz, Fs, GD, Other stuff
    # WaveformParams.csv contains waveform parameters
    # Waveforms.csv contains the waveform

    files = glob.glob(os.path.join(airsas_folder, '*'))
    flights = [x for x in files if 'Flight' in x]
    config_files = [x for x in files if 'Flight' not in x]

    # Find the system parameter and coordinates files
    sp_file = [x for x in config_files if 'SysParams' in x][0]
    coords_file = [x for x in config_files if 'Coordinates' in x][0]
    wfm_params_file = [x for x in config_files if 'WaveformParams' in x][0]
    wfm_file = [x for x in config_files if 'Waveforms.csv' in x][0]

    sys_params = read_sys_params(sp_file)

    with open(wfm_params_file) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if read_only_wfm is not None:
                if i != read_only_wfm:
                    continue
                else:
                    wfm_params = row[0]
                    break
            else:
                wfm_params = row[0]
            break

    wfm_params = wfm_params.split('_')
    f_start = int(mat_str_2_float(wfm_params[1].replace('Hz', '')))
    f_stop = int(mat_str_2_float(wfm_params[2].replace('Hz', '')))
    t_dur = float(mat_str_2_float(wfm_params[3].replace('s', '')))
    win_ratio = float(mat_str_2_float(wfm_params[6]))

    wfm_params = WfmParams()
    wfm_params[c.F_START] = f_start
    wfm_params[c.F_STOP] = f_stop
    wfm_params[c.T_DUR] = t_dur
    wfm_params[c.WIN_RATIO] = win_ratio

    wfm = []

    with open(os.path.join(wfm_file)) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if read_only_wfm is not None:
                wfm.append(float(row[read_only_wfm]))
            else:
                wfm.append(float(row[0]))

    wfm = np.array(wfm)

    if use_coords_cache and os.path.exists(os.path.join(airsas_folder, c.TX_COORDS_FILE)):
        assert read_only_wfm is None, "Cannot load from cache if using specific waveform."
        if os.path.exists(os.path.join(airsas_folder, c.TX_COORDS_FILE)):
            print("Loading cached coordinates from ", os.path.join(airsas_folder, c.TX_COORDS_FILE))
            tx_coords = np.load(os.path.join(airsas_folder, c.TX_COORDS_FILE))
            rx_coords = np.load(os.path.join(airsas_folder, c.RX_COORDS_FILE))
            temps = np.load(os.path.join(airsas_folder, c.TEMPS_FILE))
    else:
        print("Loading coordinates from scratch...")
        coords, flight_indeces = read_coords(coords_file, sys_params, read_only_wfm=read_only_wfm)
        tx_coords = coords[c.TX_COORDS]
        rx_coords = coords[c.RX_COORDS]
        temps = coords[c.TEMPS]
        np.save(os.path.join(airsas_folder, c.TX_COORDS_FILE), tx_coords)
        np.save(os.path.join(airsas_folder, c.RX_COORDS_FILE), rx_coords)
        np.save(os.path.join(airsas_folder, c.TEMPS_FILE), temps)

    if use_wfm_cache and os.path.exists(os.path.join(airsas_folder, c.WFM_FILE)):
        assert read_only_wfm is None, "Cannot load from cache if using specific waveform."
        print("Loading cached waveforms from ", os.path.join(airsas_folder, c.WFM_FILE))
        wfm_data = np.load(os.path.join(airsas_folder, c.WFM_FILE))
    else:
        print("Loading waveforms from scratch...")
        wfm_data = read_flights(airsas_folder, flights, flight_indeces)
        # np.save(os.path.join(airsas_folder, c.WFM_FILE), wfm_data)

    airsas_data = SASDataSchema()
    airsas_data[c.WFM_DATA] = wfm_data
    airsas_data[c.TX_COORDS] = tx_coords
    airsas_data[c.RX_COORDS] = rx_coords
    airsas_data[c.TEMPS] = temps
    airsas_data[c.SYS_PARAMS] = sys_params
    airsas_data[c.WFM_PARAMS] = wfm_params
    airsas_data[c.WFM] = wfm

    return airsas_data