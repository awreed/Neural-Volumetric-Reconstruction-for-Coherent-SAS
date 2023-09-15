from system_parameters import SystemParameters
from array_position import ArrayPosition
from scene import DefineScene
from SERDPBeamformer import SERDPBeamformer
import numpy as np
import os

# create a save directory
save_dir = 'sas_scenes'

# Point this to a directory containing all elements and motions/coinv2_imagery
root_path = './data/dataForAlbert/_delete30Apr2022/asasinOutput/allElementAndMotions/COINv2_Imagery'
track_id = '2019 1106 163841'
sound_speed_table_path = './data/dataForAlbert/_delete30Apr2022/navData/soundSpeedTable.csv'
image_number = 2
receiver_indeces = np.arange(68, 80, 1)
#receiver_indeces = np.array([74])
min_ping = 50
max_ping = 180

y_min = 30 * .02
y_max = 15 * .02
x_min = .5
x_max = 1.5
resolution = np.array([.01, .01, .01])

print("Setup waveform and pick ping")
SP = SystemParameters(root_path, track_id, image_number, sound_speed_table_path)
kernel = SP.gen_kernel(kernel=SP.read_kernel())


# if find_ping, then we show the raw and match-filtered time series
ping_data = SP.process_waveforms(find_ping=True)

print("Defining the Array position")
AP = ArrayPosition(root_path, track_id, image_number)
array_data = AP.define_array()

print('Defining the scene voxels')
scene = DefineScene(root_path, track_id, image_number)


# Pass in the defined array data
# Remove y_min, y_max, z_min, z_max and it will automatically set the scene dimensions based off of vehicle track
voxels, edges = scene.create_voxels_for_sas(array_data, min_depth=.75, max_depth=1.5,
                                            ct_range_override=2.0, show_path=False,
                                            y_min=y_min, y_max=y_max,
                                            x_min=x_min, x_max=x_max,
                                            resolution=resolution)

print(scene.xdim, scene.ydim, scene.zdim)
BF = SERDPBeamformer(SP, AP, voxels, edges=edges)

# show array option will show the array at each ping
# show geometry will show the scene geometry at each rx recieve
# The code will pause when the plots appear --- it will continue only when you close them
# Thus, keep these show options set to false if you actually want to beamform something
complex_scene, normalization_counts = BF.beamform(array_data, ping_data, show_geometry=False, show_array=False,
                                                  receiver_indeces=receiver_indeces,
                                                  min_ping=min_ping, max_ping=max_ping)

complex_scene = np.reshape(complex_scene, (scene.xdim, scene.ydim, scene.zdim))
normalization_counts = np.reshape(normalization_counts, (scene.xdim, scene.ydim, scene.zdim))

np.save(os.path.join(save_dir, 'complex_scene.npy'), complex_scene)
np.save(os.path.join(save_dir, 'normalization_counts.npy'), normalization_counts)
