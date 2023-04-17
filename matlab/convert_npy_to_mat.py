import numpy as np
import scipy.io
import pickle

"""I commented out the original ones"""
"""I used upsample_network.py to generate the upsampled scenes"""
scene = np.load('/home/albert/SINR_Stuff/code_release/scenes/airsas/arma_20k/npb_output/arma_20k_release_2/numpy/comp_albedo25000.npy')
print(scene.shape)

data = {
    'scene': scene,
}

with open('./data.pickle', 'wb') as handle:
    pickle.dump(data, handle)

scipy.io.savemat('./data.mat', data)
