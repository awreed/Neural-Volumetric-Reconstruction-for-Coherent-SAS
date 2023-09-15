import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'sas_scenes_out'
scene_file = '../scenes/serdp_cinder/scene_data/das/numpy/scene.npy'
#norm_file = '../scenes/serdp_cinder/scene_data/das/numpy/normalization_counts.npy'

vmin, vmax = -3, 0
thresh = False

scene = np.load(scene_file)
xdim, ydim, zdim = scene.shape

#norm_count = np.load(norm_file)

scene_abs = np.abs(scene)
scene_real = np.real(scene)
scene_imag = np.imag(scene)
#scene_abs = (scene_abs - scene_abs.min())/(scene_abs.max() - scene_abs.min())
#scene_abs = drc(scene_abs, np.median(scene_abs), 0.2)

#scene_abs = 20*np.log10(scene_abs + 1e-8)

for i in range(0, zdim):
    slice = str(i)
    fig = plt.figure()
    if thresh:
        plt.imshow(scene_abs[..., i], cmap='jet', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(scene_abs[..., i], cmap='jet')
    plt.title("Depth slice" + slice)
    plt.colorbar(label='dB')
    plt.xlabel('Cross Track')
    plt.ylabel('Along Track')
    plt.savefig(os.path.join(save_dir, 'z' + str(i) + '.png'))
    plt.close(fig)
    plt.clf()

#for i in range(0, zdim):
#    slice = str(i)
#    fig = plt.figure()
#    if thresh:
#        plt.imshow(scene_real[..., i], cmap='jet', vmin=vmin, vmax=vmax)
#    else:
#        plt.imshow(scene_real[..., i], cmap='jet')
#    plt.title("Depth slice" + slice)
#    plt.colorbar(label='dB')
#    plt.xlabel('Cross Track')
#    plt.ylabel('Along Track')
#    plt.savefig(os.path.join(save_dir, 'z_real' + str(i) + '.png'))
#    plt.close(fig)
#    plt.clf()

#for i in range(0, zdim):
#    slice = str(i)
#    fig = plt.figure()
#    if thresh:
#        plt.imshow(scene_imag[..., i], cmap='jet', vmin=vmin, vmax=vmax)
#    else:
#        plt.imshow(scene_imag[..., i], cmap='jet')
#    plt.title("Depth slice" + slice)
#    plt.colorbar(label='dB')
#    plt.xlabel('Cross Track')
#    plt.ylabel('Along Track')
#    plt.savefig(os.path.join(save_dir, 'z_imag' + str(i) + '.png'))
#    plt.close(fig)
#    plt.clf()

for i in range(0, xdim):
    slice = str(i)
    fig = plt.figure()
    if thresh:
        plt.imshow((scene_abs[i, ...]).T, cmap='jet', vmin=vmin, vmax=vmax)
    else:
        plt.imshow((scene_abs[i, ...]).T, cmap='jet')
    plt.title("Along Track Slice" + slice)
    plt.colorbar(label='dB')
    plt.xlabel('Cross Track')
    plt.ylabel('Depth')
    plt.savefig(os.path.join(save_dir, 'x' + str(i) + '.png'))
    plt.close(fig)
    plt.clf()

for i in range(0, ydim):
    slice = str(i)
    fig = plt.figure()
    if thresh:
        plt.imshow((scene_abs[:, i, :]).T, cmap='jet', vmin=vmin, vmax=vmax)
    else:
        plt.imshow((scene_abs[:, i, :]).T, cmap='jet')

    plt.title("Cross Track Slice" + slice)
    plt.colorbar(label='dB')
    plt.xlabel('Along Track')
    plt.ylabel('Depth')
    plt.savefig(os.path.join(save_dir, 'y' + str(i) + '.png'))
    #plt.savefig('scenes/' + 'y' + str(i) + '.png')
    plt.close(fig)
    plt.clf()

"""
norm_abs = scene_abs / (norm_count + 1)

for i in range(0, zdim):
    slice = str(i)
    fig = plt.figure()
    plt.imshow(norm_abs[..., i], cmap='jet')#, vmin=vmin, vmax=vmax)
    plt.title("Depth slice" + slice)
    plt.colorbar(label='dB')
    plt.xlabel('Cross Track')
    plt.ylabel('Along Track')
    plt.savefig(os.path.join(save_dir, 'z_norm' + str(i) + '.png'))
    plt.close(fig)
    plt.clf()
"""
print(scene.shape)
