import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui
from PyQt6.QtWidgets import QApplication

#from PyQt5 import QApplication
from pyqtgraph import ColorMap


import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mcubes
import scipy.interpolate

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

# ffmpeg

def render_frame(scene, w, xdim=256, ydim=256, translate_x=-50, translate_y=-50, translate_z=-50, normals=None):
    # pg.mkQApp()

    d2 = np.zeros(scene.shape + (4,), dtype=np.ubyte)
    # print(d2.shape, rgba_data.shape)
    d2[..., 3] = scene * 255


    #color = np.array([[0., 0., 0.], [43, 0, 0], [103, 37, 20], [199, 155, 97], [216, 213, 201]])
    #intensity = np.linspace(0, 256, color.shape[0])
    #intensity = np.array([0, 20, 30, 225, 256])

    #alpha = 5.0* np.array([0., 0., 0.15, 0.3, 1.0]) * 255
    #print(intensity.shape, color.shape, d2.shape)

    #interp_fn = scipy.interpolate.interp1d(intensity, color, axis=0)

    #alpha_interp_fn = scipy.interpolate.interp1d(intensity, alpha, axis=0)

    #x, y, z = d2[..., 3].shape

    #colors = interp_fn(d2[..., 3].ravel())

    #colors = np.reshape(colors, (x, y, z, 3))

    #print(colors)
    #print("c_min, c_max", colors.min(), colors.max())

    #alphas = alpha_interp_fn(d2[..., 3].ravel())

    #alphas = np.reshape(alphas, (x, y, z))
    #print(alphas.min(), alphas.max())
    #print("alphas shape", alphas.shape)
    #print(colors.shape)
    #exit(0)

    if normals is None:
        #d2[..., 0] = d2[..., 3]
        #d2[..., 1] = d2[..., 3]
        #d2[..., 2] = d2[..., 3]
        d2[..., 0] = d2[..., 3]
        d2[..., 1] = d2[..., 3]
        d2[..., 2] = d2[..., 3]
    else:
        d2[..., 0] = normals[..., 0] * 255
        d2[..., 1] = normals[..., 1] * 255
        d2[..., 2] = normals[..., 2] * 255

    # (optional) RGB orientation lines
    # d2[:40, 0, 0] = [255, 0, 0, 255]
    # d2[0, :40, 0] = [0, 255, 0, 255]
    # d2[0, 0, :40] = [0, 0, 255, 255]
    d2 = d2.astype(np.ubyte)

    v = gl.GLVolumeItem(d2, smooth=True, sliceDensity=1)
    v.translate(translate_x, translate_y, translate_z)
    #v.setGLOptions('translucent')
    w.addItem(v)

    pg.setConfigOption('background', (255, 255, 255, 100))
    d = w.renderToArray((xdim, ydim))
    # Convert to QImage
    tmp = pg.makeQImage(d)  # .save('custom3d_data/' + str(i) + '.png')
    # Convert to np array
    tmp = pg.imageToArray(tmp, copy=True, transpose=True)

    return tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render numpy arrays using PyQtGraph")
    parser.add_argument('--scene', required=True, help="Path to input numpy array path, should be 3 dimensions.")
    parser.add_argument('--norm_sigma', required=False, default=None, help="Path to input numpy array path, should be 3 dimensions.")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--video_name', required=False, default='movie.avi', help="Video name")
    parser.add_argument('--image_name', required=False, default='image.png', help="Output image name")
    parser.add_argument('--render_x', required=True, type=int, help="X render dimension")
    parser.add_argument('--render_y', required=True, type=int, help="Y render dimension")
    parser.add_argument('--x', required=False, type=int, help="X dimension")
    parser.add_argument('--y', required=False, type=int, help="Y dimension")
    parser.add_argument('--z', required=False, type=int, help="Z dimension")
    parser.add_argument('--thresh', required=False, default=0., type=float,
                        help="Linear threshold [0-1] to filter out points.")
    parser.add_argument('--distance', required=False, default=100, type=int, help="Camera distance [0-360] deg.")
    parser.add_argument('--elevation', required=False, default=45, type=int, help="Camera elevation [0-360] deg.")
    parser.add_argument('--elevation_width', required=False, default=None, type=int)
    parser.add_argument('--azimuth', required=False, default=90, type=int, help="Camera azimuth [0-360] deg.")
    parser.add_argument('--make_video', action='store_true', help="Whether to make video")
    parser.add_argument('--azimuth_min', type=int, default=0)
    parser.add_argument('--azimuth_max', type=int, default=360)
    parser.add_argument('--azimuth_step', type=int, default=1)
    parser.add_argument('--phase', required=False, action='store_true', default=False)
    parser.add_argument('--normals', required=False, default=None)
    parser.add_argument('--drc', required=False, type=float, help="Between [0, 1]. Values closer to 0 amplify small values."
                                                                "1 yields no change. ")
    parser.add_argument('--real', required=False, action='store_true', help="Only plot real value")
    parser.add_argument('--imag', required=False, action='store_true', help="Only plot imag value")
    parser.add_argument('--flip_z', required=False, action='store_true')
    parser.add_argument('--translate_x', required=False, default=-50, type=int)
    parser.add_argument('--translate_y', required=False, default=-50, type=int)
    parser.add_argument('--translate_z', required=False, default=-50, type=int)
    parser.add_argument('--mesh_name', required=False, default=None,
                        help="Path to export mesh")

    args = parser.parse_args()

    assert args.drc >= 0
    assert args.drc <= 1

    if args.phase:
        assert args.normals is None

    if args.real:
        assert not args.imag

    if args.imag:
        assert not args.real

    ##app = QtWidgets.QApplication([])

    #app = QtGui.QApplication([])
    app = QApplication([])
    #app = Application([])
    w = gl.GLViewWidget()
    w.setCameraPosition(distance=args.distance, elevation=args.elevation, azimuth=args.azimuth)
    w.show()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    scene = np.load(args.scene)

    if args.flip_z:
        scene = np.flip(scene, axis=-1)

    print("Loaded scene with shape", scene.shape)
    if scene.ndim == 1:
        scene = np.reshape(scene, (args.x, args.y, args.z))

    if args.normals is not None:
        normals = np.load(args.normals)

        normals = np.reshape(normals, (scene.shape[0], scene.shape[1], scene.shape[2], 3))
    else:
        normals = None

    if args.norm_sigma is not None:
        norm_sigma = np.load(args.norm_sigma)
        norm_sigma = np.reshape(norm_sigma, (args.x, args.y, args.z))
        scene = scene * norm_sigma

    assert scene.ndim == 3, print(scene.shape)

    #print("Loaded scene with shape (", args.x, args.y, args.z, ")")

    if args.phase:
        scene_phase = np.angle(scene)

        scene = np.abs(scene)
        scene = (scene - scene.min()) / (scene.max() - scene.min())
        scene[scene < args.thresh] = 0.
        #scene = scene ** (args.drc / 1)

        scene = scene * scene_phase
    elif args.real:
        scene = np.abs(np.real(scene))
        scene = (scene - scene.min()) / (scene.max() - scene.min())
        scene[scene < args.thresh] = 0.
        #scene = scene ** (args.drc / 1)
    elif args.imag:
        scene = np.abs(np.imag(scene))
        scene = (scene - scene.min()) / (scene.max() - scene.min())
        scene[scene < args.thresh] = 0.
        #scene = scene ** (args.drc / 1)
    elif args.normals is not None:

        if normals.dtype == complex:
            normals = (normals.real + normals.imag) / 2
            normals = normals.real

        normals = (normals + 1)/2

        #normals = (normals - normals.min()) / (normals.max() - normals.min())

        scene = np.abs(scene)
        scene = (scene - scene.min()) / (scene.max() - scene.min())

        scene[scene < args.thresh] = 0.
        scene = scene ** (args.drc / 1)
    else:
        scene = np.abs(scene)
        scene = (scene - scene.min()) / (scene.max() - scene.min())
        scene[scene < args.thresh] = 0.

        scene = scene ** (args.drc / 1)

    if args.mesh_name is not None:
        print("Marching cubes to export mesh")
        vertices, triangles = mcubes.marching_cubes(scene, 0)

        mcubes.export_mesh(vertices, triangles, os.path.join(args.output_dir, args.mesh_name), "Bunny")
        print("Mesh exported to ", os.path.join(args.output_dir, args.mesh_name))

    # Think this should happen after we export a mesh.


    angles = np.arange(args.azimuth_min, args.azimuth_max, args.azimuth_step)
    single_image = render_frame(scene, w, xdim=args.render_x, ydim=args.render_y,
                                translate_x=args.translate_x,
                                translate_y=args.translate_y,
                                translate_z=args.translate_z,
                                normals=normals)

    print(single_image[0, 0, :])

    #single_image[...] = 255 - single_image[...]

    #cond = (single_image[..., -1] == 255)#

    #single_image[cond, 0:4] = np.array([255, 255, 255, 0])

    #cv2.imwrite(os.path.join(args.output_dir, args.image_name), single_image[..., 0:3])



    #print(sign)

    #print(single_image.shape)
    #print(single_image[100:120, 100:120, :])
    #exit(0)

    plt.figure()
    if args.phase:
        plt.imshow(single_image[..., 0], cmap='hsv')
    elif args.normals is not None:
        plt.imshow(single_image[...])
    else:
        plt.imshow(single_image[..., 0])
        #plt.colorbar()
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(args.output_dir, args.image_name), dpi=600)

    if args.make_video:
        fig = plt.figure(figsize=(5, 5), dpi=125)
        ax = fig.add_subplot(1, 1, 1)

        width, height = fig.get_size_inches() * fig.get_dpi()

        del fig, ax

        if args.elevation_width is not None:
            min_elevation = args.elevation - args.elevation_width
            max_elevation = args.elevation + args.elevation_width

            num_frames = len(angles)

            num_iter = num_frames

            up = np.linspace(min_elevation, max_elevation, int(num_iter//2))
            down = np.linspace(max_elevation, min_elevation, int(num_iter//2))

            full = np.concatenate((up, down), axis=-1)

            assert full.shape == angles.shape
        else:
            full = np.ones_like(angles) * args.elevation

        images = []
        for ang, elev in tqdm(zip(angles, full), desc="Rendering movie"):
            w.setCameraPosition(distance=args.distance, elevation=elev, azimuth=ang)
            image = render_frame(scene, w, xdim=args.render_x, ydim=args.render_y,
                                 translate_x=args.translate_x,
                                 translate_y=args.translate_y,
                                 translate_z=args.translate_z,
                                 normals=normals)


            fig = plt.figure(figsize=(5, 5), dpi=125)
            ax = fig.add_subplot(1, 1, 1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            if args.phase:
                im = ax.imshow(image[..., 0], cmap='hsv')
            elif args.normals is not None:
                im = ax.imshow(image[...])
            else:
                im = ax.imshow(image[..., 0])

            fig.colorbar(im, cax=cax)
            canvas = FigureCanvas(fig)
            canvas.draw()

            tmp = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

            images.append(tmp)
            del fig, ax

        video = cv2.VideoWriter(os.path.join(args.output_dir, args.video_name),
                                         0, 5, (int(width), int(height)))

        for image_i in tqdm(images, desc="Writing video"):
            video.write(cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()

    #QtGui.QApplication.instance().exec_()



