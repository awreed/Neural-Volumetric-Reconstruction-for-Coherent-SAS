import pywavefront
import pyrender
import pyrr
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import ColorMap
import pyqtgraph.opengl as gl
import open3d as o3d
import constants as c
from evaluate.point_cloud_utils import *
from PIL import Image


def get_look_at(position, elevation_at):
    m = pyrr.matrix44.create_look_at(
        eye=np.array(position), 
        target=np.array([0.0, 0.0, elevation_at]),
        up=np.array([0.0, 0.0, 1.0]))
    m = pyrr.matrix44.inverse(m)
    return np.asarray(m).T

def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append((
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            matrix
        ))

    return nodes
    
def render_mesh_using_pyrender(mesh_path, **kwargs):
    pass

def render_voxel_using_pyqtgraph(scene, w, **kwargs):
    # pg.mkQApp()

    elevation = kwargs.get('elevation')
    distance = kwargs.get('distance')
    azimuth = kwargs.get('azimuth')
    fov = kwargs.get('fov')
    w.setCameraPosition(pos=QtGui.QVector3D(0, 0, elevation), distance=distance, elevation=elevation, azimuth=azimuth)

    d2 = np.zeros(scene.shape + (4,), dtype=np.ubyte)
    # print(d2.shape, rgba_data.shape)
    d2[..., 3] = scene * 255

    d2[..., 0] = d2[..., 3]
    d2[..., 1] = d2[..., 3]
    d2[..., 2] = d2[..., 3]

    min_x = kwargs.get("min_x", -0.2)
    max_x = kwargs.get("max_x", 0.2)
    min_y = kwargs.get("min_y", -0.2)
    max_y = kwargs.get("max_y", 0.2)
    min_z = kwargs.get("min_z", -0.0)
    max_z = kwargs.get("max_z", 0.3)
    num_x = kwargs.get("num_x", 125)
    num_y = kwargs.get("num_y", 125)
    num_z = kwargs.get("num_z", 94)
    d2 = d2.astype(np.ubyte)

    v = gl.GLVolumeItem(d2, glOptions='translucent')
    v.scale((max_x - min_x) / d2.shape[0], (max_y - min_y) / d2.shape[1], (max_z - min_z) / d2.shape[2])
    v.translate(min_x, min_y, min_z)
    w.addItem(v)
    
    d = w.renderToArray((512, 512))
    # Convert to QImage
    tmp = pg.makeQImage(d)  # .save('custom3d_data/' + str(i) + '.png')
    # Convert to np array
    tmp = pg.imageToArray(tmp, copy=True, transpose=True)
    
    w.removeItem(v)
    return tmp


def render_single_mesh(mesh_path, output_dir=None, output_file_name=None, **kwargs):
    elevation = kwargs.get('elevation')
    elevation_to = kwargs.get('elevation_to', elevation)
    distance = kwargs.get('distance')
    azimuth = kwargs.get('azimuth')
    fov = kwargs.get('fov')
    color = kwargs.get('color', None)

    # load mesh
    tm = trimesh.load(mesh_path)
    if color is not None:
        tm.visual.vertex_colors = np.tile(color, (tm.vertices.shape[0], 1))
    trimesh.repair.fix_inversion(tm)
    
    mesh = pyrender.Mesh.from_trimesh(tm)
    mesh_normal = pyrender.Mesh.from_trimesh(tm)
    for primitive in mesh_normal.primitives:
        normals_temp = np.array(primitive.normals)
        normals_temp[:, 1] *= -1
        primitive.color_0 = (normals_temp + 1) * 0.5
    
    scene = pyrender.Scene()
    scene_normal = pyrender.Scene(ambient_light=[1., 1., 1.])

    scene.add(mesh)
    scene_normal.add(mesh_normal)

    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    light = pyrender.PointLight(color=np.ones(3), intensity=0.5)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(camera_node)
    scene.add_node(light_node)
    scene_normal.add_node(camera_node)
    # scene_normal.add_node(light_node)

    r = np.sqrt(distance * distance - elevation * elevation)
    if not isinstance(azimuth, list):
        azimuth = [azimuth]
    renderer = pyrender.OffscreenRenderer(512, 512)

    # render for each azimuths
    for i, a in enumerate(azimuth):
        x = r * np.cos((-a + 90) / 180 * np.pi)
        y = r * np.sin((-a + 90) / 180 * np.pi)
        camera_pose = get_look_at([x, y, elevation], elevation_to)
        scene.set_pose(camera_node, camera_pose)
        scene.set_pose(light_node, camera_pose)
        scene_normal.set_pose(camera_node, camera_pose)
        color, depth = renderer.render(scene)

        # flags = pyrender.constants.RenderFlags.FLAT

        normal, _ = renderer.render(scene_normal)

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            np.save(os.path.join(output_dir, "%s_%d_color.npy" % (output_file_name, i)), color)
            np.save(os.path.join(output_dir, "%s_%d_depth.npy" % (output_file_name, i)), depth)
            np.save(os.path.join(output_dir, "%s_%d_normal.npy" % (output_file_name, i)), normal)
            Image.fromarray(color).save(os.path.join(output_dir, "%s_%d_color.png" % (output_file_name, i)))
            Image.fromarray(normal).save(os.path.join(output_dir, "%s_%d_normal.png" % (output_file_name, i)))
        
