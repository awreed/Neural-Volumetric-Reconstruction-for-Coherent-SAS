import open3d as o3d
import numpy as np
import pickle
import constants as c
import mcubes
import trimesh
import os
import torch

def point_cloud_to_voxel(
    points, 
    unit=0.0032, 
    **kwargs
):
    # Initialize a point cloud object
    pcd = o3d.geometry.PointCloud()
    # Add the points, colors and normals as Vectors
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create a voxel grid from the point cloud with a voxel_size of 0.01
    min_x = kwargs.get("min_x", -0.2)
    max_x = kwargs.get("max_x", 0.2)
    min_y = kwargs.get("min_y", -0.2)
    max_y = kwargs.get("max_y", 0.2)
    min_z = kwargs.get("min_z", -0.0)
    max_z = kwargs.get("max_z", 0.3)

    N_x = int((max_x - min_x) / unit)
    N_y = int((max_y - min_y) / unit)
    N_z = int((max_z - min_z) / unit)

    min_bound = np.array([min_x, min_y, min_z])
    max_bound = np.array([max_x, max_y, max_z])
    
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=unit, min_bound=min_bound, max_bound=max_bound)

    voxels = voxel_grid.get_voxels()
    
    voxels_numpy = np.zeros([N_x, N_y, N_z])

    for voxel in voxels:
        grid_idx = voxel.grid_index
        grid_idx_x = min(grid_idx[1], N_y - 1)
        grid_idx_y = min(grid_idx[0], N_x - 1)
        grid_idx_z = min(grid_idx[2], N_z - 1)
        voxels_numpy[grid_idx_x, grid_idx_y, grid_idx_z] = 1
    
    voxels_numpy_temp = np.array(voxels_numpy)
    return voxels_numpy.astype(bool)


def load_default_voxels(system_data_path):
    with open(system_data_path, 'rb') as handle:
        system_data = pickle.load(handle)
    voxels = system_data[c.GEOMETRY][c.VOXELS]
    return voxels


def voxel_to_point_cloud(voxels, condition, normal):
    inferred_points = voxels[condition, :]
    inferred_normals = None
    if normal is not None:
        inferred_normals = normal.reshape([-1, 3])[condition, :]
    
    return inferred_points, inferred_normals

def lerp(x, a, b):
    return (b - a) * x + a

def normalize_vertices(vertices, **kwargs):
    min_x = kwargs.get("min_x", -0.2)
    max_x = kwargs.get("max_x", 0.2)
    min_y = kwargs.get("min_y", -0.2)
    max_y = kwargs.get("max_y", 0.2)
    min_z = kwargs.get("min_z", -0.0)
    max_z = kwargs.get("max_z", 0.3)
    num_x = kwargs.get("num_x", 125)
    num_y = kwargs.get("num_y", 125)
    num_z = kwargs.get("num_z", 94)
    
    vertices_temp = np.array(vertices)
    vertices_temp[:, 0] = lerp(vertices[:, 1] / num_y, min_y, max_y)
    vertices_temp[:, 1] = lerp(vertices[:, 0] / num_x, min_x, max_x)
    vertices_temp[:, 2] = lerp(vertices[:, 2] / num_z, min_z, max_z)
    return vertices_temp

def point_cloud_to_mesh_marching_cube(points, output_dir, **kwargs):
    points = np.pad(points, 1)
    vertices, triangles = mcubes.marching_cubes(points, 0.5)
    vertices = normalize_vertices(vertices, **kwargs)
    mcubes.export_obj(vertices, triangles, os.path.join(output_dir, "mesh.obj"))
    points_smooth = mcubes.smooth(points)
    vertices, triangles = mcubes.marching_cubes(points_smooth, 0)
    vertices = normalize_vertices(vertices, **kwargs)
    mcubes.export_obj(vertices, triangles, os.path.join(output_dir, "mesh_smooth.obj"))

def mesh_to_point_cloud(mesh_file, output_dir, gt_n_points = 20000, gt_n_points_volume = 50000):
    gt_mesh = trimesh.load(mesh_file+".obj", force='mesh')
    gt_points, gt_faces = gt_mesh.sample(gt_n_points, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_faces]
    np.save(mesh_file+"_surface", gt_points)
    np.save(mesh_file+"_surface_normal", gt_normals)