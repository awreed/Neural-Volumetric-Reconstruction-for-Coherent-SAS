import trimesh
import numpy as np
import argparse


def export_gt_point_cloud(args):
    # Load G.T mesh
    gt_mesh = trimesh.load(args.mesh_file+".obj", force='mesh')
    gt_points, gt_faces = gt_mesh.sample(args.gt_n_points, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_faces]
    gt_points_volume = trimesh.sample.volume_mesh(gt_mesh, args.gt_n_points_volume)

    np.save(args.mesh_file+"_surface_%d" % args.gt_n_points, gt_points)
    np.save(args.mesh_file+"_surface_normal_%d" % args.gt_n_points, gt_normals)
    np.save(args.mesh_file+"_volume_%d" % args.gt_n_points_volume, gt_points_volume)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Export point cloud from ground truth mesh")
    parser.add_argument('--mesh_file', required=True, help="Mesh file directory")
    parser.add_argument('--gt_n_points', type=int, default=20000, help="Number of points for surface point cloud")
    parser.add_argument('--gt_n_points_volume', type=int, default=50000, help="Number of points for volume point cloud")
    args = parser.parse_args()
    export_gt_point_cloud(args)