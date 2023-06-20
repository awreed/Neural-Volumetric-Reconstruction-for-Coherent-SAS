import matplotlib.pyplot as plt
from sas_utils import matplotlib_render
import pandas as pd
import point_cloud_utils as pcu
import glob 
from pytorch3d.loss.chamfer import chamfer_distance
import argparse


dev = 'cuda'

from evaluate.point_cloud_utils import *


# Metric 1. : IOU
def calculate_voxel_iou(voxels_gt, points, prefix=""):
    voxels = point_cloud_to_voxel(points, unit=0.005)
    
    intersection = np.sum(np.logical_and(voxels, voxels_gt))
    union = np.sum(np.logical_or(voxels, voxels_gt))
    IoU = intersection / union

    loss = {}
    loss[prefix + "iou"] = IoU
    return loss

# Metric 2. : Chamfer distance
def calculate_chamfer_distance(gt_points_surface, gt_points_volume, points, prefix=""):
    points = torch.Tensor(points).to(dev)
    loss = {}

    # Chamfer distance with surface point clouds
    cham, _ = chamfer_distance(gt_points_surface[None, ...], points[None, ...])
    loss[prefix + "cham"] = cham.item()

    # Chamfer distance with volumetric point clouds
    cham_vol, _  = chamfer_distance(gt_points_volume[None, ...], points[None, ...])
    loss[prefix + "cham_vol"] = cham_vol.item()
    
    return loss


def calculate_3d_loss(args)
    # (1) Load G.T mesh
    # gt_mesh_dir = "../airsas_data/gt_meshes"
    gt_points_surface = np.load("%s/%s_surface.npy" % (args.gt_mesh_dir, args.mesh_name))
    gt_points_volume = np.load("%s/%s_volume.npy" % (args.gt_mesh_dir, args.mesh_name))

    gt_voxels = point_cloud_to_voxel(gt_points_volume, unit=0.005)

    gt_points_surface = torch.Tensor(gt_points_surface).to(dev)
    gt_points_volume = torch.Tensor(gt_points_volume).to(dev)


    default_voxels = load_default_voxels(args.system_data_path)

    # (2) Create output directory
    mesh_output_dir = os.path.join(args.output_dir, "reconstructed_mesh_%.2f" % (args.thresh))
    loss_dict = {}

    # (3) Calculate 3D error for each exps
    
    normal = None
    comp_albedo = np.load(args.comp_albedo_paths)   # load calculated comp_albedo
    
    mag = np.abs(comp_albedo).astype(float)
    mag = (mag - mag.min()) / (mag.max() - mag.min())
    condition = mag > args.thresh
    mag[mag < args.thresh] = 0.
    mag = mag ** 0.2
    
    # Calculate point cloud from comp_albedo
    inferred_points, inferred_normals = voxel_to_point_cloud(default_voxels, condition.ravel(), normal)
    mesh_output_dir_exp = os.path.join(mesh_output_dir, args.expname)
    if not os.path.exists(mesh_output_dir_exp):
        os.makedirs(mesh_output_dir_exp)

    if not os.path.exists(os.path.join(mesh_output_dir_exp, "mesh.obj")):
        # export point cloud to mesh using marching cube (mesh / smoothed mesh) -> we will use smoothe mesh only.
        point_cloud_to_mesh_marching_cube(condition, mesh_output_dir_exp)
        mesh_to_point_cloud(os.path.join(mesh_output_dir_exp, "mesh"), mesh_output_dir_exp)
        mesh_to_point_cloud(os.path.join(mesh_output_dir_exp, "mesh_smooth"), mesh_output_dir_exp)
    else:
        print("Mesh info alreay exists!")

    mesh_points = np.load(os.path.join(mesh_output_dir_exp, "mesh_smooth_surface.npy"))
    mesh_normals = np.load(os.path.join(mesh_output_dir_exp, "mesh_smooth_surface_normal.npy"))
    
    # Chamfer / IOU loss
    # Use two version!
    # A. From point cloud from inferred point using comp albedo
    # B. From point cloud from reconstructed mesh

    cham_loss = calculate_chamfer_distance(gt_points_surface, gt_normals_surface, gt_points_volume, inferred_points, inferred_normals)
    cham_loss_mesh = calculate_chamfer_distance(gt_points_surface, gt_normals_surface, gt_points_volume, mesh_points, mesh_normals, prefix="mesh_")
    iou_loss = calculate_voxel_iou(gt_voxels, inferred_points)
    iou_loss_mesh = calculate_voxel_iou(gt_voxels, mesh_points, prefix="mesh_")
    
    loss_dict[args.expname] = {**cham_loss, **iou_loss, **cham_loss_mesh, **iou_loss_mesh}

    # export all into csv file
    df = pd.DataFrame.from_dict(loss_dict).T

    if args.csv_file_name is None:
        df.to_csv(os.path.join(mesh_output_dir, args.expname, "result_%s.csv" % (args.mesh_name)))
    else:
        df.to_csv(os.path.join(mesh_output_dir, args.expname,"result_%s.csv" % (args.csv_file_name)))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Export 3D point cloud, mesh and 3D loss")
    parser.add_argument('--scene_inr_result_directory', required=True, help="Result folder for inr")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--system_data_path', required=True, help="System data path")
    
    parser.add_argument('--expname', required=True, help="Experiment name")
    parser.add_argument('--mesh_name', required=True, help="Mesh name")
    parser.add_argument('--comp_albedo_path', required=True, help="Complex albedo path")
    parser.add_argument('--csv_file_name', required=False, default=None, help="CSV file name")

    parser.add_argument('--gt_mesh_dir', required=True, help="Ground truth mesh directory")
    parser.add_argument('--thresh', type=float, default=0.2, help="Threshold for reconstructed inr")

    args = parser.parse_args()

    calculate_3d_loss(args)