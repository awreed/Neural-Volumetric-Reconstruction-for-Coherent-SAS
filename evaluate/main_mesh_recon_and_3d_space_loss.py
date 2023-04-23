import matplotlib.pyplot as plt
from sas_utils import matplotlib_render
import pandas as pd
import point_cloud_utils as pcu
import glob 
from pytorch3d.loss.chamfer import chamfer_distance

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


def calculate_3d_loss(
        mesh_name, 
        expnames, 
        thresh,
        csv_file_name=None,
        scene_inr_result_directory=None,
        output_dir=None,
        comp_albedo_paths=None,
        system_data_path=None
    ):

    # (1) Load G.T mesh
    gt_mesh_dir = "../airsas_data/gt_meshes"
    gt_points_surface = np.load("%s/%s_surface.npy" % (gt_mesh_dir, mesh_name))
    gt_points_volume = np.load("%s/%s_volume.npy" % (gt_mesh_dir, mesh_name))

    gt_voxels = point_cloud_to_voxel(gt_points_volume, unit=0.005)

    gt_points_surface = torch.Tensor(gt_points_surface).to(dev)
    gt_points_volume = torch.Tensor(gt_points_volume).to(dev)


    default_voxels = load_default_voxels(system_data_path)

    # (2) Create output directory
    mesh_output_dir = os.path.join(output_dir, "reconstructed_mesh_%.2f" % (thresh))
    loss_dict = {}

    # (3) Calculate 3D error for each exps
    for expname in expnames:
        normal = None
        comp_albedo = np.load(comp_albedo_paths[expname])   # load calculated comp_albedo
        
        mag = np.abs(comp_albedo).astype(float)
        mag = (mag - mag.min()) / (mag.max() - mag.min())
        condition = mag > thresh
        mag[mag < thresh] = 0.
        mag = mag ** 0.2
        
        # Calculate point cloud from comp_albedo
        inferred_points, inferred_normals = voxel_to_point_cloud(default_voxels, condition.ravel(), normal)
        mesh_output_dir_exp = os.path.join(mesh_output_dir, expname)
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
        
        loss_dict[expname] = {**cham_loss, **iou_loss, **cham_loss_mesh, **iou_loss_mesh}

    # export all into csv file
    df = pd.DataFrame.from_dict(loss_dict).T

    if csv_file_name is None:
        df.to_csv(os.path.join(mesh_output_dir, "result_%s.csv" % (mesh_name)))
    else:
        df.to_csv(os.path.join(mesh_output_dir, "result_%s.csv" % (csv_file_name)))


if __name__=="__main__":
    bunny_expnames = []
    comp_albedo_paths = {}
    for k in [5, 10, 20]:
        for n in [0, 10, 20]:
            bunny_expnames.append("bunny_%dk_%ddb" % (k, n))
            bunny_expnames.append("bunny_%dk_%ddb_no_network" % (k, n))
            bunny_expnames.append("bunny_%dk/%ddb/das" % (k, n))

            comp_albedo_paths["bunny_%dk_%ddb" % (k, n)] = ""
            comp_albedo_paths["bunny_%dk_%ddb_no_network" % (k, n)] = ""
            comp_albedo_paths["bunny_%dk/%ddb/das" % (k, n)] = ""
            
    for thresh in np.arange(.1, .5, .01):
        calculate_3d_loss(
            "bunny", bunny_expnames, thresh=thresh, 
            scene_inr_result_directory="../experiments/figure_10/scenes/scene_inr_result",
            output_dir="../reconstructed_mesh",
            comp_albedo_paths=comp_albedo_paths,
            csv_file_name="bunny",
            system_data_path = "../experiments/figure_10/scenes/configs/bunny_20k/10db/system_data_5k.pik"
        )