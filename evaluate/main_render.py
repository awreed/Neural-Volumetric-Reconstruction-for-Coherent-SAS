from evaluate.visualize_render_utils import *
from evaluate.point_cloud_utils import *
from evaluate.predefined_configs import *

import pandas as pd
import lpips
import torch
import glob


def render_gt_mesh(mesh_name, **kwargs):
    index = kwargs.get("index", None)
    gt_mesh_dir = kwargs.get("gt_mesh_dir")

    render_output_dir = kwargs.get('render_output_dir')
    if kwargs.get("is_video", False):
        render_output_dir += "_video"

    render_single_mesh(
        os.path.join(gt_mesh_dir, "%s.obj" % mesh_name), 
        output_dir=render_output_dir, 
        output_file_name="gt_%s_mesh" % mesh_name,
        **kwargs
    )


def render_mesh(mesh_name, expnames, thresh, **kwargs):
    for expname in expnames:
        mesh_output_dir = kwargs.get('mesh_output_dir')
        mesh_dir = os.path.join(mesh_output_dir, "reconstructed_mesh_%.2f/%s" % (thresh, expname))
        render_output_dir = kwargs.get('render_output_dir', mesh_output_dir)
        if kwargs.get("is_video", False):
            render_output_dir += "_video"
        index = kwargs.get("index", None)

        color_dict = kwargs.get("color_dict")

        color = None

        render_single_mesh(
            os.path.join(mesh_dir, "mesh_smooth.obj"), 
            output_dir = render_output_dir,
            output_file_name="inferred_points_mesh",
            **kwargs,
            color=color
        )


if __name__ == "__main__":
    expnames_total_dict = get_expnames()
    camera_poses = get_camera_pose_dict()

    azimuths_image = []
    N = 10
    for azimuth_i in range(N):
        azimuth = azimuth_i * 360 / N
        azimuths_image.append(azimuth)
    
    azimuths_video = []
    N = 360
    for azimuth_i in range(N):
        azimuth = azimuth_i * 360 / N
        azimuths_video.append(azimuth)

    
    target_meshes = ["budda", "dragon", "lucy", "cube", "cylinder", "xyz_dragon", "bunny", "armadilo"]

    common_config = {
        "render_output_dir": "",
        "mesh_output_dir": ""
    }

    # Render gt image
    for mesh in target_meshes:
        camera_pos = camera_poses.get(mesh, camera_poses["default"])
        render_gt_mesh(mesh, w=w, **camera_pos, azimuth = azimuths_image, **common_config)
        render_gt_mesh(mesh, w=w, **camera_pos, azimuth = azimuths_video, is_video=True, **common_config)
    
    # Render exp image
    for mesh in target_meshes:
        for thresh in np.arange(.1, .5, .01):
            camera_pos = camera_poses.get(mesh, camera_poses["default"])
            render_mesh(
                mesh, expnames_total_dict[mesh], thresh = thresh, , **camera_pos,
                azimuth=azimuths_image, **common_config
            )
            render_mesh(
                mesh, expnames_total_dict[mesh], thresh = thresh, , **camera_pos,
                azimuth=azimuths_image, **common_config
            )