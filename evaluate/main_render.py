from evaluate.visualize_render_utils import *
from evaluate.point_cloud_utils import *
from evaluate.predefined_configs import *

import pandas as pd
import lpips
import torch
import glob
import argparse


def render_gt_mesh(**kwargs):
    mesh_name = kwargs.get("mesh_name")
    index = kwargs.get("index", None)
    gt_mesh_dir = kwargs.get("gt_mesh_dir")
    n_azimuth = kwargs.get("n_azimuth")

    render_output_dir = kwargs.get('render_output_dir')
    if kwargs.get("is_video", False):
        render_output_dir += "_video"

    azimuths = []
    for azimuth_i in range(n_azimuth):
        azimuth = azimuth_i * 360 / n_azimuth
        azimuths.append(azimuth)

    render_single_mesh(
        os.path.join(gt_mesh_dir, "%s.obj" % mesh_name), 
        output_dir=render_output_dir, 
        output_file_name="gt_%s_mesh" % mesh_name,
        **kwargs,
        azimuth=azimuths
    )


def render_mesh(thresh, **kwargs):
    mesh_name = kwargs.get("mesh_name")
    expname = kwargs.get("expname")
    mesh_output_dir = kwargs.get('recon_mesh_dir')
    mesh_dir = os.path.join(mesh_output_dir, "reconstructed_mesh_%.2f/%s" % (thresh, expname))
    render_output_dir = kwargs.get('render_output_dir', mesh_output_dir)
    n_azimuth = kwargs.get("n_azimuth")
    
    if kwargs.get("is_video", False):
        render_output_dir += "_video"
    index = kwargs.get("index", None)

    color_dict = kwargs.get("color_dict")

    color = None

    azimuths = []
    for azimuth_i in range(n_azimuth):
        azimuth = azimuth_i * 360 / n_azimuth
        azimuths.append(azimuth)

    render_single_mesh(
        os.path.join(mesh_dir, "mesh_smooth.obj"), 
        output_dir = render_output_dir,
        output_file_name="inferred_points_mesh",
        **kwargs,
        color=color,
        azimuth=azimuths
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Mesh to Image")
    
    parser.add_argument('--n_azimuth', type=int, default=10, help="Number of azimuth")
    parser.add_argument('--mesh_name', required=True, help="Mesh name")
    parser.add_argument('--expname', required=True, help="Expname")
    parser.add_argument('--render_output_dir', required=True, help="Render result output dir")
    parser.add_argument('--recon_mesh_dir', required=True, help="Reconstructed mesh dir")
    parser.add_argument('--gt_mesh_dir', required=True, help="Ground truth mesh directory")
    parser.add_argument('--elevation', type=float, default=0.1, help="Camera elevation")
    parser.add_argument('--distance', type=float, default=0.3, help="Camera distance")
    parser.add_argument('--fov', type=float, default=60, help="Camera fov in degree")
    parser.add_argument('--thresh', type=float, default=0.2, help="Threshold for reconstructed inr")

    args = parser.parse_args()

    # Render gt image
    render_gt_mesh(**vars(args))

    # Render exp image
    render_mesh(**vars(args))