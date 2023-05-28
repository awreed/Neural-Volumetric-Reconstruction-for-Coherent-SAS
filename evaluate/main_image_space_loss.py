import lpips
import torch
import numpy as np
import os 
from collections import OrderedDict
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio

import pandas as pd
import argparse

from evaluate.predefined_configs import *
device = 'cuda'
import argparse


# Calculate averaged image space error (default : 10 azimuths)
def calculate_image_space_error_averaged(mesh_name, expnames, thresh, lpips_loss_fn, target_image="voxel", target_type=None, max_index=10, csv_file_name=None, **kwargs):
    image_dir = kwargs.get('image_dir')
    gt_image_dir = kwargs.get('gt_image_dir')

    v = 1.0 if target_type == "depth" else 255.0

    image_gts = []
    for index in range(max_index):
        file_name = "gt_%s_%s_%d" % (mesh_name, target_image, index)
        if target_type is not None:
            file_name += ("_%s" % target_type)
        image_gt = np.load(os.path.join(gt_image_dir, "%s.npy" % file_name))
        if image_gt.shape[-1] == 3:
            image_gt = image_gt[..., 0]

        image_gts.append(image_gt)
    image_gts = np.asarray(image_gts)
    img0 = torch.Tensor(image_gts).to(device) / v
    img0 = img0[:, None, ...]

    loss_dict_all = OrderedDict()

    mse = torch.nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    for expname in expnames:
        images = []
        for index in range(max_index):
            file_name = "inferred_points_%s_%d" % (target_image, index)
            if target_type is not None:
                file_name += ("_%s" % target_type)
            expdir = os.path.join(image_dir, expname, "%s.npy" % (file_name))
            image = np.load(expdir)
            if image.shape[-1] == 3:
                image = image[..., 0]
            images.append(image)
        images = np.asarray(images)
        img1 = torch.Tensor(images).to(device) / v
        img1 = img1[:, None, ...]
        
        loss_dict = OrderedDict()
        loss_dict["lpips"] = lpips_loss_fn(img0, img1).mean().item()
        loss_dict["mse"] = mse(img0, img1).item()
        loss_dict["ssim"] = ssim(img0, img1).item()
        loss_dict["psnr"] = psnr(img0, img1).item()
        loss_dict_all[expname] = loss_dict

    df = pd.DataFrame.from_dict(loss_dict_all).T

    assert csv_file_name is not None
    if csv_file_name is None:
        csv_file_name = mesh_name

    if target_type is not None:
        file_name = "result_image_error_%s_%s_%s.csv" % (csv_file_name, target_image, target_type)
    else:
        file_name = "result_image_error_%s_%s.csv" % (csv_file_name, target_image)
    
    df.index.name = "expname"
    df.to_csv(os.path.join(image_dir, file_name), index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image space loss evaluation")
    
    parser.add_argument('--n_azimuth', type=int, default=10, help="Number of azimuth")
    parser.add_argument('--mesh_name', required=True, help="Mesh name")
    parser.add_argument('--expname', required=True, help="Expname")
    parser.add_argument('--image_dir', required=True, help="Image directory")
    parser.add_argument('--gt_image_dir', required=True, help="GT Image directory")
    parser.add_argument('--thresh', type=float, default=0.2, help="Threshold for reconstructed inr")

    args = parser.parse_args()

    lpips_loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    expnames_total = get_expnames()

    calculate_image_space_error_averaged(args.mesh_name, args.expname, args.thresh, lpips_loss_fn_vgg, 
        target_image="mesh", target_type="depth", max_index=args.n_azimuth,
        image_dir=args.image_dir,
        gt_image_dir=args.gt_image_dir
        )