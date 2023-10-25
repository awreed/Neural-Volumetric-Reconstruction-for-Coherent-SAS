import argparse
import torch
import commentjson as json
from utils import divide_chunks
import os
import glob
import numpy as np
import pickle
import constants as c
from utils import aggressive_crop_weights, normalize
from sas_utils import hilbert_torch, figure_to_tensorboard, safe_normalize, matplotlib_render, range_normalize, find_indeces_within_scene
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
from network import Network
import shutil
from argument_io import directory_cleaup
from forward_model import scattering_model, transmission_model
from sampling import SceneSampler, normalize_vectors
import time
import math
from torch.utils.tensorboard import SummaryWriter
from reconstruct_scene_parser import *
from logging_utils import *
from timing_utils import *
from render_utils import set_axes_equal, _set_axes_radius
import pdb

if __name__ == '__main__':
    ###################################
    # (0) torch initialization
    ###################################
    torch.autograd.set_detect_anomaly(True)
    assert torch.cuda.is_available()
    dev = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logger = load_logger("Info logger")

    ###################################
    # (1) parse from config
    ###################################
    with time_measure("[1] Parse from config"):
        parser = config_parser()
        args = parser.parse_args()

        num_rays = int(math.sqrt(args.num_rays)) ** 2

        num_importance_sampling_rays = None
        if args.importance_sampling_rays is not None:
            num_importance_sampling_rays = int(math.sqrt(args.importance_sampling_rays)) ** 2

        if num_importance_sampling_rays is not None:
            logger.info("Using %d sampling rays and %d importance sampling rays" % (num_rays,
                                                                                    num_importance_sampling_rays))
        else:
            logger.info(
                "Using %d sampling rays and 0 importance sampling rays" % (num_rays))
        logger.info("Beamwidth is %s" % str(args.beamwidth))

        # assert few properties
        assert args.sampling_distribution_uniformity <= 1.00001
        assert args.sampling_distribution_uniformity >= 0.
        if args.k_normal_per_ray is not None:
            assert args.k_normal_per_ray == 1, "Only support 1 normal per ray for now"

        with open(args.scene_inr_config) as config_file:
            inr_config = json.load(config_file)

        # only runs with --clear_output_directory flag

        # make directories for output
        expname = args.expname
        if expname is None:
            raise OSError("Failed to read the --expname argument")
            expname = os.getcwd()
            expname = expname.split("/")[-1]
        basedir = os.path.join(args.output_dir, expname)

        logger.info("Deleting directory")
        directory_cleaup(args.output_dir, args.clear_output_directory)

        if not os.path.exists(basedir):
            os.makedirs(basedir)
        if not os.path.exists(os.path.join(basedir, c.IMAGES)):
            os.makedirs(os.path.join(basedir, c.IMAGES))
        if not os.path.exists(os.path.join(basedir, c.NUMPY)):
            os.makedirs(os.path.join(basedir, c.NUMPY))
        if not os.path.exists(os.path.join(basedir, 'models')):
            os.makedirs(os.path.join(basedir, 'models'))

        writer = SummaryWriter(log_dir=basedir)

        # export config file
        logger.info("Saving input arguments to " + os.path.join(basedir, 'commandline_args.txt'))
        # Place in main path
        with open(os.path.join(basedir, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        # place in image path in case I only download this directory
        with open(os.path.join(basedir, c.IMAGES, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(os.path.join(basedir, c.NUMPY, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    ###################################
    # (2) Load data and process
    ###################################
    with time_measure("[2] Data Load and Process"):
        with open(args.system_data, 'rb') as handle:
            system_data = pickle.load(handle)

        corners = torch.from_numpy(system_data[c.GEOMETRY][c.CORNERS]).to(dev)
        if args.no_factor_4:
            corners_scaled = corners
        else:
            corners_scaled = 1.5*(corners + .05)

        all_scene_coords = torch.from_numpy(system_data[c.GEOMETRY][c.VOXELS])
        scene_scale_factor = 1
        if args.no_factor_4:
            scene_scale_factor = 1 / (all_scene_coords.abs().max())
        else:
            scene_scale_factor = 1 / (4 * all_scene_coords.abs().max())
        if args.normalize_scene_dims:
            all_scene_coords = all_scene_coords * scene_scale_factor
            corners_scaled = corners_scaled * scene_scale_factor
            #corners = torch.from_numpy(corners).to(dev) * scene_scale_factor

        NUM_X = system_data[c.GEOMETRY][c.NUM_X]
        NUM_Y = system_data[c.GEOMETRY][c.NUM_Y]
        NUM_Z = system_data[c.GEOMETRY][c.NUM_Z]

        voxel_size_x = torch.abs(all_scene_coords[:, 0].max() - all_scene_coords[:, 0].min())/NUM_X
        voxel_size_y = torch.abs(all_scene_coords[:, 1].max() - all_scene_coords[:, 1].min())/NUM_Y
        voxel_size_z = torch.abs(all_scene_coords[:, 2].max() - all_scene_coords[:, 2].min())/NUM_Z

        voxel_size_avg = torch.mean(torch.tensor([voxel_size_x, voxel_size_y, voxel_size_z]))
        voxel_size_avg = torch.sqrt(voxel_size_avg**2 + voxel_size_avg**2)

        #print(all_scene_coords.shape)
        #print(all_scene_coords[:, 0].min(), all_scene_coords[:, 0].max())
        #print(all_scene_coords[:, 1].min(), all_scene_coords[:, 1].max())
        #print(all_scene_coords[:, 2].min(), all_scene_coords[:, 2].max())
        #exit(0)

        scene_voxels=None
        if args.no_network:
            assert args.normalize_scene_dims is True
            print("No Network")
            x_padded = torch.arange(all_scene_coords[:, 0].min()*(1/scene_scale_factor),
                                    all_scene_coords[:, 0].max()*(1/scene_scale_factor),
                                    voxel_size_x)
            y_padded = torch.arange(all_scene_coords[:, 1].min()*(1/scene_scale_factor),
                                    all_scene_coords[:, 1].max()*(1/scene_scale_factor),
                                    voxel_size_y)
            z_padded = torch.arange(all_scene_coords[:, 2].min()*(1/scene_scale_factor),
                                    all_scene_coords[:, 2].max()*(1/scene_scale_factor),
                                    voxel_size_z)

            # Zeros will mess up the distribution sampling
            scene_voxels = 1e-8*torch.randn((x_padded.shape[0], y_padded.shape[0], z_padded.shape[0]),
                                       dtype=torch.complex64)[None, None, ...]

            del x_padded
            del y_padded
            del z_padded

            print("Scene voxels shape", scene_voxels.shape)

            #exit(0)

            #scene_voxels = torch.stack(torch.meshgrid(x_padded, y_padded, z_padded)).permute(1, 2, 3, 0)[None, ...]

        try:
            tx_bw = torch.tensor([system_data[c.SYS_PARAMS][c.TX_BW]]).to(dev)
            assert args.bw_units is not None, "Detected a set TX beamwidth in system_data file. " \
                                              "Specifify with --bw_units whether this is degrees or radians"
            if args.bw_units == 'r':
                tx_bw = torch.rad2deg(tx_bw)

        except AssertionError:
            tx_bw = None
        except KeyError:
            tx_bw = None

        # override the system file
        if args.beamwidth is not None:
            tx_bw = torch.tensor([args.beamwidth])

        if args.two_dimensions:
            # Assert z value of corners is the same if want to only use two dimensions
            assert (corners[..., 2][0] == corners[..., 2][1:]).all()

        # The exception supports older code functionality
        try:
            speed_of_sound = system_data[c.SOUND_SPEED]
        except KeyError:
            temp = np.mean(system_data[c.TEMPS])
            speed_of_sound = 331.4 + 0.6 * temp

        weight_paths = glob.glob(os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*'))

        # Older das.sh code puts in different directory, so check:
        if len(weight_paths) == 0:
            weight_paths = glob.glob(os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*'))

        assert len(weight_paths) > 0, "Failed to load weights from " + \
                                      os.path.join(args.fit_folder, c.NUMPY, c.WEIGHT_PREFIX + '*')

        wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]
        # Sort weight paths so they align with tx indeces
        num_radial = None
        if args.use_mf_wfms:
            weights = system_data[c.WFM_RC]
            if args.use_up_to is not None:
                weights = weights[:args.use_up_to*360, :]
            weights = weights[:, wfm_crop_settings[c.MIN_SAMPLE]:
                                wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]]
            print("Using MF wfms", weights.shape)
            num_radial = weights.shape[-1]
        else:
            weight_paths = sorted(weight_paths, key=lambda x: int(x.split('_')[-2]))
            if args.use_up_to is not None:
                weight_paths = weight_paths[0:args.use_up_to]

            # Store all weights in RAM?
            weights = []

            for index in tqdm(range(len(weight_paths)), desc="Reading weight paths INR"):
                weight_path = weight_paths[index]
                split_weight_path = weight_path.split('_')
                start_index = int(split_weight_path[-2])
                stop_index = int(split_weight_path[-1].split('.')[0]) + 1
                weight = np.load(weight_path)
                num_radial = weight.shape[-1]

                weights.append(np.load(weight_path))

            weights = np.concatenate((weights), axis=0)

            # Convert weight to complex analytic signal
            weights = torch.from_numpy(weights)
            comp_weights = torch.zeros_like(weights, dtype=torch.complex64)
            for i in tqdm(range(weights.shape[0]), desc="Converting weights to complex"):
                comp_weights[i, :] = hilbert_torch(weights[i, :].float())

            weights = comp_weights.detach().cpu().numpy()
            logger.info("Loaded deconvolved weights %s" % str(weights.shape))

        new_min_sample, new_max_sample = aggressive_crop_weights(
            tx_coords=torch.from_numpy(system_data[c.TX_COORDS]).to(dev),
            rx_coords=torch.from_numpy(system_data[c.RX_COORDS]).to(dev),
            corners=corners,
            old_min_dist=wfm_crop_settings[c.MIN_DIST],
            old_max_dist=wfm_crop_settings[c.MAX_DIST],
            num_radial=num_radial)

        weights = weights[:, new_min_sample:new_max_sample]
        print("Weights", weights.shape)
        distribution = np.abs(weights)**(args.sampling_distribution_uniformity/1) / \
                        np.sum(np.abs(weights)**(args.sampling_distribution_uniformity/1), axis=-1)[..., None]

        indeces = np.arange(0, weights.shape[-1], 1)

        dists_norm = torch.linspace(0, 1, num_radial*args.upsample)
        dists_scene = wfm_crop_settings[c.MIN_DIST] + \
                      dists_norm * (wfm_crop_settings[c.MAX_DIST] - wfm_crop_settings[c.MIN_DIST])
        dists_scene = dists_scene[new_min_sample*args.upsample:new_max_sample*args.upsample].to(dev)

        min_delta_dist = torch.abs(dists_scene[0] - dists_scene[1])/2

        perturb_radii = None
        if args.perturb_radii:
            perturb_radii = min_delta_dist

    start = 0

    ###################################
    # (3) Create NeRF Model
    ###################################
    with time_measure("[3] NeRF Load"):

        scene_model = Network(inr_config=inr_config,
                              dev=dev,
                              num_layers=args.num_layers,
                              num_neurons=args.num_neurons,
                              scene_voxels=scene_voxels,
                              incoherent=args.incoherent,
                              real_only=args.real_only)

        scene_optimizer = torch.optim.Adam(scene_model.get_params(args.learning_rate), lr=args.learning_rate)

        # Load checkpoints
        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            model_path = os.path.join(basedir, 'models')
            ckpts = [os.path.join(model_path, f) for f in sorted(os.listdir(model_path)) if 'tar' in f]

        logger.info('Found ckpts: %s' % str(ckpts))

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            logger.info('Reloading from %s' % str(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start = ckpt['global_step'] + 1

            # Load model
            scene_model.load_state_dict(ckpt['network_fn_state_dict'])
            scene_optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    random.seed(0)
    global_step = start

    #assert tx_bw is not None

    ###################################
    # (4) Main train loop
    ###################################

    # Create Sampler
    # If beamwidth is not none, then this will compute the sparse rays.
    scene_sampler = SceneSampler(num_dense_rays=num_rays,
                                 num_sparse_rays=num_importance_sampling_rays,
                                 max_distance=wfm_crop_settings[c.MAX_DIST],
                                 beamwidth=tx_bw,
                                 device=dev
                                 )
    compute_normals = True
    if args.skip_normals:
        compute_normals = False

    if tx_bw is not None:
        print("TX Beamwidth is", tx_bw.detach().cpu().numpy(), "degrees")

    if args.sequential_sample:
        print("Sampling in order")
    else:
        print("Sampling random")

    if args.real_only:
        weights = np.real(weights)

    if args.incoherent:
        weights = np.abs(weights)

    #pdb.set_trace()

    dists_scene_cropped = None
    ts = []
    for epoch in range(start, args.num_epochs):
        #a = time.time()
        if args.sequential_sample:
            rand_batch = global_step % weights.shape[0]
        else:
            rand_batch = random.sample(range(weights.shape[0]), 1)

        weight_batch = weights[rand_batch].squeeze()
        dist_batch = distribution[rand_batch].squeeze()

        if args.max_weights < dist_batch.shape[-1]:
            index_batch = np.sort(np.random.choice(indeces, size=args.max_weights, replace=False, p=dist_batch))
        else:
            index_batch = np.arange(0, dist_batch.shape[-1])

        gt_weight_cropped = torch.from_numpy(weight_batch[index_batch]).to(dev) * args.scale_factor

        if gt_weight_cropped.abs().max() < args.thresh:
            continue

        dists_scene_cropped_before = dists_scene[index_batch]

        if args.perturb_sampling:
            perturb = torch.from_numpy(np.random.uniform(low=-1, high=1,
                                                         size=dists_scene_cropped_before.shape)).\
                to(dists_scene_cropped_before.device)
            dists_scene_cropped = dists_scene_cropped_before + min_delta_dist*perturb
        else:
            dists_scene_cropped = dists_scene_cropped_before

        # Skip using the weight if values don't beat thresh
        if gt_weight_cropped.shape[0] == 0:
            continue

        tx_batch = torch.from_numpy(system_data[c.TX_COORDS][rand_batch])
        rx_batch = torch.from_numpy(system_data[c.RX_COORDS][rand_batch])

        # Exceptions are to support older versions of das.sh that don't set this key
        try:
            tx_vec = torch.from_numpy(system_data[c.TX_VECS][rand_batch]).to(dev).squeeze()
        except AssertionError:
            tx_vec = None
        except KeyError:
            tx_vec = None

        if args.flip_z:
            assert tx_vec.ndim == 1
            tx_vec[-1] = -tx_vec[-1]

        sampling_distribution=None
        if args.importance_sampling_rays is not None and global_step > args.reg_start:
            # If beamwidth is not none, then this just loads the sparse rays
            vec_to, dir_to, model_out = scene_sampler.ellipsoidal_sampling(
                radii=dists_scene_cropped,
                tx_pos=tx_batch.to(dev),
                rx_pos=rx_batch.to(dev),
                cache_vectors=True,
                num_rays=num_importance_sampling_rays,
                scene_bounds=corners,
                tx_vec=tx_vec,
                create_return_vec=False,
                debug_dir=None,
                point_at_center=args.point_at_center,
                transmit_from_tx=args.transmit_from_tx,
                device=dev)

            if args.normalize_scene_dims:
                vec_to = vec_to * scene_scale_factor

            num_to_rad, num_to_rays, _ = vec_to.shape

            #if args.normalize_scene_dims:
            #    assert vec_to.min().item() >= -1., print(vec_to.min().item())
            #    assert vec_to.max().item() <= 1., print(vec_to.max().item())

            # Can't compute normals outside of model for some reason...
            # So pass coords_to and coords_back to model. Just compute normal on coords_to
            model_out = scene_model(coords_to=vec_to,
                                    compute_normals=False)

            scatterers_sparse = model_out['scatterers_to']
            scatterers_sparse = scatterers_sparse.reshape(num_to_rad, num_to_rays)
            transmission_probs = transmission_model(radii=dists_scene_cropped,
                                                    scatterers_to=scatterers_sparse,
                                                    occlusion_scale=args.occlusion_scale)
            # [int(sqrt(num_importance_sampling_rays)), int(sqrt(num_importance_sampling_rays))]
            # We will use this to importance sample
            sampling_distribution = torch.sum(transmission_probs*scatterers_sparse.abs(), dim=0)

        debug_dir = None
        if global_step % args.info_every == 0:
            debug_dir = os.path.join(basedir, c.NUMPY, 'debug' + str(global_step) + '.npy')

        vec_to, dir_to, model_out = scene_sampler.ellipsoidal_sampling(
            radii=dists_scene_cropped,
            tx_pos=tx_batch.to(dev),
            rx_pos=rx_batch.to(dev),
            num_rays=num_rays,
            scene_bounds=corners,
            tx_vec=tx_vec,
            create_return_vec=args.ray_trace_return,
            point_at_center=args.point_at_center,
            distribution=sampling_distribution,
            debug_dir=debug_dir,
            transmit_from_tx=args.transmit_from_tx,
            scene_model=scene_model,
            transmission_model=transmission_model,
            occlusion_scale=args.occlusion_scale,
            compute_normals=compute_normals,
            scene_scale_factor=scene_scale_factor,
            device=dev)

        if args.normalize_scene_dims:
            vec_to = vec_to * scene_scale_factor

        num_to_rad, num_to_rays, _ = vec_to.shape

        #if args.normalize_scene_dims:
        #    assert vec_to.min().item() >= -1., print(vec_to.min().item())
        #    assert vec_to.max().item() <= 1., print(vec_to.max().item())

        ######################
        # (4.2) Forward model
        #######################

        # If this is False then we already computed all these things in the scene sampler
        if not args.ray_trace_return:
            # Can't compute normals outside of model for some reason...
            # So pass coords_to and coords_back to model. Just compute normal on coords_to
            model_out = scene_model(coords_to=vec_to,
                                    compute_normals=compute_normals)

            scatterers_to = model_out['scatterers_to']
            normals = model_out['normals']

            scatterers_to = scatterers_to.reshape(num_to_rad, num_to_rays)
            if compute_normals:
                normals = normals.reshape(num_to_rad, num_to_rays, 3)

            transmission_probs = transmission_model(radii=dists_scene_cropped,
                                                    scatterers_to=scatterers_to,
                                                    occlusion_scale=args.occlusion_scale,
                                                    factor2=True)

        # already computed by scene sampler
        else:
            scatterers_to = model_out['scatterers_to']
            normals = model_out['normals']
            transmission_probs = model_out['transmission_probs']

            scatterers_to = scatterers_to.reshape(num_to_rad, num_to_rays)
            if compute_normals:
                normals = normals.reshape(num_to_rad, num_to_rays, 3)

        # We only integrate the to vectors (since they are definitely within TX beamwidth.)
        if normals is not None:
            # Add lambertian scattering
            scatterers_lamb = scattering_model(
                scatterers=scatterers_to,
                normals=normals,
                dir_vec=-dir_to,
                ratio=args.lambertian_ratio)
        else:
            scatterers_lamb = scatterers_to

        scatterers_to_integrate = scatterers_lamb * transmission_probs

        # Integrate rays along ellipsoid
        estimated_weights = torch.sum(scatterers_to_integrate, dim=-1)

        if args.norm_weights:
            complex_color_integrated = estimated_weights / \
                                       torch.sqrt(torch.mean(estimated_weights*torch.conj(estimated_weights)))
            gt_weight_cropped = gt_weight_cropped / \
                                torch.sqrt(torch.mean(gt_weight_cropped*torch.conj(gt_weight_cropped)))

        #######################
        # (4.3) Calculate loss
        #######################

        if args.l1_loss:
            if args.incoherent or args.real_only:
                weight_loss = torch.nn.functional.l1_loss(estimated_weights.float().squeeze(),
                                                          gt_weight_cropped.float().squeeze(), reduction='mean')

            else:
                weight_loss = torch.nn.functional.l1_loss(estimated_weights.real.float().squeeze(),
                                                            gt_weight_cropped.real.float().squeeze(), reduction='mean') \
                              + \
                                  torch.nn.functional.l1_loss(estimated_weights.imag.float().squeeze(),
                                                            gt_weight_cropped.imag.float().squeeze(), reduction='mean')
        else:
            if args.incoherent or args.real_only:
                weight_loss = torch.nn.functional.mse_loss(estimated_weights.float().squeeze(),
                                                           gt_weight_cropped.float().squeeze(), reduction='mean')
            else:
                weight_loss = torch.nn.functional.mse_loss(estimated_weights.real.float().squeeze(),
                                                          gt_weight_cropped.real.float().squeeze(), reduction='mean') \
                              + \
                              torch.nn.functional.mse_loss(estimated_weights.imag.float().squeeze(),
                                                          gt_weight_cropped.imag.float().squeeze(), reduction='mean')

        weight_loss = weight_loss.clip(max=1e1)

        #if args.normal_loss is not None and overall_count > args.reg_start:
        #    print("Min_delta_dist", min_delta_dist)
        #    normals_perturb = scene_model.normal(coords_sample + .001*torch.rand_like(coords_sample))
        #    normals_perturb = normals_perturb.reshape(num_rad, num_theta, num_phi, 3)
        #    normal_loss = args.normal_loss * torch.mean((normals - normals_perturb).abs())
        #else:
        #    normal_loss = torch.tensor([0.]).to(weight_loss.device)
        if args.smooth_weight is not None:
            smooth_weight_loss = torch.mean(torch.abs(estimated_weights.abs().float()[1:] -
                                                      estimated_weights.abs().float()[:-1]))
        else:
            smooth_weight_loss = torch.tensor([0.]).to(scatterers_to.device)

        if args.tv_loss is not None and global_step > args.reg_start:
            tv_loss = args.tv_loss * torch.mean(torch.mean(torch.abs(scatterers_to[1:, ...] -
                                                      scatterers_to[:-1, ...]), dim=-1), dim=-1)
        else:
            tv_loss = torch.tensor([0.]).to(scatterers_to.device)

        if args.sparsity is not None and global_step > args.reg_start:
            sparsity_loss = args.sparsity * torch.mean(torch.abs(scatterers_to.abs() + 1e-10))
        else:
            sparsity_loss = torch.tensor([0.]).to(weight_loss.device)

        if (args.smooth_loss is not None or args.phase_loss is not None) and (global_step > args.reg_start):
            coords_perturb = vec_to + args.smooth_delta*voxel_size_avg*normalize_vectors(torch.randn_like(vec_to))

            perturb_out = scene_model(coords_to=coords_perturb,
                                    compute_normals=False)

            perturb_out = perturb_out['scatterers_to']

            if args.incoherent or args.real_only:
                if args.smooth_loss is not None:
                    smooth_loss = args.smooth_loss * torch.mean(
                        torch.nn.functional.l1_loss(scatterers_to.reshape(-1).squeeze(), perturb_out.squeeze()))
                else:
                    smooth_loss = torch.tensor([0.]).to(scatterers_to.device)

                phase_loss = torch.tensor([0.]).to(scatterers_to.device)
            else:
                if args.smooth_loss is not None:
                    smooth_loss = args.smooth_loss * (torch.mean(
                        torch.nn.functional.l1_loss(scatterers_to.reshape(-1).real, perturb_out.real))
                      + torch.mean(torch.nn.functional.l1_loss(scatterers_to.reshape(-1).imag, perturb_out.imag)))
                else:
                    smooth_loss = torch.tensor([0.]).to(weight_loss.device)

                if args.phase_loss is not None:
                    phase_loss = args.phase_loss * torch.mean(
                        torch.nn.functional.l1_loss(torch.cos(torch.angle(scatterers_to.reshape(-1))),
                                                              torch.cos(torch.angle(perturb_out))))
                else:
                    phase_loss = torch.tensor([0.]).to(weight_loss.device)

        else:
            smooth_loss = torch.tensor([0.]).to(weight_loss.device)
            phase_loss = torch.tensor([0.]).to(weight_loss.device)

        total_loss = weight_loss + tv_loss + phase_loss + sparsity_loss + smooth_loss + smooth_weight_loss
        #+ normal_loss + tv_loss + phase_loss

        #print(total_loss.item(), weight_loss.item(), tv_loss.item(), phase_loss.item(), sparsity_loss.item(), phase_loss.item())
        # Back propagate
        total_loss.backward()
        #
        #b = time.time()
        #ts.append(b - a)
        #print("Time", sum(ts) / len(ts))
        if global_step % args.accum_grad == 0:
            # torch.nn.utils.clip_grad_norm_(scene_model.network.parameters(), 1e-8)
            scene_optimizer.step()
            scene_optimizer.zero_grad()
        #
        #######################
        # (4.4) Export weight
        #######################
        if global_step % args.export_model_every == 0:
            model_export_path = os.path.join(basedir, 'models')
            if not os.path.exists(model_export_path):
                os.makedirs(model_export_path)
            model_export_file = os.path.join(model_export_path, '{:06d}.tar'.format(global_step))

            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': scene_model.state_dict(),
                'optimizer_state_dict': scene_optimizer.state_dict(),
            }, model_export_file)
            logger.info("Weight exported %s" % model_export_file)

        ######################################
        # (4.5) Export to tensorboard / folder
        ######################################

        # (4.5.1) loss & sigma est
        if global_step % args.info_every == 0:
            # write loss
            writer.add_scalar('Loss/weight_loss', weight_loss, global_step)
            writer.add_scalar('Loss/sparsity_loss', sparsity_loss, global_step)
            writer.add_scalar('Loss/smooth_loss', smooth_loss, global_step)
            # writer.add_scalar('Loss/normal_loss', normal_loss, global_step)
            writer.add_scalar('Loss/tv_loss', tv_loss, global_step)
            writer.add_scalar('Loss/phase_loss', phase_loss, global_step)
            writer.add_scalar('Loss/phase_loss', smooth_weight_loss, global_step)

            logger.info("Count %d: Weight loss %f, Sparsity loss %f, Smooth loss %f, TV loss %f, Phase loss %f,"
                        " Weight loss %f" %
                        (global_step, weight_loss.item(), sparsity_loss.item(), smooth_loss.item(), tv_loss.item(),
                         phase_loss.item(), smooth_weight_loss.item()))


        # (4.5.2) write scene info
        if global_step % args.scene_every == 0:
            if args.incoherent or args.real_only:
                fig = plt.figure()
                plt.plot(estimated_weights.detach().cpu().numpy(), label='est', alpha=0.5)
                plt.plot(gt_weight_cropped.detach().cpu().numpy(), label='sigma', alpha=0.5)
                plt.legend()
                figure_to_tensorboard(writer, fig, 'image/sigma_est', global_step)
                plt.savefig(os.path.join(basedir, c.IMAGES, 'sigma_est' + str(global_step) + '.png'))
                plt.close('all')
            else:
                fig = plt.figure(figsize=(3, 9))
                plt.subplot(3, 1, 1)
                plt.plot(estimated_weights.real.detach().cpu().numpy(), label='est', alpha=0.5)
                plt.plot(gt_weight_cropped.real.detach().cpu().numpy(), label='sigma', alpha=0.5)
                plt.legend()
                plt.title('Real')
                plt.subplot(3, 1, 2)
                plt.plot(estimated_weights.imag.detach().cpu().numpy(), label='est', alpha=0.5)
                plt.plot(gt_weight_cropped.imag.detach().cpu().numpy(), label='sigma', alpha=0.5)
                plt.legend()
                plt.title('Imag')
                plt.subplot(3, 1, 3)
                plt.plot(estimated_weights.abs().detach().cpu().numpy(), label='est', alpha=0.5)
                plt.plot(gt_weight_cropped.abs().detach().cpu().numpy(), label='sigma', alpha=0.5)
                plt.legend()
                plt.title('Abs')
                figure_to_tensorboard(writer, fig, 'image/sigma_est', global_step)
                plt.savefig(os.path.join(basedir, c.IMAGES, 'sigma_est' + str(global_step) + '.png'))
                plt.close('all')

            print("ARGS MAX VOXELS IS:", args.max_voxels)
            with torch.no_grad():

                #if args.max_voxels is not None:
                chunks = divide_chunks(list(range(0, all_scene_coords.shape[0])), args.max_voxels)

                comp_albedo = torch.zeros((all_scene_coords.shape[0]), dtype=torch.complex64)
                normal = torch.zeros((all_scene_coords.shape[0], 3))
                for chunk in tqdm(chunks, desc="Querying network for full scene..."):
                    scene_chunk = all_scene_coords[chunk, :]
                    with torch.no_grad():
                        model_out = scene_model(coords_to=scene_chunk.cuda(),
                                    compute_normals=compute_normals)

                    comp_albedo[chunk] = model_out['scatterers_to']

                    if compute_normals:
                        normal[chunk, :] = model_out['normals']
                    del scene_chunk
               # else:
               #     model_out = scene_model(coords_to=all_scene_coords.cuda(),
               #                             compute_normals=compute_normals)
               #     comp_albedo = model_out['scatterers_to']
               #     normal = model_out['normals']

            print("Albedo min and max", comp_albedo.abs().min().item(), comp_albedo.abs().max().item())

            if args.x_then_y:
                comp_albedo = comp_albedo.reshape(NUM_X, NUM_Y,NUM_Z).detach().cpu().numpy()
            else:
                comp_albedo = comp_albedo.reshape(NUM_Y, NUM_X, NUM_Z).detach().cpu().numpy()

            if compute_normals:
                if args.x_then_y:
                    normal = normal.reshape(NUM_X, NUM_Y, NUM_Z, -1).detach().cpu().numpy()
                else:
                    normal= normal.reshape(NUM_Y, NUM_X, NUM_Z,-1).detach().cpu().numpy()
                np.save(os.path.join(basedir, c.NUMPY, 'normal' + str(global_step) + '.npy'), normal)
            else:
                normal = np.zeros((comp_albedo.shape[0], comp_albedo.shape[1], comp_albedo.shape[2], 3))

            if args.flip_z:
                comp_albedo = np.flip(comp_albedo, -1)
                normal = np.flip(normal, -2)

            np.save(os.path.join(basedir, c.NUMPY, 'comp_albedo' + str(global_step) + '.npy'), comp_albedo)

            fig = matplotlib_render(np.abs(comp_albedo).astype(float), args.plot_thresh,
                                    x_voxels=all_scene_coords[:, 0],
                                    y_voxels=all_scene_coords[:, 1],
                                    z_voxels=all_scene_coords[:, 2],
                                    x_corners=all_scene_coords[:, 0],
                                    y_corners=all_scene_coords[:, 1],
                                    z_corners=all_scene_coords[:, 2],
                                    save_path=os.path.join(basedir, c.IMAGES,
                                                           'comp_albedo' + str(global_step) + '.png'))
            figure_to_tensorboard(writer, fig, 'image/comp_albedo', global_step)
            plt.close('all')

            depths = [0]
            if args.depth_plots:
                depths = [26, 42]
                #depths = list(range(22, 25))
                #depths = list(range(35, 45))

            for depth in depths:
                if not args.incoherent and not args.real_only:
                    fig = plt.figure()
                    plt.imshow(np.real(comp_albedo[..., depth].squeeze()).astype(float))
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    figure_to_tensorboard(writer, fig, 'image/albedo_real_' + str(depth), global_step)
                    plt.savefig(os.path.join(basedir, c.IMAGES, 'albedo_real_' + str(depth) + str(global_step) + '.png'))
                    plt.close()

                    fig = plt.figure()
                    plt.imshow(np.imag(comp_albedo[..., depth].squeeze()).astype(float))
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    figure_to_tensorboard(writer, fig, 'image/albedo_imag_' + str(depth), global_step)
                    plt.savefig(os.path.join(basedir, c.IMAGES, 'albedo_imag_' + str(depth) + str(global_step) + '.png'))
                    plt.close()

                    fig = plt.figure(figsize=(3, 9))
                    plt.subplot(3, 1, 1)
                    plt.imshow(np.abs(comp_albedo[..., depth]).astype(float))
                    plt.colorbar()
                    plt.title('Complex Albedo')
                    plt.subplot(3, 1, 2)
                    plt.imshow(range_normalize(np.abs(comp_albedo[..., depth].squeeze())[..., None] *
                                               normal[..., depth, :].squeeze().astype(float)))
                    plt.colorbar()
                    plt.title('Normal')

                    plt.subplot(3, 1, 3)
                    plt.imshow(np.angle(comp_albedo[..., depth]).astype(float), cmap='hsv')
                    plt.colorbar()
                    plt.title('Phase')

                    plt.tight_layout()

                    figure_to_tensorboard(writer, fig, 'image/slicexy_' + str(depth), global_step)
                    plt.savefig(os.path.join(basedir, c.IMAGES, 'slicexy_' + str(depth) + str(global_step) + '.png'))
                    plt.close('all')

                fig = plt.figure()
                plt.imshow(np.abs(comp_albedo[..., depth].squeeze()).astype(float))
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.tight_layout()
                figure_to_tensorboard(writer, fig, 'image/albedo_abs_' + str(depth), global_step)
                plt.savefig(os.path.join(basedir, c.IMAGES, 'albedo_abs_' + str(depth) + str(global_step) + '.png'))
                plt.close()

        global_step = global_step + 1

