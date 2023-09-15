import torch
import os
from network import Network
import commentjson as json
from utils import divide_chunks
from tqdm import tqdm
from geometry import create_voxels
import constants as c
import mcubes
import numpy as np
import argparse
import scipy
import pickle

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

    print("V, T", vertices.shape, triangles.shape)
    vertices = normalize_vertices(vertices, **kwargs)
    mcubes.export_obj(vertices, triangles, os.path.join(output_dir, "mesh.obj"))
    points_smooth = mcubes.smooth(points)

    print("Points smooth", points_smooth.shape)
    vertices, triangles = mcubes.marching_cubes(points_smooth, 0)
    print("V, T smooth", vertices.shape, triangles.shape)
    vertices = normalize_vertices(vertices, **kwargs)
    mcubes.export_obj(vertices, triangles, os.path.join(output_dir, "mesh_smooth.obj"))


def config_parser():
    parser = argparse.ArgumentParser(description="Sample the network at arbitrary resolution")
    parser.add_argument('--exp_name', required=True,
                        help='experiment name')
    parser.add_argument('--experiment_dir', required=True,
                        help='experiment directory')
    parser.add_argument('--inr_config', required=True,
                        help='json file containing the network configuration')
    parser.add_argument('--output_scene_file_name', required=True,
                        help='Code will output a <output_scene_file_name>.npy and <output_scene_file_name>.mat.'
                             'Each is a complex-valued 3D array that can be visualized using the Matlab volume'
                             'renderer')
    parser.add_argument('--output_dir_name', required=True,
                        help='Name of output directory')
    parser.add_argument('--no_factor_4', required=False, action='store_true')
    parser.add_argument('--model_path', required=False, default=None,
                        help="Pass in a particular model")
    parser.add_argument('--num_layers', required=False, type=int, default=4,
                        help="Number of layers in neural network model")
    parser.add_argument('--num_neurons', required=False, type=int, default=128,
                        help="Number of layers in neural network model")
    parser.add_argument('--system_data', required=True,
                        help="system_data.pik file")
    parser.add_argument('--sf', required=False, type=int, default=1,
                        help='Scale scene by this factor')
    parser.add_argument('--normalize_scene_dims', required=False, default=False, action='store_true',
                        help="normalize scene dimensions (Check if this was done for neural_backproject.sh)")
    parser.add_argument('--flip_z', required=False, default=False, action='store_true',
                        help="Whether to flip z dimension (check if this was done for neural_backprojection.sh")
    parser.add_argument('--permute_xy', required=False, default=False, action='store_true',
                        help="Flip xy axis. Should be used for the SVSS data")
    return parser

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    with open(args.system_data, 'rb') as handle:
        system_data = pickle.load(handle)

    experiment_dir_agave = args.experiment_dir
    exp = args.exp_name

    scene_inr_config = os.path.join(experiment_dir_agave, args.inr_config)

    if args.model_path is None:
        #model_path = os.path.join(experiment_dir_agave, 'nbp_output', exp)
        model_paths = os.path.join(experiment_dir_agave, 'nbp_output', exp, 'models')
        model_names = [os.path.join(model_paths, f) for f in sorted(os.listdir(model_paths)) if 'tar' in f]

        ckpt_path = None

        if len(model_names) < 0:
            raise OSError("Did not find any saved models in " + model_paths)
        else:
            ckpt_path = model_names[-1]
            print("Using model path:", ckpt_path)
    else:
        ckpt_path = args.model_path

    assert ckpt_path is not None, "Failed to load any model checkpoints. You can pass one in using the --model_path input arg."

    dev = 'cuda:0'
    num_layers = args.num_layers
    num_neurons = args.num_neurons
    scene_voxels = None
    incoherent = False
    real_only = False
    normalize_scene_dims = args.normalize_scene_dims
    compute_normals=False
    max_voxels = 1500000

    x_min=system_data[c.GEOMETRY][c.CORNERS][:, 0].min()
    x_max=system_data[c.GEOMETRY][c.CORNERS][:, 0].max()
    y_min=system_data[c.GEOMETRY][c.CORNERS][:, 1].min()
    y_max=system_data[c.GEOMETRY][c.CORNERS][:, 1].max()
    z_min=system_data[c.GEOMETRY][c.CORNERS][:, 2].min()
    z_max=system_data[c.GEOMETRY][c.CORNERS][:, 2].max()
    sf = args.sf

    num_x = system_data[c.GEOMETRY][c.NUM_X]*sf
    num_y = system_data[c.GEOMETRY][c.NUM_Y]*sf
    num_z = system_data[c.GEOMETRY][c.NUM_Z]*sf

    print("Upsampled scene dimensions", num_x, num_y, num_z)

    #voxels = system_data[c.GEOMETRY][c.VOXELS]
    #all_scene_coords = torch.from_numpy(voxels)

    voxels = create_voxels(x_min=x_min,
                           x_max=x_max,
                           y_min=y_min,
                           y_max=y_max,
                           z_min=z_min,
                           z_max=z_max,
                           num_x=num_x,
                           num_y=num_y,
                           num_z=num_z,
                           permute_xy=args.permute_xy)

    all_scene_coords = torch.from_numpy(voxels[c.VOXELS])

    if args.no_factor_4:
        scene_scale_factor = 1 / (all_scene_coords.abs().max())
    else:
        scene_scale_factor = 1 / (4 * all_scene_coords.abs().max())

    if normalize_scene_dims:
        scene_scale_factor = 1 / (4 * all_scene_coords.abs().max())
        all_scene_coords = all_scene_coords * scene_scale_factor

    with open(scene_inr_config) as config_file:
        inr_config = json.load(config_file)

    scene_model = Network(inr_config=inr_config,
                          dev=dev,
                          num_layers=num_layers,
                          num_neurons=num_neurons)


    ckpt = torch.load(ckpt_path)

    scene_model.load_state_dict(ckpt['network_fn_state_dict'])

    if max_voxels is not None:
        chunks = divide_chunks(list(range(0, all_scene_coords.shape[0])), max_voxels)

        comp_albedo = np.zeros((all_scene_coords.shape[0]), dtype=complex)
        normal = np.zeros((all_scene_coords.shape[0], 3))
        for chunk in tqdm(chunks, desc="Querying network for full scene..."):
            scene_chunk = all_scene_coords[chunk, :]
            with torch.no_grad():
                model_out = scene_model(coords_to=scene_chunk.cuda(),
                                        compute_normals=compute_normals)

            comp_albedo[chunk] = model_out['scatterers_to'].detach().cpu().numpy()

            if compute_normals:
                normal[chunk, :] = model_out['normals'].detach().cpu().numpy()
            del scene_chunk
    else:
        model_out = scene_model(coords_to=all_scene_coords.cuda(),
                                compute_normals=compute_normals)
        comp_albedo = model_out['scatterers_to']
        normal = model_out['normals']

    if compute_normals:
        print("Normals shape", normal.shape)

    comp_albedo = np.reshape(comp_albedo, (num_x, num_y, num_z))

    if args.flip_z:
        print("Flipped z")
        comp_albedo = np.flip(comp_albedo, -1)
    else:
        print("Did not flip z")

    output_dir = os.path.join(experiment_dir_agave, args.output_dir_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, args.output_scene_file_name + '.npy'), comp_albedo)

    data = {
        'scene': comp_albedo,
    }

    scipy.io.savemat(os.path.join(output_dir, args.output_scene_file_name + '.mat'), data)

