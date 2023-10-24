import argparse


def config_parser():
    parser = argparse.ArgumentParser(description="Deconvolve SAS measurements using an INR")
    parser.add_argument('--scene_inr_config', required=True,
                        help='json file that configures tiny cuda INR')
    parser.add_argument('--system_data', required=True,
                        help='Pickle file containing system data structure')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save output')
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-4,
                        help='Learning rate for the INR')
    parser.add_argument('--num_epochs', required=False, type=int, default=10,
                        help='Number of times to iterate over all tx measurements')
    parser.add_argument('--num_rays', required=False, type=int, default=10000,
                        help='Number of vectors to project into scene')
    parser.add_argument('--occlusion', required=False, default=False, action='store_true',
                        help="Whether to handle occlusion using volume rendering equation")
    parser.add_argument('--info_every', required=False, type=int, default=250,
                        help="How often to print loss and weight curves")
    parser.add_argument('--scene_every', required=False, type=int, default=1000,
                        help="How often to save the scene")
    parser.add_argument('--sparsity', required=False, type=float, default=0.,
                        help="Weight of sparsity loss")
    parser.add_argument('--two_dimensions', required=False, default=False, action='store_true',
                        help="Whether to reconstruct to two dimensions")
    parser.add_argument('--accum_grad', required=False, default=1, type=int,
                        help="accumulate gradient (same as using larger batch size but more memory efficient")
    parser.add_argument('--scale_factor', required=False, default=1., type=float,
                        help="Amount to scale weights")
    parser.add_argument('--noise_std', required=False, default=0., type=float,
                        help="Standard deviation on alpha noise")
    parser.add_argument('--occlusion_scale', required=False, default=1., type=float,
                        help="How sensitive to occlusion")
    parser.add_argument('--clear_output_directory', required=False, default=False, action='store_true',
                        help="Whether to delete everything in output directory before running")
    parser.add_argument('--fit_folder', required=True,
                        help="Folder containing fit data")
    parser.add_argument('--plot_thresh', required=False, default=0.2, type=float,
                        help='Threshold for plotting')
    parser.add_argument('--use_up_to', required=False, default=None, type=int,
                        help="Only use subset of the weights")
    parser.add_argument('--max_weights', required=False, default=None, type=int,
                        help="Maximum number of weights per iteration")
    parser.add_argument('--sampling_distribution_uniformity', required=False, default=1, type=float,
                        help="Sampling distribution parameter between [0-1], default is 1. Value of 1 samples points"
                             "proportional to their intensity, values closer to 0 scale points closer to uniform distribution,"
                             "and value of 0 samples all points uniformly")
    parser.add_argument('--perturb_theta_phi', required=False, action='store_true', default=False)
    parser.add_argument('--perturb_sampling', required=False, action='store_true', default=False)

    parser.add_argument('--lambertian_ratio', '--lambertion_ratio',
                        required=False, type=float, default=0., help="0 is use full lambertian, "
                                                                                         "1 is no lambertian")
    parser.add_argument('--normal_loss', required=False, type=float, default=None, help="Orient normals torwards tx")
    parser.add_argument('--beamwidth', required=False, type=float, default=None, help="Beamwidth of tx/rx in degrees")
    parser.add_argument('--upsample', required=False, default=1, type=int)
    parser.add_argument('--perturb_radii', required=False, action='store_true', default=False,
                        help="Whether to perturb radii")
    parser.add_argument('--norm_weights', required=False, action='store_true', help='Normalize weights before loss')
    parser.add_argument('--num_layers', required=False, type=int, default=3, help="Beamwidth of tx/rx in degrees")
    parser.add_argument('--num_neurons', required=False, type=int, default=128, help="Beamwidth of tx/rx in degrees")
    parser.add_argument('--skip_normals', required=False, action='store_true')
    parser.add_argument('--tv_loss', required=False, default=None, type=float, help="Weight of total variation loss")
    parser.add_argument('--smooth_loss', required=False, default=None, type=float, help="Weight of smoothing loss")
    parser.add_argument('--smooth_weight', required=False, default=None, type=float, help="Smooth waveform weights")
    parser.add_argument('--smooth_delta', required=False, default=2.0, type=float,
                        help="Delta for smoothing loss")
    parser.add_argument('--phase_loss', required=False, default=None, type=float, help="Weight of phase variation loss")
    parser.add_argument('--reg_start', required=False, default=0, type=int, help="When to turn on regularization")
    parser.add_argument('--thresh', required=False, type=float, default=1e-8, help="Waveform threshold")
    parser.add_argument('--adaptive_phase', required=False, action='store_true')
    parser.add_argument('--normalize_scene_dims', required=False, action='store_true')
    parser.add_argument('--use_mf_wfms', required=False, action='store_true')
    parser.add_argument('--ray_trace_return', required=False, action='store_true', help="Whether to trace the return"
                                                                                        "path to compute transmission"
                                                                                        "probability. If false, will"
                                                                                        "assume return transmission"
                                                                                        "probability is same as "
                                                                                        "outgoing transmission "
                                                                                        "probability.")
    parser.add_argument('--point_at_center', required=False, action='store_true', help="Whether to point"
                                                                                       "transmitter at scene center.")
    parser.add_argument('--mult_with_alpha', required=False, action='store_true', help="Whether to multiply cumulative product"
                                                                                       "by alpha")
    parser.add_argument('--k_normal_per_ray', type=int, required=False, default=None, help="How many normals to "
                                                                                           "sample per ray")
    parser.add_argument('--importance_sampling_rays', type=int, required=False, default=None, help="Number"
                                                                                                   "of rays to importance"
                                                                                                   "sample with")
    parser.add_argument('--flip_z', required=False, action='store_true')
    parser.add_argument('--bw_units', default=None, type=str, help='Specify (r)adians or (d)egrees for beamwidth'
                                                                   'units')
    parser.add_argument('--x_then_y', required=False, default=False, action='store_true',
                        help="Whether to reshape scene as Y, X, Z (False) or X, Y, Z (True)")
    parser.add_argument('--l1_loss', required=False, default=False, action='store_true',
                        help="Whether to use l1 loss")
    parser.add_argument('--transmit_from_tx', required=False, default=False, action='store_true',
                        help="Whether to project rays from TX (projects from origin if False)")
    parser.add_argument('--dist_trans', required=False, default=False, action='store_true',
                        help="Whether to reshape scene as Y, X, Z (False) or X, Y, Z (True)")
    parser.add_argument('--sequential_sample', required=False, default=False, action='store_true',
                        help="Whether to sample tx/rx in order they appear in list (uses random sampling otherwise)")
    parser.add_argument('--incoherent', required=False, default=False, action='store_true',
                        help="Compute the incoherent reconstruction (ignoring phase) by using the envelope of the "
                             "deconvolved signal.")
    parser.add_argument('--real_only', required=False, default=False, action='store_true',
                        help="Uses only the real part of the deconvolved waveform for fitting "
                             "(as opposed to fit complex waveform)")
    parser.add_argument('--no_network', required=False, default=False, action='store_true',
                        help='Do not use a neural network.')
    parser.add_argument('--depth_plots', required=False, default=False, action='store_true')
    # Juhyeon added
    parser.add_argument('--ft_path', required=False, default=None, type=str, help='Directory to load trained model')
    parser.add_argument('--no_reload', required=False, action='store_true', default=False,
                        help='Force not reloading the model')
    parser.add_argument('--smooth_envelope', required=False, action='store_true', default=False,
                        help='Whether to smooth the envelope')
    parser.add_argument('--export_model_every', required=False, default=5000, type=int,
                        help="When to export model")
    parser.add_argument('--expname', required=False, default=None, help='Experiment name')
    parser.add_argument('--max_voxels', required=False, type=int, default=15000, help="Max number of scene voxels to feed"
                                                                              "network. Set lower if OOM")
    parser.add_argument('--no_factor_4', required=False, action='store_true')
    return parser