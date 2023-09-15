import argparse
import pickle
import constants as c
import tinycudann as tcnn
import commentjson as json
import torch
from sas_utils import precompute_time_series, no_rc_kernel_from_waveform, radial_delay_wfms_fast, hilbert_torch, wiener_deconvolution
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import divide_chunks
from argument_io import directory_cleaup
import warnings
import scipy
from deconv_gd import GradientDescentDeconvolution
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deconvolve SAS measurements using an INR")
    parser.add_argument('--inr_config', required=True,
                        help='json file that configures tiny cuda INR')
    parser.add_argument('--system_data', required=True,
                        help='Pickle file containing system data structure')
    parser.add_argument('--output_dir', required=True,
                        help="Directory to save output")
    parser.add_argument('--num_radial', type=int, required=False, default=None,
                        help='Number of range samples to take.')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4,
                        help='Learning rate for the INR')
    parser.add_argument('--clear_output_directory', required=False, default=False, action='store_true',
                        help="Whether to delete everything in output directory before running")
    parser.add_argument('--compare_with_mf', required=False, default=False, action='store_true',
                        help="Whether to compare with MF waveforms")
    parser.add_argument('--num_trans_per_inr', type=int, required=True, default=300)
    parser.add_argument('--number_iterations', type=int, required=False, default=500000,
                        help="Number of iterations to train for")
    parser.add_argument('--info_every', type=int, required=False, default=1000,
                        help="How often to print the loss")
    parser.add_argument('--sparsity', type=float, required=False, default=0.,
                        help="Sparsity wight")
    parser.add_argument('--phase_loss', type=float, required=False, default=None,
                        help="Phase loss wight")
    parser.add_argument('--tv_loss', type=float, required=False, default=None,
                        help="TV loss wight")
    parser.add_argument('--complex_weights', required=False, default=False, action='store_true',
                        help="Whether to fit complex waveforms as well")
    parser.add_argument('--subtract_dc', required=False, default=False, action='store_true',
                        help="Subtract mean from waveforms")
    parser.add_argument('--normalize_each', required=False, action='store_true',
                        help="Whether to normalize each waveform by its own energy")
    parser.add_argument('--log_loss', required=False, action='store_true',
                        help="Whether to use log loss")
    parser.add_argument('--load_wfm', required=False, default=None, help="Load measured waveform")
    parser.add_argument('--output', required=False, default=None, help="Network output")
    parser.add_argument('--use_debug_wfm', required=False, default=False, action='store_true',
                        help="Kernel for debugging purposes")
    parser.add_argument('--start_from', required=False, default=0, type=int, help="Start from batch")
    parser.add_argument('--drc_gt', required=False, type=float, default=None,
                        help="Between [0, 1]. Values closer to 0 amplify small values."
                             "1 yields no change. ")
    parser.add_argument('--drc_weights', required=False, type=float, default=None,
                        help="Between [0, 1]. Values closer to 0 amplify small values."
                             "1 yields no change. ")
    parser.add_argument('--linear_plot', required=False, default=False, action='store_true',
                        help="Plot on a linear scale ")
    parser.add_argument('--zero_padding', required=False, default=0, type=int, help="Zero pad waveforms")
    parser.add_argument('--norm_kernel', required=False, action='store_true', help="Dont normalize kernel")
    parser.add_argument('--wiener_deconvolution', required=False, action='store_true', help="Use Wiener deconvolution "
                                                                                            "instead of INR.")
    parser.add_argument('--wiener_lambda', required=False, type=float, default=0., help='Wiener filter SNR parameter')
    parser.add_argument('--no_network', required=False, action='store_true', help="Don't use INR. Gradient descent directly"
                                                                                  "to time bins.")
    args = parser.parse_args()

    if args.drc_gt is not None:
        assert args.drc_gt >= 0
        assert args.drc_gt <= 1

    if args.drc_weights is not None:
        assert args.drc_weights >= 0
        assert args.drc_weights <= 1

    assert torch.cuda.is_available()
    dev = 'cuda:0'

    directory_cleaup(args.output_dir, args.clear_output_directory)

    print("Saving input arguments to ", os.path.join(args.output_dir, 'commandline_args.txt'))
    # Place in main path
    with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # place in image path in case I only download this directory
    with open(os.path.join(args.output_dir, c.IMAGES, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(args.system_data, 'rb') as handle:
        system_data = pickle.load(handle)

    with open(args.inr_config) as config_file:
        inr_config = json.load(config_file)

    gt_wfms = system_data[c.WFM_DATA]

    assert gt_wfms is not None, "GT wfms is None, did you load a system_data.pik containing sonar waveforms?"
    if args.compare_with_mf:
        mf_wfms = system_data[c.WFM_RC]
    tx_coords = system_data[c.TX_COORDS]
    rx_coords = system_data[c.RX_COORDS]
    corners = system_data[c.GEOMETRY][c.CORNERS]
    system_params = system_data[c.SYS_PARAMS]

    if args.load_wfm is not None:
        print("Loading waveform")
        wfm = np.load(args.load_wfm)[0:100]
    else:
        wfm = system_data[c.WFM]

    assert wfm is not None

    speed_of_sound = system_data[c.SOUND_SPEED]

    if not tx_coords.shape[0] % args.num_trans_per_inr == 0:
        print("Num_trans_per_inr", args.num_trans_per_inr, "Number of tx", tx_coords.shape[0])
        warnings.warn("num_trans_per_inr should ideally be divisible into number of tx. If it's not, the remainder" + \
                      "of the waveforms may be sampled differently.")

    wfm_crop_settings = system_data[c.WFM_CROP_SETTINGS]

    assert tx_coords.shape[0] == rx_coords.shape[0]

    # Crop the ground truth waveforms if not already cropped
    if gt_wfms.shape[1] != wfm_crop_settings[c.NUM_SAMPLES]:
        gt_wfms = gt_wfms[:,
                  wfm_crop_settings[c.MIN_SAMPLE]:
                  wfm_crop_settings[c.MIN_SAMPLE]+wfm_crop_settings[c.NUM_SAMPLES]
        ]

    # Scale the waveforms
    gt_wfms = torch.from_numpy(gt_wfms).to(dev)


    if args.compare_with_mf:
        if mf_wfms.shape[1] != wfm_crop_settings[c.NUM_SAMPLES]:
            mf_wfms = mf_wfms[:,
                      wfm_crop_settings[c.MIN_SAMPLE]:
                      wfm_crop_settings[c.MIN_SAMPLE] + wfm_crop_settings[c.NUM_SAMPLES]
                      ]
    #np.sqrt(np.sum(np.abs(wfm_convolve) ** 2))
    if args.normalize_each:
        gt_wfms = gt_wfms / torch.sqrt(torch.sum(torch.abs(gt_wfms)**2, dim=-1))[..., None]
    else:
        gt_wfms = gt_wfms / gt_wfms.abs().max()

    if args.subtract_dc:
        print("Subtracting DC component")
        gt_wfms = gt_wfms - torch.mean(gt_wfms, dim=1, keepdim=True)

        if args.compare_with_mf:
            mf_wfms = mf_wfms - np.mean(mf_wfms, axis=1, keepdims=True)

    gt_wfms = torch.nn.functional.pad(gt_wfms, (args.zero_padding, args.zero_padding))

    # Define sampling distances
    num_radial = None
    if args.num_radial is None:
        dists_norm = torch.linspace(0, 1, wfm_crop_settings[c.NUM_SAMPLES] + args.zero_padding*2).to(dev)
        num_radial = wfm_crop_settings[c.NUM_SAMPLES] + args.zero_padding*2
    else:
        dists_norm = torch.linspace(0, 1, args.num_radial).to(dev)
        num_radial = args.num_radial

    min_dist = wfm_crop_settings[c.MIN_DIST] - args.zero_padding/system_data[c.SYS_PARAMS][c.FS] * speed_of_sound
    max_dist = wfm_crop_settings[c.MAX_DIST] + args.zero_padding/system_data[c.SYS_PARAMS][c.FS] * speed_of_sound

    dists_scene = min_dist + \
                    dists_norm * (max_dist - min_dist)

    if args.use_debug_wfm:
        print("Using debug wfm")
        wfm_dur = 255 * 1e-6
        f_start = 20000
        f_stop = 35000

        times = np.linspace(0, wfm_dur - 1 / system_data[c.SYS_PARAMS][c.FS],
                            num=int((wfm_dur) * system_data[c.SYS_PARAMS][c.FS] * 1))
        LFM = scipy.signal.chirp(times, f_start, wfm_dur, f_stop, phi=0)

        taylor_window = scipy.signal.windows.taylor(len(LFM), nbar=5, sll=40, norm=True)
        kernel = LFM * taylor_window

        wfm = kernel

    if args.drc_gt is not None:
        gt_wfms = torch.sign(gt_wfms) * torch.abs(gt_wfms)**(args.drc_gt)

    if args.zero_padding > 0:
        mask = torch.zeros(wfm_crop_settings[c.NUM_SAMPLES] + 2*args.zero_padding).to(dev)
        mask[args.zero_padding:-args.zero_padding] = 1.

    # Precompute unscaled time series
    kernel = no_rc_kernel_from_waveform(wfm, wfm_crop_settings[c.NUM_SAMPLES] + 2*args.zero_padding).to(dev)

    if args.norm_kernel:
        kernel = kernel / torch.sqrt(torch.mean(kernel * torch.conj(kernel)))

    tsd = precompute_time_series(dists_scene,
                                 min_dist,
                                 kernel,
                                 speed_of_sound,
                                 system_data[c.SYS_PARAMS][c.FS],
                                 wfm_crop_settings[c.NUM_SAMPLES] + 2*args.zero_padding)

    trans = divide_chunks(list(range(0, tx_coords.shape[0])), args.num_trans_per_inr)

    for batch_num, trans_batch in enumerate(trans):
        print("Batch", batch_num, "/", tx_coords.shape[0] // args.num_trans_per_inr)
        ndim = 1
        if args.complex_weights:
            ndim = 2

        if batch_num < args.start_from:
            continue
        if args.wiener_deconvolution:
            print("Using Wiener deconvolution")
            wfm_batch = gt_wfms[trans_batch, :].squeeze()
            wfm_batch_fft = torch.fft.fft(wfm_batch, dim=-1)

            weights = wiener_deconvolution(wfm_batch_fft, kernel, args.wiener_lambda)
            weights = weights.detach().cpu().numpy()

            if args.zero_padding > 0:
                weights = weights[:, args.zero_padding:-args.zero_padding]
            if args.drc_weights is not None:
                weights = np.sign(weights) * np.abs(weights) ** (args.drc_weights / 1)

            np.save(os.path.join(args.output_dir, c.NUMPY, c.WEIGHT_PREFIX + str(min(trans_batch)) + '_' + \
                                 str(max(trans_batch))), weights)

            fig, ax = plt.subplots(3)
            if args.linear_plot:
                ax[0].imshow(np.abs(weights + 1e-6).T)
            else:
                ax[0].imshow(20 * np.log10(np.abs(weights + 1e-6)).T)
            ax[0].title('INR Deconvolved')
            if args.linear_plot:
                ax[1].imshow(np.abs(mf_wfms[trans_batch, :] + 1e-6).T)
            else:
                ax[1].imshow(20 * np.log10(np.abs(mf_wfms[trans_batch, :] + 1e-6)).T)
            ax[1].title("MF")
            if args.linear_plot:
                ax[2].imshow(np.abs(gt_wfms[trans_batch, :].abs().squeeze().
                                  detach().cpu().numpy().astype('float32') + 1e-6).T)
            else:
                ax[2].imshow(20 * np.log10(gt_wfms[trans_batch, :].abs().squeeze().
                                         detach().cpu().numpy().astype('float32') + 1e-6).T)
            ax[2].title("Raw")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, c.IMAGES, 'comparison' + str(batch_num) + '.png'))

            plt.figure()
            plt.plot(weights[0].squeeze(), label='est', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'weights' + str(batch_num) + '.png'))
            plt.close('all')

            weight = hilbert_torch(torch.from_numpy(weights[0, :])).detach().cpu().numpy()
            plt.subplot(1, 4, 1)
            plt.plot(np.real(weight).squeeze(), label='real', alpha=0.5)
            plt.subplot(1, 4, 2)
            plt.plot(np.imag(weight).squeeze(), label='imag', alpha=0.5)
            plt.subplot(1, 4, 3)
            plt.plot(np.abs(weight).squeeze(), label='abs', alpha=0.5)
            plt.subplot(1, 4, 4)
            plt.plot(np.angle(weight).squeeze(), label='angle', alpha=0.5)
            plt.legend()
            plt.savefig(
                os.path.join(args.output_dir, c.IMAGES, 'weights_comp' + str(batch_num) + '.png'))
            plt.close('all')

        else:
            if args.no_network:
                print("Not using an INR. Gradient descent directly to time bins")
                num_batch, num_bins = gt_wfms[trans_batch].shape
                model = GradientDescentDeconvolution(num_batch=num_batch,
                                                     num_bins=num_bins,
                                                     dev=dev)

                optimizer = torch.optim.Adam(list(model.get_params(args.learning_rate)),
                                             lr=args.learning_rate)
            else:
                model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=ndim,
                                                      encoding_config=inr_config["encoding"],
                                                      network_config=inr_config["network"])


                optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate)

            # [num trans, num_radial, 2]
            samples = torch.stack(torch.meshgrid(dists_norm, torch.linspace(0, 1, len(trans_batch)).to(dev),
                                                 indexing='ij')).permute(2, 1, 0)
            ts = []
            for epoch in range(1, args.number_iterations):
                #a = time.time()
                # Model predicts the weights at each time sample
                sample_batch = samples.reshape(-1, 2)
                if args.no_network:
                    weights = model()
                else:
                    weights = model(sample_batch)

                #print("3")
                if args.complex_weights:
                    weights = weights.reshape(len(trans_batch), num_radial, 2)
                else:
                    weights = weights.reshape(len(trans_batch), num_radial)

                if args.zero_padding > 0:
                    if args.complex_weights:
                        weights = weights * mask[None, :, None]
                    else:
                        weights = weights * mask[None, :]

                sparsity = torch.mean(torch.abs(weights))
                # Use weights to scale the precomputed waveform response and integrate
                #print("4")
                if args.complex_weights:
                    est_wfm_real = radial_delay_wfms_fast(tsd, weights[..., 0]).real
                    est_wfm_imag = radial_delay_wfms_fast(tsd, weights[..., 1]).imag
                else:
                    est_wfm = radial_delay_wfms_fast(tsd, weights).real

                if args.phase_loss is not None:
                    complex_wfm = hilbert_torch(est_wfm)
                    est_wfm_angle = torch.angle(complex_wfm)
                    phase_loss = args.phase_loss * \
                                 ((torch.mean(torch.abs(torch.cos(est_wfm_angle[:, 1:]) -
                                                                          torch.cos(est_wfm_angle[:, :-1])))
                                  + torch.mean(torch.abs(torch.sin(est_wfm_angle[:, 1:]) -
                                                                           torch.sin(est_wfm_angle[:, :-1])))))
                else:
                    phase_loss = torch.tensor([0.]).to(dev)

                if args.tv_loss is not None:
                    tv_loss = args.tv_loss * torch.mean(torch.abs(est_wfm[:, 1:] - est_wfm[:, :-1]))
                else:
                    tv_loss = torch.tensor([0.]).to(dev)

                if args.complex_weights:
                    loss = torch.nn.functional.mse_loss(est_wfm_real.squeeze(),
                                                        gt_wfms[trans_batch, :].squeeze().real,
                                                        reduction='mean') + \
                        torch.nn.functional.mse_loss(est_wfm_imag.squeeze(),
                                                 gt_wfms[trans_batch, :].squeeze().imag,
                                                 reduction='mean')

                else:
                    if args.log_loss:
                    # Compute a loss with ground truth waveforms
                        loss = torch.nn.functional.l1_loss((est_wfm.squeeze() + 1e-8),
                                                            (gt_wfms[trans_batch, :].squeeze() + 1e-8),
                                                            reduction='mean')
                    else:
                        loss = torch.nn.functional.mse_loss(est_wfm.squeeze(),
                                                           gt_wfms[trans_batch, :].squeeze(),
                                                           reduction='mean')

                sparsity_loss = args.sparsity*sparsity

                total_loss = loss + sparsity_loss + phase_loss + tv_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                #b = time.time()
                #ts.append(b - a)
                #print(sum(ts)/len(ts))
                #print("5")
                # hello
                # hello
                #if args.zero_padding > 0:
                #    gt_wfms = gt_wfms.squeeze()[:, args.zero_padding:-args.zero_padding]
                #est_wfm = est_wfm.squeeze()[:, args.zero_padding:-args.zero_padding]

                if epoch % args.info_every == 0:
                    if args.complex_weights:
                        est_wfm = torch.complex(real=est_wfm_real.float(), imag=est_wfm_imag.float())
                        weights = torch.complex(real=weights[..., 0].float(),
                                                imag=weights[..., 1].float()).squeeze().detach().cpu().numpy()
                    else:
                        weights = weights.squeeze().detach().cpu().numpy().astype('float32')

                    if args.drc_weights is not None:
                        weights = np.sign(weights) * np.abs(weights) ** (args.drc_weights/1)

                    print("Epoch", epoch, "Loss", loss.item(), "Sparsity", sparsity_loss.item(), "Phase", phase_loss.item(),
                          "TV", tv_loss.item())
                    if args.compare_with_mf:
                        plt.figure(figsize=(10, 6))
                        plt.subplot(3, 1, 1)
                        if args.linear_plot:
                            plt.imshow(np.abs(weights + 1e-6).T)
                        else:
                            plt.imshow(20*np.log10(np.abs(weights + 1e-6)).T)
                        plt.title('INR Deconvolved')
                        plt.subplot(3, 1, 2)
                        if args.linear_plot:
                            plt.imshow(np.abs(mf_wfms[trans_batch, :] + 1e-6).T)
                        else:
                            plt.imshow(20 * np.log10(np.abs(mf_wfms[trans_batch, :] + 1e-6)).T)
                        plt.title("MF")
                        plt.subplot(3, 1, 3)
                        if args.linear_plot:
                            plt.imshow(np.abs(gt_wfms[trans_batch, :].abs().squeeze().
                                                     detach().cpu().numpy().astype('float32') + 1e-6).T)
                        else:
                            plt.imshow(20*np.log10(gt_wfms[trans_batch, :].abs().squeeze().
                                                   detach().cpu().numpy().astype('float32') + 1e-6).T)
                        plt.title("Raw")
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'comparison' + str(batch_num) + '_' + str(epoch) + '.png'))
                    else:
                        plt.figure(figsize=(10, 3))
                        plt.subplot(1, 2, 1)
                        plt.title('INR Deconvolved')
                        if args.zero_padding > 0:
                            plt.imshow(20 * np.log10(np.abs(weights[:, args.zero_padding:-args.zero_padding] + 1e-6)).T)
                        else:
                            plt.imshow(20 * np.log10(np.abs(weights + 1e-6)).T)
                        plt.subplot(1, 2, 2)
                        plt.title("MF")
                        if args.zero_padding > 0:
                            plt.imshow(
                                20 * np.log10(gt_wfms[trans_batch, args.zero_padding:-args.zero_padding].abs().squeeze().detach().cpu().numpy().astype('float32') + 1e-6).T)
                        else:
                            plt.imshow(
                                20 * np.log10(gt_wfms[trans_batch, :].abs().squeeze().detach().cpu().numpy().astype(
                                    'float32') + 1e-6).T)
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES,
                                                 'comparison' + str(batch_num) + '_' + str(epoch) + '.png'))

                    if args.complex_weights:
                        plt.figure()
                        plt.plot(est_wfm.real.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                        plt.plot(gt_wfms[trans_batch, :].real.squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'wfm_est_real' + str(epoch) + '.png'))
                        plt.close()

                        plt.figure()
                        plt.plot(est_wfm.imag.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                        plt.plot(gt_wfms[trans_batch, :].imag.squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'wfm_est_imag' + str(epoch) + '.png'))
                        plt.close()
                    else:
                        plt.figure()
                        plt.plot(est_wfm.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                        plt.plot(gt_wfms[trans_batch, :].squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'wfm_est_real' + str(batch_num) + '_' + str(epoch) + '.png'))
                        plt.close()

                    if args.complex_weights:
                        plt.figure()
                        plt.plot(weights[0, ...].real.squeeze(), label='est', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'weights_real' + str(epoch) + '.png'))
                        plt.close('all')

                        plt.figure()
                        plt.plot(weights[0, ...].imag.squeeze(), label='est', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'weights_imag' + str(epoch) + '.png'))
                        plt.close('all')
                    else:
                        plt.figure()
                        plt.plot(weights[0].squeeze(), label='est', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(args.output_dir, c.IMAGES, 'weights' + str(batch_num) + '_' + str(epoch) + '.png'))
                        plt.close('all')

                        weight = hilbert_torch(torch.from_numpy(weights[0, :])).detach().cpu().numpy()
                        plt.subplot(1, 4, 1)
                        plt.plot(np.real(weight).squeeze(), label='real', alpha=0.5)
                        plt.subplot(1, 4, 2)
                        plt.plot(np.imag(weight).squeeze(), label='imag', alpha=0.5)
                        plt.subplot(1, 4, 3)
                        plt.plot(np.abs(weight).squeeze(), label='abs', alpha=0.5)
                        plt.subplot(1, 4, 4)
                        plt.plot(np.angle(weight).squeeze(), label='angle', alpha=0.5)
                        plt.legend()
                        plt.savefig(
                            os.path.join(args.output_dir, c.IMAGES, 'weights_comp' + str(batch_num) + '_' + str(epoch) + '.png'))
                        plt.close('all')

            print("Saving Weights..")
            with torch.no_grad():
                weight_type = np.float64
                if args.complex_weights:
                    weight_type = complex

                if args.no_network:
                    weights_out = model()
                else:
                    weights_out = model(sample_batch)

                if args.complex_weights:
                    assert args.drc_weights is None
                    weights_out = weights_out.reshape(samples.shape[0], num_radial, 2)
                    weights_out = weights_out[..., 0] + 1j * weights_out[..., 1]
                else:
                    weights_out = weights_out.reshape(samples.shape[0], num_radial)

                weights_out = weights_out.detach().cpu().numpy()
                if args.zero_padding > 0:
                    weights_out = weights_out[:, args.zero_padding:-args.zero_padding]
                if args.drc_weights is not None:
                    weights_out = np.sign(weights_out) * np.abs(weights_out) ** (args.drc_weights / 1)

                np.save(os.path.join(args.output_dir, c.NUMPY, c.WEIGHT_PREFIX + str(min(trans_batch)) + '_' + \
                                     str(max(trans_batch))), weights_out)
