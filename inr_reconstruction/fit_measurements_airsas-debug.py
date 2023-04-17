import argparse
import pickle
import constants as c
import tinycudann as tcnn
import commentjson as json
import torch
from sas_utils import crop_wfm, precompute_time_series, no_rc_kernel_from_waveform, radial_delay_wfms_fast, hilbert_torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import inr_fit_sampling, divide_chunks
from tqdm import tqdm
import pdb
from sas_utils import gen_real_lfm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deconvolve SAS measurements using an INR")
    parser.add_argument('--inr_config', required=True,
                        help='json file that configures tiny cuda INR')
    parser.add_argument('--system_data', required=True,
                        help='Pickle file containing system data structure')
    parser.add_argument('--output_dir', required=True,
                        help="Directory to save output")
    parser.add_argument('--num_radial', type=int, required=True,
                        help='Number of range samples to take.')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4,
                        help='Learning rate for the INR')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Number of waveforms in each batch')
    parser.add_argument('--num_trans_per_inr', type=int, required=True, default=300)
    parser.add_argument('--number_iterations', type=int, required=False, default=500000,
                        help="Number of iterations to train for")
    parser.add_argument('--info_every', type=int, required=False, default=1000,
                        help="How often to print the loss")
    parser.add_argument('--sparsity', type=float, required=False, default=0.,
                        help="Sparsity wight")
    parser.add_argument('--complex_weights', required=False, default=False, action='store_true',
                        help="Whether to fit complex waveforms as well")
    parser.add_argument('--normalize_each', required=False, action='store_true',
                        help="Whether to normalize each waveform by its own energy")
    args = parser.parse_args()

    assert torch.cuda.is_available()

    dev = 'cuda:0'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.system_data, 'rb') as handle:
        system_data = pickle.load(handle)

    with open(args.inr_config) as config_file:
        inr_config = json.load(config_file)

    gt_wfms = system_data[c.WFM_DATA]

    if args.complex_weights:
        gt_wfms_complex = torch.zeros_like(torch.from_numpy(gt_wfms), dtype=torch.complex128)
        for i in tqdm(range(gt_wfms.shape[0]), desc="Computing hilbert of waveforms"):
            gt_wfms_complex[i, :] = hilbert_torch(torch.from_numpy(gt_wfms[i, ...]))

        gt_wfms = gt_wfms_complex.detach().cpu().numpy()

    #for i in range(0, gt_wfms.shape[0], 5):
    #    plt.figure()
    #    plt.plot(gt_wfms[i, :])
    #    plt.savefig(os.path.join(args.output_dir, str(i) + '.png'))
    #    plt.close('all')
    #exit(0)

    tx_coords = system_data[c.TX_COORDS]
    rx_coords = system_data[c.RX_COORDS]
    corners = system_data[c.GEOMETRY][c.CORNERS]
    system_params = system_data[c.SYS_PARAMS]
    wfm_params = system_data[c.WFM_PARAMS]
    #wfm = gen_real_lfm(system_data[c.SYS_PARAMS][c.FS],
    #                       30000,
    #                       10000,
    #                       system_data[c.WFM_PARAMS][c.T_DUR],
    #                       window=True,
    #                       win_ratio=system_data[c.WFM_PARAMS][c.WIN_RATIO],
    #                       phase=0.)
    wfm = np.load('/home/awreed/SINR3D/data/wfm/20khz_bw_lfm.npy')[0:100]

    temp = np.mean(system_data[c.TEMPS])
    speed_of_sound = 331.4 + 0.6 * temp

    assert tx_coords.shape[0] % args.num_trans_per_inr == 0, \
        "Num trans per inr must divide evenly into number of transducers (" + str(tx_coords.shape[0]) + ")"

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
    #np.sqrt(np.sum(np.abs(wfm_convolve) ** 2))
    if args.normalize_each:
        gt_wfms = gt_wfms / torch.sqrt(torch.sum(torch.abs(gt_wfms)**2, dim=-1))[..., None]
    else:
        gt_wfms = gt_wfms / gt_wfms.abs().max()

    # Define sampling distances
    #dists_norm = torch.linspace(0, 1, args.num_radial).to(dev)
    dists_norm = torch.linspace(0, 1, wfm_crop_settings[c.NUM_SAMPLES]).to(dev)
    dists_scene = wfm_crop_settings[c.MIN_DIST] + \
                  dists_norm * (wfm_crop_settings[c.MAX_DIST] - wfm_crop_settings[c.MIN_DIST])

    # Precompute unscaled time series
    kernel = no_rc_kernel_from_waveform(wfm, wfm_crop_settings[c.NUM_SAMPLES]).to(dev)

    tsd = precompute_time_series(dists_scene,
                                 wfm_crop_settings[c.MIN_DIST],
                                 kernel,
                                 speed_of_sound,
                                 system_data[c.SYS_PARAMS][c.FS],
                                 wfm_crop_settings[c.NUM_SAMPLES])

    trans = divide_chunks(list(range(0, tx_coords.shape[0])), args.num_trans_per_inr)

    for batch_num, trans_batch in enumerate(trans):
        if batch_num == 1:
            exit(0)
        ndim = 1
        if args.complex_weights:
            ndim = 2

        model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=ndim,
                                              encoding_config=inr_config["encoding"],
                                              network_config=inr_config["network"])

        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate)

        # [num trans, num_radial, 2]
        samples = torch.stack(torch.meshgrid(dists_norm, torch.linspace(0, 1, len(trans_batch)).
                                             to(dev))).permute(2, 1, 0)
        print("Batch", batch_num, "/", tx_coords.shape[0]//args.num_trans_per_inr)
        for epoch in range(1, args.number_iterations):
            # Model predicts the weights at each time sample
            sample_batch = samples.reshape(-1, 2)
            weights = model(sample_batch)

            if args.complex_weights:
                weights = weights.reshape(len(trans_batch), args.num_radial, 2)
            else:
                weights = weights.reshape(len(trans_batch), args.num_radial)

            sparsity = torch.mean(torch.abs(weights))
            # Use weights to scale the precomputed waveform response and integrate
            #print("4")
            if args.complex_weights:
                est_wfm_real = radial_delay_wfms_fast(tsd, weights[..., 0]).real
                est_wfm_imag = radial_delay_wfms_fast(tsd, weights[..., 1]).imag
            else:
                est_wfm = radial_delay_wfms_fast(tsd, weights).real

            est_wfm_angle = torch.angle(hilbert_torch(est_wfm))

            phase_weight = 1e-4
            phase_loss = phase_weight * torch.mean(torch.abs(est_wfm_angle[:, 1:] - est_wfm_angle[:, :-1]))
            #phase_loss = torch.tensor([0.]).to(dev)

            if args.complex_weights:
                loss = torch.nn.functional.mse_loss(est_wfm_real.squeeze(),
                                                    gt_wfms[trans_batch, :].squeeze().real,
                                                    reduction='mean') + \
                    torch.nn.functional.mse_loss(est_wfm_imag.squeeze(),
                                             gt_wfms[trans_batch, :].squeeze().imag,
                                             reduction='mean')

            else:
                # Compute a loss with ground truth waveforms
                loss = torch.nn.functional.mse_loss(est_wfm.squeeze(),
                                                    gt_wfms[trans_batch, :].squeeze(),
                                                    reduction='mean') \

            sparsity_loss = args.sparsity*sparsity

            total_loss = loss + sparsity_loss + phase_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #print("5")

            if epoch % args.info_every == 0:
                if args.complex_weights:
                    est_wfm = torch.complex(real=est_wfm_real.float(), imag=est_wfm_imag.float())
                    weights = torch.complex(real=weights[..., 0].float(),
                                            imag=weights[..., 1].float()).squeeze().detach().cpu().numpy()
                else:
                    weights = weights.squeeze().detach().cpu().numpy().astype('float32')

                print("Epoch", epoch, "Loss", loss.item(), "Sparsity", sparsity_loss.item(), "Phase", phase_loss.item())
                plt.figure(figsize=(40, 20))
                plt.subplot(1, 2, 1)
                plt.imshow(np.abs(weights[:, ::2]))
                plt.subplot(1, 2, 2)
                plt.imshow(gt_wfms[trans_batch, :].abs().squeeze().detach().cpu().numpy().astype('float32'))
                plt.colorbar()
                plt.savefig(os.path.join(args.output_dir, 'comparison' + str(batch_num) + '_' + str(epoch) + '.png'))

                if args.complex_weights:
                    plt.figure()
                    plt.plot(est_wfm.real.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                    plt.plot(gt_wfms[trans_batch, :].real.squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'wfm_est_real' + str(epoch) + '.png'))
                    plt.close()

                    plt.figure()
                    plt.plot(est_wfm.imag.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                    plt.plot(gt_wfms[trans_batch, :].imag.squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'wfm_est_imag' + str(epoch) + '.png'))
                    plt.close()
                else:
                    plt.figure()
                    plt.plot(est_wfm.squeeze().detach().cpu().numpy()[0], label='est', alpha=0.5)
                    plt.plot(gt_wfms[trans_batch, :].squeeze().detach().cpu().numpy()[0], label='gt', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'wfm_est_real' + str(batch_num) + '_' + str(epoch) + '.png'))
                    plt.close()

                if args.complex_weights:
                    plt.figure()
                    plt.plot(weights[0, ...].real.squeeze(), label='est', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'weights_real' + str(epoch) + '.png'))
                    plt.close('all')

                    plt.figure()
                    plt.plot(weights[0, ...].imag.squeeze(), label='est', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'weights_imag' + str(epoch) + '.png'))
                    plt.close('all')
                else:
                    plt.figure()
                    plt.plot(weights[0].squeeze(), label='est', alpha=0.5)
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, 'weights' + str(batch_num) + '_' + str(epoch) + '.png'))
                    plt.close('all')

                    plt.figure(figsize=(18, 6))

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
                    plt.savefig(os.path.join(args.output_dir, 'weights_comp' + str(batch_num) + '_' + str(epoch) + '.png'))
                    plt.close('all')

        print("Saving weights...")
        sub_trans_batch = divide_chunks(trans_batch, args.batch_size)
        if args.complex_weights:
            batch_weights = np.zeros((len(trans_batch), args.num_radial, 2))
        else:
            batch_weights = np.zeros((len(trans_batch), args.num_radial))
        for count, batch in enumerate(sub_trans_batch):
            sample_batch = samples.reshape(-1, 2)
            weights = model(sample_batch)

            if args.complex_weights:
                weights = weights.reshape(len(batch), args.num_radial, 2)
            else:
                weights = weights.reshape(len(batch), args.num_radial)

            batch_weights[count*len(batch):count*len(batch) + len(batch)] = weights.detach().cpu().numpy()

        np.save(os.path.join(args.output_dir, c.WEIGHT_PREFIX + str(min(trans_batch)) + '_' + str(max(trans_batch))
                             ), batch_weights)

        torch.save(model.state_dict(), os.path.join(args.output_dir, c.NUMPY,
                                                    c.MODEL_PREFIX +
                                                    str(min(trans_batch)) + '_' + str(max(trans_batch)) + '.pt'))
