import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from ping_data import PingData
from utils import log_img, modulate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
from bs4 import BeautifulSoup
import re
import scipy
from sas_utils import view_fft, modulate_signal, match_filter_all, baseband_signal

class SystemParameters:
    def __init__(self, root_path, track_id, image_number, spt, **kwargs):
        self.Fs = kwargs.get('Fs', 125000)
        self.Fc = kwargs.get('Fc', 27500)
        self.root_path = root_path
        self.track_id = track_id
        self.image_number = image_number
        self.num_tx = 5  # Hardware param
        self.num_rx = 80  # Hardware param

        # read the sound speed
        sound_speed_lookup = {}
        with open(spt) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for i, row in enumerate(csv_reader):
                if i == 0:
                    pass
                else:
                    sound_speed_lookup[row[0]] = row[1]

        # if we can find the temperature then use it
        if self.track_id in sound_speed_lookup:
            temp_f = float(sound_speed_lookup[self.track_id])
            print("Found temperature in sound speed csv to be ", temp_f, 'degrees F')
        else:
            temp_f = 55
            print("Didn't find temperature in sound speed csv, using ", temp_f, 'degrees F')

        # convert to C
        temp_c = (temp_f - 32)*.5556
        print("Temp", temp_c, 'degrees celcius')
        # from physics
        self.c = 1404.3 + 4.7*temp_c - 0.04*(temp_c**2)
        print("Speed of sound", self.c)
        self.num_samples = None

        self.wfm_dur = 255 * 1e-6
        self.f_start = 35000
        self.f_stop = 20000

    def gen_kernel(self, kernel=None):
        if kernel is None:
            print("Using the analytic signal")
            times = np.linspace(0, self.wfm_dur - 1 / self.Fs, num=int((self.wfm_dur) * self.Fs * 1))
            LFM = scipy.signal.chirp(times, self.f_start, self.wfm_dur, self.f_stop, phi=-225)

            taylor_window = scipy.signal.windows.taylor(len(LFM), nbar=5, sll=40, norm=False)
            kernel = LFM * taylor_window

            sig = np.zeros(int(self.num_samples))
            sig[:len(kernel)] = kernel



            self.pulse_fft_kernel = np.fft.fft(scipy.signal.hilbert(sig))
        else:
            print("Using provided kernel")
            assert kernel.ndim == 1
            assert kernel.dtype == np.complex64 or kernel.dtype == np.complex128

            #times = np.linspace(0, self.wfm_dur - 1 / self.Fs, num=int((self.wfm_dur) * self.Fs * 1))
            #LFM = scipy.signal.chirp(times, self.f_start, self.wfm_dur, self.f_stop, phi=-225)#

            #taylor_window = scipy.signal.windows.taylor(len(LFM), nbar=5, sll=40, norm=False)
            #kernel_anal = LFM * taylor_window

            #kernel_normal = kernel[0:32]
            #kernel_normal = modulate(kernel_normal, fc=self.Fc, fs=self.Fs)
            #raise Warning("Cropping kernel to 32")
            #kernel_flipped = np.flip(kernel[0:32])
            #kernel_flipped = modulate(kernel_flipped, fc=self.Fc, fs=self.Fs)
            #kernel = kernel[0:32]
            kernel = kernel[:]
            #plt.figure()
            #plt.plot(np.imag(kernel))
            #plt.show()

            kernel = kernel / np.sqrt(np.mean(kernel * np.conj(kernel)))
            kernel = modulate(kernel, fc=self.Fc, fs=self.Fs)

            #plt.figure()
            #plt.plot(kernel_anal, label="analytic")
            #plt.plot(kernel, label="measured")
            #plt.show()
            #exit(0)


            sig = np.zeros(int(self.num_samples))
            sig[:len(kernel)] = kernel

            self.pulse_fft_kernel = np.fft.fft(scipy.signal.hilbert(sig))

        return kernel

    # TODO: read parameters from xml file
    def read_kernel(self):
        path = os.path.join(self.root_path, self.track_id, self.track_id + '.xml')
        with open(path, 'r') as f:
            data = f.read()

            data = BeautifulSoup(data, "html.parser")

            kernel_struct = data.find_all('kernel')
            kernel_struct = str(kernel_struct[1])

            real = re.findall(r'(?<=<real>)(.*)(?=</real>)', kernel_struct)
            real = np.array([float(x) for x in real])

            imag = re.findall(r'(?<=<imag>)(.*)(?=</imag>)', kernel_struct)
            imag = np.array([float(x) for x in imag])

            kernel = np.zeros((len(real)), dtype=np.complex64)
            kernel.real = real
            kernel.imag = imag

            return kernel

    # Read the match filtered waveforms
    def process_waveforms(self, find_ping=False, do_mf=False):
        # Load the Raw Data
        # assumes this matches assassin output base name
        base_name = 'unpack float - LF - down - '
        ftype = '.hdf5'
        fname = base_name + str(self.image_number) + ftype
        full_name = os.path.join(self.root_path, self.track_id, fname)

        with h5py.File(full_name, 'r') as f:
            d_imag = np.array(f['DataImag'][:])
            d_real = np.array(f['DataReal'][:])

            assert d_imag.shape == d_real.shape

            raw_data = np.zeros_like(d_real, dtype=np.complex128)
            raw_data.real = d_real
            raw_data.imag = d_imag
            raw_data = raw_data.T

        # Load the MF Data
        # assumes this matches assassin output base name
        base_name = 'matched filter - LF - down - '
        ftype = '.hdf5'
        fname = base_name + str(self.image_number) + ftype
        full_name = os.path.join(self.root_path, self.track_id, fname)

        with h5py.File(full_name, 'r') as f:
            d_imag = np.array(f['DataImag'][:])
            d_real = np.array(f['DataReal'][:])

            assert d_imag.shape == d_real.shape

            mf_data = np.zeros_like(d_real, dtype=np.complex128)
            mf_data.real = d_real
            mf_data.imag = d_imag
            mf_data = mf_data.T

            # Match filtered data given in Fourier domain
            mf_data = np.fft.ifft(mf_data, axis=0)

            self.num_samples = mf_data.shape[0]

            kernel = self.gen_kernel(kernel=self.read_kernel())
            #kernel = self.gen_kernel()
            raw_up = modulate_signal(raw_data.T, fc=self.Fc, fs=self.Fs)
            mf_up = modulate_signal(mf_data.T, fc=self.Fc, fs=self.Fs, keep_quadrature=True)
            mf_up = mf_data.T

            #np.save('')
            #print(raw_up.shape, mf_up.shape)

            plt.figure()
            plt.plot(np.real(raw_up[0, :])/np.abs(np.real(raw_up[0, :])).max(), label="Raw")
            plt.plot(np.abs(mf_up[0, :])/np.abs(mf_up[0, :].max()), label="MF")

            if do_mf:
                print("Doing match filtering")
                raw_data_up = modulate_signal(raw_data.T, fc=self.Fc, fs=self.Fs)
                #raw_data_up = raw_data
                #raw_data_up = raw_data_up - np.mean(raw_data_up, axis=-1, keepdims=True)
                #b, a = scipy.signal.butter(5, 18000 * 2 / self.Fs, btype='high')
                #raw_data_up = scipy.signal.lfilter(b, a, raw_data_up)
                #raw_data = np.roll(raw_data_up, -(b.shape[0] - 1) // 2, 1)
                print(raw_data_up.shape)
                mf_data = match_filter_all(raw_data_up, kernel)

                plt.plot(np.abs(mf_data)[0, :]/np.abs(mf_data)[0, :].max(), label="My MF")
                plt.legend()
                plt.show()
                #exit(0)

                #raw_data = baseband_signal(raw_data, fc=self.Fc, fs=self.Fs).T
                #mf_data = baseband_signal(mf_data, fc=self.Fc, fs=self.Fs).T


            # Plot response of single array element over track to find canidate ping number
            if find_ping:


                #plt.figure()
                #plt.imshow(np.real(raw_data[:, 74:-1:80]))
                #plt.show()

                plt.figure()
                plt.plot(np.real(raw_data[:, ...][:, 108]/np.real(raw_data[:, 74:-1:80][:, 108].max())))
                plt.plot(np.abs(mf_data[:, ...][:, 108]/np.abs(mf_data[:, 74:-1:80][:, 108].max())))

                f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 1))
                im = ax1.imshow(log_img(np.abs(raw_data)[:, 74::80]), cmap='jet')
                ax1.set_title('Mag Raw Time Series')
                ax1.set_xlabel('Ping')
                ax2.set_ylabel('Samples')
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='2%', pad=0.05)
                f.colorbar(im, cax=cax, orientation='vertical', label='Linear')

                im2 = ax2.imshow(log_img(np.abs(mf_data)[:, 74::80]), cmap='jet')
                ax2.set_title('Mag MF Time Series')
                ax2.set_xlabel('Ping')
                ax2.set_ylabel('Samples')
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size='2%', pad=0.05)
                f.colorbar(im2, cax=cax, orientation='vertical', label='dB')
                plt.tight_layout()
                plt.show()

            PD = PingData(num_tx=self.num_tx, num_rx=self.num_rx)
            PD.create_ping_list(raw_data, mf_data)

            return PD, raw_data, mf_data, kernel










