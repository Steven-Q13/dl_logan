import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft
from scipy.optimize import brentq


# Gaussian Process Tutorial:
# https://peterroelants.github.io/posts/gaussian-process-tutorial/

# Exponentiated quadratic kernel
def rbf_kernel(sd):
    def rbf_kernel_func(x1, x2):
        sq_norm = distance.cdist(x1, x2, 'sqeuclidean') / (-2 * sd**2)
        return np.exp(sq_norm)
    return rbf_kernel_func

def quadratic_kernel(const):
    def quadratic_kernel_func(x1,x2):
        x1 = np.expand_dims(x1, axis=1)
        x2 = np.expand_dims(x2, axis=0)
        return np.square(np.add(np.dot(x1,x2),const))
    return quadratic_kernel_func

# sinc Kernel as parameterized by it spower spectral density
# sig_pow: Total power of bandpass frequencys of kernel
# center: Frequency +/- center of rectangular passband
# width: Frequency width of rectangular passband
def sinc_kernel(sig_pow, center, width):
    def sinc_kernel_func(x1, x2):
        data_size = x2.shape[0]
        x2_mat = np.zeros([data_size, data_size])
        x_diff = x1.reshape([data_size, 1]) - x2.reshape([data_size])
        return sinc_func(x_diff, sig_pow, center, width)
    return sinc_kernel_func

# sinc Kernel as parameterized by it spower spectral density
# sig_pow: Total power of bandpass frequencys of kernel
# center: Frequency +/- center of rectangular passband
# width: Frequency width of rectangular passband
def sinc_func(x, sig_pow, center, width):
    return sig_pow**2 * np.sinc(width*x) * np.cos(2*np.pi*center*x) / (2*width)


# min/max time diff need to be 1/fund_freq
## FIX DEVICE
class PeriodicTrain(torch.utils.data.IterableDataset):
    ZERO_RATIO = 0.15
    EPSILON = 0.0000001
    PI = torch.acos(torch.zeros(1)).item() * 2
    def __init__(self, 
            min_time, max_time, 
            lowpass, highpass, cache_size,
            num_freqs, avg_freq_amp,
            err_spectral_freqs, err_spectral_avg_amp,
            err_temporal, err_temporal_size,
            err_dropout, err_dropout_size,
            model, mode='spectral'):
        self.min_time = min_time
        self.max_time = max_time
        self.lowpass = lowpass
        self.highpass = highpass
        self.cache_size = cache_size
        self.num_freqs = num_freqs
        self.freq_amp = avg_freq_amp
        self.err_temporal_ratio = err_temporal
        self.err_freq_ratio = err_spectral_freqs
        self.err_amp_ratio = err_spectral_avg_amp
        self.err_dropout_ratio = err_dropout
        self.err_temporal_size = err_temporal_size
        self.err_dropout_size = err_dropout_size
        self.mode = mode
        self.model = model
        self.max_zeros = 0

        self.fund_freq = (self.highpass - self.lowpass) / self.num_freqs
        self.freqs = torch.linspace(
            self.lowpass, self.highpass-self.fund_freq, 
            steps=self.num_freqs)[:,None]
        err_freq_padding = round(
            self.lowpass / (2*self.fund_freq)) * self.fund_freq
        self.err_lowpass = self.lowpass - err_freq_padding
        self.err_highpass = self.highpass + err_freq_padding
        self.err_num_freqs = round(
            (self.err_highpass - self.err_lowpass) / self.fund_freq)
        self.err_pad_size = (self.err_num_freqs - self.num_freqs) // 2
        self.err_freqs = torch.linspace(
            self.err_lowpass, self.err_highpass-self.fund_freq, 
            steps=self.err_num_freqs)[:,None]
        self.err_freqs_filter = torch.logical_or(
            torch.le(self.err_freqs, self.err_lowpass),
            torch.gt(self.err_freqs, self.err_highpass))
        # Check min_zc_interval; assuming max freq (including spectral leakage) 
        self.min_zc_interval = 1 / self.err_highpass / 12

    def __iter__(self):
        self.batch_rand_vals()
        self.idx = 0
        return self

    def fourier_gen(coeff, freqs):
        def gen_func(x):
            flip_coeff = torch.flip(-1*coeff,[0,1])
            flip_freqs = torch.flip(-1*freqs,[0,1])
            return (torch.mm(coeff, 
                    torch.exp(2j*PeriodicTrain.PI*freqs*x)).item() 
                + torch.mm(flip_coeff, 
                    torch.exp(2j*PeriodicTrain.PI*flip_freqs*x)).item())
        return gen_func

    def cos_gen(coeff, freqs):
        def gen_func(x):
            return torch.mm(
               coeff*2, torch.cos(2*PeriodicTrain.PI*freqs*x)).item()
        return gen_func

    def real_filter(func):
        def filter(x):
            return func(x).real()
        return filter

    def find_zeros(self, func, zeros):
        min_val, max_val = self.min_time, self.min_time + self.min_zc_interval
        idx = 0
        while min_val < self.max_time:
            if func(min_val) * func(max_val) < 0:
                zeros[idx] = brentq(func, min_val, max_val)
                idx += 1
            min_val, max_val = max_val, min(
                max_val + self.min_zc_interval, self.max_time)

    def batch_rand_vals(self):
        self.coeff = torch.rand(
            [1, self.num_freqs, self.cache_size], dtype=torch.float32)
        self.coeff = (
            ((self.coeff - 0.5) * self.freq_amp) + self.freq_amp)
        # Adds randomness to nature of generated baseline signals
        self.amp_filter = (torch.rand(
            (1, self.num_freqs, self.cache_size), dtype=torch.float) 
            < 0.75)
        self.coeff *= self.amp_filter


        self.err_coeff = torch.rand(
            [1, self.err_num_freqs, self.cache_size], dtype=torch.float32) 
        self.err_coeff = (
            ((self.err_coeff - 0.5) * self.err_amp_ratio) + self.err_amp_ratio)
        self.err_amp_filter = (torch.rand(
            (1, self.err_num_freqs, self.cache_size), dtype=torch.float) 
            < self.err_freq_ratio)
        self.err_coeff *= self.err_amp_filter
        self.err_coeff[:,self.err_pad_size:self.err_pad_size+self.num_freqs] = (
            self.coeff)

        self.err_temporal = torch.empty(
            [self.err_temporal_size, self.cache_size]).normal_(
                mean=self.err_temporal_ratio, std=self.err_temporal_ratio/2)
        self.err_dropout = (torch.rand(
            [self.err_dropout_size, self.cache_size], dtype=torch.float) 
            < self.err_dropout_ratio)

    # Needs to check free zeros
    def __next__(self):
        if(self.idx >= self.cache_size): 
            self.batch_rand_vals()
            self.idx = 0
        sig_func = PeriodicTrain.cos_gen(
            self.coeff[:,:,self.idx], self.freqs)
        target = torch.zeros(self.zeros_size())
        self.find_zeros(sig_func, target)
        self.max_zeros = max(self.max_zeros, torch.count_nonzero(target))

        if self.mode == 'spectral':
            err_func = PeriodicTrain.cos_gen(
                self.err_coeff[:,:,self.idx], self.err_freqs)
            input = torch.zeros(self.zeros_size())
            self.find_zeros(err_func, input)
        if self.mode == 'temporal':
            target = self.model.encode(target, 'spectral')
            input = self.err_temporal[:,self.idx] * target
        if self.mode == 'dropout':
            target = self.model.encode(target, 'temporal')
            input = self.err_dropout[:,self.idx] * target
        self.idx += 1
        return (input.detach(), target)

    def zeros_size(self):
        return round(self.ZERO_RATIO * (self.max_time-self.min_time) 
            / self.min_zc_interval)

    def zeros_size_est(min_time, max_time, lowpass, highpass, num_freqs):
        fund_freq = (highpass - lowpass) / num_freqs
        err_freq_padding = round(lowpass / (2*fund_freq)) * fund_freq
        err_highpass = highpass + lowpass - err_freq_padding
        min_zc_interval = 1 / err_highpass / 12
        return round(
            PeriodicTrain.ZERO_RATIO * (max_time-min_time) / min_zc_interval)


    def max_zeros(self):
        return self.max_zeros

    def get_fund_freq(self):
        return self.fund_freq

    def get_min_zc_interval(self):
        return self.min_zc_interval

    def num_samples(self):
        return self.sample_rate * (self.max_time - self.min_time)

    def set_mode(self, mode):
        self.mode = mode

    def check_num_samples(max_time, min_time, sample_rate):
        return sample_rate * (max_time - min_time)



class AperiodicTrain(torch.utils.data.IterableDataset):
    ZERO_RATIO = 0.5
    def __init__(self, min_time, max_time, lowpass, highpass, kernel, batch_size):
        self.min_time = min_time
        self.max_time = max_time
        self.sample_r = round(highpass * 10)
        # self.kernel = kernel
        self.num_samples = self.sample_r * (self.max_time - self.min_time)
        self.batch_size = batch_size
        self.x = np.linspace(min_time, max_time-1/self.sample_r, num=self.num_samples)
        self.cov_func = kernel(self.x,self.x)
        self.mean_func = np.zeros(self.num_samples)
        self.max_zeros = 0

    def __iter__(self):
        self.idx = 0
        self.freqs = np.random.multivariate_normal(
            mean=self.mean_func, cov=self.cov_func, size=self.batch_size
        ).astype(np.float32)
        return self

    # Needs to check free zeros
    # Can't currently handle values of exactly 0?
    # Treats -#, 0, -# as sign change
    # Currently only linear interpolation
    def __next__(self):
        #Check num of times is called
        if(self.idx >= self.batch_size):
            self.idx = 0
            self.freqs = np.random.multivariate_normal(
                mean=self.mean_func, cov=self.cov_func, size=self.batch_size
            ).astype(np.float32)
        curr_target = self.freqs[self.idx]
        zero_idx = np.where(np.diff(np.signbit(curr_target)))[0]
        comp_arr = np.roll(curr_target, -1)
        # For linear approximation y=0, -b/m = x
        zero_offset = np.divide(curr_target, np.subtract(curr_target,comp_arr))
        zeros = np.add(np.divide(zero_offset,self.sample_r), self.x)
        zeros = np.take(zeros, zero_idx)
        pad_len = self.input_size() - zeros.size
        zeros_pad = np.pad(zeros, (0,pad_len))
        self.idx = self.idx + 1
        self.max_zeros = max(self.max_zeros, zeros.size)
        # (X, y)
        return (torch.from_numpy(zeros_pad), torch.from_numpy(curr_target))

    def time_dim(self):
        return self.x

    def input_size(self):
        return round(self.num_samples * self.ZERO_RATIO)

    def output_size(self):
        return self.num_samples

    def max_zeros():
        return self.max_zeros

    def sample_rate(self):
        return self.sample_r

