import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft
from scipy.optimize import brentq

PI = torch.acos(torch.zeros(1)).item() * 2
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

## FIX DEVICE
class PeriodicTrain_NBeats(torch.utils.data.IterableDataset):
    EPSILON = 0.000001
    def __init__(self, min_time, num_freqs, avg_freq_amp,
            lowpass, highpass, num_samples, cache_size):
        self.min_time = min_time
        self.num_freqs = num_freqs
        self.lowpass = lowpass
        self.highpass = highpass
        self.freq_amp = avg_freq_amp
        self.num_samples = num_samples
        self.cache_size = cache_size

        self.fund_freq = (self.highpass - self.lowpass) / self.num_freqs
        self.freqs = torch.linspace(self.lowpass, 
            self.highpass-self.fund_freq, steps=self.num_freqs)[:,None]
        self.max_time = min_time + 1/self.fund_freq
        self.sample_rate = self.num_samples / (self.max_time - self.min_time)
        if self.num_samples and round(highpass * 2.2) > self.sample_rate:
            raise ValueError('PeriodicTrain recieved too small num_samples.')
        self.x = torch.linspace(self.min_time, 
            self.max_time-1/self.sample_rate, self.num_samples)[None,:]
        self.min_zc_interval = 1 / self.highpass / 12
        self.zeros_size = int(1.85*(self.max_time-self.min_time)*self.highpass)
        self.max_zeros = 0

    def __iter__(self):
        self.batch_rand_vals()
        self.idx = 0
        return self

    def find_zeros(self, sig):
        zero_bool = torch.diff(torch.signbit(sig))
        comp_arr = torch.roll(sig, 1)
        zero_offset = torch.divide(comp_arr, torch.subtract(comp_arr,sig))
        zeros = torch.add(torch.divide(zero_offset,self.sample_rate), self.x)
        zeros[:,1:] = zeros[:,1:] * zero_bool
        return torch.nan_to_num(zeros, nan=0.0)

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

    # Needs to check free zeros
    def __next__(self):
        #sig[num_samples,1]
        if(self.idx >= self.cache_size): 
            self.batch_rand_vals()
            self.idx = 0
        sig = PeriodicTrain_Transformer.cos_func(
            self.x, self.coeff[:,:,self.idx], self.freqs)
        zeros = self.find_zeros(sig)
        zeros[:,0] = sig[:,0]
        zeros[:,-1] = sig[:,-1]
        self.idx += 1
        return (zeros.detach(), sig.detach())

    def cos_func(x, coeff, freqs):
        return torch.mm(coeff*2, torch.cos(2*PI*torch.mm(freqs,x)))

    def get_fund_freq(self):
        return self.fund_freq

    def get_max_time(self):
        return self.max_time

    def get_max_zeros(self):
        return self.max_zeros


## FIX DEVICE
class PeriodicTrain_Transformer(torch.utils.data.IterableDataset):
    EPSILON = 0.000001
    def __init__(self, min_time, num_freqs, avg_freq_amp,
            lowpass, highpass, num_samples, cache_size):
        self.min_time = min_time
        self.num_freqs = num_freqs
        self.lowpass = lowpass
        self.highpass = highpass
        self.freq_amp = avg_freq_amp
        self.num_samples = num_samples
        self.cache_size = cache_size

        self.fund_freq = (self.highpass - self.lowpass) / self.num_freqs
        self.freqs = torch.linspace(self.lowpass, 
            self.highpass-self.fund_freq, steps=self.num_freqs)[:,None]
        self.max_time = min_time + 1/self.fund_freq
        self.sample_rate = self.num_samples / (self.max_time - self.min_time)
        if self.num_samples and round(highpass * 2.2) > self.sample_rate:
            raise ValueError('PeriodicTrain recieved too small num_samples.')
        self.x = torch.linspace(self.min_time, 
            self.max_time-1/self.sample_rate, self.num_samples)[None,:]
        self.min_zc_interval = 1 / self.highpass / 12
        self.zeros_size = int(1.85*(self.max_time-self.min_time)*self.highpass)
        self.max_zeros = 0

    def __iter__(self):
        self.batch_rand_vals()
        self.idx = 0
        return self

    def find_zeros(self, sig):
        zero_idx = torch.where(torch.diff(torch.signbit(sig)))[1]
        comp_arr = torch.roll(sig, -1)
        zero_offset = torch.divide(sig, torch.subtract(sig,comp_arr))
        zeros = torch.add(torch.divide(zero_offset,self.sample_rate), self.x)
        zero_feats = torch.zeros(zero_idx.shape[0], sig.shape[1])
        zero_feats[torch.arange(zero_idx.shape[0]),zero_idx] = 1
        return zero_feats * zeros

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

    # Needs to check free zeros
    def __next__(self):
        #sig[num_samples,1]
        if(self.idx >= self.cache_size): 
            self.batch_rand_vals()
            self.idx = 0
        sig = PeriodicTrain_Transformer.cos_func(
            self.x, self.coeff[:,:,self.idx], self.freqs)
        zeros = self.find_zeros(sig)
        self.max_zeros = max(self.max_zeros, zeros.shape[0])
        pad_len = self.zeros_size - zeros.shape[0]
        zeros_pad = torch.nn.functional.pad(zeros, (0,0,0,pad_len))
        start = torch.zeros(sig.shape)
        start[:,0] = sig[:,0]
        self.idx += 1
        return (zeros_pad.detach(), start.detach(), sig.detach())

    def cos_func(x, coeff, freqs):
        return torch.mm(coeff*2, torch.cos(2*PI*torch.mm(freqs,x)))

    def get_fund_freq(self):
        return self.fund_freq

    def get_max_time(self):
        return self.max_time

    def get_max_zeros(self):
        return self.max_zeros


# min/max time diff need to be 1/fund_freq
## FIX DEVICE
class PeriodicTrain_SDA(torch.utils.data.IterableDataset):
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
            return (torch.mm(coeff, torch.exp(2j*PI*freqs*x)).item() 
                + torch.mm(flip_coeff, torch.exp(2j*PI*flip_freqs*x)).item())
        return gen_func

    def cos_gen(coeff, freqs):
        def gen_func(x):
            return torch.mm(coeff*2, torch.cos(2*PI*freqs*x)).item()
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





'''
class PeriodicTrain_MLP(torch.utils.data.IterableDataset):
    EPSILON = 0.000001
    def __init__(self, min_time, num_freqs, avg_freq_amp,
            lowpass, highpass, num_samples, cache_size, batch_size):
        self.gen = PeriodicTrain_Transformer(min_time, num_freqs, avg_freq_amp,
            lowpass, highpass, num_samples, cache_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.gen, batch_size=batch_size)

    def __iter__(self):
        return self

    # Needs to check free zeros
    def __next__(self):
        for batch_idx, (X,gen,y) in enumerate(train_loader):
            # X[batch,num_zeros,samples] -> X[batches,samples]
            X = torch.sum(X,1)
            batch_size = X.shape[0]
            signal_size = X.shape[1]
            X = torch.reshape(X,(batch_size * signal_size, 1))
            idx = torch.arange(0,batch_size)
            idx = torch.reshape(torch.transpose(idx.repeat(signal_size,1),0,1),(batch_size*signal_size,1))
            X = torch.cat((X,idx),1)
            #X[target, group_idx, time_idx]
            X = torch.cat((X, torch.arange(batch_size*signal_size,1)[:,None]),1)
            pX = pd.DataFrame(X).astype("float")
            pX.rename(columns={0:'target',1:'group',2:'time'})
            dataloader = TimeSeriesDataset(pX, group_ids=['group'], 
                target='target', time_idx='time',
                max_encoder_length=signal_size, 
                max_prediction_length=signal_size,
                time_varying_unknown_reals=['target'])
            return dataloader, y

    def get_fund_freq(self):
        return self.gen.fund_freq()

    def get_max_time(self):
        return self.gen.max_time()

    def get_max_zeros(self):
        return self.gen.max_zeros()
'''
