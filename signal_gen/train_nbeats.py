import matplotlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft

from signal_gen import *
from models import *
from visualize import *

# Run: python3 signal_gen/train_nbeats.py mac 400 64 900 800 test1
np.random.seed(13)
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=1000000)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

cli_args = {
    "location": sys.argv[1] if len(sys.argv) > 1 else 'mac',
    "num_epochs": int(sys.argv[2]) if len(sys.argv) > 2 else 1024,
    "batch_size": int(sys.argv[3]) if len(sys.argv) > 3 else 64,
    "fnn_size": int(sys.argv[4]) if len(sys.argv) > 4 else 1,
    "basis_size": int(sys.argv[5]) if len(sys.argv) > 5 else 2,
    "version": sys.argv[6] if len(sys.argv) > 5 else 'test1',
}

# Set hyperparameters
OCTAVE_MULT = 10 ** 0.3
MINI_OCTAVE_MULT = 1.3
CACHE_SIZE = 1000
BATCH_SIZE = cli_args['batch_size']
REDUCE_LR = 4
NUM_EPOCH = cli_args['num_epochs']

passband_mult = MINI_OCTAVE_MULT
avg_freq_amp = 16
lowpass = 4
highpass = lowpass * passband_mult
min_time = 0
num_freqs = 100
num_samples = 1024
fnn_size = cli_args['fnn_size']
basis_size = cli_args['basis_size']
block_types = ['generic', 'generic', 'generic', 'fourier', 'fourier']

save = True
name = 'nbeats'
version = cli_args['version']
if cli_args['location'] == 'mac':
    curr_path = '/Users/steven-q13/College/Research/PERL/logan/dl_logan/signal_gen'
else:
    curr_path = '/vulcanscratch/squeen0/dl_logan'
model_path = curr_path + '/models/' + name + '_V' + version + '.pyt'
plot_path = curr_path + '/results/' + name + '_V' + version + '.png'
seq_plot_path = curr_path + '/results/' + name + '_seq_V' + version + '.png'

# Setup data loaders
train = PeriodicTrain_NBeats(min_time, num_freqs, avg_freq_amp, 
    lowpass, highpass, num_samples, CACHE_SIZE)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
max_time = train.get_max_time()
model = NBeats_Model(device=DEVICE, input_size=num_samples, 
    output_size=num_samples, fnn_size=fnn_size, 
    basis_size=basis_size, block_types=block_types,
    min_freq=lowpass, time_len=max_time)

print("NBeats")
print('Number of Epochs: %d' % NUM_EPOCH)
print('Batch Size: %d' % BATCH_SIZE)
print('Block Types: ' + str(block_types))
print('FNN Size: %d' % fnn_size)
print('Basis Size: %d' % basis_size)
print('Lowpass: %d' % lowpass)
print('Number of Frequencies: %d' % num_freqs)
print('Frequency Amplitude: %d' % avg_freq_amp)
print('Version: ' + version)
print('Number of Parameters: %d' % model.count_params())

start = time.time()
# Infinite dataset so can view as mini-batch, or each batch is individual epoch
for batch_idx, (X,y) in enumerate(train_loader):
    X = X.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    y_p = model.forward(X)
    model.backward(y, y_p)

    if batch_idx == NUM_EPOCH - 1:
        print("\nMSE Loss: %.6f" % model.calc_loss(y,y_p))
        avg_dumb_mse_loss = (
            torch.sum(torch.square(y)).item()/(y.shape[1]*BATCH_SIZE))
        #print("MSE Loss from y_p=0: %.6f" % avg_dumb_mse_loss)
        #print(y_p[0,0,:])
        #print(y[0,0,:])
        #show = torch.cat((X[0:1,:],y[0:1,:],y_p[0:1,:]),0)
        #print(torch.transpose(show,0,1))
        break
    elif (batch_idx+1) % 100 == 0:
        #print('Epoch: ' + str(batch_idx+1))
        epoch = batch_idx + 1
        print("Epoch: %d - %d%%" % (epoch, epoch / NUM_EPOCH * 100))
        print('Time: %s' % timeSince(start, epoch / NUM_EPOCH))
        print("Error Last: %.6f" % model.get_train_loss()[-1])
    elif (batch_idx+1) % 100 == 0:
        plot_loss(model, skip_beg=batch_idx // 2, 
            legend=['Transformer Sequence Prediction'])
        plt.savefig(curr_path + '/results/' + name + '_loss.png')
        plt.clf()
        plot_seq(y, y_p)
        plt.savefig(seq_plot_path)
        plt.clf()
        if save: model.save_model(model_path)


plot_loss(model, legend=['NBeats Sequence Prediction'])
plt.savefig(curr_path + '/results/' + name + '_loss.png')
plt.clf()
plot_seq(y, y_p)
plt.savefig(seq_plot_path)
if cli_args['location'] == 'mac': plt.show()
if save: model.save_model(model_path)

