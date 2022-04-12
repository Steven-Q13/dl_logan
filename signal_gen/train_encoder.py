import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft

from signal_gen import *
from models import *
from visualize import *

np.random.seed(13)
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=10000)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Set hyperparameters
OCTAVE_MULT = 10 ** 0.3
MINI_OCTAVE_MULT = 1.3
CACHE_SIZE = 10000
BATCH_SIZE = 4
REDUCE_LR = 4
NUM_EPOCH = 18

passband_mult = MINI_OCTAVE_MULT
avg_freq_amp = 16
num_funcs = 1 
lowpass = 4
highpass = lowpass * passband_mult
min_time = 0
max_time = 20
kernel = sinc_kernel(avg_freq_amp, (highpass + lowpass)/2, highpass-lowpass)
num_freqs = 100
zc_std_dev_ratio = 1
loss_weights = {'temporal':0.7, 'spectral':0.0, 'sparsity':0.3}
loss_weights2 = {'temporal':1.0, 'spectral':0.0, 'sparsity':0.0}
sparsity = 0.2
orig_loss = [0] * NUM_EPOCH

spectral_encoding_size = 1000
temporal_encoding_size = 1000
dropout_encoding_size = 1000
encoder_input_size = PeriodicTrain.zeros_size_est(
    min_time, max_time, lowpass, highpass, num_freqs)
err_spectral_freqs_ratio = 0.15
err_spectral_avg_amp = 0.15
err_temporal_ratio = 0.1
err_dropout_ratio = 0.05

save = True
version = 'test1'
curr_path = '/Users/steven-q13/College/Research/PERL/logan/dl_logan/signal_gen'
model_path = curr_path + '/models/sparse_V' + version + '.pyt'
mode = 'dropout'

#model = SDA_Model(DEVICE, path=model_path)
model = SDA_Model(DEVICE, input_size=encoder_input_size, 
    spectral_encoding_size=spectral_encoding_size, 
    temporal_encoding_size=temporal_encoding_size,
    dropout_encoding_size=dropout_encoding_size,
    loss_weights=loss_weights, sparsity=sparsity, activation='PReLU')
train = PeriodicTrain(
    min_time, max_time, lowpass, highpass, CACHE_SIZE, 
    num_freqs, avg_freq_amp, 
    err_spectral_freqs_ratio, err_spectral_avg_amp,
    err_temporal_ratio, spectral_encoding_size,
    err_dropout_ratio, temporal_encoding_size, model)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

model.set_mode(mode)
train.set_mode(mode)
start = time.time()
# Infinite dataset so can view as mini-batch, or each batch is individual epoch
for batch_idx, (X,y) in enumerate(train_loader):
    X = X.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    y_p = model.forward(X)
    model.backward(y, y_p)
    orig_loss[batch_idx] = (
        torch.sum(torch.square(X-y)).item() / (train.zeros_size()*BATCH_SIZE))

    if batch_idx == NUM_EPOCH - 1:
        print("\nMSE Loss: %.6f" % model.calc_loss(y,y_p))
        avg_dumb_mse_loss = (
            torch.sum(torch.square(X-y)).item()/(train.zeros_size()*BATCH_SIZE))
        print("MSE Loss from y_p=0: %.6f" % avg_dumb_mse_loss)
        show = torch.cat((X[0:1,:],y[0:1,:],y_p[0:1,:]),0)
        #print(torch.transpose(show,0,1))
        break
    elif (batch_idx+1) % 5 == 0:
        #print('Epoch: ' + str(batch_idx+1))
        epoch = batch_idx + 1
        print("Epoch: %d - %d%%" % (epoch, epoch / NUM_EPOCH * 100))
        print('Time: %s' % timeSince(start, epoch / NUM_EPOCH))
        print(
            "Sparse Total: %.6f" % 
            torch.sum(torch.abs(model.encode_layer(y_p,set(mode)))))

    elif (batch_idx+1) % 16 == 0:
        plot_loss(model, skip_beg=batch_idx // 2, 
            legend=['SDA Sparse Encoding, PReLU'])
        plt.savefig('sda_loss.png')
        plt.clf()

        fig = matplotlib.pyplot.gcf()
        model.show_encoding(y_p, set(mode), title='Sparse Encoding')
        fig.set_size_inches(13, 6)
        plt.savefig('encodings')
        plt.clf()
        if save: model.save_model(model_path)


plot_loss(model, legend=['SDA Sparse Encoding, PReLU'])
plt.savefig('sda_loss.png')
plt.show()

fig = matplotlib.pyplot.gcf()
model.show_encoding(y_p, set(mode), title='Sparse Encoding')
fig.set_size_inches(13, 6)
plt.show()

if save: model.save_model(model_path)

