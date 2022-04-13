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
torch.set_printoptions(threshold=1000000)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Set hyperparameters
OCTAVE_MULT = 10 ** 0.3
MINI_OCTAVE_MULT = 1.3
CACHE_SIZE = 10000
BATCH_SIZE = 64
REDUCE_LR = 4
NUM_EPOCH = 32

passband_mult = MINI_OCTAVE_MULT
avg_freq_amp = 16
lowpass = 4
highpass = lowpass * passband_mult
min_time = 0
num_freqs = 100
num_samples = 1024
num_layers = 1
num_heads = 1
attention_size = 256

save = True
version = 'test1'
curr_path = '/Users/steven-q13/College/Research/PERL/logan/dl_logan/signal_gen'
model_path = curr_path + '/models/transformer_V' + version + '.pyt'
mean_func_loss = [0] * NUM_EPOCH

# Setup data loaders
train = PeriodicTrain_Transformer(min_time, num_freqs, avg_freq_amp, 
    lowpass, highpass, num_samples, CACHE_SIZE)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
max_time = train.get_max_time()

model = Transformer_Model(device=DEVICE,input_size=num_samples, 
    output_size=num_samples, num_layers=num_layers, 
    num_heads=num_heads, batch_first=True)

start = time.time()
# Infinite dataset so can view as mini-batch, or each batch is individual epoch
for batch_idx, (X,gen,y) in enumerate(train_loader):
    X = X.to(DEVICE, non_blocking=True)
    gen = gen.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    y_p = model.forward(X,gen)
    model.backward(y, y_p)

    if batch_idx == NUM_EPOCH - 1:
        print("\nMSE Loss: %.6f" % model.calc_loss(y,y_p))
        print(torch.sum(torch.square(y)).shape)
        avg_dumb_mse_loss = (
            torch.sum(torch.square(y)).item()/(y.shape[1]*BATCH_SIZE))
        print("MSE Loss from y_p=0: %.6f" % avg_dumb_mse_loss)
        print(y_p[0,0,:])
        print(y[0,0,:])
        #show = torch.cat((X[0:1,:],y[0:1,:],y_p[0:1,:]),0)
        #print(torch.transpose(show,0,1))
        break
    elif (batch_idx+1) % 5 == 0:
        #print('Epoch: ' + str(batch_idx+1))
        epoch = batch_idx + 1
        print("Epoch: %d - %d%%" % (epoch, epoch / NUM_EPOCH * 100))
        print('Time: %s' % timeSince(start, epoch / NUM_EPOCH))
        print("Error Last: %.6f" % model.get_train_loss()[-1])
    elif (batch_idx+1) % 16 == 0:
        plot_loss(model, skip_beg=batch_idx // 2, 
            legend=['Transformer Sequence Prediction'])
        plt.savefig('results/transformer_loss.png')
        plt.clf()
        if save: model.save_model(model_path)


plot_loss(model, legend=['Transformer Sequence Prediction'])
plt.savefig('results/transformer_loss.png')
plt.show()
if save: model.save_model(model_path)

