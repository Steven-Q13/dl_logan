import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import torch
from matplotlib.colors import Normalize
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft
from torch import nn

'''
TODO:
  - Add free zero checks
'''

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (~ %s)' % (asMinutes(s), asMinutes(rs))

# Logan's Theorem guarantees uniqueness within a constant time domain multiple,
#  so we fit to best amplitude multiple with a linear least squares regression.
#  Detach amplitude multiple tensor so it isn't learned/fit in backprop.
# Assumes input of 2D tensor - batch size x input dimension
def amplitude_mult(y, y_p, fit='lstsq_avg'):
    if fit == 'lstsq_indiv':
        for i in range(y.shape[0].item()):
            amp = torch.linalg.lstsq(
                torch.transpose(y_p[None,i,:],0,1), 
                torch.transpose(y[None,i,:],0,1)).solution
            y_p[i,:] *= amp[0].detach()
    elif fit == 'lstsq_avg':
        amp = torch.linalg.lstsq(
            torch.transpose(y_p,0,1), torch.transpose(y,0,1)).solution
        y_p = torch.mm(torch.transpose(amp,0,1).detach(), y_p)
    elif fit == 'var':
        # Assumes mean func f(x)=0
        y_var = torch.sum(
            torch.square(y-torch.zeros(y.shape)), dim=1) / y.shape[1]
        y_p_var = torch.sum(
            torch.square(y_p-torch.zeros(y_p.shape)), dim=1) / y_p.shape[1]
        amp = y_var / y_p_var
        y_p *= torch.sqrt(amp[:,None]).detach()
    else:
        ValueError('amplitude_mult recieved invalid fit type name.')
    return y_p

def zero_func(x):
    return torch.zeros(x)

# Very Basic Feed Forward Neural Network
class FNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN, self).__init__()
        self.linear_prelu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, round(input_size * 0.9)),
            torch.nn.PReLU(),
            torch.nn.Linear(round(input_size * 0.9), round(input_size * 2)),
            torch.nn.PReLU(),
            torch.nn.Linear(round(input_size * 2.0), round(output_size * 1.3)),
            torch.nn.PReLU(),
            torch.nn.Linear(round(output_size * 1.3), output_size)
        )

    def forward(self, x):
        # x = self.flatten(x)
        return self.linear_prelu_stack(x)


class FNN_Model():
    def __init__(self, DEVICE, input_size, output_size, 
            loss='L2', const_mult=False):
        self.DEVICE = DEVICE
        self.const_mult = const_mult
        self.net = FNN(input_size, output_size).to(self.DEVICE)
        param_list = [{'params' : self.net.parameters()}]
        self.optimizer = torch.optim.Adam(param_list, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=0.00001)
        self.loss_fn = torch.nn.MSELoss()
        self.train_loss = []
        if loss == 'L1':
            torch.nn.L1Loss()
        elif loss == 'L2':
            torch.nn.MSELoss()
        else:
            raise ValueError('FNN_MODEL recieved invalid loss type name.')

    # Makw sure amp_y_p is ignored in learning
    def calc_loss(self, y, y_p):
        if self.const_mult: y_p = amplitude_mult(y, y_p)
        return self.loss_fn(y, y_p)

    def backward(self, y, y_p):
        batch_loss = self.calc_loss(y, y_p)
        self.train_loss.append(batch_loss.detach().numpy())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        self.scheduler.step(batch_loss)
        return batch_loss

    def forward(self, X):
        return self.net(X.float())

    def get_train_loss(self):
        return self.train_loss


# Denoising autoencoder, denoises in terms of spectral leakage, 
#  zero approximation, and ?free zeros?
# Check is this a STACKED autoencoder
class SDA(torch.nn.Module):
    def __init__(self, input_size, spectral_encoding_size, 
            temporal_encoding_size, dropout_encoding_size, 
            activation, mode='temporal'):
        super(SDA, self).__init__()
        self.input_size = input_size
        self.spectral_encoding_size = spectral_encoding_size
        self.temporal_encoding_size = temporal_encoding_size
        self.dropout_encoding_size = dropout_encoding_size
        self.mode = mode

        if activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise ValueError('SDA recieved invalid activation function name.')

        self.spectral_encoder = nn.Linear(
            self.input_size, self.spectral_encoding_size)
        self.spectral_decoder = nn.Linear(
            self.spectral_encoding_size, self.input_size)
        self.temporal_encoder = nn.Linear(
            self.spectral_encoding_size, self.temporal_encoding_size)
        self.temporal_decoder = nn.Linear(
            self.temporal_encoding_size, self.spectral_encoding_size)
        self.dropout_encoder = nn.Linear(
            self.temporal_encoding_size, self.dropout_encoding_size)
        self.dropout_decoder = nn.Linear(
            self.dropout_encoding_size, self.temporal_encoding_size)

    def forward(self, x):
        if self.mode=='spectral':
            x = self.spectral_encoder(x)
            x = self.activation(x)
            x = self.spectral_decoder(x)
            x = self.activation(x)
        elif self.mode=='temporal':
            x = self.temporal_encoder(x)
            x = self.activation(x)
            x = self.temporal_decoder(x)
            x = self.activation(x)
        elif self.mode=='dropout':
            x = self.dropout_encoder(x)
            x = self.activation(x)
            x = self.dropout_decoder(x)
            x = self.activation(x)
        return x

    def encode(self, x, mode=None):
        if mode == None: mode = self.mode
        x = self.spectral_encoder(x)
        x = self.activation(x)
        if mode=='spectral': return x
        x = self.temporal_encoder(x)
        x = self.activation(x)
        if mode=='temporal': return x
        x = self.dropout_encoder(x)
        x = self.activation(x)
        if mode=='dropout': return x

    def encode_layer(self, x, modes):
        if 'spectral' in modes:
            x = self.spectral_encoder(x)
            x = self.activation(x)
        if 'temporal' in modes:
            x = self.temporal_encoder(x)
            x = self.activation(x)
        if 'dropout' in modes:
            x = self.dropout_encoder(x)
            x = self.activation(x)
        return x

    def set_mode(self, mode):
        self.mode = mode

    def get_encoding_size(self, mode):
        if mode=='spectral': return self.spectral_encoding_size
        if mode=='temporal': return self.temporal_encoding_size
        if mode=='dropout': return self.dropout_encoding_size


# Add learned loss weighting?
class SDA_Model():
    def __init__(self, device, input_size=None, 
            spectral_encoding_size=None, temporal_encoding_size=None,
            dropout_encoding_size=None, loss_weights=None, 
            sparsity=0, activation='PReLU', path=None, mode='dropout'):
        if path:
            state = torch.load(path)
            self.input_size = state['input_size']
            self.spectral_encoding_size = state['spectral_encoding_size']
            self.temporal_encoding_size = state['temporal_encoding_size']
            self.dropout_encoding_size = state['dropout_encoding_size']
            self.activation = state['activation']
            self.train_loss = state['train_loss']
            self.loss_weights = state['loss_weights']
            self.sparsity = state['sparsity']
        else:
            self.input_size = input_size
            self.spectral_encoding_size = spectral_encoding_size
            self.temporal_encoding_size = temporal_encoding_size
            self.dropout_encoding_size = dropout_encoding_size
            self.activation = activation
            self.train_loss = []
            self.loss_weights = loss_weights
            self.sparsity = sparsity
        self.DEVICE = device
        self.mode = mode
        self.net = SDA(self.input_size, self.spectral_encoding_size, 
            self.temporal_encoding_size, self.dropout_encoding_size, 
            self.activation).to(self.DEVICE)
        param_list = [{'params' : self.net.parameters()}]
        self.optimizer = torch.optim.Adam(param_list, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=0.00001)
        self.loss_fn = torch.nn.MSELoss()
        if path:
            self.net.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def temporal_loss(self, y, y_p):
        return self.loss_fn(y_p, y)

    def spectral_loss(self, y, y_p):
        return self.loss_fn(y_p, y)

    def sparsity_loss(self, hn):
        return self.kl_div(self.sparsity, hn)

    def calc_loss(self, y, y_p):
        loss = self.temporal_loss(y, y_p) * self.loss_weights['temporal']
        if self.loss_weights['spectral'] != 0:
            loss += self.spectral_loss(y, y_p) * self.loss_weights['spectral']
        if self.loss_weights['sparsity'] != 0:
            loss += (
                self.sparsity_loss(self.net.encode_layer(y_p, set(self.mode))) 
                * self.loss_weights['sparsity'])
        return loss

    def backward(self, y, y_p):
        batch_loss = self.calc_loss(y, y_p)
        self.train_loss.append(batch_loss.detach().numpy())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        self.scheduler.step(batch_loss)
        return batch_loss

    def forward(self, X):
        return self.net(X.float())

    def show_encoding(self, y, mode, ax=plt, title='Encoding Activation'):
        encoding = self.net.encode_layer(y, mode)[0,:].repeat(200,1).detach()
        c = ax.imshow(encoding, norm=Normalize(), interpolation='none')
        if ax==plt:
            ax.title(title)
        else:
            ax.set_title(title)
        plt.colorbar(c)

    def get_train_loss(self):
        return self.train_loss

    def encode(self, y, mode=None):
        return self.net.encode(y, mode=mode)

    def encode_layer(self, y, mode):
        return self.net.encode_layer(y, mode)

    def set_mode(self, mode):
        self.mode = mode
        self.net.set_mode(mode)

    def kl_div(self, y, y_p):
        y_p = torch.mean(torch.sigmoid(y_p), 1)
        y = (torch.ones(y_p.shape) * y).to(self.DEVICE)
        return torch.sum(y * torch.log(y/y_p) + (1-y)*torch.log((1-y)/(1-y_p)))

    def save_model(self, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'device': self.DEVICE,
            'train_loss': self.train_loss,
            'loss_weights': self.loss_weights,
            'sparsity': self.sparsity,
            'input_size': self.input_size,
            'activation': self.activation,
            'spectral_encoding_size': self.spectral_encoding_size,
            'temporal_encoding_size': self.temporal_encoding_size,
            'dropout_encoding_size': self.dropout_encoding_size,}
        torch.save(state, path)


# Denoising autoencoder, denoises in terms of spectral leakage, 
#  zero approximation, and ?free zeros?
# Check is this a STACKED autoencoder
class SDA_Conv(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SDA_Conv, self).__init__()
        self.input_size = input_size
        kernel_size1 = round(input_size / 8)
        kernel_size2 = round(input_size / 32)
        kernel_size3 = round(input_size / 128)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size1),
            torch.nn.PReLU(),
            torch.nn.Conv1d(16, 64, kernel_size2),
            torch.nn.PReLU(),
            torch.nn.Conv1d(64, 2, kernel_size3),
            torch.nn.PReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(2, 64, kernel_size3),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose1d(64, 16, kernel_size2),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose1d(16, 1, kernel_size1),
            torch.nn.PReLU(),
        )

    def forward(self, x):
        self.coding = self.encoder(x[:,None,:])
        return self.decoder(self.coding)[:,0,:]


class Transformer_Model():
    def __init__(self, device, input_size=None, output_size=None, 
            num_layers=None, num_heads=None, dim_feedforward=None, 
            batch_first=None, path=None):
        if path:
            state = torch.load(path, map_location=device)
            self.input_size = state['input_size']
            self.output_size = state['output_size']
            self.num_layers = state['num_layers']
            self.num_heads = state['num_heads']
            self.dim_feedforward = state['dim_feedforward']
            self.batch_first = state['batch_first']
            self.train_loss = state['train_loss']
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.dim_feedforward = (
                dim_feedforward if dim_feedforward else 2 * self.input_size)
            self.batch_first = batch_first
            self.train_loss = []
        self.DEVICE = device
        self.net = nn.Transformer(d_model=self.input_size, 
            nhead=self.num_heads, batch_first=self.batch_first,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers, 
            dim_feedforward=self.dim_feedforward).to(self.DEVICE)
        param_list = [{'params' : self.net.parameters()}]
        self.optimizer = torch.optim.Adam(param_list, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=0.00005)
        self.loss_fn = torch.nn.MSELoss()
        if path:
            self.net.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def calc_loss(self, y, y_p):
        return self.loss_fn(y_p, y)

    def backward(self, y, y_p):
        batch_loss = self.calc_loss(y, y_p)
        self.train_loss.append(batch_loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        self.scheduler.step(batch_loss)
        return batch_loss

    def forward(self, input, start):
        return self.net(input.float(), start.float())

    def get_train_loss(self):
        return self.train_loss

    def save_model(self, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'device': self.DEVICE,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dim_feedforward': self.dim_feedforward,
            'batch_first': self.batch_first,
            'train_loss': self.train_loss}
        torch.save(state, path)

    def count_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


# Denoising autoencoder, denoises in terms of spectral leakage, 
#  zero approximation, and ?free zeros?
# Check is this a STACKED autoencoder
class NBeats_Block(torch.nn.Module):
    def __init__(self, input_size, output_size, fnn_size, basis_size, 
            basis, min_freq, max_freq, time_len):
        super(NBeats_Block, self).__init__()
        self.input_size = input_size
        self.input_size = output_size
        self.fnn_size = fnn_size
        self.basis = basis
        self.activation = torch.nn.ReLU()

        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(input_size, fnn_size),
            self.activation,
            torch.nn.Linear(fnn_size, fnn_size),
            self.activation,
            torch.nn.Linear(fnn_size, fnn_size),
            self.activation,
            torch.nn.Linear(fnn_size, fnn_size),
            self.activation)
        self.input_proj = torch.nn.Linear(fnn_size, basis_size)
        self.output_proj = torch.nn.Linear(fnn_size, basis_size)
        if self.basis == 'generic':
            self.input_basis = torch.nn.Linear(basis_size, input_size)
            self.output_basis = torch.nn.Linear(basis_size, output_size)
        elif self.basis == 'fourier':
            freqs = torch.linspace(min_freq, max_freq, basis_size)[None,:]
            times = torch.linspace(0, time_len, input_size)[:,None]
            times = times.type(torch.cfloat)
            PI = torch.acos(torch.zeros(1)).item() * 2
            fund_freq = torch.Tensor([1])
            fund_freq = fund_freq.type(torch.cfloat)
            fund_freq *= -2 * PI * 1j / basis_size
            freqs = torch.mul(fund_freq, freqs)
            self.fourier = torch.mm(times, freqs)
            self.fourier = torch.exp(self.fourier) / basis_size
            #self.fourier = torch.pow(fund_freq,freqs)
            #self.fourier = torch.mm(times
            #    torch.transpose(self.fourier,0,1),self.fourier).detach()
        else:
            raise ValueError('NBeats_Block recieved invalid basis name.')
        self.temp = torch.nn.Parameter(torch.empty(0))

    def forward(self, x):
        x = self.fnn(x)
        y = self.output_proj(x)
        if self.basis == 'generic':
            y = self.output_basis(y)
        elif self.basis == 'fourier':
            y = y.type(torch.cfloat)
            #print(self.fourier.repeat(x.shape[0],1,1).shape)
            #print(self.fourier.shape)
            #print(torch.transpose(x,1,2).shape)
            y = torch.bmm(
                self.fourier.repeat(y.shape[0],1,1), torch.transpose(y,1,2))
            y = torch.real(torch.transpose(y,1,2))
        return y

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def move(self, device):
        if self.basis == 'fourier':
            self.fourier = self.fourier.to(device)
'''
    def parameters(self):
        return [self.fnn.parameters(), self.output_proj.parameters()]
'''


class NBeats_Model():
    def __init__(self, device, input_size=None, output_size=None, 
            fnn_size=None, basis_size=None, block_types=None, 
            min_freq=None, max_freq=None, time_len=None, path=None):
        if path:
            state = torch.load(path, map_location=device)
            self.input_size = state['input_size']
            self.output_size = state['output_size']
            self.fnn_size = state['fnn_size']
            self.basis_size = state['basis_size']
            self.block_types = state['block_types']
            self.min_freq = state['min_freq']
            self.max_freq = state['max_freq']
            self.time_len = state['time_len']
            self.train_loss = state['train_loss']
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.fnn_size = fnn_size
            self.basis_size = basis_size
            self.block_types = block_types
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.time_len = time_len
            self.train_loss = []
        self.DEVICE = device
        params = []
        self.blocks = []
        for basis in self.block_types:
            block = NBeats_Block(self.input_size, self.output_size,
                self.fnn_size, self.basis_size, basis, self.min_freq, 
                self.max_freq, self.time_len).to(self.DEVICE)
            block.move(self.DEVICE)
            params += block.parameters()
            self.blocks.append(block)

        param_list = [{'params' : params}]
        self.optimizer = torch.optim.Adam(param_list, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=0.00005)
        self.loss_fn = torch.nn.MSELoss()
        #self.loss_fn = torch.nn.L1Loss()
        if path:
            for i in range(self.block_types):
                self.blocks[i].load_state_dict(state['net' + str(i)])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def calc_loss(self, y, y_p):
        return self.loss_fn(y_p, y)

    def backward(self, y, y_p):
        batch_loss = self.calc_loss(y, y_p)
        self.train_loss.append(batch_loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        self.scheduler.step(batch_loss)
        return batch_loss

    def forward(self, x):
        x = x.float()
        y_total = self.blocks[0](x)
        for block in self.blocks[1:]:
            y_total += block(x)
        return y_total

    def get_train_loss(self):
        return self.train_loss

    def save_model(self, path):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'device': self.DEVICE,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'fnn_size': self.fnn_size,
            'basis_size': self.basis_size,
            'block_types':self.block_types,
            'min_freq':self.min_freq,
            'max_freq':self.max_freq,
            'time_len':self.time_len,
            'train_loss': self.train_loss}
        for i in range(len(self.block_types)):
            state['net' + str(i)] = self.blocks[i].state_dict()
        torch.save(state, path)

    def count_params(self):
        count = 0
        for block in self.blocks:
            count += block.count_params()
        return count










class NBeats_Model2():
    def __init__(self, device, input_size=None, output_size=None, 
            fnn_size=None, basis_size=None, block_types=None, 
            min_freq=None, time_len=None, path=None):
        if path:
            state = torch.load(path, map_location=device)
            self.input_size = state['input_size']
            self.output_size = state['output_size']
            self.fnn_size = state['fnn_size']
            self.basis_size = state['basis_size']
            self.block_types = state['block_types']
            self.min_freq = state['min_freq']
            self.time_len = state['time_len']
            self.train_loss = state['train_loss']
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.fnn_size = fnn_size
            self.basis_size = basis_size
            self.block_types = block_types
            self.min_freq = min_freq
            self.time_len = time_len
            self.train_loss = []
        self.DEVICE = device
        self.net = NBeats(self.input_size, self.output_size, 
            self.fnn_size, self.basis_size, self.block_types, 
            self.min_freq, self.time_len).to(self.DEVICE)
        self.net.move(self.DEVICE)
        param_list = [{'params' : self.net.parameters()}]
        self.optimizer = torch.optim.Adam(param_list, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=0.00005)
        self.loss_fn = torch.nn.MSELoss()
        #self.loss_fn = torch.nn.L1Loss()
        if path:
            self.net.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def calc_loss(self, y, y_p):
        return self.loss_fn(y_p, y)

    def backward(self, y, y_p):
        batch_loss = self.calc_loss(y, y_p)
        self.train_loss.append(batch_loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        self.scheduler.step(batch_loss)
        return batch_loss

    def forward(self, input):
        return self.net(input.float())

    def get_train_loss(self):
        return self.train_loss

    def save_model(self, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'device': self.DEVICE,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'fnn_size': self.fnn_size,
            'basis_size': self.basis_size,
            'block_types':self.block_types,
            'min_freq':self.min_freq,
            'time_len':self.time_len,
            'train_loss': self.train_loss}
        torch.save(state, path)

    def count_params(self):
        return self.net.count_params()


class NBeats(torch.nn.Module):
    def __init__(
            self, input_size, output_size, fnn_size, basis_size, 
            block_types, min_freq, max_freq, time_len):
        super(NBeats, self).__init__()
        self.blocks = []
        self.block_types = block_types
        for basis in self.block_types:
            self.blocks.append(NBeats_Block(input_size, output_size, fnn_size, 
                basis_size, basis, min_freq, max_freq, time_len))

    def forward(self, x):
        x_prev = x
        x, y_total = self.blocks[0](x)
        for block in self.blocks[1:]:
            x = torch.sub(x_prev, x)
            x_prev = x
            x, y = block(x)
            y_total += y
        return y_total

    def count_params(self):
        count = 0
        for block in self.blocks:
            count += sum(
                p.numel() for p in block.parameters() if p.requires_grad)
        return count

    def move(self, device):
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].to(device)
            self.blocks[i].move(device)
