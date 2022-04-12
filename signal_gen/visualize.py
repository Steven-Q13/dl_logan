import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from scipy.spatial import distance
from scipy.fft import fft, fftfreq, fftshift, rfft

from signal_gen import *
from models import *

# Plot basic sinc function and its transform
def plot_sinc_transform(x, sig_power, highpass, lowpass, N):
    sinc = sinc_func(x, sig_power, (highpass + lowpass)/2, highpass-lowpass)
    sinc_f = fft(sinc,N)

    plt.plot(x, sinc)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title('sinc Function')
    plt.show()

    plt.plot(xf, fftshift(np.abs(sinc_f)))
    plt.xlabel('$\u03BE$', fontsize=13)
    plt.ylabel('$f(\u03BE )$', fontsize=13)
    plt.title('sinc Function Fourier Transform')
    plt.show()

def plot_func(x, y, xlabel='', ylabel='', title='', xlim=None, show_pnts=False):
    if show_pnts:
        plt.plot(x, y, linestyle='-', marker='o', markersize=3)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title)
    if xlim != None: plt.xlim(xlim)




if __name__ == "__main__":
    np.random.seed(13)
    np.set_printoptions(threshold=np.inf)
    torch.set_printoptions(threshold=10_000)

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    # Params 
    OCTAVE_MULT = 10 ** 0.3
    MINI_OCTAVE_MULT = 1.3
    CACHE_SIZE = 64
    BATCH_SIZE = 64
    sig_power = 1
    num_funcs = 1 
    lowpass = 4
    highpass = lowpass * MINI_OCTAVE_MULT
    sample_rate = round(highpass * 10)
    min_time = -10
    max_time = 10
    kernel = rbf_kernel(1)
    kernel = sinc_kernel(sig_power, (highpass + lowpass)/2, highpass-lowpass) 
    num_samples = sample_rate * (max_time - min_time)
    N = round(num_samples * 0.55)

    # Discrete evaluation of the mean and covariance functions
    x = np.linspace(min_time, max_time-1/sample_rate, num=num_samples)
    cov_func = kernel(x, x)
    mean_func = np.zeros(num_samples)
    freqs = np.random.multivariate_normal(mean=mean_func, cov=cov_func, size=num_funcs)

    # Discrete Fourier Transform of Gaussian Process
    yt = freqs[0]
    yf = fft(yt,N)
    xf = (np.linspace(0,N-1,N) - N/2) * sample_rate / N

    # Plot the sampled function
    # plt.figure(figsize=(6, 4))
    for i in range(num_funcs):
        plot_func(x, freqs[i], xlabel='t', ylabel='y = f(t)', 
                  title='Gaussian Process with sinc Kernel', 
                  xlim=[min_time, max_time], show_pnts=True)
    plt.show()

    # Plot Fourier Transform of Gaussian process
    plot_func(xf, fftshift(np.abs(yf)), xlabel='$\u03BE$', 
              ylabel='$f(\u03BE )$', title='Gaussian Process Fourier Transform')
    plt.show()
    plot_sinc_transform(x, sig_power, highpass, lowpass, N)

    # Setup data loaders
    stdout = './' + time.strftime('%Y_%m_%d_%H%M%S', time.gmtime()) + '.txt'
    train = AperiodicTrain(
        min_time, max_time, lowpass, highpass, kernel, CACHE_SIZE)
    valid = AperiodicTrain(
        min_time, max_time, lowpass, highpass, kernel, CACHE_SIZE)
    test = AperiodicTrain(
        min_time, max_time, lowpass, highpass, kernel, CACHE_SIZE)

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)
    model = FNN_Model(DEVICE, train.input_size(), train.output_size())


    for batch_idx, (X,y) in enumerate(train_loader):
        sample_rate = train.sample_rate()
        N = round(train.output_size() * 0.55)
        xf = (np.linspace(0,N-1,N) - N/2) * sample_rate / N

        y_p = model.forward(X)
        print(torch.transpose(y_p,0,1).shape)
        print(torch.transpose(y,0,1).shape)
        amp = torch.linalg.lstsq(
            torch.transpose(y_p,0,1), torch.transpose(y,0,1)).solution
        y_p = torch.mm(torch.transpose(amp,0,1).detach(), y_p).detach().numpy()
        y = y.detach().numpy()
        y_pf = fft(y_p[0],N)
        yf = fft(y[0],N)
        print(y[0].shape)

        fig, (ax1, ax2) = plt.subplots(2, 2)
        ax1[0].plot(train.time_dim(), y[0])
        ax1[0].set_xlabel('time - t')
        ax1[0].set_ylabel('y')
        ax1[0].set_xlim([min_time, max_time])
        ax1[0].set_title('Original')
        ax2[0].plot(train.time_dim(), y_p[0])
        ax2[0].set_xlabel('time - t')
        ax2[0].set_ylabel('y')
        ax2[0].set_title('Predicted')
        ax2[0].set_xlim([min_time, max_time])

        ax1[1].plot(xf, fftshift(np.abs(yf)))
        ax1[1].set_xlabel('freq - $\u03BE$')
        ax1[1].set_ylabel('$f(\u03BE)$')
        ax1[1].set_title('Original ')
        ax2[1].plot(xf, fftshift(np.abs(y_pf)))
        ax2[1].set_xlabel('freq - $\u03BE$')
        ax2[1].set_ylabel('$f(\u03BE)$')
        ax2[1].set_title('Predicted')
        fig.set_size_inches(13, 6)
        plt.show()

        total_mse_loss = np.sum(np.square(y))
        print("MSE Loss: " + str(
            total_mse_loss.item()/(train.output_size() * BATCH_SIZE)))
        print("Max Zeros: " + str(train.max_zeros))
        break


def plot_funcs(x1, x2, y1, y2, ht=13, wd=6, xlabel1='', xlabel2='', 
        ylabel1='', ylabel2='', title1='', title2='', 
        xlim1=None, xlim2=None, show_pnts1=False, show_pnts2=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if show_pnts1:
        ax1.plot(x1, y1, linestyle='-', marker='o', markersize=3)
    else:
        ax2.plot(x1, y1)
    if show_pnts2:
        ax2.plot(x2, y2, linestyle='-', marker='o', markersize=3)
    else:
        ax2.plot(x2, y2)
    
    ax1.set_xlabel(xlabel1, fontsize=13)
    ax1.set_ylabel(ylabel1, fontsize=13)
    ax1.set_title(title1)
    if xlim1 != None: ax1.set_xlim(xlim1)

    ax2.set_xlabel(xlabel2, fontsize=13)
    ax2.set_ylabel(ylabel2, fontsize=13)
    ax2.set_title(title2)
    if xlim2 != None: ax2.set_xlim(xlim2)
    fig.set_size_inches(13, 6)


def plot_loss(model, model2=None, model3=None, skip_beg=0, legend=['Loss']):
    x = np.linspace(1, len(model.get_train_loss())-skip_beg-1, 
        num=len(model.get_train_loss())-skip_beg-1)
    plt.plot(x, model.get_train_loss()[skip_beg+1:])
    if model2:
        plt.plot(x, model2.get_train_loss()[skip_beg+1:])
    if model3:
        plt.plot(x, model3.get_train_loss()[skip_beg+1:])
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('Training/Validation Loss')
    plt.legend(legend)
