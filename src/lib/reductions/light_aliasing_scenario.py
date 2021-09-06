import numpy as np
import matplotlib.pyplot as plt
from ..tools.utils import compute_fourier_transform, gaussian

from scipy.fftpack import fft,fft2, fftshift, ifft, ifft2


def no_frequency_cutoff_subsampling(u,visualization=False):
    #print('light aliasing scenario !')
    u_fourier = compute_fourier_transform(u)
    nx, ny = u.shape
    u_fourier_cut_off = fftshift(fft2(u))
    fcutoff = 80 #
    k = gaussian(fcutoff)
    # Multiply the frequency with the gaussian kernel
    u_fourier_cut_off_Blur = np.copy(u_fourier_cut_off)
    u_fourier_cut_off_Blur[nx//2-fcutoff:nx//2+fcutoff, ny//2-fcutoff:ny//2+fcutoff] = u_fourier_cut_off[nx//2-fcutoff:nx//2+fcutoff, ny//2-fcutoff:ny//2+fcutoff]*k
    # Downsample the fourier domain by a factor 2 (crop the center)
    u_fourier_cut_off_Blur_crop = u_fourier_cut_off_Blur[nx//2-nx//4:nx//2+nx//4,ny//2-ny//4:ny//2+ny//4]
    u_reduction = np.abs(ifft2(u_fourier_cut_off_Blur_crop))

    u_reduction_fourier = compute_fourier_transform(u_reduction)
    if visualization:
        plt.figure(figsize = (10,10))
        fig, ax = plt.subplots(2,2, figsize = (20,20))

        ax[0,0].imshow(u)
        ax[0,0].set_title('Periodic Image')

        ax[0,1].imshow(u_fourier)
        ax[0,1].set_title('Fourier transform of periodic image')

        ax[1,0].imshow(u_reduction)
        ax[1,0].set_title('Reduction image')

        ax[1,1].imshow(u_reduction_fourier)
        ax[1,1].set_title('Fourier transform of reduction image')
    return u_reduction