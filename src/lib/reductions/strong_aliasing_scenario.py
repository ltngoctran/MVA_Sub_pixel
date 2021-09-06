import numpy as np
import matplotlib.pyplot as plt
from ..tools.utils import compute_fourier_transform


def direct_subsampling(u,visualization=False):
    #print('Direct subsampling !')
    u_fourier = compute_fourier_transform(u)
    u_reduction = u[::2,::2]   
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
