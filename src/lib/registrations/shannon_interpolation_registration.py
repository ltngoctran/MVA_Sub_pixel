import numpy as np
import matplotlib.pyplot as plt
from ..tools.utils import fourier_zoom
from ..tools.utils import image_cross_correlation, image_phase_cross_correlation
import skimage


def subpixel_image_registration(reference_image, moving_image,upsample_factor=4):
    w = image_cross_correlation(reference_image,moving_image)
    W = fourier_zoom(w,upsample_factor)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in W.shape])
    # maximum pixel
    I= np.unravel_index(np.argmax(W, axis=None), W.shape)
    X = np.array([[1,1,1,-1,-1,1],
                [1,0,0,-1,0,1],
                [1,-1,1,-1,1,1],
                [0,0,1, 0,-1,1],
                [0,0,0,0,0,1],
                [0,0,1,0,1,1],
                [1,-1,1,1,-1,1],
                [1,0,0,1,0,1],
                [1,1,1,1,1,1] ])
    Y = np.array([W[I[0]+i,I[1]+j] for j in [-1,0,1] for i in [-1,0,1] ]).reshape(-1,1)
    alpha = np.linalg.solve(np.matmul(X.T,X),np.matmul(X.T,Y)).flatten()
    A = np.array( [[2*alpha[0],alpha[1]],[alpha[1],2*alpha[2]] ])
    Ay = np.array([-alpha[3],-alpha[4]])
    dz = np.linalg.solve(A,Ay)
    peak_position = I + dz[::-1]
    dt = (peak_position-midpoints)/upsample_factor
    return -dt[::-1]

    


def reference_subpixel_image_registration(image,offset_image):
    shift, error, diffphase = skimage.registration.phase_cross_correlation(image, offset_image,upsample_factor=10)
    return -shift[::-1]
