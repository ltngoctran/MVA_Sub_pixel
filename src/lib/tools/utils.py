import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.fftpack import fft,fft2, fftshift, ifft, ifft2,ifftshift
import  math
import copy
from scipy import ndimage



def get_colors(n):
    cmap = plt.get_cmap('jet')
    return cmap(np.linspace(0, 1.0, n))

def check_create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def fourier_shift(u, t):
    ny, nx = u.shape
    mx = np.floor(nx/2);
    my = np.floor(ny/2);
    Tx = np.exp(-2*1.j*np.pi*t[0]/nx*(np.arange(mx,mx+nx)%nx-mx))[None]
    Ty = np.exp(-2*1.j*np.pi*t[1]/ny*(np.arange(my,my+ny)%ny-my))[None]

    v = np.real(ifft2(fft2(u)*(Ty.T*Tx)))
    return v

def translation_bilinear_interpolation(u,t):
    nx,ny = u.shape
    tu = np.zeros((nx+2,ny+2))
    tu[1:-1,1:-1]= u
    # periodic
    tu[0,:] = tu[nx,:]
    tu[nx+1,:] = tu[1,:]
    tu[:,0] = tu[:,ny]
    tu[:,ny+1] = tu[:,1]
    F = np.zeros((nx,ny))
    #F = (1-t[0])*(1-t[1])*tu[1:-1,1:-1]+t[0]*(1-t[1])*tu[1:-1,2:]+(1-t[0])*t[1]*tu[2:,1:-1]+ t[0]*t[1]*tu[2:,2:]
    F = t[0]*t[1]*tu[0:-2,0:-2]+t[0]*(1-t[1])*tu[1:-1,0:-2]+(1-t[0])*t[1]*tu[0:-2,1:-1]+ (1-t[0])*(1-t[1])*tu[1:-1,1:-1]

    return F

def perdecomp(u):
    nx, ny = u.shape
    v = np.zeros_like(u)
    v[0,:]  = u[0,:]  - u[-1,:]
    v[-1,:] = -v[0,:]
    v[:,0]  = v[:,0]  + u[:,0] - u[:,-1]
    v[:,-1] = v[:,-1] - u[:,0] + u[:,-1]
    fy = np.cos(2.*np.pi*(np.arange(1,ny+1) -1)/ny)[:,None]
    fy = np.repeat(fy, nx, axis=1)

    fx = np.cos(2.*np.pi*(np.arange(1,nx+1) -1)/nx)[None]
    fx = np.repeat(fx, ny, axis=0)
    
    fy[0,0] = 0.
    s = np.real(ifft2(fft2(v) *0.5/(2-fx-fy)))
    p = u -s
    return s, p
def compute_fourier_transform(u):
    u_fourier = np.log(0.0001+np.abs(fftshift(fft2(u))))
    return u_fourier
def gaussian(fcutoff):
    L = np.arange(0, fcutoff)
    kern1d = np.exp(-(L/(fcutoff-1))**2/2)
    kern1d = np.concatenate((kern1d[::-1], kern1d))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d

def sincd_func(x,M,flag=True):
    
    if flag:
        if x%M ==0 or x ==0 :
            sincd = 1
        elif M%2 == 0:
            sincd = np.sin(math.pi*x)/(M*np.tan(math.pi*x/M))
        else:
            sincd = np.sin(math.pi*x)/(M*np.sin(math.pi*x/M))
    else:
        sincd = np.zeros(len(x))
        Mv = M*np.ones(len(x))
        for i in range(len(x)):
            if x[i]%Mv[i] ==0 or x[i] ==0 :
                sincd[i] = 1
            elif Mv[i]%2 == 0:
                sincd[i] = np.sin(math.pi*x[i])/(Mv[i]*np.tan(math.pi*x[i]/Mv[i]))
            else:
                sincd[i] = np.sin(math.pi*x[i])/(Mv[i]*np.sin(math.pi*x[i]/Mv[i]))
    return sincd

def translation_sincd_Shannon_interpolation(u,t):
    uf = copy.deepcopy(u)
    ny,nx = u.shape
    F = np.zeros((ny,nx))
    for j in range(ny):
        for i in range(nx):
            x = i - t[0]
            y = j - t[1]
            vx = [x-k for k in range(nx)]
            vy = [y-l for l in range(ny)]
            vsincx = sincd_func(vx,nx,False)
            vsincy = sincd_func(vy,ny,False)
            msinc = np.tensordot(vsincy,vsincx,axes = 0)
            F[j,i] = np.sum(uf*msinc)
    return F

def translation_fourier_Shannon_interpolation(u,t):
    ny,nx = u.shape
    mx = np.floor(nx/2)
    my = np.floor(ny/2)

    Ix = (np.arange(mx,mx+nx)%nx-mx)
    Iy = (np.arange(my,my+ny)%ny-my)
    u_fourier = fft2(u)
    F = np.zeros((ny,nx))
    for j in range(ny):
        for i in range(nx):
            x = i - t[0]
            y = j - t[1]
            Tx = np.exp(2*1.j*np.pi*x/nx*Ix)[None]
            Ty = np.exp(2*1.j*np.pi*y/ny*Iy)[None]
            F[j,i]= np.real( 1/(nx*ny)*np.sum(u_fourier*(Ty.T*Tx)) )
    return F
def zoom_fourier_Shannon_interpolation(u,z):
    ny,nx = u.shape
    mx = np.floor(nx/2)
    my = np.floor(ny/2)
    znx = int(z*nx)
    zny = int(z*ny)

    Ix = (np.arange(mx,mx+nx)%nx-mx)
    Iy = (np.arange(my,my+ny)%ny-my)
    u_fourier = fft2(u)
    F = np.zeros((zny,znx))
    for j in range(zny):
        for i in range(znx):
            x = i/z
            y = j/z
            Tx = np.exp(2*1.j*np.pi*x/nx*Ix)[None]
            Ty = np.exp(2*1.j*np.pi*y/ny*Iy)[None]
            F[j,i]= np.real( 1/(nx*ny)*np.sum(u_fourier*(Ty.T*Tx)) )
    return F


def zoom_sincd_Shannon_interpolation(u,z):
    uf = copy.deepcopy(u)
    ny,nx = u.shape
    znx = int(z*nx)
    zny = int(z*ny)
    F = np.zeros((zny,znx))
    for j in range(zny):
        for i in range(znx):
            x = i/z
            y = j/z
            vx = [x-k for k in range(nx)]
            vy = [y-l for l in range(ny)]
            vsincx = sincd_func(vx,nx,flag=False)
            vsincy = sincd_func(vy,ny,flag=False)
            msinc = np.tensordot(vsincy,vsincx,axes = 0)
            F[j,i] = np.sum(uf*msinc)
    return F

def fourier_zoom(u,z=None):
    if z==None:
        z=2
    ny,nx = u.shape
    mx = int(np.floor(z*nx))
    my = int(np.floor(z*ny))
    dx = int(np.floor(nx/2)-np.floor(mx/2))
    dy = int(np.floor(ny/2)-np.floor(my/2))

    if z>=1 :
        v = np.zeros((my,mx),dtype = 'complex_')
        v[-dy:-dy+ny,-dx:-dx+nx] = fftshift(fft2(u))
    else:
        f = fftshift(fft2(u))
        v = f[dy:dy+my,dx:dx+mx]
        if mx%2 ==0:
            v[:,0]=0
        if my%2 ==0:
            v[0,:]=0
    
    return z*z*np.real(ifft2(ifftshift(v)))

def image_fourier_shift(image,t):
    shift = (t[1], t[0])
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image = ndimage.fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    return np.real(offset_image)

def image_phase_cross_correlation(image,offset_image):
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    image_product_normalization = image_product/np.sum(np.abs(image_product))
    phase_cc_image = np.fft.fftshift(np.fft.ifft2(image_product_normalization))
    return np.real(phase_cc_image)

def image_cross_correlation(image,offset_image):
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    return np.real(cc_image)