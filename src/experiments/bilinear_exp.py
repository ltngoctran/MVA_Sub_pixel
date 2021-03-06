
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from ..config import data_dir, results_dir
from ..lib.tools.utils import perdecomp
from ..lib.reductions.strong_aliasing_scenario import direct_subsampling
from ..lib.reductions.light_aliasing_scenario import no_frequency_cutoff_subsampling
from ..lib.reductions.no_aliasing_scenario import hard_frequency_cutoff_subsampling
from ..lib.registrations.bilinear_interpolation_registration import optim_bilinear_registration
from ..lib.tools.utils import translation_bilinear_interpolation, fourier_shift, translation_fourier_Shannon_interpolation


plt.gray()

u = plt.imread(data_dir+'barbara.png')
s0, u0 = perdecomp(u)

# fig, ax = plt.subplots(1,3, figsize = (20,7))
# fig.suptitle('periodic components')
# ax[0].imshow(u0)
# ax[1].imshow(s0)
# ax[2].imshow(u)
# plt.show()

z = 2 # zoom factor = 2 for this project

n_param = 100
# samples_t = 0.5*np.random.random((n_param,2))
# print('samples_t',samples_t)
# print(type(samples_t))
# print('-------------------------')

# pd.DataFrame(samples_t).to_csv("file.csv")

samples_t = pd.read_csv('file.csv')

samples_t = samples_t.drop(columns=['id']).values
print('samples_t2',samples_t)
# print(type(samples_t2))



total_t_hat =[]
total_t_hat_ts =[]
total_t_hat_ls =[]
total_t_hat_ns =[]

error_ts = []
error_ls = []
error_ns = []

for i in range(n_param):

    t = samples_t[i]

    #u1 = translation_bilinear_interpolation(u0,t*z)
    u1 = fourier_shift(u0,t*z)
    #u1 = translation_Shannon_interpolation(u0,t*z)

    # fig, ax = plt.subplots(1,2, figsize = (20,10))
    # ax[0].imshow(u0)
    # ax[1].imshow(u1)
    # plt.show()

    v0_ts = direct_subsampling(u0)
    v0_ls = no_frequency_cutoff_subsampling(u0)
    v0_ns = hard_frequency_cutoff_subsampling(u0)

    v1_ts = direct_subsampling(u1)
    v1_ls = no_frequency_cutoff_subsampling(u1)
    v1_ns = hard_frequency_cutoff_subsampling(u1)

    # add noise  to v0 and v1
    sigma = 0
    w0_ts = v0_ts + np.random.normal(loc=0,scale=sigma,size=v0_ts.shape)
    w0_ls = v0_ls + np.random.normal(loc=0,scale=sigma,size=v0_ls.shape)
    w0_ns = v0_ns + np.random.normal(loc=0,scale=sigma,size=v0_ns.shape)

    w1_ts = v1_ts + np.random.normal(loc=0,scale=sigma,size=v1_ts.shape)
    w1_ls = v1_ls + np.random.normal(loc=0,scale=sigma,size=v1_ls.shape)
    w1_ns = v1_ns + np.random.normal(loc=0,scale=sigma,size=v1_ns.shape)



    t_hat    = optim_bilinear_registration(u0,u1)
    t_hat_ts = optim_bilinear_registration(w0_ts,w1_ts)
    t_hat_ls = optim_bilinear_registration(w0_ls,w1_ls)
    t_hat_ns = optim_bilinear_registration(w0_ns,w1_ns)

    total_t_hat.append(t_hat/z)
    total_t_hat_ts.append(t_hat_ts)
    total_t_hat_ls.append(t_hat_ls)
    total_t_hat_ns.append(t_hat_ns)
    # store error
    error_ts.append(np.linalg.norm(t_hat_ts-t)**2)
    error_ls.append(np.linalg.norm(t_hat_ls-t)**2)
    error_ns.append(np.linalg.norm(t_hat_ns-t)**2)

# print('original',np.vstack(total_t_hat))
# print('strong scenario ',np.vstack(total_t_hat_ts))
# print('light  scenario',np.vstack(total_t_hat_ls))
# print('no scenario',np.vstack(total_t_hat_ns))


plt.figure()
plt.plot(np.arange(n_param),error_ts,'r-*')
plt.plot(np.arange(n_param),error_ls,'b-o')
plt.plot(np.arange(n_param),error_ns,'g-h')
plt.legend(['strong scenario','light scenario','no scenario'])
plt.title('Error curves')
plt.show(block=False)
plt.savefig(results_dir+'error_bilinear_registration_method.png')


plt.figure()
plt.plot(np.arange(n_param),error_ls,'b-o')
plt.plot(np.arange(n_param),error_ns,'g-h')
plt.legend(['light scenario','no scenario'])
plt.title('Error curves')
plt.show(block=False)
plt.savefig(results_dir+'error_bilinear_2cur_registration_method.png')
