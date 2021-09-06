
import numpy as np
import matplotlib.pyplot as plt
import os
from ..config import data_dir, results_dir, fit_dir
from ..lib.tools.utils import perdecomp, check_create_dir
from ..lib.reductions.strong_aliasing_scenario import direct_subsampling
from ..lib.reductions.light_aliasing_scenario import no_frequency_cutoff_subsampling
from ..lib.reductions.no_aliasing_scenario import hard_frequency_cutoff_subsampling
from ..lib.registrations.shannon_interpolation_registration import subpixel_image_registration
from ..lib.tools.utils import fourier_shift, translation_fourier_Shannon_interpolation
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
#matplotlib.use('TkAgg')

plt.gray()

data_directory_contents = os. listdir(data_dir)
print(data_directory_contents)
for filename in data_directory_contents:
    name = filename.split('.')[0]
    print('Test case for ',name)

    u = plt.imread(data_dir+filename)
    print(u.shape)
    s0, u0 = perdecomp(u)

    # fig, axs = plt.subplots(1,3, figsize = (20,7))
    # fig.suptitle('periodic components')
    # axs[0].imshow(u0)
    # axs[1].imshow(s0)
    # axs[2].imshow(u)
    # plt.show()
    

    z = 2 # zoom factor = 2 for this project

    n_param = 100
    samples_t = 0.5*np.random.random((n_param,2))
    print('samples_t',samples_t)

    total_t_hat =[]
    total_t_hat_ts =[]
    total_t_hat_ls =[]
    total_t_hat_ns =[]

    error_ts = []
    error_ls = []
    error_ns = []

    for i in range(n_param):

        t = samples_t[i]
        u1 = fourier_shift(u0,t*z)


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
        sigma = 0.01
        w0_ts = v0_ts + np.random.normal(loc=0,scale=sigma,size=v0_ts.shape)
        w0_ls = v0_ls + np.random.normal(loc=0,scale=sigma,size=v0_ls.shape)
        w0_ns = v0_ns + np.random.normal(loc=0,scale=sigma,size=v0_ns.shape)

        w1_ts = v1_ts + np.random.normal(loc=0,scale=sigma,size=v1_ts.shape)
        w1_ls = v1_ls + np.random.normal(loc=0,scale=sigma,size=v1_ls.shape)
        w1_ns = v1_ns + np.random.normal(loc=0,scale=sigma,size=v1_ns.shape)

        # smoothing
        # w0_ts = gaussian_filter(w0_ts, sigma=10)
        # w1_ts = gaussian_filter(w1_ts, sigma=10)

        t_hat    = subpixel_image_registration(u0,u1)
        t_hat_ts = subpixel_image_registration(w0_ts,w1_ts)
        t_hat_ls = subpixel_image_registration(w0_ls,w1_ls)
        t_hat_ns = subpixel_image_registration(w0_ns,w1_ns)

        total_t_hat.append(t_hat/z)
        total_t_hat_ts.append(t_hat_ts)
        total_t_hat_ls.append(t_hat_ls)
        total_t_hat_ns.append(t_hat_ns)
        # store error
        error_ts.append(np.linalg.norm(t_hat_ts-t)**2)
        error_ls.append(np.linalg.norm(t_hat_ls-t)**2)
        error_ns.append(np.linalg.norm(t_hat_ns-t)**2)

    print('original',np.vstack(total_t_hat))
    print('strong scenario ',np.vstack(total_t_hat_ts))
    print('light  scenario',np.vstack(total_t_hat_ls))
    print('no scenario',np.vstack(total_t_hat_ns))

    save_fit_dir = check_create_dir(fit_dir+'{}/'.format(name))
    pd.DataFrame(error_ts, columns=['error']).to_csv(save_fit_dir+'ts_error.csv')
    pd.DataFrame(error_ls, columns=['error']).to_csv(save_fit_dir+'ls_error.csv')
    pd.DataFrame(error_ns, columns=['error']).to_csv(save_fit_dir+'ns_error.csv')

    pd.DataFrame(samples_t , columns=['tx','ty']).to_csv(save_fit_dir+'true_T.csv')
    pd.DataFrame(np.vstack(total_t_hat_ts), columns=['tx','ty']).to_csv(save_fit_dir+'ts_T.csv')
    pd.DataFrame(np.vstack(total_t_hat_ls), columns=['tx','ty']).to_csv(save_fit_dir+'ls_T.csv')
    pd.DataFrame(np.vstack(total_t_hat_ns), columns=['tx','ty']).to_csv(save_fit_dir+'ns_T.csv')




    save_result_dir = check_create_dir(results_dir+'{}/'.format(name))
    plt.figure()
    plt.plot(np.arange(n_param),error_ts,'r-*')
    plt.plot(np.arange(n_param),error_ls,'b-o')
    plt.plot(np.arange(n_param),error_ns,'g-h')
    plt.legend(['strong scenario','light scenario','no scenario'])
    plt.title('Error curves')
    #plt.show()
    plt.savefig(save_result_dir +'error_shannon_registration_method.png')
    plt.show()