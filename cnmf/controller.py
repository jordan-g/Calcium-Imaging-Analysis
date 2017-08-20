from cnmf.param_window import ParamWindow
from cnmf.preview_window import PreviewWindow
import cnmf.utilities as utilities
import time
import json
import os
import sys
import glob
import scipy.ndimage as ndi
import scipy.signal
import numpy as np
from skimage.external.tifffile import imread

import pdb

import numpy as np
import glob
import os
import scipy
from ipyparallel import Client
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%%

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.utilities import extract_DF_F
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar

# import the Qt library
try:
    from PyQt4.QtCore import pyqtSignal, Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 4
except:
    from PyQt5.QtCore import pyqtSignal, Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 5

DEFAULT_PARAMS = {}

PARAMS_FILENAME = "params.txt"

class Controller():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(PARAMS_FILENAME):
            try:
                self.params = DEFAULT_PARAMS
                params = json.load(open(PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_PARAMS
        else:
            self.params = DEFAULT_PARAMS

        self.image = None

        # create parameters window
        self.param_window = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

    def open_image(self, image_path):
        base_name = os.path.basename(image_path)
        if base_name.endswith('.npy'):
            self.video = np.transpose(np.load(image_path), (1, 2, 0))
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = np.transpose(imread(image_path), (1, 2, 0))

        self.video_path = image_path

        self.image = utilities.normalize(utilities.mean(self.video)).astype(np.float32)

        self.param_window.show()
        self.preview_window.show()

        self.preview_window.plot_image(self.image)
        # self.preview_window.zoom(2)

    def show_image(self):
        if self.image is not None:
            self.preview_window.plot_image(self.image)

    def refine_watershed_components(self):
        c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = None,single_thread = False)
        init_method = 'greedy_roi'
        alpha_snmf=None #10e2  # this controls sparsity

        fnames = [self.video_path]

        add_to_movie=300 # the movie must be positive!!!
        downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
        idx_xy=None
        base_name='Yr'
        name_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy,add_to_movie=add_to_movie)
        name_new.sort()
        print(name_new)
        #%%
        if len(name_new)>1:
            fname_new = cm.save_memmap_join(name_new, base_name='Yr', n_chunks=12, dview=dview)
        else:
            print('One file only, not saving!')
            fname_new = name_new[0]
        #%%
        # fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr, dims + (T,), order='F')

        Cn = cm.local_correlations(Y)

        #%%
        if np.min(images)<0:
            raise Exception('Movie too negative, add_to_movie should be larger')
        if np.sum(np.isnan(images))>0:
            raise Exception('Movie contains nan! You did not remove enough borders')   
        #%%
        Cn = cm.local_correlations(Y[:,:,:3000])
        pl.imshow(Cn,cmap='gray')

        #%%
        rf = 14  # half-size of the patches in pixels. rf=25, patches are 50x50
        stride = 6  # amounpl.it of overlap between the patches in pixels
        K = 6  # number of neurons expected per patch
        gSig = [6, 6]  # expected half size of neurons
        merge_thresh = 0.8  # merging threshold, max correlation allowed
        p = 1  # order of the autoregressive system
        save_results = False

        cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1,method_deconvolution='oasis')

        #Todo : to compartiment
        T = images.shape[0]
        dims = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        print((T,) + dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        Y.filename = images.filename
        Yr.filename = images.filename

        options = CNMFSetParms(Y, n_processes, p=0, gSig=gSig, K=K, ssub=2, tsub=2,
                               p_ssub=1, p_tsub=1, method_init='greedy_roi',
                               n_pixels_per_process=4000, block_size=20000,
                               check_nan=True, nb=1, normalize_init = True,
                               options_local_NMF = None,
                               remove_very_bad_comps = False)

        print('preprocessing ...')
        Yr, sn, g, psx = preprocess_data(Yr, dview=dview, **options['preprocess_params'])

        if stride is None:
            stride = np.int(rf * 2 * .1)
            print(('**** Setting the stride to 10% of 2*rf automatically:' + str(stride)))

        if type(images) is np.ndarray:
            raise Exception(
                'You need to provide a memory mapped file as input if you use patches!!')

        if only_init:
            options['patch_params']['only_init'] = True

        if alpha_snmf is not None:
            options['init_params']['alpha_snmf'] = alpha_snmf

        print('update spatial ...')
        A, b, Cin, f_in = update_spatial_components(Yr, Cin, f_in, Ain,
                                                         sn=sn, dview=dview, **options['spatial_params'])

        print('update temporal ...')
        # set this to zero for fast updating without deconvolution
        options['temporal_params']['p'] = 0

        print('deconvolution ...')
        options['temporal_params']['method'] = 'oasis'

        C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(
            Yr, A, b, Cin, f_in, dview=dview, **options['temporal_params'])

        print('refinement...')
        print('merge components ...')
        A, C, nr, merged_ROIs, S, bl, c1, sn1, g1 = merge_components(
            Yr, A, b, C, f, S, sn, options['temporal_params'], options['spatial_params'],
            dview=dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8,
            mx=50, fast_merge=True)
        print((A.shape))
        print('update spatial ...')
        A, b, C, f = update_spatial_components(
            Yr, C, f, A, sn=sn, dview=dview, **options['spatial_params'])
        # set it back to original value to perform full deconvolution
        options['temporal_params']['p'] = 2
        print('update temporal ...')
        C, A, b, f, S, bl, c1, neurons_sn, g1, YrA = update_temporal_components(
            Yr, A, b, C, f, dview=dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])


        pl.subplot(1, 2, 1)
        crd = plot_contours(A.tocsc(), Cn, thr=0.9)

    def process_video(self):
        init_method = 'greedy_roi'

        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        
        alpha_snmf = None

        fnames = [self.video_path]
        add_to_movie = 300
        downsample_factor = 1
        idx_xy = None
        base_name = 'Yr'
        name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie)
        name_new.sort()
        fname_new = name_new[0]

        Yr, dims, T = cm.load_memmap(fname_new)
        # print(dims, T)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr, dims + (T,), order='F')

        # print(Y.shape)

        # pdb.set_trace()

        Cn = cm.local_correlations(Y)

        rf           = 25     # half-size of the patches in pixels. rf=25, patches are 50x50
        stride       = 10     # amount of overlap between the patches in pixels
        K            = 10     # number of neurons expected per patch
        gSig         = [7, 7] # expected half size of neurons
        merge_thresh = 0.8    # merging threshold, max correlation allowed
        p            = 1      # order of the autoregressive system
        save_results = False

        cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis')
        cnm = cnm.fit(images)

        A_tot   = cnm.A
        C_tot   = cnm.C
        YrA_tot = cnm.YrA
        b_tot   = cnm.b
        f_tot   = cnm.f
        sn_tot  = cnm.sn

        # pdb.set_trace()

        # pl.subplot(1, 2, 1)
        # crd = plot_contours(A_tot.tocsc(), self.image, thr=0.9, display_numbers=False)

        print(('Number of components:' + str(A_tot.shape[-1])))

        #%%
        final_frate = 10 # approx final rate  (after eventual downsampling )
        Npeaks = 10
        traces = C_tot + YrA_tot
        #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
        #        traces_b=np.diff(traces,axis=1)
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
            Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

        idx_components_r = np.where(r_values >= .5)[0]
        idx_components_raw = np.where(fitness_raw < -40)[0]
        idx_components_delta = np.where(fitness_delta < -20)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

        print(('Keeping ' + str(len(idx_components)) +
               ' and discarding  ' + str(len(idx_components_bad))))
        #%%
        pl.figure()
        crd = plot_contours(A_tot.tocsc()[:, idx_components], self.image, thr=0.9, display_numbers=False)
        #%%
        A_tot = A_tot.tocsc()[:, idx_components]
        C_tot = C_tot[idx_components]
        #%%
        save_results = True
        if save_results:
            np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
                     YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)

        #%%
        cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                        f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
        cnm = cnm.fit(images)

        #%%
        A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
        #%%
        final_frate = 10

        Npeaks = 10
        traces = C + YrA
        #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
        #        traces_b=np.diff(traces,axis=1)
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
            evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                                              N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

        idx_components_r = np.where(r_values >= .95)[0]
        idx_components_raw = np.where(fitness_raw < -100)[0]
        idx_components_delta = np.where(fitness_delta < -100)[0]


        #min_radius = gSig[0] - 2
        #masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
        #    A.tocsc(), min_radius, dims, num_std_threshold=1,
        #    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        #idx_blobs = np.intersect1d(idx_components, idx_blobs)
        idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

        print(' ***** ')
        print((len(traces)))
        print((len(idx_components)))
        #print((len(idx_blobs)))
        #%%
        save_results = True
        if save_results:
            np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'), Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)

        #%% visualize components
        # pl.figure();
        pl.subplot(1, 2, 1)
        crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
        #pl.subplot(1, 3, 2)
        #crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
        pl.subplot(1, 2, 2)
        crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
        #%%
        view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
                                       idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
        #%%
        view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
                                       idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
        #%%
        C_dff = extract_DF_F(Yr, A.tocsc()[:, idx_components], C[idx_components, :], cnm.bl[idx_components], quantileMin = 8, frames_window = 200, dview = dview)
        pl.plot(C_dff.T)

        #%% STOP CLUSTER and clean up log files
        cm.stop_server()

        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

    def close_all(self):
        self.closing = True
        self.param_window.close()
        self.preview_window.close()

        json.dump(self.params, open(PARAMS_FILENAME, "w"))