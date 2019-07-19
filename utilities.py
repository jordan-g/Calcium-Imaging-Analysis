import numpy as np
import cv2
import skimage
import sys
import os
import glob
import tifffile
import time
import shutil
import h5py
import scipy
import peakutils
import matplotlib.pyplot as plt
from scipy import sparse

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as cnmf_params
# from caiman.source_extraction.cnmf import estimates as estimates
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf.temporal import update_temporal_components
from caiman.source_extraction.cnmf.pre_processing import preprocess_data
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import logging

try:
    import suite2p
    from suite2p.run_s2p import run_s2p
    suite2p_enabled = True
except:
    suite2p_enabled = False

if sys.version_info[0] < 3:
    python_version = 2
else:
    python_version = 3

def mean(movie, z=0):
    return np.mean(movie[:, z, :, :], axis=0)

def correlation(movie, z=0):
    return np.abs(cm.local_correlations_fft(movie[:, z, :, :], swap_dim=False))

def adjust_contrast(image, contrast):
    return image*contrast

def adjust_gamma(image, gamma):
    return skimage.exposure.adjust_gamma(image, gamma)

def calculate_adjusted_image(image, contrast, gamma):
    return adjust_gamma(adjust_contrast(image, contrast), gamma)

def update_temporal_traces_multiple_videos(video_paths, video_groups, params, roi_spatial_footprints, bg_spatial_footprints, use_multiprocessing=True):
    group_nums = np.unique(video_groups)

    new_roi_spatial_footprints  = {}
    new_roi_temporal_footprints = {}
    new_roi_temporal_residuals  = {}
    new_bg_spatial_footprints   = {}
    new_bg_temporal_footprints  = {}

    for n in range(len(group_nums)):
        group_num = group_nums[n]
        paths = [ video_paths[i] for i in range(len(video_paths)) if video_groups[i] == group_num ]

        for i in range(len(paths)):
            video_path = paths[i]

            video = tifffile.imread(video_path)
                
            if len(video.shape) == 3:
                # add a z dimension
                video = video[:, np.newaxis, :, :]

            # flip video 90 degrees to match what is shown in Fiji
            video = video.transpose((0, 1, 3, 2))
                
            if i == 0:
                final_video = video
            else:
                final_video = np.concatenate([final_video, video], axis=0)

        for z in range(final_video.shape[1]):
            roi_spatial_footprints_z, roi_temporal_footprints_z, roi_temporal_residuals_z, bg_spatial_footprints_z, bg_temporal_footprints_z = perform_cnmf(final_video[:, z, :, :], params, roi_spatial_footprints[z], np.ones((roi_spatial_footprints[z].shape[1], final_video.shape[0])), np.ones((roi_spatial_footprints[z].shape[1], final_video.shape[0])), bg_spatial_footprints[z], np.ones((bg_spatial_footprints[z].shape[1], final_video.shape[0])), use_multiprocessing=use_multiprocessing)

            new_roi_spatial_footprints[group_num][z]  = roi_spatial_footprints_z
            new_roi_temporal_footprints[group_num][z] = roi_temporal_footprints_z
            new_roi_temporal_residuals[group_num][z]  = roi_temporal_residuals_z
            new_bg_spatial_footprints[group_num][z]   = bg_spatial_footprints_z
            new_bg_temporal_footprints[group_num][z]  = bg_temporal_footprints_z

        # calculate temporal components based on spatial components
        new_roi_temporal_footprints[group_num], new_roi_temporal_residuals[group_num], new_bg_temporal_footprints[group_num] = calculate_temporal_components(paths, new_roi_spatial_footprints[group_num], new_roi_temporal_footprints[group_num], new_roi_temporal_residuals[group_num], new_bg_spatial_footprints[group_num], new_bg_temporal_footprints[group_num])
    
    return new_roi_spatial_footprints, new_roi_temporal_footprints, new_roi_temporal_residuals, new_bg_spatial_footprints, new_bg_temporal_footpri

def perform_cnmf(video, params, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, use_multiprocessing=True):
    if use_multiprocessing:
        backend = 'multiprocessing'

        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    video_path = "video_temp.tif"
    tifffile.imsave(video_path, video)

    # dataset dependent parameters
    fnames         = [video_path]          # filename to be processed
    fr             = params['imaging_fps'] # imaging rate in frames per second
    decay_time     = params['decay_time']  # length of a typical transient in seconds
    
    # parameters for source extraction and deconvolution
    p              = params['autoregressive_order']             # order of the autoregressive system
    gnb            = params['num_bg_components']                # number of global background components
    merge_thresh   = params['merge_threshold']                  # merging threshold, max correlation allowed
    rf             = None                                       # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf    = 6                                          # amount of overlap between the patches in pixels
    K              = roi_spatial_footprints.shape[1]                   # number of components per patch
    gSig           = [params['half_size'], params['half_size']] # expected half size of neurons
    init_method    = 'greedy_roi'                               # initialization method (if analyzing dendritic data using 'sparse_nmf')
    rolling_sum    = True
    rolling_length = 50
    is_dendrites   = False                                      # flag for analyzing dendritic data
    alpha_snmf     = None                                       # sparsity penalty for dendritic data analysis through sparse NMF

    # parameters for component evaluation
    min_SNR        = params['min_snr']          # signal to noise ratio for accepting a component
    rval_thr       = params['min_spatial_corr'] # space correlation threshold for accepting a component
    cnn_thr        = params['cnn_threshold']    # threshold for CNN based classifier

    border_pix = 0

    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C') # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')


    cnm = cnmf.CNMF(n_processes=8, k=K, gSig=gSig, merge_thresh= merge_thresh, 
                    p = p,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf, rolling_sum=rolling_sum,
                    only_init_patch = True, skip_refinement=True, gnb = gnb, border_pix = border_pix, ssub=1, ssub_B=1, tsub=1, Ain=roi_spatial_footprints, Cin=roi_temporal_footprints, b_in=bg_spatial_footprints, f_in=bg_temporal_footprints, do_merge=False)

    cnm = cnm.fit(images)

    roi_spatial_footprints  = cnm.A
    roi_temporal_footprints = cnm.C
    roi_temporal_residuals  = cnm.YrA
    bg_spatial_footprints   = cnm.b
    bg_temporal_footprints  = cnm.f

    if use_multiprocessing:
        if backend == 'multiprocessing':
            dview.close()
        else:
            try:
                dview.terminate()
            except:
                dview.shutdown()
        cm.stop_server()

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def calculate_temporal_components(video_paths, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints):
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = tifffile.imread(video_path)
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]

        if i == 0:
            final_video = video.copy()
        else:
            final_video = np.concatenate([final_video, video], axis=0)

    roi_temporal_footprints = [ None for i in range(final_video.shape[1]) ]
    roi_temporal_residuals  = [ None for i in range(final_video.shape[1]) ]
    bg_temporal_footprints  = [ None for i in range(final_video.shape[1]) ]

    for z in range(final_video.shape[1]):
        final_video_path = "video_concatenated_z_{}.tif".format(z)
        tifffile.imsave(final_video_path, final_video[:, z, :, :])

        fname_new = cm.save_memmap([final_video_path], base_name='memmap_z_{}'.format(z), order='C') # exclude borders

        # # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)

        images = final_video[:, z, :, :].transpose((1, 2, 0)).reshape((final_video.shape[2]*final_video.shape[3], final_video.shape[0]))

        Cin = np.zeros((roi_spatial_footprints[z].shape[1], final_video.shape[0]))
        fin = np.zeros((bg_spatial_footprints[z].shape[1], final_video.shape[0]))

        Yr, sn, g, psx = preprocess_data(Yr, dview=None)

        roi_temporal_footprints[z], _, _, bg_temporal_footprints[z], _, _, _, _, _, roi_temporal_residuals[z], _ = update_temporal_components(images, roi_spatial_footprints[z], bg_spatial_footprints[z], Cin, fin, bl=None, c1=None, g=g, sn=sn, nb=2, ITER=2, block_size=5000, num_blocks_per_run=20, debug=False, dview=None, p=0, method='cvx')

    return roi_temporal_footprints, roi_temporal_residuals, bg_temporal_footprints

def motion_correct_multiple_videos(video_paths, video_groups, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None, use_multiprocessing=True):
    start_time = time.time()

    mc_video_paths = []
    mc_videos  = []
    mc_borders = {}

    if use_multiprocessing:
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            # backend = 'ipyparallel'
            backend = 'multiprocessing'

        # Create the cluster
        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        c           = None
        dview       = None
        n_processes = None

    group_nums = np.unique(video_groups)

    for n in range(len(group_nums)):
        group_num = group_nums[n]
        paths = [ video_paths[i] for i in range(len(video_paths)) if video_groups[i] == group_num ]

        video_lengths = []

        directory = os.path.dirname(paths[0])

        final_video_path = os.path.join(directory, "final_video_temp.tif")

        with tifffile.TiffWriter(final_video_path) as tif:
            for i in range(len(paths)):
                print(i)
                video_path = paths[i]
                new_video_path = os.path.join(directory, "video_temp.tif")

                shutil.copyfile(video_path, new_video_path)

                video = tifffile.memmap(new_video_path)

                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]

                # flip video 90 degrees to match what is shown in Fiji
                video = video.transpose((0, 1, 3, 2))

                video_lengths.append(video.shape[0])

                for k in range(video.shape[0]):
                    tif.save(video[k])

                del video

                if os.path.exists(new_video_path):
                    os.remove(new_video_path)

        final_video = tifffile.memmap(final_video_path)

        print("final video shape: {}".format(final_video.shape))

        if len(final_video.shape) == 5:
            final_video_path_2 = os.path.join(directory, "final_video_temp_2.tif")
            tifffile.imsave(final_video_path_2, final_video.reshape((final_video.shape[0]*final_video.shape[1], final_video.shape[2], final_video.shape[3], final_video.shape[4])))
            if os.path.exists(final_video_path):
                os.remove(final_video_path)
        else:
            final_video_path_2 = final_video_path

        del final_video

        mc_video, new_video_path, mc_borders[group_num] = motion_correct(final_video_path_2, max_shift, patch_stride, patch_overlap, use_multiprocessing=use_multiprocessing, c=c, dview=dview, n_processes=n_processes)
        
        mc_video = mc_video.transpose((0, 1, 3, 2))

        for i in range(len(paths)):
            video_path    = paths[i]
            directory     = os.path.dirname(video_path)
            filename      = os.path.basename(video_path)
            mc_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_mc.tif")

            if i == 0:
                tifffile.imsave(mc_video_path, mc_video[:video_lengths[0]])
            else:
                tifffile.imsave(mc_video_path, mc_video[np.sum(video_lengths[:i]):np.sum(video_lengths[:i]) + video_lengths[i]])

            mc_video_paths.append(mc_video_path)

        if progress_signal is not None:
            progress_signal.emit(n)

        if os.path.exists(final_video_path_2):
            os.remove(final_video_path_2)
            os.remove(new_video_path)

        del mc_video

    if use_multiprocessing:
        if backend == 'multiprocessing':
            dview.close()
        else:
            try:
                dview.terminate()
            except:
                dview.shutdown()
        cm.stop_server()

    end_time = time.time()

    print("---- Motion correction finished. Elapsed time: {} s.".format(end_time - start_time))
            
    return mc_video_paths, mc_borders

def motion_correct(video_path, max_shift, patch_stride, patch_overlap, use_multiprocessing=True, c=None, dview=None, n_processes=None):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    memmap_video = tifffile.memmap(video_path)

    z_range = list(range(memmap_video.shape[1]))

    new_video_path = os.path.join(directory, "mc_video_temp.tif")

    shutil.copyfile(video_path, new_video_path)

    mc_video = tifffile.memmap(new_video_path).astype(np.uint16)

    mc_borders = [ None for z in z_range ]

    counter = 0

    for z in z_range:
        print("Motion correcting plane z={}...".format(z))
        z_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_z_{}_temp.tif".format(z))
        tifffile.imsave(z_video_path, memmap_video[:, z, :, :])

        mc_video[:, z, :, :] *= 0

        # --- PARAMETERS --- #

        params_movie = {'fname': z_video_path,
                        'max_shifts': (max_shift, max_shift),  # maximum allow rigid shift (2,2)
                        'niter_rig': 3,
                        'splits_rig': 1,  # for parallelization split the movies in  num_splits chuncks across time
                        'num_splits_to_process_rig': None,  # if none all the splits are processed and the movie is saved
                        'strides': (patch_stride, patch_stride),  # intervals at which patches are laid out for motion correction
                        'overlaps': (patch_overlap, patch_overlap),  # overlap between pathes (size of patch strides+overlaps)
                        'splits_els': 1,  # for parallelization split the movies in  num_splits chuncks across time
                        'num_splits_to_process_els': [None],  # if none all the splits are processed and the movie is saved
                        'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                        'max_deviation_rigid': 3,  # maximum deviation allowed for patch with respect to rigid shift         
                        }

        # load movie (in memory!)
        fname = params_movie['fname']
        niter_rig = params_movie['niter_rig']
        # maximum allow rigid shift
        max_shifts = params_movie['max_shifts']  
        # for parallelization split the movies in  num_splits chuncks across time
        splits_rig = params_movie['splits_rig']  
        # if none all the splits are processed and the movie is saved
        num_splits_to_process_rig = params_movie['num_splits_to_process_rig']
        # intervals at which patches are laid out for motion correction
        strides = params_movie['strides']
        # overlap between pathes (size of patch strides+overlaps)
        overlaps = params_movie['overlaps']
        # for parallelization split the movies in  num_splits chuncks across time
        splits_els = params_movie['splits_els'] 
        # if none all the splits are processed and the movie is saved
        num_splits_to_process_els = params_movie['num_splits_to_process_els']
        # upsample factor to avoid smearing when merging patches
        upsample_factor_grid = params_movie['upsample_factor_grid'] 
        # maximum deviation allowed for patch with respect to rigid
        # shift
        max_deviation_rigid = params_movie['max_deviation_rigid']

        # --- RIGID MOTION CORRECTION --- #

        # Load the original movie
        m_orig = tifffile.memmap(fname)
        # m_orig = cm.load(fname)
        min_mov = np.min(m_orig) # movie must be mostly positive for this to work

        offset_mov = -min_mov

        # Create motion correction object
        mc = MotionCorrect(fname, min_mov,
                           dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig, 
                           num_splits_to_process_rig=num_splits_to_process_rig, 
                        strides= strides, overlaps= overlaps, splits_els=splits_els,
                        num_splits_to_process_els=num_splits_to_process_els, 
                        upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid, 
                        shifts_opencv = True, nonneg_movie = True, border_nan='min')

        # Do rigid motion correction
        mc.motion_correct_rigid(save_movie=False)

        # --- ELASTIC MOTION CORRECTION --- #

        # Do elastic motion correction
        mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig, show_template=False)

        # # Save elastic shift border
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)  
        # np.savez(mc.fname_tot_els + "_bord_px_els.npz", bord_px_els)

        fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
        border_to_0 = bord_px_els     # number of pixels to exclude
        fname_new = cm.save_memmap(fnames, base_name='memmap_z_{}'.format(z), order = 'C',
                                   border_to_0 = bord_px_els) # exclude borders

        # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F') 

        mc_borders[z] = bord_px_els

        # images += np.amin(images)

        # print(np.amax(images))
        # print(np.amin(images))
        # print(type(images))

        mc_video[:, z, :, :] = (images - np.amin(images)).astype(memmap_video.dtype)

        del m_orig
        if os.path.exists(z_video_path):
            os.remove(z_video_path)

        try:
            os.remove(mc.fname_tot_rig)
            os.remove(mc.fname_tot_els)
        except:
            pass

        counter += 1

    mmap_files = glob.glob(os.path.join(directory, '*.mmap'))
    for mmap_file in mmap_files:
        try:
            os.remove(mmap_file)
        except:
            pass

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return mc_video, new_video_path, mc_borders

def find_rois_multiple_videos(video_paths, video_groups, params, mc_borders={}, progress_signal=None, thread=None, use_multiprocessing=True, method="cnmf", mask_points=[]):
    start_time = time.time()

    group_nums = np.unique(video_groups)

    if use_multiprocessing and method == "cnmf":
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            # backend = 'ipyparallel'
            backend = 'multiprocessing'

        # Create the cluster
        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        c           = None
        dview       = None
        n_processes = None

    new_roi_spatial_footprints  = {}
    new_roi_temporal_footprints = {}
    new_roi_temporal_residuals  = {}
    new_bg_spatial_footprints   = {}
    new_bg_temporal_footprints  = {}

    for n in range(len(group_nums)):
        group_num = group_nums[n]
        paths = [ video_paths[i] for i in range(len(video_paths)) if video_groups[i] == group_num ]

        directory = os.path.dirname(paths[0])

        final_video_path = os.path.join(directory, "final_video_temp.tif")

        with tifffile.TiffWriter(final_video_path, bigtiff=False) as tif:
            print("# videos: {}".format(len(paths)))
            for i in range(len(paths)):
                video_path = paths[i]
                new_video_path = os.path.join(directory, "video_temp.tif")

                shutil.copyfile(video_path, new_video_path)

                video = tifffile.memmap(new_video_path)

                print("memmap shape", video.shape)

                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]

                video = video.transpose((0, 1, 3, 2))

                print("video shape: {}".format(video.shape))

                if len(mask_points) > 0 and n in mask_points.keys():
                    mask = np.zeros(video.shape[1:]).astype(np.uint8)
                    for z in range(video.shape[1]):
                        if len(mask_points[n][z]) > 0:
                            for p in mask_points[n][z]:
                                # create mask image
                                p = np.fliplr(np.array(p + [p[0]])).astype(int)

                                # print(p.shape)

                                cv2.fillConvexPoly(mask[z, :, :], p, 1)

                        if np.sum(mask[z]) == 0:
                            mask[z] = 1

                    mask = mask.astype(bool)

                    if not params['invert_masks']:
                        mask = mask == False

                    mask = np.repeat(mask[np.newaxis, :, :, :], video.shape[0], axis=0)


                    video[mask] = 0

                for k in range(video.shape[0]):
                    tif.save(video[k])

                del video

                if os.path.exists(new_video_path):
                    os.remove(new_video_path)

        final_video = tifffile.memmap(final_video_path).astype(np.uint16)

        print("final video shape: {}".format(final_video.shape))

        if len(final_video.shape) == 5:
            final_video_path_2 = os.path.join(directory, "final_video_temp_2.tif")
            print("final video shape: {}".format(final_video.reshape((final_video.shape[0]*final_video.shape[1], final_video.shape[2], final_video.shape[3], final_video.shape[4])).shape))
            tifffile.imsave(final_video_path_2, final_video.reshape((final_video.shape[0]*final_video.shape[1], final_video.shape[2], final_video.shape[3], final_video.shape[4])))
            if os.path.exists(final_video_path):
                os.remove(final_video_path)
        else:
            final_video_path_2 = final_video_path

        del final_video

        if len(mc_borders.keys()) > 0:
            borders = mc_borders[group_num]
        else:
            borders = None

        if method == "cnmf":
            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = find_rois_cnmf(final_video_path_2, params, mc_borders=borders, use_multiprocessing=use_multiprocessing, c=c, dview=dview, n_processes=n_processes)
        else:
            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = find_rois_suite2p(final_video_path_2, params, mc_borders=borders, use_multiprocessing=use_multiprocessing)

        print("temporal footprints shape: {}".format(roi_temporal_footprints[0].shape))

        new_roi_spatial_footprints[group_num]  = roi_spatial_footprints
        new_roi_temporal_footprints[group_num] = roi_temporal_footprints
        new_roi_temporal_residuals[group_num]  = roi_temporal_residuals
        new_bg_spatial_footprints[group_num]   = bg_spatial_footprints
        new_bg_temporal_footprints[group_num]  = bg_temporal_footprints

        if os.path.exists(final_video_path_2):
            os.remove(final_video_path_2)

        if progress_signal is not None:
            progress_signal.emit(n)

    if use_multiprocessing and method == "cnmf":
        if backend == 'multiprocessing':
            dview.close()
        else:
            try:
                dview.terminate()
            except:
                dview.shutdown()
        cm.stop_server()

    end_time = time.time()

    print("---- ROI finding finished. Elapsed time: {} s.".format(end_time - start_time))

    return new_roi_spatial_footprints, new_roi_temporal_footprints, new_roi_temporal_residuals, new_bg_spatial_footprints, new_bg_temporal_footprints

def find_rois_cnmf(video_path, params, mc_borders=None, use_multiprocessing=True, c=None, dview=None, n_processes=None):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    new_video_path = os.path.join(directory, "cnmf_video_temp.tif")

    shutil.copyfile(video_path, new_video_path)

    memmap_video = tifffile.memmap(video_path)
    print(memmap_video.shape)

    if len(memmap_video.shape) == 5:
        num_z = memmap_video.shape[2]
    else:
        num_z = memmap_video.shape[1]

    roi_spatial_footprints  = [ None for i in range(num_z) ]
    roi_temporal_footprints = [ None for i in range(num_z) ]
    roi_temporal_residuals  = [ None for i in range(num_z) ]
    bg_spatial_footprints   = [ None for i in range(num_z) ]
    bg_temporal_footprints  = [ None for i in range(num_z) ]

    for z in range(num_z):
        fname = os.path.splitext(filename)[0] + "_masked_z_{}.tif".format(z)

        z_video_path = os.path.join(directory, fname)

        if len(memmap_video.shape) == 5:
            tifffile.imsave(z_video_path, memmap_video[:, :, z, :, :].reshape((-1, memmap_video.shape[3], memmap_video.shape[4])))
        else:
            tifffile.imsave(z_video_path, memmap_video[:, z, :, :])

        # dataset dependent parameters
        fnames         = [z_video_path]        # filename to be processed
        fr             = params['imaging_fps'] # imaging rate in frames per second
        decay_time     = params['decay_time']  # length of a typical transient in seconds
        
        # parameters for source extraction and deconvolution
        p              = params['autoregressive_order']             # order of the autoregressive system
        gnb            = params['num_bg_components']                # number of global background components
        merge_thresh   = params['merge_threshold']                  # merging threshold, max correlation allowed
        if params['use_patches']:
            rf             = params['cnmf_patch_size']
            stride         = params['cnmf_patch_stride']
        else:
            rf             = None                                       # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride         = None                                       # amount of overlap between the patches in pixels
        K              = params['num_components']                   # number of components per patch
        gSig           = [params['half_size'], params['half_size']] # expected half size of neurons
        init_method    = params['init_method']                               # initialization method (if analyzing dendritic data using 'sparse_nmf')
        # rolling_sum    = True
        # rolling_length = 50
        is_dendrites   = False                                      # flag for analyzing dendritic data
        alpha_snmf     = None                                       # sparsity penalty for dendritic data analysis through sparse NMF

        # parameters for component evaluation
        min_SNR        = params['min_snr']          # signal to noise ratio for accepting a component
        rval_thr       = params['min_spatial_corr'] # space correlation threshold for accepting a component
        # cnn_thr        = params['cnn_threshold']    # threshold for CNN based classifier
        max_merge_area = params['max_merge_area']

        if mc_borders is not None:
            border_pix = mc_borders[z]
        else:
            border_pix = 0

        fname_new = cm.save_memmap(fnames, base_name='memmap_z_{}'.format(z), order='C') # exclude borders

        # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        params_dict = {'fnames': fnames,
                       'fr': fr,
                       'decay_time': decay_time,
                       'rf': rf,
                       'stride': stride,
                       'K': K,
                       'gSig': gSig,
                       'merge_thr': merge_thresh,
                       'p': p,
                       'nb': gnb,
                       'init_method': init_method,
                       'dims': memmap_video.shape[-2:],
                       'max_merge_area': max_merge_area}

        # cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh= merge_thresh, 
        #                 p = p,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
        #                 method_init=init_method, alpha_snmf=alpha_snmf, rolling_sum=rolling_sum,
        #                 only_init_patch = False, gnb = gnb, border_pix = border_pix, ssub=1, ssub_B=1, tsub=1)

        # cnm = cnm.fit(images)

        opts = cnmf_params.CNMFParams(params_dict=params_dict)
        # %% Now RUN CaImAn Batch (CNMF)
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit_file()

        # %% load memory mapped file
        Yr, dims, T = cm.load_memmap(cnm.mmap_file)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        # try:
        #     A_in, C_in, b_in, f_in = cnm.A, cnm.C, cnm.b, cnm.f
        # except:
        #     A_in, C_in, b_in, f_in = cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f

        # cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
        #                 merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
        #                 f_in=f_in, rf = None, stride = None, gnb = gnb, 
        #                 method_deconvolution='oasis', check_nan = True)

        # cnm2 = cnm2.fit(images)

        cnm2 = cnm.refit(images, dview=dview)

        try:
            roi_spatial_footprints[z]  = cnm2.A
            roi_temporal_footprints[z] = cnm2.C
            roi_temporal_residuals[z]  = cnm2.YrA
            bg_spatial_footprints[z]   = cnm2.b
            bg_temporal_footprints[z]  = cnm2.f
        except:
            roi_spatial_footprints[z]  = cnm2.estimates.A
            roi_temporal_footprints[z] = cnm2.estimates.C
            roi_temporal_residuals[z]  = cnm2.estimates.YrA
            bg_spatial_footprints[z]   = cnm2.estimates.b
            bg_temporal_footprints[z]  = cnm2.estimates.f

        if os.path.exists(z_video_path):
            os.remove(z_video_path)

    del memmap_video

    if os.path.exists(new_video_path):
        os.remove(new_video_path)

    mmap_files = glob.glob(os.path.join(directory, "*.mmap"))
    for mmap_file in mmap_files:
        try:
            os.remove(mmap_file)
        except:
            pass

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def find_rois_suite2p(video_path, params, mc_borders=None, use_multiprocessing=True):
    if suite2p_enabled:
        full_video_path = video_path

        video = tifffile.memmap(video_path)

        directory = os.path.dirname(full_video_path)
        filename  = os.path.basename(full_video_path)

        roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
        roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
        roi_temporal_residuals  = [ None for i in range(video.shape[1]) ]
        bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
        bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]

        if os.path.exists("suite2p"):
            shutil.rmtree("suite2p")

        for z in range(video.shape[1]):
            fname = os.path.splitext(filename)[0] + "_masked_z_{}.h5".format(z)

            z_video_path = os.path.join(directory, fname)

            h5f = h5py.File(z_video_path, 'w')
            h5f.create_dataset('data', data=video[:, z, :, :])
            h5f.close()

            ops = {
                'fast_disk': [], # used to store temporary binary file, defaults to save_path0
                'save_path0': '', # stores results, defaults to first item in data_path
                'delete_bin': False, # whether to delete binary file after processing
                # main settings
                'nplanes' : 1, # each tiff has these many planes in sequence
                'nchannels' : 1, # each tiff has these many channels per plane
                'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
                'diameter':params['diameter'], # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
                'tau':  1., # this is the main parameter for deconvolution
                'fs': params['sampling_rate'],  # sampling rate (total across planes)
                # output settings
                'save_mat': False, # whether to save output as matlab files
                'combined': True, # combine multiple planes into a single result /single canvas for GUI
                # parallel settings
                'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
                'num_workers_roi': 0, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
                # registration settings
                'do_registration': False, # whether to register data
                'nimg_init': 200, # subsampled frames for finding reference image
                'batch_size': 200, # number of frames per batch
                'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
                'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
                'reg_tif': False, # whether to save registered tiffs
                'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
                # cell detection settings
                'connected': params['connected'], # whether or not to keep ROIs fully connected (set to 0 for dendrites)
                'navg_frames_svd': 5000, # max number of binned frames for the SVD
                'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
                'max_iterations': 20, # maximum number of iterations to do cell detection
                'ratio_neuropil': params['neuropil_basis_ratio'], # ratio between neuropil basis size and cell radius
                'ratio_neuropil_to_cell': params['neuropil_radius_ratio'], # minimum ratio between neuropil radius and cell radius
                'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
                'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
                'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
                'inner_neuropil_radius': params['inner_neuropil_radius'], # number of pixels to keep between ROI and neuropil donut
                'outer_neuropil_radius': np.inf, # maximum neuropil radius
                'min_neuropil_pixels': params['min_neuropil_pixels'], # minimum number of pixels in the neuropil
                # deconvolution settings
                'baseline': 'maximin', # baselining mode
                'win_baseline': 60., # window for maximin
                'sig_baseline': 10., # smoothing constant for gaussian filter
                'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
                'neucoeff': .7,  # neuropil coefficient
            }

            db = {
                'h5py': z_video_path, # a single h5 file path
                'h5py_key': 'data',
                'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
                'data_path': [], # a list of folders with tiffs 
                                                     # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                'subfolders': [] # choose subfolders of 'data_path' to look in (optional)
            }

            opsEnd=run_s2p(ops=ops,db=db)

            stat = np.load("suite2p/plane0/stat.npy")
            F    = np.load("suite2p/plane0/F.npy")
            Fneu = np.load("suite2p/plane0/Fneu.npy")

            spatial_components = np.zeros((video.shape[2], video.shape[3], len(stat)))
            for i in range(len(stat)):
                spatial_components[stat[i]['xpix'], stat[i]['ypix'], i] = stat[i]['lam']

            roi_spatial_footprints[z]  = scipy.sparse.coo_matrix(spatial_components.reshape((video.shape[2]*video.shape[3], len(stat))))
            roi_temporal_footprints[z] = F - ops["neucoeff"]*Fneu
            roi_temporal_residuals[z]  = np.zeros(F.shape)
            bg_spatial_footprints[z]   = None
            bg_temporal_footprints[z]  = None

            os.remove(fname)
            shutil.rmtree("suite2p")

        return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def filter_rois(video_paths, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, params):
    directory = os.path.dirname(video_paths[0])

    final_video_path = os.path.join(directory, "final_video_temp.tif")

    with tifffile.TiffWriter(final_video_path, bigtiff=True) as tif:
        for i in range(len(video_paths)):
            video_path = video_paths[i]

            video = tifffile.memmap(video_path)
                
            if len(video.shape) == 3:
                # add a z dimension
                video = video[:, np.newaxis, :, :]

            for k in range(video.shape[0]):
                tif.save(video[k])

            del video

    final_video = tifffile.memmap(final_video_path)

    if len(final_video.shape) == 5:
        final_video_path_2 = os.path.join(directory, "final_video_temp_2.tif")
        tifffile.imsave(final_video_path_2, final_video.reshape((final_video.shape[0]*final_video.shape[1], final_video.shape[2], final_video.shape[3], final_video.shape[4])))
        if os.path.exists(final_video_path):
            os.remove(final_video_path)
    else:
        final_video_path_2 = final_video_path

    del final_video

    filtered_out_rois = []

    memmap_video = tifffile.memmap(final_video_path_2)

    if len(memmap_video.shape) == 5:
        n_frames = memmap_video.shape[1]
        num_z = memmap_video.shape[2]
        height = memmap_video.shape[3]
        width = memmap_video.shape[4]
    else:
        n_frames = memmap_video.shape[0]
        num_z = memmap_video.shape[1]
        height = memmap_video.shape[2]
        width = memmap_video.shape[3]

    for z in range(num_z):
        directory = os.path.dirname(final_video_path_2)
        filename  = os.path.basename(final_video_path_2)

        fname = os.path.splitext(filename)[0] + "_masked_z_{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap".format(z, height, width, n_frames)

        video_path = os.path.join(directory, fname)

        if len(memmap_video.shape) == 5:
            tifffile.imsave(video_path, memmap_video[:, :, z, :, :].reshape((-1, memmap_video.shape[3], memmap_video.shape[4])).transpose([1, 2, 0]).astype(np.float32))
        else:
            tifffile.imsave(video_path, memmap_video[:, z, :, :].transpose([1, 2, 0]).astype(np.float32))

        video = tifffile.memmap(video_path)

        print(video.shape)

        # e = estimates.Estimates(A=roi_spatial_footprints[z], b=bg_spatial_footprints[z], C=roi_temporal_footprints[z], f=bg_temporal_footprints[z], R=roi_temporal_residuals[z])

        # e = e.filter_components(video, )

        dims = video.shape[:2]

        print("dims", dims)

        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
                estimate_components_quality_auto(video, roi_spatial_footprints[z], roi_temporal_footprints[z], bg_spatial_footprints[z], bg_temporal_footprints[z], 
                                                 roi_temporal_residuals[z], params['imaging_fps']/num_z, params['decay_time'], [params['half_size'], params['half_size']], dims, 
                                                 dview = None, min_SNR=params['min_snr'], 
                                                 r_values_min = params['min_spatial_corr'], use_cnn = params['use_cnn'], 
                                                 thresh_cnn_min = params['cnn_accept_threshold'], thresh_cnn_lowest=params['cnn_reject_threshold'], gSig_range=[ (i, i) for i in range(max(1, params['half_size']-2), params['half_size']+2) ])

        # gSig_range=[ (i, i) for i in range(max(1, params['half_size']-5), params['half_size']+5) ]

        # filtered_out_rois.append(list(idx_components_bad))

        print(idx_components_bad.shape)

        if isinstance(roi_spatial_footprints[z], scipy.sparse.coo_matrix):
            f = roi_spatial_footprints[z].toarray()
        else:
            f = roi_spatial_footprints[z]

        size_neurons_gt = (f > 0).sum(0)
        neurons_to_discard = np.where((size_neurons_gt < params['min_area']) | (size_neurons_gt > params['max_area']))[0]

        print(neurons_to_discard.shape)

        # for i in range(f.shape[-1]):
        #     area = np.sum(f[:, i] > 0)
        #     if (area < params['min_area'] or area > params['max_area']) and i not in filtered_out_rois[-1]:
        #         filtered_out_rois[-1].append(i)

        idx_components_bad = np.union1d(idx_components_bad, neurons_to_discard)

        zscores = (roi_temporal_footprints[z] - np.mean(roi_temporal_footprints[z], axis=1)[:, np.newaxis])/np.std(roi_temporal_footprints[z], axis=1)[:, np.newaxis]
        zscore_diffs = np.diff(zscores, axis=0)
        min_zscore_diffs = np.amin(zscore_diffs, axis=1)

        neurons_to_discard = np.where(min_zscore_diffs < -params['artifact_decay_speed'])[0]

        print(neurons_to_discard.shape)

        idx_components_bad = np.union1d(idx_components_bad, neurons_to_discard)

        abs_traces = np.abs(roi_temporal_footprints[z])
        df_f = np.abs(np.amax(roi_temporal_footprints[z], axis=1) - np.amin(roi_temporal_footprints[z], axis=1))/np.mean(abs_traces[:, :10])
        print(df_f)
        neurons_to_discard = np.where(df_f < params['min_df_f'])[0]

        idx_components_bad = np.union1d(idx_components_bad, neurons_to_discard)

        print(idx_components_bad)

        filtered_out_rois.append(list(idx_components_bad))

        print(filtered_out_rois)

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass

        del video

    del memmap_video

    if os.path.exists(final_video_path_2):
        os.remove(final_video_path_2)

    mmap_files = glob.glob(os.path.join(directory, "*.mmap"))
    for mmap_file in mmap_files:
        try:
            os.remove(mmap_file)
        except:
            pass

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return filtered_out_rois

def get_roi_containing_point(spatial_footprints, roi_point, video_shape):
    flattened_point = roi_point[0]*video_shape[0] + roi_point[1]

    if flattened_point >= video_shape[0]*video_shape[1]:
        return None

    try:
        spatial_footprints = spatial_footprints.toarray()
    except:
        pass

    if spatial_footprints is not None:
        roi = np.argmax(spatial_footprints[flattened_point, :])
        
        if spatial_footprints[flattened_point, roi] != 0:

            print(np.sum(spatial_footprints[:, roi] > 0))
            return roi
    else:
        return None

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def calculate_tail_beat_frequency(fps, tail_angle_array):
    tail_angles = tail_angle_array.copy()

    baseline = np.mean(tail_angles[:100])
    tail_angles -= baseline

    N = 10
    smoothed_tail_angles = np.convolve(tail_angles, np.ones((N,))/N, mode='valid')

    derivative = np.abs(np.diff(smoothed_tail_angles, append=[0]))/0.01
    smoothed_derivative = np.convolve(derivative, np.ones((N,))/N, mode='same')

    threshold = 2
    min_dist = 5
    min_deriv = 10
    highs = peakutils.peak.indexes(smoothed_tail_angles, thres=threshold/max(smoothed_tail_angles), min_dist=min_dist)
    highs = np.array([ h for h in highs if smoothed_derivative[h] > min_deriv ])

    lows = peakutils.peak.indexes(-smoothed_tail_angles, thres=threshold/max(-smoothed_tail_angles), min_dist=min_dist)
    lows = np.array([ h for h in lows if smoothed_derivative[h] > min_deriv ])

    low_freqs = [ 1.0/(lows[i] - lows[i-1]) for i in range(1, len(lows)) ]

    low_freqs_array = np.zeros(smoothed_tail_angles.shape)
    for i in range(len(low_freqs)):
        low_freqs_array[lows[i]:lows[i+1]] = low_freqs[i]

    high_freqs = [ 1.0/(highs[i] - highs[i-1]) for i in range(1, len(highs)) ]

    high_freqs_array = np.zeros(smoothed_tail_angles.shape)
    for i in range(len(high_freqs)):
        high_freqs_array[highs[i]:highs[i+1]] = high_freqs[i]

    freqs_array = (low_freqs_array + high_freqs_array)/2

    return fps*freqs_array

def add_data_to_dataset(roi_spatial_footprints, mean_image, positive_rois, negative_rois, half_size, dataset_filename="zebrafish_gcamp_dataset.h5"):
    num_total_rois = len(positive_rois) + len(negative_rois)

    # make sure ROI and non-ROI training data are the same size
    if len(negative_rois) == 0 or len(positive_rois) == 0:
        return

    num_negative_rois = len(negative_rois)
    num_positive_rois = len(positive_rois)

    num_samples_of_each = np.minimum(num_negative_rois, num_positive_rois)

    positive_roi_spatial_footprints = roi_spatial_footprints[:, positive_rois[:num_samples_of_each]]
    negative_roi_spatial_footprints = roi_spatial_footprints[:, negative_rois[:num_samples_of_each]]

    # generate labels for each set of training data

    positive_roi_labels = np.zeros((num_samples_of_each, 2))
    positive_roi_labels[:, 0] = 1
    negative_roi_labels = np.zeros((num_samples_of_each, 2))
    negative_roi_labels[:, 1] = 1

    print(positive_roi_spatial_footprints.shape, negative_roi_spatial_footprints.shape)

    print(positive_roi_labels.shape, negative_roi_labels.shape)

    # final_roi_spatial_footprints = np.concatenate([positive_roi_spatial_footprints, negative_roi_spatial_footprints], axis=-1)

    final_roi_spatial_footprints = sparse.hstack([positive_roi_spatial_footprints, negative_roi_spatial_footprints])
    final_roi_labels = np.concatenate([positive_roi_labels, negative_roi_labels], axis=0).T
    # final_roi_labels = sparse.hstack([positive_roi_labels, negative_roi_labels])

    # shuffle data
    final_roi_spatial_footprints, final_roi_labels = shuffle_arrays(final_roi_spatial_footprints, final_roi_labels)

    input_data, _ = preprocess_spatial_footprints(final_roi_spatial_footprints, mean_image, half_size)

    existing_images, existing_labels = load_dataset(dataset_filename)

    # print(existing_images)
    # print(type(existing_images))

    if existing_images.shape[0] > 1 or np.sum(existing_images) != 0:
        print("Adding to existing dataset.")
        final_images = np.concatenate([existing_images, input_data], axis=0)
        final_labels = np.concatenate([existing_labels, final_roi_labels.T], axis=0)
    else:
        final_images = input_data
        final_labels = final_roi_labels.T

    print(final_images.shape)
    print(final_labels.shape)

    save_dataset(final_images, final_labels, dataset_filename)

def train_cnn_on_data(roi_spatial_footprints, mean_image, positive_rois, negative_rois, half_size, lr=1e-4):
    loaded_model = load_model()

    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])

    # num_total_rois = len(positive_rois) + len(negative_rois)

    # # make sure ROI and non-ROI training data are the same size
    # if len(negative_rois) == 0 or len(positive_rois) == 0:
    #     return

    # num_negative_rois = len(negative_rois)
    # num_positive_rois = len(positive_rois)

    # num_samples_of_each = np.minimum(num_negative_rois, num_positive_rois)

    # positive_roi_spatial_footprints = roi_spatial_footprints[:, positive_rois[:num_samples_of_each]]
    # negative_roi_spatial_footprints = roi_spatial_footprints[:, negative_rois[:num_samples_of_each]]

    # # generate labels for each set of training data

    # positive_roi_labels = np.zeros((num_samples_of_each, 2))
    # positive_roi_labels[:, 0] = 1
    # negative_roi_labels = np.zeros((num_samples_of_each, 2))
    # negative_roi_labels[:, 1] = 1

    # print(positive_roi_spatial_footprints.shape, negative_roi_spatial_footprints.shape)

    # print(positive_roi_labels.shape, negative_roi_labels.shape)

    # # final_roi_spatial_footprints = np.concatenate([positive_roi_spatial_footprints, negative_roi_spatial_footprints], axis=-1)

    # final_roi_spatial_footprints = sparse.hstack([positive_roi_spatial_footprints, negative_roi_spatial_footprints])
    # final_roi_labels = np.concatenate([positive_roi_labels, negative_roi_labels], axis=0).T
    # # final_roi_labels = sparse.hstack([positive_roi_labels, negative_roi_labels])

    # # shuffle data
    # final_roi_spatial_footprints, final_roi_labels = shuffle_arrays(final_roi_spatial_footprints, final_roi_labels)

    # input_data, _ = preprocess_spatial_footprints(final_roi_spatial_footprints, mean_image, half_size)
    
    # plot_sample_data(input_data, final_roi_labels.T, None)

    input_data, final_roi_labels = load_dataset(filename="zebrafish_gcamp_dataset.h5")

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        )

    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=True)

    datagen.fit(input_data)

    # validation_datagen = ImageDataGenerator()

    # train_generator = datagen.flow(input_data, final_roi_labels.T, batch_size=32, shuffle=True, subset="training")
     
    # validation_generator = datagen.flow(input_data, final_roi_labels.T, batch_size=32, shuffle=False, subset="validation")

    # fit the model
    # loaded_model.fit(input_data, final_roi_labels.T, nb_epoch=5, batch_size=1, validation_split=0.15, verbose=1)

    print("Training...")

    loaded_model.fit_generator(datagen.flow(input_data, final_roi_labels, batch_size=32), steps_per_epoch=input_data.shape[0] / 32, epochs=5, verbose=1)

    # history = loaded_model.fit_generator(
    #       train_generator,
    #       steps_per_epoch=input_data.shape[0]/32,
    #       epochs=50,
    #       validation_data=validation_generator,
    #       validation_steps=input_data.shape[0]/32,
    #       verbose=1)

    # save the new model
    save_model(loaded_model)

def shuffle_arrays(*args):
    '''
    Shuffle multiple arrays using the same random permutation.
    Arguments:
        args (tuple of ndarrays) : Arrays to shuffle.
    Returns:
        results (tuple of ndarrays) : Shuffled arrays.
    '''

    p = np.random.permutation(args[0].shape[1])
    results = (a[:, p] for a in args)
    return results

def preprocess_spatial_footprints(roi_spatial_footprints, mean_image, half_size, roi_overlays=None):
    dims = mean_image.shape

    half_crop = (half_size * 2 + 1, half_size * 2 + 1)

    dims = np.array(dims)
    coms = [scipy.ndimage.center_of_mass(
        mm.toarray().reshape(dims, order='F')) for mm in roi_spatial_footprints.tocsc().T]
    coms = np.maximum(coms, half_crop)
    coms = np.array([np.minimum(cms, dims - half_crop)
                     for cms in coms]).astype(np.int)

    crop_imgs = [mm.toarray().reshape(dims, order='F')[com[0] - half_crop[0]:com[0] + half_crop[0],
                                                       com[1] - half_crop[1]:com[1] + half_crop[1]] for mm, com in zip(roi_spatial_footprints.tocsc().T, coms)]

    # crop mean image instead of using just the ROI spatial footprint
    mean_image_crops = [mean_image[com[0] - half_crop[0]:com[0] + half_crop[0],
                            com[1] - half_crop[1]:com[1] + half_crop[1]] for com in coms]

    if roi_overlays is not None:
        overlay_crops = [roi_overlays[i, coms[i][0] - half_crop[0]:coms[i][0] + half_crop[0],
                                coms[i][1] - half_crop[1]:coms[i][1] + half_crop[1], :] for i in range(len(coms))]

    final_crops = np.array([cv2.resize(
        im / np.linalg.norm(im), (50, 50)) for im in crop_imgs])[:, :, :, np.newaxis]

    final_crops = np.repeat(final_crops, 3, -1)

    mean_image_crops = np.array([cv2.resize(
        im / np.linalg.norm(im), (50, 50)) for im in mean_image_crops])[:, :, :, np.newaxis]

    mean_image_crops = np.repeat(mean_image_crops, 3, -1)

    final_crops[:, :, :, :2] = mean_image_crops[:, :, :, :2]

    if roi_overlays is not None:
        overlay_crops = np.array([cv2.resize(
            im, (50, 50)) for im in overlay_crops])

        overlay_crops = np.array(overlay_crops)

    if roi_overlays is not None:
        return final_crops, mean_image_crops, overlay_crops
    else:
        return final_crops, mean_image_crops


def load_model(model_filename="vggz_model.h5", reset=False):

    if os.path.exists(model_filename) and not reset:
        print("Loading model from file...")

        model = models.load_model(model_filename)
    else:
        print("Creating a new model...")

        # load in pre-trained VGG model without the fully-connected layers
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))

        # freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False

        # create the model
        model = models.Sequential()
         
        # add the vgg convolutional base model
        model.add(vgg_conv)
         
        # add new layers
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        # save the model
        model.save(model_filename)

    print(model.layers[0].summary())
    print(model.summary())

    for layer in model.layers[0].layers:
        print(layer, layer.trainable)

    for layer in model.layers[1:]:
        print(layer, layer.trainable)

    # plot_conv_weights(model.layers[0], 'block1_conv2')
    # plot_conv_weights(model.layers[0], 'block2_conv2')
    # plot_conv_weights(model.layers[0], 'block3_conv3')
    # plot_conv_weights(model.layers[0], 'block4_conv3')
    # plot_conv_weights(model.layers[0], 'block5_conv3')

    print("Done.")

    return model

def save_model(model, model_filename="vggz_model.h5"):
    print("Saving model...")

    model.save(model_filename)

    print("Done.")

def test_cnn_on_data(roi_spatial_footprints, mean_image, discarded_rois, half_size):
    loaded_model = load_model()

    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    input_data, _ = preprocess_spatial_footprints(roi_spatial_footprints, mean_image, half_size)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        )

    datagen.fit(input_data)

    datagen.standardize(input_data)

    kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in discarded_rois ]

    print("{} kept ROIs, {} discarded ROIs.".format(len(kept_rois), len(discarded_rois)))

    roi_labels = np.zeros((roi_spatial_footprints.shape[-1], 2))
    roi_labels[kept_rois, 0] = 1
    roi_labels[discarded_rois, 1] = 1

    # scores = loaded_model.evaluate(input_data, roi_labels, verbose=1)

    # print("{}: {:.2f}%".format(loaded_model.metrics_names[1], scores[1]*100))
    
    predictions = loaded_model.predict(input_data, batch_size=32, verbose=1)

    # plot_sample_data(input_data, roi_labels, predictions)

    return predictions, input_data

def plot_sample_data(input_data, roi_labels, predictions=None):
    categories = ["Y", "N"]
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.ravel()
    indices = np.random.choice(list(range(input_data.shape[0])), 25).astype(int)
    for i in range(25):
        axes[i].imshow(input_data[indices[i], :, :, 0])
        if predictions is not None:
            axes[i].set_title("{} | {:.1f}%, {:.1f}%.".format(categories[np.argmax(roi_labels[indices[i]])], 100*predictions[indices[i], 0], 100*predictions[indices[i], 1]))
        else:
            axes[i].set_title("{}".format(categories[np.argmax(roi_labels[indices[i]])]))

    plt.tight_layout()
    plt.show()

def plot_conv_weights(model, layer):
    W = model.get_layer(name=layer).get_weights()[0]
    if len(W.shape) == 4:
        W = np.squeeze(W)
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
        fig, axs = plt.subplots(5,5, figsize=(8,8))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(W[:,:,i])
            axs[i].set_title(str(i))

        plt.show()

def load_dataset(filename="zebrafish_gcamp_dataset.h5", reset=False):
    if os.path.exists(filename) and not reset:
        print("Loading dataset from file...")

        f = h5py.File(filename, "r")

        images = f["images"].value
        labels = f["labels"].value

        f.close()
    else:
        print("Creating a new dataset...")

        f = h5py.File(filename, "w")

        images = f.create_dataset("images", data=np.zeros((1, 50, 50, 3))).value
        labels = f.create_dataset("labels", data=np.zeros((1, 2))).value

        f.close()

    return images, labels

def save_dataset(images, labels, filename="zebrafish_gcamp_dataset.h5"):
    print("Saving dataset...")

    f = h5py.File(filename, "w")

    f.create_dataset("images", data=images)
    f.create_dataset("labels", data=labels)

    f.close()

    print("Done.")

def create_dataset_subset(images, labels, kept_rois):

    final_images = images[kept_rois]
    final_labels = labels[kept_rois]
    # num_total_rois = len(positive_rois) + len(negative_rois)

    # print(num_total_rois)

    # # make sure ROI and non-ROI training data are the same size
    # if len(negative_rois) == 0 or len(positive_rois) == 0:
    #     return

    # num_negative_rois = len(negative_rois)
    # num_positive_rois = len(positive_rois)

    # num_samples_of_each = np.minimum(num_negative_rois, num_positive_rois)

    # # generate labels for each set of training data

    # positive_roi_labels = np.zeros((num_samples_of_each, 2))
    # positive_roi_labels[:, 0] = 1
    # negative_roi_labels = np.zeros((num_samples_of_each, 2))
    # negative_roi_labels[:, 1] = 1

    # labels = np.concatenate([positive_roi_labels, negative_roi_labels], axis=0)

    # positive_images = images[positive_rois[:num_samples_of_each]]
    # negative_images = images[negative_rois[:num_samples_of_each]]
    # final_images = np.concatenate([positive_images, negative_images], axis=0)

    return final_images, labels