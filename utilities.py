import numpy as np
import cv2
import skimage
import sys
import os
import glob
import skimage.external.tifffile as tifffile
import time
import shutil
import h5py

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf.temporal import update_temporal_components
from caiman.source_extraction.cnmf.pre_processing import preprocess_data

import suite2p
from suite2p.run_s2p import run_s2p

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
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            backend = 'ipyparallel'

        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    # print("~")
    # print(roi_spatial_footprints.shape)

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

    mc_videos  = []
    mc_borders = {}
    
    group_nums = np.unique(video_groups)

    for n in range(len(group_nums)):
        group_num = group_nums[n]
        paths = [ video_paths[i] for i in range(len(video_paths)) if video_groups[i] == group_num ]

        video_lengths = []

        for i in range(len(paths)):
            video_path = paths[i]

            video = tifffile.imread(video_path)
                
            if len(video.shape) == 3:
                # add a z dimension
                video = video[:, np.newaxis, :, :]

            # flip video 90 degrees to match what is shown in Fiji
            video = video.transpose((0, 1, 3, 2))
            
            video_lengths.append(video.shape[0])
                
            if i == 0:
                final_video = video
            else:
                final_video = np.concatenate([final_video, video], axis=0)
        
        final_video_path = "video_temp.tif"

        mc_video, mc_borders[group_num] = motion_correct(final_video, final_video_path, max_shift, patch_stride, patch_overlap, use_multiprocessing=use_multiprocessing)
        
        mc_video = mc_video.transpose((0, 1, 3, 2))

        for i in range(len(paths)):
            if i == 0:
                mc_videos.append(mc_video[:video_lengths[0]])
            else:
                mc_videos.append(mc_video[np.sum(video_lengths[:i]):np.sum(video_lengths[:i]) + video_lengths[i]])

        if progress_signal is not None:
            progress_signal.emit(n)

    end_time = time.time()

    print("---- Motion correction finished. Elapsed time: {} s.".format(end_time - start_time))
            
    return mc_videos, mc_borders

def motion_correct(video, video_path, max_shift, patch_stride, patch_overlap, use_multiprocessing=True):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    if use_multiprocessing:
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            backend = 'ipyparallel'

        # Create the cluster
        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    z_range = list(range(video.shape[1]))

    mc_video = video.copy()

    mc_borders = [ None for z in z_range ]

    counter = 0

    for z in z_range:
        video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_z_{}_temp.tif".format(z))
        tifffile.imsave(video_path, video[:, z, :, :])

        mc_video[:, z, :, :] *= 0

        a = tifffile.imread(video_path)

        # --- PARAMETERS --- #

        params_movie = {'fname': video_path,
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
        m_orig = cm.load(fname)
        min_mov = np.min(m_orig) # movie must be mostly positive for this to work

        offset_mov = -min_mov

        # Create motion correction object
        mc = MotionCorrect(fname, min_mov,
                           dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig, 
                           num_splits_to_process_rig=num_splits_to_process_rig, 
                        strides= strides, overlaps= overlaps, splits_els=splits_els,
                        num_splits_to_process_els=num_splits_to_process_els, 
                        upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid, 
                        shifts_opencv = True, nonneg_movie = True)

        # Do rigid motion correction
        mc.motion_correct_rigid(save_movie=False)

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(counter + (1/3))/len(z_range))
            progress_signal.emit(percent_complete)

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

        mc_video[:, z, :, :] = images

        os.remove(video_path)

        try:
            os.remove(mc.fname_tot_rig)
            os.remove(mc.fname_tot_els)
            os.remove(video_path)
        except:
            pass

        counter += 1

    if use_multiprocessing:
        if backend == 'multiprocessing':
            dview.close()
        else:
            try:
                dview.terminate()
            except:
                dview.shutdown()
        cm.stop_server()

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return mc_video, mc_borders

def find_rois_multiple_videos(video_paths, video_groups, params, mc_borders={}, progress_signal=None, thread=None, use_multiprocessing=True, method="cnmf"):
    start_time = time.time()

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

            video = video.transpose((0, 1, 3, 2))

            if i == 0:
                final_video = video.copy()
            else:
                final_video = np.concatenate([final_video, video], axis=0)

        final_video_path = "video_temp.tif"

        if len(mc_borders.keys()) > 0:
            borders = mc_borders[group_num]
        else:
            borders = None

        if method == "cnmf":
            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = find_rois_cnmf(final_video, final_video_path, params, mc_borders=borders, use_multiprocessing=use_multiprocessing)
        else:
            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = find_rois_suite2p(final_video, final_video_path, params, mc_borders=borders, use_multiprocessing=use_multiprocessing)

        new_roi_spatial_footprints[group_num]  = roi_spatial_footprints
        new_roi_temporal_footprints[group_num] = roi_temporal_footprints
        new_roi_temporal_residuals[group_num]  = roi_temporal_residuals
        new_bg_spatial_footprints[group_num]   = bg_spatial_footprints
        new_bg_temporal_footprints[group_num]  = bg_temporal_footprints

        if progress_signal is not None:
            progress_signal.emit(n)

    end_time = time.time()

    print("---- ROI finding finished. Elapsed time: {} s.".format(end_time - start_time))

    return new_roi_spatial_footprints, new_roi_temporal_footprints, new_roi_temporal_residuals, new_bg_spatial_footprints, new_bg_temporal_footprints

def find_rois_cnmf(video, video_path, params, mc_borders=None, use_multiprocessing=True):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
    roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
    roi_temporal_residuals  = [ None for i in range(video.shape[1]) ]
    bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
    bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]

    # Create the cluster
    if use_multiprocessing:
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            backend = 'ipyparallel'

        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    for z in range(video.shape[1]):
        fname = os.path.splitext(filename)[0] + "_masked_z_{}.tif".format(z)

        video_path = os.path.join(directory, fname)
        tifffile.imsave(video_path, video[:, z, :, :])

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
        K              = params['num_components']                   # number of components per patch
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

        if mc_borders is not None:
            border_pix = mc_borders[z]
        else:
            border_pix = 0

        fname_new = cm.save_memmap(fnames, base_name='memmap_z_{}'.format(z), order='C') # exclude borders

        # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh= merge_thresh, 
                        p = p,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf, rolling_sum=rolling_sum,
                        only_init_patch = False, gnb = gnb, border_pix = border_pix, ssub=1, ssub_B=1, tsub=1)

        cnm = cnm.fit(images)

        A_in, C_in, b_in, f_in = cnm.A, cnm.C, cnm.b, cnm.f
        cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                        merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
                        f_in=f_in, rf = None, stride = None, gnb = gnb, 
                        method_deconvolution='oasis', check_nan = True)

        cnm2 = cnm2.fit(images)

        roi_spatial_footprints[z]  = cnm2.A
        roi_temporal_footprints[z] = cnm2.C
        roi_temporal_residuals[z]  = cnm2.YrA
        bg_spatial_footprints[z]   = cnm2.b
        bg_temporal_footprints[z]  = cnm2.f

        os.remove(fname)

    if use_multiprocessing:
        if backend == 'multiprocessing':
            dview.close()
        else:
            try:
                dview.terminate()
            except:
                dview.shutdown()
        cm.stop_server()

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def find_rois_suite2p(video, video_path, params, mc_borders=None, use_multiprocessing=True):
    full_video_path = video_path

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

        video_path = os.path.join(directory, fname)

        h5f = h5py.File(video_path, 'w')
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
            'h5py': video_path, # a single h5 file path
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
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = tifffile.imread(video_path)
            
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]
            
        if i == 0:
            final_video = video
        else:
            final_video = np.concatenate([final_video, video], axis=0)

    filtered_out_rois = []
    for z in range(final_video.shape[1]):
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
                estimate_components_quality_auto(final_video[:, z, :, :].transpose([1, 2, 0]), roi_spatial_footprints[z], roi_temporal_footprints[z], bg_spatial_footprints[z], bg_temporal_footprints[z], 
                                                 roi_temporal_residuals[z], params['imaging_fps'], params['decay_time'], params['half_size'], (video.shape[-2], video.shape[-1]), 
                                                 dview = None, min_SNR=params['min_snr'], 
                                                 r_values_min = params['min_spatial_corr'], use_cnn = params['use_cnn'], 
                                                 thresh_cnn_lowest = params['cnn_threshold'], gSig_range=[ (i, i) for i in range(max(1, params['half_size']-5), params['half_size']+5) ])

        filtered_out_rois.append(list(idx_components_bad))

        if isinstance(roi_spatial_footprints[z], scipy.sparse.coo_matrix):
            f = roi_spatial_footprints[z].toarray()
        else:
            f = roi_spatial_footprints[z]

        for i in range(f.shape[-1]):
            area = np.sum(f[:, i] > 0)
            if (area < params['min_area'] or area > params['max_area']) and i not in filtered_out_rois[-1]:
                filtered_out_rois[-1].append(i)

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
