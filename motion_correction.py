from __future__ import division
import numpy as np
import cv2
import sys
import os
import math
import glob
import pdb
from functools import partial

from multiprocessing import Pool
from imimposemin import imimposemin

import scipy.ndimage as ndi
import scipy.stats
import scipy.signal
from scipy.sparse import hstack

import skimage
from skimage.feature import peak_local_max
from skimage.feature import register_translation
from skimage.morphology import *
from skimage.restoration import *
from skimage.filters import rank
from skimage.external.tifffile import imread, imsave

import caiman as cm
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise, MotionCorrect
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour, get_contours
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.base.rois import register_ROIs
from caiman.source_extraction.cnmf.temporal import update_temporal_components
from caiman.source_extraction.cnmf.pre_processing import preprocess_data

def motion_correct_videos(video_paths, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None, use_multiprocessing=True):
    # initialize list to store length (# of frames) of each video
    video_lengths = []
    
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = imread(video_path)
            
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]
        
        # store the video length (# of frames)
        video_lengths.append(video.shape[0])
           
        # add to a video that is concatenated along the time dimension
        if i == 0:
            concatenated_video = video
        else:
            concatenated_video = np.concatenate([concatenated_video, video], axis=0)

    # perform motion correction on the concatenated video
    concatenated_mc_video, mc_borders = motion_correct_video(concatenated_video, max_shift, patch_stride, patch_overlap, progress_signal=progress_signal, thread=thread, use_multiprocessing=use_multiprocessing)

    if concatenated_mc_video is not None:
	    # initialize list to store separated motion-corrected videos
	    mc_videos = []
	    
	    for i in range(len(video_paths)):
	    	# append to motion-corrected video list
	        if i == 0:
	            mc_videos.append(concatenated_mc_video[:video_lengths[0]])
	        else:
	            mc_videos.append(concatenated_mc_video[np.sum(video_lengths[:i]):np.sum(video_lengths[:i]) + video_lengths[i]])
	            
	    return mc_videos, mc_borders
	else:
		return [], []

def motion_correct_video(video, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None, use_multiprocessing=True):
    video_path = None
    dview      = None

    def check_if_cancelled():
    	# check if the user has requested for motion correction to stop
        if thread is not None and thread.running == False:
            if use_multiprocessing:
            	if dview is not None:
		        	if backend == 'multiprocessing':
		        	    dview.close()
		        	else:
		        	    try:
		        	        dview.terminate()
		        	    except:
		        	        dview.shutdown()
	        	cm.stop_server()

            # remove any log files created by CaImAn
            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            # remove any temporary video
            if video_path is not None:
            	os.remove(video_path)

            return None, None

    if use_multiprocessing:
    	# set a different backend depending on whether the code is running on macOS or Windows
    	# (multiprocessing backend has issues on macOS)
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            backend = 'ipyparallel'

        # stop any existing cluster of processes and create a new one
        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    check_if_cancelled()

    # create a range of z values
    z_range = list(range(video.shape[1]))

    if progress_signal is not None:
        # send an update signal to the GUI
        percent_complete = int(100.0*float(0.1)/len(z_range))
        progress_signal.emit(percent_complete)

    # initialize motion corrected video and motion correction borders
	mc_video   = 0*video.copy()
	mc_borders = [ None for z in z_range ]

    for z in z_range:
    	# save the video for this z plane
        video_path = "video_z_{}_temp.tif".format(z)
        imsave(video_path, video[:, z, :, :])
        
        # create parameters dictionary
        params_movie = {'fname': video_path,
                        'max_shifts': (max_shift, max_shift),  # maximum allow rigid shift (2,2)
                        'niter_rig': 1,
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

        check_if_cancelled()

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + (1/3))/len(z_range))
            progress_signal.emit(percent_complete)

        # print(mc.fname_tot_rig)

        # Load rigid motion corrected movie
        # m_rig = cm.load(mc.fname_tot_rig)

        # --- ELASTIC MOTION CORRECTION --- #

        # Do elastic motion correction
        mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig, show_template=False)

        check_if_cancelled()

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + (2/3))/len(z_range))
            progress_signal.emit(percent_complete)

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

        check_if_cancelled()

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + 1)/len(z_range))
            progress_signal.emit(percent_complete)

        try:
            os.remove(mc.fname_tot_rig)
            os.remove(mc.fname_tot_els)
            os.remove(video_path)
        except:
            pass

    check_if_cancelled()

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
