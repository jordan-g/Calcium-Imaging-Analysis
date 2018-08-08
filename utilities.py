from __future__ import division

import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.stats
from skimage.feature import peak_local_max
import scipy.signal
from skimage.feature import register_translation

import skimage
from skimage.morphology import *
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.filters import rank
from skimage.external.tifffile import imread, imsave
# from skimage.exposure import *

import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]),'CaImAn'))
# print(sys.path)

from imimposemin import imimposemin
import math

import glob

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
import pdb
from functools import partial
from scipy.sparse import hstack

from multiprocessing import Pool

# from Watershed import Watershed

if sys.version_info[0] < 3:
    python_version = 2
else:
    python_version = 3

colors = [ np.random.randint(50, 255, size=3) for i in range(100000) ]

def play_movie(movie):
    t = movie.shape[-1]

    frame_counter = 0
    while True:
        cv2.imshow('Movie', cv2.resize(movie[:, :, frame_counter]/255, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        frame_counter += 1
        if frame_counter == t:
            frame_counter = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def normalize(image, video_max=255):
    new_image = 255.0*image/video_max
    new_image[new_image > 255.0] = 255.0
    return new_image.astype(np.uint8)

def order_statistic(image, percentile, window_size):
    order = int(np.floor(percentile*window_size**2))
    return ndi.rank_filter(image, order, size=(window_size, window_size))

def rescale_0_1(image):
    (h, w) = image.shape
    n_pixels = h*w
    S = np.reshape(image, (1, n_pixels))

    percent = 5.0/10000
    min_percentile = int(np.maximum(np.floor(n_pixels*percent), 1))
    max_percentile = n_pixels - min_percentile

    S = np.sort(S)

    denominator = S[0, max_percentile] - S[0, min_percentile]

    if abs(denominator) == 0:
        denominator = 1e-6

    return (image - S[0, min_percentile])/denominator

def calculate_local_correlations(movie):
    # (t, h, w) = movie.shape

    # cross_correlations = np.zeros((h, w, 2, 2))

    # index_range = [-1, 1]

    # for i in range(h):
    #     for j in range(w):
    #         for k in range(2):
    #             for l in range(2):
    #                 ind_1 = index_range[k]
    #                 ind_2 = index_range[l]
    #                 cross_correlations[i, j, ind_1, ind_2] = scipy.stats.pearsonr(movie[:, i, j], movie[:, min(max(i+ind_1, 0), h-1), min(max(j+ind_2, 0), w-1)])[0]
    
    # correlations = np.mean(np.mean(cross_correlations, axis=-1), axis=-1)

    # return correlations
    return cm.local_correlations(movie, swap_dim=False)

def mean(movie, z=0):
    return np.mean(movie[:, z, :, :], axis=0)
    # return np.sum(movie[:, z, :, :], axis=0)

def correlation(movie, z=0):
    return np.abs(cm.local_correlations_fft(movie[:, z, :, :], swap_dim=False))

def std(movie):
    return np.median(movie, axis=0)

# def adjust_contrast(image, contrast):
#     print("Adjusting contrast...")
#     new_image = contrast*image
#     new_image[new_image > 255] = 255
#     return new_image

# def adjust_gamma(image, gamma):
#     print("Adjusting gamma...")
#     new_image = 255*(image/255.0)**(1.0/gamma)
#     new_image[new_image > 255] = 255
#     return new_image

def sharpen(image):
    kernel      = np.ones((3,3))*(-1)
    kernel[1,1] = 8
    Lap         = ndi.filters.convolve(image, kernel)
    Laps        = Lap*100.0/np.amax(Lap) #Sharpening factor!
    # Laps += np.amin(Laps)

    A           = image + Laps

    A = abs(A)

    A           *= 255.0/np.amax(A)

    A_cv2       = A
    A_cv2       = A_cv2.astype(np.uint8)

    tile_s0     = 8
    tile_s1     = 8

    clahe       = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    A_cv2       = clahe.apply(A_cv2)

    return A_cv2

def adjust_contrast(image, contrast):
    return image*contrast

def adjust_gamma(image, gamma):
    return skimage.exposure.adjust_gamma(image, gamma)
    
def motion_correct_multiple_videos(video_paths, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None, use_multiprocessing=True):
    video_lengths = []
    
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = imread(video_path)
            
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]
        
        video_lengths.append(video.shape[0])
            
        if i == 0:
            final_video = video
        else:
            final_video = np.concatenate([final_video, video], axis=0)
    
    final_video_path = "video_temp.tif"

    # imsave(final_video_path, final_video)

    # print(video_lengths)

    mc_video, mc_borders = motion_correct(final_video, final_video_path, max_shift, patch_stride, patch_overlap, progress_signal=progress_signal, thread=thread, use_multiprocessing=use_multiprocessing)
    
    # print(mc_video.shape)

    if mc_video.shape[0] == 1:
        return [], []
    
    mc_videos = []
    
    for i in range(len(video_paths)):
        if i == 0:
            mc_videos.append(mc_video[:video_lengths[0]])
        else:
            mc_videos.append(mc_video[np.sum(video_lengths[:i]):np.sum(video_lengths[:i]) + video_lengths[i]])

    # os.remove(final_video_path)

    # print([ video.shape for video in mc_videos ])
            
    return mc_videos, mc_borders

def motion_correct(video, video_path, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None, use_multiprocessing=True):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    if thread is not None and thread.running == False:
        return np.zeros(1), [ None for z in z_range ]

    def check_if_cancelled():
        if thread is not None and thread.running == False:
            if use_multiprocessing:
                cm.stop_server()

            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            return np.zeros(1), [ None for z in z_range ]

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

    check_if_cancelled()

    z_range = list(range(video.shape[1]))

    if progress_signal:
        # send an update signal to the GUI
        percent_complete = int(100.0*float(0.1)/len(z_range))
        progress_signal.emit(percent_complete)

    mc_video = video.copy()

    mc_borders = [ None for z in z_range ]

    counter = 0

    for z in z_range:
        video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_z_{}_temp.tif".format(z))
        imsave(video_path, video[:, z, :, :])

        mc_video[:, z, :, :] *= 0

        a = imread(video_path)

        # --- PARAMETERS --- #

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
            percent_complete = int(100.0*float(counter + (1/3))/len(z_range))
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
            percent_complete = int(100.0*float(counter + (2/3))/len(z_range))
            progress_signal.emit(percent_complete)

        # # Save elastic shift border
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)  
        # np.savez(mc.fname_tot_els + "_bord_px_els.npz", bord_px_els)

        fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
        border_to_0 = bord_px_els     # number of pixels to exclude
        fname_new = cm.save_memmap(fnames, base_name='memmap_', order = 'C',
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
            percent_complete = int(100.0*float(counter + 1)/len(z_range))
            progress_signal.emit(percent_complete)

        try:
            os.remove(mc.fname_tot_rig)
            os.remove(mc.fname_tot_els)
            os.remove(video_path)
        except:
            pass

        counter += 1

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

def calculate_adjusted_image(image, contrast, gamma):
    return adjust_gamma(adjust_contrast(image, contrast), gamma)

def calculate_background_mask(adjusted_image, background_threshold, video_max):
    return dilation(adjusted_image*255.0/video_max < background_threshold, disk(1))

def calculate_equalized_image(adjusted_image, background_mask, window_size, video_max):
    image = adjusted_image/np.amax(adjusted_image)

    # print(np.amax(image), image.dtype)
    new_image_10 = order_statistic(image, 0.1, int(window_size))
    new_image_90 = order_statistic(image, 0.9, int(window_size))

    # print(new_image_10)
    # print(new_image_90)

    image_difference = image - new_image_10
    image_difference[image_difference < 0] = 0

    image_range = new_image_90 - new_image_10
    image_range[image_range <= 0] = 1e-6

    equalized_image = rescale_0_1(image_difference/image_range)

    # print(equalized_image)

    equalized_image[equalized_image < 0.1] = 0
    equalized_image[equalized_image > 1] = 1

    equalized_image[background_mask] = 0

    equalized_image = (1.0 - equalized_image)

    return equalized_image*video_max

def calculate_soma_threshold_image(equalized_image, soma_threshold, video_max):
    nuclei_image = equalized_image/video_max

    nuclei_image[nuclei_image < 1] = 0

    nuclei_image = remove_small_objects(nuclei_image.astype(bool), 2, connectivity=2, in_place=True).astype(float)

    soma_mask = local_maxima(h_maxima(nuclei_image, soma_threshold/255.0, selem=square(3)), selem=square(3))
    soma_mask = remove_small_objects(soma_mask.astype(bool), 2, connectivity=2, in_place=True)
    # self.soma_masks[i] = remove_small_objects(self.soma_masks[i].astype(bool), 2, connectivity=2, in_place=True)

    # print(soma_mask)
    # print(np.amax(soma_mask))

    # cv2.imshow('image',soma_mask.astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    nuclei_image_c = 1 - nuclei_image

    I_mod = imimposemin(nuclei_image_c.astype(float), soma_mask)

    try:
        inf = math.inf
    except:
        inf = float("inf")

    soma_threshold_image = I_mod/np.amax(I_mod)
    soma_threshold_image[soma_threshold_image == -inf] = 0

    return soma_mask, I_mod, soma_threshold_image

def calculate_roi_properties(rois, mean_image):
    unique_rois = np.unique(rois)

    n = len(unique_rois)

    roi_areas = np.zeros(n)
    roi_circs = np.zeros(n)

    for i in range(len(unique_rois)):
        l = unique_rois[i]

        if l <= 1:
            continue

        mask = rois == l

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            area = cv2.contourArea(c)

            perimeter = cv2.arcLength(c, True)

            if area > 0:
                roi_circs[i] = (perimeter**2)/(4*np.pi*area)

            roi_areas[i] = area

    return roi_areas, roi_circs

def calculate_translation(image_1, image_2):
    # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = image_1.astype('float')
   im2_gray = image_2.astype('float')

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   correlation = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

   return np.unravel_index(np.argmax(correlation), correlation.shape)

def calculate_centroids_and_traces(rois, video):
    roi_nums = np.unique(rois).tolist()

    # print(roi_nums)

    # remove ROI #0 (this is the background)
    try:
        index = roi_nums.index(0)
        del roi_nums[index]
    except:
        pass

    n_rois = len(roi_nums)

    centroids = np.zeros((n_rois, 2))
    traces = np.zeros((video.shape[0], n_rois))

    # traces[1:, 0] = np.arange(video.shape[0])

    for i in range(n_rois):
        roi = roi_nums[i]
        mask = rois == roi

        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            contour = contours[0]

            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            centroids[i] = [center_x, center_y]

        # traces[0, i+1]  = roi
        # a = np.ma.array(video, mask=np.repeat(np.invert(mask)[np.newaxis, :, :], video.shape[0], axis=0))
        # traces[1:, i+1] = np.ma.average(a, axis=(1, 2))


        coords = np.nonzero(mask)
        min_y = min(coords[0])
        max_y = max(coords[0])
        min_x = min(coords[1])
        max_x = max(coords[1])

        mask = mask[min_y:max_y+1, min_x:max_x+1]

        mask[mask == 0] = np.nan
        # trace_sum = np.sum(video*mask[np.newaxis, :, :], axis=(1, 2))
        # count = np.count_nonzero(mask)
        # traces[1:, i+1] = trace_sum/count
        traces[:, i] = np.nanmean(video[:, min_y:max_y+1, min_x:max_x+1]*mask[np.newaxis, :, :], axis=(1, 2))
        # traces[1:, i+1] = np.nanmean(np.where(mask[np.newaxis, :, :], video, np.nan), axis=(1, 2))

    return centroids, traces

# def find_rois(cells_mask, starting_image, mean_image):
#     # w = Watershed()
#     # rois = w.apply(equalized_image)
#     rois = watershed(starting_image, label(cells_mask))

#     roi_areas, roi_circs = calculate_roi_properties(rois, mean_image)

#     return rois, roi_areas, roi_circs

def find_rois_multiple_videos(video_paths, params, use_mc_video=True, masks=None, background_mask=None, mc_borders=None, progress_signal=None, thread=None, use_multiprocessing=True):
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = imread(video_path)
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]

        if i == 0:
            final_video = video.copy()
        else:
            final_video = np.concatenate([final_video, video], axis=0)

    final_video_path = "video_concatenated.tif"
    imsave(final_video_path, final_video)

    roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = find_rois(final_video, final_video_path, params, masks=masks, background_mask=background_mask, mc_borders=mc_borders, progress_signal=progress_signal, thread=thread, use_multiprocessing=use_multiprocessing)

    # A = np.dot(roi_spatial_footprints[0].toarray(), roi_temporal_footprints[0]).reshape((final_video.shape[2], final_video.shape[3], final_video.shape[0])).transpose((2, 0, 1)).astype(np.uint16)
    # imsave("A.tif", A)

    os.remove(final_video_path)

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def calculate_temporal_components(video_paths, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints):
    for i in range(len(video_paths)):
        video_path = video_paths[i]

        video = imread(video_path)
        if len(video.shape) == 3:
            # add a z dimension
            video = video[:, np.newaxis, :, :]

        if i == 0:
            final_video = video.copy()
        else:
            final_video = np.concatenate([final_video, video], axis=0)

    roi_temporal_footprints = [ None for i in range(final_video.shape[1]) ]
    roi_temporal_residuals = [ None for i in range(final_video.shape[1]) ]
    bg_temporal_footprints = [ None for i in range(final_video.shape[1]) ]

    for z in range(final_video.shape[1]):
        print(z)
        final_video_path = "video_concatenated_z_{}.tif".format(z)
        imsave(final_video_path, final_video[:, z, :, :])

        fname_new = cm.save_memmap([final_video_path], base_name='memmap_', order='C') # exclude borders

        # # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)
        # d1, d2 = dims
        # images = np.reshape(Yr.T, [T] + list(dims), order='F').transpose((1, 2, 0)).reshape((d1*d2, T))

        images = final_video[:, z, :, :].transpose((1, 2, 0)).reshape((final_video.shape[2]*final_video.shape[3], final_video.shape[0]))

        # print(images.shape)
        # print(roi_spatial_footprints[z].shape)
        # print(bg_spatial_footprints[z].shape)
        Cin = np.zeros((roi_spatial_footprints[z].shape[1], final_video.shape[0]))
        fin = np.zeros((bg_spatial_footprints[z].shape[1], final_video.shape[0]))

        print(images.shape)

        Yr, sn, g, psx = preprocess_data(Yr, dview=None)

        roi_temporal_footprints[z], _, _, bg_temporal_footprints[z], _, _, _, _, _, roi_temporal_residuals[z], _ = update_temporal_components(images, roi_spatial_footprints[z], bg_spatial_footprints[z], Cin, fin, bl=None, c1=None, g=g, sn=sn, nb=2, ITER=2, block_size=5000, num_blocks_per_run=20, debug=False, dview=None, p=0, method='cvx')

        # for i in range(10):
        #     roi_temporal_footprints[z], _, _, bg_temporal_footprints[z], _, _, _, _, _, roi_temporal_residuals[z], _ = update_temporal_components(images, roi_spatial_footprints[z], bg_spatial_footprints[z], roi_temporal_footprints[z], bg_temporal_footprints[z], bl=None, c1=None, g=None, sn=None, nb=2, ITER=2000, block_size=5000, num_blocks_per_run=20, debug=False, dview=None, p=0)

        # os.remove(final_video_path)

        print(roi_temporal_footprints[z].shape)

    return roi_temporal_footprints, roi_temporal_residuals, bg_temporal_footprints

def do_cnmf(video, params, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, use_multiprocessing=True):
    if use_multiprocessing:
        if os.name == 'nt':
            backend = 'multiprocessing'
        else:
            backend = 'ipyparallel'

        cm.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
    else:
        dview = None

    video_path = "video_temp.tif"
    imsave(video_path, video)

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
                    only_init_patch = True, skip_refinement=False, gnb = gnb, border_pix = border_pix, ssub=1, ssub_B=1, tsub=1, Ain=roi_spatial_footprints, Cin=roi_temporal_footprints, b_in=bg_spatial_footprints, f_in=bg_temporal_footprints, do_merge=False)

    cnm = cnm.fit(images)

    C, A, b, f, S, bl, c1, sn, g, YrA, lam = update_temporal_components(Yr, roi_spatial_footprints, bg_spatial_footprints, cnm.C, cnm.f, bl=None, c1=None, g=None, sn=None, nb=1, ITER=2, block_size=5000, num_blocks_per_run=20, debug=False, dview=None, p=0)

    roi_spatial_footprints  = A
    roi_temporal_footprints = C
    roi_temporal_residuals  = YrA
    bg_spatial_footprints   = b
    bg_temporal_footprints  = f

    # pdb.set_trace()

    if use_multiprocessing:
        dview.terminate()
        cm.stop_server()

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

def find_rois(video, video_path, params, masks=None, background_mask=None, mc_borders=None, progress_signal=None, thread=None, use_multiprocessing=True):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
    roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
    roi_temporal_residuals  = [ None for i in range(video.shape[1]) ]
    bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
    bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]

    if thread is not None and thread.running == False:
        if use_multiprocessing:
            cm.stop_server()
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

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
        imsave(video_path, video[:, z, :, :])

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

        if mc_borders[z] is not None:
            border_pix = mc_borders[z]
        else:
            border_pix = 0

        fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C') # exclude borders

        # now load the file
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')


        cnm = cnmf.CNMF(n_processes=8, k=K, gSig=gSig, merge_thresh= merge_thresh, 
                        p = p,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf, rolling_sum=rolling_sum,
                        only_init_patch = False, gnb = gnb, border_pix = border_pix, ssub=1, ssub_B=1, tsub=1)

        cnm = cnm.fit(images)

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z+0.5)/video.shape[1])
            progress_signal.emit(percent_complete)

        if thread is not None and thread.running == False:
            if use_multiprocessing:
                cm.stop_server()
            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
            roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
            roi_temporal_residuals  = [ None for i in range(video.shape[1]) ]
            bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
            bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]
            return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z+1)/video.shape[1])
            progress_signal.emit(percent_complete)

        roi_spatial_footprints[z]  = cnm.A
        roi_temporal_footprints[z] = cnm.C
        roi_temporal_residuals[z]  = cnm.YrA
        bg_spatial_footprints[z]   = cnm.b
        bg_temporal_footprints[z]  = cnm.f

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

def filter_rois(video, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, params):
    # filtered_out_rois = [ None for i in range(video.shape[1]) ]

    # pdb.set_trace()

    # for z in range(video.shape[1]):
    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(video.transpose([1, 2, 0]), roi_spatial_footprints, roi_temporal_footprints, bg_spatial_footprints, bg_temporal_footprints, 
                                             roi_temporal_residuals, params['imaging_fps'], params['decay_time'], params['half_size'], (video.shape[-2], video.shape[-1]), 
                                             dview = None, min_SNR=params['min_snr'], 
                                             r_values_min = params['min_spatial_corr'], use_cnn = params['use_cnn'], 
                                             thresh_cnn_lowest = params['cnn_threshold'], gSig_range=[ (i, i) for i in range(max(1, params['half_size']-5), params['half_size']+5) ])

    filtered_out_rois = list(idx_components_bad)

    for i in range(roi_spatial_footprints.shape[-1]):
        area = np.sum(roi_spatial_footprints[:, i] > 0)
        if (area < params['min_area'] or area > params['max_area']) and i not in filtered_out_rois:
            filtered_out_rois.append(i)

    return filtered_out_rois

def find_rois_refine(video, video_path, params, masks=None, background_mask=None, mc_borders=None, roi_spatial_footprints=None, roi_temporal_footprints=None, roi_temporal_residuals=None, bg_spatial_footprints=None, bg_temporal_footprints=None, progress_signal=None, thread=None):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
    roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
    roi_temporal_residuals  = [ None for i in range(video.shape[1]) ]
    bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
    bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]

    try:
        if thread is not None and thread.running == False:
            # cm.stop_server()
            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

        # Create the cluster
        # cm.stop_server()
        # c, dview, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', n_processes=8, single_thread=False)

        for z in range(video.shape[1]):
            # if masks is not None and len(masks) > 0 and masks[0] is not None:
            #     masks = np.array(masks[z])
            #     if params["invert_masks"]:
            #         mask = np.prod(masks, axis=0).astype(bool)
            #     else:
            #         mask = np.sum(masks, axis=0).astype(bool)

            #     if background_mask is not None:
            #         final_mask = (background_mask + mask).astype(bool)
            #     else:
            #         final_mask = mask.astype(bool)
            # elif background_mask is not None:
            #     final_mask = background_mask.astype(bool)
            # else:
            #     final_mask = None

            # if final_mask is not None:
            #     video[:, z, final_mask] = 0

            fname = os.path.splitext(filename)[0] + "_masked_z_{}.tif".format(z)

            video_path = os.path.join(directory, fname)
            imsave(video_path, video[:, z, :, :])

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

            if mc_borders[z] is not None:
                border_pix = mc_borders[z]
            else:
                border_pix = 0

            fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C') # exclude borders

            # now load the file
            Yr, dims, T = cm.load_memmap(fname_new)
            d1, d2 = dims
            images = np.reshape(Yr.T, [T] + list(dims), order='F')

            A_in, C_in, b_in, f_in = roi_spatial_footprints[z][:,idx_components], roi_temporal_footprints[z][idx_components], bg_spatial_footprints[z], bg_temporal_footprints[z]
            cnm = cnmf.CNMF(n_processes=8, k=A_in.shape[-1], gSig=gSig, p=p, dview=None,
                            merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
                            f_in=f_in, rf = None, stride = None, gnb = gnb, 
                            method_deconvolution='oasis', check_nan = True)

            cnm = cnm.fit(images)

            if progress_signal:
                # send an update signal to the GUI
                percent_complete = int(100.0*float(z+0.5)/video.shape[1])
                progress_signal.emit(percent_complete)

            if thread is not None and thread.running == False:
                # cm.stop_server()
                log_files = glob.glob('Yr*_LOG_*')
                for log_file in log_files:
                    os.remove(log_file)

                roi_spatial_footprints  = [ None for i in range(video.shape[1]) ]
                roi_temporal_footprints = [ None for i in range(video.shape[1]) ]
                bg_spatial_footprints   = [ None for i in range(video.shape[1]) ]
                bg_temporal_footprints  = [ None for i in range(video.shape[1]) ]
                return roi_spatial_footprints, roi_temporal_footprints, bg_spatial_footprints, bg_temporal_footprints
                
            if progress_signal:
                # send an update signal to the GUI
                percent_complete = int(100.0*float(z+1)/video.shape[1])
                progress_signal.emit(percent_complete)

            roi_spatial_footprints[z]  = cnm.A
            roi_temporal_footprints[z] = cnm.C
            roi_temporal_residuals[z]  = cnm.YrA
            bg_spatial_footprints[z]   = cnm.b
            bg_temporal_footprints[z]  = cnm.f

        cm.stop_server()
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
    except:
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

    return roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints

# def filter_rois(image, rois, min_area, max_area, min_circ, max_circ, roi_areas, roi_circs, locked_rois=[]):
#     filtered_rois = rois.copy()
#     filtered_out_rois = []

#     for l in np.unique(rois):
#         if ((not (min_area <= roi_areas[l-1] <= max_area)) or (not (min_circ <= roi_circs[l-1] <= max_circ)) or l <= 1) and l not in locked_rois:
#             mask = rois == l
            
#             filtered_rois[mask] = 0
#             filtered_out_rois.append(l)

#     return filtered_rois, filtered_out_rois

def remove_rois(rois, rois_to_remove):
    if rois_to_remove is not None:
        if python_version == 3:
            new_rois = rois.copy()
        else:
            new_rois = rois[:]

        for i in range(len(new_rois)):
            for roi in rois_to_remove[i]:
                new_rois[i][new_rois[i] == roi] = 0

        return new_rois
    else:
        return rois

def get_roi_containing_point(rois, manual_rois, roi_point, video_shape):
    flattened_point = roi_point[0]*video_shape[0] + roi_point[1]

    try:
        rois = rois.toarray()
    except:
        pass
    # if isinstance(rois, scipy.sparse.coo_matrix)

    if rois is not None:
        roi = np.argmax(rois[flattened_point, :])
        
        if rois[flattened_point, roi] != 0:
            return False, roi

    if manual_rois is not None:
        roi = np.argmax(manual_rois[flattened_point, :])

        if manual_rois[flattened_point, roi] != 0:
            return True, roi

    return False, None


    # roi = rois[roi_point[1], roi_point[0]]

    # if roi < 1:
    #     return None

def get_rois_near_point(rois, roi_point, radius, video_shape):
    # pdb.set_trace()

    mask = np.zeros(video_shape)

    cv2.circle(mask, roi_point, radius, 1, -1)

    rois_2d = rois.toarray().reshape((video_shape[0], video_shape[1], rois.shape[-1]))

    rois = np.nonzero(np.sum(rois_2d*mask[:, :, np.newaxis], axis=(0, 1)))[0].tolist()

    # rois = np.unique(rois[mask == 1]).tolist()

    return rois

def get_mask_containing_point(masks, mask_point, inverted=False):
    for i in range(len(masks)):
        mask = masks[i]
        if inverted:
            if mask[mask_point[1], mask_point[0]] == 0:
                return mask, i
        else:
            if mask[mask_point[1], mask_point[0]] > 0:
                return mask, i
    return None, -1

def calc_activity_of_roi(mask, video, z):
    time_mask = np.repeat(mask[np.newaxis, :, :], video.shape[0], axis=0)

    # pdb.set_trace()

    # print(mask.shape, vid.shape, vid[mask].shape)
    # print(np.mean(np.multiply(video[0, z, :, :], rois == roi)))
    # print(np.nanmax(np.where(mask, video, np.nan), axis=(0, 1)))
    return np.nanmean(np.where(time_mask, video[:, z, :, :], np.nan), axis=(1, 2))

    # return rois[roi]

def add_roi_to_overlay(overlay, roi_mask, rois):
    l = np.amax(rois[roi_mask > 0])

    mask = rois == l

    b = erosion(mask, disk(1))

    mask = mask ^ b
    
    overlay[mask] = np.array([255, 0, 0]).astype(np.uint8)

def calculate_shift(mean_image_1, mean_image_2):
    nonzeros_1 = np.nonzero(mean_image_1 > 0)
    nonzeros_2 = np.nonzero(mean_image_2 > 0)

    crop_y = max([nonzeros_1[0][0], nonzeros_2[0][0]]) + 20
    crop_x = max([nonzeros_1[1][0], nonzeros_2[1][0]]) + 20

    image_1 = mean_image_1[crop_y:-crop_y, crop_x:-crop_x]
    image_2 = mean_image_2[crop_y:-crop_y, crop_x:-crop_x]

    shift, error, diffphase = register_translation(image_1, image_2)

    # print("shift", shift)

    return int(shift[0]), int(shift[1])

def draw_rois(rgb_image, rois, selected_roi, erased_rois, filtered_out_rois, locked_rois, newly_erased_rois=None, roi_overlay=None):
    image = rgb_image.copy()

    if roi_overlay is None:
        n_rois = len(np.unique(rois))
        roi_overlay = np.zeros(image.shape)

        for l in np.unique(rois):
            if (filtered_out_rois is not None and erased_rois is not None) and (l < 1 or l in filtered_out_rois or l in erased_rois):
                continue

            mask = rois == l

            b = erosion(mask, disk(1))

            mask = mask ^ b

            roi_overlay[mask] = np.array([255, 0, 0]).astype(np.uint8)

    if newly_erased_rois is not None:
        for l in newly_erased_rois:
            mask = rois == l
            roi_overlay[mask] = 0
    elif erased_rois is not None:
        for l in erased_rois:
            mask = rois == l
            roi_overlay[mask] = 0

    final_overlay = image.copy()
    final_overlay[roi_overlay > 0] = roi_overlay[roi_overlay > 0]
    cv2.addWeighted(final_overlay, 0.5, image, 0.5, 0, image)

    if selected_roi is not None:
        mask = rois == selected_roi

        b = erosion(mask, disk(1))

        mask = mask ^ b

        image[mask] = np.array([0, 255, 0]).astype(np.uint8)

    if locked_rois is not None:
        for l in locked_rois:
            mask = rois == l

            b = erosion(mask, disk(1))

            mask = mask ^ b

            image[mask] = np.array([255, 255, 0]).astype(np.uint8)

    return image, roi_overlay
