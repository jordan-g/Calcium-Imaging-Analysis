import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.stats
from skimage.feature import peak_local_max

from skimage.morphology import *
from skimage.filters import rank
from skimage.external.tifffile import imread, imsave

import caiman as cm
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.utils import download_demo
from mahotas.labeled import bwperim

import os
import glob

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

def normalize(image, force=False):
    min_val = np.amin(image)

    image_new = image.copy()

    if min_val < 0:
        image_new -= min_val

    max_val = np.amax(image_new)

    if force:
        return 255*image_new/max_val

    if max_val <= 1:
        return 255*image_new
    elif max_val <= 255:
        return image_new
    else:
        return 255*image_new/max_val

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

def adjust_contrast(image, contrast):
    # print(np.amax(image), np.amin(image))
    table = np.array([i*contrast
        for i in np.arange(0, 256)])
    table[table > 255] = 255

    return cv2.LUT(image, table.astype(np.uint8))

def adjust_gamma(image, gamma):
    # print("Adjusting gamma...")
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    table[table > 255] = 255
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table.astype(np.uint8))

def motion_correct(video, video_path, max_shift, patch_stride, patch_overlap):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    mc_video = None

    # Create the cluster
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    for z in range(video.shape[1]):
        video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_z_{}.tif".format(z))
        imsave(video_path, video[:, z, :, :])

        a = imread(video_path)
        print(a.shape)

        # --- PARAMETERS --- #

        params_movie = {'fname': video_path,
                        'max_shifts': (max_shift, max_shift),  # maximum allow rigid shift (2,2)
                        'niter_rig': 2,
                        'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
                        'num_splits_to_process_rig': 10,  # if none all the splits are processed and the movie is saved
                        'strides': (patch_stride, patch_stride),  # intervals at which patches are laid out for motion correction
                        'overlaps': (patch_overlap, patch_overlap),  # overlap between pathes (size of patch strides+overlaps)
                        'splits_els': 20,  # for parallelization split the movies in  num_splits chuncks across time
                        'num_splits_to_process_els': [None],  # if none all the splits are processed and the movie is saved
                        'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                        'max_deviation_rigid': 10,  # maximum deviation allowed for patch with respect to rigid shift         
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
        mc.motion_correct_rigid(save_movie=True)

        # Load rigid motion corrected movie
        m_rig = cm.load(mc.fname_tot_rig)

        # --- ELASTIC MOTION CORRECTION --- #

        # Do elastic motion correction
        mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig, show_template=False)

        # Save elastic shift border
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        np.savez(mc.fname_tot_els + "_bord_px_els.npz", bord_px_els)

        # Load elastic motion corrected movie
        m_els = cm.load(mc.fname_tot_els)

        # downsample_factor = 1
        # cm.concatenate([m_orig.resize(1, 1, downsample_factor)+offset_mov, m_rig.resize(1, 1, downsample_factor), m_els.resize(
        #     1, 1, downsample_factor)], axis=2).play(fr=60, gain=5, magnification=0.75, offset=0)

        # Crop elastic shifts out of the movie and save
        fnames = [mc.fname_tot_els]
        border_to_0 = bord_px_els
        idx_x=slice(border_to_0,-border_to_0,None)
        idx_y=slice(border_to_0,-border_to_0,None)
        idx_xy=(idx_x,idx_y)
        # idx_xy = None
        add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
        remove_init = 0 # if you need to remove frames from the beginning of each file
        downsample_factor = 1 
        base_name = fname.split('/')[-1][:-4]
        name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
            1, 1, downsample_factor), remove_init=remove_init, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=0)
        name_new.sort()

        # If multiple files were saved in C format, put them together in a single large file 
        if len(name_new) > 1:
            fname_new = cm.save_memmap_join(
                name_new, base_name='Yr', n_chunks=20, dview=dview)
        else:
            print('One file only, not saving!')
            fname_new = name_new[0]

        print("Final movie saved in: {}.".format(fname_new))

        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr, dims + (T,), order='F')

        if mc_video is None:
            mc_video = np.zeros((video.shape[0], video.shape[1], images.shape[1], images.shape[2]))
        mc_video[:, z, :, :] = images

        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        out = np.zeros(m_els.shape)
        out[:] = m_els[:]

        out = np.nan_to_num(out)

    motion_corrected_video_path = os.path.splitext(os.path.basename(full_video_path))[0] + "_mc.npy"
    np.save(motion_corrected_video_path, mc_video)

    return mc_video, motion_corrected_video_path

def apply_watershed(original_image, cells_mask, starting_image):
    if len(original_image.shape) == 2:
        rgb_image = cv2.cvtColor((original_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = (original_image*255).copy()

    labels = watershed(starting_image, label(cells_mask))

    unique_labels = np.unique(labels)

    n = len(unique_labels)

    roi_areas = np.zeros(n)
    roi_circs = np.zeros(n)

    for l in unique_labels:
        if l <= 1:
            continue

        mask = np.zeros(original_image.shape, dtype="uint8")
        mask[labels == l] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)

        roi_areas[l-1] = area

    return labels, roi_areas, roi_circs

def filter_rois(image, labels, min_area, max_area, min_circ, max_circ, roi_areas, roi_circs, locked_rois=[]):
    filtered_labels = labels.copy()
    filtered_out_rois = []

    for l in np.unique(labels):
        mask = labels == l

        perim = bwperim(mask.astype(int), n=4) == 1

        diff = np.mean(image[perim]) - np.mean(image[mask - perim])

        if ((not (min_area <= roi_areas[l-1] <= max_area)) or l <= 1 or (diff != np.nan and diff < 1.0)) and l not in locked_rois:
            filtered_labels[mask] = 0
            filtered_out_rois.append(l)

    return filtered_labels, filtered_out_rois

def get_roi_containing_point(labels, roi_point):
    roi = labels[roi_point[1], roi_point[0]]

    if roi <= 1:
        return None

    return roi

def get_mask_containing_point(masks, mask_point):
    for i in range(len(masks)):
        mask = masks[i]
        if mask[mask_point[1], mask_point[0]] > 0:
            return mask, i
    return None, -1

def calc_activity_of_roi(labels, video, roi, z=0):
    z_video = video[:, z, :, :]
    mask = (labels == roi)[np.newaxis, :, :].repeat(z_video.shape[0], 0)

    return np.mean(z_video*mask, axis=(1, 2))

def draw_rois(rgb_image, labels, selected_roi, removed_rois, locked_rois, prev_labels=None):
    global colors
    image = rgb_image.copy()

    n_rois = len(np.unique(labels))
    roi_overlay = np.zeros(image.shape).astype(np.uint8)

    for l in np.unique(labels):
        if l <= 1 or l in removed_rois:
            continue

        roi_overlay[labels == l] = colors[l]

    roi_overlay_2 = roi_overlay.copy()

    if selected_roi is not None:
        perim = bwperim((labels == selected_roi).astype(int), n=4) == 1

        roi_overlay_2[perim > 0] = np.array([0, 255, 0]).astype(np.uint8)
    
    if locked_rois is not None:
        for l in locked_rois:
            perim = bwperim((labels == l).astype(int), n=4) == 1

            roi_overlay_2[perim > 0] = np.array([255, 255, 0]).astype(np.uint8)

    final_overlay = image.copy()

    mask = np.sum(roi_overlay_2, axis=-1) > 0

    final_overlay[mask] = roi_overlay_2[mask]

    cv2.addWeighted(final_overlay, 0.5, image, 0.5, 0, image)

    return image, roi_overlay, final_overlay