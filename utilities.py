import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.stats
from skimage.feature import peak_local_max

from skimage.morphology import *
from skimage.filters import rank
from skimage.external.tifffile import imread, imsave

import caiman as cm
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise, MotionCorrect
from imimposemin import imimposemin
import math

import os
import glob
import sys

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

def sharpen(image):
    print(np.amin(image), np.amax(image))
    kernel      = np.ones((3,3))*(-1)
    kernel[1,1] = 8
    Lap         = ndi.filters.convolve(image, kernel)
    Laps        = Lap*100.0/np.amax(Lap) #Sharpening factor!
    # Laps += np.amin(Laps)

    A           = image + Laps

    print(np.amin(A), np.amax(A))

    A = abs(A)

    A           *= 255.0/np.amax(A)

    print(np.amin(A), np.amax(A))

    A_cv2       = A
    A_cv2       = A_cv2.astype(np.uint8)

    tile_s0     = 8
    tile_s1     = 8

    clahe       = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    A_cv2       = clahe.apply(A_cv2)

    return A_cv2

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

def motion_correct(video, video_path, max_shift, patch_stride, patch_overlap, progress_signal=None, thread=None):
    full_video_path = video_path

    directory = os.path.dirname(full_video_path)
    filename  = os.path.basename(full_video_path)

    mc_videos_list = []

    if thread is not None and thread.running == False:
        return [None]*2

    # Create the cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    if thread is not None and thread.running == False:
        cm.stop_server()
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        return [None]*2

    if progress_signal:
        # send an update signal to the GUI
        percent_complete = int(100.0*float(0.1)/video.shape[1])
        progress_signal.emit(percent_complete)

    for z in range(video.shape[1]):
        print(z)
        video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_z_{}.tif".format(z))
        imsave(video_path, video[:, z, :, :])

        a = imread(video_path)
        # print(a.shape)

        # --- PARAMETERS --- #

        params_movie = {'fname': video_path,
                        'max_shifts': (max_shift, max_shift),  # maximum allow rigid shift (2,2)
                        'niter_rig': 2,
                        'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
                        'num_splits_to_process_rig': None,  # if none all the splits are processed and the movie is saved
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

        if thread is not None and thread.running == False:
            cm.stop_server()
            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            return None

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + (1/3))/video.shape[1])
            progress_signal.emit(percent_complete)

        # print(mc.fname_tot_rig)

        # Load rigid motion corrected movie
        m_rig = cm.load(mc.fname_tot_rig)

        # --- ELASTIC MOTION CORRECTION --- #

        # Do elastic motion correction
        mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig, show_template=False)

        if thread is not None and thread.running == False:
            cm.stop_server()
            log_files = glob.glob('Yr*_LOG_*')
            for log_file in log_files:
                os.remove(log_file)

            return None

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + (2/3))/video.shape[1])
            progress_signal.emit(percent_complete)

        # # Save elastic shift border
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        # np.savez(mc.fname_tot_els + "_bord_px_els.npz", bord_px_els)

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

        # print(m_els.shape)
        images = m_els[:, idx_xy[0], idx_xy[1]].copy()
        images += add_to_movie

        # remove_init = 0 # if you need to remove frames from the beginning of each file
        # downsample_factor = 1 
        # base_name = fname.split('/')[-1][:-4]
        # name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
        #     1, 1, downsample_factor), remove_init=remove_init, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=0)
        # name_new.sort()

        # # If multiple files were saved in C format, put them together in a single large file 
        # if len(name_new) > 1:
        #     fname_new = cm.save_memmap_join(
        #         name_new, base_name='Yr', n_chunks=20, dview=dview)
        # else:
        #     print('One file only, not saving!')
        #     fname_new = name_new[0]

        # print("Final movie saved in: {}.".format(fname_new))

        # Yr, dims, T = cm.load_memmap(fname_new)
        # d1, d2 = dims
        # images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # Y = np.reshape(Yr, dims + (T,), order='F')

        mc_videos_list.append(np.nan_to_num(images))

        os.remove(video_path)

        # out = np.zeros(m_els.shape)
        # out[:] = m_els[:]

        # out = np.nan_to_num(out)

        if thread is not None and thread.running == False:
            return None
            

        if progress_signal:
            # send an update signal to the GUI
            percent_complete = int(100.0*float(z + 1)/video.shape[1])
            progress_signal.emit(percent_complete)


    max_height = max([ a.shape[1] for a in mc_videos_list ])
    max_width  = max([ a.shape[2] for a in mc_videos_list ])

    # print("Max", max_height, max_width)

    mc_video = np.zeros((video.shape[0], video.shape[1], max_height, max_width))

    # print(mc_video.shape)

    offset = ((video.shape[2] - max_height)//2, (video.shape[3] - max_width)//2)

    for z in range(len(mc_videos_list)):
        a = mc_videos_list[z]
        height_pad = max_height - a.shape[1]
        width_pad  = max_width - a.shape[2]

        # print("Pad", height_pad, width_pad)

        height_pad_pre  = height_pad//2
        height_pad_post = height_pad - height_pad_pre

        width_pad_pre  = width_pad//2
        width_pad_post = width_pad - width_pad_pre

        b = np.pad(a, ((0, 0), (height_pad_pre, height_pad_post), (width_pad_pre, width_pad_post)), 'constant')
        # print(b.shape)

        mc_video[:, z, :, :] = b

        # print(np.amax(mc_video), np.amin(mc_video))

    cm.stop_server()
    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    if thread is not None and thread.running == False:
        return None

    return mc_video

def calculate_adjusted_image(normalized_image, contrast, gamma):
    return adjust_gamma(adjust_contrast(normalized_image, contrast), gamma)/255.0

def calculate_background_mask(adjusted_image, background_threshold):
    return adjusted_image < background_threshold/255.0

def calculate_equalized_image(adjusted_image, background_mask, window_size):
    new_image_10 = order_statistic(adjusted_image, 0.1, int(window_size))
    new_image_90 = order_statistic(adjusted_image, 0.9, int(window_size))

    image_difference = adjusted_image - new_image_10
    image_difference[image_difference < 0] = 0

    image_range = new_image_90 - new_image_10
    image_range[image_range <= 0] = 1e-6

    equalized_image = rescale_0_1(image_difference/image_range)

    equalized_image[equalized_image < 0] = 0
    equalized_image[equalized_image > 1] = 1

    equalized_image[background_mask] = 0

    return 1.0 - equalized_image

def calculate_soma_threshold_image(equalized_image, soma_threshold):
    nuclei_image = equalized_image.copy()

    soma_mask = local_maxima(h_maxima(nuclei_image, soma_threshold/255.0, selem=square(3)), selem=square(3))
    # self.soma_masks[i] = remove_small_objects(self.soma_masks[i].astype(bool), 2, connectivity=2, in_place=True)

    nuclei_image_c = 1 - nuclei_image

    I_mod = imimposemin(nuclei_image_c.astype(float), soma_mask)

    try:
        inf = math.inf
    except:
        inf = float("inf")

    soma_threshold_image = I_mod/np.amax(I_mod)
    soma_threshold_image[soma_threshold_image == -inf] = 0

    return soma_mask, I_mod, soma_threshold_image

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

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            area = cv2.contourArea(c)

            perimeter = cv2.arcLength(c, True)

            if area > 0:
                roi_circs[l-1] = (perimeter**2)/(4*np.pi*area)

            roi_areas[l-1] = area

    return labels, roi_areas, roi_circs

def filter_rois(image, labels, min_area, max_area, min_circ, max_circ, roi_areas, roi_circs, locked_rois=[]):
    filtered_labels = labels.copy()
    filtered_out_rois = []

    for l in np.unique(labels):
        if ((not (min_area <= roi_areas[l-1] <= max_area)) or (not (min_circ <= roi_circs[l-1] <= max_circ)) or l <= 1) and l not in locked_rois:
            mask = labels == l
            
            filtered_labels[mask] = 0
            filtered_out_rois.append(l)
        else:
            mask = labels == l

            a = dilation(mask, disk(1))
            b = erosion(mask, disk(1))

            difference = np.mean(image[a - b]) - np.mean(image[b])

            if (difference != np.nan and difference < 4.0):
                filtered_labels[mask] = 0
                filtered_out_rois.append(l)

    return filtered_labels, filtered_out_rois

def filter_labels(labels, removed_rois):
    if removed_rois is not None:
        if python_version == 3:
            new_labels = labels.copy()
        else:
            new_labels = labels[:]

        for i in range(len(new_labels)):
            for roi in removed_rois[i]:
                new_labels[i][new_labels[i] == roi] = 0

        return new_labels
    else:
        return labels

def get_roi_containing_point(labels, roi_point):
    roi = labels[roi_point[1], roi_point[0]]

    if roi <= 1:
        return None

    return roi

def get_rois_near_point(labels, roi_point, radius):
    mask = np.zeros(labels.shape)

    cv2.circle(mask, roi_point, radius, 1, -1)

    rois = np.unique(labels[mask == 1]).tolist()

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

def calc_activity_of_roi(labels, video, roi, z=0):
    return np.mean(video[:, z, :, :].transpose(1, 2, 0) * ((labels == roi)[:, :, np.newaxis]), axis=(0, 1))

def add_roi_to_overlay(overlay, roi_mask, labels):
    l = np.amax(labels[roi_mask > 0])
    overlay[roi_mask > 0] = colors[l]

def draw_rois(rgb_image, labels, selected_roi, erased_rois, filtered_out_rois, locked_rois, newly_erased_rois=None, roi_overlay=None):
    global colors
    image = rgb_image.copy()

    if roi_overlay is None:
        n_rois = len(np.unique(labels))
        roi_overlay = np.zeros(image.shape).astype(np.uint8)

        for l in np.unique(labels):
            if l <= 1 or l in filtered_out_rois:
                continue

            roi_overlay[labels == l] = colors[l]

    if newly_erased_rois is not None:
        for l in newly_erased_rois:
            mask = labels == l
            roi_overlay[mask] = 0
    elif erased_rois is not None:
        for l in erased_rois:
            mask = labels == l
            roi_overlay[mask] = 0

    final_roi_overlay = image.copy()
    mask = roi_overlay != 0
    final_roi_overlay[mask] = roi_overlay[mask]

    cv2.addWeighted(final_roi_overlay, 0.5, image, 0.5, 0, image)

    if selected_roi is not None:
        mask = labels == selected_roi

        b = erosion(mask, disk(1))

        mask = mask - b

        image[mask] = np.array([0, 255, 0]).astype(np.uint8)

    if locked_rois is not None:
        for l in locked_rois:
            mask = labels == l

            b = erosion(mask, disk(1))

            mask = mask - b

            image[mask] = np.array([255, 255, 0]).astype(np.uint8)

    return image, roi_overlay