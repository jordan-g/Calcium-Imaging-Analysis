from __future__ import division
from param_window import ParamWindow
from preview_window import PreviewWindow
from skimage.morphology import *
import utilities
import time
import json
import os
import sys
import scipy.ndimage as ndi
import scipy.signal
import numpy as np
from skimage.external.tifffile import imread, imsave
from skimage.measure import find_contours, regionprops
from skimage.filters import gaussian
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import exposure
import cv2
import matplotlib.pyplot as plt
import csv
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf

# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    pyqt_version = 4
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    pyqt_version = 5

if sys.version_info[0] < 3:
    python_version = 2
else:
    python_version = 3
import pdb

# set default parameters dictionary
DEFAULT_PARAMS = {'gamma'               : 1.0,
                  'contrast'            : 1.0,
                  'fps'                 : 60,
                  'z'                   : 0,
                  'max_shift'           : 6,
                  'patch_stride'        : 24,
                  'patch_overlap'       : 6,
                  'window_size'         : 7,
                  'background_threshold': 10,
                  'invert_masks'        : False,
                  'soma_threshold'      : 0.8,
                  'min_area'            : 10,
                  'max_area'            : 100,
                  'min_circ'            : 0,
                  'max_circ'            : 2,
                  'imaging_fps'         : 30,
                  'decay_time'          : 0.4,
                  'autoregressive_order': 0,
                  'num_bg_components'   : 2,
                  'merge_threshold'     : 0.85,
                  'num_components'      : 100,
                  'half_size'           : 4,
                  'use_cnn'             : False,
                  'min_snr'             : 1.1,
                  'min_spatial_corr'    : 0.8,
                  'use_cnn'             : False,
                  'cnn_threshold'       : 0.5}

# set filename for saving current parameters
PARAMS_FILENAME = "params.txt"

class Controller():
    def __init__(self):
        # load parameters
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

        # initialize other variables
        self.video          = None  # video that is being previewed
        self.video_path     = None  # path of the video that is being previewed
        self.video_paths    = []    # paths of all videos to process
        self.mean_images    = None  # mean images for all z planes
        self.video_lengths  = []

        # initialize settings variables
        self.motion_correct_all_videos = False # whether to use motion correction when processing videos
        self.use_mc_video              = False # whether to use the motion-corrected video for finding ROIs
        self.mc_current_z              = False # whether to motion-correct only the current z plane
        self.use_multiprocessing       = True # whether to motion-correct only the current z plane
        self.find_new_rois             = True  # whether we need to find new ROIs
        self.mc_rois                   = False # whether found ROIs are based on the motion-corrected video
        self.apply_blur                = False

        # initialize motion correction, ROI finding & ROI filtering variables
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables(reset_rois=True)
        self.reset_roi_filtering_variables(reset_rois=True)

    def reset_motion_correction_variables(self):
        self.mc_video       = None
        self.mc_video_paths = []
        if self.video is not None:
            self.mc_borders = [ None for i in range(self.video.shape[1]) ]
        else:
            self.mc_borders = None

    def reset_roi_finding_variables(self, reset_rois=False):
        if reset_rois:
            self.n_masks = 0

            if self.video is not None:
                self.masks                   = [ [] for i in range(self.video.shape[1]) ]
                self.mask_points             = [ [] for i in range(self.video.shape[1]) ]
                self.roi_spatial_footprints  = [ None for i in range(self.video.shape[1]) ]
                self.roi_temporal_footprints = [ None for i in range(self.video.shape[1]) ]
                self.roi_temporal_residuals  = [ None for i in range(self.video.shape[1]) ]
                self.bg_spatial_footprints   = [ None for i in range(self.video.shape[1]) ]
                self.bg_temporal_footprints  = [ None for i in range(self.video.shape[1]) ]
                self.filtered_out_rois       = [ [] for i in range(self.video.shape[1]) ]
            else:
                self.masks                   = None
                self.mask_points             = None
                self.roi_spatial_footprints  = None
                self.roi_temporal_footprints = None
                self.roi_temporal_residuals  = None
                self.bg_spatial_footprints   = None
                self.bg_temporal_footprints  = None
                self.filtered_out_rois       = None

    def reset_roi_filtering_variables(self, reset_rois=False):
        if reset_rois:
            if self.video is not None:
                self.erased_rois                    = [ [] for i in range(self.video.shape[1]) ]
                self.removed_rois                   = [ [] for i in range(self.video.shape[1]) ]
                self.locked_rois                    = [ [] for i in range(self.video.shape[1]) ]
                self.manual_roi_spatial_footprints  = [ None for i in range(self.video.shape[1]) ]
                self.manual_roi_temporal_footprints = [ None for i in range(self.video.shape[1]) ]
            else:
                self.erased_rois                    = None
                self.removed_rois                   = None
                self.locked_rois                    = None
                self.manual_roi_spatial_footprints  = None
                self.manual_roi_temporal_footprints = None
                self.last_erased_rois               = None

    def import_videos(self, video_paths):
        if self.video_path is None:
            # open the first video for previewing
            success = self.open_video(video_paths[0])
        else:
            success = True

        # if there was an error opening the video, try opening the next one until there is no error
        while not success:
            del video_paths[0]

            if len(video_paths) == 0:
                return

            success = self.open_video(video_paths[0])

        # add the new video paths to the currently loaded video paths
        self.video_paths += video_paths

        for video_path in video_paths:
            video = imread(video_path)
            self.video_lengths.append(video.shape[0])

    def open_video(self, video_path):
        # get the shape of the previously-previewed video, if any
        if self.video is None:
            previous_video_shape = None
        else:
            previous_video_shape = self.video.shape

        # open the video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.tif') or base_name.endswith('.tiff'):
            video = imread(video_path)
        else:
            return False

        if len(video.shape) < 3:
            print("Error: Opened file is not a video -- not enough dimensions.")
            return False

        # there is a bug with z-stack OIR files where the first frame of the first z plane is wrong,
        # so we have to throw out the first frame of the video here
        self.video = video

        # figure out the dynamic range of the video
        max_value = np.amax(self.video)
        if max_value > 2047:
            self.video_max = 4095
        elif max_value > 1023:
            self.video_max = 2047
        elif max_value > 511:
            self.video_max = 1023
        elif max_value > 255:
            self.video_max = 511
        elif max_value > 1:
            self.video_max = 255
        else:
            self.video_max = 1

        print("Video max: {}.".format(self.video_max))

        # set the path to the previewed video
        self.video_path = video_path

        if len(self.video.shape) == 3:
            # add a z dimension
            self.video = self.video[:, np.newaxis, :, :]
            # self.video = self.video[np.newaxis, :, :, :]

        # remove nans
        self.video = np.nan_to_num(self.video).astype(np.float32)

        print("Opened video with shape {}.".format(self.video.shape))

        # # if the video is a different shape than the previous one, get rid of any exising ROI information
        # if previous_video_shape is None or self.video.shape[2] != previous_video_shape[2] or self.video.shape[3] != previous_video_shape[3]:
        #     reset_rois = True
        # else:
        #     reset_rois = False

        # reset the ROI finding & filtering variables
        # self.reset_motion_correction_variables()
        # self.reset_roi_finding_variables(reset_rois=reset_rois)
        # self.reset_roi_filtering_variables(reset_rois=reset_rois)
        if self.mc_borders is None:
            self.mc_borders = [ None for i in range(self.video.shape[1]) ]

        return True

    def open_mc_video(self, video_path):
        self.mc_video = imread(video_path)

        if len(self.mc_video.shape) == 3:
            # add a z dimension
            self.mc_video = self.mc_video[:, np.newaxis, :, :]

        # remove nans
        self.mc_video = np.nan_to_num(self.mc_video).astype(np.float32)

    def save_mc_video(self, save_path):
        if self.mc_video is not None:
            # save the video
            imsave(save_path, self.mc_video)

    def save_rois(self, save_path):
        if self.roi_spatial_footprints[0] is not None:
            # create a dictionary to hold the ROI data
            roi_data = {'roi_spatial_footprints'         : self.roi_spatial_footprints,
                        'roi_temporal_footprints'        : self.roi_temporal_footprints,
                        'roi_temporal_residuals'         : self.roi_temporal_residuals,
                        'bg_spatial_footprints'          : self.bg_spatial_footprints,
                        'bg_temporal_footprints'         : self.bg_temporal_footprints,
                        'manual_roi_spatial_footprints'  : self.manual_roi_spatial_footprints,
                        'manual_roi_temporal_footprints' : self.manual_roi_temporal_footprints,
                        'filtered_out_rois'              : self.filtered_out_rois,
                        'erased_rois'                    : self.erased_rois,
                        'removed_rois'                   : self.removed_rois,
                        'locked_rois'                    : self.locked_rois}

            # save the ROI data
            np.save(save_path, roi_data)

    def load_rois(self, load_path):
        # load the saved ROIs
        roi_data = np.load(load_path)

        # we are loading a dictionary containing an ROI array and other ROI variables

        # extract the dictionary
        roi_data = roi_data[()]

        # make sure the ROI array shape matches the video
        # if np.array(roi_data['roi_spatial_footprints']).shape != self.video.shape[1:]:
        #     print("Error: ROI array shape does not match the video shape.")
        #     return

        # set ROI variables
        self.roi_spatial_footprints         = roi_data['roi_spatial_footprints']
        self.roi_temporal_footprints        = roi_data['roi_temporal_footprints']
        self.roi_temporal_residuals         = roi_data['roi_temporal_residuals']
        self.bg_spatial_footprints          = roi_data['bg_spatial_footprints']
        self.bg_temporal_footprints         = roi_data['bg_temporal_footprints']
        self.manual_roi_spatial_footprints  = roi_data['manual_roi_spatial_footprints']
        self.manual_roi_temporal_footprints = roi_data['manual_roi_temporal_footprints']
        self.filtered_out_rois              = roi_data['filtered_out_rois']
        self.erased_rois                    = roi_data['erased_rois']
        self.removed_rois                   = roi_data['removed_rois']
        self.locked_rois                    = roi_data['locked_rois']

        # A = np.dot(self.roi_spatial_footprints[0].toarray(), self.roi_temporal_footprints[0]).reshape((self.video.shape[2], self.video.shape[3], self.video.shape[0])).transpose((2, 0, 1)).astype(np.uint16)
        # imsave("A.tif", A)

        self.find_new_rois = False
        self.use_mc_video  = False
        self.mc_rois       = False

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths at the indices, in reverse order
            index = indices[i]
            del self.video_paths[index]
            del self.video_lengths[index]

        if len(self.video_paths) == 0:
            # reset variables
            self.video_path    = None
            self.use_mc_video  = False
            self.video         = None
            self.mean_images   = None
            self.find_new_rois = True
        elif 0 in indices:
            # the first video was removed; open the next one for previewing
            self.open_video(self.video_paths[0])

    def calculate_mean_images(self):
        if self.use_mc_video and self.mc_video is not None:
            video = self.mc_video
        else:
            video = self.video

        if self.apply_blur:
            self.mean_images = [ ndi.median_filter(utilities.sharpen(ndi.gaussian_filter(denoise_wavelet(utilities.mean(video, z)/self.video_max)*self.video_max, 1)), 3) for z in range(video.shape[1]) ]
        else:
            self.mean_images = [ denoise_wavelet(utilities.mean(video, z)/self.video_max)*self.video_max for z in range(video.shape[1]) ]

        if self.video.shape[1] > 1:
            # set size of squares whose mean brightness we will calculate
            window_size = 50

            # get the coordinates of the top-left corner of the z=0 plane, ignoring any black borders due to motion correction
            nonzeros = np.nonzero(self.mean_images[0] > 0)
            crop_y   = nonzeros[0][0] + 20
            crop_x   = nonzeros[1][0] + 20

            # crop the z=0 mean image to remove the black borders
            image = self.mean_images[0][crop_y:-crop_y, crop_x:-crop_x]

            # get the mean brightness of squares at each corner of the image
            mean_vals = [ np.mean(image[:window_size, :window_size]), np.mean(image[:window_size, -window_size:]), np.mean(image[-window_size:, :window_size]), np.mean(image[-window_size:, -window_size:]) ]
            
            # find which corner has the lowest brightness -- we will assume that corner contains the background
            bg_brightness_0 = min(mean_vals)
            bg_window_index = mean_vals.index(bg_brightness_0)

            for z in range(1, self.video.shape[1]):
                # get the coordinates of the top-left corner of this z plane, ignoring any black borders due to motion correction
                nonzeros = np.nonzero(self.mean_images[z] > 0)
                crop_y   = nonzeros[0][0] + 20
                crop_x   = nonzeros[1][0] + 20

                # crop this z plane's mean image to remove the black borders
                image = self.mean_images[z][crop_y:-crop_y, crop_x:-crop_x]

                # get the mean brightness of the corner at this z plane
                if bg_window_index == 0:
                    bg_brightness = np.mean(image[:window_size, :window_size])
                elif bg_window_index == 1:
                    bg_brightness = np.mean(image[:window_size, -window_size:])
                elif bg_window_index == 2:
                    bg_brightness = np.mean(image[-window_size:, :window_size])
                else:
                    bg_brightness = np.mean(image[-window_size:, -window_size:])

                # calculate the difference between this brightness and that of the z=0 plane
                difference = int(round(bg_brightness - bg_brightness_0))

                # subtract this difference from this z plane's mean image
                self.mean_images[z] = np.maximum(self.mean_images[z] - difference, 0)

    def set_invert_masks(self, boolean):
        self.params['invert_masks'] = boolean

        # invert the masks
        for i in range(len(self.masks)):
            for j in range(len(self.masks[i])):
                self.masks[i][j] = self.masks[i][j] == False

    def filter_rois(self, z): # TODO: Update this
        if self.use_mc_video and self.mc_video is not None:
            video = self.mc_video
        else:
            video = self.video

        # filter out ROIs and update the removed ROIs
        self.filtered_out_rois[z] = utilities.filter_rois(video[:, z, :, :], self.roi_spatial_footprints[z], self.roi_temporal_footprints[z], self.roi_temporal_residuals[z], self.bg_spatial_footprints[z], self.bg_temporal_footprints[z], self.params)
        
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

    def create_roi(self, start_point, end_point, z): # TODO: Update this
        # find the center of the ROI
        center_point = (int(round((start_point[0] + end_point[0])/2)), int(round((start_point[1] + end_point[1])/2)))
        axis_1 = np.abs(center_point[0] - end_point[0])
        axis_2 = np.abs(center_point[1] - end_point[1])

        # create a mask
        mask = np.zeros((self.video.shape[2], self.video.shape[3])).astype(np.uint8)

        # draw an ellipse on the mask
        cv2.ellipse(mask, center_point, (axis_1, axis_2), 0, 0, 360, 1, -1)

        if self.use_mc_video and self.mc_video is not None:
            video = self.mc_video
        else:
            video = self.video

        # pdb.set_trace()

        trace = utilities.calc_activity_of_roi(mask, video, z)

        mask = mask.T.reshape((self.video.shape[2]*self.video.shape[3], 1))

        if self.manual_roi_spatial_footprints[z] is None:
            self.manual_roi_spatial_footprints[z] = mask
            self.manual_roi_temporal_footprints[z] = trace[np.newaxis, :]
        else:
            self.manual_roi_spatial_footprints[z]  = np.concatenate((self.manual_roi_spatial_footprints[z], mask), axis=1)
            self.manual_roi_temporal_footprints[z] = np.concatenate((self.manual_roi_temporal_footprints[z], trace[np.newaxis, :]), axis=0)

    def create_roi_magic_wand(self, image, point, z):
        if self.use_mc_video and self.mc_video is not None:
            video = self.mc_video
        else:
            video = self.video

        mask = cm.external.cell_magic_wand.cell_magic_wand(image, point, min_radius=2, max_radius=10, roughness=1)

        trace = utilities.calc_activity_of_roi(mask, video, z)

        mask = mask.reshape((self.video.shape[2]*self.video.shape[3], 1))

        if self.manual_roi_spatial_footprints[z] is None:
            self.manual_roi_spatial_footprints[z] = mask
            self.manual_roi_temporal_footprints[z] = trace[np.newaxis, :]
        else:
            self.manual_roi_spatial_footprints[z]  = np.concatenate((self.manual_roi_spatial_footprints[z], mask), axis=1)
            self.manual_roi_temporal_footprints[z] = np.concatenate((self.manual_roi_temporal_footprints[z], trace[np.newaxis, :]), axis=0)

    def shift_rois(self, start_point, end_point, z): # TODO: Update this
        # get the x and y shift
        y_shift = end_point[1] - start_point[1]
        x_shift = end_point[0] - start_point[0]

        # shift the ROI & ROI overlay arrays
        self.rois[z] = np.roll(self.rois[z], y_shift, axis=0)
        self.rois[z] = np.roll(self.rois[z], x_shift, axis=1)

    def select_rois_near_point(self, roi_point, z, radius=10): # TODO: Update this
        # find out which ROIs to select
        # rois_to_select = utilities.get_rois_near_point(self.roi_spatial_footprints[z], roi_point, radius, self.video.shape[2:])

        _, selected_roi = utilities.get_roi_containing_point(self.roi_spatial_footprints[z], None, roi_point, self.video.shape[2:])

        rois_to_select = []
        if selected_roi is not None:
            rois_to_select.append(selected_roi)

        # print(rois_to_select)

        return rois_to_select

    def erase_roi(self, label, z): # TODO: call roi_unselected() method of the param window
        # update ROI filtering variables
        self.erased_rois[z].append(label)
        # self.last_erased_rois[z].append([label])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

        if label in self.locked_rois[z]:
            index = self.locked_rois[z].index(label)
            del self.locked_rois[z][index]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))
