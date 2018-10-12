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
DEFAULT_PARAMS = {'gamma'                : 1.0,
                  'contrast'             : 1.0,
                  'fps'                  : 60,
                  'z'                    : 0,
                  'max_shift'            : 6,
                  'patch_stride'         : 48,
                  'patch_overlap'        : 24,
                  'window_size'          : 7,
                  'background_threshold' : 10,
                  'invert_masks'         : False,
                  'soma_threshold'       : 0.8,
                  'min_area'             : 10,
                  'max_area'             : 100,
                  'min_circ'             : 0,
                  'max_circ'             : 2,
                  'imaging_fps'          : 30,
                  'decay_time'           : 0.4,
                  'autoregressive_order' : 0,
                  'num_bg_components'    : 2,
                  'merge_threshold'      : 0.8,
                  'num_components'       : 400,
                  'half_size'            : 4,
                  'use_cnn'              : False,
                  'min_snr'              : 1.3,
                  'min_spatial_corr'     : 0.8,
                  'use_cnn'              : False,
                  'cnn_threshold'        : 0.5,
                  'diameter'             : 10,
                  'sampling_rate'        : 3,
                  'connected'            : True,
                  'neuropil_basis_ratio' : 6,
                  'neuropil_radius_ratio': 3,
                  'inner_neuropil_radius': 2,
                  'min_neuropil_pixels'  : 350
                  }

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
        self.video_paths   = [] # paths of all videos to process
        self.video_lengths = [] # lengths (# of frames) of all videos
        self.video_groups  = [] # groups that videos belong to

        # initialize all variables
        self.reset_variables()
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables()
        self.reset_roi_filtering_variables()

    def reset_variables(self):
        self.use_mc_video        = False # whether to use the motion-corrected video for finding ROIs
        self.use_multiprocessing = True  # whether to use multi-processing
        self.find_new_rois       = True  # whether we need to find new ROIs
        self.mc_rois             = False # whether found ROIs are based on the motion-corrected video

    def reset_motion_correction_variables(self):
        self.mc_video_paths = [] # paths of all motion-corrected videos
        self.mc_borders     = [] # borders of all motion-corrected videos

    def reset_roi_finding_variables(self):
        self.roi_spatial_footprints  = {}
        self.roi_temporal_footprints = {}
        self.roi_temporal_residuals  = {}
        self.bg_spatial_footprints   = {}
        self.bg_temporal_footprints  = {}
        self.filtered_out_rois       = {}

    def reset_roi_filtering_variables(self):
        self.discarded_rois = {}
        self.removed_rois   = {}
        self.locked_rois    = {}

    def import_videos(self, video_paths):
        # add the new video paths to the currently loaded video paths
        self.video_paths += video_paths

        # assign a group number to the new videos
        if len(self.video_groups) > 0:
            group_num = np.amax(np.unique(self.video_groups)) + 1
        else:
            group_num = 0

        # store video lengths and group numbers
        for video_path in video_paths:
            video = imread(video_path)
            self.video_lengths.append(video.shape[0])
            self.video_groups.append(group_num)

    def save_rois(self, save_path):
        # create a dictionary to hold the ROI data
        roi_data = {'roi_spatial_footprints' : self.roi_spatial_footprints,
                    'roi_temporal_footprints': self.roi_temporal_footprints,
                    'roi_temporal_residuals' : self.roi_temporal_residuals,
                    'bg_spatial_footprints'  : self.bg_spatial_footprints,
                    'bg_temporal_footprints' : self.bg_temporal_footprints,
                    'filtered_out_rois'      : self.filtered_out_rois,
                    'erased_rois'            : self.discarded_rois,
                    'removed_rois'           : self.removed_rois,
                    'locked_rois'            : self.locked_rois}

        # save the ROI data
        np.save(save_path, roi_data)

    def load_rois(self, load_path):
        # load the saved ROIs
        roi_data = np.load(load_path)

        # extract the dictionary
        roi_data = roi_data[()]

        # set ROI variables
        self.roi_spatial_footprints  = roi_data['roi_spatial_footprints']
        self.roi_temporal_footprints = roi_data['roi_temporal_footprints']
        self.roi_temporal_residuals  = roi_data['roi_temporal_residuals']
        self.bg_spatial_footprints   = roi_data['bg_spatial_footprints']
        self.bg_temporal_footprints  = roi_data['bg_temporal_footprints']
        self.filtered_out_rois       = roi_data['filtered_out_rois']
        self.discarded_rois          = roi_data['erased_rois']
        self.removed_rois            = roi_data['removed_rois']
        self.locked_rois             = roi_data['locked_rois']

        self.find_new_rois = False

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths, lengths and groups at the indices, in reverse order
            index = indices[i]
            del self.video_paths[index]
            del self.video_lengths[index]
            del self.video_groups[index]

            if len(self.mc_video_paths) > 0:
                del self.mc_video_paths[index]

        if len(self.video_paths) == 0:
            # reset variables
            self.reset_variables()

    def calculate_mean_images(self):
        if self.use_mc_video and self.mc_video is not None:
            video = self.mc_video
        else:
            video = self.video

        if self.apply_blur:
            self.mean_images = [ ndi.median_filter(utilities.sharpen(ndi.gaussian_filter(denoise_wavelet(utilities.mean(video, z)/self.video_max)*self.video_max, 1)), 3) for z in range(video.shape[1]) ]
        else:
            self.mean_images = [ (utilities.mean(video, z)/self.video_max)*self.video_max for z in range(video.shape[1]) ]

    def set_invert_masks(self, boolean):
        self.params['invert_masks'] = boolean

        # invert the masks
        for i in range(len(self.masks)):
            for j in range(len(self.masks[i])):
                self.masks[i][j] = self.masks[i][j] == False

    def filter_rois(self): # TODO: Update this
        if self.use_mc_video and self.mc_video is not None:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        # filter out ROIs and update the removed ROIs
        self.filtered_out_rois = utilities.filter_rois(video_paths, self.roi_spatial_footprints, self.roi_temporal_footprints, self.roi_temporal_residuals, self.bg_spatial_footprints, self.bg_temporal_footprints, self.params)
        
        print(self.filtered_out_rois)
        for z in range(self.video.shape[1]):
            self.filtered_out_rois[z] = [ roi for roi in self.filtered_out_rois[z] if roi not in self.locked_rois[z] ]
            self.removed_rois[z] = self.filtered_out_rois[z] + self.discarded_rois[z]

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
        self.discarded_rois[z].append(label)
        # self.last_erased_rois[z].append([label])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.discarded_rois[z]

        if label in self.locked_rois[z]:
            index = self.locked_rois[z].index(label)
            del self.locked_rois[z][index]

    def unerase_roi(self, label, z): # TODO: call roi_unselected() method of the param window
        # update ROI filtering variables
        if label in self.discarded_rois[z]:
            i = self.discarded_rois[z].index(label)
            del self.discarded_rois[z][i]
        elif label in self.filtered_out_rois[z]:
            i = self.filtered_out_rois[z].index(label)
            del self.filtered_out_rois[z][i]

            if label not in self.locked_rois[z]:
                self.locked_rois.append(label)

        self.removed_rois[z] = self.filtered_out_rois[z] + self.discarded_rois[z]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))
