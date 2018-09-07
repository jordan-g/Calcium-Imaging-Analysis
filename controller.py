from __future__ import division
import utilities
import time
import json
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pdb
import motion_correction

import scipy.ndimage as ndi
import scipy.signal

from skimage.morphology import *
from skimage.restoration import *
from skimage.external.tifffile import imread, imsave
from skimage.measure import find_contours, regionprops
from skimage.filters import gaussian
from skimage import exposure

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf

from param_window import ParamWindow
from preview_window import PreviewWindow

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

# check which Python version is being used
if sys.version_info[0] < 3:
    python_version = 2
else:
    python_version = 3

# set default parameters dictionary
DEFAULT_PARAMS = {'max_shift'           : 6,
                  'patch_stride'        : 48,
                  'patch_overlap'       : 24,
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
                  'merge_threshold'     : 0.8,
                  'num_components'      : 400,
                  'half_size'           : 4,
                  'use_cnn'             : False,
                  'min_snr'             : 1.3,
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

        # initialize settings variables
        self.use_motion_correction = False # whether to use motion correction when processing videos
        self.use_multiprocessing   = True  # whether to use multiprocessing

        # initialize all other variables
        self.reset_variables()
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables()
        self.reset_roi_filtering_variables()

    def reset_variables(self):
        self.video_paths   = []   # paths of all videos to process
        self.video_lengths = []   # lengths (# of frames) of all of the videos
        self.num_z_planes  = 0    # number of z planes in loaded videos

    def reset_motion_correction_variables(self):
        self.mc_video_paths = None
        self.mc_borders     = None

    def reset_roi_finding_variables(self):
        if len(self.video_paths) == 0:
            self.roi_spatial_footprints  = None
            self.roi_temporal_footprints = None
            self.roi_temporal_residuals  = None
            self.bg_spatial_footprints   = None
            self.bg_temporal_footprints  = None
            self.filtered_out_rois       = None
        else:
            self.roi_spatial_footprints  = [ None for i in range(self.num_z_planes) ]
            self.roi_temporal_footprints = [ None for i in range(self.num_z_planes) ]
            self.roi_temporal_residuals  = [ None for i in range(self.num_z_planes) ]
            self.bg_spatial_footprints   = [ None for i in range(self.num_z_planes) ]
            self.bg_temporal_footprints  = [ None for i in range(self.num_z_planes) ]

    def reset_roi_filtering_variables(self):
        if len(self.video_paths) == 0:
            self.filtered_out_rois = None
            self.erased_rois       = None
            self.removed_rois      = None
            self.locked_rois       = None
        else:
            self.filtered_out_rois = [ [] for i in range(self.num_z_planes) ]
            self.erased_rois       = [ [] for i in range(self.num_z_planes) ]
            self.removed_rois      = [ [] for i in range(self.num_z_planes) ]
            self.locked_rois       = [ [] for i in range(self.num_z_planes) ]

    def import_videos(self, video_paths):
        # add the new video paths to the currently loaded video paths
        self.video_paths += video_paths

        # get lengths (# of frames) of each of the new videos
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            video = imread(video_path)
            self.video_lengths.append(video.shape[0])

            if i == 0:
                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]

                # set number of z planes
                self.num_z_planes = video.shape[1]

    def save_rois(self, save_path):
        if self.roi_spatial_footprints is not None and self.roi_spatial_footprints[0] is not None:
            # create a dictionary to hold the ROI data
            roi_data = {'roi_spatial_footprints' : self.roi_spatial_footprints,
                        'roi_temporal_footprints': self.roi_temporal_footprints,
                        'roi_temporal_residuals' : self.roi_temporal_residuals,
                        'bg_spatial_footprints'  : self.bg_spatial_footprints,
                        'bg_temporal_footprints' : self.bg_temporal_footprints,
                        'filtered_out_rois'      : self.filtered_out_rois,
                        'erased_rois'            : self.erased_rois,
                        'removed_rois'           : self.removed_rois,
                        'locked_rois'            : self.locked_rois}

            # save the ROI data
            np.save(save_path, roi_data)

    def load_rois(self, load_path):
        # load the saved ROIs
        roi_data = np.load(load_path)

        # we are loading a dictionary containing an ROI array and other ROI variables
        # extract the dictionary
        roi_data = roi_data[()]

        # set ROI variables
        self.roi_spatial_footprints  = roi_data['roi_spatial_footprints']
        self.roi_temporal_footprints = roi_data['roi_temporal_footprints']
        self.roi_temporal_residuals  = roi_data['roi_temporal_residuals']
        self.bg_spatial_footprints   = roi_data['bg_spatial_footprints']
        self.bg_temporal_footprints  = roi_data['bg_temporal_footprints']
        self.filtered_out_rois       = roi_data['filtered_out_rois']
        self.erased_rois             = roi_data['erased_rois']
        self.removed_rois            = roi_data['removed_rois']
        self.locked_rois             = roi_data['locked_rois']

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths and lengths at the indices, in reverse order
            index = indices[i]
            del self.video_paths[index]
            del self.video_lengths[index]

        if len(self.video_paths) == 0:
            # all videos removed; reset variables
            self.reset_variables()
            self.reset_motion_correction_variables()
            self.reset_roi_finding_variables()
            self.reset_roi_filtering_variables()

    def motion_correct_videos(self):
        mc_videos, mc_borders = motion_correction.motion_correct_videos(self.video_paths, self.params["max_shift"], self.params["patch_stride"], self.params["patch_overlap"], use_multiprocessing=self.use_multiprocessing)

        if len(mc_videos) != 0:
            self.mc_video_paths = []
            for i in range(len(mc_videos)):
                # generate a path for the motion-corrected video
                video_path    = video_paths[i]
                directory     = os.path.dirname(video_path)
                filename      = os.path.basename(video_path)
                mc_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_mc.tif")
                
                # save the motion-corrected video
                imsave(mc_video_path, mc_videos[i])
                
                # add to motion-corrected video paths list
                self.mc_video_paths.append(mc_video_path)
            
            # update motion-corrected video borders list
            self.mc_borders = mc_borders

    def find_rois(self):
        if self.use_motion_correction and self.mc_video_paths is not None:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = roi_finding.find_rois_from_videos(video_paths, self.params, mc_borders=self.mc_borders, use_multiprocessing=self.use_multiprocessing)

    def filter_rois(self):
        if self.use_motion_correction and self.mc_video_paths is not None:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        # get filtered-out ROIs
        self.filtered_out_rois = utilities.filter_rois(video_paths, self.roi_spatial_footprints, self.roi_temporal_footprints, self.roi_temporal_residuals, self.bg_spatial_footprints, self.bg_temporal_footprints, self.params)
        
        for z in range(self.num_z_planes):
            # don't filter out ROIs that are locked
            self.filtered_out_rois[z] = [ roi for roi in self.filtered_out_rois[z] if roi not in self.locked_rois[z] ]

            # update removed ROIs list
            self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

    def erase_roi(self, roi, z):
        # update ROI filtering variables
        self.erased_rois[z].append(roi)
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

        if roi in self.locked_rois[z]:
            # remove ROI from locked ROIs list
            index = self.locked_rois[z].index(roi)
            del self.locked_rois[z][index]

    def unerase_roi(self, roi, z):
        # update ROI filtering variables
        if roi in self.erased_rois[z]:
            i = self.erased_rois[z].index(roi)
            del self.erased_rois[z][i]
        elif roi in self.filtered_out_rois[z]:
            i = self.filtered_out_rois[z].index(roi)
            del self.filtered_out_rois[z][i]

            # add to locked ROIs list
            if roi not in self.locked_rois[z]:
                self.locked_rois.append(roi)

        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))
