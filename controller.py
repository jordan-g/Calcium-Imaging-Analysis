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
                  'max_circ'            : 2}

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

        # initialize settings variables
        self.motion_correct_all_videos = False # whether to use motion correction when processing videos
        self.use_mc_video              = False # whether to use the motion-corrected video for finding ROIs
        self.mc_current_z              = False # whether to motion-correct only the current z plane
        self.find_new_rois             = True  # whether we need to find new ROIs
        self.mc_rois                   = False # whether found ROIs are based on the motion-corrected video
        self.apply_blur                = False

        # initialize motion correction, ROI finding & ROI filtering variables
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables(reset_rois=True)
        self.reset_roi_filtering_variables(reset_rois=True)

    def reset_motion_correction_variables(self):
        self.mc_video          = None

    def reset_roi_finding_variables(self, reset_rois=False):
        if reset_rois:
            self.n_masks = 0

            if self.video is not None:
                self.masks             = [ [] for i in range(self.video.shape[1]) ]
                self.mask_points       = [ [] for i in range(self.video.shape[1]) ]
                self.rois              = [ np.zeros(self.video.shape[2:]).astype(int) for i in range(self.video.shape[1]) ]
                self.original_rois     = [ np.zeros(self.video.shape[2:]).astype(int) for i in range(self.video.shape[1]) ]
                self.roi_areas         = [ [] for i in range(self.video.shape[1]) ]
                self.roi_circs         = [ [] for i in range(self.video.shape[1]) ]
                self.filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]
            else:
                self.masks             = None
                self.mask_points       = None
                self.rois              = None
                self.roi_areas         = None
                self.roi_circs         = None
                self.filtered_out_rois = None

    def reset_roi_filtering_variables(self, reset_rois=False):
        if reset_rois:
            if self.video is not None:
                self.erased_rois  = [ [] for i in range(self.video.shape[1]) ]
                self.removed_rois = [ [] for i in range(self.video.shape[1]) ]
                self.locked_rois  = [ [] for i in range(self.video.shape[1]) ]
            else:
                self.erased_rois      = None
                self.removed_rois     = None
                self.locked_rois      = None
                self.last_erased_rois = None

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
        self.video = video[1:]

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
            # self.video = self.video[:, np.newaxis, :, :]
            self.video = self.video[np.newaxis, :, :, :]

        # remove nans
        self.video = np.nan_to_num(self.video).astype(np.float32)

        print("Opened video with shape {}.".format(self.video.shape))

        # if the video is a different shape than the previous one, get rid of any exising ROI information
        if previous_video_shape is None or self.video.shape[2] != previous_video_shape[2] or self.video.shape[3] != previous_video_shape[3]:
            reset_rois = True
        else:
            reset_rois = False

        # reset the ROI finding & filtering variables
        self.reset_roi_finding_variables(reset_rois=reset_rois)
        self.reset_roi_filtering_variables(reset_rois=reset_rois)

        return True

    def save_mc_video(self, save_path):
        if self.mc_video is not None:
            # save the video
            imsave(save_path, self.mc_video)

    def save_rois(self, save_path):
        if self.rois[0] is not None:
            # create a dictionary to hold the ROI data
            roi_data = {'rois'             : self.rois,
                        'roi_areas'        : self.roi_areas,
                        'roi_circs'        : self.roi_circs,
                        'filtered_out_rois': self.filtered_out_rois,
                        'erased_rois'      : self.erased_rois,
                        'removed_rois'     : self.removed_rois,
                        'locked_rois'      : self.locked_rois}

            # save the ROI data
            np.save(save_path, roi_data)

    def load_rois(self, load_path):
        # load the saved ROIs
        roi_data = np.load(load_path)

        if len(roi_data.shape) == 3:
            # we are loading just an ROI array

            # make sure the ROI array shape matches the video
            if roi_data.shape != self.video.shape[1:]:
                print("Error: ROI array shape does not match the video shape.")
                return

            # set ROI variables
            self.rois              = roi_data
            self.original_rois     = self.rois[:]
            self.filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]
            self.erased_rois       = [ [] for i in range(self.video.shape[1]) ]
            self.removed_rois      = [ [] for i in range(self.video.shape[1]) ]
            self.locked_rois       = [ [] for i in range(self.video.shape[1]) ]

            # calculate ROI areas and circulatures for the ROIs
            for z in range(self.video.shape[1]):
                self.roi_areas[z], self.roi_circs[z] = utilities.calculate_roi_properties(roi_data[z], self.mean_images[z])
        else:
            # we are loading a dictionary containing an ROI array and other ROI variables

            # extract the dictionary
            roi_data = roi_data[()]

            # make sure the ROI array shape matches the video
            if np.array(roi_data['rois']).shape != self.video.shape[1:]:
                print("Error: ROI array shape does not match the video shape.")
                return

            # set ROI variables
            self.rois              = roi_data['rois']
            self.original_rois     = self.rois[:]
            self.roi_areas         = roi_data['roi_areas']
            self.roi_circs         = roi_data['roi_circs']
            self.filtered_out_rois = roi_data['filtered_out_rois']
            self.erased_rois       = roi_data['erased_rois']
            self.removed_rois      = roi_data['removed_rois']
            self.locked_rois       = roi_data['locked_rois']

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

        if len(self.video_paths) == 0:
            # reset variables
            self.video_path   = None
            self.use_mc_video = False
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

    def filter_rois(self, z):
        # filter out ROIs and update the removed ROIs
        _, self.filtered_out_rois[z] = utilities.filter_rois(self.mean_images[z], self.rois[z], self.params['min_area'], self.params['max_area'], self.params['min_circ'], self.params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.locked_rois[z])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

    def create_roi(self, start_point, end_point, label, z):
        # find the center of the ROI
        center_point = (int(round((start_point[0] + end_point[0])/2)), int(round((start_point[1] + end_point[1])/2)))
        axis_1 = np.abs(center_point[0] - end_point[0])
        axis_2 = np.abs(center_point[1] - end_point[1])

        # create a mask
        mask = np.zeros(self.rois[z].shape).astype(np.uint8)

        # draw an ellipse on the mask
        cv2.ellipse(mask, center_point, (axis_1, axis_2), 0, 0, 360, 1, -1)

        # detect contours in the mask and grab the largest one
        c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2][0]

        # get the area of the ROI
        area = cv2.contourArea(c)

        if area > 0:
            # get the perimeter of the ROI
            perimeter = cv2.arcLength(c, True)

            # record the circulature and area of the ROI
            self.roi_circs[z] = np.append(self.roi_circs[z], [(perimeter**2)/(4*np.pi*area)])
            self.roi_areas[z] = np.append(self.roi_areas[z], [area])

            # update the ROI array
            self.rois[z][mask == 1] = l

            # make the ROI locked by default
            self.locked_rois[z].append(l)

    def shift_rois(self, start_point, end_point, z):
        # get the x and y shift
        y_shift = end_point[1] - start_point[1]
        x_shift = end_point[0] - start_point[0]

        # shift the ROI & ROI overlay arrays
        self.rois[z] = np.roll(self.rois[z], y_shift, axis=0)
        self.rois[z] = np.roll(self.rois[z], x_shift, axis=1)

    def erase_rois_near_point(self, roi_point, z, radius=10):
        # find out which ROIs to erase
        rois_to_erase = utilities.get_rois_near_point(self.rois[z], roi_point, radius)

        # remove the ROIs
        for i in range(len(rois_to_erase)-1, -1, -1):
            roi = rois_to_erase[i]
            if roi in self.locked_rois[z] or roi in self.erased_rois[z]:
                del rois_to_erase[i]

        if len(rois_to_erase) > 0:
            # update ROI variables
            self.erased_rois[z]          += rois_to_erase
            self.last_erased_rois[z][-1] += rois_to_erase
            self.removed_rois[z]         = self.filtered_out_rois[z] + self.erased_rois[z]

        return rois_to_erase

    def erase_roi(self, label, z): # TODO: call roi_unselected() method of the param window
        # update ROI filtering variables
        self.erased_rois[z].append(label)
        self.last_erased_rois[z].append([label])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

        if label in self.locked_rois[z]:
            index = self.locked_rois[z].index(label)
            del self.locked_rois[z][index]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))