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
                  'soma_threshold'      : 1,
                  'min_area'            : 10,
                  'max_area'            : 100,
                  'min_circ'            : 0,
                  'max_circ'            : 2,
                  'min_correlation'     : 0.2}

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
        self.mean_images    = None
        self.adjusted_image = None

        # initialize state variables
        self.closing                      = False # whether the controller is in the process of closing
        self.performing_motion_correction = False # whether motion correction is being performed
        self.finding_rois                 = False # whether ROIs are currently being found
        self.processing_videos            = False # whether videos are currently being processed

        # initialize settings variables
        self.motion_correct_all_videos = False # whether to use motion correction when processing videos
        self.use_mc_video              = False # whether to use the motion-corrected video for finding ROIs
        self.mc_current_z              = False # whether to motion-correct only the current z plane
        self.find_new_rois             = True  # whether we need to find new ROIs
        self.mc_rois                   = False # whether found ROIs are based on the motion-corrected video
        self.apply_blur                = False # whether to apply blurring to remove noise

        # initialize thread variables
        self.motion_correction_thread = None
        self.roi_finding_thread       = None
        self.video_processing_thread  = None  # thread for processing all videos
        
        # create windows
        self.param_window   = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

        # set the mode -- "motion_correcting" / "roi_finding" / "roi_filtering"
        self.mode = "motion_correcting"

        # set the current z plane to 0
        self.z = 0

        # set references to param widgets & preview window
        self.param_widget                   = self.param_window.main_param_widget
        self.motion_correction_param_widget = self.param_window.motion_correction_widget
        self.roi_finding_param_widget       = self.param_window.roi_finding_widget
        self.roi_filtering_param_widget     = self.param_window.roi_filtering_widget

        # initialize motion correction, ROI finding & ROI filtering variables
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables(reset_rois=True)
        self.reset_roi_filtering_variables(reset_rois=True)

    def reset_motion_correction_variables(self):
        self.mc_video          = None
        self.adjusted_video    = None
        self.adjusted_mc_video = None
        self.adjusted_frame    = None
        self.adjusted_mc_frame = None

        # cancel any ongoing motion correction
        self.cancel_motion_correction()

    def reset_roi_finding_variables(self, reset_rois=False):
        self.background_mask      = None
        self.equalized_image      = None
        self.soma_mask            = None
        self.I_mod                = None
        self.soma_threshold_image = None
        self.roi_image            = None

        if reset_rois:
            self.roi_overlay       = None
            self.selected_mask     = None
            self.selected_mask_num = -1
            self.n_masks           = 0

            if self.video is not None:
                self.masks             = [ [] for i in range(self.video.shape[1]) ]
                self.mask_points       = [ [] for i in range(self.video.shape[1]) ]
                self.rois              = [ np.zeros(self.video.shape[2:]).astype(int) for i in range(self.video.shape[1]) ]
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

            # cancel any ongoing ROI finding
            self.cancel_roi_finding()

    def reset_roi_filtering_variables(self, reset_rois=False):
        self.roi_image = None
        self.figure    = None
        self.axis      = None

        if reset_rois:
            self.roi_overlay        = None
            self.original_rois      = None
            self.selected_roi       = None
            self.correlation_images = None

            if self.video is not None:
                self.erased_rois  = [ [] for i in range(self.video.shape[1]) ]
                self.removed_rois = [ [] for i in range(self.video.shape[1]) ]
                self.locked_rois  = [ [] for i in range(self.video.shape[1]) ]
                self.reset_history()
            else:
                self.erased_rois      = None
                self.removed_rois     = None
                self.locked_rois      = None
                self.last_erased_rois = None

    def select_videos_to_import(self):
        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')

            video_paths = [ str(path) for path in video_paths ]
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')[0]

        # import the videos (only the first video is actually opened and previewed, the rest are added to a list of videos to process)
        if video_paths is not None and len(video_paths) > 0:
            self.import_videos(video_paths)

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

        # notify the param window
        self.param_window.videos_imported(video_paths)

    def open_video(self, video_path):
        # get the shape of the previously-previewed video, if any
        if self.video is None:
            previous_video_shape = None
        else:
            previous_video_shape = self.video.shape

        # open the video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.npy'):
            video = np.load(video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            video = imread(video_path)

        if len(video.shape) < 3:
            print("Error: Opened file is not a video -- not enough dimensions.")
            return False

        # there is a bug with z-stack OIR files where the first frame of the first z plane is wrong,
        # so we have to throw out the first frame of the video here
        self.video = video[1:]

        # figure out the dynamic range of the video
        max_value = np.amax(self.video)
        if max_value > 511:
            self.video_max = 1023
        elif max_value > 255:
            self.video_max = 511
        elif max_value > 1:
            self.video_max = 255
        else:
            self.video_max = 1

        # set the path to the previewed video
        self.video_path = video_path

        if len(self.video.shape) == 3:
            # add a z dimension
            self.video = self.video[:, np.newaxis, :, :]

        # set z to 0 if necessary
        if self.z >= self.video.shape[1]:
            self.z = 0

        # remove nans
        self.video = np.nan_to_num(self.video).astype(np.float32)

        # calculate gamma- and contrast-adjusted video
        self.adjusted_video = self.calculate_adjusted_video(self.video, z=self.z)

        print(self.adjusted_video.dtype)
        print(np.amax(self.adjusted_video), self.adjusted_video[:, 0])

        print("Opened video with shape {}.".format(self.video.shape))

        # notify the preview window
        self.preview_window.video_opened(self.video_path)

        # notify the param window
        self.param_window.video_opened(max_z=self.video.shape[1]-1, z=self.z)

        # if the video is a different shape than the previous one, get rid of any exising ROI information
        if previous_video_shape is None or self.video.shape[2] != previous_video_shape[2] or self.video.shape[3] != previous_video_shape[3]:
            reset_rois = True
        else:
            reset_rois = False

        # reset the ROI finding & filtering variables
        self.reset_roi_finding_variables(reset_rois=reset_rois)
        self.reset_roi_filtering_variables(reset_rois=reset_rois)

        # reset history variables
        self.reset_history()

        # play the adjusted video
        self.play_video(self.adjusted_video)

        return True

    def save_mc_video(self):
        if self.mc_video is not None:
            # let the user pick where to save the video
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save Video', '{}_motion_corrected'.format(os.path.splitext(self.video_path)[0]), 'Videos (*.tif *.tiff *.npy)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save Video', '{}_motion_corrected'.format(os.path.splitext(self.video_path)[0]), 'Videos (*.tif *.tiff *.npy)')[0])
            if not (save_path.endswith('.npy') or save_path.endswith('.tif') or save_path.endswith('.tiff')):
                save_path += ".tif"

            # save the video
            if save_path.endswith('.npy'):
                np.save(save_path, self.mc_video)
            else:
                imsave(save_path, self.mc_video)

    def save_rois(self):
        if self.rois[0] is not None:
            # let the user pick where to save the ROIs
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)')[0])
            if not save_path.endswith('.npy'):
                save_path += ".npy"

            if save_path is not None and len(save_path) > 0:
                # create a dictionary to hold the ROI data
                roi_data = {'labels'           : self.rois,
                            'roi_areas'        : self.roi_areas,
                            'roi_circs'        : self.roi_circs,
                            'filtered_out_rois': self.filtered_out_rois,
                            'erased_rois'      : self.erased_rois,
                            'removed_rois'     : self.removed_rois,
                            'locked_rois'      : self.locked_rois}

                # save the ROI data
                np.save(save_path, roi_data)

    def load_rois(self):
        # let the user pick saved ROIs
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        if load_path is not None and len(load_path) > 0:
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
                self.filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]
                self.erased_rois       = [ [] for i in range(self.video.shape[1]) ]
                self.removed_rois      = [ [] for i in range(self.video.shape[1]) ]
                self.locked_rois       = [ [] for i in range(self.video.shape[1]) ]

                # calculate ROI areas and circulatures for the ROIs
                for z in range(self.video.shape[1]):
                    self.roi_areas[z], self.roi_circs[z] = utilities.calculate_roi_properties(roi_data[z])
            else:
                # we are loading a dictionary containing an ROI array and other ROI variables

                # extract the dictionary
                roi_data = roi_data[()]

                # make sure the ROI array shape matches the video
                if np.array(roi_data['labels']).shape != self.video.shape[1:]:
                    print("Error: ROI array shape does not match the video shape.")
                    return

                # set ROI variables
                self.rois              = roi_data['labels']
                self.roi_areas         = roi_data['roi_areas']
                self.roi_circs         = roi_data['roi_circs']
                self.filtered_out_rois = roi_data['filtered_out_rois']
                self.erased_rois       = roi_data['erased_rois']
                self.removed_rois      = roi_data['removed_rois']
                self.locked_rois       = roi_data['locked_rois']

            # stop any motion correction or ROI finding process
            self.cancel_motion_correction()
            self.cancel_roi_finding()

            self.preview_window.timer.stop()

            self.find_new_rois = False
            self.use_mc_video  = False
            self.mc_rois       = False

            # show ROI filtering parameters
            self.show_roi_filtering_params(loading_rois=True)

    def cancel_motion_correction(self):
        if self.motion_correction_thread is not None:
            # inform the thread to stop running
            self.motion_correction_thread.running = False

        # reset motion correction progress text
        self.param_widget.update_motion_correction_progress(-1)

        # reset motion correction progress variables
        self.performing_motion_correction = False
        self.motion_correction_thread     = None

    def cancel_roi_finding(self):
        if self.roi_finding_thread is not None:
            # inform the thread to stop running
            self.roi_finding_thread.running = False

        # reset ROI finding progress text
        self.param_widget.update_roi_finding_progress(-1)

        # reset ROI finding progress variables
        self.finding_rois       = False
        self.roi_finding_thread = None

    def cancel_processing_videos(self):
        if self.video_processing_thread is not None:
            # inform the thread to stop running
            self.video_processing_thread.running = False

        # reset video processing progress text
        self.param_window.update_process_videos_progress(-1)

        # reset video processing progress variables
        self.processing_videos       = False
        self.video_processing_thread = None

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths at the indices, in reverse order
            index = indices[i]
            del self.video_paths[index]

        if len(self.video_paths) == 0:
            # cancel any ongoing motion correction
            self.cancel_motion_correction()

            # reset variables
            self.video_path   = None
            self.use_mc_video = False

            # switch to showing motion correction params
            self.show_motion_correction_params()

            # reset param window & preview window to their initial states
            self.param_window.set_initial_state()
            self.preview_window.set_initial_state()
        elif 0 in indices:
            # the first video was removed; open the next one for previewing
            self.open_video(self.video_paths[0])

    def process_all_videos(self):
        # stop any motion correction or ROI finding process
        self.cancel_motion_correction()
        self.cancel_roi_finding()

        # get rid of filtered out and/or manually erased rois
        if self.removed_rois is not None:
            rois = utilities.remove_rois(self.rois, self.removed_rois)
        else:
            rois = utilities.remove_rois(self.rois, self.filtered_out_rois)

        if not self.processing_videos:
            # cancel any ongoing processing of videos
            self.cancel_processing_videos()

            # create a new thread for processing the videos
            self.video_processing_thread = ProcessVideosThread(self.param_window)
            self.video_processing_thread.progress.connect(self.process_videos_progress)
            self.video_processing_thread.finished.connect(self.process_videos_finished)

            # set its parameters
            self.video_processing_thread.set_parameters(self.video_paths, labels, self.motion_correct_all_videos, self.params["max_shift"], self.params["patch_stride"], self.params["patch_overlap"], self.params)

            # start the thread
            self.video_processing_thread.start()

            # notify the param window
            self.param_window.process_videos_started()

            self.processing_videos = True
        else:
            # cancel any ongoing processing of videos
            self.cancel_processing_videos()

    def process_videos_progress(self, percent):
        self.param_window.update_process_videos_progress(percent)

    def process_videos_finished(self):
        self.param_window.update_process_videos_progress(100)

    def set_motion_correct(self, boolean):
        self.motion_correct_all_videos = boolean

    def show_motion_correction_params(self):
        # switch to showing motion correction params
        self.param_window.stacked_widget.setCurrentIndex(0)
        self.mode = "motion_correcting"

        self.preview_window.timer.stop()

        # play the video
        if self.use_mc_video and self.adjusted_mc_video is not None:
            self.play_video(self.adjusted_mc_video)
        else:
            self.play_video(self.adjusted_video)

        self.param_window.statusBar().showMessage("")

    def show_roi_finding_params(self):
        # cancel any ongoing motion correction
        self.cancel_motion_correction()

        self.preview_window.timer.stop()

        # switch to showing ROI finding params
        self.param_window.stacked_widget.setCurrentIndex(1)
        self.mode = "roi_finding"

        if self.mc_rois != self.use_mc_video:
            self.find_new_rois = True

        if self.find_new_rois:
            self.reset_roi_finding_variables(reset_rois=True)
            self.reset_roi_filtering_variables(reset_rois=True)

            # calculate mean images & correlation images
            self.calculate_mean_images()
            self.correlation_images = [ utilities.correlation(self.video, z).astype(np.float32) for z in range(self.video.shape[1]) ]

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
                    self.mean_images[z] = np.maximum(self.mean_images[z].astype(np.float32) - difference, 0).astype(np.uint8)

            # uncheck "Show ROIs" checkbox
            self.roi_finding_param_widget.show_rois_checkbox.setChecked(False)

            self.find_new_rois = False

            # calculate ROI finding variables
            self.adjusted_image                                   = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])
            self.background_mask                                  = utilities.calculate_background_mask(self.adjusted_image, self.params['background_threshold'])
            self.equalized_image                                  = utilities.calculate_equalized_image(self.adjusted_image, self.background_mask, self.params['window_size'])
            self.soma_mask, self.I_mod, self.soma_threshold_image = utilities.calculate_soma_threshold_image(self.equalized_image, self.params['soma_threshold'])

        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())

        self.param_window.statusBar().showMessage("")

        self.preview_window.setWindowTitle(os.path.basename(self.video_path))

    def show_roi_filtering_params(self, loading_rois=False):
        # switch to showing ROI filtering params
        self.param_window.stacked_widget.setCurrentIndex(2)
        self.mode = "roi_filtering"

        if loading_rois:
            # calculate mean images & correlation images
            self.calculate_mean_images()
            self.correlation_images = [ utilities.correlation(self.video, z).astype(np.float32) for z in range(self.video.shape[1]) ]

            # calculate adjusted image
            self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

            # reset history variables
            self.reset_history()

        print(self.rois, self.erased_rois, self.filtered_out_rois, self.locked_rois)

        # calculate the ROI image
        self.calculate_roi_image(z=self.z, update_overlay=self.roi_overlay is None)

        # show the ROI image
        self.roi_filtering_param_widget.show_rois_checkbox.setChecked(True)
        self.show_roi_image(self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # add the current state to the history
        self.add_to_history()

        self.param_window.statusBar().showMessage("")

    def reset_history(self, z=None):
        # initialize history variables
        if z is None:
            self.last_erased_rois           = [ [] for i in range(self.video.shape[1]) ]
            self.previous_labels            = [ [] for i in range(self.video.shape[1]) ]
            self.previous_roi_overlays      = [ [] for i in range(self.video.shape[1]) ]
            self.previous_erased_rois       = [ [] for i in range(self.video.shape[1]) ]
            self.previous_filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]
            self.previous_adjusted_images   = [ [] for i in range(self.video.shape[1]) ]
            self.previous_roi_images        = [ [] for i in range(self.video.shape[1]) ]
            self.previous_selected_rois     = [ [] for i in range(self.video.shape[1]) ]
            self.previous_removed_rois      = [ [] for i in range(self.video.shape[1]) ]
            self.previous_locked_rois       = [ [] for i in range(self.video.shape[1]) ]
            self.previous_params            = [ [] for i in range(self.video.shape[1]) ]
        else:
            self.last_erased_rois[z]           = []
            self.previous_labels[z]            = []
            self.previous_roi_overlays[z]      = []
            self.previous_erased_rois[z]       = []
            self.previous_filtered_out_rois[z] = []
            self.previous_adjusted_images[z]   = []
            self.previous_roi_images[z]        = []
            self.previous_selected_rois[z]     = []
            self.previous_removed_rois[z]      = []
            self.previous_locked_rois[z]       = []
            self.previous_params[z]            = []

    def close_all(self):
        # cancel any ongoing threads
        self.cancel_motion_correction()
        self.cancel_roi_finding()
        self.cancel_processing_videos()

        self.closing = True

        # close param & preview windows
        self.param_window.close()
        self.preview_window.close()

        # save the current parameters
        self.save_params()

    def preview_contrast(self, contrast):
        self.params['contrast'] = contrast

        if self.mode == "motion_correcting":
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            if self.use_mc_video:
                adjusted_frame = self.calculate_adjusted_frame(self.mc_video)
            else:
                adjusted_frame = self.calculate_adjusted_frame(self.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("contrast", contrast)

    def preview_gamma(self, gamma):
        self.params['gamma'] = gamma

        if self.mode == "motion_correcting":
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            if self.use_mc_video:
                adjusted_frame = self.calculate_adjusted_frame(self.mc_video)
            else:
                adjusted_frame = self.calculate_adjusted_frame(self.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("gamma", gamma)

    def update_param(self, param, value):
        # update the parameter
        if param in self.params.keys():
            self.params[param] = value

        if self.mode == "motion_correcting":
            if param in ("contrast, gamma"):
                self.preview_window.timer.stop()

                # calculate a contrast- and gamma-adjusted version of the current video at the current z plane and play it
                if self.use_mc_video:
                    self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
                    self.play_video(self.adjusted_mc_video)
                else:
                    self.adjusted_video = self.calculate_adjusted_video(self.video, self.z)
                    self.play_video(self.adjusted_video)
            elif param == "fps":
                # update the FPS of the preview window
                self.preview_window.set_fps(self.params['fps'])
            elif param == "z":
                self.z = value

                self.preview_window.timer.stop()

                # calculate a contrast- and gamma-adjusted version of the current video at the new z plane and play it
                if self.use_mc_video:
                    self.adjusted_video    = None
                    self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
                    self.play_video(self.adjusted_mc_video)
                else:
                    self.adjusted_mc_video = None
                    self.adjusted_video    = self.calculate_adjusted_video(self.video, self.z)
                    self.play_video(self.adjusted_video)
        elif self.mode == "roi_finding":
            if param in ("contrast, gamma"):
                # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
                self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

                # update the ROI image using the new adjusted image
                if self.rois is not None:
                    self.calculate_roi_image(self.z, update_overlay=False)

                # show the ROI image
                self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())
            elif param == "background_threshold":
                # calculate the background mask using the new threshold
                self.background_mask = utilities.calculate_background_mask(self.adjusted_image, self.params['background_threshold'])

                # uncheck the "Show ROIs" checkbox
                self.roi_finding_param_widget.show_rois_checkbox.setChecked(False)
                self.param_window.show_rois_action.setChecked(False)

                # show the background mask
                # TODO: only show the background mask while dragging the slider
                self.preview_window.plot_image(self.adjusted_image, background_mask=self.background_mask)
            elif param == "window_size":
                # calculate the equalized image using the new window size
                self.equalized_image = utilities.calculate_equalized_image(self.adjusted_image, self.background_mask, self.params['window_size'])

                # uncheck the "Show ROIs" checkbox
                self.roi_finding_param_widget.show_rois_checkbox.setChecked(False)
                self.param_window.show_rois_action.setChecked(False)

                # show the equalized image
                # TODO: only show the equalized image while dragging the slider
                self.preview_window.plot_image(self.equalized_image)
            elif param == "z":
                self.z = value

                # calculate a contrast- and gamma-adjusted version of the new z plane's mean image
                self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

                # update the ROI image using the new adjusted image
                if self.rois is not None:
                    self.calculate_roi_image(self.z, update_overlay=True)

                # show the ROI image
                self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())
        elif self.mode == "roi_filtering":
            if param in ("contrast, gamma"):
                # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
                self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

                # update the ROI image using the new adjusted image
                if self.rois is not None:
                    self.calculate_roi_image(self.z, update_overlay=False)

                # show the ROI image
                self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())
            if param == "z":
                self.z = value

                # calculate a contrast- and gamma-adjusted version of the new z plane's mean image
                self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

                # filter the ROIs at the new z plane
                self.filter_rois(z=self.z)

                # update the ROI image using the new adjusted image
                self.calculate_roi_image(z=self.z, update_overlay=True)

                # show the ROI image
                self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

                # add the current state to the history only if this is the first time we've switched to this z plane
                self.add_to_history(only_if_new=True)
            elif param in ("min_area", "max_area", "min_circ", "max_circ", "min_correlation"):
                pass

    def calculate_adjusted_video(self, video, z=None):
        if z is not None:
            # calculate the adjusted video only at this z plane
            return utilities.adjust_gamma(utilities.adjust_contrast((video[:, z, :, :]*255.0/self.video_max).astype(np.uint8), self.params['contrast']), self.params['gamma']).astype(np.float32)*self.video_max/255.0
        else:
            # calculate the adjusted video at all z planes
            return utilities.adjust_gamma(utilities.adjust_contrast((video*255.0/self.video_max).astype(np.uint8), self.params['contrast']), self.params['gamma']).astype(np.float32)*self.video_max/255.0

    def calculate_adjusted_frame(self, video):
        # calculate the adjusted frame
        return utilities.adjust_gamma(utilities.adjust_contrast((video[self.preview_window.frame_num, self.z]*255.0/self.video_max).astype(np.uint8), self.params['contrast']), self.params['gamma']).astype(np.float32)*self.video_max/255.0

    def motion_correct_video(self):
        if not self.performing_motion_correction:
            # cancel any ongoing motion correction
            self.cancel_motion_correction()

            # create a motion correction thread
            self.motion_correction_thread = MotionCorrectThread(self.motion_correction_param_widget)
            self.motion_correction_thread.progress.connect(self.motion_correction_progress)
            self.motion_correction_thread.finished.connect(self.motion_correction_ended)

            # determine whether to motion correct all z planes or just the current one (-1 means all z planes)
            if self.mc_current_z:
                mc_z = self.z
            else:
                mc_z = -1

            # set the parameters of the motion correction thread
            self.motion_correction_thread.set_parameters(self.video, self.video_path, int(self.params["max_shift"]), int(self.params["patch_stride"]), int(self.params["patch_overlap"]), mc_z=mc_z)

            # start the thread
            self.motion_correction_thread.start()

            # notify the param widget
            self.motion_correction_param_widget.motion_correction_started()

            self.performing_motion_correction = True
        else:
            # cancel any ongoing motion correction
            self.cancel_motion_correction()

    def motion_correction_progress(self, percent):
        # notify the param widget
        self.motion_correction_param_widget.update_motion_correction_progress(percent)

    def motion_correction_ended(self, mc_video): # TODO: create a motion_correction_ended() method for the param window
        # notify the param widget
        self.motion_correction_param_widget.update_motion_correction_progress(100)

        if np.sum(mc_video) != 0:
            self.mc_video     = mc_video
            self.use_mc_video = True

            # calculate the adjusted motion-corrected video at the current z plane
            self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)

            # update the param window
            self.param_window.videos_widget.save_mc_video_button.setEnabled(True)
            self.motion_correction_param_widget.use_mc_video_checkbox.setEnabled(True)
            self.motion_correction_param_widget.use_mc_video_checkbox.setChecked(True)

            self.set_use_mc_video(True)

    def cancel_motion_correction(self):
        if self.motion_correction_thread is not None:
            self.motion_correction_thread.running = False

        self.motion_correction_param_widget.update_motion_correction_progress(100)

        self.performing_motion_correction = False
        self.motion_correction_thread     = None

    def play_video(self, video):
        self.preview_window.play_movie(video, fps=self.params['fps'])

    def set_use_mc_video(self, use_mc_video):
        self.use_mc_video = use_mc_video

        # calculate the corresponding adjusted video and play it
        if self.use_mc_video:
            if self.adjusted_mc_video is None:
                self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
            self.preview_window.play_movie(self.adjusted_mc_video, fps=self.params['fps'])
        else:
            if self.adjusted_video is None:
                self.adjusted_video = self.calculate_adjusted_video(self.video, self.z)
            self.preview_window.play_movie(self.adjusted_video, fps=self.params['fps'])

    def set_mc_current_z(self, mc_current_z):
        self.mc_current_z = mc_current_z

    def calculate_mean_images(self):
        if self.use_mc_video and self.adjusted_mc_video is not None:
            video = self.mc_video*255.0/self.video_max
        else:
            video = self.video*255.0/self.video_max

        if self.apply_blur:
            self.mean_images = [ ndi.median_filter(utilities.sharpen(ndi.gaussian_filter(denoise_tv_chambolle(utilities.mean(video, z).astype(np.float32), weight=0.01, multichannel=False), 1)), 3).astype(np.uint8) for z in range(video.shape[1]) ]
        else:
            self.mean_images = [ utilities.mean(video, z).astype(np.uint8) for z in range(video.shape[1]) ]

    def set_apply_blur(self, apply_blur):
        self.apply_blur = apply_blur

        # calculate new mean images
        self.calculate_mean_images()

        # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
        self.adjusted_image = utilities.calculate_adjusted_image(self.mean_images[self.z], self.params['contrast'], self.params['gamma'])

        # update the ROI image using the new adjusted image
        if self.rois is not None:
            self.calculate_roi_image(self.z, update_overlay=False)

        # show the ROI image
        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())

    def show_roi_image(self, show):
        # plot the ROI image (or the regular image if show is False)
        if show:
            self.preview_window.plot_image(self.roi_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

        # update the param window
        self.param_window.show_rois_action.setChecked(show)
        if self.mode == "roi_finding":
            self.roi_finding_param_widget.show_rois_checkbox.setChecked(show)
        elif self.mode == "roi_filtering":
            self.roi_filtering_param_widget.show_rois_checkbox.setChecked(show)

    def save_roi_image(self):
        # let the user pick where to save the ROI images
        if pyqt_version == 4:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROI image', '{}_rois_z_{}'.format(os.path.splitext(self.video_path)[0], self.z), 'PNG (*.png)'))
        elif pyqt_version == 5:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROI image', '{}_rois_z_{}'.format(os.path.splitext(self.video_path)[0], self.z), 'PNG (*.png)')[0])
        if not save_path.endswith('.png'):
            save_path += ".png"

        if save_path is not None and len(save_path) > 0:
            # save the ROIs image
            scipy.misc.imsave(save_path, self.roi_image)

    def set_invert_masks(self, boolean):
        self.params['invert_masks'] = boolean

        # invert the masks
        for i in range(len(self.masks)):
            for j in range(len(self.masks[i])):
                self.masks[i][j] = self.masks[i][j] == False

        self.preview_window.plot_image(self.adjusted_image)

    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.adjusted_image)

            # notify the preview window that we are in mask drawing mode
            self.preview_window.start_drawing_mask()

            self.selected_mask     = None
            self.selected_mask_num = -1

            # update the param widget
            self.roi_finding_param_widget.draw_mask_button.setText("Done")
            self.roi_finding_param_widget.draw_mask_button.previous_message = "Draw a mask on the image preview."
            self.roi_finding_param_widget.param_widget.setEnabled(False)
            self.roi_finding_param_widget.button_widget.setEnabled(False)
            self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)
            self.roi_finding_param_widget.draw_mask_button.setEnabled(True)
        else:
            if len(self.preview_window.mask_points) > 0:
                # update the mask points
                mask_points = self.preview_window.mask_points
                mask_points += [mask_points[0]]
                self.mask_points[self.z].append(mask_points)
                mask_points = np.array(mask_points)

                # create the mask
                mask = np.zeros(self.adjusted_image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)

                if self.params['invert_masks']:
                    mask = mask == False

                self.masks[self.z].append(mask)

                self.n_masks += 1

            # notify the preview window that we are no longer in mask drawing mode
            self.preview_window.end_drawing_mask()

            self.preview_window.plot_image(self.adjusted_image)

            # update the param widget
            self.roi_finding_param_widget.draw_mask_button.setText("Draw Mask")
            self.roi_finding_param_widget.draw_mask_button.previous_message = ""
            self.roi_finding_param_widget.param_widget.setEnabled(True)
            self.roi_finding_param_widget.button_widget.setEnabled(True)

    def calculate_roi_image(self, z, update_overlay=True, newly_erased_rois=None):
        if update_overlay:
            roi_overlay = None
        else:
            roi_overlay = self.roi_overlay

        # create ROI image
        rgb_image = cv2.cvtColor(self.adjusted_image, cv2.COLOR_GRAY2RGB)
        self.roi_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.rois[z], self.selected_roi, self.erased_rois[z], self.filtered_out_rois[z], self.locked_rois[z], newly_erased_rois=newly_erased_rois, roi_overlay=roi_overlay)

    def show_roi_image(self, show):
        if show:
            self.preview_window.plot_image(self.roi_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

        self.param_window.show_rois_action.setChecked(show)
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(show)

    def find_rois(self):
        if not self.finding_rois:
            # cancel any ongoing ROI finding
            self.cancel_roi_finding()

            # create an ROI finding thread
            self.roi_finding_thread = ROIFindingThread(self.roi_finding_param_widget)
            self.roi_finding_thread.progress.connect(self.roi_finding_progress)
            self.roi_finding_thread.finished.connect(self.roi_finding_ended)

            # set the parameters of the ROI finding thread
            self.roi_finding_thread.set_parameters(self.video, self.mean_images, self.masks, self.params["min_area"], self.params["max_area"], self.params["min_circ"], self.params["max_circ"], self.params['min_correlation'], self.params['soma_threshold'], self.params['window_size'], self.params['background_threshold'], self.params['contrast'], self.params['gamma'], self.correlation_images)

            # start the thread
            self.roi_finding_thread.start()

            # notify the param widget
            self.roi_finding_param_widget.roi_finding_started()

            self.finding_rois = True
        else:
            # cancel any ongoing ROI finding
            self.cancel_roi_finding()

    def roi_finding_progress(self, percent):
        # notify the param widget
        self.roi_finding_param_widget.update_roi_finding_progress(percent)

    def roi_finding_ended(self, rois, roi_areas, roi_circs, filtered_out_rois): # TODO: create an roi_finding_ended() method for the param window
        # notify the param widget
        self.roi_finding_param_widget.update_roi_finding_progress(100)

        self.finding_rois = False

        self.rois              = rois
        self.original_rois     = rois[:]
        self.roi_areas         = roi_areas
        self.roi_circs         = roi_circs
        self.filtered_out_rois = filtered_out_rois

        # update the param window
        self.roi_finding_param_widget.show_rois_checkbox.setDisabled(False)
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(True)
        self.param_window.show_rois_action.setDisabled(False)
        self.param_window.save_roi_image_action.setDisabled(False)
        self.param_window.show_rois_action.setChecked(True)
        self.roi_finding_param_widget.filter_rois_button.setDisabled(False)

        self.mc_rois = self.use_mc_video

        # create ROI image
        rgb_image = cv2.cvtColor(self.adjusted_image, cv2.COLOR_GRAY2RGB)
        self.roi_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.rois[self.z], None, None, self.filtered_out_rois[self.z], None)

        # show the ROI image
        self.show_roi_image(True)

    def cancel_roi_finding(self):
        if self.roi_finding_thread is not None:
            self.roi_finding_thread.running = False

        self.roi_finding_param_widget.update_roi_finding_progress(-1)

        self.finding_rois       = False
        self.roi_finding_thread = None

    def select_mask(self, mask_point):
        # figure out which mask is selected (if any)
        selected_mask, selected_mask_num = utilities.get_mask_containing_point(self.masks[self.z], mask_point, inverted=self.params['invert_masks'])

        if selected_mask is not None:
            # update the param widget
            self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(True)

            self.selected_mask     = selected_mask
            self.selected_mask_num = selected_mask_num
        else:
            # update the param widget
            self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)

            self.selected_mask     = None
            self.selected_mask_num = -1

        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())

    def erase_selected_mask(self):
        if self.selected_mask is not None:
            # remove the mask
            del self.masks[self.z][self.selected_mask_num]
            del self.mask_points[self.z][self.selected_mask_num]

            self.selected_mask     = None
            self.selected_mask_num = -1

            # update the param widget
            self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)

            self.preview_window.plot_image(self.adjusted_image)

    def filter_rois(self, z):
        # filter out ROIs and update the removed ROIs
        _, self.filtered_out_rois[z] = utilities.filter_rois(self.mean_images[z], self.rois[z], self.params['min_area'], self.params['max_area'], self.params['min_circ'], self.params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.correlation_images[z], self.params['min_correlation'], self.locked_rois[z])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

        # update the ROI image
        self.calculate_roi_image(z=self.z, update_overlay=True)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def draw_rois(self): # TODO: create roi_drawing_started() and roi_drawing_ended() methods for the preview window
        if not self.preview_window.drawing_rois:
            self.preview_window.drawing_rois = True

            # notify the param window
            self.param_window.roi_drawing_started()
        else:
            self.preview_window.drawing_rois = False

            # notify the param window
            self.param_window.roi_drawing_ended()

    def create_roi(self, start_point, end_point):
        # find the center of the ROI
        center_point = (int(round((start_point[0] + end_point[0])/2)), int(round((start_point[1] + end_point[1])/2)))
        axis_1 = np.abs(center_point[0] - end_point[0])
        axis_2 = np.abs(center_point[1] - end_point[1])

        # pick a number for this ROI
        l = np.amax(self.rois[self.z])+1

        # create a mask
        mask = np.zeros(self.rois[self.z].shape).astype(np.uint8)

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
            self.roi_circs[self.z] = np.append(self.roi_circs[self.z], [(perimeter**2)/(4*np.pi*area)])
            self.roi_areas[self.z] = np.append(self.roi_areas[self.z], [area])

            # update the ROI array
            self.rois[self.z][mask == 1] = l

            # update the ROI overlay
            utilities.add_roi_to_overlay(self.roi_overlay, self.rois[self.z] == l, self.rois[self.z])

            # make the ROI locked by default
            self.locked_rois[self.z].append(l)

            # calculate the ROI image
            self.calculate_roi_image(z=self.z, update_overlay=False)

            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

            # add this state to the history
            self.add_to_history()

    def shift_labels(self, start_point, end_point):
        # get the x and y shift
        y_shift = end_point[1] - start_point[1]
        x_shift = end_point[0] - start_point[0]

        # shift the ROI & ROI overlay arrays
        self.rois[self.z] = np.roll(self.rois[self.z], y_shift, axis=0)
        self.rois[self.z] = np.roll(self.rois[self.z], x_shift, axis=1)
        self.roi_overlay  = np.roll(self.roi_overlay, y_shift, axis=0)
        self.roi_overlay  = np.roll(self.roi_overlay, x_shift, axis=1)

        # calculate the ROI image
        self.calculate_roi_image(z=self.z, update_overlay=False)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def erase_rois(self): # TODO: create roi_erasing_started() and roi_erasing_ended() methods for the preview window
        self.rois_erased = False

        if not self.preview_window.erasing_rois:
            self.selected_roi = None

            self.preview_window.erasing_rois = True

            # notify the param window
            self.param_window.roi_erasing_started()
        else:
            self.preview_window.erasing_rois = False

            # notify the param window
            self.param_window.roi_erasing_ended()

            # add the current state to the history
            self.add_to_history()

    def erase_rois_near_point(self, roi_point, radius=10):
        if not self.rois_erased:
            # create a new list storing the ROIs that are being erased in this erasing operation
            self.last_erased_rois[self.z].append([])
            self.rois_erased = True

        # find out which ROIs to erase
        rois_to_erase = utilities.get_rois_near_point(self.rois[self.z], roi_point, radius)

        # remove the ROIs
        for i in range(len(rois_to_erase)-1, -1, -1):
            roi = rois_to_erase[i]
            if roi in self.locked_rois[self.z] or roi in self.erased_rois[self.z]:
                del rois_to_erase[i]

        if len(rois_to_erase) > 0:
            # update ROI variables
            self.erased_rois[self.z]          += rois_to_erase
            self.last_erased_rois[self.z][-1] += rois_to_erase
            self.removed_rois[self.z]         = self.filtered_out_rois[self.z] + self.erased_rois[self.z]

            # create & show the new ROI image
            self.calculate_roi_image(z=self.z, update_overlay=False, newly_erased_rois=rois_to_erase)
            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def select_roi(self, roi_point): # TODO: create roi_selected() and roi_unselected() methods for the param window
        # find out which ROI to select
        selected_roi = utilities.get_roi_containing_point(self.rois[self.z], roi_point)

        if selected_roi is not None and selected_roi not in self.removed_rois[self.z]:
            # an ROI is selected

            self.selected_roi = selected_roi

            # create & show the new ROI image
            self.calculate_roi_image(z=self.z, update_overlay=False)
            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

            # update the param window
            self.roi_filtering_param_widget.lock_roi_button.setEnabled(True)
            self.roi_filtering_param_widget.enlarge_roi_button.setEnabled(True)
            self.roi_filtering_param_widget.shrink_roi_button.setEnabled(True)
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
            if selected_roi in self.locked_rois[self.z]:
                self.roi_filtering_param_widget.lock_roi_button.setText("Unlock ROI")
            else:
                self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")

            # get the activity of this ROI
            activity = utilities.calc_activity_of_roi(self.rois[self.z], self.video[:, self.z, :, :].transpose(1, 2, 0), self.selected_roi, z=self.z)

            # plot the activity of the ROI
            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.mpl_connect('close_event', self.figure_closed)
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()
            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))
        else:
            # no ROI is selected

            self.selected_roi = -1

            # create & show the new ROI image
            self.calculate_roi_image(z=self.z, update_overlay=False)
            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

            # update the param window
            self.roi_filtering_param_widget.lock_roi_button.setEnabled(False)
            self.roi_filtering_param_widget.enlarge_roi_button.setEnabled(False)
            self.roi_filtering_param_widget.shrink_roi_button.setEnabled(False)
            self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")

    def add_to_history(self, only_if_new=False):
        if (not only_if_new) or len(self.previous_labels[self.z]) == 0:
            print("Adding to history.")

            # only store up to 20 history states
            if len(self.previous_labels[self.z]) > 20:
                del self.previous_labels[self.z][0]
            if len(self.previous_roi_overlays[self.z]) > 20:
                del self.previous_roi_overlays[self.z][0]
            if len(self.previous_erased_rois[self.z]) > 20:
                del self.previous_erased_rois[self.z][0]
            if len(self.previous_filtered_out_rois[self.z]) > 20:
                del self.previous_filtered_out_rois[self.z][0]
            if len(self.previous_adjusted_images[self.z]) > 20:
                del self.previous_adjusted_images[self.z][0]
            if len(self.previous_roi_images[self.z]) > 20:
                del self.previous_roi_images[self.z][0]
            if len(self.previous_selected_rois[self.z]) > 20:
                del self.previous_selected_rois[self.z][0]
            if len(self.previous_removed_rois[self.z]) > 20:
                del self.previous_removed_rois[self.z][0]
            if len(self.previous_locked_rois[self.z]) > 20:
                del self.previous_locked_rois[self.z][0]

            # store the current state
            self.previous_labels[self.z].append(self.rois[self.z][:])
            self.previous_erased_rois[self.z].append(self.erased_rois[self.z][:])
            self.previous_filtered_out_rois[self.z].append(self.filtered_out_rois[self.z][:])
            self.previous_removed_rois[self.z].append(self.removed_rois[self.z][:])
            self.previous_locked_rois[self.z].append(self.locked_rois[self.z][:])

            self.previous_roi_overlays[self.z].append(self.roi_overlay.copy())
            self.previous_adjusted_images[self.z].append(self.adjusted_image.copy())
            self.previous_roi_images[self.z].append(self.roi_image.copy())

            if self.selected_roi is not None:
                self.previous_selected_rois[self.z].append(self.selected_roi)

    def undo(self):
        if len(self.previous_labels[self.z]) > 1:
            del self.previous_labels[self.z][-1]

            self.rois[self.z] = self.previous_labels[self.z][-1][:]
        if len(self.previous_roi_overlays[self.z]) > 1:
            del self.previous_roi_overlays[self.z][-1]

            self.roi_overlay = self.previous_roi_overlays[self.z][-1].copy()
        if len(self.previous_erased_rois[self.z]) > 1:
            del self.previous_erased_rois[self.z][-1]

            self.erased_rois[self.z] = self.previous_erased_rois[self.z][-1][:]
        if len(self.previous_adjusted_images[self.z]) > 1:
            del self.previous_adjusted_images[self.z][-1]

            self.adjusted_image  = self.previous_adjusted_images[self.z][-1].copy()
        if len(self.previous_roi_images[self.z]) > 1:
            del self.previous_roi_images[self.z][-1]

            self.roi_image = self.previous_roi_images[self.z][-1].copy()
        if len(self.previous_selected_rois[self.z]) > 1:
            del self.previous_selected_rois[self.z][-1]

            self.selected_roi = self.previous_selected_rois[self.z][-1]
        if len(self.previous_locked_rois[self.z]) > 1:
            del self.previous_locked_rois[self.z][-1]

            self.locked_rois[self.z] = self.previous_locked_rois[self.z][-1][:]

        self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]

        # create & show the new ROI image
        self.calculate_roi_image(z=self.z, update_overlay=False)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def reset_erase(self):
        self.rois[self.z]         = self.original_rois[self.z][:]
        self.removed_rois[self.z] = self.filtered_out_rois[self.z][:]
        self.erased_rois[self.z]  = []

        # reset the history for this z plane
        self.reset_history(z=self.z)

        # create & show the new ROI image
        self.calculate_roi_image(z=self.z, update_overlay=True)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def erase_selected_roi(self): # TODO: call roi_unselected() method of the param window
        # update ROI filtering variables
        self.erased_rois[self.z].append(self.selected_roi)
        self.last_erased_rois[self.z].append([self.selected_roi])
        self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]

        self.selected_roi = None

        # create & show the new ROI image
        self.calculate_roi_image(z=self.z, update_overlay=True)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # update param widget
        self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)

        # add current state to the history
        self.add_to_history()

    def lock_roi(self): # TODO: create roi_locked() and roi_unlocked() methods for the param window
        if self.selected_roi not in self.locked_rois[self.z]:
            self.locked_rois[self.z].append(self.selected_roi)

            # update the param widget
            self.roi_filtering_param_widget.lock_roi_button.setText("Unlock ROI")
        else:
            index = self.locked_rois[self.z].index(self.selected_roi)
            del self.locked_rois[self.z][index]

            # update the param widget
            self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")

        # create & show the new ROI image
        self.calculate_roi_image(z=self.z, update_overlay=True)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def enlarge_roi(self):
        if self.selected_roi >= 1:
            # create a mask of the enlarged ROI
            mask = self.rois[self.z] == self.selected_roi
            mask = binary_dilation(mask, disk(1))

            # draw the new ROI
            self.rois[self.z][mask] = self.selected_roi

            # create & show the new ROI image
            self.calculate_roi_image(z=self.z, update_overlay=True)
            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

            # get the activity of this ROI
            activity = utilities.calc_activity_of_roi(self.rois[self.z], self.video[:, self.z, :, :].transpose(1, 2, 0), self.selected_roi, z=self.z)

            # plot the activity of the ROI
            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.mpl_connect('close_event', self.figure_closed)
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()
            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

            # add current state to the history
            self.add_to_history()

    def shrink_roi(self):
        if self.selected_roi >= 1:
            # create & draw the shrunken ROI
            rois              = self.rois[self.z].copy()
            mask              = self.rois[self.z] == self.selected_roi
            rois[mask]        = 0
            mask              = erosion(mask, disk(1))
            rois[mask]        = self.selected_roi
            self.rois[self.z] = rois.copy()

            # create & show the new ROI image
            self.calculate_roi_image(z=self.z, update_overlay=True)
            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

            # get the activity of this ROI
            activity = utilities.calc_activity_of_roi(self.rois[self.z], self.video[:, self.z, :, :].transpose(1, 2, 0), self.selected_roi, z=self.z)

            # plot the activity of the ROI
            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.mpl_connect('close_event', self.figure_closed)
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()
            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

            # add current state to the history
            self.add_to_history()

    def figure_closed(self, event):
        self.figure = None

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))

class MotionCorrectThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video, video_path, max_shift, patch_stride, patch_overlap, mc_z=-1):
        self.video         = video
        self.video_path    = video_path
        self.max_shift     = max_shift
        self.patch_stride  = patch_stride
        self.patch_overlap = patch_overlap
        self.mc_z          = mc_z

    def run(self):
        self.running = True

        mc_video = utilities.motion_correct(self.video, self.video_path, self.max_shift, self.patch_stride, self.patch_overlap, progress_signal=self.progress, thread=self, mc_z=self.mc_z)

        self.finished.emit(mc_video)

        self.running = False

class ROIFindingThread(QThread):
    finished = pyqtSignal(list, list, list, list)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video, mean_images, masks, min_area, max_area, min_circ, max_circ, min_correlation, soma_threshold, window_size, background_threshold, contrast, gamma, correlation_images):
        self.video                = video
        self.mean_images          = mean_images
        self.masks                = masks
        self.min_area             = min_area
        self.max_area             = max_area
        self.min_circ             = min_circ
        self.max_circ             = max_circ
        self.min_correlation      = min_correlation
        self.soma_threshold       = soma_threshold
        self.window_size          = window_size
        self.background_threshold = background_threshold
        self.contrast             = contrast
        self.gamma                = gamma
        self.correlation_images   = correlation_images

        # print(self.min_area, self.max_area, self.min_circ, self.max_circ)

    def run(self):
        labels            = [ [] for i in range(self.video.shape[1]) ]
        roi_areas         = [ [] for i in range(self.video.shape[1]) ]
        roi_circs         = [ [] for i in range(self.video.shape[1]) ]
        filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]

        self.running = True

        for z in range(self.video.shape[1]):
            adjusted_image  = utilities.calculate_adjusted_image(self.mean_images[z], self.contrast, self.gamma)
            background_mask = utilities.calculate_background_mask(adjusted_image, self.background_threshold)
            equalized_image = utilities.calculate_equalized_image(adjusted_image, background_mask, self.window_size)
            soma_mask, I_mod, soma_threshold_image = utilities.calculate_soma_threshold_image(equalized_image, self.soma_threshold)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(z + (1/3))/self.video.shape[1]))

            if len(self.masks[z]) > 0:
                masks = np.array(self.masks[z])
                mask = np.sum(masks, axis=0).astype(bool)

                out = np.zeros(adjusted_image.shape)
                out[mask] = adjusted_image[mask]
                adjusted_image = out.copy()

                out = np.zeros(soma_mask.shape)
                out[mask] = soma_mask[mask]
                soma_mask = out.copy()

                out = np.zeros(I_mod.shape)
                out[mask] = I_mod[mask]
                I_mod = out.copy()

            labels[z], roi_areas[z], roi_circs[z] = utilities.find_rois(adjusted_image, soma_mask, I_mod)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(z + (2/3))/self.video.shape[1]))

            if len(self.masks[z]) > 0:
                masks = np.array(self.masks[z])
                mask = np.sum(masks, axis=0).astype(bool)

                out = np.zeros(labels[z].shape).astype(int)
                out[mask] = labels[z][mask]
                labels[z] = out.copy()

            _, filtered_out_rois[z] = utilities.filter_rois(self.mean_images[z], labels[z], self.min_area, self.max_area, self.min_circ, self.max_circ, roi_areas[z], roi_circs[z], self.correlation_images[z], self.min_correlation)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(z + 1)/self.video.shape[1]))

        if labels is not None:
            self.finished.emit(labels, roi_areas, roi_circs, filtered_out_rois)

        self.running = False

class ProcessVideosThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video_paths, labels, motion_correct, max_shift, patch_stride, patch_overlap, params):
        self.video_paths    = video_paths
        self.rois         = labels
        self.motion_correct = motion_correct
        self.max_shift      = max_shift
        self.patch_stride   = patch_stride
        self.patch_overlap  = patch_overlap
        self.params         = params

    def run(self):
        self.running = True

        video_shape = None

        first_mean_images = None
        mean_images = None

        for i in range(len(self.video_paths)):
            video_path = self.video_paths[i]

            # open video
            base_name = os.path.basename(video_path)
            if base_name.endswith('.npy'):
                video = np.load(video_path)
            elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
                video = imread(video_path)

            print("Processing {}.".format(base_name))

            if len(video.shape) < 3:
                print("Skipping, this file is not a video -- not enough dimensions.")
                continue

            if len(video.shape) == 3:
                # add z dimension
                video = video[:, np.newaxis, :, :]

            if video_shape is None and not self.motion_correct:
                video_shape = video.shape

            # elif (video.shape[2], video.shape[3]) != video_shape:
            #     print("Skipping {} due to shape mismatch.".format(video_path))
            #     continue

            # print("Loaded video with shape {}.".format(video.shape))

            video = np.nan_to_num(video).astype(np.float32)

            # figure out the dynamic range of the video
            max_value = np.amax(video)
            if max_value > 511:
                video_max = 1023
            elif max_value > 255:
                video_max = 511
            elif max_value > 1:
                video_max = 255
            else:
                video_max = 1

            name = os.path.splitext(base_name)[0]
            directory = os.path.dirname(video_path)
            video_dir_path = os.path.join(directory, name)

            # make a folder to hold the results
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + (1/3))/len(self.video_paths)))

            if self.motion_correct:
                print("Performing motion correction...")
                mc_video = utilities.motion_correct(video, video_path, self.max_shift, self.patch_stride, self.patch_overlap)

                if video_shape is None:
                    video_shape = mc_video.shape

                np.save(os.path.join(video_dir_path, '{}_motion_corrected.npy'.format(name)), mc_video)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + (2/3))/len(self.video_paths)))

            if python_version == 3:
                labels = self.rois.copy()
            else:
                labels = self.rois[:]

            if self.motion_correct:
                vid = mc_video*255.0/max_video
            else:
                vid = video*255.0/max_video

            # print(labels[0].shape, vid.shape, video_shape)

            # if labels[0].shape[0] > vid.shape[2] or labels[0].shape[1] > vid.shape[3]:
            #     print("Cropping labels...")
            #     height_pad =  (labels[0].shape[0] - vid.shape[2])//2
            #     width_pad  =  (labels[0].shape[1] - vid.shape[3])//2

            #     for i in range(len(labels)):
            #         labels[i] = labels[i][height_pad:, width_pad:]
            #         labels[i] = labels[i][:vid.shape[2], :vid.shape[3]]
            # elif labels[0].shape[0] < vid.shape[2] or labels[0].shape[1] < vid.shape[3]:
            #     print("Padding labels...")
            #     height_pad_pre =  (vid.shape[2] - labels[0].shape[0])//2
            #     width_pad_pre  =  (vid.shape[3] - labels[0].shape[1])//2

            #     height_pad_post = vid.shape[2] - labels[0].shape[0] - height_pad_pre
            #     width_pad_post  = vid.shape[3] - labels[0].shape[1] - width_pad_pre

            #     # print(height_pad_pre, height_pad_post, width_pad_pre, width_pad_post)

            #     for i in range(len(labels)):
            #         labels[i] = np.pad(labels[i], ((height_pad_pre, height_pad_post), (width_pad_pre, width_pad_post)), 'constant')

            # print(labels[0].shape, vid.shape, video_shape)

            # shift the labels to match the first video
            mean_images = [ ndi.median_filter(utilities.sharpen(ndi.gaussian_filter(denoise_tv_chambolle(utilities.mean(vid, z).astype(np.float32), weight=0.01, multichannel=False), 1)), 3).astype(np.uint8) for z in range(vid.shape[1]) ]

            for z in range(vid.shape[1]):
                if first_mean_images is not None:
                    y_shift, x_shift = utilities.calculate_shift(first_mean_images[z], mean_images[z])

                    if np.abs(y_shift) < 20 and np.abs(x_shift) < 20:
                        labels[z] = np.roll(labels[z], -y_shift, axis=0)
                        labels[z] = np.roll(labels[z], -x_shift, axis=1)

                        if y_shift >= 0 and x_shift >= 0:
                            labels[z][:y_shift, :] = 0
                            labels[z][:, :x_shift] = 0
                        elif y_shift < 0 and x_shift >= 0:
                            labels[z][y_shift:, :] = 0
                            labels[z][:, :x_shift] = 0
                        elif y_shift >= 0 and x_shift < 0:
                            labels[z][:y_shift, :] = 0
                            labels[z][:, x_shift:] = 0
                        else:
                            labels[z][y_shift:, :] = 0
                            labels[z][:, x_shift:] = 0

                adjusted_image = utilities.calculate_adjusted_image(mean_images[z], self.params['contrast'], self.params['gamma'])

                rgb_image = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2RGB)

                roi_image, _ = utilities.draw_rois(rgb_image, labels[z], None, None, [], None, roi_overlay=None)

                cv2.imwrite(os.path.join(video_dir_path, 'z_{}_rois.png'.format(z)), roi_image)

            np.save(os.path.join(video_dir_path, 'all_rois.npy'), labels)

            if first_mean_images is None:
                first_mean_images = mean_images[:]

            results = [ {} for z in range(video.shape[1]) ]

            for z in range(video.shape[1]):
                np.save(os.path.join(video_dir_path, 'z_{}_rois.npy'.format(z)), labels[z])

                print("Calculating ROI activities for z={}...".format(z))
                vid_z = vid[:, z, :, :].transpose(1, 2, 0)
                for l in np.unique(labels[z]):
                    activity = utilities.calc_activity_of_roi(labels[z], vid_z, l, z=z)

                    results[z][l] = activity

                # add CSV saving here
                print("Saving CSV for z={}...".format(z))
                with open(os.path.join(video_dir_path, 'z_{}_traces.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['ROI #'] + [ 'Frame {}'.format(i) for i in range(video.shape[0]) ])

                    for l in np.unique(self.rois[z])[1:]:
                        writer.writerow([l] + results[z][l].tolist())
                print("Done.")

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + 1)/len(self.video_paths)))

            np.savez(os.path.join(video_dir_path, '{}_roi_traces.npz'.format(os.path.splitext(video_path)[0])), results)

        self.finished.emit()

        self.running = False
