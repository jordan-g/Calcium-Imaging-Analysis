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

DEFAULT_VIEWING_PARAMS = {'gamma'   : 1.0,
                          'contrast': 1.0,
                          'fps'     : 60,
                          'z'       : 0}

DEFAULT_WATERSHED_PARAMS = {'window_size'         : 7,
                            'background_threshold': 10,
                            'invert_masks'        : False,
                            'soma_threshold'      : 1}

DEFAULT_MOTION_CORRECTION_PARAMS = {'max_shift'    : 6,
                                    'patch_stride' : 24,
                                    'patch_overlap': 6}

DEFAULT_ROI_FILTERING_PARAMS = {'min_area'       : 10,
                                'max_area'       : 100,
                                'min_circ'       : 0,
                                'max_circ'       : 2,
                                'min_correlation': 0.2}

VIEWING_PARAMS_FILENAME           = "viewing_params.txt"
WATERSHED_PARAMS_FILENAME         = "watershed_params.txt"
MOTION_CORRECTION_PARAMS_FILENAME = "motion_correction_params.txt"
ROI_FILTERING_PARAMS_FILENAME     = "roi_filtering_params.txt"

class Controller():
    def __init__(self):
        # load parameters
        if os.path.exists(VIEWING_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_VIEWING_PARAMS
                params = json.load(open(VIEWING_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_VIEWING_PARAMS
        else:
            self.params = DEFAULT_VIEWING_PARAMS

        # create controllers
        self.motion_correction_controller = MotionCorrectionController(self)
        self.watershed_controller         = WatershedController(self)
        self.roi_filtering_controller     = ROIFilteringController(self)

        # create windows
        self.param_window   = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

        # create variables
        self.video                     = None  # video that is being previewed
        self.video_path                = None  # path of the video that is being previewed
        self.video_paths               = []    # paths of all videos to process
        self.closing                   = False # whether the controller is in the process of closing
        self.processing_videos         = False # whether videos are currently being processed
        self.motion_correct_all_videos = False # whether to use motion correction when processing videos
        self.process_videos_thread     = None  # thread for processing videos

        # set the mode -- "motion_correct" / "watershed" / "filter"
        self.mode = "motion_correct"

        # set references to param widgets & preview window
        self.param_widget                                = self.param_window.main_param_widget
        self.motion_correction_controller.param_widget   = self.param_window.motion_correction_widget
        self.motion_correction_controller.preview_window = self.preview_window
        self.watershed_controller.param_widget           = self.param_window.watershed_widget
        self.watershed_controller.preview_window         = self.preview_window
        self.roi_filtering_controller.param_widget       = self.param_window.roi_filtering_widget
        self.roi_filtering_controller.preview_window     = self.preview_window

    def select_and_open_video(self):
        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')

            video_paths = [ str(path) for path in video_paths ]
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')[0]

        # open the videos (only the first video is actually opened and previewed, the rest are added to a list of videos to process)
        if video_paths is not None and len(video_paths) > 0:
            self.open_videos(video_paths)

    def save_rois(self):
        if self.watershed_controller.labels[0] is not None:
            # let the user pick where to save the ROIs
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)')[0])
            if not save_path.endswith('.npy'):
                save_path += ".npy"

            # create a dictionary to hold the ROI data
            roi_data = {'labels'           : self.watershed_controller.labels,
                        'roi_areas'        : self.watershed_controller.roi_areas,
                        'roi_circs'        : self.watershed_controller.roi_circs,
                        'filtered_out_rois': self.roi_filtering_controller.filtered_out_rois,
                        'erased_rois'      : self.roi_filtering_controller.erased_rois,
                        'removed_rois'     : self.roi_filtering_controller.removed_rois,
                        'locked_rois'      : self.roi_filtering_controller.locked_rois}

            # save the ROI data
            np.save(save_path, roi_data)

    def load_rois(self):
        # let the user pick saved ROIs
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        # load the saved ROIs
        roi_data = np.load(load_path)[()]

        # stop any motion correction or watershed process
        if self.mode == "motion_correct":
            self.motion_correction_controller.cancel_motion_correction()
        elif self.mode == "watershed":
            self.watershed_controller.cancel_watershed()

        # set parameters of watershed & ROI filtering controllers
        self.watershed_controller.labels                = roi_data['labels']
        self.watershed_controller.roi_areas             = roi_data['roi_areas']
        self.watershed_controller.roi_circs             = roi_data['roi_circs']
        self.roi_filtering_controller.filtered_out_rois = roi_data['filtered_out_rois']
        self.roi_filtering_controller.erased_rois       = roi_data['erased_rois']
        self.roi_filtering_controller.removed_rois      = roi_data['removed_rois']
        self.roi_filtering_controller.locked_rois       = roi_data['locked_rois']

        # show ROI filtering parameters
        self.show_roi_filtering_params(self.watershed_controller.labels, self.watershed_controller.roi_areas, self.watershed_controller.roi_circs, None, None, None, None, loading_rois=True)

    def open_videos(self, video_paths):
        # add the new video paths to the currently loaded video paths
        self.video_paths += video_paths

        # notify the param window
        self.param_window.videos_opened(video_paths)

        if self.video_path is None:
            # open the first video for previewing
            self.open_video(self.video_paths[0])

    def open_video(self, video_path):
        # set the path to the previewed video
        self.video_path = video_path

        # get the shape of the previously-previewed video, if any
        if self.video is None:
            previous_video_shape = None
        else:
            previous_video_shape = self.video.shape

        # open the video
        base_name = os.path.basename(self.video_path)
        if base_name.endswith('.npy'):
            self.video = np.load(self.video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = imread(self.video_path)

        self.video = self.video[1:]

        if len(self.video.shape) == 3:
            # add a z dimension
            self.video = self.video[:, np.newaxis, :, :]

        # set z parameter to 0 if necessary
        if self.params['z'] >= self.video.shape[1]:
            self.params['z'] = 0

        # remove nans
        self.video = np.nan_to_num(self.video).astype(np.float32)

        # calculate normalized video (between 0 and 255)
        self.normalized_video = utilities.normalize(self.video).astype(np.uint8)

        print("Loaded video with shape {}.".format(self.video.shape))

        # update preview window
        self.preview_window.title_label.setText(os.path.basename(self.video_path))

        # update param window
        self.param_window.stacked_widget.setDisabled(False)
        self.param_window.statusBar().showMessage("")
        self.param_widget.param_sliders["z"].setMaximum(self.video.shape[1]-1)

        # if the video is a different shape than the previous one, get rid of any exising roi information
        if previous_video_shape is None or self.video.shape[2] != previous_video_shape[2] or self.video.shape[3] != previous_video_shape[3]:
            clear_progress = True
        else:
            clear_progress = False

        # reset the states of the watershed & roi filtering controllers
        self.watershed_controller.reset_state(clear_progress=clear_progress)
        self.roi_filtering_controller.reset_state(clear_progress=clear_progress)

        # update the motion correction controller
        self.motion_correction_controller.video_opened(self.normalized_video, self.video_path)

    def remove_videos_at_indices(self, indices):
        indices = sorted(indices)
        for i in range(len(indices)-1, -1, -1):
            index = indices[i]
            del self.video_paths[index]

        if len(self.video_paths) == 0:
            if self.mode == "motion_correct":
                self.motion_correction_controller.cancel_motion_correction()

            self.video_path = None
            self.use_mc_video = False

            self.show_motion_correction_params()
            self.param_window.toggle_initial_state(True)
            self.preview_window.timer.stop()
            self.preview_window.title_label.setText("")
            self.preview_window.setWindowTitle("Preview")
            self.preview_window.plot_image(None)
        elif 0 in indices:
            self.open_video(self.video_paths[0])

    def process_all_videos(self):
        if self.mode == "motion_correct":
            self.motion_correction_controller.cancel_motion_correction()
        elif self.mode == "watershed":
            self.watershed_controller.cancel_watershed()

        if self.roi_filtering_controller.labels is not None:
            labels = utilities.filter_labels(self.roi_filtering_controller.labels, self.roi_filtering_controller.removed_rois)
        else:
            labels = utilities.filter_labels(self.watershed_controller.labels, self.watershed_controller.filtered_out_rois)

        if not self.processing_videos:
            if self.process_videos_thread is None:
                self.process_videos_thread = ProcessVideosThread(self.param_window)
                self.process_videos_thread.progress.connect(self.process_videos_progress)
                self.process_videos_thread.finished.connect(self.process_videos_finished)
            else:
                self.process_videos_thread.running = False

            self.process_videos_thread.set_parameters(self.video_paths, labels, self.motion_correct_all_videos, self.motion_correction_controller.params["max_shift"], self.motion_correction_controller.params["patch_stride"], self.motion_correction_controller.params["patch_overlap"])

            self.process_videos_thread.start()

            self.param_window.process_videos_started()

            self.processing_videos = True
        else:
            self.cancel_processing_videos()

    def process_videos_progress(self, percent):
        self.param_window.update_process_videos_progress(percent)

    def process_videos_finished(self):
        self.param_window.update_process_videos_progress(100)

    def cancel_processing_videos(self):
        if self.process_videos_thread is not None:
            self.process_videos_thread.running = False

        self.param_window.update_process_videos_progress(-1)

        self.processing_videos = False
        self.process_videos_thread = None

    def set_motion_correct(self, boolean):
        self.motion_correct_all_videos = boolean

    def show_watershed_params(self, video=None, video_path=None, roi_overlay=None):
        if video is None:
            video = self.normalized_video

        if video_path is None:
            video_path = self.video_path

        self.watershed_controller.filtering_params = self.roi_filtering_controller.params

        self.param_window.stacked_widget.setCurrentIndex(1)
        self.mode = "watershed"
        self.preview_window.controller = self.watershed_controller
        self.watershed_controller.video_opened(video, video_path, roi_overlay)
        self.param_window.statusBar().showMessage("")

        self.preview_window.setWindowTitle("Preview")

    def show_motion_correction_params(self, switched_to=False):
        self.param_window.stacked_widget.setCurrentIndex(0)
        self.mode = "motion_correct"
        self.preview_window.controller = self.motion_correction_controller
        if switched_to:
            self.motion_correction_controller.switched_to()
        else:
            self.motion_correction_controller.video_opened(self.normalized_video, self.video_path)
        self.param_window.statusBar().showMessage("")

    def show_roi_filtering_params(self, labels, roi_areas, roi_circs, mean_images, normalized_images, correlation_images, filtered_out_rois, roi_overlay, loading_rois=False):
        self.param_window.stacked_widget.setCurrentIndex(2)
        self.mode = "filter"
        self.preview_window.controller = self.roi_filtering_controller
        self.roi_filtering_controller.video_opened(self.normalized_video, self.video_path, labels, roi_areas, roi_circs, mean_images, normalized_images, correlation_images, filtered_out_rois, roi_overlay, loading_rois=loading_rois)
        self.param_window.statusBar().showMessage("")

    def rois_created(self):
        self.param_window.rois_created()

    def close_all(self):
        if self.mode == "motion_correct":
            self.motion_correction_controller.cancel_motion_correction()

        self.closing = True
        self.param_window.close()
        self.preview_window.close()

        self.save_params()
        self.motion_correction_controller.save_params()
        self.watershed_controller.save_params()
        self.roi_filtering_controller.save_params()

    def preview_contrast(self, contrast):
        self.params['contrast'] = contrast

        if self.mode == "motion_correct":
            self.motion_correction_controller.preview_contrast(contrast)
        elif self.mode == "watershed":
            self.watershed_controller.update_param("contrast", contrast)
        elif self.mode == "filter":
            self.roi_filtering_controller.update_param("contrast", contrast)

    def preview_gamma(self, gamma):
        self.params['gamma'] = gamma

        if self.mode == "motion_correct":
            self.motion_correction_controller.preview_gamma(gamma)
        elif self.mode == "watershed":
            self.watershed_controller.update_param("gamma", gamma)
        elif self.mode == "filter":
            self.roi_filtering_controller.update_param("gamma", gamma)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if self.mode == "motion_correct":
            self.motion_correction_controller.update_param(param, value)
        elif self.mode == "watershed":
            self.watershed_controller.update_param(param, value)
        elif self.mode == "filter":
            self.roi_filtering_controller.update_param(param, value)

    def save_params(self):
        json.dump(self.params, open(VIEWING_PARAMS_FILENAME, "w"))

class MotionCorrectionController():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(MOTION_CORRECTION_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_MOTION_CORRECTION_PARAMS
                params = json.load(open(MOTION_CORRECTION_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_MOTION_CORRECTION_PARAMS
        else:
            self.params = DEFAULT_MOTION_CORRECTION_PARAMS

        self.video    = None
        self.mc_video = None
        
        self.adjusted_video    = None
        self.adjusted_mc_video = None
        
        self.adjusted_frame    = None
        self.adjusted_mc_frame = None
        
        self.video_path    = None

        self.use_mc_video = False

        self.z = 0

        self.motion_correct_thread        = None
        self.performing_motion_correction = False

    def video_opened(self, video, video_path):
        self.video      = video
        self.video_path = video_path

        self.z = self.main_controller.params['z']

        self.preview_window.timer.stop()

        self.adjusted_video = self.calculate_adjusted_video(self.video, z=self.z)

        self.play_video(self.adjusted_video)

        self.param_widget.use_mc_video_checkbox.setChecked(False)
        self.param_widget.use_mc_video_checkbox.setDisabled(True)

    def switched_to(self):
        if self.z != self.main_controller.params['z']:
            z = self.main_controller.params['z']

            if self.use_mc_video:
                self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, z=self.main_controller.params['z'])
            else:
                self.adjusted_video = self.calculate_adjusted_video(self.video, z=self.main_controller.params['z'])

            self.z = z

        self.preview_window.timer.stop()

        if self.use_mc_video:
            self.play_video(self.adjusted_mc_video)
        else:
            self.play_video(self.adjusted_video)

    def preview_contrast(self, contrast):
        self.preview_window.timer.stop()

        if self.use_mc_video:
            adjusted_frame = self.calculate_adjusted_frame(self.mc_video)
        else:
            adjusted_frame = self.calculate_adjusted_frame(self.video)
          
        self.preview_window.show_frame(adjusted_frame)

    def preview_gamma(self, gamma):
        self.preview_window.timer.stop()

        if self.use_mc_video:
            adjusted_frame = self.calculate_adjusted_frame(self.mc_video)
        else:
            adjusted_frame = self.calculate_adjusted_frame(self.video)

        self.preview_window.show_frame(adjusted_frame)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.preview_window.timer.stop()

            if self.use_mc_video:
                self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
                self.play_video(self.adjusted_mc_video)
            else:
                self.adjusted_video = self.calculate_adjusted_video(self.video, self.z)
                self.play_video(self.adjusted_video)
        elif param == "fps":
            self.preview_window.set_fps(self.main_controller.params['fps'])
        elif param == "z":
            self.z = value

            if self.use_mc_video:
                self.adjusted_video    = None
                self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
            else:
                self.adjusted_mc_video = None
                self.adjusted_video    = self.calculate_adjusted_video(self.video, self.z)

            self.preview_window.timer.stop()

            if self.use_mc_video:
                self.play_video(self.adjusted_mc_video)
            else:
                self.play_video(self.adjusted_video)

    def calculate_adjusted_video(self, video, z=None):
        if z is not None:
            return utilities.adjust_gamma(utilities.adjust_contrast(video[:, z, :, :], self.main_controller.params['contrast']), self.main_controller.params['gamma'])
        else:
            return utilities.adjust_gamma(utilities.adjust_contrast(video, self.main_controller.params['contrast']), self.main_controller.params['gamma'])

    def calculate_adjusted_frame(self, video):
        return utilities.adjust_gamma(utilities.adjust_contrast(video[self.preview_window.frame_num, self.z], self.main_controller.params['contrast']), self.main_controller.params['gamma'])

    def motion_correct_video(self):
        if not self.performing_motion_correction:
            if self.motion_correct_thread is None:
                self.motion_correct_thread = MotionCorrectThread(self.param_widget)
                self.motion_correct_thread.progress.connect(self.motion_correction_progress)
                self.motion_correct_thread.finished.connect(self.motion_correction_finished)
            else:
                self.motion_correct_thread.running = False

            self.motion_correct_thread.set_parameters(self.video, self.video_path, int(self.params["max_shift"]), int(self.params["patch_stride"]), int(self.params["patch_overlap"]))

            self.motion_correct_thread.start()

            self.param_widget.motion_correction_started()

            self.performing_motion_correction = True
        else:
            self.cancel_motion_correction()

    def motion_correction_progress(self, percent):
        self.param_widget.update_motion_correction_progress(percent)

    def motion_correction_finished(self, mc_video):
        self.mc_video      = mc_video

        self.param_widget.update_motion_correction_progress(100)

        self.mc_video = utilities.normalize(self.mc_video).astype(np.uint8)

        self.use_mc_video = True

        self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)

        self.param_widget.use_mc_video_checkbox.setEnabled(True)
        self.param_widget.use_mc_video_checkbox.setChecked(True)

        self.set_use_mc_video(True)

    def cancel_motion_correction(self):
        if self.motion_correct_thread is not None:
            self.motion_correct_thread.running = False

        self.param_widget.update_motion_correction_progress(100)

        self.performing_motion_correction = False

    def play_video(self, video):
        self.preview_window.play_movie(video, fps=self.main_controller.params['fps'])

    def set_use_mc_video(self, use_mc_video):
        self.use_mc_video = use_mc_video

        if self.use_mc_video:
            if self.adjusted_mc_video is None:
                self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)
            self.preview_window.play_movie(self.adjusted_mc_video, fps=self.main_controller.params['fps'])
        else:
            if self.adjusted_video is None:
                self.adjusted_video = self.calculate_adjusted_video(self.video, self.z)
            self.preview_window.play_movie(self.adjusted_video, fps=self.main_controller.params['fps'])

    def accept(self):
        self.cancel_motion_correction()

        self.preview_window.timer.stop()

        if self.use_mc_video:
            self.main_controller.show_watershed_params(video=self.mc_video, video_path=self.video_path)
        else:
            self.main_controller.show_watershed_params(video=self.video, video_path=self.video_path)

    def save_params(self):
        json.dump(self.params, open(MOTION_CORRECTION_PARAMS_FILENAME, "w"))

class WatershedController():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(WATERSHED_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_WATERSHED_PARAMS
                params = json.load(open(WATERSHED_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_WATERSHED_PARAMS
        else:
            self.params = DEFAULT_WATERSHED_PARAMS

        self.reset_state(clear_progress=True)

    def reset_state(self, clear_progress=False):
        self.video      = None
        self.video_path = None

        self.mean_images       = None
        self.normalized_images = None

        self.adjusted_image       = None
        self.background_mask      = None
        self.equalized_image      = None
        self.soma_mask            = None
        self.I_mod                = None
        self.soma_threshold_image = None
        self.watershed_image      = None

        if clear_progress:
            self.roi_overlay          = None
        
            self.masks             = None
            self.mask_points       = None
            self.selected_mask     = None
            self.selected_mask_num = -1
            self.n_masks           = 0

            self.labels            = None
            self.roi_areas         = None
            self.roi_circs         = None
            self.filtered_out_rois = None

        self.z = 0

        self.watershed_thread     = None
        self.performing_watershed = False

    def set_invert_masks(self, boolean):
        self.params['invert_masks'] = boolean

        for i in range(len(self.masks)):
            for j in range(len(self.masks[i])):
                self.masks[i][j] = self.masks[i][j] == False

        self.preview_window.plot_image(self.adjusted_image)

    def video_opened(self, video, video_path, roi_overlay):
        if video_path != self.video_path:
            self.video       = video
            self.video_path  = video_path

            self.z = self.main_controller.params['z']

            self.preview_window.timer.stop()

            self.mean_images = [ ndi.median_filter(utilities.sharpen(ndi.gaussian_filter(denoise_tv_chambolle(utilities.mean(self.video, z).astype(np.float32), weight=0.01, multichannel=False), 1)), 3) for z in range(video.shape[1]) ]
            # self.mean_images = [ utilities.mean(self.video, z).astype(np.float32) for z in range(video.shape[1]) ]

            self.correlation_images = [ utilities.correlation(self.video, z).astype(np.float32) for z in range(video.shape[1]) ]

            self.normalized_images = [ utilities.normalize(mean_image).astype(np.uint8) for mean_image in self.mean_images ]

            self.masks             = [ [] for i in range(video.shape[1]) ]
            self.mask_points       = [ [] for i in range(video.shape[1]) ]
            
            self.adjusted_image       = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])
            self.background_mask      = utilities.calculate_background_mask(self.adjusted_image, self.params['background_threshold'])
            self.equalized_image      = utilities.calculate_equalized_image(self.adjusted_image, self.background_mask, self.params['window_size'])
            self.soma_mask, self.I_mod, self.soma_threshold_image = utilities.calculate_soma_threshold_image(self.equalized_image, self.params['soma_threshold'])

        if roi_overlay is not None:
            self.roi_overlay = roi_overlay
            self.calculate_watershed_image(z=self.z, update_overlay=False)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def draw_mask(self):
        # print(self.preview_window.drawing_mask)
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.adjusted_image)

            self.preview_window.start_drawing_mask()

            self.param_widget.draw_mask_button.setText("Done")
            self.param_widget.draw_mask_button.previous_message = "Draw a mask on the image preview."
            self.param_widget.param_widget.setEnabled(False)
            self.param_widget.button_widget.setEnabled(False)
            self.selected_mask = None
            self.selected_mask_num = -1
            self.param_widget.erase_selected_mask_button.setEnabled(False)
            self.param_widget.draw_mask_button.setEnabled(True)
        else:
            if len(self.preview_window.mask_points) > 0:
                mask_points = self.preview_window.mask_points
                mask_points += [mask_points[0]]
                self.mask_points[self.z].append(mask_points)
                mask_points = np.array(mask_points)

                mask = np.zeros(self.adjusted_image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)

                if self.params['invert_masks']:
                    mask = mask == False

                self.masks[self.z].append(mask)

                self.n_masks += 1

            self.preview_window.end_drawing_mask()
            self.preview_window.plot_image(self.adjusted_image)

            self.param_widget.draw_mask_button.setText("Draw Mask")
            self.param_widget.draw_mask_button.previous_message = ""
            self.param_widget.param_widget.setEnabled(True)
            self.param_widget.button_widget.setEnabled(True)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.adjusted_image = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])

            if self.labels is not None:
                self.calculate_watershed_image(self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())
        elif param == "background_threshold":
            self.background_mask = utilities.calculate_background_mask(self.adjusted_image, self.params['background_threshold'])

            self.param_widget.show_watershed_checkbox.setChecked(False)

            self.preview_window.plot_image(self.adjusted_image, mask=self.background_mask)
        elif param == "window_size":
            self.equalized_image = utilities.calculate_equalized_image(self.adjusted_image, self.background_mask, self.params['window_size'])

            self.param_widget.show_watershed_checkbox.setChecked(False)

            self.preview_window.plot_image(self.equalized_image, mask=None)
        elif param == "z":
            self.z = value

            self.adjusted_image = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])

            if self.labels is not None:
                self.calculate_watershed_image(self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def calculate_watershed_image(self, z, update_overlay=True):
        if update_overlay:
            roi_overlay = None
        else:
            roi_overlay = self.roi_overlay

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[z], None, None, self.filtered_out_rois[z], None, roi_overlay=roi_overlay)

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def process_video(self):
        if not self.performing_watershed:
            if self.watershed_thread is None:
                self.watershed_thread = WatershedThread(self.param_widget)
                self.watershed_thread.progress.connect(self.watershed_progress)
                self.watershed_thread.finished.connect(self.watershed_finished)
            else:
                self.watershed_thread.running = False

            self.watershed_thread.set_parameters(self.video, self.mean_images, self.masks, self.filtering_params["min_area"], self.filtering_params["max_area"], self.filtering_params["min_circ"], self.filtering_params["max_circ"], self.filtering_params['min_correlation'], self.params['soma_threshold'], self.params['window_size'], self.params['background_threshold'], self.main_controller.params['contrast'], self.main_controller.params['gamma'], self.correlation_images)

            self.watershed_thread.start()

            self.param_widget.watershed_started()

            self.performing_watershed = True
        else:
            self.cancel_watershed()

    def watershed_progress(self, percent):
        self.param_widget.update_watershed_progress(percent)

    def watershed_finished(self, labels, roi_areas, roi_circs, filtered_out_rois):
        self.labels            = labels
        self.roi_areas         = roi_areas
        self.roi_circs         = roi_circs
        self.filtered_out_rois = filtered_out_rois

        self.param_widget.update_watershed_progress(100)

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.param_widget.filter_rois_button.setDisabled(False)

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[self.z], None, None, self.filtered_out_rois[self.z], None)

        self.main_controller.rois_created()

        self.show_watershed_image(True)

    def cancel_watershed(self):
        if self.watershed_thread is not None:
            self.watershed_thread.running = False

        self.param_widget.update_watershed_progress(-1)

        self.performing_watershed = False
        self.watershed_thread = None

    def select_mask(self, mask_point):
        selected_mask, selected_mask_num = utilities.get_mask_containing_point(self.masks[self.z], mask_point, inverted=self.params['invert_masks'])

        if selected_mask is not None:
            self.param_widget.erase_selected_mask_button.setEnabled(True)

            self.selected_mask     = selected_mask
            self.selected_mask_num = selected_mask_num
        else:
            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def erase_selected_mask(self):
        if self.selected_mask is not None:
            del self.masks[self.z][self.selected_mask_num]
            del self.mask_points[self.z][self.selected_mask_num]

            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)
            
            self.preview_window.plot_image(self.adjusted_image)

    def motion_correct(self):
        self.cancel_watershed()

        self.main_controller.show_motion_correction_params(switched_to=True)

    def filter_rois(self):
        self.cancel_watershed()

        self.main_controller.show_roi_filtering_params(self.labels, self.roi_areas, self.roi_circs, self.mean_images, self.normalized_images, self.correlation_images, self.filtered_out_rois, self.roi_overlay)

    def save_params(self):
        json.dump(self.params, open(WATERSHED_PARAMS_FILENAME, "w"))

class ROIFilteringController():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(ROI_FILTERING_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_ROI_FILTERING_PARAMS
                params = json.load(open(ROI_FILTERING_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_ROI_FILTERING_PARAMS
        else:
            self.params = DEFAULT_ROI_FILTERING_PARAMS

        self.reset_state(clear_progress=True)

    def reset_state(self, clear_progress=False):
        self.mean_images       = None
        self.normalized_images = None

        self.adjusted_image  = None
        self.watershed_image = None

        if clear_progress:
            self.roi_overlay     = None

            self.original_labels   = None
            self.labels            = None
            self.roi_areas         = None
            self.roi_circs         = None
            self.selected_roi      = None

            self.filtered_out_rois = None
            self.erased_rois       = None
            self.removed_rois      = None
            self.last_erased_rois  = None
            self.locked_rois       = None

        self.figure = None
        self.axis   = None

        self.z = 0

    def video_opened(self, video, video_path, labels, roi_areas, roi_circs, mean_images, normalized_images, correlation_images, filtered_out_rois, roi_overlay, loading_rois=False):
        if labels is not self.original_labels:
            self.video      = video
            self.video_path = video_path

            self.z = self.main_controller.params['z']

            self.preview_window.timer.stop()
            
            if mean_images is not None:
                self.mean_images = mean_images
            else:
                self.mean_images = [ ndi.median_filter(denoise_tv_chambolle(utilities.mean(self.video, z).astype(np.float32), weight=0.01, multichannel=False), 3) for z in range(video.shape[1]) ]

            if normalized_images is not None:
                self.normalized_images = normalized_images
            else:
                self.normalized_images = [ utilities.normalize(mean_image).astype(np.uint8) for mean_image in self.mean_images ]

            if correlation_images is not None:
                self.correlation_images = correlation_images
            else:
                self.correlation_images = [ utilities.correlation(self.video, z).astype(np.float32) for z in range(video.shape[1]) ]

            if python_version == 3:
                self.original_labels   = labels.copy()
                self.labels            = self.original_labels.copy()
            else:
                self.original_labels   = labels[:]
                self.labels            = self.original_labels[:]
            self.roi_areas         = roi_areas
            self.roi_circs         = roi_circs
            self.filtered_out_rois = filtered_out_rois
            self.roi_overlay       = roi_overlay

            if not loading_rois:
                self.erased_rois = [ [] for i in range(video.shape[1]) ]
                self.locked_rois = [ [] for i in range(video.shape[1]) ]

                if self.filtered_out_rois is None:
                    if python_version == 3:
                        self.filtered_out_rois = filtered_out_rois.copy()
                    else:
                        self.filtered_out_rois = filtered_out_rois[:]

                if self.removed_rois is None:
                    if python_version == 3:
                        self.removed_rois = filtered_out_rois.copy()
                    else:
                        self.removed_rois = filtered_out_rois[:]
            
            self.last_erased_rois  = [ [] for i in range(video.shape[1]) ]

            self.previous_labels            = [ [] for i in range(video.shape[1]) ]
            self.previous_roi_overlays      = [ [] for i in range(video.shape[1]) ]
            self.previous_erased_rois       = [ [] for i in range(video.shape[1]) ]
            self.previous_filtered_out_rois = [ [] for i in range(video.shape[1]) ]
            self.previous_adjusted_images   = [ [] for i in range(video.shape[1]) ]
            self.previous_watershed_images  = [ [] for i in range(video.shape[1]) ]
            self.previous_selected_rois     = [ [] for i in range(video.shape[1]) ]
            self.previous_removed_rois      = [ [] for i in range(video.shape[1]) ]
            self.previous_locked_rois       = [ [] for i in range(video.shape[1]) ]
            self.previous_params            = [ [] for i in range(video.shape[1]) ]

            self.rois_erased = False

            self.adjusted_image = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])

            if self.filtered_out_rois is None:
                self.filter_rois(z=self.z)

            self.calculate_watershed_image(z=self.z, update_overlay=self.roi_overlay is None)

            self.param_widget.show_watershed_checkbox.setDisabled(False)
            self.param_widget.show_watershed_checkbox.setChecked(True)

            self.show_watershed_image(True)

            self.add_to_history()

    def calculate_adjusted_image(self, normalized_image):
        return utilities.adjust_gamma(utilities.adjust_contrast(normalized_image, self.main_controller.params['contrast']), self.main_controller.params['gamma'])/255.0

    def calculate_watershed_image(self, z, update_overlay=True, newly_erased_rois=None):
        if update_overlay:
            roi_overlay = None
        else:
            roi_overlay = self.roi_overlay

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[z], self.selected_roi, self.erased_rois[z], self.filtered_out_rois[z], self.locked_rois[z], newly_erased_rois=newly_erased_rois, roi_overlay=roi_overlay)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.adjusted_image = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])

            self.calculate_watershed_image(self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())
        elif param == "z":
            self.z = value

            self.adjusted_image = utilities.calculate_adjusted_image(self.normalized_images[self.z], self.main_controller.params['contrast'], self.main_controller.params['gamma'])

            self.filter_rois(z=self.z)

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            self.add_to_history()
        elif param in ("min_area", "max_area", "min_circ", "max_circ", "min_correlation"):
            pass

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def filter_rois(self, z, update_overlay=False):
        _, self.filtered_out_rois[z] = utilities.filter_rois(self.mean_images[z], self.labels[z], self.params['min_area'], self.params['max_area'], self.params['min_circ'], self.params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.correlation_images[z], self.params['min_correlation'], self.locked_rois[z])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

        if update_overlay:
            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def draw_rois(self):
        if not self.preview_window.drawing_rois:
            self.preview_window.drawing_rois = True

            self.main_controller.param_window.roi_drawing_started()

            self.param_widget.draw_rois_button.setText("Finished")
        else:
            self.preview_window.drawing_rois = False

            self.main_controller.param_window.roi_drawing_ended()

            self.param_widget.draw_rois_button.setText("Draw")

    def create_roi(self, start_point, end_point):
        center_point = (int(round((start_point[0] + end_point[0])/2)), int(round((start_point[1] + end_point[1])/2)))
        axis_1 = np.abs(center_point[0] - end_point[0])
        axis_2 = np.abs(center_point[1] - end_point[1])

        l = np.amax(self.labels[self.z])+1

        # print(l)
        # print(self.labels[self.z].shape, self.labels[self.z].dtype)

        mask = np.zeros(self.labels[self.z].shape).astype(np.uint8)

        # add to ROI mask
        cv2.ellipse(mask, center_point, (axis_1, axis_2), 0, 0, 360, 1, -1)

        self.labels[self.z][mask == 1] = l

        utilities.add_roi_to_overlay(self.roi_overlay, self.labels[self.z] == l, self.labels[self.z])

        self.locked_rois.append(l)

        self.calculate_watershed_image(z=self.z, update_overlay=False)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

        self.add_to_history()

    def erase_rois(self):
        self.rois_erased = False

        if not self.preview_window.erasing_rois:
            self.preview_window.erasing_rois = True

            self.selected_roi = None

            self.main_controller.param_window.roi_erasing_started()

            self.param_widget.erase_rois_button.setText("Finished")
        else:
            self.preview_window.erasing_rois = False

            self.main_controller.param_window.roi_erasing_ended()

            self.param_widget.erase_rois_button.setText("Erase ROIs")

            self.add_to_history()

    def erase_rois_near_point(self, roi_point, radius=10):
        if not self.rois_erased:
            self.last_erased_rois[self.z].append([])
            self.rois_erased = True

        rois_to_erase = utilities.get_rois_near_point(self.labels[self.z], roi_point, radius)

        for i in range(len(rois_to_erase)):
            roi = rois_to_erase[i]
            if roi in self.locked_rois or roi in self.erased_rois:
                del rois_to_erase[i]

        if len(rois_to_erase) > 0:
            self.erased_rois[self.z] += rois_to_erase
            self.last_erased_rois[self.z][-1] += rois_to_erase
            self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]
            
            self.calculate_watershed_image(z=self.z, update_overlay=False, newly_erased_rois=rois_to_erase)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def select_roi(self, roi_point):
        selected_roi = utilities.get_roi_containing_point(self.labels[self.z], roi_point)

        if selected_roi is not None and selected_roi not in self.removed_rois[self.z]:
            self.param_widget.lock_roi_button.setEnabled(True)
            self.param_widget.enlarge_roi_button.setEnabled(True)
            self.param_widget.shrink_roi_button.setEnabled(True)
            if selected_roi in self.locked_rois[self.z]:
                self.param_widget.lock_roi_button.setText("Unlock ROI")
            else:
                self.param_widget.lock_roi_button.setText("Lock ROI")

            self.param_widget.erase_selected_roi_button.setEnabled(True)

            self.selected_roi = selected_roi

            self.calculate_watershed_image(z=self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            activity = utilities.calc_activity_of_roi(self.labels[self.z], self.video, self.selected_roi, z=self.z)

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
            self.selected_roi = -1

            self.calculate_watershed_image(z=self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            self.param_widget.lock_roi_button.setEnabled(False)
            self.param_widget.enlarge_roi_button.setEnabled(False)
            self.param_widget.shrink_roi_button.setEnabled(False)
            self.param_widget.lock_roi_button.setText("Lock ROI")

    def add_to_history(self):
        print("Adding to history")
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
        if len(self.previous_watershed_images[self.z]) > 20:
            del self.previous_watershed_images[self.z][0]
        if len(self.previous_selected_rois[self.z]) > 20:
            del self.previous_selected_rois[self.z][0]
        if len(self.previous_removed_rois[self.z]) > 20:
            del self.previous_removed_rois[self.z][0]
        if len(self.previous_locked_rois[self.z]) > 20:
            del self.previous_locked_rois[self.z][0]

        if python_version == 3:
            self.previous_labels[self.z].append(self.labels[self.z].copy())
            self.previous_erased_rois[self.z].append(self.erased_rois[self.z].copy())
            self.previous_filtered_out_rois[self.z].append(self.filtered_out_rois[self.z].copy())
            self.previous_removed_rois[self.z].append(self.removed_rois[self.z].copy())
            self.previous_locked_rois[self.z].append(self.locked_rois[self.z].copy())
        else:
            self.previous_labels[self.z].append(self.labels[self.z][:])
            self.previous_erased_rois[self.z].append(self.erased_rois[self.z][:])
            self.previous_filtered_out_rois[self.z].append(self.filtered_out_rois[self.z][:])
            self.previous_removed_rois[self.z].append(self.removed_rois[self.z][:])
            self.previous_locked_rois[self.z].append(self.locked_rois[self.z][:])
        
        self.previous_roi_overlays[self.z].append(self.roi_overlay.copy())
        self.previous_adjusted_images[self.z].append(self.adjusted_image.copy())
        self.previous_watershed_images[self.z].append(self.watershed_image.copy())
        if self.selected_roi is not None:
            self.previous_selected_rois[self.z].append(self.selected_roi.copy())

    def undo(self):
        if len(self.previous_labels[self.z]) > 1:
            del self.previous_labels[self.z][-1]

            if python_version == 3:
                self.labels[self.z] = self.previous_labels[self.z][-1].copy()
            else:
                self.labels[self.z] = self.previous_labels[self.z][-1][:]
        if len(self.previous_roi_overlays[self.z]) > 1:
            del self.previous_roi_overlays[self.z][-1]

            self.roi_overlay = self.previous_roi_overlays[self.z][-1].copy()
        if len(self.previous_erased_rois[self.z]) > 1:
            del self.previous_erased_rois[self.z][-1]

            if python_version == 3:
                self.erased_rois[self.z] = self.previous_erased_rois[self.z][-1].copy()
            else:
                self.erased_rois[self.z] = self.previous_erased_rois[self.z][-1][:]
        if len(self.previous_adjusted_images[self.z]) > 1:
            del self.previous_adjusted_images[self.z][-1]

            self.adjusted_image  = self.previous_adjusted_images[self.z][-1].copy()
        if len(self.previous_watershed_images[self.z]) > 1:
            del self.previous_watershed_images[self.z][-1]

            self.watershed_image = self.previous_watershed_images[self.z][-1].copy()
        if len(self.previous_selected_rois[self.z]) > 1:
            del self.previous_selected_rois[self.z][-1]

            self.selected_roi = self.previous_selected_rois[self.z][-1].copy()
        if len(self.previous_locked_rois[self.z]) > 1:
            del self.previous_locked_rois[self.z][-1]

            if python_version == 3:
                self.locked_rois[self.z] = self.previous_locked_rois[self.z][-1].copy()
            else:
                self.locked_rois[self.z] = self.previous_locked_rois[self.z][-1][:]

        # print(self.erased_rois)

        self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]

        self.calculate_watershed_image(z=self.z, update_overlay=False)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def undo_erase(self):
        self.undo()

    def reset_erase(self):
        if pyqt_version == 3:
            self.labels[self.z]           = self.original_labels[self.z].copy()
            self.removed_rois[self.z]     = self.filtered_out_rois[self.z].copy()
        else:
            self.labels[self.z]           = self.original_labels[self.z][:]
            self.removed_rois[self.z]     = self.filtered_out_rois[self.z][:]
        self.erased_rois[self.z]      = []
        self.last_erased_rois[self.z] = []

        self.calculate_watershed_image(z=self.z, update_overlay=True)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def erase_selected_roi(self):
        self.erased_rois[self.z].append(self.selected_roi)
        self.last_erased_rois[self.z].append([self.selected_roi])
        self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]
        self.selected_roi = None

        self.calculate_watershed_image(z=self.z, update_overlay=True)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

        self.param_widget.erase_selected_roi_button.setEnabled(False)

        self.add_to_history()

    def lock_roi(self):
        if self.selected_roi not in self.locked_rois[self.z]:
            self.locked_rois[self.z].append(self.selected_roi)
            self.param_widget.lock_roi_button.setText("Unlock ROI")
        else:
            index = self.locked_rois[self.z].index(self.selected_roi)
            del self.locked_rois[self.z][index]
            self.param_widget.lock_roi_button.setText("Lock ROI")

        self.calculate_watershed_image(z=self.z, update_overlay=True)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

        self.add_to_history()

    def enlarge_roi(self):
        if self.selected_roi >= 1:
            mask = self.labels[self.z] == self.selected_roi
            mask = binary_dilation(mask, disk(1))

            self.labels[self.z][mask] = self.selected_roi

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            activity = utilities.calc_activity_of_roi(self.labels[self.z], self.video, self.selected_roi, z=self.z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.mpl_connect('close_event', self.figure_closed)
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

            self.add_to_history()

    def shrink_roi(self):
        if self.selected_roi >= 1:
            labels = self.labels[self.z].copy()
            mask = self.labels[self.z] == self.selected_roi
            labels[mask] = 0

            mask = erosion(mask, disk(1))
            labels[mask] = self.selected_roi

            self.labels[self.z] = labels.copy()

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            activity = utilities.calc_activity_of_roi(self.labels[self.z], self.video, self.selected_roi, z=self.z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.mpl_connect('close_event', self.figure_closed)
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

            self.add_to_history()

    def figure_closed(self, event):
        self.figure = None

    def motion_correct(self):
        self.main_controller.show_motion_correction_params(switched_to=True)

    def watershed(self):
        self.main_controller.show_watershed_params(roi_overlay=self.roi_overlay)

    def save_params(self):
        json.dump(self.params, open(ROI_FILTERING_PARAMS_FILENAME, "w"))

class MotionCorrectThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video, video_path, max_shift, patch_stride, patch_overlap):
        self.video         = video
        self.video_path    = video_path
        self.max_shift     = max_shift
        self.patch_stride  = patch_stride
        self.patch_overlap = patch_overlap

    def run(self):
        self.running = True

        mc_video = utilities.motion_correct(self.video, self.video_path, self.max_shift, self.patch_stride, self.patch_overlap, progress_signal=self.progress, thread=self)

        if mc_video is not None:
            self.finished.emit(mc_video)

        self.running = False

class WatershedThread(QThread):
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

            labels[z], roi_areas[z], roi_circs[z] = utilities.apply_watershed(adjusted_image, soma_mask, I_mod)

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

    def set_parameters(self, video_paths, labels, motion_correct, max_shift, patch_stride, patch_overlap):
        self.video_paths    = video_paths
        self.labels         = labels
        self.motion_correct = motion_correct
        self.max_shift      = max_shift
        self.patch_stride   = patch_stride
        self.patch_overlap  = patch_overlap

    def run(self):
        self.running = True

        video_shape = None

        for i in range(len(self.video_paths)):
            video_path = self.video_paths[i]

            # open video
            base_name = os.path.basename(video_path)
            if base_name.endswith('.npy'):
                video = np.load(video_path)
            elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
                video = imread(video_path)

            print("Processing {}.".format(base_name))

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

            results = [ {} for z in range(video.shape[1]) ]

            if python_version == 3:
                labels = self.labels.copy()
            else:
                labels = self.labels[:]

            if self.motion_correct:
                vid = mc_video
            else:
                vid = video

            # print(labels[0].shape, vid.shape, video_shape)

            if labels[0].shape[0] > vid.shape[2] or labels[0].shape[1] > vid.shape[3]:
                print("Cropping labels...")
                height_pad =  (labels[0].shape[0] - vid.shape[2])//2
                width_pad  =  (labels[0].shape[1] - vid.shape[3])//2

                for i in range(len(labels)):
                    labels[i] = labels[i][height_pad:, width_pad:]
                    labels[i] = labels[i][:vid.shape[2], :vid.shape[3]]
            elif labels[0].shape[0] < vid.shape[2] or labels[0].shape[1] < vid.shape[3]:
                print("Padding labels...")
                height_pad_pre =  (vid.shape[2] - labels[0].shape[0])//2
                width_pad_pre  =  (vid.shape[3] - labels[0].shape[1])//2

                height_pad_post = vid.shape[2] - labels[0].shape[0] - height_pad_pre
                width_pad_post  = vid.shape[3] - labels[0].shape[1] - width_pad_pre

                # print(height_pad_pre, height_pad_post, width_pad_pre, width_pad_post)

                for i in range(len(labels)):
                    labels[i] = np.pad(labels[i], ((height_pad_pre, height_pad_post), (width_pad_pre, width_pad_post)), 'constant')

            # print(labels[0].shape, vid.shape, video_shape)

            for z in range(video.shape[1]):
                np.save(os.path.join(video_dir_path, 'z_{}_rois.npy'.format(z)), labels[z])

                print("Calculating ROI activities for z={}...".format(z))
                for l in np.unique(labels[z]):
                    activity = utilities.calc_activity_of_roi(labels[z], vid, l, z=z)

                    results[z][l] = activity

                # add CSV saving here
                print("Saving CSV for z={}...".format(z))
                with open(os.path.join(video_dir_path, 'z_{}_traces.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['ROI #'] + [ 'Frame {}'.format(i) for i in range(video.shape[0]) ])

                    for l in np.unique(self.labels[z]):
                        writer.writerow([l] + results[z][l].tolist())
                print("Done.")

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + 1)/len(self.video_paths)))

            np.savez(os.path.join(video_dir_path, '{}_roi_traces.npz'.format(os.path.splitext(video_path)[0])), results)

        self.finished.emit()

        self.running = False
