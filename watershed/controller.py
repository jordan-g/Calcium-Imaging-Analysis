from watershed.param_window import ParamWindow
from watershed.preview_window import PreviewWindow
from watershed.preview_widget import PreviewWidget
from skimage.morphology import *
import watershed.utilities as utilities
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
from watershed.imimposemin import imimposemin
from mahotas.labeled import bwperim
import cv2
import math
import matplotlib.pyplot as plt

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

DEFAULT_VIEWING_PARAMS = {'gamma': 1.0,
                          'contrast': 1.0,
                          'fps': 60,
                          'z': 0}

DEFAULT_WATERSHED_PARAMS = {'window_size': 5,
                            'neuropil_threshold': 50,
                            'soma_threshold': 10,
                            'background_threshold': 10,
                            'compactness': 10}

DEFAULT_MOTION_CORRECTION_PARAMS = {'max_shift': 6,
                                    'patch_stride': 24,
                                    'patch_overlap': 6}

DEFAULT_ROI_FILTERING_PARAMS = {'min_area': 10,
                                'max_area': 100,
                                'min_circ': 0,
                                'max_circ': 2}

VIEWING_PARAMS_FILENAME           = "viewing_params.txt"
WATERSHED_PARAMS_FILENAME         = "watershed_params.txt"
MOTION_CORRECTION_PARAMS_FILENAME = "motion_correction_params.txt"
ROI_filtering_PARAMS_FILENAME     = "roi_filtering_params.txt"

class Controller():
    def __init__(self):
        # set parameters
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

        self.mode = "motion_correct"

        self.param_widget                                = self.param_window.main_param_widget
        self.motion_correction_controller.param_widget   = self.param_window.motion_correction_widget
        self.motion_correction_controller.preview_window = self.preview_window
        self.watershed_controller.param_widget           = self.param_window.watershed_widget
        self.watershed_controller.preview_window         = self.preview_window
        self.roi_filtering_controller.param_widget       = self.param_window.roi_filtering_widget
        self.roi_filtering_controller.preview_window     = self.preview_window
        self.watershed_controller.filtering_params       = self.roi_filtering_controller.params

        self.closing = False

    def select_and_open_video(self):
        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')[0]

        # open the first selected video (note: needs to be updates to process batches of videos)
        if video_paths is not None and len(video_paths) > 0:
            self.open_video(video_paths[0])

    def save_rois(self):
        if self.watershed_controller.labels[0] is not None:
            if pyqt_version == 4:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)'))
            elif pyqt_version == 5:
                save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.video_path)[0]), 'Numpy (*.npy)')[0])
            if not save_path.endswith('.npy'):
                save_path += ".npy"

            roi_data = {'labels': self.watershed_controller.labels,
                        'roi_areas': self.watershed_controller.roi_areas,
                        'roi_circs': self.watershed_controller.roi_circs,
                        'filtered_labels': self.roi_filtering_controller.filtered_labels,
                        'filtered_out_rois': self.roi_filtering_controller.filtered_out_rois,
                        'erased_rois': self.roi_filtering_controller.erased_rois,
                        'removed_rois': self.roi_filtering_controller.removed_rois,
                        'locked_rois': self.roi_filtering_controller.locked_rois }

            np.save(save_path, roi_data)

    def load_rois(self):
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        roi_data = np.load(load_path)[()]

        self.watershed_controller.labels                = roi_data['labels']
        self.watershed_controller.roi_areas             = roi_data['roi_areas']
        self.watershed_controller.roi_circs             = roi_data['roi_circs']
        self.roi_filtering_controller.filtered_labels   = roi_data['filtered_labels']
        self.roi_filtering_controller.filtered_out_rois = roi_data['filtered_out_rois']
        self.roi_filtering_controller.erased_rois       = roi_data['erased_rois']
        self.roi_filtering_controller.removed_rois      = roi_data['removed_rois']
        self.roi_filtering_controller.locked_rois       = roi_data['locked_rois']

        self.show_roi_filtering_params(self.watershed_controller.labels, self.watershed_controller.roi_areas, self.watershed_controller.roi_circs, None)

    def open_video(self, video_path):
        self.video_path = video_path

        # open video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.npy'):
            self.video = np.load(video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = imread(video_path)

        # imsave("test.tif", self.video)

        if len(self.video.shape) == 3:
            # add z dimension
            self.video = self.video[:, np.newaxis, :, :]

        if self.params['z'] >= self.video.shape[1]:
            self.params['z'] = 0

        print("Loaded video with shape {}.".format(self.video.shape))

        self.param_widget.param_sliders["z"].setMaximum(self.video.shape[1]-1)

        self.video = np.nan_to_num(self.video).astype(np.float32)

        self.normalized_video = utilities.normalize(self.video).astype(np.uint8)

        self.param_window.stacked_widget.setDisabled(False)
        self.param_window.statusBar().showMessage("")

        # update the motion correction controller
        self.motion_correction_controller.video_opened(self.video, self.normalized_video, self.video_path, plot=True)

    def show_watershed_params(self, video=None, video_path=None):
        if video is None:
            video = self.video

        if video_path is None:
            video_path = self.video_path
        self.param_window.stacked_widget.setCurrentIndex(1)
        self.watershed_controller.video_opened(video, video_path, plot=True)
        self.preview_window.controller = self.watershed_controller
        self.param_window.statusBar().showMessage("")

        self.preview_window.setWindowTitle("Preview")
        self.mode = "watershed"

    def show_motion_correction_params(self):
        self.param_window.stacked_widget.setCurrentIndex(0)
        self.motion_correction_controller.video_opened(self.video, self.normalized_video, self.video_path, plot=True)
        self.preview_window.controller = self.motion_correction_controller
        self.param_window.statusBar().showMessage("")

        self.mode = "motion_correct"

    def show_roi_filtering_params(self, labels, roi_areas, roi_circs, roi_overlay):
        self.param_window.stacked_widget.setCurrentIndex(2)
        self.roi_filtering_controller.video_opened(self.video, self.video_path, labels, roi_areas, roi_circs, roi_overlay, plot=True)
        self.preview_window.controller = self.roi_filtering_controller
        self.param_window.statusBar().showMessage("")

        self.mode = "filter"

    def close_all(self):
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

        self.video                             = None
        self.normalized_video                  = None
        self.adjusted_video                    = None
        self.adjusted_frame                    = None
        self.motion_corrected_video            = None
        self.normalized_motion_corrected_video = None
        self.current_video                     = None
        self.current_adjusted_video            = None
        self.adjusted_motion_corrected_video   = None
        self.video_path                        = None
        self.motion_corrected_video_path       = None

        self.showing_motion_corrected_video = False

    def video_opened(self, video, normalized_video, video_path, plot=False):
        self.video            = video
        self.normalized_video = normalized_video
        self.video_path       = video_path

        self.preview_window.timer.stop()

        self.calculate_adjusted_video()

        if plot:
            self.play_adjusted_video()

    def preview_contrast(self, contrast):
        self.preview_window.timer.stop()

        self.calculate_adjusted_frame()
          
        self.preview_window.show_frame(self.adjusted_frame)

    def preview_gamma(self, gamma):
        self.preview_window.timer.stop()

        self.calculate_adjusted_frame()

        self.preview_window.show_frame(self.adjusted_frame)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.preview_window.timer.stop()

            self.calculate_adjusted_video()

            self.preview_window.play_movie(self.adjusted_video, fps=self.main_controller.params['fps'])
        elif param == "fps":
            self.preview_window.set_fps(self.main_controller.params['fps'])
        elif param == "z":
            self.preview_window.timer.stop()

            self.preview_window.play_movie(self.adjusted_video, fps=self.main_controller.params['fps'])

    def calculate_adjusted_video(self):
        if self.showing_motion_corrected_video:
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_motion_corrected_video, self.main_controller.params['contrast']), self.main_controller.params['gamma'])
        else:
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_video, self.main_controller.params['contrast']), self.main_controller.params['gamma'])

    def calculate_adjusted_frame(self):
        if self.showing_motion_corrected_video:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_motion_corrected_video[self.preview_window.frame_num, self.main_controller.params['z']], self.main_controller.params['contrast']), self.main_controller.params['gamma'])
        else:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_video[self.preview_window.frame_num, self.main_controller.params['z']], self.main_controller.params['contrast']), self.main_controller.params['gamma'])

    def process_video(self):
        self.motion_corrected_video, self.motion_corrected_video_path = utilities.motion_correct(self.video_path, int(self.params["max_shift"]), int(self.params["patch_stride"]), int(self.params["patch_overlap"]))

        self.normalized_motion_corrected_video = utilities.normalize(self.motion_corrected_video).astype(np.uint8)

        self.showing_motion_corrected_video = True
        self.calculate_adjusted_video()

        self.param_widget.play_motion_corrected_video_checkbox.setEnabled(True)
        self.param_widget.play_motion_corrected_video_checkbox.setChecked(True)
        self.play_motion_corrected_video(True)

        self.param_widget.accept_button.setEnabled(True)

    def play_adjusted_video(self):
        if self.adjusted_video is not None:
            self.preview_window.play_movie(self.adjusted_video, fps=self.main_controller.params['fps'])

    def play_motion_corrected_video(self, show):
        self.showing_motion_corrected_video = show

        self.calculate_adjusted_video()
        self.preview_window.play_movie(self.adjusted_video, fps=self.main_controller.params['fps'])

    def skip(self):
        self.preview_window.timer.stop()
        self.main_controller.show_watershed_params()

    def accept(self):
        self.preview_window.timer.stop()
        self.main_controller.show_watershed_params(video=self.motion_corrected_video, video_path=self.motion_corrected_video_path)

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

        self.mean_images           = None
        self.normalized_images     = None
        self.adjusted_images       = None
        self.equalized_images      = None
        self.soma_masks            = None
        self.I_mods                = None
        self.soma_threshold_images = None
        self.masks                 = None
        self.mask_points           = None
        self.background_masks      = None
        self.selected_mask         = None
        self.labels                = None
        self.roi_areas             = None
        self.roi_circs             = None
        self.watershed_images      = None
        self.roi_overlays          = None
        self.selected_mask_num     = -1
        self.n_masks               = 0

    def video_opened(self, video, video_path, plot=False):
        self.video       = video
        self.video_path  = video_path
        
        self.mean_images = [ ndi.median_filter(denoise_tv_chambolle(utilities.normalize(utilities.mean(video, i)).astype(np.float32), weight=0.01, multichannel=False), 3) for i in range(video.shape[1]) ]

        self.normalized_images = [ utilities.normalize(mean_image).astype(np.uint8) for mean_image in self.mean_images ]

        self.adjusted_images       = [ [] for i in range(video.shape[1]) ]
        self.background_masks      = [ [] for i in range(video.shape[1]) ]
        self.equalized_images      = [ [] for i in range(video.shape[1]) ]
        self.soma_masks            = [ [] for i in range(video.shape[1]) ]
        self.I_mods                = [ [] for i in range(video.shape[1]) ]
        self.soma_threshold_images = [ [] for i in range(video.shape[1]) ]
        self.masks                 = [ [] for i in range(video.shape[1]) ]
        self.mask_points           = [ [] for i in range(video.shape[1]) ]
        self.labels                = [ [] for i in range(video.shape[1]) ]
        self.roi_areas             = [ [] for i in range(video.shape[1]) ]
        self.roi_circs             = [ [] for i in range(video.shape[1]) ]
        self.watershed_images      = [ [] for i in range(video.shape[1]) ]
        self.roi_overlays          = [ [] for i in range(video.shape[1]) ]
        
        self.calculate_adjusted_images(z_vals=range(self.video.shape[1]))
        self.calculate_background_masks(z_vals=range(self.video.shape[1]))
        self.calculate_equalized_images(z_vals=range(self.video.shape[1]))
        self.calculate_soma_threshold_images(z_vals=range(self.video.shape[1]))

        if plot:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def calculate_background_masks(self, z_vals=[0]):
        for z in z_vals:
            self.background_masks[z] = self.adjusted_images[z] < self.params['background_threshold']/255.0

    def calculate_adjusted_images(self, z_vals=[0]):
        for z in z_vals:
            self.adjusted_images[z] = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_images[z], self.main_controller.params['contrast']), self.main_controller.params['gamma'])/255.0
    
    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

            self.preview_window.drawing_mask = True

            self.param_widget.draw_mask_button.setText("Done Drawing Mask")
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
                self.mask_points[self.main_controller.params['z']].append(mask_points)
                mask_points = np.array(mask_points)

                mask = np.zeros(self.mean_image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)
                self.masks[self.main_controller.params['z']].append(mask)

                self.calculate_adjusted_images()
                self.calculate_equalized_images()
                self.calculate_soma_threshold_images()

                self.n_masks += 1

            self.preview_window.end_drawing_mask()
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

            self.param_widget.draw_mask_button.setText("Draw Mask")
            self.param_widget.draw_mask_button.previous_message = ""
            self.param_widget.param_widget.setEnabled(True)
            self.param_widget.button_widget.setEnabled(True)

    def calculate_equalized_images(self, z_vals=[0]):
        for z in z_vals:
            adjusted_image  = self.adjusted_images[z]
            background_mask = self.background_masks[z]

            new_image_10 = utilities.order_statistic(adjusted_image, 0.1, int(self.params['window_size']))
            new_image_90 = utilities.order_statistic(adjusted_image, 0.9, int(self.params['window_size']))

            image_difference = adjusted_image - new_image_10
            image_difference[image_difference < 0] = 0

            image_range = new_image_90 - new_image_10
            image_range[image_range <= 0] = 1e-6

            equalized_image = utilities.rescale_0_1(image_difference/image_range)

            equalized_image[equalized_image < 0] = 0
            equalized_image[equalized_image > 1] = 1

            equalized_image[background_mask] = 0

            self.equalized_images[z] = 1.0 - equalized_image

    def calculate_soma_threshold_images(self, z_vals=[0]):
        for z in z_vals:
            equalized_image  = self.equalized_images[z]

            nuclei_image = equalized_image.copy()

            self.soma_masks[z] = local_maxima(h_maxima(nuclei_image, self.params['soma_threshold']/255.0, selem=square(3)), selem=square(3))
            # self.soma_masks[i] = remove_small_objects(self.soma_masks[i].astype(bool), 2, connectivity=2, in_place=True)

            nuclei_image_c = 1 - nuclei_image

            self.I_mods[z] = imimposemin(nuclei_image_c.astype(float), self.soma_masks[z])

            self.soma_threshold_images[z] = self.I_mods[z]/np.amax(self.I_mods[z])
            self.soma_threshold_images[z][self.soma_threshold_images[z] == -math.inf] = 0

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        mask = None
        if param in ("contrast, gamma"):
            self.calculate_adjusted_images(z_vals=[self.main_controller.params['z']])

            image = self.adjusted_images[self.main_controller.params['z']]
        elif param == "background_threshold":
            self.calculate_background_masks(z_vals=[self.main_controller.params['z']])

            image = self.adjusted_images[self.main_controller.params['z']].copy()
            # image[self.background_mask] = 0
            mask = self.background_masks[self.main_controller.params['z']]
        elif param == "window_size":
            self.calculate_equalized_images(z_vals=[self.main_controller.params['z']])

            image = self.equalized_images[self.main_controller.params['z']]
        elif param == "z":
            self.calculate_adjusted_images(z_vals=[self.main_controller.params['z']])
            # self.video_opened(self.video, self.video_path, plot=False)
            if self.param_widget.show_watershed_checkbox.isChecked():
                image = self.watershed_images[self.main_controller.params['z']]
            else:
                image = self.adjusted_images[self.main_controller.params['z']]

        self.preview_window.plot_image(image, mask=mask)

    def show_background_mask(self):
        if self.soma_threshold_images[self.main_controller.params['z']] is None:
            self.update_background_threshold(self.params['background_threshold'])

        image = self.adjusted_images[self.main_controller.params['z']].copy()

        self.preview_window.plot_image(image, mask=self.background_masks[self.main_controller.params['z']])

    def show_equalized_image(self):
        if self.equalized_images[self.main_controller.params['z']] is None:
            self.update_window_size(self.params['window_size'])
        self.preview_window.plot_image(self.equalized_images[self.main_controller.params['z']])

    def show_soma_threshold_image(self):
        self.preview_window.plot_image(self.soma_threshold_images[self.main_controller.params['z']])

    def show_adjusted_image(self):
        if self.adjusted_images[self.main_controller.params['z']] is not None:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_images[self.main_controller.params['z']])
        else:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def process_video(self):
        centers_list = []

        self.calculate_adjusted_images(z_vals=range(self.video.shape[1]))
        self.calculate_background_masks(z_vals=range(self.video.shape[1]))
        self.calculate_equalized_images(z_vals=range(self.video.shape[1]))
        self.calculate_soma_threshold_images(z_vals=range(self.video.shape[1]))

        for i in range(self.video.shape[1]):
            if len(self.masks[i]) > 0:
                masks = np.array(self.masks[i])
                mask = np.sum(masks, axis=0).astype(bool)

                out = np.zeros(self.adjusted_images[i].shape)
                out[mask] = self.adjusted_images[i][mask]
                adjusted_image = out.copy()

                out = np.zeros(self.soma_masks[i].shape)
                out[mask] = self.soma_masks[i][mask]
                soma_mask = out.copy()

                out = np.zeros(self.I_mods[i].shape)
                out[mask] = self.I_mods[i][mask]
                I_mod = out.copy()
            else:
                adjusted_image = self.adjusted_images[i]
                soma_mask      = self.soma_masks[i]
                I_mod          = self.soma_threshold_images[i]

            rgb_image = cv2.cvtColor((adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            start = time.time()

            self.labels[i], self.roi_areas[i], self.roi_circs[i] = utilities.apply_watershed(adjusted_image, soma_mask, I_mod)

            end = time.time()

            print("Time: {}s.".format(end - start))

            start = time.time()

            filtered_labels, filtered_out_rois = utilities.filter_rois(self.labels[i], self.filtering_params['min_area'], self.filtering_params['max_area'], self.filtering_params['min_circ'], self.filtering_params['max_circ'], self.roi_areas[i], self.roi_circs[i], self.roi_areas[i])

            self.watershed_images[i], self.roi_overlays[i], _ = utilities.draw_rois(rgb_image, self.labels[i], None, filtered_out_rois, None)

            end = time.time()

            print("Time: {}s.".format(end - start))

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.show_watershed_image(True)

        self.param_widget.filter_rois_button.setDisabled(False)

    def select_mask(self, mask_point):
        selected_mask, selected_mask_num = utilities.get_mask_containing_point(self.masks[self.main_controller.params['z']], mask_point)

        if selected_mask is not None:
            self.param_widget.erase_selected_mask_button.setEnabled(True)

            self.selected_mask     = selected_mask
            self.selected_mask_num = selected_mask_num
        else:
            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)

        self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def erase_selected_mask(self):
        if self.selected_mask is not None:
            del self.masks[self.main_controller.params['z']][self.selected_mask_num]
            del self.mask_points[self.main_controller.params['z']][self.selected_mask_num]

            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)
            
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def filter_rois(self):
        self.main_controller.show_roi_filtering_params(self.labels, self.roi_areas, self.roi_circs, self.roi_overlays)

    def save_params(self):
        json.dump(self.params, open(WATERSHED_PARAMS_FILENAME, "w"))

class ROIFilteringController():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(ROI_filtering_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_ROI_FILTERING_PARAMS
                params = json.load(open(ROI_filtering_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_ROI_FILTERING_PARAMS
        else:
            self.params = DEFAULT_ROI_FILTERING_PARAMS

        self.mean_images       = None
        self.normalized_images = None
        self.adjusted_images   = None
        self.labels            = None
        self.filtered_labels   = None
        self.watershed_images  = None
        self.roi_areas         = None
        self.roi_circs         = None
        self.selected_roi      = None
        self.filtered_out_rois = []
        self.erased_rois       = []
        self.removed_rois      = []
        self.last_erased_rois  = []
        self.locked_rois       = []

        self.figure = None
        self.axis   = None

    def video_opened(self, video, video_path, labels, roi_areas, roi_circs, roi_overlay, plot=False):
        self.video      = video
        self.video_path = video_path
        if labels is not None:
            self.labels = labels
        if roi_areas is not None:
            self.roi_areas = roi_areas
        if roi_circs is not None:
            self.roi_circs = roi_circs
        if roi_overlay is not None:
            self.roi_overlay = roi_overlay

        self.mean_images = [ ndi.median_filter(denoise_tv_chambolle(utilities.normalize(utilities.mean(video, i)).astype(np.float32), weight=0.01, multichannel=False), 3) for i in range(video.shape[1]) ]

        self.normalized_images = [ utilities.normalize(mean_image).astype(np.uint8) for mean_image in self.mean_images ]

        self.adjusted_images   = [ [] for i in range(video.shape[1]) ]
        self.filtered_labels   = [ [] for i in range(video.shape[1]) ]
        self.watershed_images  = [ [] for i in range(video.shape[1]) ]
        self.filtered_out_rois = [ [] for i in range(video.shape[1]) ]
        self.erased_rois       = [ [] for i in range(video.shape[1]) ]
        self.removed_rois      = [ [] for i in range(video.shape[1]) ]
        self.last_erased_rois  = [ [] for i in range(video.shape[1]) ]
        self.locked_rois       = [ [] for i in range(video.shape[1]) ]

        self.rois_erased = False

        self.calculate_adjusted_images(z_vals=[self.main_controller.params['z']])
        self.filter_rois(z_vals=[self.main_controller.params['z']])

        # self.preview_window.plot_image(self.watershed_images[self.main_controller.params['z']])

    def calculate_adjusted_images(self, z_vals=[0]):
        for z in z_vals:
            self.adjusted_images[z] = utilities.adjust_gamma(utilities.adjust_contrast(self.normalized_images[z], self.main_controller.params['contrast']), self.main_controller.params['gamma'])/255.0

    def calculate_watershed_images(self, z_vals=[0]):
        for z in z_vals:
            rgb_image = cv2.cvtColor((self.adjusted_images[z]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            self.watershed_images[z], roi_overlay, final_overlay = utilities.draw_rois(rgb_image, self.labels[z], self.selected_roi, self.removed_rois[z], self.locked_rois[z])

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.show_watershed_image(True)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.calculate_adjusted_images(z_vals=[self.main_controller.params['z']])
            self.preview_window.plot_image(self.watershed_images[self.main_controller.params['z']])
        elif param == "z":
            self.calculate_adjusted_images(z_vals=[self.main_controller.params['z']])
            self.filter_rois(z_vals=[self.main_controller.params['z']])
        elif param in ("min_area", "max_area", "min_circ", "max_circ"):
            self.filter_rois(z_vals=[self.main_controller.params['z']])

    def show_adjusted_image(self):
        if self.adjusted_images[self.main_controller.params['z']] is not None:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_images[self.main_controller.params['z']])
        else:
            self.preview_window.plot_image(self.adjusted_images[self.main_controller.params['z']])

    def filter_rois(self, z_vals=[0]):
        for z in z_vals:
            self.filtered_labels[z], self.filtered_out_rois[z] = utilities.filter_rois(self.labels[z], self.params['min_area'], self.params['max_area'], self.params['min_circ'], self.params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.locked_rois[z])
            self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]
        self.calculate_watershed_images(z_vals=z_vals)

    def erase_rois(self):
        z = self.main_controller.params['z']

        self.rois_erased = False

        if not self.preview_window.erasing_rois:
            self.preview_window.erasing_rois = True

            self.param_widget.erase_rois_button.setText("Finished")
        else:
            self.preview_window.erasing_rois = False

            self.param_widget.erase_rois_button.setText("Erase ROIs")

    def erase_roi_at_point(self, roi_point, radius=1):
        z = self.main_controller.params['z']

        if not self.rois_erased:
            self.last_erased_rois[z].append([])
            self.rois_erased = True

        roi_to_erase = utilities.get_roi_containing_point(self.filtered_labels[z], roi_point)

        if roi_to_erase is not None and roi_to_erase not in self.erased_rois[z] and roi_to_erase not in self.locked_rois[z]:
            self.erased_rois[z].append(roi_to_erase)
            self.last_erased_rois[z][-1].append(roi_to_erase)
            self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]
            self.calculate_watershed_images(z_vals=[z])

    def select_roi(self, roi_point):
        z = self.main_controller.params['z']

        selected_roi = utilities.get_roi_containing_point(self.filtered_labels[z], roi_point)

        if selected_roi is not None:
            self.param_widget.lock_roi_button.setEnabled(True)
            self.param_widget.enlarge_roi_button.setEnabled(True)
            self.param_widget.shrink_roi_button.setEnabled(True)
            if selected_roi in self.locked_rois[z]:
                self.param_widget.lock_roi_button.setText("Unlock ROI")
            else:
                self.param_widget.lock_roi_button.setText("Lock ROI")

            self.param_widget.erase_selected_roi_button.setEnabled(True)

            self.selected_roi = selected_roi

            activity = utilities.calc_activity_of_roi(self.filtered_labels[z], self.video, self.selected_roi, z=z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(selected_roi))
        else:
            self.selected_roi = -1

            self.param_widget.lock_roi_button.setEnabled(False)
            self.param_widget.enlarge_roi_button.setEnabled(False)
            self.param_widget.shrink_roi_button.setEnabled(False)
            self.param_widget.lock_roi_button.setText("Lock ROI")

        self.calculate_watershed_images(z_vals=[z])

    def undo_erase(self):
        z = self.main_controller.params['z']

        if len(self.last_erased_rois[z]) > 0:
            self.erased_rois[z] = self.erased_rois[z][:-len(self.last_erased_rois[-1])]
            del self.last_erased_rois[z][-1]
            self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

            self.calculate_watershed_images(z_vals=[z])

    def reset_erase(self):
        z = self.main_controller.params['z']

        if len(self.last_erased_rois[z]) > 0:
            self.erased_rois[z]      = []
            self.last_erased_rois[z] = []

            self.removed_rois[z] = self.filtered_out_rois[z]

            self.calculate_watershed_images(z_vals=[z])

    def erase_selected_roi(self):
        z = self.main_controller.params['z']

        self.erased_rois[z].append(self.selected_roi)
        self.last_erased_rois[z].append([self.selected_roi])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]
        self.selected_roi = None

        self.calculate_watershed_images(z_vals=[z])

        self.param_widget.erase_selected_roi_button.setEnabled(False)

    def lock_roi(self):
        z = self.main_controller.params['z']

        if self.selected_roi not in self.locked_rois[z]:
            self.locked_rois[z].append(self.selected_roi)
            self.param_widget.lock_roi_button.setText("Unlock ROI")
        else:
            index = self.locked_rois[z].index(self.selected_roi)
            del self.locked_rois[z][index]
            self.param_widget.lock_roi_button.setText("Lock ROI")

    def enlarge_roi(self):
        z = self.main_controller.params['z']

        if self.selected_roi >= 1:
            prev_labels = self.labels[z].copy()
            mask = self.labels[z] == self.selected_roi
            mask = binary_dilation(mask, disk(1))

            self.labels[z][mask] = self.selected_roi

            self.calculate_watershed_images(rois_to_update=[self.selected_roi], prev_labels=prev_labels, z_vals=[z])
            self.filter_rois(z_vals=[z])

    def shrink_roi(self):
        z = self.main_controller.params['z']

        if self.selected_roi >= 1:
            prev_labels = self.labels[z].copy()
            
            labels = self.labels[z].copy()
            mask = prev_labels == self.selected_roi
            labels[mask] = 0

            mask = self.labels[z] == self.selected_roi
            mask = erosion(mask, disk(1))
            labels[mask] = self.selected_roi

            self.labels[z] = labels.copy()

            self.calculate_watershed_images(rois_to_update=[self.selected_roi], prev_labels=prev_labels, z_vals=[z])
            self.filter_rois(z_vals=[z])

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def watershed(self):
        self.main_controller.show_watershed_params()

    def save_params(self):
        json.dump(self.params, open(ROI_filtering_PARAMS_FILENAME, "w"))