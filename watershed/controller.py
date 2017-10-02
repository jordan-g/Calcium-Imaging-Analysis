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

        self.video_path = None
        self.video_paths = []

        self.mode = "motion_correct"

        self.param_widget                                = self.param_window.main_param_widget
        self.motion_correction_controller.param_widget   = self.param_window.motion_correction_widget
        self.motion_correction_controller.preview_window = self.preview_window
        self.watershed_controller.param_widget           = self.param_window.watershed_widget
        self.watershed_controller.preview_window         = self.preview_window
        self.roi_filtering_controller.param_widget       = self.param_window.roi_filtering_widget
        self.roi_filtering_controller.preview_window     = self.preview_window
        self.watershed_controller.filtering_params       = self.roi_filtering_controller.params

        self.use_mc_video = False

        self.closing = False

    def select_and_open_video(self):
        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff *.npy)')[0]

        # open the first selected video (note: needs to be updates to process batches of videos)
        if video_paths is not None and len(video_paths) > 0:
            self.open_videos(video_paths)

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

    def open_videos(self, video_paths):
        self.video_paths += video_paths
        self.param_window.videos_opened(video_paths)

        if self.video_path is None:
            self.open_video(self.video_paths[0])

    def open_video(self, video_path):
        self.video_path = video_path

        # open video
        base_name = os.path.basename(self.video_path)
        if base_name.endswith('.npy'):
            self.video = np.load(self.video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = imread(self.video_path)

        if len(self.video.shape) == 3:
            # add z dimension
            self.video = self.video[:, np.newaxis, :, :]

        if self.params['z'] >= self.video.shape[1]:
            self.params['z'] = 0

        print("Loaded video with shape {}.".format(self.video.shape))

        self.param_widget.param_sliders["z"].setMaximum(self.video.shape[1]-1)

        self.video = np.nan_to_num(self.video).astype(np.float32)

        self.video = utilities.normalize(self.video).astype(np.uint8)

        self.normalized_video = self.video

        self.param_window.stacked_widget.setDisabled(False)
        self.param_window.statusBar().showMessage("")

        # update the motion correction controller
        self.motion_correction_controller.video_opened(self.video, self.video_path)

    def remove_videos_at_indices(self, indices):
        indices = sorted(indices)
        for i in range(len(indices)-1, -1, -1):
            index = indices[i]
            del self.video_paths[index]

        self.param_window.remove_selected_items()

        if len(self.video_paths) == 0:
            self.video_path = None
            self.use_mc_video = False

            self.show_motion_correction_params()
            self.param_window.toggle_initial_state(True)
            self.preview_window.plot_image(None)
        elif 0 in indices:
            self.open_video(self.video_paths[0])

    def process_all_videos(self):
        ...

    def show_watershed_params(self, video=None, video_path=None):
        if video is None:
            video = self.video

        if video_path is None:
            video_path = self.video_path

        self.param_window.stacked_widget.setCurrentIndex(1)
        self.mode = "watershed"
        self.preview_window.controller = self.watershed_controller
        self.watershed_controller.video_opened(video, video_path)
        self.param_window.statusBar().showMessage("")

        self.preview_window.setWindowTitle("Preview")

    def show_motion_correction_params(self):
        self.param_window.stacked_widget.setCurrentIndex(0)
        self.mode = "motion_correct"
        self.preview_window.controller = self.motion_correction_controller
        self.motion_correction_controller.video_opened(self.video, self.video_path)
        self.param_window.statusBar().showMessage("")

    def show_roi_filtering_params(self, labels, roi_areas, roi_circs, mean_images, normalized_images):
        self.param_window.stacked_widget.setCurrentIndex(2)
        self.mode = "filter"
        self.preview_window.controller = self.roi_filtering_controller
        self.roi_filtering_controller.video_opened(self.video, self.video_path, labels, roi_areas, roi_circs, mean_images, normalized_images)
        self.param_window.statusBar().showMessage("")

    def rois_created(self):
        self.param_window.rois_created()

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

        self.video    = None
        self.mc_video = None
        
        self.adjusted_video    = None
        self.adjusted_mc_video = None
        
        self.adjusted_frame    = None
        self.adjusted_mc_frame = None
        
        self.video_path    = None
        self.mc_video_path = None

        self.use_mc_video = False

        self.z = 0

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
        self.mc_video, self.mc_video_path = utilities.motion_correct(self.video, self.video_path, int(self.params["max_shift"]), int(self.params["patch_stride"]), int(self.params["patch_overlap"]))

        self.mc_video = utilities.normalize(self.mc_video).astype(np.uint8)

        self.use_mc_video = True

        self.adjusted_mc_video = self.calculate_adjusted_video(self.mc_video, self.z)

        self.param_widget.use_mc_video_checkbox.setEnabled(True)
        self.param_widget.use_mc_video_checkbox.setChecked(True)

        self.set_use_mc_video(True)

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
        self.preview_window.timer.stop()

        if self.use_mc_video:
            self.main_controller.show_watershed_params(video=self.mc_video, video_path=self.mc_video_path)
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

    def video_opened(self, video, video_path):
        if video_path != self.video_path:
            self.video       = video
            self.video_path  = video_path

            self.z = self.main_controller.params['z']

            self.mean_images = [ ndi.median_filter(denoise_tv_chambolle(utilities.mean(self.video, z).astype(np.float32), weight=0.01, multichannel=False), 3) for z in range(video.shape[1]) ]

            self.normalized_images = [ utilities.normalize(mean_image).astype(np.uint8) for mean_image in self.mean_images ]

            self.masks             = [ [] for i in range(video.shape[1]) ]
            self.mask_points       = [ [] for i in range(video.shape[1]) ]
            self.labels            = [ [] for i in range(video.shape[1]) ]
            self.roi_areas         = [ [] for i in range(video.shape[1]) ]
            self.roi_circs         = [ [] for i in range(video.shape[1]) ]
            self.filtered_out_rois = [ [] for i in range(video.shape[1]) ]
            
            self.adjusted_image       = self.calculate_adjusted_image(self.normalized_images[self.z])
            self.background_mask      = self.calculate_background_mask(self.adjusted_image)
            self.equalized_image      = self.calculate_equalized_image(self.adjusted_image, self.background_mask)
            self.soma_mask, self.I_mod, self.soma_threshold_image = self.calculate_soma_threshold_image(self.equalized_image)

        self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def calculate_adjusted_image(self, normalized_image):
        return utilities.adjust_gamma(utilities.adjust_contrast(normalized_image, self.main_controller.params['contrast']), self.main_controller.params['gamma'])/255.0

    def calculate_background_mask(self, adjusted_image):
        return adjusted_image < self.params['background_threshold']/255.0
    
    def calculate_equalized_image(self, adjusted_image, background_mask):
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

        return 1.0 - equalized_image
    
    def calculate_soma_threshold_image(self, equalized_image):
        nuclei_image = equalized_image.copy()

        soma_mask = local_maxima(h_maxima(nuclei_image, self.params['soma_threshold']/255.0, selem=square(3)), selem=square(3))
        # self.soma_masks[i] = remove_small_objects(self.soma_masks[i].astype(bool), 2, connectivity=2, in_place=True)

        nuclei_image_c = 1 - nuclei_image

        I_mod = imimposemin(nuclei_image_c.astype(float), soma_mask)

        soma_threshold_image = I_mod/np.amax(I_mod)
        soma_threshold_image[soma_threshold_image == -math.inf] = 0

        return soma_mask, I_mod, soma_threshold_image
    
    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.adjusted_image)

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
                self.mask_points[self.z].append(mask_points)
                mask_points = np.array(mask_points)

                mask = np.zeros(self.adjusted_image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)
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
            self.adjusted_image = self.calculate_adjusted_image(self.normalized_images[self.z])

            if self.labels is not None:
                self.calculate_watershed_image(self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())
        elif param == "background_threshold":
            self.background_mask = self.calculate_background_mask(self.adjusted_image)

            self.param_widget.show_watershed_checkbox.setChecked(False)

            self.preview_window.plot_image(self.adjusted_image, mask=self.background_mask)
        elif param == "window_size":
            self.equalized_image = self.calculate_equalized_image(self.adjusted_image, self.background_mask)

            self.param_widget.show_watershed_checkbox.setChecked(False)

            self.preview_window.plot_image(self.equalized_image, mask=None)
        elif param == "z":
            self.z = value

            self.adjusted_image = self.calculate_adjusted_image(self.normalized_images[self.z])

            if self.labels is not None:
                self.calculate_watershed_image(self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def calculate_watershed_image(self, z, update_overlay=True):
        if update_overlay:
            roi_overlay = None
        else:
            roi_overlay = self.roi_overlay

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[z], None, self.filtered_out_rois[z], None, roi_overlay=roi_overlay)

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def process_video(self):
        centers_list = []

        for z in range(self.video.shape[1]):
            adjusted_image  = self.calculate_adjusted_image(self.normalized_images[z])
            background_mask = self.calculate_background_mask(adjusted_image)
            equalized_image = self.calculate_equalized_image(adjusted_image, background_mask)
            soma_mask, I_mod, soma_threshold_image = self.calculate_soma_threshold_image(equalized_image)

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

            start = time.time()

            self.labels[z], self.roi_areas[z], self.roi_circs[z] = utilities.apply_watershed(adjusted_image, soma_mask, I_mod)

            if len(self.masks[z]) > 0:
                masks = np.array(self.masks[z])
                mask = np.sum(masks, axis=0).astype(bool)

                out = np.zeros(self.labels[z].shape).astype(int)
                out[mask] = self.labels[z][mask]
                self.labels[z] = out.copy()

            end = time.time()

            print("Time: {}s.".format(end - start))

            start = time.time()

            filtered_labels, self.filtered_out_rois[z] = utilities.filter_rois(self.normalized_images[z], self.labels[z], self.filtering_params['min_area'], self.filtering_params['max_area'], self.filtering_params['min_circ'], self.filtering_params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.roi_areas[z])

            end = time.time()

            print("Time: {}s.".format(end - start))

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.param_widget.filter_rois_button.setDisabled(False)

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[self.z], None, self.filtered_out_rois[self.z], None)

        self.main_controller.rois_created()

        self.show_watershed_image(True)

    def select_mask(self, mask_point):
        selected_mask, selected_mask_num = utilities.get_mask_containing_point(self.masks[self.z], mask_point)

        if selected_mask is not None:
            self.param_widget.erase_selected_mask_button.setEnabled(True)

            self.selected_mask     = selected_mask
            self.selected_mask_num = selected_mask_num
        else:
            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)

        self.preview_window.plot_image(self.adjusted_image)

    def erase_selected_mask(self):
        if self.selected_mask is not None:
            del self.masks[self.z][self.selected_mask_num]
            del self.mask_points[self.z][self.selected_mask_num]

            self.selected_mask     = None
            self.selected_mask_num = -1

            self.param_widget.erase_selected_mask_button.setEnabled(False)
            
            self.preview_window.plot_image(self.adjusted_image)

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def filter_rois(self):
        self.main_controller.show_roi_filtering_params(self.labels, self.roi_areas, self.roi_circs, self.mean_images, self.normalized_images)

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

        self.adjusted_image  = None
        self.watershed_image = None
        self.roi_overlay     = None

        self.labels            = None
        self.filtered_labels   = None
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

    def video_opened(self, video, video_path, labels, roi_areas, roi_circs, mean_images, normalized_images):
        if labels is not self.labels:
            self.video      = video
            self.video_path = video_path

            self.z = self.main_controller.params['z']
            
            self.mean_images       = mean_images
            self.normalized_images = normalized_images

            self.labels          = labels
            self.filtered_labels = [ [] for i in range(video.shape[1]) ]
            self.roi_areas       = roi_areas
            self.roi_circs       = roi_circs

            self.filtered_out_rois = [ [] for i in range(video.shape[1]) ]
            self.erased_rois       = [ [] for i in range(video.shape[1]) ]
            self.removed_rois      = [ [] for i in range(video.shape[1]) ]
            self.last_erased_rois  = [ [] for i in range(video.shape[1]) ]
            self.locked_rois       = [ [] for i in range(video.shape[1]) ]

            self.rois_erased = False

            self.adjusted_image = self.calculate_adjusted_image(self.normalized_images[self.z])

            self.filter_rois(z=self.z)

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.param_widget.show_watershed_checkbox.setDisabled(False)
            self.param_widget.show_watershed_checkbox.setChecked(True)

            self.show_watershed_image(True)

    def calculate_adjusted_image(self, normalized_image):
        return utilities.adjust_gamma(utilities.adjust_contrast(normalized_image, self.main_controller.params['contrast']), self.main_controller.params['gamma'])/255.0

    def calculate_watershed_image(self, z, update_overlay=True):
        if update_overlay:
            roi_overlay = None
        else:
            roi_overlay = self.roi_overlay

        rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        self.watershed_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.labels[z], self.selected_roi, self.removed_rois[z], self.locked_rois[z], roi_overlay=roi_overlay)

    def update_param(self, param, value):
        if param in self.params.keys():
            self.params[param] = value

        if param in ("contrast, gamma"):
            self.adjusted_image = self.calculate_adjusted_image(self.normalized_images[self.z])

            self.calculate_watershed_image(self.z, update_overlay=False)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())
        elif param == "z":
            self.adjusted_image = self.calculate_adjusted_image(self.normalized_images[self.z])

            self.filter_rois(z=self.z)

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())
        elif param in ("min_area", "max_area", "min_circ", "max_circ"):
            self.filter_rois(z=self.z)

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def filter_rois(self, z):
        self.filtered_labels[z], self.filtered_out_rois[z] = utilities.filter_rois(self.mean_images[z], self.labels[z], self.params['min_area'], self.params['max_area'], self.params['min_circ'], self.params['max_circ'], self.roi_areas[z], self.roi_circs[z], self.locked_rois[z])
        self.removed_rois[z] = self.filtered_out_rois[z] + self.erased_rois[z]

    def erase_rois(self):
        self.rois_erased = False

        if not self.preview_window.erasing_rois:
            self.preview_window.erasing_rois = True

            self.param_widget.erase_rois_button.setText("Finished")
        else:
            self.preview_window.erasing_rois = False

            self.param_widget.erase_rois_button.setText("Erase ROIs")

    def erase_roi_at_point(self, roi_point, radius=1):
        if not self.rois_erased:
            self.last_erased_rois[self.z].append([])
            self.rois_erased = True

        roi_to_erase = utilities.get_roi_containing_point(self.filtered_labels[self.z], roi_point)

        if roi_to_erase is not None and roi_to_erase not in self.erased_rois[self.z] and roi_to_erase not in self.locked_rois[self.z]:
            self.erased_rois[self.z].append(roi_to_erase)
            self.last_erased_rois[self.z][-1].append(roi_to_erase)
            self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]
            
            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def select_roi(self, roi_point):
        selected_roi = utilities.get_roi_containing_point(self.filtered_labels[self.z], roi_point)

        if selected_roi is not None:
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

            activity = utilities.calc_activity_of_roi(self.filtered_labels[self.z], self.video, self.selected_roi, z=self.z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
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

    def undo_erase(self):
        if len(self.last_erased_rois[self.z]) > 0:
            self.erased_rois[self.z] = self.erased_rois[self.z][:-len(self.last_erased_rois[-1])]
            del self.last_erased_rois[self.z][-1]
            self.removed_rois[self.z] = self.filtered_out_rois[self.z] + self.erased_rois[self.z]

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

    def reset_erase(self):
        if len(self.last_erased_rois[self.z]) > 0:
            self.erased_rois[self.z]      = []
            self.last_erased_rois[self.z] = []

            self.removed_rois[self.z] = self.filtered_out_rois[self.z]

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

    def enlarge_roi(self):
        if self.selected_roi >= 1:
            prev_labels = self.labels[self.z].copy()
            mask = self.labels[self.z] == self.selected_roi
            mask = binary_dilation(mask, disk(1))

            self.labels[self.z][mask] = self.selected_roi

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            activity = utilities.calc_activity_of_roi(self.filtered_labels[self.z], self.video, self.selected_roi, z=self.z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

    def shrink_roi(self):
        if self.selected_roi >= 1:
            prev_labels = self.labels[self.z].copy()
            
            labels = self.labels[self.z].copy()
            mask = prev_labels == self.selected_roi
            labels[mask] = 0

            mask = self.labels[self.z] == self.selected_roi
            mask = erosion(mask, disk(1))
            labels[mask] = self.selected_roi

            self.labels[self.z] = labels.copy()

            self.calculate_watershed_image(z=self.z, update_overlay=True)

            self.show_watershed_image(show=self.param_widget.show_watershed_checkbox.isChecked())

            activity = utilities.calc_activity_of_roi(self.filtered_labels[self.z], self.video, self.selected_roi, z=self.z)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(self.selected_roi))

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def watershed(self):
        self.main_controller.show_watershed_params()

    def save_params(self):
        json.dump(self.params, open(ROI_filtering_PARAMS_FILENAME, "w"))