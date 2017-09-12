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

DEFAULT_WATERSHED_PARAMS = {'gamma': 1.0,
                            'contrast': 1.0,
                            'window_size': 5,
                            'neuropil_threshold': 50,
                            'soma_threshold': 10,
                            'background_threshold': 10,
                            'compactness': 10}

DEFAULT_MOTION_CORRECTION_PARAMS = {'gamma': 1.0,
                                    'contrast': 1.0,
                                    'fps': 60,
                                    'max_shift': 6,
                                    'patch_stride': 24,
                                    'patch_overlap': 6}

DEFAULT_ROI_PRUNING_PARAMS = {'gamma': 1.0,
                              'contrast': 1.0,
                              'min_area': 10,
                              'max_area': 100}

WATERSHED_PARAMS_FILENAME         = "watershed_params.txt"
MOTION_CORRECTION_PARAMS_FILENAME = "motion_correction_params.txt"
ROI_PRUNING_PARAMS_FILENAME       = "roi_pruning_params.txt"

class Controller():
    def __init__(self):
        # create controllers
        self.motion_correction_controller = MotionCorrectionController(self)
        self.watershed_controller         = WatershedController(self)
        self.roi_pruning_controller       = ROIPruningController(self)

        # create windows
        self.param_window   = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

        self.preview_window.mode = "motion_correct"

        self.motion_correction_controller.param_widget   = self.param_window.motion_correction_widget
        self.motion_correction_controller.preview_window = self.preview_window
        self.watershed_controller.param_widget           = self.param_window.watershed_widget
        self.watershed_controller.preview_window         = self.preview_window
        self.roi_pruning_controller.param_widget         = self.param_window.roi_pruning_widget
        self.roi_pruning_controller.preview_window       = self.preview_window
        self.watershed_controller.pruning_params         = self.roi_pruning_controller.params

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

    def open_video(self, video_path):
        self.video_path = video_path

        # open video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.npy'):
            self.video = np.load(video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = imread(video_path)
        self.video = np.nan_to_num(self.video)

        self.param_window.stacked_widget.setDisabled(False)
        self.param_window.statusBar().showMessage("")

        # update the motion correction controller
        self.motion_correction_controller.video_opened(self.video, self.video_path, plot=True)

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
        self.preview_window.mode = "watershed"

    def show_motion_correction_params(self):
        self.param_window.stacked_widget.setCurrentIndex(0)
        self.motion_correction_controller.video_opened(self.video, self.video_path, plot=True)
        self.preview_window.controller = self.motion_correction_controller
        self.param_window.statusBar().showMessage("")

        self.preview_window.mode = "motion_correct"

    def show_roi_pruning_params(self, labels, roi_areas, roi_overlay):
        self.param_window.stacked_widget.setCurrentIndex(2)
        self.roi_pruning_controller.video_opened(self.video, self.video_path, labels, roi_areas, roi_overlay, plot=True)
        self.preview_window.controller = self.roi_pruning_controller
        self.param_window.statusBar().showMessage("")

        self.preview_window.mode = "prune"

    def close_all(self):
        self.closing = True
        self.param_window.close()
        self.preview_window.close()

        self.motion_correction_controller.save_params()
        self.watershed_controller.save_params()
        self.roi_pruning_controller.save_params()

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

        self.video                           = None
        self.adjusted_video                  = None
        self.adjusted_frame                  = None
        self.motion_corrected_video          = None
        self.current_video                   = None
        self.current_adjusted_video          = None
        self.adjusted_motion_corrected_video = None
        self.video_path                      = None
        self.motion_corrected_video_path     = None

        self.showing_motion_corrected_video  = False

    def video_opened(self, video, video_path, plot=False):
        self.video      = video
        self.video_path = video_path

        self.calculate_adjusted_video()

        if plot:
            self.play_adjusted_video()

    def preview_contrast(self, contrast):
        self.preview_window.timer.stop()
        
        self.params['contrast'] = contrast
        
        self.calculate_adjusted_frame()
          
        self.preview_window.show_frame(self.adjusted_frame)

    def preview_gamma(self, gamma):
        self.preview_window.timer.stop()
        
        self.params['gamma'] = gamma

        self.calculate_adjusted_frame()

        self.preview_window.show_frame(self.adjusted_frame)

    def update_param(self, param, value):
        self.params[param] = value

        if param in ("contrast, gamma"):
            self.preview_window.timer.stop()

            self.calculate_adjusted_video()

            self.preview_window.play_movie(self.adjusted_video, fps=self.params['fps'])
        elif param == "fps":
            self.preview_window.set_fps(self.params['fps'])

    def calculate_adjusted_video(self):
        if self.showing_motion_corrected_video:
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.motion_corrected_video, self.params['contrast']), self.params['gamma'])
        else:
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.video, self.params['contrast']), self.params['gamma'])

    def calculate_adjusted_frame(self):
        if self.showing_motion_corrected_video:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.motion_corrected_video[self.preview_window.frame_num], self.params['contrast']), self.params['gamma'])
        else:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.video[self.preview_window.frame_num], self.params['contrast']), self.params['gamma'])

    def process_video(self):
        self.motion_corrected_video, self.motion_corrected_video_path = utilities.motion_correct(self.video_path, int(self.params["max_shift"]), int(self.params["patch_stride"]), int(self.params["patch_overlap"]))

        self.showing_motion_corrected_video = True
        self.calculate_adjusted_video()

        self.param_widget.play_motion_corrected_video_checkbox.setEnabled(True)
        self.param_widget.play_motion_corrected_video_checkbox.setChecked(True)
        self.play_motion_corrected_video(True)

        self.param_widget.accept_button.setEnabled(True)

    def play_video(self):
        if self.video is not None:
            self.preview_window.play_movie(self.video, fps=self.params['fps'])

    def play_adjusted_video(self):
        if self.adjusted_video is not None:
            self.preview_window.play_movie(self.adjusted_video, fps=self.params['fps'])

    def play_motion_corrected_video(self, show):
        self.showing_motion_corrected_video = show

        self.calculate_adjusted_video()
        self.preview_window.play_movie(self.adjusted_video, fps=self.params['fps'])

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

        self.image                = None
        self.adjusted_image       = None
        self.normalized_image     = None
        self.soma_threshold_image = None
        self.masks                = []
        self.mask_points          = []
        self.neuropil_mask        = None
        self.background_mask      = None

    def video_opened(self, video, video_path, plot=False):
        self.image = utilities.normalize(utilities.mean(video)).astype(np.float32)
        
        self.image = denoise_tv_chambolle(self.image, weight=0.01, multichannel=False)
        self.image = ndi.median_filter(self.image, 3) #Added

        self.calculate_adjusted_image()
        self.calculate_normalized_image()
        self.calculate_background_mask()
        self.calculate_neuropil_mask()
        self.calculate_soma_threshold_image()

        if plot:
            self.preview_window.plot_image(self.adjusted_image)

    def calculate_background_mask(self):
        mask = self.adjusted_image >= self.params['background_threshold']/255.0

        self.background_mask = mask == 0

    def calculate_neuropil_mask(self):
        # print(np.amax(self.normalized_image))
        mask_1 = self.normalized_image >= self.params['neuropil_threshold']/255.0

        connected_components, n = ndi.label(mask_1, disk(1))

        self.mask_2 = np.zeros(self.normalized_image.shape).astype(bool)
        for i in range(n):
            self.mask_2[connected_components == i+1] = True
        self.mask_2 = binary_closing(self.mask_2, disk(1))
        self.mask_2 = remove_small_objects(self.mask_2, 7, connectivity=2, in_place=True)
        self.mask_2 = binary_dilation(self.mask_2, disk(1))
        self.mask_2_perimeter = bwperim(self.mask_2, n=4)

        self.neuropil_mask = self.mask_2 == 0

    def calculate_adjusted_image(self):
        self.adjusted_image = utilities.adjust_gamma(utilities.adjust_contrast(self.image, self.params['contrast']), self.params['gamma'])/255.0

        if len(self.masks) > 0:
            out = self.adjusted_image.copy()
            masks = np.array(self.masks)
            mask = np.sum(masks, axis=0).astype(bool)
            out[mask == False] *= 0.5

            self.adjusted_image = out.copy()

    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.adjusted_image)

            self.preview_window.drawing_mask = True

            self.param_widget.draw_mask_button.setText("\u2713 Done")
            self.param_widget.draw_mask_button.previous_message = "Draw a mask on the image preview."
        else:
            if len(self.preview_window.mask_points) > 0:
                mask_points = self.preview_window.mask_points
                mask_points += [mask_points[0]]
                self.mask_points.append(mask_points)
                mask_points = np.array(mask_points)

                mask = np.zeros(self.image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)
                self.masks.append(mask)

                self.calculate_adjusted_image()
                self.calculate_normalized_image()
                self.calculate_soma_threshold_image()

            self.preview_window.end_drawing_mask()
            self.preview_window.plot_image(self.adjusted_image)

            self.param_widget.draw_mask_button.setText("Draw Mask")
            self.param_widget.draw_mask_button.previous_message = ""

    def calculate_normalized_image(self):
        # image = self.adjusted_image.copy()
        # image[self.background_mask] = 0
        # print(np.amax(self.adjusted_image))
        new_image_10 = utilities.order_statistic(self.adjusted_image, 0.1, int(self.params['window_size']))
        new_image_90 = utilities.order_statistic(self.adjusted_image, 0.9, int(self.params['window_size']))

        image_difference = self.adjusted_image - new_image_10
        image_difference[image_difference < 0] = 0

        image_range = new_image_90 - new_image_10
        image_range[image_range <= 0] = 1e-6

        normalized_image = utilities.rescale_0_1(image_difference/image_range)

        normalized_image[normalized_image < 0] = 0
        normalized_image[normalized_image > 1] = 1

        normalized_image[self.background_mask] = 0

        # normalized_image = 1.0 - normalized_image

        self.normalized_image = 1.0 - normalized_image

        # print(self.normalized_image)

        # self.normalized_image[self.neuropil_mask] = 0

        if len(self.masks) > 0:
            out = self.normalized_image.copy()
            masks = np.array(self.masks)
            mask = np.sum(masks, axis=0).astype(bool)
            out[mask == False] *= 0.5

            self.normalized_image = out.copy()

    def calculate_soma_threshold_image(self):
        nuclei_image = self.normalized_image.copy()
        nuclei_image[self.neuropil_mask] = 0
        self.soma_mask = local_maxima(h_maxima(nuclei_image, self.params['soma_threshold']/255.0, selem=square(3)), selem=square(3))
        # self.soma_mask = remove_small_objects(self.soma_mask.astype(bool), 2, connectivity=2, in_place=True)

        nuclei_image_c = 1 - nuclei_image

        # filter_blurred_f = ndi.gaussian_filter(nuclei_image_c, 1)
        # alpha = 30
        # nuclei_image_c = nuclei_image_c + alpha * (nuclei_image_c - filter_blurred_f)

        a = np.logical_and((self.mask_2_perimeter == 0), (bwperim(erosion(self.mask_2, disk(1))) == 0))

        L = label((a == 0), connectivity=2)
        props = regionprops(L)

        idxs = []
        for i in range(len(props)):
            if props[i]['area'] == props[i]['filled_area']:
                idxs.append(i+1)
                
        mask_3 = np.in1d(L, idxs).reshape(L.shape)

        self.I_mod = imimposemin(nuclei_image_c.astype(float), np.logical_or(self.neuropil_mask, np.logical_and(self.soma_mask, np.logical_or((a + mask_3) == 0, self.mask_2_perimeter) == 0)))

        self.soma_threshold_image = self.I_mod/np.amax(self.I_mod)
        self.soma_threshold_image[self.soma_threshold_image == -math.inf] = 0

        # print(np.amax(self.soma_threshold_image), np.amin(self.soma_threshold_image))

        if len(self.masks) > 0:
            out = self.soma_threshold_image.copy()
            masks = np.array(self.masks)
            mask = np.sum(masks, axis=0).astype(bool)
            out[mask == False] *= 0.5

            self.soma_threshold_image = out.copy()

    def update_param(self, param, value):
        self.params[param] = value

        if param in ("contrast, gamma"):
            self.calculate_adjusted_image()

            image = self.adjusted_image
        elif param == "neuropil_threshold":
            self.calculate_neuropil_mask()
            
            image = self.normalized_image.copy()
            image[self.neuropil_mask] = 0
        elif param == "background_threshold":
            self.calculate_background_mask()

            image = self.adjusted_image.copy()
            image[self.background_mask] = 0
        elif param == "window_size":
            self.calculate_normalized_image()

            image = self.normalized_image
        elif param == "soma_threshold":
            self.calculate_soma_threshold_image()

            image = self.soma_threshold_image
        elif param == "compactness":
            image = self.adjusted_image

        self.preview_window.plot_image(image)

    def show_background_mask(self):
        if self.soma_threshold_image is None:
            self.update_background_threshold(self.params['background_threshold'])

        image = self.adjusted_image.copy()
        image[self.background_mask] = 0

        self.preview_window.plot_image(image)

    def show_neuropil_mask(self):
        if self.soma_threshold_image is None:
            self.update_neuropil_threshold(self.params['neuropil_threshold'])

        image = self.normalized_image.copy()
        image[self.neuropil_mask] = 0

        self.preview_window.plot_image(image)

    def show_normalized_image(self):
        if self.normalized_image is None:
            self.update_window_size(self.params['window_size'])
        self.preview_window.plot_image(self.normalized_image)

    def show_soma_threshold_image(self):
        if self.soma_threshold_image is None:
            self.update_soma_threshold(self.params['soma_threshold'])
        self.preview_window.plot_image(self.soma_threshold_image)

    def show_image(self):
        if self.image is not None:
            self.preview_window.plot_image(self.image)

    def show_adjusted_image(self):
        if self.adjusted_image is not None:
            self.preview_window.plot_image(self.adjusted_image)

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def process_video(self):
        centers_list = []

        self.adjusted_image = self.image.copy()
        self.calculate_adjusted_image()
        self.calculate_normalized_image()
        self.calculate_neuropil_mask()
        self.calculate_soma_threshold_image()

        if len(self.masks) > 0:
            masks = np.array(self.masks)
            mask = np.sum(masks, axis=0).astype(bool)

            out = np.zeros(self.image.shape)
            out[mask] = self.adjusted_image[mask]
            adjusted_image = out.copy()

            out = np.zeros(self.image.shape)
            out[mask] = self.soma_mask[mask]
            soma_mask = out.copy()

            out = np.zeros(self.image.shape)
            out[mask] = self.I_mod[mask]
            I_mod = out.copy()
        else:
            adjusted_image = self.adjusted_image
            soma_mask      = self.soma_mask
            I_mod          = self.I_mod

        self.rgb_image = cv2.cvtColor((adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        self.labels, self.roi_areas = utilities.apply_watershed(adjusted_image, soma_mask, I_mod)

        self.pruned_labels, _ = utilities.prune_rois(self.labels, self.pruning_params['min_area'], self.pruning_params['max_area'], self.roi_areas)
        
        if len(self.masks) > 0:
            out = self.labels.copy().astype(np.float64)
            masks = np.array(self.masks)
            mask = np.sum(masks, axis=0).astype(bool)
            out[mask == False] *= 0.5

            self.labels = out.copy()

        self.watershed_image, self.roi_overlay, _ = utilities.draw_rois(self.rgb_image, self.pruned_labels, None, None, None, None, create_initial_overlay=True)
        

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.show_watershed_image(True)

        self.param_widget.prune_rois_button.setDisabled(False)

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def prune_rois(self):
        self.main_controller.show_roi_pruning_params(self.labels, self.roi_areas, self.roi_overlay)

    def save_params(self):
        json.dump(self.params, open(WATERSHED_PARAMS_FILENAME, "w"))

class ROIPruningController():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
        if os.path.exists(ROI_PRUNING_PARAMS_FILENAME):
            try:
                self.params = DEFAULT_ROI_PRUNING_PARAMS
                params = json.load(open(ROI_PRUNING_PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_ROI_PRUNING_PARAMS
        else:
            self.params = DEFAULT_ROI_PRUNING_PARAMS

        self.image             = None
        self.adjusted_image    = None
        self.rgb_image         = None
        self.labels            = None
        self.pruned_labels     = None
        self.watershed_image   = None
        self.roi_overlay       = None
        self.roi_areas         = None
        self.selected_roi      = None
        self.final_overlay     = None
        self.pruned_rois       = []
        self.erased_rois       = []
        self.removed_rois      = []
        self.last_erased_rois  = []

        self.figure = None
        self.axis   = None

    def video_opened(self, video, video_path, labels, roi_areas, roi_overlay, plot=False):
        self.video       = video
        self.labels      = labels
        self.roi_areas   = roi_areas
        self.roi_overlay = roi_overlay
        self.image       = utilities.normalize(utilities.mean(video)).astype(np.float32)
        self.calculate_adjusted_image()
        self.prune_rois()

        if plot:
            self.preview_window.plot_image(self.watershed_image)

    def calculate_adjusted_image(self):
        self.adjusted_image = utilities.adjust_gamma(utilities.adjust_contrast(self.image, self.params['contrast']), self.params['gamma'])/255.0

        sigma_est = estimate_sigma(self.adjusted_image, multichannel=False)

        self.adjusted_image = denoise_tv_chambolle(self.adjusted_image, weight=0.01, multichannel=False)
        self.adjusted_image = ndi.median_filter(self.adjusted_image, 3) #Added

        self.rgb_image = cv2.cvtColor((self.adjusted_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def calculate_watershed_image(self, create_initial_overlay=False, update_overlay=False):
        self.watershed_image, self.roi_overlay, self.final_overlay = utilities.draw_rois(self.rgb_image, self.labels, self.roi_overlay, self.final_overlay, self.selected_roi, self.removed_rois, create_initial_overlay=create_initial_overlay, update_overlay=update_overlay)

        self.param_widget.show_watershed_checkbox.setDisabled(False)
        self.param_widget.show_watershed_checkbox.setChecked(True)
        self.show_watershed_image(True)

    def update_param(self, param, value):
        self.params[param] = value

        if param in ("contrast, gamma"):
            self.calculate_adjusted_image()
        
        self.calculate_watershed_image(update_overlay=False)

        self.preview_window.plot_image(self.watershed_image)

    def show_image(self):
        if self.image is not None:
            self.preview_window.plot_image(self.image)

    def show_adjusted_image(self):
        if self.adjusted_image is not None:
            self.preview_window.plot_image(self.adjusted_image)

    def show_watershed_image(self, show):
        if show:
            self.preview_window.plot_image(self.watershed_image)
        else:
            self.preview_window.plot_image(self.adjusted_image)

    def prune_rois(self):
        self.pruned_labels, self.pruned_rois = utilities.prune_rois(self.labels, self.params['min_area'], self.params['max_area'], self.roi_areas)
        self.removed_rois = self.pruned_rois + self.erased_rois
        self.calculate_watershed_image(update_overlay=True)

    def erase_rois(self):
        if not self.preview_window.erasing_rois:
            self.preview_window.erasing_rois = True

            self.last_erased_rois = []

            self.param_widget.erase_rois_button.setText("Finished")
        else:
            self.preview_window.erasing_rois = False

            self.param_widget.erase_rois_button.setText("Erase ROIs")

    def erase_roi_at_point(self, roi_point):
        roi_to_erase = utilities.get_roi_containing_point(self.pruned_labels, roi_point)

        if roi_to_erase is not None and roi_to_erase not in self.erased_rois:
            self.erased_rois.append(roi_to_erase)
            self.last_erased_rois.append(roi_to_erase)
            self.removed_rois = self.pruned_rois + self.erased_rois
            self.calculate_watershed_image(update_overlay=True)

    def select_roi(self, roi_point):
        selected_roi = utilities.get_roi_containing_point(self.pruned_labels, roi_point)

        if selected_roi is not None:
            self.selected_roi = selected_roi
            self.calculate_watershed_image(update_overlay=True)

            activity = utilities.calc_activity_of_roi(self.pruned_labels, self.video, self.selected_roi)

            if self.figure is None:
                plt.close('all')
                self.figure, self.axis = plt.subplots(figsize=(5, 3))
                self.figure.canvas.set_window_title('ROI Activity')
                self.figure.tight_layout()

            self.axis.clear()
            self.axis.plot(activity, c="#FF6666")
            self.figure.canvas.set_window_title('ROI {} Activity'.format(selected_roi))

    def undo_erase(self):
        if len(self.last_erased_rois) > 0:
            self.erased_rois = self.erased_rois[:-len(self.last_erased_rois)]
            self.last_erased_rois = []
            self.removed_rois = self.pruned_rois + self.erased_rois

            self.calculate_watershed_image(update_overlay=True)

    def reset_erase(self):
        if len(self.last_erased_rois) > 0:
            self.erased_rois = []
            self.last_erased_rois = []

            self.removed_rois = self.pruned_rois

            self.calculate_watershed_image(update_overlay=True)

    def keep_roi(self):
        pass

    def motion_correct(self):
        self.main_controller.show_motion_correction_params()

    def watershed(self):
        self.main_controller.show_watershed_params()

    def save_params(self):
        json.dump(self.params, open(ROI_PRUNING_PARAMS_FILENAME, "w"))