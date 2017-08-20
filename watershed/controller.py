from watershed.param_window import ParamWindow
from watershed.preview_window import PreviewWindow
import watershed.utilities as utilities
import time
import json
import os
import sys
import scipy.ndimage as ndi
import scipy.signal
import numpy as np
from skimage.external.tifffile import imread
from skimage.morphology import *
from imimposemin import imimposemin
import cv2

# import the Qt library
try:
    from PyQt4.QtCore import pyqtSignal, Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 4
except:
    from PyQt5.QtCore import pyqtSignal, Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog
    pyqt_version = 5

DEFAULT_PARAMS = {'gamma': 1.0,
                  'contrast': 1.0,
                  'window_size': 5,
                  'background_threshold': 50,
                  'soma_threshold': 10,
                  'compactness': 10}

PARAMS_FILENAME = "watershed_params.txt"

class Controller():
    def __init__(self, main_controller):
        self.main_controller = main_controller

        # set parameters
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

        self.image                = None
        self.adjusted_image       = None
        self.normalized_image     = None
        self.soma_threshold_image = None
        self.mask                 = None
        self.background_mask      = None

        # create parameters window
        self.param_window = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

    def open_image(self, image_path):
        base_name = os.path.basename(image_path)
        if base_name.endswith('.npy'):
            self.video = np.load(image_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = np.transpose(imread(image_path), (1, 2, 0))
        self.video = np.nan_to_num(self.video)

        self.image = utilities.normalize(utilities.mean(self.video)).astype(np.float32)

        # self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        self.adjusted_image = self.image.copy()
        self.calculate_adjusted_image()
        self.calculate_normalized_image()
        self.calculate_background_mask()
        self.calculate_soma_threshold_image()

        self.param_window.show()
        self.preview_window.show()

        self.preview_window.plot_image(self.adjusted_image)
        # self.preview_window.zoom(2)

    def calculate_background_mask(self):
        self.background_mask = self.normalized_image < self.params['background_threshold']

    def calculate_adjusted_image(self):
        self.adjusted_image = utilities.adjust_gamma(utilities.adjust_contrast(self.image, self.params['contrast']), self.params['gamma'])

        if self.mask is not None:
            out = np.zeros(self.image.shape)
            out[self.mask] = self.adjusted_image[self.mask]

            self.adjusted_image = out.copy()

    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.mask = None

            self.calculate_adjusted_image()
            self.calculate_normalized_image()
            self.calculate_soma_threshold_image()

            self.preview_window.plot_image(self.adjusted_image)

            self.preview_window.drawing_mask = True

            self.param_window.draw_mask_button.setText("Finished")
        else:
            if len(self.preview_window.image_label.mask_points) > 0:
                self.mask_points = self.preview_window.image_label.mask_points
                self.mask_points += [self.mask_points[0]]
                self.mask_points = np.array(self.mask_points)

                self.mask = np.zeros(self.image.shape)
                cv2.fillConvexPoly(self.mask, self.mask_points, 1)
                self.mask = self.mask.astype(np.bool)

                self.calculate_adjusted_image()
                self.calculate_normalized_image()
                self.calculate_soma_threshold_image()
            else:
                self.mask = None

            self.preview_window.end_drawing_mask()
            self.preview_window.plot_image(self.adjusted_image)

            self.param_window.draw_mask_button.setText("Draw Mask")

    def calculate_normalized_image(self):
        new_image_10 = utilities.order_statistic(self.adjusted_image, 0.1, self.params['window_size'])
        new_image_90 = utilities.order_statistic(self.adjusted_image, 0.9, self.params['window_size'])

        image_difference = self.adjusted_image - new_image_10
        image_difference[image_difference < 0] = 0

        image_range = new_image_90 - new_image_10
        image_range[image_range <= 0] = 1e-6

        normalized_image = utilities.rescale_0_1(image_difference/image_range)

        normalized_image[normalized_image < 0] = 0
        normalized_image[normalized_image > 1] = 1

        # normalized_image = 1.0 - normalized_image

        self.normalized_image = 255.0*(1.0 - normalized_image)

        if self.mask is not None:
            out = np.zeros(self.image.shape)
            out[self.mask] = self.normalized_image[self.mask]

            self.normalized_image = out.copy()

    # def calculate_regional_minima(self):
    #     image = self.normalized_image/255.0

    #     border_threshold = 0.15
    #     border_center    = 0.1



    def calculate_soma_threshold_image(self):
        I = 1 - self.normalized_image.copy()/255.0

        I[self.mask == 0] = 0

        mybw = cv2.threshold(I, self.params['background_threshold'])

        CC = cv2.findContours(mybw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        test = np.zeros((self.image.shape[0], self.image.shape[1], len(CC)))

        for i in range(len(CC)):
            temp = np.zeros(self.image.shape).astype(bool)
            temp[]

        soma_threshold_image = local_maxima(h_maxima(soma_threshold_image, self.params['soma_threshold']/255.0))
        
        I_mod = imimposemin()

        soma_threshold_image = soma_threshold_image*255        

        # soma_threshold_image[soma_threshold_image < self.params['soma_threshold']] = 0
        # soma_threshold_image[soma_threshold_image > 0] = 255

        # soma_threshold_image[self.background_mask] = 0

        # soma_threshold_image = erosion(soma_threshold_image)
        # soma_threshold_image = erosion(soma_threshold_image)

        # soma_threshold_image = 255 - soma_threshold_image

        self.soma_threshold_image = soma_threshold_image

        if self.mask is not None:
            out = np.zeros(self.image.shape)
            out[self.mask] = self.soma_threshold_image[self.mask]

            self.soma_threshold_image = out.copy()

    def update_background_threshold(self, background_threshold):
        self.params['background_threshold'] = background_threshold
        
        self.calculate_background_mask()

        image = self.normalized_image.copy()
        image[self.background_mask] = 0

        if self.mask is not None:
            out = np.zeros(self.image.shape)
            out[self.mask] = image[self.mask]

            image = out.copy()

        self.preview_window.plot_image(image)

    def update_contrast(self, contrast):
        self.params['contrast'] = contrast
        
        self.calculate_adjusted_image()
        
        self.preview_window.plot_image(self.adjusted_image)

    def update_gamma(self, gamma):
        self.params['gamma'] = gamma

        self.calculate_adjusted_image()

        self.preview_window.plot_image(self.adjusted_image)

    def update_window_size(self, window_size):
        self.params['window_size'] = window_size

        self.calculate_normalized_image()

        self.preview_window.plot_image(self.normalized_image)

    def update_soma_threshold(self, soma_threshold):
        self.params['soma_threshold'] = soma_threshold

        self.calculate_soma_threshold_image()

        self.preview_window.plot_image(self.soma_threshold_image)

    def update_compactness(self, compactness):
        self.params['compactness'] = compactness

    def show_background_mask(self):
        if self.soma_threshold_image is None:
            self.update_background_threshold(self.params['background_threshold'])

        image = self.normalized_image.copy()
        image[self.background_mask] = 0

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
        self.calculate_background_mask()
        self.calculate_soma_threshold_image()

        self.watershed_image = self.adjusted_image.copy()

        threshold_range = list(range(self.params['soma_threshold']-40, self.params['soma_threshold'] + 40, 5))
        threshold_range = list(range(self.params['soma_threshold'], self.params['soma_threshold'] + 1))

        cells_mask = np.ones(self.watershed_image.shape)*255

        for i in range(len(threshold_range)):
            print("Iteration {}/{}.".format(i+1, len(threshold_range)))
            t = threshold_range[i]
            self.watershed_image, cells_mask, centers_list = utilities.apply_watershed(self.watershed_image, cells_mask, self.soma_threshold_image, t, self.params['compactness'], centers_list)
            self.param_window.show_watershed_checkbox.setDisabled(False)
            self.param_window.show_watershed_checkbox.setChecked(True)
            self.show_watershed_image(True)

    def close_all(self):
        self.closing = True
        self.param_window.close()
        self.preview_window.close()

        json.dump(self.params, open(PARAMS_FILENAME, "w"))