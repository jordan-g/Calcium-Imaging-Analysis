from motion_correction.param_window import ParamWindow
from motion_correction.preview_window import PreviewWindow
import motion_correction.utilities as utilities
import time
import json
import os
import glob
import scipy.ndimage as ndi
import scipy.signal
import numpy as np
from skimage.external.tifffile import imread

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

import caiman as cm

DEFAULT_PARAMS = {'gamma': 1.0,
                  'contrast': 1.0}

PARAMS_FILENAME = "motion_correction_params.txt"

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

        self.video                       = None
        self.adjusted_video              = None
        self.adjusted_frame              = None
        self.motion_corrected_video      = None
        self.video_path                  = None
        self.motion_corrected_video_path = None

        # create parameters & preview windows
        self.param_window   = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

    def open_video(self, video_path):
        self.video_path = video_path
        base_name = os.path.basename(video_path)

        if base_name.endswith('.npy'):
            self.video = np.load(video_path)
        elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = imread(video_path)

        print(self.video.shape)

        # remove nan elements
        self.video = np.nan_to_num(self.video)

        self.calculate_adjusted_video()

        self.param_window.show()
        self.preview_window.show()

        self.play_adjusted_video()

    def calculate_adjusted_video(self):
        if self.motion_corrected_video is not None:
            print(np.amax(self.video), np.amin(self.video), np.amax(self.motion_corrected_video), np.amin(self.motion_corrected_video))
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.motion_corrected_video, self.params['contrast']), self.params['gamma'])
        else:
            self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.video, self.params['contrast']), self.params['gamma'])

    def calculate_adjusted_frame(self):
        if self.motion_corrected_video is not None:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.motion_corrected_video[self.preview_window.frame_num], self.params['contrast']), self.params['gamma'])
        else:
            self.adjusted_frame = utilities.adjust_gamma(utilities.adjust_contrast(self.video[self.preview_window.frame_num], self.params['contrast']), self.params['gamma'])

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

    def update_contrast(self, contrast):
        self.preview_window.timer.stop()

        self.params['contrast'] = contrast
        
        self.calculate_adjusted_video()
        
        self.preview_window.play_movie(self.adjusted_video)

    def update_gamma(self, gamma):
        self.preview_window.timer.stop()

        self.params['gamma'] = gamma

        self.calculate_adjusted_video()

        self.preview_window.play_movie(self.adjusted_video)

    def play_frames(self):
        if self.video is not None:
            self.preview_window.play_movie(self.video)

    def play_adjusted_video(self):
        if self.adjusted_video is not None:
            self.preview_window.play_movie(self.adjusted_video)

    def play_motion_corrected_video(self):
        if self.motion_corrected_video is not None:
            self.preview_window.play_movie(self.adjusted_video)

    def process_video(self):
        self.motion_corrected_video, self.motion_corrected_video_path = utilities.motion_correct(self.video_path)

        self.calculate_adjusted_video()

        self.play_motion_corrected_video()

        self.param_window.accept_button.setEnabled(True)

    def accept(self):
        self.close_all()

        self.main_controller.motion_correction_done(self.motion_corrected_video_path)

    def skip(self):
        self.close_all()

        self.main_controller.motion_correction_done(self.video_path)

    def accept_cnmf(self):
        self.close_all()

        self.main_controller.motion_correction_done_cnmf(self.motion_corrected_video_path)

    def skip_cnmf(self):
        self.close_all()

        self.main_controller.motion_correction_done_cnmf(self.video_path)

    def close_all(self):
        cm.stop_server()

        self.closing = True
        self.param_window.close()
        self.preview_window.close()

        json.dump(self.params, open(PARAMS_FILENAME, "w"))