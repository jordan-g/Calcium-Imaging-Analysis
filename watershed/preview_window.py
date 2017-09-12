import sys
import os
import time
import json

import numpy as np
import cv2

import watershed.utilities as utilities
import matplotlib.pyplot as plt

# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class PreviewQLabel(QLabel):
    def __init__(self, preview_window):
        QLabel.__init__(self)

        self.preview_window = preview_window
        self.scale_factor   = None
        self.pix            = None  # image label's pixmap
        self.pix_size       = None  # size of image label's pixmap

        # accept clicks
        self.setAcceptDrops(True)

        self.erase_points   = []
        self.pressed_point  = None
        self.released_point = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.scale_factor is not None:
            if self.preview_window.drawing_mask:
                self.preview_window.add_mask_point((int(round(event.x()/self.scale_factor)), int(round(event.y()/self.scale_factor))))
            elif not self.preview_window.erasing_rois:
                self.pressed_point = (int(round(event.x()/self.scale_factor)), int(round(event.y()/self.scale_factor)))

    def mouseMoveEvent(self, event):
        if self.scale_factor is not None:
            if self.preview_window.drawing_mask:
                self.preview_window.add_tentative_mask_point((int(round(event.x()/self.scale_factor)), int(round(event.y()/self.scale_factor))))
            elif self.preview_window.erasing_rois and event.buttons() & Qt.LeftButton:
                self.preview_window.erase_roi_at_point((int(round(event.x()/self.scale_factor)), int(round(event.y()/self.scale_factor))))
    
    def mouseReleaseEvent(self, event):
        self.released_point = (int(round(event.x()/self.scale_factor)), int(round(event.y()/self.scale_factor)))
        if (not self.preview_window.erasing_rois) and (not self.preview_window.drawing_mask) and self.preview_window.mode == "prune":
            if self.released_point == self.pressed_point:
                self.preview_window.select_roi(self.released_point)
        elif self.preview_window.erasing_rois and event.buttons() & Qt.LeftButton:
            self.preview_window.end_erase_rois()

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def update_size(self):
        if self.pix:
            # calculate new label vs. image scale factor
            scale_factor = float(self.pix_size)/max(self.pix.width(), self.pix.height())
            self.scale_factor = scale_factor

            # scale pixmap
            pix = self.pix.scaled(self.pix.width()*scale_factor, self.pix.height()*scale_factor, Qt.KeepAspectRatio, Qt.FastTransformation)

            # update pixmap & size
            self.setPixmap(pix)
            self.setFixedSize(pix.size())

    def update_pixmap(self, image):
        if image is None:
            self.scale_factor   = None
            self.pix            = None  # image label's pixmap
            self.pix_size       = None  # size of image label's pixmap
            self.clear()
        else:
            # get image info
            height, width, bytesPerComponent = image.shape
            bytesPerLine = bytesPerComponent * width
            
            # create qimage
            qimage = QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QImage.Format_RGB888)
            qimage.setColorTable(gray_color_table)

            # generate pixmap
            self.pix = QPixmap(qimage)

class PreviewWindow(QMainWindow):
    def __init__(self, main_controller):
        QMainWindow.__init__(self)

        # set controller
        self.main_controller = main_controller

        # set title
        self.setWindowTitle("Preview")

        # get parameter window position & size
        param_window_x      = self.main_controller.param_window.x()
        param_window_y      = self.main_controller.param_window.y()
        param_window_width  = self.main_controller.param_window.width()
        param_window_height = self.main_controller.param_window.height()

        # set position & size
        self.setGeometry(param_window_x + param_window_width, param_window_y, 10, 10)

        # create main widget
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: #000;")
        self.main_widget.setMinimumSize(QSize(500, 500))

        # create main layout
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(0)

        # create label that shows frames
        self.image_widget = QWidget(self)
        self.image_layout = QHBoxLayout(self.image_widget)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = PreviewQLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()
        self.image_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.image_widget, 0, 0)

        # initialize variables
        self.image     = None  # image to show
        self.frames    = None  # frames to play
        self.frame_num = 0
        self.n_frames  = 1

        self.drawing_mask = False
        self.erasing_rois = False
        self.mask_points = []

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.show()

    def resizeEvent(self, event):
        # update label's pixmap size
        self.image_label.pix_size = self.get_available_pix_size()

        # update size of image label
        self.image_label.update_size()

    def get_available_pix_size(self):
        available_width  = self.width() - 20
        available_height = self.height() - 20

        if available_height < available_width:
            return available_height
        else:
            return available_width

    def plot_image(self, image):
        if image is None:
            self.update_image_label(None)
            self.image_label.hide()
        else:
            self.image_label.show()

            if self.image is None:
                self.main_widget.setMinimumSize(QSize(image.shape[0] + 20, image.shape[1] + 20))

            normalized_image = utilities.normalize(image)

            # convert to RGB
            if len(normalized_image.shape) == 2:
                normalized_image = cv2.cvtColor(normalized_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # update image
            self.image = normalized_image

            if self.mode == "watershed" and len(self.controller.mask_points) > 0:
                image = self.image.copy()
                for mask_points in self.controller.mask_points:
                    self.draw_mask_points(mask_points, image)

                self.update_image_label(image)
            else:
                # update image label
                self.update_image_label(self.image)

    def play_movie(self, frames, fps=60):
        if frames is None:
            self.update_image_label(None)
            self.frame_num = 0
            self.image_label.hide()
            self.n_frames = 1
        else:
            if self.frames is None:
                self.main_widget.setMinimumSize(QSize(frames.shape[1] + 20, frames.shape[2] + 20))

            self.frame_num = 0
            self.image_label.show()

            self.frames = utilities.normalize(frames).astype(np.uint8)

            self.n_frames = self.frames.shape[0]

            self.timer.start(int(1000.0/fps))

    def set_fps(self, fps):
        self.timer.stop()
        self.timer.start(int(1000.0/fps))

    def show_frame(self, frame):
        if frame is None:
            self.update_image_label(None)
            self.frame_num = 0
            self.image_label.hide()
            self.n_frames = 1
        else:
            self.frame_num = 0
            self.image_label.show()

            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            self.image_label.update_pixmap(frame)

            # update label's pixmap size
            self.image_label.pix_size = self.get_available_pix_size()

            # update size of image label
            self.image_label.update_size()

    def update_frame(self):
        if self.frames is not None:
            frame = cv2.cvtColor(self.frames[self.frame_num], cv2.COLOR_GRAY2RGB)

            self.image_label.update_pixmap(frame)

            self.frame_num += 1

            self.frame_num = self.frame_num % self.n_frames

            # update label's pixmap size
            self.image_label.pix_size = self.get_available_pix_size()

            # update size of image label
            self.image_label.update_size()

            self.setWindowTitle("Preview: Frame {}/{}.".format(self.frame_num + 1, self.n_frames))

    def zoom(self, zoom_level):
        self.resize(self.width()*zoom_level, self.height()*zoom_level)

    def update_image_label(self, image):
        self.image_label.update_pixmap(image)

        # update label's pixmap size
        self.image_label.pix_size = self.get_available_pix_size()

        # update size of image label
        self.image_label.update_size()

    def draw_mask_points(self, mask_points, image=None):
        if image is None:
            image = self.image.copy()

        n_points = len(mask_points)
        if n_points >= 1:
            for i in range(n_points):
                if i < n_points - 1:
                    cv2.line(image, mask_points[i], mask_points[i+1], (255, 255, 0), 1)
                cv2.circle(image, mask_points[i], 2, (255, 255, 0), -1)

    def draw_tentative_mask_points(self, mask_points, image=None):
        if image is None:
            image = self.image.copy()

        n_points = len(mask_points)
        if n_points >= 1:
            for i in range(n_points):
                if i < n_points - 1:
                    cv2.line(image, mask_points[i], mask_points[i+1], (128, 128, 128), 1)

    def add_mask_point(self, mask_point):
        self.mask_points.append(mask_point)

        image = self.image.copy()
        for mask_points in self.controller.mask_points:
            self.draw_mask_points(mask_points, image)
        self.draw_mask_points(self.mask_points + [self.mask_points[0]], image)

        self.update_image_label(image)

    def add_tentative_mask_point(self, mask_point):
        image = self.image.copy()
        for mask_points in self.controller.mask_points:
            self.draw_mask_points(mask_points, image)
        if len(self.mask_points) > 0:
            self.draw_mask_points(self.mask_points + [self.mask_points[0]], image)
            self.draw_tentative_mask_points([self.mask_points[-1], mask_point], image)

        self.update_image_label(image)

    def erase_roi_at_point(self, roi_point):
        self.controller.erase_roi_at_point(roi_point)

    def end_erase_rois(self):
        self.image_label.erased_rois = []

    def end_drawing_mask(self):
        self.drawing_mask = False
        self.mask_points = []
        self.image_label.mask_points = []

    def select_roi(self, roi_point):
        self.controller.select_roi(roi_point)

    def closeEvent(self, ce):
        if not self.main_controller.closing:
            ce.ignore()
        else:
            ce.accept()
