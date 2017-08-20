import sys
import os
import time
import json

import numpy as np
import cv2

import motion_correction.utilities as utilities

# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread, QSize, QTimer
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QGraphicsDropShadowEffect, QColor
except:
    from PyQt5.QtCore import Qt, QThread, QSize, QTimer
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon, QColor
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout, QGraphicsDropShadowEffect

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

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

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
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set title
        self.setWindowTitle("Preview")

        # get parameter window position & size
        param_window_x      = self.controller.param_window.x()
        param_window_y      = self.controller.param_window.y()
        param_window_width  = self.controller.param_window.width()
        param_window_height = self.controller.param_window.height()

        print(param_window_height)

        # set position & size
        self.setGeometry(param_window_x + param_window_width, param_window_y + param_window_height, 10, 10)

        # create main widget
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: #b3b9bc;")
        self.main_widget.setMinimumSize(QSize(500, 500))

        # create main layout
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
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
        self.frames    = None  # frames to play
        self.frame_num = 0
        self.n_frames  = 1

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def resizeEvent(self, event):
        # update label's pixmap size
        self.image_label.pix_size = self.get_available_pix_size()

        # update size of image label
        self.image_label.update_size()

    def get_available_pix_size(self):
        available_width  = self.width()
        available_height = self.height()

        if available_height < available_width:
            return available_height
        else:
            return available_width

    def play_movie(self, frames):
        if frames is None:
            self.update_image_label(None)
            self.frame_num = 0
            self.image_label.hide()
            self.n_frames = 1
        else:
            self.frame_num = 0
            self.image_label.show()

            self.frames = utilities.normalize(frames).astype(np.uint8)

            self.n_frames = self.frames.shape[0]

            self.timer.start(10)

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

    def zoom(self, zoom_level):
        self.resize(self.width()*zoom_level, self.height()*zoom_level)

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

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
