from __future__ import division
import sys
import os
import time
import json

import numpy as np
import cv2

import utilities
import matplotlib.pyplot as plt
from skimage.morphology import *

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

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class PreviewQLabel(QLabel):
    """
    QLabel subclass used to show a preview image.
    """

    def __init__(self, preview_window):
        QLabel.__init__(self)
        
        self.preview_window    = preview_window
        self.scale_factor      = None
        self.pix               = None  # image label's pixmap
        self.pix_size          = None  # size of image label's pixmap
        self.image             = None
        self.click_start_coord = None # coordinate where the user pressed down the mouse button

        # accept clicks
        self.setAcceptDrops(True)

        # track when the user moves the mouse over the label, even if they haven't clicked
        self.setMouseTracking(True)

    def resizeEvent(self, event):
        """
        Function called when the user resizes the window.
        """

        if self.pix is not None:
            self.setPixmap(self.pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio))

            # update scale factor
            self.scale_factor = self.pixmap().height()/self.image.shape[0]

    def mousePressEvent(self, event):
        """
        Function called when the user presses down the mouse button.
        """

        if self.scale_factor:
            self.click_start_coord = (int(event.x()/self.scale_factor), int(event.y()/self.scale_factor))

            self.prev_coord = self.click_start_coord

            self.preview_window.mouse_pressed(self.click_start_coord)

    def mouseMoveEvent(self, event):
        """
        Function called when the user presses moves the mouse button over the label.
        """

        if self.scale_factor:
            self.click_end_coord = (int(event.x()/self.scale_factor), int(event.y()/self.scale_factor))

            self.preview_window.mouse_moved(self.click_start_coord, self.click_end_coord, clicked=(event.buttons() & Qt.LeftButton))
    
            self.prev_coord = self.click_end_coord

    def mouseReleaseEvent(self, event):
        """
        Function called when the user releases the mouse button.
        """

        if self.scale_factor:
            self.click_end_coord = (int(event.x()/self.scale_factor), int(event.y()/self.scale_factor))

            print("User clicked {}.".format(self.click_end_coord))

            self.preview_window.mouse_released(self.click_start_coord, self.click_end_coord, mouse_moved=(self.click_end_coord != self.click_start_coord))

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

            self.setPixmap(self.pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.FastTransformation))

            # self.scale_factor = min(self.height()/image.shape[0], self.width()/image.shape[1])
            self.scale_factor = self.pixmap().height()/image.shape[0]
        
        self.image = image

class PreviewWindow(QMainWindow):
    """
    QMainWindow subclass used to show frames & tracking.
    """

    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # get parameter window position & size
        param_window_x      = self.controller.param_window.x()
        param_window_y      = self.controller.param_window.y()
        param_window_width  = self.controller.param_window.width()

        # set position & size
        self.setGeometry(param_window_x + param_window_width, param_window_y, 10, 10)

        # create main widget
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: #000;")
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
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.image_widget, 0, 0)

        self.drawing_mask = False # whether the user is drawing a mask
        self.erasing_rois = False # whether the user is erasing ROIs
        self.drawing_rois = False # whether the user is drawing ROIs
        self.mask_points  = []    # list holding the points of the mask being drawn

        # create a shortcut for ending mask creation
        self.done_creating_mask_shortcut = QShortcut(QKeySequence('Return'), self)

        # set main widget
        self.setCentralWidget(self.main_widget)

        # set window buttons
        if pyqt_version == 5:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)
        else:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        # create a timer for updating the frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.set_initial_state()

        self.show()

    def set_initial_state(self):
        # reset variables
        self.image      = None  # image to show
        self.frames     = None  # frames to play
        self.frame_num  = 0     # current frame #
        self.n_frames   = 1     # total number of frames
        self.video_name = ""    # name of the currently showing video

        self.update_image_label(None)
        self.image_label.hide()
        self.timer.stop()
        self.setWindowTitle("Preview")

    def plot_image(self, image, background_mask=None, video_max=255):
        if self.image is None:
            self.image_label.show()
            self.main_widget.setMinimumSize(QSize(image.shape[1], image.shape[0]))

        # normalize the image (to be between 0 and 255)
        normalized_image = utilities.normalize(image, video_max)

        # convert to RGB
        if len(normalized_image.shape) == 2:
            normalized_image = cv2.cvtColor(normalized_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # update image
        self.image = normalized_image

        # draw the background mask as red
        if background_mask is not None:
            self.image[background_mask > 0] = np.array([255, 0, 0]).astype(np.uint8)

        # draw user-drawn masks
        if self.controller.mode == "roi_finding" and len(self.controller.mask_points[self.controller.z]) > 0:
            # # make a copy of the image
            # image = self.image.copy()

            # combine the masks into one using their union or intersection, depending on whether they are inverted or not
            masks = np.array(self.controller.masks[self.controller.z])
            if self.controller.params['invert_masks']:
                mask = np.prod(masks, axis=0).astype(bool)
            else:
                mask = np.sum(masks, axis=0).astype(bool)

            # darken the image outside of the mask
            self.image[mask == False] = self.image[mask == False]/2

            # draw the points along the individual masks
            for i in range(len(self.controller.mask_points[self.controller.z])):
                mask_points = self.controller.mask_points[self.controller.z][i]

                self.draw_mask_points(mask_points, self.image, selected=(i == self.controller.selected_mask_num))

            # self.image = image
        
        # update image label
        self.update_image_label(self.image)

    def play_movie(self, frames, fps=60):
        if frames is None:
            # reset variables
            self.image                 = None
            self.frames                = None
            self.frame_num             = 0
            self.n_frames              = 1
            self.video_name            = ""

            self.update_image_label(None)
            self.image_label.hide()
        else:
            if self.frames is None:
                self.image_label.show()
                self.main_widget.setMinimumSize(QSize(frames.shape[2], frames.shape[1]))
                self.image_label.new_load = True

            # set frame number to 0
            self.frame_num = 0

            # normalize the frames (to be between 0 and 255)
            self.frames = (utilities.normalize(frames)).astype(np.uint8)

            # get the number of frames
            self.n_frames = self.frames.shape[0]

            # start the timer to update the frames
            self.timer.start(int(1000.0/fps))

    def video_opened(self, video_path):
        self.video_name = os.path.basename(video_path)
        self.timer.stop()

    def set_fps(self, fps):
        # restart the timer with the new fps
        self.timer.stop()
        self.timer.start(int(1000.0/fps))

    def show_frame(self, frame):
        if frame is None:
            self.update_image_label(None)
            self.frame_num = 0
            self.n_frames  = 1
            self.image_label.hide()
        else:
            # convert to RGB
            frame = cv2.cvtColor(utilities.normalize(frame).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # update image label
            self.update_image_label(frame)

    def update_frame(self):
        if self.frames is not None:
            # convert the current frame to RGB
            frame = cv2.cvtColor(self.frames[self.frame_num], cv2.COLOR_GRAY2RGB)

            # update image label
            self.update_image_label(frame)

            # increment frame number (keeping it between 0 and n_frames)
            self.frame_num += 1
            self.frame_num = self.frame_num % self.n_frames

            # update window title
            self.setWindowTitle("{}. Z={}. Frame {}/{}.".format(self.video_name, self.controller.params['z'], self.frame_num + 1, self.n_frames))

    def update_image_label(self, image):
        self.image_label.update_pixmap(image)

    def draw_mask_points(self, mask_points, image=None, selected=False):
        # make a copy of the current image if none is provided
        if image is None:
            image = self.image.copy()

        # set color of points along the selected vs. not selected mask
        if selected:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)

        # get the number of points along the mask
        n_points = len(mask_points)

        # draw the points along the mask
        if n_points >= 1:
            for i in range(n_points):
                if i < n_points - 1:
                    cv2.line(image, mask_points[i], mask_points[i+1], color, 1)
                cv2.circle(image, mask_points[i], 2, color, -1)

    def draw_tentative_mask_point(self, tentative_mask_point, last_mask_point, image=None):
        # make a copy of the current image if none is provided
        if image is None:
            image = self.image.copy()

        # draw a line connecting the tentative point to the last mask point
        cv2.line(image, last_mask_point, tentative_mask_point, (128, 128, 128), 1)

    def add_mask_point(self, mask_point):
        # add the point to the list of mask points
        self.mask_points.append(mask_point)

        # make a copy of the current image
        image = self.image.copy()

        # draw any existing masks that have been created
        for mask_points in self.controller.mask_points[self.controller.z]:
            self.draw_mask_points(mask_points, image)
        
        # draw the points of the current mask being created
        self.draw_mask_points(self.mask_points + [self.mask_points[0]], image)

        # update image label
        self.update_image_label(image)

    def add_tentative_mask_point(self, mask_point):
        # make a copy of the current image
        image = self.image.copy()

        # draw any existing masks that have been created
        for mask_points in self.controller.mask_points[self.controller.z]:
            self.draw_mask_points(mask_points, image)

        # draw the points of the current mask being created, including the tentative point
        if len(self.mask_points) > 0:
            self.draw_mask_points(self.mask_points + [self.mask_points[0]], image)
            self.draw_tentative_mask_point(mask_point, self.mask_points[-1], image)

        # update image label
        self.update_image_label(image)

    def erase_roi_at_point(self, roi_point):
        # tell the controller to remove ROIs near this point
        self.controller.erase_rois_near_point(roi_point)

        # draw a circle showing the radius of the eraser around this point
        image   = self.image.copy()
        overlay = image.copy()
        cv2.circle(overlay, roi_point, 10, (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # update image label
        self.update_image_label(image)

    def end_erase_rois(self):
        # update image label
        self.update_image_label(self.image)

    def start_drawing_mask(self):
        self.drawing_mask = True

        # enable the shortcut to finish drawing the mask
        self.done_creating_mask_shortcut.activated.connect(self.controller.draw_mask)

    def end_drawing_mask(self):
        self.drawing_mask = False

        # reset mask points list
        self.mask_points = []

        # disable the shortcut to finish drawing the mask
        self.done_creating_mask_shortcut.activated.disconnect()

    def select_roi(self, roi_point):
        self.controller.select_roi(roi_point)

    def select_mask(self, roi_point):
        self.controller.select_mask(roi_point)

    def draw_tentative_roi(self, start_point, end_point):
        if end_point != start_point:
            # make a copy of the current image
            image = self.image.copy()

            # create a mask that shows the boundary of an elliptical ROI fitting between the start & end points
            mask  = np.zeros((self.image.shape[0], self.image.shape[1])).astype(np.uint8)
            center_point = (int(round((end_point[0] + start_point[0])/2)), int(round((end_point[1] + start_point[1])/2)))
            axis_1 = np.abs(center_point[0] - end_point[0])
            axis_2 = np.abs(center_point[1] - end_point[1])
            cv2.ellipse(mask, center_point, (axis_1, axis_2), 0, 0, 360, 1, -1)
            b = erosion(mask, disk(1))
            mask = mask - b

            # draw the boundary on the image
            image[mask == 1] = np.array([255, 255, 0]).astype(np.uint8)

            # update image label
            self.update_image_label(image)

    def draw_roi(self, start_point, end_point):
        self.controller.create_roi(start_point, end_point)

    def shift_labels(self, previous_point, current_point):
        self.controller.shift_labels(previous_point, current_point)

    def mouse_pressed(self, point):
        if self.controller.mode == "roi_finding" and self.drawing_mask:
            self.add_mask_point(point)
        elif self.controller.mode == "roi_filtering" and not self.drawing_rois:
            # store this point
            self.click_end_point = point

    def mouse_moved(self, start_point, end_point, clicked=False):
        if self.drawing_mask:
            self.add_tentative_mask_point(end_point)
        elif self.erasing_rois and clicked:
            self.erase_roi_at_point(end_point)
        elif self.drawing_rois and clicked:
            self.draw_tentative_roi(start_point, end_point)
        elif self.controller.mode == "roi_filtering" and clicked:
            self.shift_labels(self.click_end_point, end_point)

            # store this point
            self.click_end_point = end_point

    def mouse_released(self, start_point, end_point, mouse_moved=False):
        if self.controller.mode == "roi_finding":
            if not self.drawing_mask and not mouse_moved:
                self.select_mask(end_point)
        elif self.controller.mode == "roi_filtering":
            if self.erasing_rois:
                self.end_erase_rois()
            elif self.drawing_rois:
                self.draw_roi(start_point, end_point)
            elif not mouse_moved:
                self.select_roi(end_point)

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
