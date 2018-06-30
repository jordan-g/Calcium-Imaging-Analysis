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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

import random

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class PlotCanvas(FigureCanvas):
    def __init__(self, window=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = plt.Axes(fig, [0., 0., 1., 1.])
        self.axes.set_axis_off()
        fig.add_axes(self.axes)
        self.new_load = True
        self.preview_window = window
 
        FigureCanvas.__init__(self, fig)
 
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        FigureCanvas.updateGeometry(self)
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        fig.canvas.mpl_connect('button_release_event', self.on_release)

        # initialize imshow
        self.plot = None
        self.background_mask_plot = None
        self.background_mask_contour_plot = None
        self.mask_line_collections = None
        self.mask_plot = None
        self.click_start_coord = None # coordinate where the user pressed down the mouse button

    def on_click(self, event):
        x, y = event.xdata, event.ydata

        self.click_start_coord = (int(x), int(y))

        self.prev_coord = self.click_start_coord

        self.preview_window.mouse_pressed(self.click_start_coord)

    def on_mouse_move(self, event):
        x, y = event.xdata, event.ydata

        self.click_end_coord = (int(x), int(y))

        self.preview_window.mouse_moved(self.click_start_coord, self.click_end_coord, clicked=(event.button == 1))
        
        self.prev_coord = self.click_end_coord

    def on_release(self, event):
        x, y = event.xdata, event.ydata

        self.click_end_coord = (int(x), int(y))

        # print("User clicked {}.".format(self.click_end_coord))

        self.preview_window.mouse_released(self.click_start_coord, self.click_end_coord, mouse_moved=(self.click_end_coord != self.click_start_coord))
 
    def show_image(self, image, vmax, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1):
        self.axes.draw_artist(self.axes.patch)

        if self.plot is None:
            self.plot = self.axes.imshow(image, interpolation='nearest', vmin=0, vmax=vmax, cmap='gray')
        else:
            self.plot.set_data(image)

            self.axes.draw_artist(self.plot)

        if self.background_mask_contour_plot is not None:
            for coll in self.background_mask_contour_plot.collections:
                coll.remove()
            self.background_mask_contour_plot = None

        if self.mask_line_collections is not None:
            for coll in self.mask_line_collections:
                coll.remove()
            self.mask_line_collections = None

        if self.background_mask_plot is None:
            if background_mask is not None:
                background_mask_image = np.zeros(background_mask.shape + (4,))
                background_mask_image[:, :, 0] = 1
                background_mask_image[:, :, -1] = 0.5
                background_mask_image[background_mask == 0] = 0

                self.background_mask_plot = self.axes.imshow(background_mask_image, interpolation='nearest')

                self.axes.draw_artist(self.background_mask_plot)

                self.background_mask_contour_plot = self.axes.contour(np.arange(background_mask_image.shape[1]), np.arange(background_mask_image.shape[1]), background_mask_image[:, :, 0], levels=[0], colors='r', linewidths=1)

                for coll in self.background_mask_contour_plot.collections:
                    self.axes.draw_artist(coll)
        else:
            if background_mask is not None:
                background_mask_image = np.zeros(background_mask.shape + (4,))
                background_mask_image[:, :, 0] = 1
                background_mask_image[:, :, -1] = 0.5
                background_mask_image[background_mask == 0] = 0

                self.background_mask_plot.set_data(background_mask_image)

                self.axes.draw_artist(self.background_mask_plot)

                self.background_mask_contour_plot = self.axes.contour(np.arange(background_mask_image.shape[1]), np.arange(background_mask_image.shape[1]), background_mask_image[:, :, 0], levels=[0], colors='r', linewidths=1)

                for coll in self.background_mask_contour_plot.collections:
                    self.axes.draw_artist(coll)
            else:
                try:
                    self.background_mask_plot.remove()
                    self.background_mask_plot = None
                except:
                    pass

        if self.mask_plot is None:
            if mask is not None:
                mask_image = np.zeros(mask.shape + (4,))
                mask_image[:, :, -1] = 0.6
                mask_image[mask != 0] = 0

                self.mask_plot = self.axes.imshow(mask_image, interpolation='nearest')

                self.axes.draw_artist(self.mask_plot)
        else:
            if mask is not None:
                mask_image = np.zeros(mask.shape + (4,))
                mask_image[:, :, -1] = 0.6
                mask_image[mask != 0] = 0

                self.mask_plot.set_data(mask_image)

                self.axes.draw_artist(self.mask_plot)
            else:
                try:
                    self.mask_plot.remove()
                    self.mask_plot = None
                except:
                    pass

        if mask_points is not None:
            self.mask_line_collections = []

            # draw the points along the individual masks
            for i in range(len(mask_points)):
                points = mask_points[i]

                if len(points) > 0:
                    segments = np.zeros((len(points)-1, 2, 2))
                    for j in range(segments.shape[0]):
                        segments[j, 0, :] = points[j]
                        segments[j, 1, :] = points[j+1]

                    if selected_mask_num == i:
                        color = 'g'
                    else:
                        color = 'y'

                    collection = LineCollection(segments, linewidths=1, colors=color, linestyle='solid')
                    self.axes.add_collection(collection)
                    self.mask_line_collections.append(collection)

                    self.axes.draw_artist(collection)

                    scatter_plot = self.axes.scatter([ point[0] for point in points[:-1] ], [ point[1] for point in points[:-1] ], c=color, s=10)
                    self.axes.draw_artist(scatter_plot)

                    if i == len(mask_points)-1 and tentative_mask_point is not None:
                        segment = np.zeros((1, 2, 2))
                        segment[0, 0, :] = points[-2]
                        segment[0, 1, :] = tentative_mask_point

                        collection = LineCollection(segment, linewidths=1, colors='gray', linestyle='solid')
                        self.axes.add_collection(collection)
                        self.mask_line_collections.append(collection)

                        self.axes.draw_artist(collection)

                        scatter_plot = self.axes.scatter(tentative_mask_point[0], tentative_mask_point[1], c='gray', s=10)
                        self.axes.draw_artist(scatter_plot)

        self.blit(self.axes.bbox)
        self.flush_events()

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
        self.main_layout.addWidget(self.image_widget, 0, 0)

        self.drawing_mask    = False # whether the user is drawing a mask
        self.erasing_rois    = False # whether the user is erasing ROIs
        self.drawing_rois    = False # whether the user is drawing ROIs
        self.mask_points     = []    # list holding the points of the mask being drawn
        self.mask            = None
        self.background_mask = None

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

        self.image_plot = PlotCanvas(self, width=5, height=5)
        self.image_layout.addWidget(self.image_plot)
        
        self.set_initial_state()

        self.show()

    def set_initial_state(self):
        # reset variables
        self.image      = None  # image to show
        self.frames     = None  # frames to play
        self.frame_num  = 0     # current frame #
        self.n_frames   = 1     # total number of frames
        self.video_name = ""    # name of the currently showing video

        self.image_plot.hide()
        self.timer.stop()
        self.setWindowTitle("Preview")

    def plot_image(self, image, background_mask=None, video_max=255):
        if self.image is None:
            self.image_plot.show()
            self.main_widget.setMinimumSize(QSize(image.shape[1], image.shape[0]))

        # update image
        self.image = image
        self.background_mask = background_mask

        # draw user-drawn masks
        if self.controller.mode == "roi_finding" and len(self.controller.controller.mask_points[self.controller.z]) > 0:
            # combine the masks into one using their union or intersection, depending on whether they are inverted or not
            masks = np.array(self.controller.controller.masks[self.controller.z])
            if self.controller.controller.params['invert_masks']:
                self.mask = np.prod(masks, axis=0).astype(bool)
            else:
                self.mask = np.sum(masks, axis=0).astype(bool)

            # self.image = image
            mask_points = self.controller.controller.mask_points[self.controller.z]
        else:
            mask_points = []
            self.mask = None
        
        # update image plot
        self.update_image_plot(self.image, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask, selected_mask_num=self.controller.selected_mask_num)

    def play_movie(self, frames, fps=60):
        if frames is None:
            # reset variables
            self.image                 = None
            self.frames                = None
            self.frame_num             = 0
            self.n_frames              = 1
            self.video_name            = ""

            self.update_image_plot(None)
            self.image_plot.hide()
        else:
            if self.frames is None:
                self.image_plot.show()
                self.main_widget.setMinimumSize(QSize(frames.shape[2], frames.shape[1]))
                self.image_plot.new_load = True

            # set frame number to 0
            self.frame_num = 0

            # normalize the frames (to be between 0 and 255)
            self.frames = (utilities.normalize(frames)).astype(np.uint8)

            # get the number of frames
            self.n_frames = self.frames.shape[0]

            # start the timer to update the frames
            self.timer.start(int(1000.0/fps))

    def play_video(self, video, video_path, fps):
        if video_path is not None:
            self.video_name = os.path.basename(video_path)
        else:
            self.video_name = ""
        self.play_movie(video, fps=fps)

    def set_fps(self, fps):
        # restart the timer with the new fps
        self.timer.stop()
        self.timer.start(int(1000.0/fps))

    def show_frame(self, frame):
        if frame is None:
            self.update_image_plot(None)
            self.frame_num = 0
            self.n_frames  = 1
            self.image_plot.hide()
        else:
            # convert to RGB
            frame = cv2.cvtColor(utilities.normalize(frame).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # update image plot
            self.update_image_plot(frame)

    def update_frame(self):
        if self.frames is not None:
            # convert the current frame to RGB
            frame = cv2.cvtColor(self.frames[self.frame_num], cv2.COLOR_GRAY2RGB)

            # update image plot
            self.update_image_plot(frame)

            # increment frame number (keeping it between 0 and n_frames)
            self.frame_num += 1
            self.frame_num = self.frame_num % self.n_frames

            # update window title
            self.setWindowTitle("{}. Z={}. Frame {}/{}.".format(self.video_name, self.controller.z, self.frame_num + 1, self.n_frames))

    def update_image_plot(self, image, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1):
        self.image_plot.show_image(image, self.controller.controller.video_max, background_mask=background_mask, mask_points=mask_points, tentative_mask_point=tentative_mask_point, mask=mask, selected_mask_num=selected_mask_num)

    def add_mask_point(self, mask_point):
        # add the point to the list of mask points
        self.mask_points.append(mask_point)

        mask_points = self.controller.controller.mask_points[self.controller.z] + [self.mask_points + [self.mask_points[0]]]

        # update image plot
        self.update_image_plot(self.image, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask)

    def add_tentative_mask_point(self, mask_point):
        # make a copy of the current image
        image = self.image.copy()

        if len(self.mask_points) > 0:
            mask_points = self.controller.controller.mask_points[self.controller.z] + [self.mask_points + [self.mask_points[0]]]
        else:
            mask_points = self.controller.controller.mask_points[self.controller.z] + [[]]

        # update image plot
        self.update_image_plot(image, background_mask=self.background_mask, mask_points=mask_points, tentative_mask_point=mask_point, mask=self.mask)

    def erase_roi_at_point(self, roi_point):
        # tell the controller to remove ROIs near this point
        self.controller.erase_rois_near_point(roi_point)

        # draw a circle showing the radius of the eraser around this point
        image   = self.image.copy()
        overlay = image.copy()
        cv2.circle(overlay, roi_point, 10, (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # update image plot
        self.update_image_plot(image)

    def end_erase_rois(self):
        # update image plot
        self.update_image_plot(self.image)

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

            # update image plot
            self.update_image_plot(image)

    def draw_roi(self, start_point, end_point):
        self.controller.create_roi(start_point, end_point)

    def shift_rois(self, previous_point, current_point):
        self.controller.shift_rois(previous_point, current_point)

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
            self.shift_rois(self.click_end_point, end_point)

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
