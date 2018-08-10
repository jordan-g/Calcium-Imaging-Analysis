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
import pdb

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), ]

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
        self.roi_overlay       = None
        self.flat_contours     = None

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

            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                print("Shift key held.")
                shift_held = True
            else:
                shift_held = False

            self.preview_window.mouse_pressed(self.click_start_coord, shift_held=shift_held)

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

            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                print("Shift key held.")
                shift_held = True
            else:
                shift_held = False

            self.preview_window.mouse_released(self.click_start_coord, self.click_end_coord, mouse_moved=(self.click_end_coord != self.click_start_coord), shift_held=shift_held)

    def update_pixmap(self, image, update_stored_image=True):
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
        
        if update_stored_image:
            self.image = image

    def select_roi(self, roi_to_select):
        print("Selecting ROI {}.".format(roi_to_select))
        cv2.drawContours(self.image, self.all_contours[roi_to_select], -1, (255, 255, 0), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def select_rois(self, rois_to_select):
        for roi in rois_to_select:
            cv2.drawContours(self.image, self.all_contours[roi], -1, (255, 255, 0), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def show_eraser(self, point):
        image   = self.image.copy()
        overlay = image.copy()
        cv2.circle(overlay, point, 10, (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        self.update_pixmap(image, update_stored_image=False)

    def deselect_rois(self):
        cv2.drawContours(self.image, self.flat_contours, -1, (255, 0, 0), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def erase_roi(self, roi_to_erase):
        cv2.drawContours(self.image, self.all_contours[roi_to_erase], -1, (128, 128, 128), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def unerase_roi(self, roi):
        cv2.drawContours(self.image, self.all_contours[roi], -1, (255, 0, 0), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def erase_rois(self, rois_to_erase):
        print(len(self.all_contours))
        for roi_to_erase in rois_to_erase:
            cv2.drawContours(self.image, self.all_contours[roi_to_erase], -1, (128, 128, 128), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def unerase_rois(self, rois_to_unerase):
        for roi in rois_to_unerase:
            cv2.drawContours(self.image, self.all_contours[roi], -1, (255, 0, 0), 1)

        self.update_pixmap(self.image, update_stored_image=False)

    def show_roi_nums(self, roi_nums):
        image = self.image.copy()

        for roi in roi_nums:
            contour = self.all_contours[roi][0]
            M = cv2.moments(contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                centroid = (center_x + 10, center_y + 10)

                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image, "{}".format(roi), centroid, font, 1, (255, 255, 0), 1, cv2.LINE_AA)

        self.update_pixmap(image, update_stored_image=False)

    def show_image(self, image, vmax, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, show_rois=False, update_stored_image=True, use_existing_roi_overlay=False):
        image = 255.0*image/vmax
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        # if background_mask is not None:
        #     background_mask_image = image.copy()
        #     background_mask_image[background_mask, :] = 0
        #     # background_mask_image[background_mask, 0]  = 255

        #     cv2.addWeighted(background_mask_image, 0.5, image, 0.5, 0, image)

        #     contours = cv2.findContours(background_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        #     cv2.drawContours(image, contours, -1, (128, 128, 128), 1)

        # if mask is not None:
        #     mask_image = image.copy()
        #     mask_image[mask == 0, :] = 0

        #     cv2.addWeighted(mask_image, 0.6, image, 0.4, 0, image)

        # if mask_points is not None:
        #     # draw the points along the individual masks
        #     for i in range(len(mask_points)):
        #         points = mask_points[i]

        #         if i == selected_mask_num:
        #             color = (0, 255, 0)
        #         else:
        #             color = (255, 255, 0)

        #         if len(points) > 1:
        #             for j in range(len(points)):
        #                 if j < len(points) - 1:
        #                     cv2.line(image, points[j], points[j+1], color, 1)
        #                 cv2.circle(image, points[j], 2, color, -1)

        #         if i == len(mask_points)-1 and tentative_mask_point is not None:
        #             color = (128, 128, 128)
        #             if len(points) > 1:
        #                 cv2.line(image, points[-2], tentative_mask_point, color, 1)
        #             cv2.circle(image, tentative_mask_point, 2, color, -1)

        if show_rois:
            if roi_spatial_footprints is not None:
                roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((video_dimensions[1], video_dimensions[2], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

            if video_dimensions is not None:
                    # overlay = np.zeros(image.shape).astype(np.uint8)
                if (not use_existing_roi_overlay or self.roi_overlay is None) and roi_spatial_footprints is not None:
                    print("Computing new contours.....")
                    self.all_contours = []
                    self.flat_contours = []

                    total_mask = np.zeros((video_dimensions[1], video_dimensions[2])).astype(bool)

                    maximum_mask = np.ones((image.shape[0], image.shape[1]))

                    kept_footprints = []

                    for i in range(roi_spatial_footprints.shape[-1]):
                        # if removed_rois is not None and i not in removed_rois:
                        maximum = np.amax(roi_spatial_footprints[:, :, i])
                        mask = (roi_spatial_footprints[:, :, i] > 0.1*maximum).copy()

                        # maximum_mask[mask] = maximum

                        # total_mask += mask
                        
                        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                        self.all_contours.append(contours)
                        self.flat_contours += contours

                            # kept_footprints.append(i)

                    # roi_sum = np.sum(roi_spatial_footprints[:, :, kept_footprints], axis=-1)

                    # total_mask = roi_sum > 0

                    # overlay[total_mask, 0] = (255.0*roi_sum/maximum_mask).astype(np.uint8)[total_mask]

                    cv2.drawContours(image, self.flat_contours, -1, (255, 0, 0), 1)

                    self.image = image

                # image = self.roi_overlay.copy()

                # all_selected_contours = []

                # for i in selected_rois:
                #     all_selected_contours += self.all_contours[i]

                # # # plot selected ROIs
                # if selected_rois is not None and len(selected_rois) > 0:
                #     cv2.drawContours(image, all_selected_contours, -1, (255, 255, 0), 1)

                # if update_overlay:
                #     self.roi_overlay = image
                #     if roi_spatial_footprints is not None:
                #         # maximum_mask = np.ones((image.shape[0], image.shape[1]))

                #         a = roi_spatial_footprints[:, :, selected_rois]

                #         maximum = np.amax(roi_spatial_footprints[:, :, selected_rois])

                #         # for i in selected_rois:
                #         #     # print(roi_spatial_footprints.shape)
                #         #     maximum = np.amax(roi_spatial_footprints[:, :, i])
                #         #     mask = (roi_spatial_footprints[:, :, i] > 0.1*maximum).copy()
                #         #     maximum_mask[mask] = maximum

                #         roi_sum = np.sum(roi_spatial_footprints[:, :, selected_rois], axis=-1)
                #         total_mask = roi_sum > 0

                #         # total_mask = 

                #         overlay[total_mask, :-1] = (255.0*roi_sum/maximum).astype(np.uint8)[total_mask][:, np.newaxis]

                # cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

                # # plot selected ROIs
                # if selected_rois is not None and len(selected_rois) > 0:
                #     if roi_spatial_footprints is not None:
                #         all_contours = []

                #         for i in range(len(selected_rois)):
                #             roi = selected_rois[i]

                #             if i < len(colors):
                #                 color = colors[i]
                #             else:
                #                 color = colors[-1]

                #             maximum = np.amax(roi_spatial_footprints[:, :, roi])
                #             mask = (roi_spatial_footprints[:, :, roi] > 0.1*maximum).copy()
                #             contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
                #             all_contours += contours

                #             font = cv2.FONT_HERSHEY_PLAIN
                #             coord = np.unravel_index(mask.argmax(), mask.shape)
                #             coord = (coord[1], coord[0])
                #             cv2.putText(image, "{}".format(roi), coord, font, 1, color, 1, cv2.LINE_AA)

                #         cv2.drawContours(image, all_contours, -1, (255, 255, 0), 1)

                    # self.roi_selected_

        self.update_pixmap(image, update_stored_image=update_stored_image)

        return image

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
        self.setGeometry(param_window_x + param_window_width + 32, param_window_y + 32, 10, 10)

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
        self.image_plot = PreviewQLabel(self)
        self.image_plot.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.image_plot.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_layout.addWidget(self.image_plot)
        self.main_layout.addWidget(self.image_widget, 0, 0)

        self.drawing_mask    = False # whether the user is drawing a mask
        self.erasing_rois    = False # whether the user is erasing ROIs
        self.drawing_rois    = False # whether the user is drawing ROIs
        self.mask_points     = []    # list holding the points of the mask being drawn
        self.mask            = None
        self.background_mask = None

        # create a shortcut for ending mask creation
        self.done_creating_mask_shortcut = QShortcut(QKeySequence('Return'), self)

        # self.trace_rois_action = QAction('Plot Traces', self)
        # self.trace_rois_action.setShortcut('T')
        # self.trace_rois_action.setStatusTip('Plot traces for the selected ROIs.')
        # self.trace_rois_action.triggered.connect(self.controller.update_trace_plot)
        # self.trace_rois_action.setEnabled(False)

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

        # self.image_plot = PlotCanvas(self, width=5, height=5)
        # self.image_layout.addWidget(self.image_plot)
        
        self.set_initial_state()

        self.show()

    def set_initial_state(self):
        # reset variables
        self.image           = None  # image to show
        self.final_image     = None
        self.frames          = None  # frames to play
        self.frame_num       = 0     # current frame #
        self.n_frames        = 1     # total number of frames
        self.video_name      = ""    # name of the currently showing video
        self.mask_points     = []    # list holding the points of the mask being drawn
        self.mask            = None
        self.background_mask = None

        self.image_plot.hide()
        self.timer.stop()
        self.setWindowTitle("Preview")

    def plot_image(self, image, background_mask=None, video_max=255, update_overlay=True):
        if self.image is None:
            self.image_plot.show()
            self.main_widget.setMinimumSize(QSize(image.shape[1], image.shape[0]))

        # update image
        self.image = image
        self.background_mask = background_mask

        # # draw user-drawn masks
        # if self.controller.mode == "roi_finding" and len(self.controller.controller.mask_points[self.controller.z]) > 0:
        #     # combine the masks into one using their union or intersection, depending on whether they are inverted or not
        #     masks = np.array(self.controller.controller.masks[self.controller.z])
        #     if self.controller.controller.params['invert_masks']:
        #         self.mask = np.prod(masks, axis=0).astype(bool)
        #     else:
        #         self.mask = np.sum(masks, axis=0).astype(bool)

        #     # self.image = image
        #     mask_points = self.controller.controller.mask_points[self.controller.z]
        # else:
        mask_points = []
        self.mask = None
        
        # update image plot
        self.update_image_plot(self.image, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask, selected_mask_num=self.controller.selected_mask_num, roi_spatial_footprints=self.controller.controller.roi_spatial_footprints[self.controller.z], manual_roi_spatial_footprints=self.controller.controller.manual_roi_spatial_footprints[self.controller.z], video_dimensions=self.controller.video.shape, removed_rois=self.controller.controller.removed_rois[self.controller.z], selected_rois=self.controller.selected_rois, manual_roi_selected=self.controller.manual_roi_selected, use_existing_roi_overlay=(not update_overlay))

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
            self.frames = frames

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
            # frame = cv2.cvtColor(utilities.normalize(frame).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # update image plot
            self.update_image_plot(frame, background_mask=self.background_mask)

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

    def update_image_plot(self, image, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, use_existing_roi_overlay=False):
        final_image = self.image_plot.show_image(image, self.controller.controller.video_max, background_mask=background_mask, mask_points=mask_points, tentative_mask_point=tentative_mask_point, mask=mask, selected_mask_num=selected_mask_num, roi_spatial_footprints=roi_spatial_footprints, manual_roi_spatial_footprints=manual_roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, manual_roi_selected=manual_roi_selected, show_rois=self.controller.show_rois, use_existing_roi_overlay=use_existing_roi_overlay)

        # if update_final_image:
        self.final_image = final_image.copy()

    def update_image_plot_simple(self, image):
        self.image_plot.show_image(image, 255, update_stored_image=False)

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

    def select_roi_at_point(self, roi_point):
        # tell the controller to remove ROIs near this point
        new_roi_selected = self.controller.select_rois_near_point(roi_point)

        # draw a circle showing the radius of the eraser around this point
        # image   = self.image_plot.image.copy()
        # overlay = image.copy()
        # cv2.circle(overlay, roi_point, 10, (255, 0, 0), -1)
        # cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # update image plot
        if new_roi_selected:
            self.image_plot.select_roi(self.controller.selected_rois[-1])

        # self.image_plot.show_eraser(roi_point)
        # self.update_image_plot(image, background_mask=self.background_mask, mask=self.mask, selected_mask_num=self.controller.selected_mask_num, roi_spatial_footprints=self.controller.controller.roi_spatial_footprints[self.controller.z], video_dimensions=self.controller.video.shape, removed_rois=self.controller.controller.removed_rois[self.controller.z], selected_rois=self.controller.selected_rois, use_existing_roi_overlay=True)

    def end_erase_rois(self):
        # update image plot
        # self.update_image_plot(self.image)
        self.controller.erase_rois()

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

    def select_roi(self, roi_point, shift_held=False):
        self.controller.select_roi(roi_point, shift_held=shift_held)

        if (not shift_held) or len(self.controller.selected_rois) == 0:
            self.image_plot.deselect_rois()
            self.image_plot.erase_rois(self.controller.controller.removed_rois[self.controller.z])

        if len(self.controller.selected_rois) > 0:
            self.image_plot.select_roi(self.controller.selected_rois[-1])

    def select_mask(self, roi_point):
        self.controller.select_mask(roi_point)

    def draw_tentative_roi(self, start_point, end_point):
        if end_point != start_point:
            # make a copy of the current image
            image = self.image_plot.image.copy()

            print(np.amax(image), image.shape)

            # create a mask that shows the boundary of an elliptical ROI fitting between the start & end points
            mask  = np.zeros((self.image.shape[0], self.image.shape[1])).astype(np.uint8)
            center_point = (int(round((end_point[0] + start_point[0])/2)), int(round((end_point[1] + start_point[1])/2)))
            axis_1 = np.abs(center_point[0] - end_point[0])
            axis_2 = np.abs(center_point[1] - end_point[1])
            cv2.ellipse(mask, center_point, (axis_1, axis_2), 0, 0, 360, 1, -1)
            b = erosion(mask, disk(1))
            mask = mask - b

            # draw the boundary on the image
            # image[mask] = 255
            image[mask == 1] = np.array([255, 255, 0]).astype(np.uint8)

            # update image plot
            self.update_image_plot_simple(image)

    def draw_roi(self, start_point, end_point):
        self.controller.create_roi(start_point, end_point)

    def draw_roi_magic_wand(self, point):
        self.controller.create_roi_magic_wand(point)

    def show_roi_nums(self):
        self.image_plot.show_roi_nums(self.controller.selected_rois)

    # def shift_rois(self, previous_point, current_point):
    #     self.controller.shift_rois(previous_point, current_point)

    def mouse_pressed(self, point, shift_held=False):
        if self.controller.mode == "roi_finding" and self.drawing_mask:
            self.add_mask_point(point)
        elif self.controller.mode == "roi_filtering" and not self.drawing_rois:
            if not shift_held:
                # self.controller.select_roi(None, shift_held=False)
                self.select_roi(None, shift_held=False)

            # store this point
            self.click_end_point = point

    def mouse_moved(self, start_point, end_point, clicked=False):
        if self.drawing_mask:
            self.add_tentative_mask_point(end_point)
        elif self.controller.mode == "roi_filtering" and clicked:
            if self.erasing_rois: 
                self.select_roi_at_point(end_point)
            else:
                self.controller.erase_rois()
        elif self.drawing_rois and clicked:
            self.draw_tentative_roi(start_point, end_point)
        # elif self.controller.mode == "roi_filtering" and clicked:
        #     self.shift_rois(self.click_end_point, end_point)

            # store this point
            # self.click_end_point = end_point

    def mouse_released(self, start_point, end_point, mouse_moved=False, shift_held=False):
        # if self.controller.mode == "roi_finding":
        #     if not self.drawing_mask and not mouse_moved:
        #         self.select_mask(end_point)
        if self.controller.mode == "roi_filtering":
            if self.erasing_rois:
                self.end_erase_rois()
            elif self.drawing_rois:
                if end_point != start_point:
                    self.draw_roi(start_point, end_point)
                else:
                    self.draw_roi_magic_wand(start_point)
            elif (not mouse_moved):
                self.select_roi(end_point, shift_held=shift_held)

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
