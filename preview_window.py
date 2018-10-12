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
import pyqtgraph as pg
from matplotlib import cm

# color table to use for showing images
gray_color_table = [qRgb(i, i, i) for i in range(256)]

colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), ]

all_colors = [ (np.random.uniform(50, 200), np.random.uniform(50, 200), np.random.uniform(50, 200)) for i in range(10000) ]

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
        # self.main_widget.setStyleSheet("background-color: #000;")
        self.main_widget.setMinimumSize(QSize(800, 900))

        # create main layout
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # create label that shows frames
        self.image_widget = QWidget(self)
        self.image_layout = QHBoxLayout(self.image_widget)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        bg_color = (self.palette().color(self.backgroundRole()).red(), self.palette().color(self.backgroundRole()).green(), self.palette().color(self.backgroundRole()).blue())
        pg.setConfigOption('background', bg_color)
        if bg_color[0] < 100:
            pg.setConfigOption('foreground', (200, 200, 200))
        else:
            pg.setConfigOption('foreground', (50, 50, 50))
        self.image_plot = pg.GraphicsLayoutWidget()
        # self.image_plot.setBackground((10, 10, 10))
        self.viewbox1 = self.image_plot.addViewBox(lockAspect=True,name='left_plot',border=None, row=0,col=0, invertY=True)
        self.viewbox2 = self.image_plot.addViewBox(lockAspect=True,name='right_plot',border=None, row=0,col=1, invertY=True)
        self.viewbox1.setLimits(minXRange=10, minYRange=10, maxXRange=2000, maxYRange=2000)
        self.viewbox3 = self.image_plot.addPlot(name='trace_plot', row=1,col=0,colspan=2)
        self.viewbox3.setLabel('left', "Fluorescence")
        self.viewbox3.setLabel('bottom', "Frame #")
        self.viewbox3.showButtons()
        self.viewbox3.setMouseEnabled(x=True,y=False)
        self.viewbox3.setYRange(0, 1)
        self.viewbox4 = self.image_plot.addPlot(name='heatmap_plot', row=2,col=0, colspan=2)
        self.viewbox4.setLabel('left', "ROI #")
        self.viewbox4.setMouseEnabled(x=True,y=False)
        self.viewbox4.setLabel('bottom', "Frame #")
        self.viewbox5 = self.image_plot.addPlot(name='heatmap2_plot', row=3,col=0, colspan=2)
        self.viewbox5.setLabel('left', "ROI #")
        self.viewbox5.setMouseEnabled(x=True,y=False)
        self.viewbox5.setLabel('bottom', "Frame #")
        self.viewbox6 = self.image_plot.addPlot(name='tail_angle_plot', row=4,col=0,colspan=2)
        self.viewbox6.setLabel('left', "Tail Angle (º)")
        self.viewbox6.setLabel('bottom', "Frame #")
        self.viewbox6.setMouseEnabled(x=True,y=False)
        self.heatmap_plot = pg.ImageItem()
        self.heatmap_plot_2 = pg.ImageItem()
        self.viewbox4.addItem(self.heatmap_plot)
        self.viewbox5.addItem(self.heatmap_plot_2)
        self.image_plot.ci.layout.setRowStretchFactor(0,2)
        self.left_plot = pg.ImageItem()
        self.right_plot = pg.ImageItem()
        self.viewbox1.addItem(self.left_plot)
        self.viewbox2.addItem(self.right_plot)
        self.viewbox1.setBackgroundColor(bg_color)
        self.viewbox2.setBackgroundColor(bg_color)
        self.viewbox2.setXLink('left_plot')
        self.viewbox2.setYLink('left_plot')
        self.viewbox4.setXLink('trace_plot')
        self.viewbox5.setXLink('trace_plot')
        self.viewbox6.setXLink('trace_plot')
        colormap = cm.get_cmap("inferno")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.heatmap_plot.setLookupTable(lut)
        self.heatmap_plot_2.setLookupTable(lut)
        # self.image_plot = PreviewQLabel(self)
        self.image_plot.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.image_plot.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_layout.addWidget(self.image_plot)
        self.main_layout.addWidget(self.image_widget, 0, 0)
        self.image_plot.scene().sigMouseClicked.connect(self.plot_clicked)
        self.viewbox1.setMenuEnabled(False)
        self.viewbox2.setMenuEnabled(False)
        self.image_plot.ci.layout.setRowStretchFactor(0, 8)
        self.image_plot.ci.layout.setRowStretchFactor(1, 2)
        self.image_plot.ci.layout.setRowStretchFactor(2, 2)
        self.image_plot.ci.layout.setRowStretchFactor(3, 2)
        self.image_plot.ci.layout.setRowStretchFactor(4, 2)

        self.drawing_mask      = False # whether the user is drawing a mask
        self.erasing_rois      = False # whether the user is erasing ROIs
        self.drawing_rois      = False # whether the user is drawing ROIs
        self.mask_points       = []    # list holding the points of the mask being drawn
        self.mask              = None
        self.background_mask   = None
        self.roi_overlay       = None
        self.roi_overlay_right = None
        self.overlays          = []
        self.all_contours      = None
        self.text_items        = []
        self.outline_items     = []

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

    def plot_tail_angles(self, tail_angles):
        self.viewbox6.clear()

        # print(tail_angles.shape)

        fps = 349
        fps_calcium = 3
        one_frame = int(np.floor((1.0/fps_calcium)*fps))
        total_frames = int(np.floor(one_frame*self.controller.video.shape[0]))

        x = np.linspace(0, self.controller.video.shape[0], total_frames)
        # for i in range(tail_angles.shape[1]):
        # self.viewbox6.plot(x, tail_angles[one_frame:total_frames+one_frame, -1], pen=pg.mkPen((255, 255, 0), width=2))
        self.viewbox6.plot(x, tail_angles[:total_frames, -1], pen=pg.mkPen((255, 255, 0), width=2))

    def plot_traces(self, roi_temporal_footprints, removed_rois=[], selected_rois=[], mean_traces=None):
        self.viewbox3.clear()
        if len(selected_rois) > 0:
            max_value = np.amax(roi_temporal_footprints)

            for i in range(len(selected_rois)):
                roi = selected_rois[i]
                # print(roi, roi_temporal_footprints.shape)
                y_offset = 0

                self.viewbox3.plot(roi_temporal_footprints[roi]/max_value + y_offset, pen=pg.mkPen((all_colors[roi][0], all_colors[roi][1], all_colors[roi][2]), width=2))

    def plot_clicked(self, event):
        if self.controller.mode not in ("loading", "motion_correcting"):
            items = self.image_plot.scene().items(event.scenePos())

            if self.left_plot in items:
                pos = self.viewbox1.mapSceneToView(event.scenePos())
            elif self.right_plot in items:
                pos = self.viewbox2.mapSceneToView(event.scenePos())
            else:
                return
            x = pos.x()
            y = pos.y()

            shift_held = event.modifiers() == Qt.ControlModifier

            for text_item in self.text_items:
                self.viewbox1.removeItem(text_item)
                self.viewbox2.removeItem(text_item)
                self.text_items = []

            for outline_item in self.outline_items:
                self.viewbox1.removeItem(outline_item)
                self.viewbox2.removeItem(outline_item)
                self.outline_items = []

            if event.button() == 1:
                self.controller.select_roi((int(y), int(x)), shift_held=shift_held)

                # don't allow selecting removed & kept ROIs at the same time
                removed_count = 0
                for i in self.controller.selected_rois:
                    if i in self.controller.controller.removed_rois[self.controller.z]:
                        removed_count += 1
                if removed_count !=0 and removed_count != len(self.controller.selected_rois):
                    self.controller.selected_rois = [self.controller.selected_rois[-1]]

                if len(self.controller.selected_rois) > 0:
                    if self.controller.selected_rois[-1] in self.controller.controller.removed_rois[self.controller.z] and self.left_plot in items:
                        self.controller.selected_rois = []
                    elif self.controller.selected_rois[-1] not in self.controller.controller.removed_rois[self.controller.z] and self.right_plot in items:
                        self.controller.selected_rois = []

                if len(self.controller.selected_rois) > 0:
                    roi_to_select = self.controller.selected_rois[0]

                    if self.left_plot in items:
                        image = self.final_image.copy()
                        contours = []
                        for i in self.controller.selected_rois:
                            contours += self.all_contours[i]
                            # print([ self.all_contours[i][j].shape for j in range(len(self.all_contours[i])) ])
                            x = np.amax([ np.amax(self.all_contours[i][j][:, 0, 0]) for j in range(len(self.all_contours[i])) ])
                            y = np.amax([ np.amax(self.all_contours[i][j][:, 0, 1]) for j in range(len(self.all_contours[i])) ])
                            text_item = pg.TextItem("{}".format(i), color=all_colors[i])
                            text_item.setPos(QPoint(int(y), int(x)))
                            self.text_items.append(text_item)
                            self.viewbox1.addItem(text_item)
                            for j in range(len(self.all_contours[i])):
                                outline_item = pg.PlotDataItem(np.concatenate([self.all_contours[i][j][:, 0, 1], np.array([self.all_contours[i][j][0, 0, 1]])]), np.concatenate([self.all_contours[i][j][:, 0, 0], np.array([self.all_contours[i][j][0, 0, 0]])]), pen=pg.mkPen((all_colors[i][0], all_colors[i][1], all_colors[i][2]), width=3))
                                self.outline_items.append(outline_item)
                                self.viewbox1.addItem(outline_item)
                        # cv2.drawContours(image, contours, -1, (255, 255, 0), 1)

                        self.left_plot.setImage(image)
                        self.right_plot.setImage(self.final_right_image)
                    else:
                        image = self.final_right_image.copy()
                        contours = []
                        for i in self.controller.selected_rois:
                            contours += self.all_contours[i]
                            # print([ self.all_contours[i][j].shape for j in range(len(self.all_contours[i])) ])
                            x = np.amax([ np.amax(self.all_contours[i][j][:, 0, 0]) for j in range(len(self.all_contours[i])) ])
                            y = np.amax([ np.amax(self.all_contours[i][j][:, 0, 1]) for j in range(len(self.all_contours[i])) ])
                            text_item = pg.TextItem("{}".format(i), color=all_colors[i])
                            text_item.setPos(QPoint(int(y), int(x)))
                            self.text_items.append(text_item)
                            self.viewbox2.addItem(text_item)
                            for j in range(len(self.all_contours[i])):
                                outline_item = pg.PlotDataItem(np.concatenate([self.all_contours[i][j][:, 0, 1], np.array([self.all_contours[i][j][0, 0, 1]])]), np.concatenate([self.all_contours[i][j][:, 0, 0], np.array([self.all_contours[i][j][0, 0, 0]])]), pen=pg.mkPen((all_colors[i][0], all_colors[i][1], all_colors[i][2]), width=3))
                                self.outline_items.append(outline_item)
                                self.viewbox2.addItem(outline_item)
                        # cv2.drawContours(image, contours, -1, (255, 255, 0), 1)

                        self.right_plot.setImage(image)
                        self.left_plot.setImage(self.final_image)
                else:
                    self.left_plot.setImage(self.final_image)
                    self.right_plot.setImage(self.final_right_image)
            elif event.button() == 2:
                if self.left_plot in items:
                    manual_roi_selected, selected_roi = utilities.get_roi_containing_point(self.controller.controller.roi_spatial_footprints[self.controller.z], self.controller.controller.manual_roi_spatial_footprints[self.controller.z], (int(y), int(x)), self.controller.image.shape)

                    if selected_roi is not None:
                        if selected_roi not in self.controller.selected_rois:
                            self.controller.selected_rois.append(selected_roi)

                        print("ROIs selected: {}".format(self.controller.selected_rois))

                        self.controller.erase_selected_rois()

                        self.controller.selected_rois = []

                        self.controller.show_roi_image(show=self.controller.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=True)
                else:
                    manual_roi_selected, selected_roi = utilities.get_roi_containing_point(self.controller.controller.roi_spatial_footprints[self.controller.z], self.controller.controller.manual_roi_spatial_footprints[self.controller.z], (int(y), int(x)), self.controller.image.shape)

                    if selected_roi is not None:
                        if selected_roi not in self.controller.selected_rois:
                            self.controller.selected_rois.append(selected_roi)

                        print("ROIs selected: {}".format(self.controller.selected_rois))

                        self.controller.unerase_selected_rois()

                        self.controller.selected_rois = []

                        self.controller.show_roi_image(show=self.controller.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=True)

            # print(x, y)

            self.controller.update_trace_plot()

    def set_initial_state(self):
        # reset variables
        self.image             = None  # image to show
        self.final_image       = None
        self.final_right_image = None
        self.frames            = None  # frames to play
        self.frame_num         = 0     # current frame #
        self.n_frames          = 1     # total number of frames
        self.video_name        = ""    # name of the currently showing video
        self.mask_points       = []    # list holding the points of the mask being drawn
        self.mask              = None
        self.background_mask   = None
        self.roi_overlay       = None
        self.roi_overlay_right = None
        self.overlays          = []
        self.all_contours      = None
        self.text_items        = []
        self.outline_items     = []

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
        self.update_right_image_plot(self.image, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask, selected_mask_num=self.controller.selected_mask_num, roi_spatial_footprints=self.controller.controller.roi_spatial_footprints[self.controller.z], manual_roi_spatial_footprints=self.controller.controller.manual_roi_spatial_footprints[self.controller.z], video_dimensions=self.controller.video.shape, removed_rois=self.controller.controller.removed_rois[self.controller.z], selected_rois=self.controller.selected_rois, manual_roi_selected=self.controller.manual_roi_selected, use_existing_roi_overlay=(not update_overlay))

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
            # self.update_image_plot(frame)

            mask_points = []
            self.mask = None

            if self.controller.controller.roi_spatial_footprints is not None:
                spatial_footprints = self.controller.controller.roi_spatial_footprints[self.controller.z]
                removed_rois = self.controller.controller.removed_rois[self.controller.z]
            else:
                spatial_footprints = None
                removed_rois = None
            self.update_image_plot(frame, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask, selected_mask_num=self.controller.selected_mask_num, roi_spatial_footprints=spatial_footprints, manual_roi_spatial_footprints=None, video_dimensions=self.controller.video.shape, removed_rois=removed_rois, selected_rois=self.controller.selected_rois, manual_roi_selected=self.controller.manual_roi_selected, use_existing_roi_overlay=True)
            # self.update_right_image_plot(self.image, background_mask=self.background_mask, mask_points=mask_points, mask=self.mask, selected_mask_num=self.controller.selected_mask_num, roi_spatial_footprints=self.controller.controller.roi_spatial_footprints[self.controller.z], manual_roi_spatial_footprints=self.controller.controller.manual_roi_spatial_footprints[self.controller.z], video_dimensions=self.controller.video.shape, removed_rois=self.controller.controller.removed_rois[self.controller.z], selected_rois=self.controller.selected_rois, manual_roi_selected=self.controller.manual_roi_selected, use_existing_roi_overlay=(not update_overlay))

            # increment frame number (keeping it between 0 and n_frames)
            self.frame_num += 1
            self.frame_num = self.frame_num % self.n_frames

            # update window title
            self.setWindowTitle("{}. Z={}. Frame {}/{}.".format(self.video_name, self.controller.z, self.frame_num + 1, self.n_frames))

    def update_image_plot(self, image, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, use_existing_roi_overlay=False):
        # print(use_existing_roi_overlay)
        final_image = self.create_final_image(image, self.controller.controller.video_max, background_mask=background_mask, mask_points=mask_points, tentative_mask_point=tentative_mask_point, mask=mask, selected_mask_num=selected_mask_num, roi_spatial_footprints=roi_spatial_footprints, manual_roi_spatial_footprints=manual_roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, manual_roi_selected=manual_roi_selected, show_rois=self.controller.show_rois, use_existing_roi_overlay=use_existing_roi_overlay)

        self.left_plot.setImage(final_image)

        # if update_final_image:
        self.final_image = final_image.copy()

    def create_final_image(self, image, vmax, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, show_rois=False, update_stored_image=True, use_existing_roi_overlay=False):
        image = 255.0*image/vmax
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if show_rois:
            if roi_spatial_footprints is not None:
                roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((video_dimensions[1], video_dimensions[2], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

            if video_dimensions is not None:
                if self.all_contours is None and roi_spatial_footprints is not None:
                    print("Computing new contours.....")
                    self.all_contours = [ None for i in range(roi_spatial_footprints.shape[-1]) ]
                    self.flat_contours = []
                    
                    self.roi_overlay = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)
                    self.overlays    = np.zeros((roi_spatial_footprints.shape[-1], image.shape[0], image.shape[1], 4)).astype(np.uint8)

                    for i in range(roi_spatial_footprints.shape[-1]):
                        maximum = np.amax(roi_spatial_footprints[:, :, i])
                        mask = (roi_spatial_footprints[:, :, i] > 0.1*maximum).copy()
                        
                        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                        color = all_colors[i]

                        overlay = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)
                        overlay[mask, :-1] = color
                        overlay[mask, -1] = 255.0*roi_spatial_footprints[mask, i]/maximum
                        self.overlays[i] = overlay

                        self.all_contours[i] = contours
                        self.flat_contours += contours

                    # overlay = np.zeros(image.shape).astype(np.uint8)
                if (not use_existing_roi_overlay or self.roi_overlay is None) and roi_spatial_footprints is not None:

                    kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]

                    denominator = np.count_nonzero(self.overlays[kept_rois], axis=0)
                    denominator[denominator == 0] = 1

                    self.roi_overlay = (np.sum(self.overlays[kept_rois], axis=0)/denominator).astype(np.uint8)

                    # self.roi_overlay = self.roi_overlay.transpose((1, 0, 2))

                if len(self.roi_overlay.shape) > 0 and roi_spatial_footprints is not None:
                    image = utilities.blend_transparent(image, self.roi_overlay)

                    kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]

                    heatmap = self.controller.controller.roi_temporal_footprints[self.controller.z][kept_rois].T
                    # print(heatmap.shape, self.controller.controller.video_lengths)
                    if self.controller.selected_video == 0:
                        heatmap = heatmap[:self.controller.controller.video_lengths[0]]
                    else:
                        heatmap = heatmap[np.sum(self.controller.controller.video_lengths[:i]):np.sum(self.controller.controller.video_lengths[:i+1])]

                    # print(heatmap.shape)
                    if heatmap.shape[1] > 0:
                        heatmap = (heatmap - np.mean(heatmap, axis=0)[np.newaxis, :])/np.std(heatmap, axis=0)[np.newaxis, :]

                        # heatmap[heatmap > 2] = 2
                        # heatmap[heatmap < -2] = -2

                        heatmap2 = np.sort(heatmap, axis=1)

                        self.heatmap_plot.setImage(heatmap)
                        self.heatmap_plot_2.setImage(heatmap2)
                    else:
                        self.heatmap_plot.setImage(None)
                        self.heatmap_plot_2.setImage(None)
                else:
                    self.heatmap_plot.setImage(None)
                    self.heatmap_plot_2.setImage(None)

        return image

    def create_final_right_image(self, image, vmax, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, show_rois=False, update_stored_image=True, use_existing_roi_overlay=False):
        image = 255.0*image/vmax
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if show_rois:
            if roi_spatial_footprints is not None:
                roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((video_dimensions[1], video_dimensions[2], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

            if video_dimensions is not None:
                    # overlay = np.zeros(image.shape).astype(np.uint8)
                if (not use_existing_roi_overlay or self.roi_overlay_right is None) and roi_spatial_footprints is not None:

                    # self.roi_overlay_right = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)

                    denominator = np.count_nonzero(self.overlays[removed_rois], axis=0)
                    denominator[denominator == 0] = 1

                    self.roi_overlay_right = (np.sum(self.overlays[removed_rois], axis=0)/denominator).astype(np.uint8)

                if len(self.roi_overlay_right.shape) > 0 and roi_spatial_footprints is not None:
                    image = utilities.blend_transparent(image, self.roi_overlay_right)

        return image

    def plot_mean_image(self, image, vmax):
        image = 255.0*image/vmax
        image[image > 255] = 255
        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        self.right_plot.setImage(image)

    def update_right_image_plot(self, image, background_mask=None, mask_points=None, tentative_mask_point=None, mask=None, selected_mask_num=-1, roi_spatial_footprints=None, manual_roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, manual_roi_selected=False, use_existing_roi_overlay=False):
        # print(use_existing_roi_overlay)
        self.final_right_image = self.create_final_right_image(image, self.controller.controller.video_max, background_mask=background_mask, mask_points=mask_points, tentative_mask_point=tentative_mask_point, mask=mask, selected_mask_num=selected_mask_num, roi_spatial_footprints=roi_spatial_footprints, manual_roi_spatial_footprints=manual_roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, manual_roi_selected=manual_roi_selected, show_rois=self.controller.show_rois, use_existing_roi_overlay=use_existing_roi_overlay)

        self.right_plot.setImage(self.final_right_image)

    def reset_zoom(self):
        self.viewbox1.autoRange()
        self.viewbox2.autoRange()

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

            # print(np.amax(image), image.shape)

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

    def select_roi(self, roi_to_select):
        print("Selecting ROI {}.".format(roi_to_select))
        # pass
        image = self.final_image.copy()
        cv2.drawContours(image, self.all_contours[roi_to_select], -1, (255, 255, 0), 1)

        self.left_plot.setImage(image)

    def select_rois(self, rois_to_select):
        pass
        # for roi in rois_to_select:
        #     cv2.drawContours(self.final_image, self.all_contours[roi], -1, (255, 255, 0), 1)

        # self.left_plot.setImage(self.final_image)

    def show_eraser(self, point):
        image   = self.final_image.copy()
        overlay = image.copy()
        cv2.circle(overlay, point, 10, (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        self.left_plot.setImage(image)

    def deselect_rois(self):
        # pass
        # cv2.drawContours(self.final_image, self.flat_contours, -1, (255, 0, 0), 1)

        self.left_plot.setImage(self.final_image)

    def erase_roi(self, roi_to_erase):
        pass
        # cv2.drawContours(self.final_image, self.all_contours[roi_to_erase], -1, (128, 128, 128), 1)

        # self.left_plot.setImage(self.final_image)

    def unerase_roi(self, roi):
        pass
        # cv2.drawContours(self.final_image, self.all_contours[roi], -1, (255, 0, 0), 1)

        # self.left_plot.setImage(self.final_image)

    def erase_rois(self, rois_to_erase):
        pass
        # print(rois_to_erase)
        # # pdb.set_trace()
        # print(len(self.all_contours))
        # for roi_to_erase in rois_to_erase:
        #     cv2.drawContours(self.final_image, self.all_contours[roi_to_erase], -1, (128, 128, 128), 1)

        # self.left_plot.setImage(self.final_image)

    def unerase_rois(self, rois_to_unerase):
        pass
        # for roi in rois_to_unerase:
        #     cv2.drawContours(self.image, self.all_contours[roi], -1, (255, 0, 0), 1)

        # self.left_plot.setImage(self.final_final_image)

    def show_roi_nums(self, roi_nums):
        pass
        # image = self.final_image.copy()

        # for roi in roi_nums:
        #     contour = self.all_contours[roi][0]
        #     M = cv2.moments(contour)
        #     if M["m00"] > 0:
        #         center_x = int(M["m10"] / M["m00"])
        #         center_y = int(M["m01"] / M["m00"])

        #         centroid = (center_x + 10, center_y + 10)

        #         font = cv2.FONT_HERSHEY_PLAIN
        #         cv2.putText(image, "{}".format(roi), centroid, font, 1, (255, 255, 0), 1, cv2.LINE_AA)

        # self.left_plot.setImage(image)

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
