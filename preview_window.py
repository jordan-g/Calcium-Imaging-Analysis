import os
import numpy as np
import cv2
import pyqtgraph as pg
from matplotlib import cm

import utilities

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

roi_colors = [ np.random.permutation((np.random.uniform(120, 200), np.random.uniform(50, 200), np.random.uniform(50, 200))) for i in range(10000) ]

class PreviewWindow(QMainWindow):
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
        self.main_widget.setMinimumSize(QSize(800, 900))

        # create main layout
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.image_widget = QWidget(self)
        self.image_layout = QHBoxLayout(self.image_widget)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        
        bg_color = (50, 50, 50)
        self.main_widget.setStyleSheet("background-color: rgba({}, {}, {}, 1);".format(bg_color[0], bg_color[1], bg_color[2]))
        pg.setConfigOption('background', bg_color)
        if bg_color[0] < 100:
            pg.setConfigOption('foreground', (150, 150, 150))
        else:
            pg.setConfigOption('foreground', (50, 50, 50))
        self.image_plot = pg.GraphicsLayoutWidget()
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
        colormap = cm.get_cmap("inferno")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.heatmap_plot.setLookupTable(lut)
        self.heatmap_plot_2.setLookupTable(lut)
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
        self.kept_rois_overlay      = None
        self.discarded_rois_overlay = None
        self.kept_rois_image        = None
        self.discarded_rois_image   = None
        self.roi_overlays           = None
        self.roi_contours           = []
        self.text_items             = []
        self.outline_items          = []
        self.image                  = None # image to show
        self.frames                 = None # frames to play
        self.frame_num              = 0    # current frame #
        self.n_frames               = 1    # total number of frames
        self.video_name             = ""   # name of the currently showing video

        self.image_plot.hide()
        self.timer.stop()
        self.setWindowTitle("Preview")

    def plot_tail_angles(self, tail_angles, tail_data_fps, calcium_data_fps):
        self.viewbox6.clear()

        if tail_angles is not None:
            one_frame    = int(np.floor((1.0/calcium_data_fps)*tail_data_fps))
            total_frames = int(np.floor(one_frame*self.controller.video.shape[0]))

            x = np.linspace(0, self.controller.video.shape[0], total_frames)
            self.viewbox6.plot(x, tail_angles[:total_frames, -1], pen=pg.mkPen((255, 255, 0), width=2))

    def plot_traces(self, roi_temporal_footprints, selected_rois=[]):
        self.viewbox3.clear()
        if len(selected_rois) > 0:
            max_value = np.amax(roi_temporal_footprints)

            for i in range(len(selected_rois)):
                roi = selected_rois[i]

                self.viewbox3.plot(roi_temporal_footprints[roi]/max_value, pen=pg.mkPen((roi_colors[roi][0], roi_colors[roi][1], roi_colors[roi][2]), width=2))

    def clear_text_and_outline_items(self):
        # remove all text and outline items from left and right plots
        for text_item in self.text_items:
            self.viewbox1.removeItem(text_item)
            self.viewbox2.removeItem(text_item)
            self.text_items = []
        for outline_item in self.outline_items:
            self.viewbox1.removeItem(outline_item)
            self.viewbox2.removeItem(outline_item)
            self.outline_items = []

    def plot_clicked(self, event):
        if self.controller.mode not in ("loading", "motion_correcting"):
            # get x-y coordinates of where the user clicked
            items = self.image_plot.scene().items(event.scenePos())
            if self.left_plot in items:
                pos = self.viewbox1.mapSceneToView(event.scenePos())
            elif self.right_plot in items:
                pos = self.viewbox2.mapSceneToView(event.scenePos())
            else:
                return
            x = pos.x()
            y = pos.y()

            # check whether the user is holding Ctrl
            ctrl_held = event.modifiers() == Qt.ControlModifier

            # remove all text and outline items from left and right plots
            self.clear_text_and_outline_items()

            if event.button() == 1:
                # left click means selecting ROIs

                self.controller.select_roi((int(y), int(x)), ctrl_held=ctrl_held)

                # don't allow selecting removed & kept ROIs at the same time
                removed_count = 0
                for i in self.controller.selected_rois:
                    if i in self.controller.removed_rois():
                        removed_count += 1
                if removed_count !=0 and removed_count != len(self.controller.selected_rois):
                    self.controller.selected_rois = [self.controller.selected_rois[-1]]

                if len(self.controller.selected_rois) > 0:
                    if self.controller.selected_rois[-1] in self.controller.removed_rois() and self.left_plot in items:
                        self.controller.selected_rois = []
                    elif self.controller.selected_rois[-1] not in self.controller.removed_rois() and self.right_plot in items:
                        self.controller.selected_rois = []

                if len(self.controller.selected_rois) > 0:
                    roi_to_select = self.controller.selected_rois[0]

                    if self.left_plot in items:
                        image = self.kept_rois_image.copy()
                        contours = []
                        for i in self.controller.selected_rois:
                            contours += self.roi_contours[i]
                            x = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 0]) for j in range(len(self.roi_contours[i])) ])
                            y = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 1]) for j in range(len(self.roi_contours[i])) ])
                            text_item = pg.TextItem("{}".format(i), color=roi_colors[i])
                            text_item.setPos(QPoint(int(y), int(x)))
                            self.text_items.append(text_item)
                            self.viewbox1.addItem(text_item)
                            for j in range(len(self.roi_contours[i])):
                                outline_item = pg.PlotDataItem(np.concatenate([self.roi_contours[i][j][:, 0, 1], np.array([self.roi_contours[i][j][0, 0, 1]])]), np.concatenate([self.roi_contours[i][j][:, 0, 0], np.array([self.roi_contours[i][j][0, 0, 0]])]), pen=pg.mkPen((roi_colors[i][0], roi_colors[i][1], roi_colors[i][2]), width=3))
                                self.outline_items.append(outline_item)
                                self.viewbox1.addItem(outline_item)

                        self.left_plot.setImage(image)
                        self.right_plot.setImage(self.discarded_rois_image)
                    else:
                        image = self.discarded_rois_image.copy()
                        contours = []
                        for i in self.controller.selected_rois:
                            contours += self.roi_contours[i]
                            # print([ self.roi_contours[i][j].shape for j in range(len(self.roi_contours[i])) ])
                            x = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 0]) for j in range(len(self.roi_contours[i])) ])
                            y = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 1]) for j in range(len(self.roi_contours[i])) ])
                            text_item = pg.TextItem("{}".format(i), color=roi_colors[i])
                            text_item.setPos(QPoint(int(y), int(x)))
                            self.text_items.append(text_item)
                            self.viewbox2.addItem(text_item)
                            for j in range(len(self.roi_contours[i])):
                                outline_item = pg.PlotDataItem(np.concatenate([self.roi_contours[i][j][:, 0, 1], np.array([self.roi_contours[i][j][0, 0, 1]])]), np.concatenate([self.roi_contours[i][j][:, 0, 0], np.array([self.roi_contours[i][j][0, 0, 0]])]), pen=pg.mkPen((roi_colors[i][0], roi_colors[i][1], roi_colors[i][2]), width=3))
                                self.outline_items.append(outline_item)
                                self.viewbox2.addItem(outline_item)

                        self.right_plot.setImage(image)
                        self.left_plot.setImage(self.kept_rois_image)
                else:
                    self.left_plot.setImage(self.kept_rois_image)
                    self.right_plot.setImage(self.discarded_rois_image)
            elif event.button() == 2:
                if self.left_plot in items:
                    selected_roi = utilities.get_roi_containing_point(self.controller.roi_spatial_footprints(), (int(y), int(x)), self.controller.selected_video_mean_image().shape)

                    if selected_roi is not None:
                        if selected_roi not in self.controller.selected_rois:
                            self.controller.selected_rois.append(selected_roi)

                        print("ROIs selected: {}.".format(self.controller.selected_rois))

                        self.controller.discard_selected_rois()
                else:
                    selected_roi = utilities.get_roi_containing_point(self.controller.roi_spatial_footprints(), (int(y), int(x)), self.controller.selected_video_mean_image().shape)

                    if selected_roi is not None:
                        if selected_roi not in self.controller.selected_rois:
                            self.controller.selected_rois.append(selected_roi)

                        print("ROIs selected: {}.".format(self.controller.selected_rois))

                        self.controller.keep_selected_rois()

            self.controller.update_trace_plot()

    def plot_image(self, image, video_max=255, show_rois=False, update_overlay=True, recreate_roi_images=True):
        if update_overlay:
            self.kept_rois_overlay      = None
            self.discarded_rois_overlay = None
            self.kept_rois_image        = None
            self.discarded_rois_image   = None
            self.roi_overlays           = None
            self.roi_contours           = []

        if self.image is None:
            self.image_plot.show()

        # update image
        self.image = image

        # update image plot
        if recreate_roi_images:
            self.update_left_image_plot(self.image, roi_spatial_footprints=self.controller.roi_spatial_footprints(), video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois, use_existing_roi_overlay=(not update_overlay))
            self.update_right_image_plot(self.image, roi_spatial_footprints=self.controller.roi_spatial_footprints(), video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois, use_existing_roi_overlay=(not update_overlay))
        else:
            if not show_rois:
                image = 255.0*self.image/self.controller.video_max
                image[image > 255] = 255

                if len(image.shape) < 3:
                    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    image = image.astype(np.uint8)

                self.left_plot.setImage(image, autoLevels=False)
                self.right_plot.setImage(image, autoLevels=False)
            else:
                self.left_plot.setImage(self.kept_rois_image, autoLevels=False)
                self.right_plot.setImage(self.discarded_rois_image, autoLevels=False)

    def play_video(self, video, video_path, fps):
        self.video_name = os.path.basename(video_path)
        
        if self.frames is None:
            self.image_plot.show()

        # set frame number to 0
        self.frame_num = 0

        # normalize the frames (to be between 0 and 255)
        self.frames = video

        # get the number of frames
        self.n_frames = self.frames.shape[0]

        # start the timer to update the frames
        self.timer.start(int(1000.0/fps))

        self.viewbox4.setXRange(0, self.controller.video.shape[0])

    def set_fps(self, fps):
        # restart the timer with the new fps
        self.timer.stop()
        self.timer.start(int(1000.0/fps))

    def show_frame(self, frame):
        self.update_left_image_plot(frame, show_rois=self.controller.show_rois)

    def update_frame(self):
        if self.frames is not None:
            # convert the current frame to RGB
            frame = cv2.cvtColor(self.frames[self.frame_num], cv2.COLOR_GRAY2RGB)

            spatial_footprints = self.controller.roi_spatial_footprints()
            removed_rois       = self.controller.removed_rois()

            self.update_left_image_plot(frame, roi_spatial_footprints=spatial_footprints, video_dimensions=self.controller.video.shape, removed_rois=removed_rois, selected_rois=self.controller.selected_rois, show_rois=self.controller.show_rois, use_existing_roi_overlay=True)

            # increment frame number (keeping it between 0 and n_frames)
            self.frame_num += 1
            self.frame_num = self.frame_num % self.n_frames

            # update window title
            self.setWindowTitle("{}. Z={}. Frame {}/{}.".format(self.video_name, self.controller.z, self.frame_num + 1, self.n_frames))

    def update_left_image_plot(self, image, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False, use_existing_roi_overlay=False):
        self.kept_rois_image = self.create_kept_rois_image(image, self.controller.video_max, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, show_rois=show_rois, use_existing_roi_overlay=use_existing_roi_overlay)
            
        self.left_plot.setImage(self.kept_rois_image, autoLevels=False)

    def create_kept_rois_image(self, image, video_max, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False, use_existing_roi_overlay=False):
        image = 255.0*image/video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if show_rois:
            if roi_spatial_footprints is not None:
                roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((video_dimensions[2], video_dimensions[3], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

            if video_dimensions is not None:
                if len(self.roi_contours) == 0 and roi_spatial_footprints is not None:
                    print("Computing new contours.....")
                    self.roi_contours  = [ None for i in range(roi_spatial_footprints.shape[-1]) ]
                    self.flat_contours = []
                    
                    self.roi_overlay  = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)
                    self.roi_overlays = np.zeros((roi_spatial_footprints.shape[-1], image.shape[0], image.shape[1], 4)).astype(np.uint8)

                    for i in range(roi_spatial_footprints.shape[-1]):
                        maximum = np.amax(roi_spatial_footprints[:, :, i])
                        mask = (roi_spatial_footprints[:, :, i] > 0.1*maximum).copy()
                        
                        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                        color = roi_colors[i]

                        overlay = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)
                        overlay[mask, :-1] = color
                        overlay[mask, -1] = 255.0*roi_spatial_footprints[mask, i]/maximum
                        self.roi_overlays[i] = overlay

                        self.roi_contours[i] = contours
                        self.flat_contours += contours

                if roi_spatial_footprints is not None:
                    kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]

                    denominator = np.count_nonzero(self.roi_overlays[kept_rois], axis=0)
                    denominator[denominator == 0] = 1

                    self.roi_overlay = (np.sum(self.roi_overlays[kept_rois], axis=0)/denominator).astype(np.uint8)

                if len(self.roi_overlay.shape) > 0 and roi_spatial_footprints is not None:
                    image = utilities.blend_transparent(image, self.roi_overlay)

                    kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]

                    heatmap = self.controller.roi_temporal_footprints()[kept_rois].T
                    
                    video_lengths = self.controller.selected_group_video_lengths()

                    index = self.controller.selected_group_video_paths().index(self.controller.selected_video_path())

                    if index == 0:
                        heatmap = heatmap[:video_lengths[0]]
                    else:
                        heatmap = heatmap[np.sum(video_lengths[:index]):np.sum(video_lengths[:index+1])]

                    if heatmap.shape[1] > 0:
                        if self.controller.show_zscore:
                            heatmap = (heatmap - np.mean(heatmap, axis=0)[np.newaxis, :])/np.std(heatmap, axis=0)[np.newaxis, :]

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

    def create_discarded_rois_image(self, image, video_max, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False, use_existing_roi_overlay=False):
        image = 255.0*image/video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if show_rois:
            if roi_spatial_footprints is not None:
                roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((video_dimensions[2], video_dimensions[3], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

            if video_dimensions is not None:
                if roi_spatial_footprints is not None:

                    denominator = np.count_nonzero(self.roi_overlays[removed_rois], axis=0)
                    denominator[denominator == 0] = 1

                    self.discarded_rois_overlay = (np.sum(self.roi_overlays[removed_rois], axis=0)/denominator).astype(np.uint8)

                if len(self.discarded_rois_overlay.shape) > 0 and roi_spatial_footprints is not None:
                    image = utilities.blend_transparent(image, self.discarded_rois_overlay)

        return image

    def plot_mean_image(self, image, video_max):
        image = 255.0*image/video_max
        image[image > 255] = 255
        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        self.right_plot.setImage(image, autoLevels=False)

    def update_right_image_plot(self, image, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False, use_existing_roi_overlay=False):
        self.discarded_rois_image = self.create_discarded_rois_image(image, self.controller.video_max, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, show_rois=show_rois, use_existing_roi_overlay=use_existing_roi_overlay)

        self.right_plot.setImage(self.discarded_rois_image, autoLevels=False)

    def reset_zoom(self):
        self.viewbox1.autoRange()
        self.viewbox2.autoRange()

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()
