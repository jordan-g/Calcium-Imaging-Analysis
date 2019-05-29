import os
import numpy as np
import cv2
import pyqtgraph as pg
from matplotlib import cm
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

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

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

n_colors = 20
cmap = get_cmap(n_colors)

colormaps  = ["inferno", "plasma", "viridis", "magma", "Reds", "Greens", "Blues", "Greys", "gray", "hot"]

ROUNDED_STYLESHEET   = "QLineEdit { background-color: rgba(255, 255, 255, 0.3); border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.5); padding: 2px; color: white; };"
STATUSBAR_STYLESHEET = "background-color: rgba(50, 50, 50, 1); border-top: 1px solid rgba(0, 0, 0, 1); font-size: 12px; font-style: italic; color: white;"
TOP_LABEL_STYLESHEET = "QLabel{color: white; font-weight: bold; font-size: 14px;}"

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
        self.main_widget.setMinimumSize(QSize(800, 700))
        self.resize(800, 1000)

        # create main layout
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # set background and foreground colors
        bg_color = (20, 20, 20)
        fg_color = (150, 150, 150)
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor(bg_color[0], bg_color[1], bg_color[2], 255))
        self.main_widget.setAutoFillBackground(True)
        self.main_widget.setPalette(palette)
        pg.setConfigOption('background', bg_color)
        pg.setConfigOption('foreground', fg_color)

        # create top widget
        self.top_widget = QWidget()
        self.top_widget.setFixedHeight(35)
        self.top_layout = QHBoxLayout(self.top_widget)
        self.top_layout.setContentsMargins(20, 10, 20, 0)
        self.main_layout.addWidget(self.top_widget)

        # create left label
        self.left_label = QLabel("Video ▼")
        self.left_label.setStyleSheet(TOP_LABEL_STYLESHEET)
        self.top_layout.addWidget(self.left_label)

        self.top_layout.addStretch()

        # create right label
        self.right_label = QLabel("▼ Mean Image")
        self.right_label.setStyleSheet(TOP_LABEL_STYLESHEET)
        self.top_layout.addWidget(self.right_label)

        # create PyQTGraph widget
        self.pg_widget = pg.GraphicsLayoutWidget()
        self.pg_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.pg_widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.main_layout.addWidget(self.pg_widget)

        # create left and right image viewboxes
        self.left_image_viewbox  = self.pg_widget.addViewBox(lockAspect=True, name='left_image', border=None, row=0,col=0, invertY=True)
        self.right_image_viewbox = self.pg_widget.addViewBox(lockAspect=True, name='right_image', border=None, row=0,col=1, invertY=True)
        self.left_image_viewbox.setLimits(minXRange=10, minYRange=10, maxXRange=2000, maxYRange=2000)
        self.right_image_viewbox.setLimits(minXRange=10, minYRange=10, maxXRange=2000, maxYRange=2000)
        self.left_image_viewbox.setBackgroundColor(bg_color)
        self.right_image_viewbox.setBackgroundColor(bg_color)
        self.right_image_viewbox.setXLink('left_image')
        self.right_image_viewbox.setYLink('left_image')
        self.left_image_viewbox.setMenuEnabled(False)
        self.right_image_viewbox.setMenuEnabled(False)

        # create left and right image and overlay items
        self.left_image          = pg.ImageItem()
        self.left_image_overlay  = pg.ImageItem()
        self.right_image         = pg.ImageItem()
        self.right_image_overlay = pg.ImageItem()
        self.left_image_viewbox.addItem(self.left_image)
        self.left_image_viewbox.addItem(self.left_image_overlay)
        self.right_image_viewbox.addItem(self.right_image)
        self.right_image_viewbox.addItem(self.right_image_overlay)

        # create selected ROI trace viewbox
        self.roi_trace_viewbox = self.pg_widget.addPlot(name='roi_trace', row=1, col=0, colspan=2)
        self.roi_trace_viewbox.setLabel('bottom', "Frame #")
        self.roi_trace_viewbox.showButtons()
        self.roi_trace_viewbox.setMouseEnabled(x=True,y=False)
        if self.controller.show_zscore:
            self.roi_trace_viewbox.setYRange(-2, 3)
            self.roi_trace_viewbox.setLabel('left', "Z-Score")
        else:
            self.roi_trace_viewbox.setYRange(0, 1)
            self.roi_trace_viewbox.setLabel('left', "Fluorescence")

        # create kept ROI traces viewbox
        self.kept_traces_viewbox = self.pg_widget.addPlot(name='heatmap_plot', row=2, col=0, colspan=2)
        self.kept_traces_viewbox.setLabel('bottom', "Frame #")
        self.kept_traces_viewbox.setLabel('left', "ROI #")
        self.kept_traces_viewbox.setMouseEnabled(x=True,y=False)
        self.kept_traces_viewbox.setXLink('roi_trace')

        # create kept ROI traces item
        self.kept_traces_image = pg.ImageItem()
        self.kept_traces_viewbox.addItem(self.kept_traces_image)

        # set kept ROI traces colormap
        colormap = cm.get_cmap("inferno")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.kept_traces_image.setLookupTable(lut)

        # create tail angle viewbox
        self.tail_angle_viewbox = self.pg_widget.addPlot(name='tail_angle_plot', row=3, col=0, colspan=2)
        self.tail_angle_viewbox.setLabel('bottom', "Frame #")
        self.tail_angle_viewbox.setLabel('left', "Tail Angle (º)")
        self.tail_angle_viewbox.setMouseEnabled(x=True,y=False)
        self.tail_angle_viewbox.setXLink('roi_trace')

        # register a callback function for when the PyQTGraph widget is clicked
        self.pg_widget.scene().sigMouseClicked.connect(self.plot_clicked)

        # set stretch factors for each row
        self.pg_widget.ci.layout.setRowStretchFactor(0, 32)
        self.pg_widget.ci.layout.setRowStretchFactor(1, 8)
        self.pg_widget.ci.layout.setRowStretchFactor(2, 8)
        self.pg_widget.ci.layout.setRowStretchFactor(3, 8)

        # create current frame line items
        self.current_frame_line_1 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(255, 255, 0, 100), width=5))
        self.current_frame_line_2 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(255, 255, 0, 100), width=5))
        self.current_frame_line_3 = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(255, 255, 0, 100), width=5))
        self.roi_trace_viewbox.addItem(self.current_frame_line_1)
        self.kept_traces_viewbox.addItem(self.current_frame_line_2)
        self.tail_angle_viewbox.addItem(self.current_frame_line_3)

        # create bottom widget
        self.bottom_widget = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_widget)
        self.main_layout.addWidget(self.bottom_widget)

        # create checkbox to show/hide ROIs
        self.show_rois_checkbox = HoverCheckBox("Show ROIs", self, self.statusBar())
        self.show_rois_checkbox.setHoverMessage("Toggle showing ROIs.")
        self.show_rois_checkbox.setObjectName("Show ROIs")
        self.show_rois_checkbox.setStyleSheet("color: rgba(150, 150, 150, 1);")
        self.show_rois_checkbox.setChecked(False)
        self.show_rois_checkbox.setEnabled(False)
        self.show_rois_checkbox.clicked.connect(self.toggle_show_rois)
        self.bottom_layout.addWidget(self.show_rois_checkbox)

        # create checkbox to toggle video playback
        self.play_video_checkbox = HoverCheckBox("Play Video", self, self.statusBar())
        self.play_video_checkbox.setHoverMessage("Play the video (if unchecked, the mean image will be shown).")
        self.play_video_checkbox.setObjectName("Play Video")
        self.play_video_checkbox.setStyleSheet("color: rgba(150, 150, 150, 1);")
        self.play_video_checkbox.setChecked(True)
        self.show_rois_checkbox.setEnabled(False)
        self.play_video_checkbox.clicked.connect(self.toggle_play_video)
        self.bottom_layout.addWidget(self.play_video_checkbox)

        self.bottom_layout.addStretch()

        # create textbox to add a frame offset to calcium imaging data
        label = HoverLabel("Frame Offset:")
        label.setHoverMessage("Set number of frames to offset the calcium imaging data from tail angle data (for previewing only).")
        label.setStyleSheet("color: rgba(150, 150, 150, 1);")
        self.bottom_layout.addWidget(label)

        self.frame_offset_textbox = QLineEdit("Frame Offset")
        self.frame_offset_textbox.setObjectName("Frame Offset")
        self.frame_offset_textbox.setText("0")
        self.frame_offset_textbox.editingFinished.connect(lambda:self.update_frame_offset())
        self.frame_offset_textbox.setStyleSheet(ROUNDED_STYLESHEET)
        self.bottom_layout.addWidget(self.frame_offset_textbox)

        self.bottom_layout.addStretch()

        # create combobox to change the colormap used to plot kept ROI traces
        label = QLabel("Colormap:")
        label.setStyleSheet("color: rgba(150, 150, 150, 1);")
        self.bottom_layout.addWidget(label)

        combobox = QComboBox()
        combobox.addItems([ colormap.title() for colormap in colormaps ])
        combobox.currentIndexChanged.connect(self.change_colormap)
        self.bottom_layout.addWidget(combobox)

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

        self.item_hovered = False

        # set up the status bar
        self.statusBar().setStyleSheet(STATUSBAR_STYLESHEET)

        self.show()

    def set_default_statusbar_message(self, message):
        self.default_statusbar_message = message
        self.statusBar().showMessage(self.default_statusbar_message)

    def reset_default_statusbar_message(self):
        self.set_default_statusbar_message("To view a video, click the arrow next to its name.")

    def set_initial_state(self):
        self.kept_rois_overlay      = None
        self.discarded_rois_overlay = None
        self.kept_rois_image        = None
        self.discarded_rois_image   = None
        self.roi_overlays           = None
        self.roi_contours           = []
        self.text_items             = []
        self.outline_items          = []
        self.mask_items             = []
        self.temp_mask_item         = None
        self.image                  = None # image to show
        self.frames                 = None # frames to play
        self.frame_num              = 0    # current frame #
        self.n_frames               = 1    # total number of frames
        self.video_name             = ""   # name of the currently showing video
        self.mask_points            = []
        self.mask                   = None
        self.frame_offset           = 0
        self.heatmap                = None
        self.selected_rois          = []
        self.roi_temporal_footprints = None
        self.play_right             = False

        self.show_rois_checkbox.setEnabled(False)
        self.pg_widget.hide()
        self.top_widget.hide()
        self.bottom_widget.hide()
        self.timer.stop()
        self.setWindowTitle("Preview")
        self.reset_default_statusbar_message()

    def update_frame_offset(self):
        try:
            self.frame_offset = int(float(self.frame_offset_textbox.text()))

            if self.heatmap is not None:
                self.kept_traces_image.setRect(QRectF(self.frame_offset + self.controller.z/self.controller.video.shape[1], 0, self.heatmap.shape[0], self.heatmap.shape[1]))

            if self.roi_temporal_footprints is not None:
                self.plot_traces(self.roi_temporal_footprints, self.selected_rois)

            self.set_play_video(play_video_bool=self.play_video_checkbox.isChecked())
        except:
            pass

    def toggle_show_rois(self):
        show_rois = self.show_rois_checkbox.isChecked()

        self.set_show_rois(show_rois)

    def toggle_play_video(self):
        play_video_bool = self.play_video_checkbox.isChecked()

        self.set_play_video(play_video_bool)

    def uncheck_play_video(self):
        self.play_video_checkbox.setChecked(False)

        self.set_play_video(False)

    def check_play_video(self):
        self.play_video_checkbox.setChecked(True)

        self.set_play_video(True)

    def set_show_rois(self, show_rois):
        print("Setting show ROIs to {}.".format(show_rois))
        self.controller.set_show_rois(show_rois)

    def show_plot(self):
        self.pg_widget.show()
        self.bottom_widget.show()
        self.top_widget.show()

    def hide_plot(self):
        self.pg_widget.hide()
        self.bottom_widget.hide()
        self.top_widget.hide()

    def change_colormap(self, i):
        colormap_name = colormaps[i]

        colormap = cm.get_cmap(colormap_name)
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.kept_traces_image.setLookupTable(lut)

    def plot_tail_angles(self, tail_angles, tail_data_fps, imaging_fps):
        self.tail_angle_viewbox.clear()

        imaging_fps_one_plane = imaging_fps

        if tail_angles is not None:
            one_frame    = (1.0/imaging_fps_one_plane)*tail_data_fps
            total_frames = int(np.floor(one_frame*self.controller.video.shape[0] + self.frame_offset + 1))

            # print(tail_angles.shape)
            # print(total_frames)

            if total_frames < tail_angles.shape[0]:
                x = np.linspace(0, self.controller.video.shape[0]+self.frame_offset+1, total_frames)
                self.tail_angle_viewbox.plot(x, tail_angles[:total_frames, -1], pen=pg.mkPen((255, 255, 0), width=2))

                self.tail_angle_viewbox.addItem(self.current_frame_line_3)

                return True
            else:
                self.tail_angle_viewbox.addItem(self.current_frame_line_3)
                return False
        else:
            self.tail_angle_viewbox.addItem(self.current_frame_line_3)
            return False

    def plot_traces(self, roi_temporal_footprints, selected_rois=[]):
        self.roi_trace_viewbox.clear()
        self.selected_rois = selected_rois
        self.roi_temporal_footprints = roi_temporal_footprints
        if roi_temporal_footprints is not None and len(selected_rois) > 0:
            if self.controller.show_zscore:
                max_value = 1
            else:
                max_value = np.amax(roi_temporal_footprints)

            x = np.arange(roi_temporal_footprints.shape[1]) + self.controller.z/self.controller.video.shape[1] + self.frame_offset
            for i in range(len(selected_rois)):
                roi = selected_rois[i]

                color = cmap(roi % n_colors)[:3]
                color = [255*color[0], 255*color[1], 255*color[2]]

                self.roi_trace_viewbox.plot(x, roi_temporal_footprints[roi]/max_value, pen=pg.mkPen(color, width=2))

        self.roi_trace_viewbox.addItem(self.current_frame_line_1)

    def clear_text_and_outline_items(self):
        # remove all text and outline items from left and right plots
        for text_item in self.text_items:
            self.left_image_viewbox.removeItem(text_item)
            self.right_image_viewbox.removeItem(text_item)
            self.text_items = []
        for outline_item in self.outline_items:
            self.left_image_viewbox.removeItem(outline_item)
            self.right_image_viewbox.removeItem(outline_item)
            self.outline_items = []

    def clear_mask_items(self):
        for mask_item in self.mask_items:
            self.left_image_viewbox.removeItem(mask_item)
            self.mask_items = []

        if self.temp_mask_item is not None:
            self.left_image_viewbox.removeItem(self.temp_mask_item)
            self.temp_mask_item = None
            self.mask_points = []

    def plot_clicked(self, event):
        if self.controller.mode not in ("loading", "motion_correcting"):
            # get x-y coordinates of where the user clicked
            items = self.pg_widget.scene().items(event.scenePos())
            if self.left_image in items:
                pos = self.left_image_viewbox.mapSceneToView(event.scenePos())
            elif self.right_image in items:
                pos = self.right_image_viewbox.mapSceneToView(event.scenePos())
            else:
                return
            x = pos.x()
            y = pos.y()

            # check whether the user is holding Ctrl
            ctrl_held = event.modifiers() == Qt.ControlModifier

            # remove all text and outline items from left and right plots
            self.clear_text_and_outline_items()

            if event.button() == 1:
                # left click means selecting/deselecting ROIs or masks
                if not self.controller.drawing_mask:
                    self.controller.select_roi((int(y), int(x)), ctrl_held=ctrl_held)

                    # don't allow selecting removed & kept ROIs at the same time
                    removed_count = 0
                    for i in self.controller.selected_rois:
                        if i in self.controller.removed_rois():
                            removed_count += 1
                    if removed_count !=0 and removed_count != len(self.controller.selected_rois):
                        self.controller.selected_rois = [self.controller.selected_rois[-1]]

                    if len(self.controller.selected_rois) > 0:
                        if self.controller.selected_rois[-1] in self.controller.removed_rois() and self.left_image in items:
                            self.controller.selected_rois = []
                        elif self.controller.selected_rois[-1] not in self.controller.removed_rois() and self.right_image in items:
                            self.controller.selected_rois = []

                    if len(self.controller.selected_rois) > 0:
                        roi_to_select = self.controller.selected_rois[0]

                        if self.left_image in items:
                            image = self.kept_rois_image.copy()
                            contours = []
                            for i in self.controller.selected_rois:
                                contours += self.roi_contours[i]
                                x = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 0]) for j in range(len(self.roi_contours[i])) ])
                                y = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 1]) for j in range(len(self.roi_contours[i])) ])
                                
                                color = cmap(i % n_colors)[:3]
                                color = [255*color[0], 255*color[1], 255*color[2]]

                                text_item = pg.TextItem("{}".format(i), color=color)
                                text_item.setPos(QPoint(int(y), int(x)))
                                self.text_items.append(text_item)
                                self.left_image_viewbox.addItem(text_item)
                                for j in range(len(self.roi_contours[i])):
                                    outline_item = pg.PlotDataItem(np.concatenate([self.roi_contours[i][j][:, 0, 1], np.array([self.roi_contours[i][j][0, 0, 1]])]), np.concatenate([self.roi_contours[i][j][:, 0, 0], np.array([self.roi_contours[i][j][0, 0, 0]])]), pen=pg.mkPen(color, width=3))
                                    self.outline_items.append(outline_item)
                                    self.left_image_viewbox.addItem(outline_item)

                            # self.left_image.setImage(image, autoLevels=False)
                            # self.right_image.setImage(self.discarded_rois_image, autoLevels=False)
                        else:
                            image = self.discarded_rois_image.copy()
                            contours = []
                            for i in self.controller.selected_rois:
                                contours += self.roi_contours[i]
                                # print([ self.roi_contours[i][j].shape for j in range(len(self.roi_contours[i])) ])
                                x = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 0]) for j in range(len(self.roi_contours[i])) ])
                                y = np.amax([ np.amax(self.roi_contours[i][j][:, 0, 1]) for j in range(len(self.roi_contours[i])) ])
                                
                                color = cmap(i % n_colors)[:3]
                                color = [255*color[0], 255*color[1], 255*color[2]]

                                text_item = pg.TextItem("{}".format(i), color=color)
                                text_item.setPos(QPoint(int(y), int(x)))
                                self.text_items.append(text_item)
                                self.right_image_viewbox.addItem(text_item)
                                for j in range(len(self.roi_contours[i])):
                                    outline_item = pg.PlotDataItem(np.concatenate([self.roi_contours[i][j][:, 0, 1], np.array([self.roi_contours[i][j][0, 0, 1]])]), np.concatenate([self.roi_contours[i][j][:, 0, 0], np.array([self.roi_contours[i][j][0, 0, 0]])]), pen=pg.mkPen(color, width=3))
                                    self.outline_items.append(outline_item)
                                    self.right_image_viewbox.addItem(outline_item)

                            # self.right_image.setImage(image, autoLevels=False)
                            # self.left_image.setImage(self.kept_rois_image, autoLevels=False)
                    else:
                        # self.left_image.setImage(self.kept_rois_image, autoLevels=False, show_rois=self.show_rois_checkbox.isChecked())
                        # self.right_image.setImage(self.discarded_rois_image, autoLevels=False, show_rois=self.show_rois_checkbox.isChecked())
                        pass
                elif self.left_image in items:
                    if ctrl_held:
                        # add mask point
                        self.mask_points.append([x, y])

                        if self.temp_mask_item is not None:
                            self.left_image_viewbox.removeItem(self.temp_mask_item)

                        self.temp_mask_item = self.create_mask_item(self.mask_points, temporary=True)
                        self.left_image_viewbox.addItem(self.temp_mask_item)
                    else:
                        # determine which mask was clicked, if any
                        mask_num = -1

                        if self.controller.mask_images is not None:
                            for i in range(len(self.controller.mask_images[self.controller.z])):
                                mask = self.controller.mask_images[self.controller.z][i]

                                if mask[int(y), int(x)] > 0:
                                    mask_num = i

                        if mask_num == -1 and len(self.mask_points) >= 3:
                            self.controller.create_mask(self.mask_points)

                            self.mask_points = []

                        self.update_mask_items(selected_mask=mask_num)

            elif event.button() == 2:
                if not self.controller.drawing_mask:
                    if self.left_image in items:
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

                    self.create_roi_heatmap(roi_spatial_footprints=self.controller.roi_spatial_footprints(), removed_rois=self.controller.removed_rois())
                else:
                    # if not ctrl_held:
                    #     if len(self.mask_points) >= 3:
                    #         self.controller.create_mask(self.mask_points)

                    #         self.update_mask_items(selected_mask=len(self.controller.mask_points())-1)

                    #         self.mask_points = []
                    if not ctrl_held:
                        # determine which mask was clicked, if any
                        mask_num = -1

                        if self.controller.mask_images is not None:
                            for i in range(len(self.controller.mask_images[self.controller.z])):
                                mask = self.controller.mask_images[self.controller.z][i]

                                if mask[int(y), int(x)] > 0:
                                    mask_num = i

                            if mask_num >= 0:
                                # delete this maximumsk
                                self.controller.delete_mask(mask_num)

                                self.update_mask_items()

            self.controller.update_trace_plot()

    def create_mask_item(self, mask_points, temporary=False, selected=False):
        if temporary:
            return pg.PlotDataItem([ p[0] for p in mask_points ] + [mask_points[0][0]], [ p[1] for p in mask_points ] + [mask_points[0][1]], symbolSize=5, pen=pg.mkPen((255, 255, 255)), symbolPen=pg.mkPen((255, 255, 255)))
        elif selected:
            return pg.PlotDataItem([ p[0] for p in mask_points ] + [mask_points[0][0]], [ p[1] for p in mask_points ] + [mask_points[0][1]], symbolSize=5, pen=pg.mkPen((0, 255, 0)), symbolPen=pg.mkPen((0, 255, 0)))
        return pg.PlotDataItem([ p[0] for p in mask_points ] + [mask_points[0][0]], [ p[1] for p in mask_points ] + [mask_points[0][1]], symbolSize=5, pen=pg.mkPen((255, 255, 0)), symbolPen=pg.mkPen((255, 255, 0)))

    def plot_image(self, image, roi_spatial_footprints=None, video_max=255, show_rois=False):
        # if update_overlay or recreate_overlays or recreate_roi_images:
        #     roi_spatial_footprints = self.controller.roi_spatial_footprints()
        #     if roi_spatial_footprints is not None:
        #         roi_spatial_footprints = roi_spatial_footprints.toarray().reshape((self.controller.video.shape[2], self.controller.video.shape[3], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))

        # if update_overlay and roi_spatial_footprints is not None:
        #     recreate_overlays = True
        #     recreate_roi_images = True
        #     self.compute_contours_and_overlays(image.shape, roi_spatial_footprints)

        # if recreate_overlays:
        #     recreate_roi_images = True
        #     removed_rois = self.controller.removed_rois()

        #     self.compute_kept_rois_overlay(roi_spatial_footprints, removed_rois)
        #     self.compute_discarded_rois_overlay(roi_spatial_footprints, removed_rois)

        if image is None:
            self.hide_plot()
        else:
            self.show_plot()

        # update image
        self.image = image

        self.update_left_image_plot(self.image, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois)
        self.update_right_image_plot(self.image, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois)

        # # update image plot
        # if recreate_roi_images:
        #     print("Recreating ROI images.")
        #     self.update_left_image_plot(self.image, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois)
        #     self.update_right_image_plot(self.image, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=show_rois)

        #     self.create_roi_heatmap(roi_spatial_footprints=self.controller.roi_spatial_footprints(), removed_rois=self.controller.removed_rois())
        # else:
        #     if not show_rois:
        #         image = 255.0*self.image/self.controller.video_max
        #         image[image > 255] = 255

        #         if len(image.shape) < 3:
        #             image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        #         else:
        #             image = image.astype(np.uint8)

        #         self.left_image.setImage(image, autoLevels=False)
        #         self.right_image.setImage(image, autoLevels=False)
        #     else:
        #         self.left_image.setImage(self.kept_rois_image, autoLevels=False)
        #         self.right_image.setImage(self.discarded_rois_image, autoLevels=False)

    def set_play_video(self, play_video_bool=True):
        self.timer.stop()

        if play_video_bool:
            self.controller.play_video()
        else:
            self.controller.show_mean_image()

            self.current_frame_line_1.setValue(self.frame_offset)
            self.current_frame_line_2.setValue(self.frame_offset)
            self.current_frame_line_3.setValue(self.frame_offset)

        if self.controller.mode in ("loading", "motion correcting"):
            if play_video_bool:
                self.left_label.setText("Video ▼")
            else:
                self.left_label.setText("Mean Image ▼")

        self.controller.set_play_video(play_video_bool)

    def play_video(self, video, video_path, fps, play_right=False):
        print("Playing video...")
        print(video.shape)
        self.video_name = os.path.basename(video_path)

        self.play_right = play_right
        
        self.show_plot()

        # set frame number to 0
        self.frame_num = 0

        # normalize the frames (to be between 0 and 255)
        self.frames = video

        # get the number of frames
        self.n_frames = self.frames.shape[0]

        # start the timer to update the frames
        self.timer.start(int(1000.0/fps))

        self.kept_traces_viewbox.setXRange(0, self.controller.video.shape[0])

    def set_fps(self, fps):
        # restart the timer with the new fps
        self.timer.stop()
        self.timer.start(int(1000.0/fps))

    def show_frame(self, frame):
        self.update_left_image_plot(frame, roi_spatial_footprints=self.controller.roi_spatial_footprints(), video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=self.controller.show_rois)

    def update_frame(self):
        if self.frames is not None:
            # convert the current frame to RGB
            # frame = cv2.cvtColor(self.frames[self.frame_num].astype(np.float32), cv2.COLOR_GRAY2RGB)
            frame = self.frames[self.frame_num]

            # self.show_frame(frame)
            self.update_left_image_plot(frame, roi_spatial_footprints=self.controller.roi_spatial_footprints(), video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=self.controller.show_rois)
            
            if self.play_right:
                self.update_right_image_plot(frame, roi_spatial_footprints=self.controller.roi_spatial_footprints(), video_dimensions=self.controller.video.shape, removed_rois=self.controller.removed_rois(), selected_rois=self.controller.selected_rois, show_rois=self.controller.show_rois)

            # increment frame number (keeping it between 0 and n_frames)
            self.frame_num += 1
            self.frame_num = self.frame_num % self.n_frames

            self.current_frame_line_1.setValue(self.frame_num + self.frame_offset)
            self.current_frame_line_2.setValue(self.frame_num + self.frame_offset)
            self.current_frame_line_3.setValue(self.frame_num + self.frame_offset)

            if not self.item_hovered:
                # update status bar
                self.set_default_statusbar_message("Viewing {}. Z={}. Frame {}/{}.".format(self.video_name, self.controller.z, self.frame_num + 1, self.n_frames))

    def update_left_image_plot(self, image, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False):
        # print("Updating left image plot...")
        # image = self.create_kept_rois_image(image, self.controller.video_max, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, show_rois=show_rois)
        # if show_rois:
        #     self.kept_rois_image = image

        image = 255.0*image/self.controller.video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        self.kept_rois_image = image

        self.left_image.setImage(image, autoLevels=False)

        if show_rois:
            self.left_image_overlay.setImage(self.kept_rois_overlay, levels=(0, 255))
        else:
            if self.kept_rois_overlay is not None:
                self.left_image_overlay.setImage(0*self.kept_rois_overlay, levels=(0, 255))

        self.update_mask_items()

        if not self.item_hovered:
            self.set_default_statusbar_message("Viewing {}. Z={}.".format(self.video_name, self.controller.z))

    def update_mask_items(self, selected_mask=-1):
        self.clear_mask_items()
        mask_points = self.controller.mask_points()

        for i in range(len(mask_points)):
            mask_item = self.create_mask_item(mask_points[i], selected=selected_mask==i)
            self.left_image_viewbox.addItem(mask_item)
            self.mask_items.append(mask_item)

    def compute_contours_and_overlays(self, shape, roi_spatial_footprints):
        print("Computing new contours.....")
        
        if roi_spatial_footprints is not None:
            self.roi_contours  = [ None for i in range(roi_spatial_footprints.shape[-1]) ]
            self.flat_contours = []
            
            self.roi_overlays = np.zeros((roi_spatial_footprints.shape[-1], shape[0], shape[1], 4)).astype(np.uint8)

            for i in range(roi_spatial_footprints.shape[-1]):
                maximum = np.amax(roi_spatial_footprints[:, :, i])

                mask = (roi_spatial_footprints[:, :, i] > 0).copy()
                
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                color = cmap(i % n_colors)[:3]
                color = [255*color[0], 255*color[1], 255*color[2]]

                overlay = np.zeros((shape[0], shape[1], 4)).astype(np.uint8)
                overlay[mask, :-1] = color
                overlay[mask, -1] = 255.0*roi_spatial_footprints[mask, i]/maximum
                self.roi_overlays[i] = overlay

                self.roi_contours[i] = contours
                self.flat_contours += contours

    def compute_kept_rois_overlay(self, roi_spatial_footprints, removed_rois):
        print("Computing kept ROIs overlay.")

        if roi_spatial_footprints is not None:
            kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]
            if len(kept_rois) > 0:
                a = Image.fromarray(self.roi_overlays[kept_rois[0]])
                for roi in kept_rois[1:]:
                    print(self.roi_overlays[roi][:, :, -1])
                    b = Image.fromarray(self.roi_overlays[roi])
                    a.alpha_composite(b)
                self.kept_rois_overlay = np.asarray(a)

    def compute_discarded_rois_overlay(self, roi_spatial_footprints, removed_rois):
        if roi_spatial_footprints is not None:
            if len(removed_rois) > 0:
                a = Image.fromarray(self.roi_overlays[removed_rois[0]])
                for roi in removed_rois[1:]:
                    print(self.roi_overlays[roi][:, :, -1])
                    b = Image.fromarray(self.roi_overlays[roi])
                    a.alpha_composite(b)
                self.discarded_rois_overlay = np.asarray(a)

    def create_kept_rois_image(self, image, video_max, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False):
        image = 255.0*image/video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if self.kept_rois_overlay is not None and roi_spatial_footprints is not None:
            if show_rois:
                image = utilities.blend_transparent(image, self.kept_rois_overlay)

        return image

    def create_roi_heatmap(self, roi_spatial_footprints=None, removed_rois=None):
        print("Creating ROI heatmap...")
        # self.kept_traces_viewbox.clear()
        if roi_spatial_footprints is not None:
            kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in removed_rois ]

            heatmap = self.controller.roi_temporal_footprints()[kept_rois]

            print("heatmap shape: {}".format(heatmap.shape))
            
            video_lengths = self.controller.selected_group_video_lengths()

            print("video lengths: {}".format(video_lengths))

            index = self.controller.selected_group_video_paths().index(self.controller.selected_video_path())

            if index == 0:
                heatmap = heatmap[:, :video_lengths[0]]
            else:
                heatmap = heatmap[:, np.sum(video_lengths[:index]):np.sum(video_lengths[:index+1])]

            print(heatmap.shape)

            if heatmap.shape[0] > 0:
                # heatmap = scipy.ndimage.interpolation.shift(heatmap, (self.controller.z, 0), cval=0)
                if self.controller.show_zscore:
                    heatmap = (heatmap - np.mean(heatmap, axis=1)[:, np.newaxis])/np.std(heatmap, axis=1)[:, np.newaxis]

                    if heatmap.shape[0] > 2:
                        correlations = np.corrcoef(heatmap)
                        i, j = np.unravel_index(correlations.argmin(), correlations.shape)
                    
                        heatmap_sorted = heatmap.copy()
                        heatmap_sorted[0]  = heatmap[i]
                        heatmap_sorted[-1] = heatmap[j]
                        
                        remaining_indices = [ index for index in range(heatmap.shape[0]) if index not in (i, j) ]
                        for k in range(1, heatmap.shape[0]-1):
                            corrs_1 = [ correlations[i, index] for index in remaining_indices ]
                            corrs_2 = [ correlations[j, index] for index in remaining_indices ]
                            
                            difference = [ corrs_1[l] - corrs_2[l] for l in range(len(remaining_indices)) ]
                            l = np.argmax(difference)
                            index = remaining_indices[l]
                            
                            heatmap_sorted[k] = heatmap[index]
                            
                            del remaining_indices[l]

                        heatmap = heatmap_sorted
                else:
                    heatmap = heatmap/np.amax(heatmap)

                if self.controller.show_zscore:
                    heatmap[heatmap > 3] = 3
                    heatmap[heatmap < -2] = -2
                # else:
                #     heatmap[heatmap > 1] = 1
                #     heatmap[heatmap < 0] = 0

                self.heatmap = heatmap.T

                # print(np.amin(heatmap), np.amax(heatmap))
                # print(heatmap[0])

                if self.controller.show_zscore:
                    self.kept_traces_image.setImage(self.heatmap, levels=(-2.01, 3.01))
                else:
                    self.kept_traces_image.setImage(self.heatmap, levels=(0, 1.01))

                self.kept_traces_image.setRect(QRectF(self.frame_offset + self.controller.z/self.controller.video.shape[1], 0, heatmap.shape[1], heatmap.shape[0]))
            else:
                self.kept_traces_image.setImage(None)
        else:
            self.kept_traces_image.setImage(None)

        # self.kept_traces_viewbox.addItem(self.kept_traces_image)
        # self.kept_traces_viewbox.addItem(self.current_frame_line_2)
        # self.current_frame_line_2.setValue(self.frame_offset)

    def create_discarded_rois_image(self, image, video_max, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False):
        image = 255.0*image/video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        if show_rois and self.discarded_rois_overlay is not None and roi_spatial_footprints is not None:
            image = utilities.blend_transparent(image, self.discarded_rois_overlay)

        return image

    def plot_mean_image(self, image, video_max):
        image = 255.0*image/video_max
        image[image > 255] = 255
        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        self.right_image.setImage(image, autoLevels=False)

    def update_right_image_plot(self, image, roi_spatial_footprints=None, video_dimensions=None, removed_rois=None, selected_rois=None, show_rois=False):
        # image = self.create_discarded_rois_image(image, self.controller.video_max, roi_spatial_footprints=roi_spatial_footprints, video_dimensions=video_dimensions, removed_rois=removed_rois, selected_rois=selected_rois, show_rois=show_rois)
        # if show_rois:
            # self.discarded_rois_image = image

        image = 255.0*image/self.controller.video_max
        image[image > 255] = 255

        if len(image.shape) < 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image = image.astype(np.uint8)

        self.discarded_rois_image = image

        self.right_image.setImage(image, autoLevels=False)

        if show_rois:
            self.right_image_overlay.setImage(self.discarded_rois_overlay, levels=(0, 255))
        else:
            if self.discarded_rois_overlay is not None:
                self.right_image_overlay.setImage(0*self.discarded_rois_overlay, levels=(0, 255))

    def reset_zoom(self):
        self.left_image_viewbox.autoRange()
        self.right_image_viewbox.autoRange()

    def item_hover_entered(self):
        self.item_hovered = True

    def item_hover_exited(self):
        self.item_hovered = False

    def closeEvent(self, ce):
        if not self.controller.closing:
            ce.ignore()
        else:
            ce.accept()

class HoverCheckBox(QCheckBox):
    def __init__(self, text, parent=None, status_bar=None):
        QCheckBox.__init__(self, text, parent)
        self.setMouseTracking(True)

        self.parent        = parent
        self.status_bar    = status_bar
        self.hover_message = ""

    def setHoverMessage(self, message):
        self.hover_message = message

    def enterEvent(self, event):
        if self.status_bar is not None:
            self.status_bar.showMessage(self.hover_message)
            self.parent.item_hover_entered()

    def leaveEvent(self, event):
        if self.status_bar is not None:
            self.status_bar.showMessage(self.parent.default_statusbar_message)
            self.parent.item_hover_exited()

class HoverLabel(QLabel):
    def __init__(self, text, parent=None, status_bar=None):
        QLabel.__init__(self, text, parent)
        self.setMouseTracking(True)

        self.parent        = parent
        self.status_bar    = status_bar
        self.hover_message = ""

    def setHoverMessage(self, message):
        self.hover_message = message

    def enterEvent(self, event):
        if self.status_bar is not None:
            self.status_bar.showMessage(self.hover_message)

    def leaveEvent(self, event):
        if self.status_bar is not None:
            self.status_bar.showMessage(self.parent.default_statusbar_message)

def HLine():
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    frame.setFrameShadow(QFrame.Plain)

    frame.setStyleSheet("color: rgba(0, 0, 0, 0.2);")

    return frame
