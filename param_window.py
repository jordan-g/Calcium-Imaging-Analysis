from __future__ import division
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

import os

# set styles of title and subtitle labels
TITLE_STYLESHEET    = "font-size: 16px; font-weight: bold;"
SUBTITLE_STYLESHEET = "font-size: 14px; font-weight: bold;"

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Automatic ROI Segmentation")

        # set initial position
        self.setGeometry(0, 32, 10, 10)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set up the status bar
        self.statusBar().setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-top: 1px solid rgba(0, 0, 0, 0.1); font-size: 10px; font-style: italic;")
        self.statusBar().showMessage("To begin, open one or more video files. TIFF and NPY files are supported.")

        # create video list widget
        self.videos_widget = VideosWidget(self, self.controller)
        self.main_layout.addWidget(self.videos_widget, 0, 0)
        self.delete_shortcut = QShortcut(QKeySequence('Backspace'), self.videos_widget.videos_list)
        self.delete_shortcut.activated.connect(self.remove_selected_items)

        self.main_param_widget = MainParamWidget(self, self.controller)
        self.main_layout.addWidget(self.main_param_widget, 1, 0)

        self.main_layout.addWidget(HLine(), 2, 0)

        # create stacked widget
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.stacked_widget, 3, 0)

        # create motion correction widget
        self.motion_correction_widget = MotionCorrectionWidget(self, self.controller)
        self.stacked_widget.addWidget(self.motion_correction_widget)

        # create ROI finding widget
        self.roi_finding_widget = ROIFindingWidget(self, self.controller)
        self.stacked_widget.addWidget(self.roi_finding_widget)

        # create ROI filtering widget
        self.roi_filtering_widget = ROIFilteringWidget(self, self.controller)
        self.stacked_widget.addWidget(self.roi_filtering_widget)

        # create menus
        self.create_menus()

        # set initial state of widgets, buttons & menu items
        self.set_initial_state()

        # set window title bar buttons
        if pyqt_version == 5:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)
        else:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.show()

    def set_initial_state(self):
        # disable buttons, widgets & menu items
        self.main_param_widget.setDisabled(True)
        self.stacked_widget.setDisabled(True)
        self.videos_widget.load_rois_button.setDisabled(True)
        self.videos_widget.save_rois_button.setDisabled(True)
        self.videos_widget.process_all_button.setDisabled(True)
        self.show_rois_action.setEnabled(False)
        self.save_roi_image_action.setEnabled(False)

    def create_menus(self):
        self.add_videos_action = QAction('Add Videos...', self)
        self.add_videos_action.setShortcut('Ctrl+O')
        self.add_videos_action.setStatusTip('Add video files for processing.')
        self.add_videos_action.triggered.connect(self.controller.select_videos_to_import)

        self.show_rois_action = QAction('Show ROIs', self, checkable=True)
        self.show_rois_action.setShortcut('R')
        self.show_rois_action.setStatusTip('Toggle showing the ROIs.')
        self.show_rois_action.triggered.connect(lambda:self.controller.show_roi_image(self.show_rois_action.isChecked()))
        self.show_rois_action.setEnabled(False)
        self.show_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.save_roi_image_action = QAction('Save ROI Image...', self)
        self.save_roi_image_action.setShortcut('Ctrl+Alt+S')
        self.save_roi_image_action.setStatusTip('Save an image of the current ROIs.')
        self.save_roi_image_action.triggered.connect(self.controller.save_roi_image)
        self.save_roi_image_action.setEnabled(False)

        # create menu bar
        menubar = self.menuBar()

        # add menu items
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.add_videos_action)
        file_menu.addAction(self.save_roi_image_action)
        # file_menu.addSeparator()

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(self.show_rois_action)

    def video_opened(self, max_z, z):
        self.stacked_widget.setDisabled(False)
        self.statusBar().showMessage("")
        self.main_param_widget.param_sliders["z"].setMaximum(max_z)
        self.main_param_widget.param_sliders["z"].setValue(z)
        self.main_param_widget.param_textboxes["z"].setText(str(z))
        self.videos_widget.save_mc_video_button.setEnabled(False)
        self.motion_correction_widget.use_mc_video_checkbox.setChecked(False)
        self.motion_correction_widget.use_mc_video_checkbox.setDisabled(True)

    def videos_imported(self, video_paths):
        self.videos_widget.videos_imported(video_paths)

        self.main_param_widget.setDisabled(False)
        self.stacked_widget.setDisabled(False)
        self.videos_widget.load_rois_button.setDisabled(False)
        self.videos_widget.save_rois_button.setEnabled(True)
        self.videos_widget.process_all_button.setEnabled(True)

    def remove_selected_items(self):
        self.videos_widget.remove_selected_items()

    def process_videos_started(self):
        self.videos_widget.process_videos_started()

    def roi_erasing_started(self):
        self.main_param_widget.setEnabled(False)
        self.videos_widget.setEnabled(False)

        self.roi_filtering_widget.roi_erasing_started()

    def roi_erasing_ended(self):
        self.main_param_widget.setEnabled(True)
        self.videos_widget.setEnabled(True)

        self.roi_filtering_widget.roi_erasing_ended()

    def roi_drawing_started(self):
        self.main_param_widget.setEnabled(False)
        self.videos_widget.setEnabled(False)

        self.roi_filtering_widget.roi_drawing_started()

    def roi_drawing_ended(self):
        self.main_param_widget.setEnabled(True)
        self.videos_widget.setEnabled(True)

        self.roi_filtering_widget.roi_drawing_ended()

    def update_process_videos_progress(self, percent):
        self.videos_widget.update_process_videos_progress(percent)

    def rois_created(self):
        pass

    def closeEvent(self, event):
        self.controller.close_all()

class VideosWidget(QWidget):
    def __init__(self, parent_widget, controller):
        QWidget.__init__(self)

        self.parent_widget = parent_widget

        self.controller = controller

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # create main buttons
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.open_file_button = HoverButton('Add...', None, self.parent_widget.statusBar())
        self.open_file_button.setHoverMessage("Add video files for processing.")
        self.open_file_button.setStyleSheet('font-weight: bold;')
        self.open_file_button.setIcon(QIcon("icons/open_file_icon.png"))
        self.open_file_button.setIconSize(QSize(16,16))
        self.open_file_button.clicked.connect(self.controller.select_videos_to_import)
        self.button_layout.addWidget(self.open_file_button)

        self.remove_videos_button = HoverButton('Remove', None, self.parent_widget.statusBar())
        self.remove_videos_button.setHoverMessage("Remove the currently selected videos.")
        self.remove_videos_button.setIcon(QIcon("icons/trash_icon.png"))
        self.remove_videos_button.setIconSize(QSize(16,16))
        self.remove_videos_button.setDisabled(True)
        self.remove_videos_button.clicked.connect(self.remove_selected_items)
        self.button_layout.addWidget(self.remove_videos_button)

        self.button_layout.addStretch()

        self.save_mc_video_button = HoverButton('Save Motion-Corrected Video...', None, self.parent_widget.statusBar())
        self.save_mc_video_button.setHoverMessage("Save the current motion-corrected video.")
        self.save_mc_video_button.setIcon(QIcon("icons/save_icon.png"))
        self.save_mc_video_button.setIconSize(QSize(16,16))
        self.save_mc_video_button.setEnabled(False)
        self.save_mc_video_button.clicked.connect(self.controller.save_mc_video)
        self.button_layout.addWidget(self.save_mc_video_button)

        self.save_rois_button = HoverButton('Save ROIs...', None, self.parent_widget.statusBar())
        self.save_rois_button.setHoverMessage("Save the current ROIs.")
        self.save_rois_button.setIcon(QIcon("icons/save_icon.png"))
        self.save_rois_button.setIconSize(QSize(16,16))
        self.save_rois_button.clicked.connect(self.controller.save_rois)
        self.button_layout.addWidget(self.save_rois_button)

        self.load_rois_button = HoverButton('Load ROIs...', None, self.parent_widget.statusBar())
        self.load_rois_button.setHoverMessage("Load saved ROIs.")
        self.load_rois_button.setIcon(QIcon("icons/load_icon.png"))
        self.load_rois_button.setIconSize(QSize(16,16))
        self.load_rois_button.clicked.connect(self.controller.load_rois)
        self.button_layout.addWidget(self.load_rois_button)

        self.title_widget = QWidget(self)
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setContentsMargins(10, 0, 10, 0)
        self.main_layout.addWidget(self.title_widget)

        self.title_label = QLabel("Videos to Process")
        self.title_label.setStyleSheet(TITLE_STYLESHEET)
        self.title_layout.addWidget(self.title_label)
        self.main_layout.setAlignment(self.title_widget, Qt.AlignTop)

        self.videos_list_widget = QWidget(self)
        self.videos_list_layout = QHBoxLayout(self.videos_list_widget)
        self.videos_list_layout.setContentsMargins(10, 0, 10, 10)
        self.main_layout.addWidget(self.videos_list_widget)
        
        self.videos_list = QListWidget(self)
        self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.videos_list.itemSelectionChanged.connect(self.item_selected)
        self.videos_list_layout.addWidget(self.videos_list)

        # create secondary buttons
        self.button_widget_2 = QWidget(self)
        self.button_layout_2 = QHBoxLayout(self.button_widget_2)
        self.button_layout_2.setContentsMargins(10, 0, 0, 0)
        self.button_layout_2.setSpacing(5)
        self.main_layout.addWidget(self.button_widget_2)

        self.motion_correct_checkbox = QCheckBox("Motion-correct all videos")
        self.motion_correct_checkbox.setObjectName("Motion-correct all videos")
        self.motion_correct_checkbox.setChecked(False)
        self.motion_correct_checkbox.setEnabled(False)
        self.motion_correct_checkbox.clicked.connect(lambda:self.controller.set_motion_correct(self.motion_correct_checkbox.isChecked()))
        self.button_layout_2.addWidget(self.motion_correct_checkbox)

        self.button_layout_2.addStretch()

        self.process_videos_progress_label = QLabel("")
        self.process_videos_progress_label.setStyleSheet("font-size: 10px; font-style: italic;")
        self.process_videos_progress_label.setMinimumWidth(200)
        self.process_videos_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.button_layout_2.addWidget(self.process_videos_progress_label)

        self.process_all_button = HoverButton('Process All', None, self.parent_widget.statusBar())
        self.process_all_button.setHoverMessage("Extract activities of current ROIs from all loaded videos.")
        self.process_all_button.setStyleSheet('font-weight: bold;')
        self.process_all_button.setIcon(QIcon("icons/play_icon.png"))
        self.process_all_button.setIconSize(QSize(16,16))
        self.process_all_button.clicked.connect(self.controller.process_all_videos)
        self.button_layout_2.addWidget(self.process_all_button)

        self.main_layout.addWidget(HLine())

    def videos_imported(self, video_paths):
        self.motion_correct_checkbox.setEnabled(True)
        for video_path in video_paths:
            self.videos_list.addItem(os.path.basename(video_path))

    def remove_selected_items(self):
        selected_items = self.videos_list.selectedItems()

        self.controller.remove_videos_at_indices([ x.row() for x in self.videos_list.selectedIndexes() ])

        for i in range(len(selected_items)-1, -1, -1):
            self.videos_list.takeItem(self.videos_list.row(selected_items[i]))
        
        if self.videos_list.count() == 0:
            self.motion_correct_checkbox.setEnabled(False)

    def item_selected(self):
        selected_items = self.videos_list.selectedItems()

        if len(selected_items) > 0:
            self.remove_videos_button.setDisabled(False)
        else:
            self.remove_videos_button.setDisabled(True)

    def process_videos_started(self):
        self.process_videos_progress_label.setText("Processing videos... 0%.")
        self.process_all_button.setText('Cancel')
        self.process_all_button.setHoverMessage("Stop processing videos.")
        self.motion_correct_checkbox.setEnabled(False)
        self.save_rois_button.setEnabled(False)
        self.load_rois_button.setEnabled(False)
        self.open_file_button.setEnabled(False)
        self.remove_videos_button.setEnabled(False)

    def update_process_videos_progress(self, percent):
        if percent == 100 or percent == -1:
            self.process_videos_progress_label.setText("")
            self.process_all_button.setText('Process All')
            self.process_all_button.setHoverMessage("Extract activities of current ROIs from all loaded videos.")
            self.motion_correct_checkbox.setEnabled(True)
            self.save_rois_button.setEnabled(True)
            self.load_rois_button.setEnabled(True)
            self.open_file_button.setEnabled(True)
            self.item_selected()
        else:
            self.process_videos_progress_label.setText("Processing videos... {}%.".format(int(percent)))

class ParamWidget(QWidget):
    def __init__(self, parent_widget, controller, title, stylesheet=TITLE_STYLESHEET):
        QWidget.__init__(self)

        self.parent_widget = parent_widget

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.title_widget = QWidget(self)
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setContentsMargins(5, 5, 0, 0)
        self.main_layout.addWidget(self.title_widget)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(stylesheet)
        self.title_layout.addWidget(self.title_label)
        self.main_layout.setAlignment(self.title_widget, Qt.AlignTop)
        
        self.param_widget = QWidget(self)
        self.param_layout = QGridLayout(self.param_widget)
        self.param_layout.setContentsMargins(5, 5, 5, 5)
        self.param_layout.setSpacing(5)
        self.main_layout.addWidget(self.param_widget)

        self.param_sliders            = {}
        self.param_slider_multipliers = {}
        self.param_textboxes          = {}

    def add_param_slider(self, label_name, name, minimum, maximum, moved, multiplier=1, pressed=None, released=None, description=None, int_values=False):
        row = len(self.param_sliders.keys())

        if released == self.update_param:
            released = lambda:self.update_param(name, int_values=int_values)
        if moved == self.update_param:
            moved = lambda:self.update_param(name, int_values=int_values)
        if pressed == self.update_param:
            pressed = lambda:self.update_param(name, int_values=int_values)

        label = HoverLabel("{}:".format(label_name), None, self.parent_widget.statusBar())
        label.setHoverMessage(description)
        self.param_layout.addWidget(label, row, 0)

        slider = QSlider(Qt.Horizontal)
        # slider.setFixedWidth(300)
        slider.setObjectName(name)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(self.controller.params[name]*multiplier)
        slider.sliderMoved.connect(moved)
        if pressed:
            slider.sliderPressed.connect(pressed)
        if released:
            slider.sliderReleased.connect(released)
        self.param_layout.addWidget(slider, row, 1)

        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier, int_values))
        slider.sliderReleased.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier, int_values))

        # make textbox & add to layout
        textbox = QLineEdit()
        textbox.setStyleSheet("border-radius: 3px; border: 1px solid #ccc; padding: 2px;")
        textbox.setAlignment(Qt.AlignHCenter)
        textbox.setObjectName(name)
        textbox.setFixedWidth(60)
        textbox.setFixedHeight(20)
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier, int_values))
        textbox.editingFinished.connect(released)
        self.update_textbox_from_slider(slider, textbox, multiplier, int_values)
        self.param_layout.addWidget(textbox, row, 2)

        self.param_sliders[name]            = slider
        self.param_slider_multipliers[name] = multiplier
        self.param_textboxes[name]          = textbox

    def update_textbox_from_slider(self, slider, textbox, multiplier=1, int_values=False):
        if int_values:
            textbox.setText(str(int(slider.sliderPosition()/multiplier)))
        else:
            textbox.setText(str(slider.sliderPosition()/multiplier))

    def update_slider_from_textbox(self, slider, textbox, multiplier=1, int_values=False):
        try:
            if int_values:
                value = int(float(textbox.text()))
            else:
                value = float(textbox.text())
            
            slider.setValue(value*multiplier)
            textbox.setText(str(value))
        except:
            pass

    def update_param(self, param, int_values=False):
        value = self.param_sliders[param].sliderPosition()/float(self.param_slider_multipliers[param])

        if int_values:
            value = int(value)

        self.controller.update_param(param, value)

class MainParamWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "Preview Parameters")

        self.controller = controller

        self.add_param_slider(label_name="Gamma", name="gamma", minimum=1, maximum=500, moved=self.preview_gamma, multiplier=100, released=self.update_param, description="Gamma of the video preview.")
        self.add_param_slider(label_name="Contrast", name="contrast", minimum=1, maximum=500, moved=self.preview_contrast, multiplier=100, released=self.update_param, description="Contrast of the video preview.")
        self.add_param_slider(label_name="FPS", name="fps", minimum=1, maximum=60, moved=self.update_param, released=self.update_param, description="Frames per second of the video preview.", int_values=True)
        self.add_param_slider(label_name="Z", name="z", minimum=0, maximum=0, moved=self.update_param, released=self.update_param, description="Z plane of the video preview.", int_values=True)

    def preview_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/float(self.param_slider_multipliers["contrast"])

        self.controller.preview_contrast(contrast)

    def preview_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/float(self.param_slider_multipliers["gamma"])

        self.controller.preview_gamma(gamma)

class MotionCorrectionWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "Motion Correction Parameters", stylesheet=TITLE_STYLESHEET)

        self.controller = controller

        self.add_param_slider(label_name="Maximum Shift", name="max_shift", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Maximum shift (in pixels) allowed for motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Stride", name="patch_stride", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Stride length (in pixels) of each patch used in motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Overlap", name="patch_overlap", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Overlap (in pixels) of patches used in motion correction.", int_values=True)
        
        self.main_layout.addStretch()

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.mc_current_z_checkbox = HoverCheckBox("Motion-correct only this z plane", None, self.parent_widget.statusBar())
        self.mc_current_z_checkbox.setHoverMessage("Apply motion correction only to the current z plane. Useful for quick troubleshooting.")
        self.mc_current_z_checkbox.setChecked(False)
        self.mc_current_z_checkbox.clicked.connect(lambda:self.controller.set_mc_current_z(self.mc_current_z_checkbox.isChecked()))
        self.button_layout.addWidget(self.mc_current_z_checkbox)

        self.button_widget_2 = QWidget(self)
        self.button_layout_2 = QHBoxLayout(self.button_widget_2)
        self.button_layout_2.setContentsMargins(5, 0, 0, 0)
        self.button_layout_2.setSpacing(5)
        self.main_layout.addWidget(self.button_widget_2)

        self.use_mc_video_checkbox = QCheckBox("Use motion-corrected video")
        self.use_mc_video_checkbox.setObjectName("Use motion-corrected video")
        self.use_mc_video_checkbox.setChecked(False)
        self.use_mc_video_checkbox.clicked.connect(lambda:self.controller.set_use_mc_video(self.use_mc_video_checkbox.isChecked()))
        self.use_mc_video_checkbox.setDisabled(True)
        self.button_layout_2.addWidget(self.use_mc_video_checkbox)

        self.button_layout_2.addStretch()

        self.mc_progress_label = QLabel("")
        self.mc_progress_label.setStyleSheet("font-size: 10px; font-style: italic;")
        self.mc_progress_label.setMinimumWidth(200)
        self.mc_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.button_layout_2.addWidget(self.mc_progress_label)

        self.motion_correct_button = HoverButton('Motion Correct', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Perform motion correction on the video.")
        self.motion_correct_button.setIcon(QIcon("icons/accept_icon.png"))
        self.motion_correct_button.setIconSize(QSize(16,16))
        self.motion_correct_button.setStyleSheet('font-weight: bold;')
        self.motion_correct_button.clicked.connect(self.controller.motion_correct_video)
        self.button_layout_2.addWidget(self.motion_correct_button)
        
        self.accept_button = HoverButton('ROI Finding', None, self.parent_widget.statusBar())
        self.accept_button.setHoverMessage("Automatically find ROIs.")
        self.accept_button.setIcon(QIcon("icons/skip_icon.png"))
        self.accept_button.setIconSize(QSize(16,16))
        # self.accept_button.setMaximumWidth(100)
        self.accept_button.clicked.connect(self.controller.accept_motion_correction)
        self.button_layout_2.addWidget(self.accept_button)

    def preview_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/float(self.param_slider_multipliers["contrast"])

        self.controller.preview_contrast(contrast)

    def preview_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/float(self.param_slider_multipliers["gamma"])

        self.controller.preview_gamma(gamma)

    def motion_correction_started(self):
        self.mc_progress_label.setText("Motion correcting... 0%.")
        self.motion_correct_button.setText('Cancel')
        self.motion_correct_button.setHoverMessage("Stop motion correction calculation.")
        self.accept_button.setEnabled(False)

    def update_motion_correction_progress(self, percent):
        if percent == 100:
            self.mc_progress_label.setText("")
            self.motion_correct_button.setText('Motion Correct')
            self.motion_correct_button.setHoverMessage("Perform motion correction on the video.")
            self.accept_button.setEnabled(True)
        elif percent == -1:
            self.mc_progress_label.setText("")
            self.motion_correct_button.setText('Motion Correct')
            self.motion_correct_button.setHoverMessage("Perform motion correction on the video.")
        else:
            self.mc_progress_label.setText("Motion correcting... {}%.".format(int(percent)))

class ROIFindingWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "ROI Finding Parameters")

        self.controller = controller

        self.add_param_slider(label_name="Normalization Window Size", name="window_size", minimum=2, maximum=30, moved=self.update_param, multiplier=1, pressed=self.update_param, released=self.update_param, description="Size (in pixels) of the window used to normalize brightness across the image.", int_values=True)
        # self.add_param_slider(label_name="Soma Threshold", name="soma_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.controller.show_soma_threshold_image, released=self.update_param, description="Threshold for soma centers.", int_values=True)
        # self.add_param_slider(label_name="Neuropil Threshold", name="neuropil_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.controller.show_neuropil_mask, released=self.update_param, description="Threshold for neuropil.", int_values=True)
        self.add_param_slider(label_name="Background Threshold", name="background_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.update_param, released=self.update_param, description="Threshold for background.", int_values=True)
        # self.add_param_slider(label_name="Compactness", name="compactness", minimum=1, maximum=50, moved=self.update_param, multiplier=1, released=self.update_param, description="Compactness parameter (not currently used).", int_values=True)

        self.main_layout.addStretch()

        self.main_layout.addWidget(HLine())

        self.mask_button_widget = QWidget(self)
        self.mask_button_layout = QHBoxLayout(self.mask_button_widget)
        self.mask_button_layout.setContentsMargins(5, 0, 0, 0)
        self.mask_button_layout.setSpacing(5)
        self.main_layout.addWidget(self.mask_button_widget)

        self.invert_masks_checkbox = QCheckBox("Invert masks")
        self.invert_masks_checkbox.setObjectName("Invert masks")
        self.invert_masks_checkbox.setChecked(self.controller.params['invert_masks'])
        self.invert_masks_checkbox.clicked.connect(lambda:self.controller.set_invert_masks(self.invert_masks_checkbox.isChecked()))
        self.mask_button_layout.addWidget(self.invert_masks_checkbox)

        self.mask_button_layout.addStretch()

        self.draw_mask_button = HoverButton('Draw Mask', None, self.parent_widget.statusBar())
        self.draw_mask_button.setHoverMessage("Draw a mask.")
        self.draw_mask_button.setIcon(QIcon("icons/draw_icon.png"))
        self.draw_mask_button.setIconSize(QSize(16,16))
        # self.draw_mask_button.setMinimumWidth(150)
        self.draw_mask_button.clicked.connect(self.controller.draw_mask)
        self.mask_button_layout.addWidget(self.draw_mask_button)

        self.erase_selected_mask_button = HoverButton('Remove Mask', None, self.parent_widget.statusBar())
        self.erase_selected_mask_button.setHoverMessage("Erase the selected mask.")
        self.erase_selected_mask_button.setIcon(QIcon("icons/trash_icon.png"))
        self.erase_selected_mask_button.setIconSize(QSize(16,16))
        self.erase_selected_mask_button.clicked.connect(self.controller.erase_selected_mask)
        self.erase_selected_mask_button.setEnabled(False)
        self.mask_button_layout.addWidget(self.erase_selected_mask_button)

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.show_rois_checkbox = QCheckBox("Show ROIs")
        self.show_rois_checkbox.setObjectName("Show ROIs")
        self.show_rois_checkbox.setChecked(False)
        self.show_rois_checkbox.clicked.connect(lambda:self.controller.show_roi_image(self.show_rois_checkbox.isChecked()))
        self.show_rois_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.show_rois_checkbox)

        self.button_layout.addStretch()

        self.roi_finding_progress_label = QLabel("")
        self.roi_finding_progress_label.setStyleSheet("font-size: 10px; font-style: italic;")
        self.roi_finding_progress_label.setMinimumWidth(200)
        self.roi_finding_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.button_layout.addWidget(self.roi_finding_progress_label)

        self.motion_correct_button = HoverButton('Motion Correction', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Go back to motion correction.")
        self.motion_correct_button.setIcon(QIcon("icons/skip_back_icon.png"))
        self.motion_correct_button.setIconSize(QSize(16,16))
        self.motion_correct_button.clicked.connect(lambda:self.controller.show_motion_correction_params(switched_to=True))
        self.button_layout.addWidget(self.motion_correct_button)

        self.process_video_button = HoverButton('Find ROIs', None, self.parent_widget.statusBar())
        self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
        self.process_video_button.setIcon(QIcon("icons/accept_icon.png"))
        self.process_video_button.setIconSize(QSize(16,16))
        self.process_video_button.setStyleSheet('font-weight: bold;')
        self.process_video_button.clicked.connect(self.controller.find_rois)
        self.button_layout.addWidget(self.process_video_button)

        self.filter_rois_button = HoverButton('ROI Filtering', None, self.parent_widget.statusBar())
        self.filter_rois_button.setHoverMessage("Automatically and manually filter the found ROIs.")
        self.filter_rois_button.setIcon(QIcon("icons/skip_icon.png"))
        self.filter_rois_button.setIconSize(QSize(16,16))
        self.filter_rois_button.clicked.connect(self.controller.show_roi_filtering_params)
        # self.filter_rois_button.setDisabled(True)
        self.button_layout.addWidget(self.filter_rois_button)

    def roi_finding_started(self):
        self.roi_finding_progress_label.setText("Finding ROIs... 0%.")
        self.process_video_button.setText('Cancel')
        self.process_video_button.setHoverMessage("Stop finding ROIs.")
        self.filter_rois_button.setEnabled(False)
        self.motion_correct_button.setEnabled(False)

    def update_roi_finding_progress(self, percent):
        if percent == 100:
            self.roi_finding_progress_label.setText("")
            self.process_video_button.setText('Find ROIs')
            self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
            self.filter_rois_button.setEnabled(True)
        elif percent == -1:
            self.roi_finding_progress_label.setText("")
            self.process_video_button.setText('Find ROIs')
            self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
        else:
            self.roi_finding_progress_label.setText("Finding ROIs... {}%.".format(int(percent)))

class ROIFilteringWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "ROI Filtering Parameters")

        self.controller = controller

        self.add_param_slider(label_name="Minimum Area", name="min_area", minimum=1, maximum=500, moved=self.update_param, multiplier=1, released=self.update_param, description="Minimum ROI area.")
        self.add_param_slider(label_name="Maximum Area", name="max_area", minimum=1, maximum=500, moved=self.update_param, multiplier=1, released=self.update_param, description="Maximum ROI area.")
        self.add_param_slider(label_name="Minimum Circulature", name="min_circ", minimum=0, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Minimum ROI circulature.")
        self.add_param_slider(label_name="Maximum Circulature", name="max_circ", minimum=0, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Maximum ROI circulature.")
        self.add_param_slider(label_name="Minimum Correlation", name="min_correlation", minimum=0, maximum=1000, moved=self.update_param, multiplier=1000, released=self.update_param, description="Minimum mean pixel correlation of ROI.")

        self.main_layout.addStretch()
        
        self.main_layout.addWidget(HLine())

        self.roi_button_widget = QWidget(self)
        self.roi_button_layout = QHBoxLayout(self.roi_button_widget)
        self.roi_button_layout.setContentsMargins(5, 0, 0, 0)
        self.roi_button_layout.setSpacing(5)
        self.main_layout.addWidget(self.roi_button_widget)

        label = QLabel("Manual Controls")
        label.setStyleSheet(SUBTITLE_STYLESHEET)
        self.roi_button_layout.addWidget(label)

        self.roi_button_layout.addStretch()

        self.erase_rois_button = HoverButton('Eraser', None, self.parent_widget.statusBar())
        self.erase_rois_button.setHoverMessage("Manually remove ROIs using an eraser tool.")
        self.erase_rois_button.setIcon(QIcon("icons/eraser_icon.png"))
        self.erase_rois_button.setIconSize(QSize(16, 16))
        self.erase_rois_button.clicked.connect(self.controller.erase_rois)
        self.roi_button_layout.addWidget(self.erase_rois_button)

        self.undo_button = HoverButton('Undo', None, self.parent_widget.statusBar())
        self.undo_button.setHoverMessage("Undo the previous erase action.")
        self.undo_button.setIcon(QIcon("icons/undo_icon.png"))
        self.undo_button.setIconSize(QSize(16, 16))
        self.undo_button.clicked.connect(self.controller.undo_erase)
        self.roi_button_layout.addWidget(self.undo_button)

        self.reset_button = HoverButton('Reset', None, self.parent_widget.statusBar())
        self.reset_button.setHoverMessage("Reset erased ROIs.")
        self.reset_button.setIcon(QIcon("icons/reset_icon.png"))
        self.reset_button.setIconSize(QSize(16, 16))
        self.reset_button.clicked.connect(self.controller.reset_erase)
        self.roi_button_layout.addWidget(self.reset_button)

        self.draw_rois_button = HoverButton('Draw', None, self.parent_widget.statusBar())
        self.draw_rois_button.setHoverMessage("Manually draw circular ROIs.")
        self.draw_rois_button.setIcon(QIcon("icons/draw_icon.png"))
        self.draw_rois_button.setIconSize(QSize(16, 16))
        self.draw_rois_button.clicked.connect(self.controller.draw_rois)
        self.roi_button_layout.addWidget(self.draw_rois_button)

        self.roi_button_widget_2 = QWidget(self)
        self.roi_button_layout_2 = QHBoxLayout(self.roi_button_widget_2)
        self.roi_button_layout_2.setContentsMargins(5, 0, 0, 0)
        self.roi_button_layout_2.setSpacing(5)
        self.main_layout.addWidget(self.roi_button_widget_2)

        label = QLabel("Selected ROI")
        label.setStyleSheet(SUBTITLE_STYLESHEET)
        self.roi_button_layout_2.addWidget(label)

        self.roi_button_layout_2.addStretch()

        self.erase_selected_roi_button = HoverButton('Erase', None, self.parent_widget.statusBar())
        self.erase_selected_roi_button.setHoverMessage("Erase the selected ROI.")
        self.erase_selected_roi_button.setIcon(QIcon("icons/trash_icon.png"))
        self.erase_selected_roi_button.setIconSize(QSize(16, 16))
        self.erase_selected_roi_button.clicked.connect(self.controller.erase_selected_roi)
        self.erase_selected_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.erase_selected_roi_button)

        self.lock_roi_button = HoverButton('Lock', None, self.parent_widget.statusBar())
        self.lock_roi_button.setHoverMessage("Lock the currently selected ROI. This prevents it from being filtered out or erased.")
        self.lock_roi_button.setIcon(QIcon("icons/lock_icon.png"))
        self.lock_roi_button.setIconSize(QSize(16, 16))
        self.lock_roi_button.clicked.connect(self.controller.lock_roi)
        self.lock_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.lock_roi_button)

        self.shrink_roi_button = HoverButton('Shrink', None, self.parent_widget.statusBar())
        self.shrink_roi_button.setHoverMessage("Shrink the currently selected ROI.")
        self.shrink_roi_button.setIcon(QIcon("icons/shrink_icon.png"))
        self.shrink_roi_button.setIconSize(QSize(16, 16))
        self.shrink_roi_button.clicked.connect(self.controller.shrink_roi)
        self.shrink_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.shrink_roi_button)

        self.enlarge_roi_button = HoverButton('Enlarge', None, self.parent_widget.statusBar())
        self.enlarge_roi_button.setHoverMessage("Enlarge the currently selected ROI.")
        self.enlarge_roi_button.setIcon(QIcon("icons/enlarge_icon.png"))
        self.enlarge_roi_button.setIconSize(QSize(16, 16))
        self.enlarge_roi_button.clicked.connect(self.controller.enlarge_roi)
        self.enlarge_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.enlarge_roi_button)

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.show_rois_checkbox = QCheckBox("Show ROIs")
        self.show_rois_checkbox.setObjectName("Show ROIs")
        self.show_rois_checkbox.setChecked(False)
        self.show_rois_checkbox.clicked.connect(lambda:self.controller.show_roi_image(self.show_rois_checkbox.isChecked()))
        self.show_rois_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.show_rois_checkbox)

        self.button_layout.addStretch()

        self.motion_correct_button = HoverButton('Motion Correction', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Go back to motion correction.")
        self.motion_correct_button.setIcon(QIcon("icons/skip_back_icon.png"))
        self.motion_correct_button.setIconSize(QSize(16,16))
        self.motion_correct_button.clicked.connect(lambda:self.controller.show_motion_correction_params(switched_to=True))
        self.button_layout.addWidget(self.motion_correct_button)

        self.find_rois_button = HoverButton('ROI Finding', None, self.parent_widget.statusBar())
        self.find_rois_button.setHoverMessage("Go back to ROI segmentation.")
        self.find_rois_button.setIcon(QIcon("icons/skip_back_icon.png"))
        self.find_rois_button.setIconSize(QSize(16,16))
        self.find_rois_button.clicked.connect(self.controller.show_roi_finding_params)
        self.button_layout.addWidget(self.find_rois_button)

        self.filter_rois_button = HoverButton('Filter ROIs', None, self.parent_widget.statusBar())
        self.filter_rois_button.setHoverMessage("Automatically filter ROIs with the current parameters.")
        self.filter_rois_button.setIcon(QIcon("icons/accept_icon.png"))
        self.filter_rois_button.setIconSize(QSize(16,16))
        self.filter_rois_button.setStyleSheet('font-weight: bold;')
        self.filter_rois_button.clicked.connect(lambda:self.controller.filter_rois(self.controller.z, update_overlay=True))
        self.button_layout.addWidget(self.filter_rois_button)

    def roi_erasing_started(self):
        self.filter_rois_button.setEnabled(False)
        self.find_rois_button.setEnabled(False)
        self.motion_correct_button.setEnabled(False)
        self.enlarge_roi_button.setEnabled(False)
        self.shrink_roi_button.setEnabled(False)
        self.lock_roi_button.setEnabled(False)
        self.erase_selected_roi_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.param_widget.setEnabled(False)
        self.draw_rois_button.setEnabled(False)

    def roi_erasing_ended(self):
        self.filter_rois_button.setEnabled(True)
        self.find_rois_button.setEnabled(True)
        self.motion_correct_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.param_widget.setEnabled(True)
        self.draw_rois_button.setEnabled(True)

    def roi_drawing_started(self):
        self.filter_rois_button.setEnabled(False)
        self.find_rois_button.setEnabled(False)
        self.motion_correct_button.setEnabled(False)
        self.enlarge_roi_button.setEnabled(False)
        self.shrink_roi_button.setEnabled(False)
        self.lock_roi_button.setEnabled(False)
        self.erase_selected_roi_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.param_widget.setEnabled(False)
        self.erase_rois_button.setEnabled(False)

    def roi_drawing_ended(self):
        self.filter_rois_button.setEnabled(True)
        self.find_rois_button.setEnabled(True)
        self.motion_correct_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.param_widget.setEnabled(True)
        self.erase_rois_button.setEnabled(True)

class HoverCheckBox(QCheckBox):
    def __init__(self, text, parent=None, status_bar=None):
        QCheckBox.__init__(self, text, parent)
        self.setMouseTracking(True)

        self.status_bar = status_bar
        self.hover_message = ""

    def setHoverMessage(self, message):
        self.hover_message = message

    def enterEvent(self, event):
        self.previous_message = self.status_bar.currentMessage()
        self.status_bar.showMessage(self.hover_message)

    def leaveEvent(self, event):
        if self.status_bar.currentMessage() != "":
            self.status_bar.showMessage(self.previous_message)

class HoverButton(QPushButton):
    def __init__(self, text, parent=None, status_bar=None):
        QPushButton.__init__(self, text, parent)
        self.setMouseTracking(True)

        self.status_bar = status_bar
        self.hover_message = ""

    def setHoverMessage(self, message):
        self.hover_message = message

    def enterEvent(self, event):
        self.previous_message = self.status_bar.currentMessage()
        self.status_bar.showMessage(self.hover_message)

    def leaveEvent(self, event):
        if self.status_bar.currentMessage() != "":
            self.status_bar.showMessage(self.previous_message)

class HoverLabel(QLabel):
    def __init__(self, text, parent=None, status_bar=None):
        QLabel.__init__(self, text, parent)
        self.setMouseTracking(True)

        self.status_bar = status_bar
        self.hover_message = ""

    def setHoverMessage(self, message):
        self.hover_message = message

    def enterEvent(self, event):
        self.previous_message = self.status_bar.currentMessage()
        self.status_bar.showMessage(self.hover_message)

    def leaveEvent(self, event):
        if self.status_bar.currentMessage() != "":
            self.status_bar.showMessage(self.previous_message)

def HLine():
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    frame.setFrameShadow(QFrame.Plain)

    frame.setStyleSheet("color: #ccc;")

    return frame