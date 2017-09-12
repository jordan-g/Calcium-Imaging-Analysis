# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

TITLE_STYLESHEET = "font-size: 18px; font-weight: bold;"

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Automatic ROI Segmentation Parameters")

        # set initial position
        self.setGeometry(0, 32, 10, 10)

        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        self.statusBar().setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border-top: 1px solid rgba(0, 0, 0, 0.1); font-size: 10px; font-style: italic;")
        self.statusBar().showMessage("To begin, open a video file. TIFF and NPY files are supported.")

        # create main buttons
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.open_file_button = HoverButton('Open...', None, self.statusBar())
        self.open_file_button.setHoverMessage("Open a video file for processing.")
        self.open_file_button.setStyleSheet('font-weight: bold;')
        self.open_file_button.setMaximumWidth(100)
        self.open_file_button.clicked.connect(self.controller.select_and_open_video)
        self.button_layout.addWidget(self.open_file_button)
        self.button_layout.setAlignment(self.open_file_button, Qt.AlignLeft)

        self.main_layout.addWidget(HLine())

        # create stacked widget
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.stacked_widget.setDisabled(True)
        self.main_layout.addWidget(self.stacked_widget)

        # create motion correction widget
        self.motion_correction_widget = MotionCorrectionWidget(self, self.controller)
        self.stacked_widget.addWidget(self.motion_correction_widget)

        # create watershed widget
        self.watershed_widget = WatershedWidget(self, self.controller)
        self.stacked_widget.addWidget(self.watershed_widget)

        # create ROI pruning widget
        self.roi_pruning_widget = ROIPruningWidget(self, self.controller)
        self.stacked_widget.addWidget(self.roi_pruning_widget)

        # set window title bar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

        self.show()

    def closeEvent(self, event):
        self.controller.close_all()

class ParamWidget(QWidget):
    def __init__(self, parent_widget, controller, title):
        QWidget.__init__(self)

        self.parent_widget = parent_widget

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.title_widget = QWidget(self)
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setContentsMargins(10, 10, 0, 0)
        self.main_layout.addWidget(self.title_widget)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(TITLE_STYLESHEET)
        self.title_layout.addWidget(self.title_label)
        self.main_layout.setAlignment(self.title_widget, Qt.AlignTop)
        
        self.param_widget = QWidget(self)
        self.param_layout = QGridLayout(self.param_widget)
        self.param_layout.setContentsMargins(10, 10, 10, 10)
        self.param_layout.setSpacing(5)
        self.main_layout.addWidget(self.param_widget)

        self.param_sliders            = {}
        self.param_slider_multipliers = {}
        self.param_textboxes          = {}

    def add_param_slider(self, label_name, name, minimum, maximum, moved, multiplier=1, pressed=None, released=None, description=None, int_values=False):
        row = len(self.param_sliders.keys())

        if released == self.update_param:
            released = lambda:self.update_param(name)
        if moved == self.update_param:
            moved = lambda:self.update_param(name)
        if pressed == self.update_param:
            pressed = lambda:self.update_param(name)

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
        textbox.setStyleSheet("border-radius: 3px; border: 1px solid #ddd; padding: 2px;")
        textbox.setAlignment(Qt.AlignHCenter)
        textbox.setObjectName(name)
        textbox.setFixedWidth(40)
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
        if int_values:
            value = int(float(textbox.text()))
        else:
            value = float(textbox.text())
        
        slider.setValue(value*multiplier)
        textbox.setText(str(value))

    def update_param(self, param):
        value = self.param_sliders[param].sliderPosition()/float(self.param_slider_multipliers[param])

        self.controller.update_param(param, value)

class MotionCorrectionWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "Motion Correction Parameters")

        self.controller = controller.motion_correction_controller

        self.add_param_slider(label_name="Gamma", name="gamma", minimum=1, maximum=500, moved=self.preview_gamma, multiplier=100, released=self.update_param, description="Gamma of the video preview.")
        self.add_param_slider(label_name="Contrast", name="contrast", minimum=1, maximum=500, moved=self.preview_contrast, multiplier=100, released=self.update_param, description="Contrast of the video preview.")
        self.add_param_slider(label_name="FPS", name="fps", minimum=1, maximum=60, moved=self.update_param, released=self.update_param, description="Frames per second of the video preview.", int_values=True)
        self.add_param_slider(label_name="Maximum Shift", name="max_shift", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Maximum shift (in pixels) allowed for motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Stride", name="patch_stride", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Stride length (in pixels) of each patch used in motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Overlap", name="patch_overlap", minimum=1, maximum=100, moved=self.update_param, released=self.update_param, description="Overlap (in pixels) of patches used in motion correction.", int_values=True)
        
        self.main_layout.addStretch()

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.process_video_button = HoverButton('Motion Correct', None, self.parent_widget.statusBar())
        self.process_video_button.setHoverMessage("Perform motion correction on the video.")
        self.process_video_button.setMaximumWidth(150)
        self.process_video_button.clicked.connect(self.controller.process_video)
        self.button_layout.addWidget(self.process_video_button)

        self.play_motion_corrected_video_checkbox = QCheckBox("Show Motion-Corrected Video")
        self.play_motion_corrected_video_checkbox.setObjectName("Show Motion-Corrected Video")
        self.play_motion_corrected_video_checkbox.setChecked(False)
        self.play_motion_corrected_video_checkbox.clicked.connect(lambda:self.controller.play_motion_corrected_video(self.play_motion_corrected_video_checkbox.isChecked()))
        self.play_motion_corrected_video_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.play_motion_corrected_video_checkbox)

        self.button_layout.addStretch()

        self.skip_button = HoverButton('\u2192 Skip', None, self.parent_widget.statusBar())
        self.skip_button.setHoverMessage("Skip motion correction.")
        self.skip_button.setMaximumWidth(100)
        self.skip_button.clicked.connect(self.controller.skip)
        self.button_layout.addWidget(self.skip_button)
        
        self.accept_button = HoverButton('\u2192 Accept', None, self.parent_widget.statusBar())
        self.accept_button.setHoverMessage("Accept the motion-corrected video and move to the next step.")
        self.accept_button.setStyleSheet('font-weight: bold;')
        self.accept_button.setMaximumWidth(100)
        self.accept_button.clicked.connect(self.controller.accept)
        self.accept_button.setEnabled(False)
        self.button_layout.addWidget(self.accept_button)

    def preview_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/float(self.param_slider_multipliers["contrast"])

        self.controller.preview_contrast(contrast)

    def preview_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/float(self.param_slider_multipliers["gamma"])

        self.controller.preview_gamma(gamma)

class WatershedWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "ROI Segmentation Parameters")

        self.controller = controller.watershed_controller

        self.add_param_slider(label_name="Gamma", name="gamma", minimum=1, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Gamma of the video preview.")
        self.add_param_slider(label_name="Contrast", name="contrast", minimum=1, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Contrast of the video preview.")
        self.add_param_slider(label_name="Window Size", name="window_size", minimum=2, maximum=30, moved=self.update_param, multiplier=1, pressed=self.controller.show_normalized_image, released=self.update_param, description="Size (in pixels) of the window used to normalize brightness across the image.", int_values=True)
        self.add_param_slider(label_name="Soma Threshold", name="soma_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.controller.show_soma_threshold_image, released=self.update_param, description="Threshold for soma centers.", int_values=True)
        self.add_param_slider(label_name="Neuropil Threshold", name="neuropil_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.controller.show_neuropil_mask, released=self.update_param, description="Threshold for neuropil.", int_values=True)
        self.add_param_slider(label_name="Background Threshold", name="background_threshold", minimum=1, maximum=255, moved=self.update_param, multiplier=1, pressed=self.controller.show_background_mask, released=self.update_param, description="Threshold for background.", int_values=True)
        self.add_param_slider(label_name="Compactness", name="compactness", minimum=1, maximum=50, moved=self.update_param, multiplier=1, released=self.update_param, description="Compactness parameter (not currently used).", int_values=True)

        self.main_layout.addStretch()

        self.main_layout.addWidget(HLine())

        self.mask_button_widget = QWidget(self)
        self.mask_button_layout = QHBoxLayout(self.mask_button_widget)
        self.mask_button_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_button_layout.setSpacing(5)
        self.main_layout.addWidget(self.mask_button_widget)

        self.draw_mask_button = HoverButton('Draw Mask', None, self.parent_widget.statusBar())
        self.draw_mask_button.setHoverMessage("Draw a mask.")
        self.draw_mask_button.setMinimumWidth(150)
        self.draw_mask_button.clicked.connect(self.controller.draw_mask)
        self.mask_button_layout.addWidget(self.draw_mask_button)

        self.mask_button_layout.addStretch()

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.process_video_button = HoverButton('Find ROIs', None, self.parent_widget.statusBar())
        self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
        self.process_video_button.clicked.connect(self.controller.process_video)
        self.button_layout.addWidget(self.process_video_button)

        self.show_watershed_checkbox = QCheckBox("Show ROIs")
        self.show_watershed_checkbox.setObjectName("Show ROIs")
        self.show_watershed_checkbox.setChecked(False)
        self.show_watershed_checkbox.clicked.connect(lambda:self.controller.show_watershed_image(self.show_watershed_checkbox.isChecked()))
        self.show_watershed_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.show_watershed_checkbox)

        self.button_layout.addStretch()

        self.motion_correct_button = HoverButton('\u2190 Motion Correction', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Go back to motion correction.")
        self.motion_correct_button.clicked.connect(self.controller.motion_correct)
        self.button_layout.addWidget(self.motion_correct_button)

        self.prune_rois_button = HoverButton('\u2192 Prune ROIs', None, self.parent_widget.statusBar())
        self.prune_rois_button.setHoverMessage("Automatically and manually prune the found ROIs.")
        self.prune_rois_button.clicked.connect(self.controller.prune_rois)
        self.prune_rois_button.setStyleSheet('font-weight: bold;')
        self.prune_rois_button.setDisabled(True)
        self.button_layout.addWidget(self.prune_rois_button)

class ROIPruningWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "ROI Pruning Parameters")

        self.controller = controller.roi_pruning_controller

        self.add_param_slider(label_name="Gamma", name="gamma", minimum=1, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Gamma of the video preview.")
        self.add_param_slider(label_name="Contrast", name="contrast", minimum=1, maximum=500, moved=self.update_param, multiplier=100, released=self.update_param, description="Contrast of the video preview.")
        self.add_param_slider(label_name="Minimum Area", name="min_area", minimum=1, maximum=500, moved=self.update_param, multiplier=1, released=self.update_param, description="Minimum ROI area.")
        self.add_param_slider(label_name="Maximum Area", name="max_area", minimum=1, maximum=500, moved=self.update_param, multiplier=1, released=self.update_param, description="Maximum ROI area.")

        self.main_layout.addStretch()
        
        self.main_layout.addWidget(HLine())

        self.roi_button_widget = QWidget(self)
        self.roi_button_layout = QHBoxLayout(self.roi_button_widget)
        self.roi_button_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_button_layout.setSpacing(5)
        self.main_layout.addWidget(self.roi_button_widget)

        self.erase_rois_button = HoverButton('Erase ROIs', None, self.parent_widget.statusBar())
        self.erase_rois_button.setHoverMessage("Manually remove ROIs using an eraser tool.")
        self.erase_rois_button.clicked.connect(self.controller.erase_rois)
        self.roi_button_layout.addWidget(self.erase_rois_button)

        self.undo_button = HoverButton('Undo', None, self.parent_widget.statusBar())
        self.undo_button.setHoverMessage("Undo the previous erase action.")
        self.undo_button.clicked.connect(self.controller.undo_erase)
        self.roi_button_layout.addWidget(self.undo_button)

        self.reset_button = HoverButton('Reset', None, self.parent_widget.statusBar())
        self.reset_button.setHoverMessage("Reset erased ROIs.")
        self.reset_button.clicked.connect(self.controller.reset_erase)
        self.reset_button.setEnabled(False)
        self.roi_button_layout.addWidget(self.reset_button)

        self.roi_button_layout.addStretch()

        self.keep_roi_button = HoverButton('Keep ROI', None, self.parent_widget.statusBar())
        self.keep_roi_button.setHoverMessage("Keep the current ROI.")
        self.keep_roi_button.clicked.connect(self.controller.keep_roi)
        self.roi_button_layout.addWidget(self.keep_roi_button)

        self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.prune_rois_button = HoverButton('Prune ROIs', None, self.parent_widget.statusBar())
        self.prune_rois_button.setHoverMessage("Automatically prune ROIs with the current parameters.")
        self.prune_rois_button.clicked.connect(self.controller.prune_rois)
        self.button_layout.addWidget(self.prune_rois_button)

        self.show_watershed_checkbox = QCheckBox("Show ROIs")
        self.show_watershed_checkbox.setObjectName("Show ROIs")
        self.show_watershed_checkbox.setChecked(False)
        self.show_watershed_checkbox.clicked.connect(lambda:self.controller.show_watershed_image(self.show_watershed_checkbox.isChecked()))
        self.show_watershed_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.show_watershed_checkbox)

        self.button_layout.addStretch()

        self.motion_correct_button = HoverButton('\u2190 Motion Correction', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Go back to motion correction.")
        self.motion_correct_button.clicked.connect(self.controller.motion_correct)
        self.button_layout.addWidget(self.motion_correct_button)

        self.watershed_button = HoverButton('\u2190 ROI Segmentation', None, self.parent_widget.statusBar())
        self.watershed_button.clicked.connect(self.controller.watershed)
        self.watershed_button.setToolTip("Go back to ROI segmentation.")
        self.button_layout.addWidget(self.watershed_button)

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
        self.status_bar.showMessage(self.previous_message)

def HLine():
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    frame.setFrameShadow(QFrame.Plain)

    frame.setStyleSheet("color: #ccc;")

    return frame