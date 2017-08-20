# import the Qt library
try:
    from PyQt4.QtCore import Qt, QThread
    from PyQt4.QtGui import qRgb, QImage, QPixmap, QIcon, QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout
except:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtGui import qRgb, QImage, QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QAction, QMessageBox, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QSlider, QFileDialog, QGridLayout

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Parameters")

        # set initial position
        self.setGeometry(0, 32, 10, 10)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("font-size: 12px;")

        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        self.param_sliders = {}
        self.param_value_labels = {}

        self.add_param_slider(label_name="Gamma", name="gamma", min=1, max=500, multiplier=100, moved=self.update_gamma, row=0)
        self.add_param_slider(label_name="Contrast", name="contrast", min=1, max=500, multiplier=100, moved=self.update_contrast, row=1)
        self.add_param_slider(label_name="Window Size", name="window_size", min=2, max=30, multiplier=1, moved=self.update_window_size, row=2, pressed=self.controller.show_normalized_image, released=self.controller.show_adjusted_image)
        self.add_param_slider(label_name="Soma Threshold", name="soma_threshold", min=1, max=255, multiplier=1, moved=self.update_soma_threshold, row=3, pressed=self.controller.show_soma_threshold_image, released=self.controller.show_adjusted_image)
        self.add_param_slider(label_name="Background Threshold", name="background_threshold", min=1, max=255, multiplier=1, moved=self.update_background_threshold, row=4, pressed=self.controller.show_background_mask, released=self.controller.show_adjusted_image)
        self.add_param_slider(label_name="Compactness", name="compactness", min=1, max=50, multiplier=1, moved=self.update_compactness, row=5)

        self.process_video_button = QPushButton('Process')
        self.process_video_button.clicked.connect(self.controller.process_video)
        self.process_video_button.setToolTip("Process the video.")
        self.main_layout.addWidget(self.process_video_button, 6, 1)

        self.draw_mask_button = QPushButton('Draw Mask')
        self.draw_mask_button.clicked.connect(self.controller.draw_mask)
        self.draw_mask_button.setToolTip("Draw a mask.")
        self.main_layout.addWidget(self.draw_mask_button, 6, 2)

        self.show_watershed_checkbox = QCheckBox("Show Watershed")
        self.show_watershed_checkbox.setObjectName("Show Watershed")
        self.show_watershed_checkbox.setChecked(False)
        self.show_watershed_checkbox.clicked.connect(lambda:self.controller.show_watershed_image(self.show_watershed_checkbox.isChecked()))
        self.show_watershed_checkbox.setDisabled(True)
        self.main_layout.addWidget(self.show_watershed_checkbox, 7, 1)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set window title bar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

    def add_param_slider(self, label_name, name, min, max, multiplier, moved, row, pressed=None, released=None):
        label = QLabel("{}:".format(label_name))
        self.main_layout.addWidget(label, row, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(300)
        slider.setObjectName(name)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(self.controller.params[name]*multiplier)
        slider.sliderMoved.connect(moved)
        if pressed:
            slider.sliderPressed.connect(pressed)
        if released:
            slider.sliderReleased.connect(released)
        self.main_layout.addWidget(slider, row, 1)

        value_label = QLabel(str(self.controller.params[name]))
        self.main_layout.addWidget(value_label, row, 2)

        self.param_sliders[name] = slider
        self.param_value_labels[name] = value_label

    def update_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/100

        self.controller.update_contrast(contrast)

        self.param_value_labels["contrast"].setText(str(self.controller.params['contrast']))

    def update_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/100

        self.controller.update_gamma(gamma)

        self.param_value_labels["gamma"].setText(str(self.controller.params['gamma']))

    def update_background_threshold(self):
        background_threshold = self.param_sliders["background_threshold"].sliderPosition()

        self.controller.update_background_threshold(background_threshold)

        self.param_value_labels["background_threshold"].setText(str(self.controller.params['background_threshold']))

    def update_window_size(self):
        window_size = self.param_sliders["window_size"].sliderPosition()

        self.controller.update_window_size(window_size)

        self.param_value_labels["window_size"].setText(str(self.controller.params['window_size']))

    def update_soma_threshold(self):
        soma_threshold = self.param_sliders["soma_threshold"].sliderPosition()

        self.controller.update_soma_threshold(soma_threshold)

        self.param_value_labels["soma_threshold"].setText(str(self.controller.params['soma_threshold']))

    def update_compactness(self):
        compactness = self.param_sliders["compactness"].sliderPosition()

        self.controller.update_compactness(compactness)

        self.param_value_labels["compactness"].setText(str(self.controller.params['compactness']))

    def closeEvent(self, event):
        self.controller.close_all()