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

        self.main_layout = QVBoxLayout(self.main_widget)

        self.param_widget = QWidget(self)
        self.param_layout = QGridLayout(self.param_widget)
        self.param_layout.setContentsMargins(5, 5, 5, 5)
        self.param_layout.setSpacing(5)
        self.main_layout.addWidget(self.param_widget)

        self.param_sliders = {}
        self.param_value_labels = {}

        self.add_param_slider(label_name="Gamma", name="gamma", min=1, max=500, multiplier=100, moved=self.preview_gamma, released=self.update_gamma, row=0)
        self.add_param_slider(label_name="Contrast", name="contrast", min=1, max=500, multiplier=100, moved=self.preview_contrast, released=self.update_contrast, row=1)

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 5, 5, 5)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.process_video_button = QPushButton('Process')
        self.process_video_button.clicked.connect(self.controller.process_video)
        self.process_video_button.setToolTip("Process the video.")
        self.button_layout.addWidget(self.process_video_button)

        self.accept_button = QPushButton('Accept')
        self.accept_button.clicked.connect(self.controller.accept)
        self.accept_button.setToolTip("Accept the motion-corrected video.")
        self.accept_button.setEnabled(False)
        self.button_layout.addWidget(self.accept_button)

        self.skip_button = QPushButton('Skip')
        self.skip_button.clicked.connect(self.controller.skip)
        self.skip_button.setToolTip("Skip motion correction.")
        self.button_layout.addWidget(self.skip_button)

        self.accept_2_button = QPushButton('Accept and CNMF')
        self.accept_2_button.clicked.connect(self.controller.accept_cnmf)
        self.accept_2_button.setToolTip("Accept the motion-corrected video.")
        self.accept_2_button.setEnabled(False)
        self.button_layout.addWidget(self.accept_2_button)

        self.skip_2_button = QPushButton('Skip and CNMF')
        self.skip_2_button.clicked.connect(self.controller.skip_cnmf)
        self.skip_2_button.setToolTip("Skip motion correction.")
        self.button_layout.addWidget(self.skip_2_button)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set window title bar buttons
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)

    def add_param_slider(self, label_name, name, min, max, multiplier, moved, row, pressed=None, released=None):
        label = QLabel("{}:".format(label_name))
        self.param_layout.addWidget(label, row, 0)

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
        self.param_layout.addWidget(slider, row, 1)

        value_label = QLabel(str(self.controller.params[name]))
        self.param_layout.addWidget(value_label, row, 2)

        self.param_sliders[name] = slider
        self.param_value_labels[name] = value_label

    def preview_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/100

        self.controller.preview_contrast(contrast)

        self.param_value_labels["contrast"].setText(str(self.controller.params['contrast']))

    def preview_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/100

        self.controller.preview_gamma(gamma)

        self.param_value_labels["gamma"].setText(str(self.controller.params['gamma']))

    def update_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/100

        self.controller.update_contrast(contrast)

        self.param_value_labels["contrast"].setText(str(self.controller.params['contrast']))

    def update_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/100

        self.controller.update_gamma(gamma)

        self.param_value_labels["gamma"].setText(str(self.controller.params['gamma']))

    def closeEvent(self, event):
        self.controller.close_all()
