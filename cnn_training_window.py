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
import numpy as np
import pyqtgraph as pg
import utilities

from dataset_editing_window import *

# set styles of title and subtitle labels
TITLE_STYLESHEET           = "font-size: 16px; font-weight: bold;"
SUBTITLE_STYLESHEET        = "font-size: 14px; font-weight: bold;"
ROUNDED_STYLESHEET_DARK    = "QLineEdit { background-color: rgba(0, 0, 0, 0.3); border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.5); padding: 2px };"
ROUNDED_STYLESHEET_LIGHT   = "QLineEdit { background-color: rgba(255, 255, 255, 1); border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.2); padding: 2px; }"
LIST_STYLESHEET_DARK       = "QListWidget { background-color: rgba(0, 0, 0, 0.3); border-radius: 5px; border: 1px solid rgba(0, 0, 0, 0.5); padding: 2px };"
LIST_STYLESHEET_LIGHT      = "QListWidget { background-color: rgba(255, 255, 255, 1); border-radius: 5px; border: 1px solid rgba(0, 0, 0, 0.2); padding: 2px; }"
LIST_ITEM_STYLESHEET_LIGHT = "QListWidget { background-color: rgba(255, 255, 255, 1); }"
STATUSBAR_STYLESHEET_LIGHT = "background-color: rgba(255, 255, 255, 0.3); border-top: 1px solid rgba(0, 0, 0, 0.2); font-size: 12px; font-style: italic;"
STATUSBAR_STYLESHEET_DARK  = "background-color: rgba(255, 255, 255, 0.1); border-top: 1px solid rgba(0, 0, 0, 0.5); font-size: 12px; font-style: italic;"
rounded_stylesheet         = ROUNDED_STYLESHEET_LIGHT
statusbar_stylesheet       = STATUSBAR_STYLESHEET_LIGHT
SHOWING_VIDEO_COLOR_LIGHT  = QColor(255, 220, 0, 60)
SHOWING_VIDEO_COLOR_DARK   = QColor(255, 220, 0, 30)
showing_video_color        = SHOWING_VIDEO_COLOR_LIGHT
SHOW_VIDEO_BUTTON_DISABLED_STYLESHEET_DARK  = "QPushButton{border: none; background-image:url(icons/play_icon_disabled_inverted.png);} QPushButton:hover{background-image:url(icons/play_icon_inverted.png);}"
SHOW_VIDEO_BUTTON_DISABLED_STYLESHEET_LIGHT = "QPushButton{border: none; background-image:url(icons/play_icon_disabled.png);} QPushButton:hover{background-image:url(icons/play_icon.png);}"
SHOW_VIDEO_BUTTON_ENABLED_STYLESHEET_DARK  = "QPushButton{border: none; background-image:url(icons/play_icon_enabled_inverted.png);} QPushButton:hover{background-image:url(icons/play_icon_enabled_inverted.png);}"
SHOW_VIDEO_BUTTON_ENABLED_STYLESHEET_LIGHT = "QPushButton{border: none; background-image:url(icons/play_icon_enabled.png);} QPushButton:hover{background-image:url(icons/play_icon_enabled.png);}"
show_video_button_disabled_stylesheet = SHOW_VIDEO_BUTTON_DISABLED_STYLESHEET_LIGHT
show_video_button_enabled_stylesheet  = SHOW_VIDEO_BUTTON_ENABLED_STYLESHEET_LIGHT
VIDEO_LABEL_SELECTED_COLOR_LIGHT = "QLabel{color: white;}"
VIDEO_LABEL_SELECTED_COLOR_DARK = "QLabel{color: white;}"
VIDEO_LABEL_UNSELECTED_COLOR_LIGHT = "QLabel{color: black;}"
VIDEO_LABEL_UNSELECTED_COLOR_DARK = "QLabel{color: white;}"
video_label_selected_color   = VIDEO_LABEL_SELECTED_COLOR_LIGHT
video_label_unselected_color = VIDEO_LABEL_UNSELECTED_COLOR_LIGHT

categoryFont = QFont()
categoryFont.setBold(True)

class CNNTrainingWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        self.controller = controller

        self.dataset_editing_window = DatasetEditingWindow(controller)

        # set window title
        self.setWindowTitle("Label ROIs and Train CNN")

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 10)
        self.main_layout.setSpacing(0)

        self.resize(1200, 1000)

        group_box = QGroupBox()
        # group_box.setFlat(True)
        widget = QWidget(self)
        hbox_layout = QHBoxLayout(widget)

        self.form_layouts = []

        for i in range(5):
            widget = QWidget()
            form_layout = QFormLayout(widget)
            hbox_layout.addWidget(widget)
            self.form_layouts.append(form_layout)

        group_box.setLayout(hbox_layout)
        scroll_area = QScrollArea()
        scroll_area.setFrameStyle(0)
        scroll_area.setWidget(group_box)
        scroll_area.setWidgetResizable(True)
        # scroll_area.setFixedHeight(400)
        self.main_layout.addWidget(scroll_area)

        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        self.main_layout.addWidget(button_widget)

        self.show_rois_checkbox = QCheckBox("Show spatial footprints")
        self.show_rois_checkbox.setChecked(True)
        self.show_rois_checkbox.clicked.connect(self.toggle_show_rois)
        button_layout.addWidget(self.show_rois_checkbox)

        button_layout.addStretch()

        auto_label_button = QPushButton("Auto-Label")
        auto_label_button.setIcon(QIcon("icons/action_icon.png"))
        auto_label_button.setIconSize(QSize(13, 16))
        auto_label_button.clicked.connect(self.auto_label)
        button_layout.addWidget(auto_label_button)

        reset_button = QPushButton("Reset CNN")
        reset_button.setIcon(QIcon("icons/action_icon.png"))
        reset_button.setIconSize(QSize(13, 16))
        reset_button.clicked.connect(self.controller.reset_cnn)
        button_layout.addWidget(reset_button)

        test_button = QPushButton("Test CNN")
        test_button.setIcon(QIcon("icons/action_icon.png"))
        test_button.setIconSize(QSize(13, 16))
        test_button.clicked.connect(self.test_cnn)
        button_layout.addWidget(test_button)

        train_button = QPushButton("Train CNN")
        train_button.setIcon(QIcon("icons/action_icon.png"))
        train_button.setIconSize(QSize(13, 16))
        train_button.clicked.connect(self.train_cnn)
        button_layout.addWidget(train_button)

        add_to_dataset_button = QPushButton("Add to Dataset")
        add_to_dataset_button.setIcon(QIcon("icons/action_icon.png"))
        add_to_dataset_button.setIconSize(QSize(13, 16))
        add_to_dataset_button.clicked.connect(self.add_to_dataset)
        button_layout.addWidget(add_to_dataset_button)

        edit_dataset_button = QPushButton("Edit Dataset...")
        edit_dataset_button.setIcon(QIcon("icons/action_icon.png"))
        edit_dataset_button.setIconSize(QSize(13, 16))
        edit_dataset_button.clicked.connect(self.edit_dataset)
        button_layout.addWidget(edit_dataset_button)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set window title bar buttons
        if pyqt_version == 5:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)
        else:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.image_list = []
        self.widget_list = []
        self.left_buttons = []
        self.right_buttons = []

        self.roi_images = None
        self.mean_images = None

    def edit_dataset(self):
        self.dataset_editing_window.show()

    def add_to_dataset(self):
        positive_rois = []
        negative_rois = []

        for i in range(len(self.image_list)):
            if self.left_buttons[i].isChecked():
                positive_rois.append(i)
            elif self.right_buttons[i].isChecked():
                negative_rois.append(i)

        utilities.add_data_to_dataset(self.controller.roi_spatial_footprints(), self.controller.adjusted_mean_images[self.controller.z], positive_rois, negative_rois, self.controller.controller.params['half_size'])

        self.dataset_editing_window.refresh()

    def refresh(self, roi_images, mean_images, show_rois=True):
        self.roi_images = roi_images
        self.mean_images = mean_images

        self.show_rois_checkbox.setChecked(show_rois)

        for i in range(len(self.left_buttons)):
            self.left_buttons[i].disconnect()
            self.right_buttons[i].disconnect()

        for i in range(len(self.form_layouts)):
            for j in range(self.form_layouts[i].rowCount(), -1, -1):
                self.form_layouts[i].removeRow(j)

        self.image_list = []
        self.widget_list = []
        self.left_buttons = []
        self.right_buttons = []

        print(np.amin(self.mean_images), np.amax(self.mean_images))

        for i in range(roi_images.shape[0]):
            col = i % 5
            form_layout = self.form_layouts[col]

            image = (255*mean_images[i]).astype(np.uint8)
            if self.show_rois_checkbox.isChecked():
                image = utilities.blend_transparent(image, roi_images[i])

            image = np.transpose(image, axes=(1, 0, 2))

            label = QLabel()
            qimage = QImage(image.copy(), image.shape[1], image.shape[0], 3*50, QImage.Format_RGB888)                                                                                                                                                                 
            pixmap = QPixmap(qimage)
            label.setPixmap(pixmap)

            label.mouseReleaseEvent = self.make_image_clicked(i)

            widget = QWidget()
            layout = QHBoxLayout(widget)
            left_button = QPushButton()
            left_button.setCheckable(True)
            left_button.setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 3px;")
            left_button.setFixedSize(QSize(50, 50))
            left_button.setIcon(QIcon("icons/checkmark_icon.png"))
            left_button.setIconSize(QSize(50, 50))
            right_button = QPushButton()
            right_button.setCheckable(True)
            right_button.setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 3px;")
            right_button.setFixedSize(QSize(50, 50))
            right_button.setIcon(QIcon("icons/cross_icon.png"))
            right_button.setIconSize(QSize(50, 50))
            layout.addWidget(left_button)
            layout.addWidget(right_button)
            
            self.image_list.append(label)
            self.widget_list.append(widget)
            self.left_buttons.append(left_button)
            self.right_buttons.append(right_button)
            
            left_button.clicked.connect(self.make_left_button_clicked(i))
            right_button.clicked.connect(self.make_right_button_clicked(i))

            form_layout.addRow(self.image_list[i], self.widget_list[i])

    def make_image_clicked(self, i):
        def image_clicked(event):
            print("ROI {} clicked".format(i))
            self.controller.select_single_roi(i)

        return image_clicked

    def make_right_button_clicked(self, i):
        def right_button_clicked():
            if self.right_buttons[i].isChecked():
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")
                self.right_buttons[i].setStyleSheet("background-color: rgba(255, 0, 0, 0.5); border-radius: 5px;")
            else:
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")
                self.right_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")

            self.left_buttons[i].setChecked(False)

        return right_button_clicked

    def make_left_button_clicked(self, i):
        def left_button_clicked():
            if self.left_buttons[i].isChecked():
                self.right_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 255, 0, 0.5); border-radius: 5px;")
            else:
                self.right_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")

            self.right_buttons[i].setChecked(False)

        return left_button_clicked

    def toggle_show_rois(self):
        show_rois = self.show_rois_checkbox.isChecked()

        self.controller.preview_window.set_show_rois(show_rois)

    def set_show_rois(self, show_rois):
        self.show_rois_checkbox.setChecked(show_rois)

        for i in range(len(self.image_list)):
            image = (255*self.mean_images[i]).astype(np.uint8)
            if self.show_rois_checkbox.isChecked():
                image = utilities.blend_transparent(image, self.roi_images[i])

            qimage = QImage(image.copy(), image.shape[1], image.shape[0], 3*50, QImage.Format_RGB888)                                                                                                                                                                 
            pixmap = QPixmap(qimage)
            self.image_list[i].setPixmap(pixmap)

    def train_cnn(self):
        positive_rois = []
        negative_rois = []

        for i in range(len(self.image_list)):
            if self.left_buttons[i].isChecked():
                positive_rois.append(i)
            elif self.right_buttons[i].isChecked():
                negative_rois.append(i)

        print("{} positive, {} negative ROIs.".format(len(positive_rois), len(negative_rois)))

        self.controller.train_cnn_on_data(positive_rois, negative_rois)

    def test_cnn(self):
        self.controller.test_cnn_on_data()

    def update_with_predictions(self, predictions):
        for i in range(len(self.image_list)):
            # print(predictions[i, 0] + predictions[i, 1])

            if self.left_buttons[i].isChecked():
                left_stylesheet = "background-color: rgba(0, 255, 0, 0.5); border-radius: 5px;"
            else:
                left_stylesheet = "background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;"

            if self.right_buttons[i].isChecked():
                right_stylesheet = "background-color: rgba(255, 0, 0, 0.5); border-radius: 5px;"
            else:
                right_stylesheet = "background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;"

            # if predictions[i, 0] > predictions[i, 1]:
            # else:
            # self.left_buttons[i].setStyleSheet("background-color: rgba(0, 255, 255, {:.1f}); border-radius: 5px;".format(100*predictions[i, 0]))
            # self.right_buttons[i].setStyleSheet("background-color: rgba(255, 0, 255, {:.1f}); border-radius: 5px;".format(100*predictions[i, 1]))

            if predictions[i, 0] > self.controller.controller.params['cnn_accept_threshold']:
                self.left_buttons[i].setStyleSheet(left_stylesheet + "border: 3px solid rgba(255, 255, 255, 1)")
            else:
                self.left_buttons[i].setStyleSheet(left_stylesheet)
            
            if predictions[i, 0] < self.controller.controller.params['cnn_reject_threshold']:
                self.right_buttons[i].setStyleSheet(right_stylesheet + "border: 3px solid rgba(255, 255, 255, 1)")
            else:
                self.right_buttons[i].setStyleSheet(right_stylesheet)

    def auto_label(self):
        for i in range(len(self.image_list)):
            if i in self.controller.removed_rois():
                self.right_buttons[i].setChecked(True)
                self.right_buttons[i].setStyleSheet("background-color: rgba(255, 0, 0, 0.5); border-radius: 5px;")
                
                self.left_buttons[i].setChecked(False)
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")
            else:
                self.right_buttons[i].setChecked(False)
                self.right_buttons[i].setStyleSheet("background-color: rgba(0, 0, 0, 0.1); border-radius: 5px;")

                self.left_buttons[i].setChecked(True)
                self.left_buttons[i].setStyleSheet("background-color: rgba(0, 255, 0, 0.5); border-radius: 5px;")