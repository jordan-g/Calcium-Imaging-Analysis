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
import matplotlib.pyplot as plt
import cv2

n_colors = 20
cmap = utilities.get_cmap(n_colors)

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

class DatasetEditingWindow(QMainWindow):
    def __init__(self, controller):
        QMainWindow.__init__(self)

        self.controller = controller

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
        scroll_area = QScrollArea(self)
        scroll_area.setFrameStyle(0)
        scroll_area.setWidget(group_box)
        scroll_area.setWidgetResizable(True)
        # scroll_area.setFixedHeight(400)
        self.main_layout.addWidget(scroll_area)

        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        self.main_layout.addWidget(button_widget)

        button_layout.addStretch()

        save_dataset_button = QPushButton("Save Dataset")
        save_dataset_button.setIcon(QIcon("icons/save_icon.png"))
        save_dataset_button.setIconSize(QSize(13, 16))
        save_dataset_button.setStyleSheet('font-weight: bold;')
        save_dataset_button.clicked.connect(self.save_dataset)
        button_layout.addWidget(save_dataset_button)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set window title bar buttons
        if pyqt_version == 5:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowFullscreenButtonHint)
        else:
            self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.image_list    = []
        self.widget_list   = []
        self.left_buttons  = []
        self.right_buttons = []

        self.dataset_filename = "zebrafish_gcamp_dataset.h5"

        self.refresh(reload_dataset=True)

    def save_dataset(self):
        kept_rois = []

        for i in range(len(self.image_list)):
            if self.left_buttons[i].isChecked() or self.right_buttons[i].isChecked():
                kept_rois.append(i)

        self.images, self.labels = self.images[kept_rois], self.labels[kept_rois]

        utilities.save_dataset(self.images, self.labels, filename=self.dataset_filename)
        
        self.refresh(reload_dataset=True)

    def refresh(self, reload_dataset=True):
        if reload_dataset:
            self.images, self.labels = utilities.load_dataset(filename=self.dataset_filename)

        for i in range(len(self.left_buttons)):
            self.left_buttons[i].disconnect()
            self.right_buttons[i].disconnect()

        for i in range(len(self.form_layouts)):
            for j in range(self.form_layouts[i].rowCount()-1, -1, -1):
                self.form_layouts[i].removeRow(j)

        self.image_list = []
        self.widget_list = []
        self.left_buttons = []
        self.right_buttons = []

        print(self.images.shape)

        for i in range(self.images.shape[0]):
            col = i % 5
            form_layout = self.form_layouts[col]

            image = np.repeat(self.images[i, :, :, 0][:, :, np.newaxis], 3, axis=-1)

            maximum = np.amax(self.images[i, :, :, -1])

            mask = (self.images[i, :, :, -1] > 0).copy()
            
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

            color = cmap(i % n_colors)[:3]
            color = [255*color[0], 255*color[1], 255*color[2]]

            overlay = np.zeros((image.shape[0], image.shape[1], 4)).astype(np.uint8)
            overlay[mask, :-1] = color
            overlay[mask, -1] = 255.0*self.images[i, mask, -1]/maximum

            image = (255*image/np.amax(image)).astype(np.uint8)
            image = utilities.blend_transparent(image, overlay)

            image = np.transpose(image, axes=(1, 0, 2))

            label = QLabel()
            qimage = QImage(image.copy(), image.shape[1], image.shape[0], 3*50, QImage.Format_RGB888)                                                                                                                                                                 
            pixmap = QPixmap(qimage)
            label.setPixmap(pixmap)

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
            
            left_button_func = self.make_left_button_clicked(i)
            right_button_func = self.make_right_button_clicked(i)

            if self.labels[i, 0] == 1:
                left_button.setChecked(True)
                left_button_func()
            else:
                right_button.setChecked(True)
                right_button_func()

            left_button.clicked.connect(left_button_func)
            right_button.clicked.connect(right_button_func)

            form_layout.addRow(self.image_list[i], self.widget_list[i])

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