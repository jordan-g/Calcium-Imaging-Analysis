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
import numpy as np

# set styles of title and subtitle labels
TITLE_STYLESHEET    = "font-size: 16px; font-weight: bold;"
SUBTITLE_STYLESHEET = "font-size: 14px; font-weight: bold;"
ROUNDED_STYLESHEET_DARK   = "background-color: rgba(255, 255, 255, 0.2); border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.5); padding: 2px;"
ROUNDED_STYLESHEET_LIGHT  = "background-color: rgba(255, 255, 255, 1); border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.2); padding: 2px;"
rounded_stylesheet = ROUNDED_STYLESHEET_LIGHT

categoryFont = QFont()
categoryFont.setBold(True)

class ParamWindow(QMainWindow):
    def __init__(self, controller):
        global rounded_stylesheet
        QMainWindow.__init__(self)

        self.bg_color = (self.palette().color(self.backgroundRole()).red(), self.palette().color(self.backgroundRole()).green(), self.palette().color(self.backgroundRole()).blue())
        if self.bg_color[0] < 100:
            rounded_stylesheet  = ROUNDED_STYLESHEET_DARK
        else:
            rounded_stylesheet  = ROUNDED_STYLESHEET_LIGHT

        # set controller
        self.controller = controller

        # set window title
        self.setWindowTitle("Automatic ROI Segmentation")

        # set initial position
        self.setGeometry(0, 32, 10, 10)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(0)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        # set up the status bar
        self.statusBar().setStyleSheet("background-color: rgba(255, 255, 255, 0.3); border-top: 1px solid rgba(0, 0, 0, 0.2); font-size: 10px; font-style: italic;")
        self.statusBar().showMessage("To begin, open one or more video files. Only TIFF files are currently supported.")

        self.main_param_widget = MainParamWidget(self, self.controller)
        self.main_layout.addWidget(self.main_param_widget, 0, 0)

        # self.main_layout.addWidget(HLine(), 2, 0)

        # create stacked widget
        # self.stacked_widget = QStackedWidget(self)
        # self.stacked_widget.setContentsMargins(5, 5, 5, 5)
        # self.main_layout.addWidget(self.stacked_widget, 3, 0)

        # create loading widget
        self.loading_widget = LoadingWidget(self, self.controller)
        # self.stacked_widget.addWidget(self.loading_widget)

        # create motion correction widget
        self.motion_correction_widget = MotionCorrectionWidget(self, self.controller)
        # self.stacked_widget.addWidget(self.motion_correction_widget)

        # create ROI finding widget
        self.roi_finding_widget = ROIFindingWidget(self, self.controller)
        # self.stacked_widget.addWidget(self.roi_finding_widget)

        # create ROI filtering widget
        self.roi_filtering_widget = ROIFilteringWidget(self, self.controller)
        # self.stacked_widget.addWidget(self.roi_filtering_widget)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(self.tab_widget, 3, 0)
        self.tab_widget.addTab(self.loading_widget, "1. Video Selection")
        self.tab_widget.addTab(self.motion_correction_widget, "2. Motion Correction")
        self.tab_widget.addTab(self.roi_finding_widget, "3. ROI Finding")
        self.tab_widget.addTab(self.roi_filtering_widget, "4. Refinement && Saving")
        self.tab_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        for i in range(self.tab_widget.count()):
            if i == 0:
                self.tab_widget.widget(i).setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            else:
                self.tab_widget.widget(i).setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Ignored)

        # self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.currentChanged.connect(self.tab_selected)

        self.videos_list_widget = QWidget(self)
        self.videos_list_layout = QHBoxLayout(self.videos_list_widget)
        self.videos_list_layout.setContentsMargins(5, 0, 5, 0)
        self.main_layout.addWidget(self.videos_list_widget, 4, 0)
        self.videos_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

        # checkbox_widget = QWidget()
        # checkbox_layout = QHBoxLayout(checkbox_widget)
        # checkbox_widget.setContentsMargins(0, 0, 0, 0)
        # self.use_multiprocessing_checkbox = HoverCheckBox("Use multiprocessing", None, self.statusBar())
        # self.use_multiprocessing_checkbox.setHoverMessage("Use multiple cores to speed up computations.")
        # self.use_multiprocessing_checkbox.setChecked(True)
        # self.use_multiprocessing_checkbox.clicked.connect(lambda:self.controller.set_use_multiprocessing(self.use_multiprocessing_checkbox.isChecked()))
        # checkbox_layout.addWidget(self.use_multiprocessing_checkbox)

        # self.main_layout.addWidget(checkbox_widget, 5, 0)
        
        self.videos_list = QListWidget(self)
        self.videos_list.setStyleSheet(rounded_stylesheet)
        # self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.videos_list.itemSelectionChanged.connect(self.item_selected)
        self.videos_list_layout.addWidget(self.videos_list)
        self.videos_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.videos_list.installEventFilter(self)

        self.delete_shortcut = QShortcut(QKeySequence('Delete'), self.videos_list)
        self.delete_shortcut.activated.connect(self.remove_selected_items)

        self.group_nums = []

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

    def rois_loaded(self):
        # enable ROI filtering tab
        self.param_window.tab_widget.setTabEnabled(3, True)

        # enable showing and saving ROIs
        self.param_window.show_rois_action.setEnabled(True)
        self.param_window.save_rois_action.setEnabled(True)

        # show ROIs
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(True)

    def toggle_show_rois(self):
        show_rois = self.show_rois_checkbox.isChecked()
        self.controller.show_roi_image(show_rois)

        self.show_rois_action.setChecked(show_rois)
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(show_rois)
        self.roi_filtering_param_widget.show_rois_checkbox.setChecked(show_rois)

    def tab_selected(self):
        index = self.tab_widget.currentIndex()

        for i in range(self.tab_widget.count()):
            if i == index:
                self.tab_widget.widget(i).setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            else:
                self.tab_widget.widget(i).setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Ignored)

        if index == 0:
            self.controller.show_video_loading_params()
        elif index == 1:
            # self.motion_correction_widget.videos_list.clear()
            # for video_path in self.controller.controller.video_paths:
            #     self.motion_correction_widget.videos_list.addItem(os.path.basename(video_path))
            self.controller.show_motion_correction_params()
        elif index == 2:
            # self.roi_finding_widget.videos_list.clear()
            # for video_path in self.controller.controller.video_paths:
            #     self.roi_finding_widget.videos_list.addItem(os.path.basename(video_path))
            self.controller.show_roi_finding_params()
        elif index == 3:
            # self.roi_filtering_widget.videos_list.clear()
            # for video_path in self.controller.controller.video_paths:
            #     self.roi_filtering_widget.videos_list.addItem(os.path.basename(video_path))
            self.controller.show_roi_filtering_params()

    def single_roi_selected(self, discarded=False):
        if not discarded:
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
            self.erase_rois_action.setEnabled(True)
            self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(False)
            self.unerase_rois_action.setEnabled(False)
        else:
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
            self.erase_rois_action.setEnabled(False)
            self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
            self.unerase_rois_action.setEnabled(True)
        self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
        self.merge_rois_action.setEnabled(False)

    def multiple_rois_selected(self, discarded=False, merge_enabled=True):
        if not discarded:
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
            self.erase_rois_action.setEnabled(True)
            self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(False)
            self.unerase_rois_action.setEnabled(False)
            self.roi_filtering_param_widget.merge_rois_button.setEnabled(merge_enabled)
            self.merge_rois_action.setEnabled(merge_enabled)
        else:
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
            self.erase_rois_action.setEnabled(False)
            self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
            self.unerase_rois_action.setEnabled(True)
            self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
            self.merge_rois_action.setEnabled(False)

    def no_rois_selected(self):
        self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
        self.erase_rois_action.setEnabled(False)
        self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(False)
        self.unerase_rois_action.setEnabled(False)
        self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
        self.merge_rois_action.setEnabled(False)

    def eventFilter(self, sender, event):
        if (event.type() == QEvent.ChildRemoved):
            self.on_order_changed()
        return False # don't actually interrupt anything

    def on_order_changed(self):
        # do magic things with our new-found knowledge
        print("Order changed.")

        item = self.videos_list.item(0)
        if item.font() != categoryFont:
            self.videos_list.takeItem(0)
            self.videos_list.insertItem(1, item)

        groups = []
        old_indices = []

        group_num = -1

        for i in range(self.videos_list.count()):
            item = self.videos_list.item(i)

            if item.font() == categoryFont:
                group_num = int(item.text().split(" ")[-1])-1
            else:
                groups.append(group_num)

                old_index = self.controller.controller.video_paths.index(item.text())
                old_indices.append(old_index)

        self.controller.videos_rearranged(old_indices, groups)

    def add_group(self, group_num):
        item = QListWidgetItem("Group {}".format(group_num+1))
        item.setFont(categoryFont)
        if self.bg_color[0] < 100:
            color = QColor(20, 30, 40, 120)
        else:
            color = QColor(36, 87, 201, 60)
        item.setBackground(QBrush(color, Qt.SolidPattern))
        item.setFlags(item.flags() & ~Qt.ItemIsDragEnabled)
        self.videos_list.addItem(item)
        self.group_nums.append(group_num)

    def add_new_group(self):
        if len(self.group_nums) > 0:
            group_num = np.amax(self.group_nums)+1
        else:
            group_num = 0

        self.add_group(group_num)

    def remove_selected_items(self):
        selected_items = self.videos_list.selectedItems()

        self.controller.remove_videos_at_indices([ self.controller.controller.video_paths.index(item.text()) for item in selected_items ])

        for i in range(len(selected_items)-1, -1, -1):
            self.videos_list.takeItem(self.videos_list.row(selected_items[i]))

    def remove_items(self, items):
        self.controller.remove_videos_at_indices([ self.controller.controller.video_paths.index(item.text()) for item in items ])

        for i in range(len(items)-1, -1, -1):
            self.videos_list.takeItem(self.videos_list.row(items[i]))

    def remove_selected_group(self):
        selected_items = self.videos_list.selectedItems()

        self.videos_list.takeItem(self.videos_list.row(selected_items[0]))

        group_num = int(selected_items[0].text().split(" ")[-1])-1
        
        print("Removing group {}.".format(group_num+1))

        if group_num in self.group_nums:
            index = self.group_nums.index(group_num)
            del self.group_nums[index]

        items_to_remove = []

        for i in range(self.videos_list.count()):
            item = self.videos_list.item(i)

            if item.font() != categoryFont:
                index = self.controller.controller.video_paths.index(item.text())
                if self.controller.controller.groups[index] == group_num:
                    items_to_remove.append(item)

        if len(items_to_remove) > 0:
            self.remove_items(items_to_remove)

    def item_selected(self):
        tab_index = self.tab_widget.currentIndex()
        
        selected_items = self.videos_list.selectedItems()

        if len(selected_items) > 0:
            if selected_items[0].font() == categoryFont:
                group_num = int(selected_items[0].text().split(" ")[-1])-1

                print("Group {} clicked.".format(group_num+1))

                self.loading_widget.remove_videos_button.setDisabled(True)
                self.loading_widget.remove_group_button.setDisabled(False)
                if tab_index == 0:
                    self.remove_videos_action.setEnabled(False)
                    self.remove_group_action.setEnabled(True)
            else:
                self.loading_widget.remove_videos_button.setDisabled(False)
                self.loading_widget.remove_group_button.setDisabled(True)
                if tab_index == 0:
                    self.remove_videos_action.setEnabled(True)
                    self.remove_group_action.setEnabled(False)

                # index = self.videos_list.selectedIndexes()[0].row()
                index = self.controller.controller.video_paths.index(selected_items[0].text())
                self.controller.video_selected(index)
        else:
            self.loading_widget.remove_videos_button.setDisabled(True)
            self.loading_widget.remove_group_button.setDisabled(True)
            if tab_index == 0:
                self.remove_videos_action.setEnabled(False)
                self.remove_group_action.setEnabled(False)

            index = None

    def set_initial_state(self):
        # disable buttons, widgets & menu items
        self.main_param_widget.setDisabled(True)
        self.tab_widget.setCurrentIndex(0)
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)
        self.tab_widget.setTabEnabled(3, False)
        # self.stacked_widget.setDisabled(True)
        self.show_rois_action.setEnabled(False)
        # self.save_roi_image_action.setEnabled(False)

    def create_menus(self):
        self.add_videos_action = QAction('Add Videos...', self)
        self.add_videos_action.setShortcut('Ctrl+O')
        self.add_videos_action.setStatusTip('Add video files for processing.')
        self.add_videos_action.triggered.connect(self.controller.import_videos)

        self.remove_videos_action = QAction('Remove Video', self)
        self.remove_videos_action.setShortcut('Delete')
        self.remove_videos_action.setStatusTip('Remove the selected video.')
        self.remove_videos_action.setEnabled(False)
        self.remove_videos_action.triggered.connect(self.remove_selected_items)

        self.remove_group_action = QAction('Remove Group', self)
        self.remove_group_action.setShortcut('Delete')
        self.remove_group_action.setStatusTip('Remove the selected group.')
        self.remove_group_action.setEnabled(False)
        self.remove_group_action.triggered.connect(self.remove_selected_group)

        self.show_rois_action = QAction('Show ROIs', self, checkable=True)
        self.show_rois_action.setShortcut('R')
        self.show_rois_action.setStatusTip('Toggle showing the ROIs.')
        self.show_rois_action.triggered.connect(lambda:self.controller.show_roi_image(self.show_rois_action.isChecked()))
        self.show_rois_action.setEnabled(False)
        self.show_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.load_rois_action = QAction('Load ROIs...', self)
        self.load_rois_action.setShortcut('Alt+O')
        self.load_rois_action.setStatusTip('Load ROIs from a file.')
        self.load_rois_action.triggered.connect(self.controller.load_rois)
        self.load_rois_action.setShortcutContext(Qt.ApplicationShortcut)
        # self.load_rois_action.setEnabled(False)
        # self.load_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.save_rois_action = QAction('Save ROIs...', self)
        self.save_rois_action.setShortcut('Alt+S')
        self.save_rois_action.setStatusTip('Save the current ROIs.')
        self.save_rois_action.triggered.connect(self.controller.save_rois)
        self.save_rois_action.setEnabled(False)
        self.save_rois_action.setShortcutContext(Qt.ApplicationShortcut)
        # self.save_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.erase_rois_action = QAction('Erase Selected ROIs', self)
        self.erase_rois_action.setShortcut('Delete')
        self.erase_rois_action.setStatusTip('Erase the selected ROIs.')
        self.erase_rois_action.triggered.connect(self.controller.erase_selected_rois)
        self.erase_rois_action.setEnabled(False)
        self.erase_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.merge_rois_action = QAction('Merge Selected ROIs', self)
        self.merge_rois_action.setShortcut('M')
        self.merge_rois_action.setStatusTip('Merge the selected ROIs.')
        self.merge_rois_action.triggered.connect(self.controller.merge_selected_rois)
        self.merge_rois_action.setEnabled(False)
        self.merge_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.trace_rois_action = QAction('Plot Traces', self)
        self.trace_rois_action.setShortcut('T')
        self.trace_rois_action.setStatusTip('Plot traces for the selected ROIs.')
        self.trace_rois_action.triggered.connect(self.controller.update_trace_plot)
        self.trace_rois_action.setEnabled(False)
        self.trace_rois_action.setShortcutContext(Qt.ApplicationShortcut)

        self.load_tail_angles_action = QAction('Load Tail Angle Trace', self)
        self.load_tail_angles_action.setShortcut('Ctrl+T')
        self.load_tail_angles_action.setStatusTip('Load tail angle CSV.')
        self.load_tail_angles_action.triggered.connect(self.controller.load_tail_angles)
        self.load_tail_angles_action.setEnabled(True)
        self.load_tail_angles_action.setShortcutContext(Qt.ApplicationShortcut)

        # self.save_roi_image_action = QAction('Save ROI Image...', self)
        # self.save_roi_image_action.setShortcut('Ctrl+Alt+S')
        # self.save_roi_image_action.setStatusTip('Save an image of the current ROIs.')
        # self.save_roi_image_action.triggered.connect(self.controller.save_roi_image)
        # self.save_roi_image_action.setEnabled(False)

        # create menu bar
        menubar = self.menuBar()

        # add menu items
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(self.add_videos_action)
        file_menu.addAction(self.remove_videos_action)
        file_menu.addAction(self.load_tail_angles_action)
        # file_menu.addAction(self.save_rois_action)
        # file_menu.addAction(self.save_roi_image_action)
        # file_menu.addSeparator()

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(self.show_rois_action)

        rois_menu = menubar.addMenu('&ROIs')
        rois_menu.addAction(self.load_rois_action)
        rois_menu.addAction(self.save_rois_action)
        rois_menu.addAction(self.erase_rois_action)
        rois_menu.addAction(self.merge_rois_action)
        rois_menu.addAction(self.trace_rois_action)

    def video_opened(self, max_z, z):
        # self.stacked_widget.setDisabled(False)
        self.statusBar().showMessage("")
        self.main_param_widget.param_sliders["z"].setMaximum(max_z)
        self.main_param_widget.param_sliders["z"].setValue(z)
        self.main_param_widget.param_textboxes["z"].setText(str(z))
        # self.loading_widget.save_mc_video_button.setEnabled(False)
        self.motion_correction_widget.use_mc_video_checkbox.setChecked(False)
        self.motion_correction_widget.use_mc_video_checkbox.setDisabled(True)

    def videos_imported(self, video_paths):
        # self.videos_imported(video_paths)
        self.add_new_group()
        for video_path in video_paths:
            self.videos_list.addItem(video_path)

        # print(self.controller.controller.video_groups)

        self.main_param_widget.setDisabled(False)
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(2, True)
        self.tab_widget.setTabEnabled(3, False)
        self.tab_widget.setCurrentIndex(0)
        # self.stacked_widget.setDisabled(False)

    def process_videos_started(self):
        self.loading_widget.process_videos_started()

    def roi_finding_ended(self):
        self.roi_finding_param_widget.show_rois_checkbox.setDisabled(False)
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(True)
        self.roi_finding_param_widget.process_video_button.setEnabled(True)
        self.show_rois_action.setDisabled(False)
        self.show_rois_action.setChecked(True)
        self.save_rois_action.setDisabled(False)
        self.tab_widget.setTabEnabled(0, True)
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(3, True)

    def motion_correction_ended(self):
        self.motion_correction_param_widget.use_mc_video_checkbox.setEnabled(True)
        self.motion_correction_param_widget.use_mc_video_checkbox.setChecked(True)

    def roi_erasing_started(self):
        self.main_param_widget.setEnabled(False)
        self.loading_widget.setEnabled(False)

        self.roi_filtering_widget.roi_erasing_started()

    def roi_erasing_ended(self):
        self.main_param_widget.setEnabled(True)
        self.loading_widget.setEnabled(True)

        self.roi_filtering_widget.roi_erasing_ended()

    def roi_drawing_started(self):
        self.main_param_widget.setEnabled(False)
        self.loading_widget.setEnabled(False)

        self.roi_filtering_widget.roi_drawing_started()

    def roi_drawing_ended(self):
        self.main_param_widget.setEnabled(True)
        self.loading_widget.setEnabled(True)

        self.roi_filtering_widget.roi_drawing_ended()

    def update_process_videos_progress(self, percent):
        pass
        # self.loading_widget.update_process_videos_progress(percent)

    def update_motion_correction_progress(self, percent):
        self.motion_correction_widget.update_motion_correction_progress(percent)

    def update_roi_finding_progress(self, percent):
        self.roi_finding_widget.update_roi_finding_progress(percent)

    def closeEvent(self, event):
        self.controller.close_all()

class LoadingWidget(QWidget):
    def __init__(self, parent_widget, controller):
        QWidget.__init__(self)

        self.parent_widget = parent_widget

        self.controller = controller

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.title_widget = QWidget(self)
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setContentsMargins(5, 5, 0, 0)
        self.main_layout.addWidget(self.title_widget)

        self.title_label = QLabel("Videos to Process")
        self.title_label.setStyleSheet(TITLE_STYLESHEET)
        self.title_layout.addWidget(self.title_label)
        self.main_layout.setAlignment(self.title_widget, Qt.AlignTop)

        # create main buttons
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 5, 5, 5)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.open_file_button = HoverButton('Add Videos...', None, self.parent_widget.statusBar())
        self.open_file_button.setHoverMessage("Add video files for processing.")
        self.open_file_button.setStyleSheet('font-weight: bold;')
        self.open_file_button.setIcon(QIcon("icons/open_file_icon.png"))
        self.open_file_button.setIconSize(QSize(16,16))
        self.open_file_button.clicked.connect(self.controller.import_videos)
        self.button_layout.addWidget(self.open_file_button)

        self.remove_videos_button = HoverButton('Remove', None, self.parent_widget.statusBar())
        self.remove_videos_button.setHoverMessage("Remove the selected video.")
        self.remove_videos_button.setIcon(QIcon("icons/trash_icon.png"))
        self.remove_videos_button.setIconSize(QSize(16,16))
        self.remove_videos_button.setDisabled(True)
        self.remove_videos_button.clicked.connect(self.parent_widget.remove_selected_items)
        self.button_layout.addWidget(self.remove_videos_button)

        self.button_layout.addStretch()

        self.remove_group_button = HoverButton('Remove Group', None, self.parent_widget.statusBar())
        self.remove_group_button.setHoverMessage("Remove the selected group.")
        self.remove_group_button.setIcon(QIcon("icons/remove_group_icon.png"))
        self.remove_group_button.setIconSize(QSize(16,16))
        self.remove_group_button.setDisabled(True)
        self.remove_group_button.clicked.connect(self.parent_widget.remove_selected_group)
        self.button_layout.addWidget(self.remove_group_button)

        self.add_group_button = HoverButton('Add Group', None, self.parent_widget.statusBar())
        self.add_group_button.setHoverMessage("Add a new group.")
        self.add_group_button.setIcon(QIcon("icons/add_group_icon.png"))
        self.add_group_button.setIconSize(QSize(16,16))
        # self.add_group_button.setDisabled(True)
        self.add_group_button.clicked.connect(self.parent_widget.add_new_group)
        self.button_layout.addWidget(self.add_group_button)

        # create secondary buttons
        self.button_widget_2 = QWidget(self)
        self.button_layout_2 = QHBoxLayout(self.button_widget_2)
        self.button_layout_2.setContentsMargins(10, 0, 0, 0)
        self.button_layout_2.setSpacing(15)
        self.main_layout.addWidget(self.button_widget_2)

        self.button_layout_2.addStretch()

        self.use_multiprocessing_checkbox = HoverCheckBox("Use multiprocessing", None, self.parent_widget.statusBar())
        self.use_multiprocessing_checkbox.setHoverMessage("Use multiple cores to speed up computations.")
        self.use_multiprocessing_checkbox.setChecked(True)
        self.use_multiprocessing_checkbox.clicked.connect(lambda:self.controller.set_use_multiprocessing(self.use_multiprocessing_checkbox.isChecked()))
        self.button_layout_2.addWidget(self.use_multiprocessing_checkbox)

        self.process_all_button = HoverButton('Motion Correct && Find ROIs...', None, self.parent_widget.statusBar())
        self.process_all_button.setHoverMessage("Motion correct and find ROIs for all videos using the current parameters.")
        self.process_all_button.setStyleSheet('font-weight: bold;')
        self.process_all_button.setIcon(QIcon("icons/skip_icon.png"))
        self.process_all_button.setIconSize(QSize(16,16))
        self.button_layout_2.addWidget(self.process_all_button)

class ParamWidget(QWidget):
    def __init__(self, parent_widget, controller, title, stylesheet=TITLE_STYLESHEET):
        QWidget.__init__(self)

        self.parent_widget = parent_widget

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        if len(title) > 0:
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
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setSpacing(0)
        self.main_layout.addWidget(self.param_widget)

        self.param_sliders            = {}
        self.param_slider_multipliers = {}
        self.param_textboxes          = {}

    def add_param_slider(self, label_name, name, minimum, maximum, moved, num, multiplier=1, pressed=None, released=None, description=None, int_values=False):
        row = np.floor(num/2)
        col = num % 2

        if released == self.update_param:
            released = lambda:self.update_param(name, int_values=int_values)
        if moved == self.update_param:
            moved = lambda:self.update_param(name, int_values=int_values)
        if pressed == self.update_param:
            pressed = lambda:self.update_param(name, int_values=int_values)

        widget = QWidget(self.param_widget)
        layout = QHBoxLayout(widget)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.param_layout.addWidget(widget, row, col)
        label = HoverLabel("{}:".format(label_name), None, self.parent_widget.statusBar())
        label.setHoverMessage(description)
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        # slider.setFixedWidth(300)
        slider.setObjectName(name)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(self.controller.controller.params[name]*multiplier)
        slider.sliderMoved.connect(moved)
        if pressed:
            slider.sliderPressed.connect(pressed)
        if released:
            slider.sliderReleased.connect(released)
        layout.addWidget(slider)

        slider.sliderMoved.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier, int_values))
        slider.sliderReleased.connect(lambda:self.update_textbox_from_slider(slider, textbox, multiplier, int_values))

        # make textbox & add to layout
        textbox = QLineEdit()
        textbox.setStyleSheet(rounded_stylesheet)
        textbox.setAlignment(Qt.AlignHCenter)
        textbox.setObjectName(name)
        textbox.setFixedWidth(60)
        textbox.setFixedHeight(20)
        textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        textbox.editingFinished.connect(lambda:self.update_slider_from_textbox(slider, textbox, multiplier, int_values))
        textbox.editingFinished.connect(released)
        self.update_textbox_from_slider(slider, textbox, multiplier, int_values)
        layout.addWidget(textbox)

        self.param_sliders[name]            = slider
        self.param_slider_multipliers[name] = multiplier
        self.param_textboxes[name]          = textbox

    def add_param_checkbox(self, label_name, name, clicked, num, description=None):
        row = np.floor(num/2)
        col = num % 2

        widget = QWidget(self.param_widget)
        layout = QHBoxLayout(widget)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.param_layout.addWidget(widget, row, col)
        label = HoverLabel("{}:".format(label_name), None, self.parent_widget.statusBar())
        label.setHoverMessage(description)
        layout.addWidget(label)

        layout.addStretch()

        checkbox = HoverCheckBox("", None, self.parent_widget.statusBar())
        checkbox.setHoverMessage(description)
        checkbox.setChecked(self.controller.controller.params[name])
        widget.setContentsMargins(0, 0, 5, 0)
        checkbox.clicked.connect(lambda:clicked(checkbox.isChecked()))
        layout.addWidget(checkbox)

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

        self.add_param_slider(label_name="Gamma", name="gamma", minimum=1, maximum=500, moved=self.preview_gamma, num=0, multiplier=100, released=self.update_param, description="Gamma of the video preview.")
        self.add_param_slider(label_name="Contrast", name="contrast", minimum=1, maximum=500, moved=self.preview_contrast, num=1, multiplier=100, released=self.update_param, description="Contrast of the video preview.")
        self.add_param_slider(label_name="FPS", name="fps", minimum=1, maximum=60, moved=self.update_param, num=2, released=self.update_param, description="Frames per second of the video preview.", int_values=True)
        self.add_param_slider(label_name="Z", name="z", minimum=0, maximum=0, moved=self.update_param, num=3, released=self.update_param, description="Z plane of the video preview.", int_values=True)

        self.setFixedHeight(120)
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

        self.main_layout.addStretch()

        # self.videos_list_widget = QWidget(self)
        # self.videos_list_layout = QHBoxLayout(self.videos_list_widget)
        # self.videos_list_layout.setContentsMargins(5, 0, 5, 0)
        # self.main_layout.addWidget(self.videos_list_widget)
        
        # self.videos_list = QListWidget(self)
        # self.videos_list.setStyleSheet(rounded_stylesheet)
        # # self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.videos_list.itemSelectionChanged.connect(self.item_selected)
        # self.videos_list_layout.addWidget(self.videos_list)

        self.add_param_slider(label_name="Maximum Shift", name="max_shift", minimum=1, maximum=100, moved=self.update_param, num=0, released=self.update_param, description="Maximum shift (in pixels) allowed for motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Stride", name="patch_stride", minimum=1, maximum=100, moved=self.update_param, num=1, released=self.update_param, description="Stride length (in pixels) of each patch used in motion correction.", int_values=True)
        self.add_param_slider(label_name="Patch Overlap", name="patch_overlap", minimum=1, maximum=100, moved=self.update_param, num=2, released=self.update_param, description="Overlap (in pixels) of patches used in motion correction.", int_values=True)
        

        # self.main_layout.addWidget(HLine())

        # self.button_widget = QWidget(self)
        # self.button_layout = QHBoxLayout(self.button_widget)
        # self.button_layout.setContentsMargins(5, 0, 0, 0)
        # self.button_layout.setSpacing(5)
        # self.main_layout.addWidget(self.button_widget)

        # self.mc_current_z_checkbox = HoverCheckBox("Motion-correct only this z plane", None, self.parent_widget.statusBar())
        # self.mc_current_z_checkbox.setHoverMessage("Apply motion correction only to the current z plane. Useful for quick troubleshooting.")
        # self.mc_current_z_checkbox.setChecked(False)
        # self.mc_current_z_checkbox.clicked.connect(lambda:self.controller.set_mc_current_z(self.mc_current_z_checkbox.isChecked()))
        # self.button_layout.addWidget(self.mc_current_z_checkbox)

        self.button_widget_2 = QWidget(self)
        self.button_layout_2 = QHBoxLayout(self.button_widget_2)
        self.button_layout_2.setContentsMargins(10, 10, 10, 10)
        self.button_layout_2.setSpacing(15)
        self.main_layout.addWidget(self.button_widget_2)

        self.use_mc_video_checkbox = HoverCheckBox("Use motion-corrected videos", None, self.parent_widget.statusBar())
        self.use_mc_video_checkbox.setHoverMessage("Use the motion-corrected videos for finding ROIs.")
        self.use_mc_video_checkbox.setChecked(False)
        self.use_mc_video_checkbox.clicked.connect(lambda:self.controller.set_use_mc_video(self.use_mc_video_checkbox.isChecked()))
        self.use_mc_video_checkbox.setDisabled(True)
        self.button_layout_2.addWidget(self.use_mc_video_checkbox)

        self.button_layout_2.addStretch()

        self.use_multiprocessing_checkbox = HoverCheckBox("Use multiprocessing", None, self.parent_widget.statusBar())
        self.use_multiprocessing_checkbox.setHoverMessage("Use multiple cores to speed up computations.")
        self.use_multiprocessing_checkbox.setChecked(True)
        self.use_multiprocessing_checkbox.clicked.connect(lambda:self.controller.set_use_multiprocessing(self.use_multiprocessing_checkbox.isChecked()))
        self.button_layout_2.addWidget(self.use_multiprocessing_checkbox)

        # self.mc_progress_label = QLabel("")
        # self.mc_progress_label.setStyleSheet("font-size: 10px; font-style: italic;")
        # self.mc_progress_label.setMinimumWidth(200)
        # self.mc_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # self.button_layout_2.addWidget(self.mc_progress_label)

        self.motion_correct_button = HoverButton('Motion Correct', None, self.parent_widget.statusBar())
        self.motion_correct_button.setHoverMessage("Perform motion correction on the videos.")
        self.motion_correct_button.setIcon(QIcon("icons/accept_icon.png"))
        self.motion_correct_button.setIconSize(QSize(16,16))
        self.motion_correct_button.setStyleSheet('font-weight: bold;')
        self.motion_correct_button.clicked.connect(self.controller.motion_correct_video)
        self.button_layout_2.addWidget(self.motion_correct_button)
        
        # self.accept_button = HoverButton('ROI Finding', None, self.parent_widget.statusBar())
        # self.accept_button.setHoverMessage("Automatically find ROIs.")
        # self.accept_button.setIcon(QIcon("icons/skip_icon.png"))
        # self.accept_button.setIconSize(QSize(16,16))
        # # self.accept_button.setMaximumWidth(100)
        # self.accept_button.clicked.connect(lambda:self.controller.show_roi_finding_params())
        # self.button_layout_2.addWidget(self.accept_button)
    
    def remove_selected_items(self):
        self.videos_widget.remove_selected_items()

    def preview_contrast(self):
        contrast = self.param_sliders["contrast"].sliderPosition()/float(self.param_slider_multipliers["contrast"])

        self.controller.preview_contrast(contrast)

    def preview_gamma(self):
        gamma = self.param_sliders["gamma"].sliderPosition()/float(self.param_slider_multipliers["gamma"])

        self.controller.preview_gamma(gamma)

    def motion_correction_started(self):
        self.motion_correct_button.setText("Motion correcting... 0%")
        # self.motion_correct_button.setText('Cancel')
        self.motion_correct_button.setEnabled(False)
        # self.motion_correct_button.setHoverMessage("Stop motion correction.")
        self.parent_widget.tab_widget.setTabEnabled(0, False)
        self.parent_widget.tab_widget.setTabEnabled(2, False)
        self.parent_widget.tab_widget.setTabEnabled(3, False)
        # self.accept_button.setEnabled(False)

    def motion_correction_ended(self):
        self.motion_correct_button.setEnabled(True)
        self.parent_widget.tab_widget.setTabEnabled(0, True)
        self.parent_widget.tab_widget.setTabEnabled(2, True)
        # self.parent_widget.tab_widget.setTabEnabled(3, True)

    def update_motion_correction_progress(self, percent):
        if percent == 100:
            # self.motion_correct_button.setEnabled(True)
            # self.mc_progress_label.setText("")
            self.motion_correct_button.setText('Motion Correct')
            self.motion_correct_button.setHoverMessage("Perform motion correction on the videos.")
            # self.accept_button.setEnabled(True)
        elif percent == -1:
            # self.motion_correct_button.setEnabled(True)
            # self.mc_progress_label.setText("")
            self.motion_correct_button.setText('Motion Correct')
            self.motion_correct_button.setHoverMessage("Perform motion correction on the videos.")
        else:
            # self.motion_correct_button.setEnabled(True)
            self.motion_correct_button.setText("Motion correcting... {}%".format(int(percent)))

    def cancelling_motion_correction(self):
        self.motion_correct_button.setEnabled(False)
        # self.motion_correct_button.setText("")
        # self.mc_progress_label.setText("Cancelling motion correction...")

    def item_selected(self):
        selected_items = self.videos_list.selectedItems()

        if len(selected_items) > 0:
            index = self.videos_list.selectedIndexes()[0].row()
            self.controller.video_selected(index)
        else:
            index = None

class ROIFindingWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "")

        # self.parent_widget = parent_widget
        self.controller = controller

        # self.main_layout = QVBoxLayout(self)
        # self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.cnmf_roi_finding_widget = CNMFROIFindingWidget(self.parent_widget, self.controller)
        self.suite2p_roi_finding_widget = Suite2pROIFindingWidget(self.parent_widget, self.controller)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(self.tab_widget)
        self.tab_widget.addTab(self.cnmf_roi_finding_widget, "CNMF")
        self.tab_widget.addTab(self.suite2p_roi_finding_widget, "Suite2p")
        self.tab_widget.currentChanged.connect(self.tab_selected)

        # self.videos_list_widget = QWidget(self)
        # self.videos_list_layout = QHBoxLayout(self.videos_list_widget)
        # self.videos_list_layout.setContentsMargins(5, 0, 5, 0)
        # self.main_layout.addWidget(self.videos_list_widget)
        
        # self.videos_list = QListWidget(self)
        # self.videos_list.setStyleSheet(rounded_stylesheet)
        # # self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.videos_list.itemSelectionChanged.connect(self.item_selected)
        # self.videos_list_layout.addWidget(self.videos_list)

        self.main_layout.addStretch()

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(10, 10, 10, 10)
        self.button_layout.setSpacing(15)
        self.main_layout.addWidget(self.button_widget)

        self.show_rois_checkbox = QCheckBox("Show ROIs")
        self.show_rois_checkbox.setObjectName("Show ROIs")
        self.show_rois_checkbox.setChecked(False)
        self.show_rois_checkbox.setEnabled(False)
        self.show_rois_checkbox.clicked.connect(self.parent_widget.toggle_show_rois)
        # self.show_rois_checkbox.setDisabled(True)
        self.button_layout.addWidget(self.show_rois_checkbox)

        self.button_layout.addStretch()

        self.use_multiprocessing_checkbox = HoverCheckBox("Use multiprocessing", None, self.parent_widget.statusBar())
        self.use_multiprocessing_checkbox.setHoverMessage("Use multiple cores to speed up computations.")
        self.use_multiprocessing_checkbox.setChecked(True)
        self.use_multiprocessing_checkbox.clicked.connect(lambda:self.controller.set_use_multiprocessing(self.use_multiprocessing_checkbox.isChecked()))
        self.button_layout.addWidget(self.use_multiprocessing_checkbox)

        # self.roi_finding_progress_label = QLabel("")
        # self.roi_finding_progress_label.setStyleSheet("font-size: 10px; font-style: italic;")
        # self.roi_finding_progress_label.setMinimumWidth(200)
        # self.roi_finding_progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # self.button_layout.addWidget(self.roi_finding_progress_label)

        self.process_video_button = HoverButton('Find ROIs', None, self.parent_widget.statusBar())
        self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
        self.process_video_button.setIcon(QIcon("icons/accept_icon.png"))
        self.process_video_button.setIconSize(QSize(16,16))
        self.process_video_button.setStyleSheet('font-weight: bold;')
        self.process_video_button.clicked.connect(self.controller.find_rois)
        self.button_layout.addWidget(self.process_video_button)

    def item_selected(self):
        selected_items = self.videos_list.selectedItems()

        if len(selected_items) > 0:
            index = self.videos_list.selectedIndexes()[0].row()
            self.controller.video_selected(index)
        else:
            index = None

    def roi_finding_started(self):
        self.process_video_button.setText("Finding ROIs... 0%")
        # self.process_video_button.setText('Cancel')
        self.process_video_button.setEnabled(False)
        # self.process_video_button.setHoverMessage("Stop finding ROIs.")
        self.parent_widget.tab_widget.setTabEnabled(0, False)
        self.parent_widget.tab_widget.setTabEnabled(1, False)
        self.parent_widget.tab_widget.setTabEnabled(3, False)
        # self.filter_rois_button.setEnabled(False)
        # self.motion_correct_button.setEnabled(False)

    def update_roi_finding_progress(self, percent):
        if percent == 100:
            # self.roi_finding_progress_label.setText("")
            self.process_video_button.setText('Find ROIs')
            self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
            # self.filter_rois_button.setEnabled(True)
        elif percent == -1:
            # self.roi_finding_progress_label.setText("")
            self.process_video_button.setText('Find ROIs')
            self.process_video_button.setHoverMessage("Find ROIs using the watershed algorithm.")
        else:
            self.process_video_button.setText("Finding ROIs... {}%".format(int(percent)))

    def tab_selected(self):
        index = self.tab_widget.currentIndex()
        if index == 0:
            self.controller.roi_finding_mode = "cnmf"
        elif index == 1:
            self.controller.roi_finding_mode = "suite2p"

class CNMFROIFindingWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "CNMF Parameters")

        # self.parent_widget = parent_widget
        self.controller = controller

        self.add_param_slider(label_name="Autoregressive Model Order", name="autoregressive_order", minimum=0, maximum=2, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Background Components", name="num_bg_components", minimum=1, maximum=100, moved=self.update_param, num=1, multiplier=1, pressed=self.update_param, released=self.update_param, description="Number of background components.", int_values=True)
        self.add_param_slider(label_name="Merge Threshold", name="merge_threshold", minimum=1, maximum=200, moved=self.update_param, num=2, multiplier=200, pressed=self.update_param, released=self.update_param, description="Merging threshold (maximum correlation allowed before merging two components).", int_values=False)
        self.add_param_slider(label_name="Components", name="num_components", minimum=1, maximum=5000, moved=self.update_param, num=3, multiplier=1, pressed=self.update_param, released=self.update_param, description="Number of components to start with.", int_values=True)
        self.add_param_slider(label_name="Neuron Half-Size", name="half_size", minimum=1, maximum=50, moved=self.update_param, num=4, multiplier=1, pressed=self.update_param, released=self.update_param, description="Expected half-size of neurons (pixels).", int_values=True)
        
        self.main_layout.addStretch()

class Suite2pROIFindingWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "Suite2p Parameters")

        # self.parent_widget = parent_widget
        self.controller = controller

        self.add_param_slider(label_name="Diameter", name="diameter", minimum=1, maximum=100, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Sampling Rate", name="sampling_rate", minimum=1, maximum=60, moved=self.update_param, num=1, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_checkbox(label_name="Connected", name="connected", clicked=self.toggle_connected, description="Whether to use a convolutional neural network for determining which ROIs are neurons.", num=2)
        self.add_param_slider(label_name="Neuropil Basis Ratio", name="neuropil_basis_ratio", minimum=1, maximum=20, moved=self.update_param, num=3, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Neuropil Radius Ratio", name="neuropil_radius_ratio", minimum=1, maximum=50, moved=self.update_param, num=4, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Inner Neropil Radius", name="inner_neuropil_radius", minimum=1, maximum=50, moved=self.update_param, num=5, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Min. Neuropil Pixels", name="min_neuropil_pixels", minimum=1, maximum=500, moved=self.update_param, num=6, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        # self.add_param_slider(label_name="Diameter", name="diameter", minimum=1, maximum=50, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
 
        self.main_layout.addStretch()

    def toggle_connected(self, boolean):
        self.controller.controller.params['connected'] = boolean

class CNMFWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "CNMF Parameters")

        self.controller = controller

        # self.add_param_slider(label_name="Background Threshold", name="background_threshold", minimum=1, maximum=255, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Threshold for background.", int_values=True)
        self.add_param_slider(label_name="Autoregressive Model Order", name="autoregressive_order", minimum=0, maximum=2, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Order of the autoregressive model (0, 1 or 2).", int_values=True)
        self.add_param_slider(label_name="Background Components", name="num_bg_components", minimum=1, maximum=10, moved=self.update_param, num=1, multiplier=1, pressed=self.update_param, released=self.update_param, description="Number of background components.", int_values=True)
        self.add_param_slider(label_name="Merge Threshold", name="merge_threshold", minimum=1, maximum=200, moved=self.update_param, num=2, multiplier=200, pressed=self.update_param, released=self.update_param, description="Merging threshold (maximum correlation allowed before merging two components).", int_values=False)
        self.add_param_slider(label_name="Components", name="num_components", minimum=1, maximum=2000, moved=self.update_param, num=3, multiplier=1, pressed=self.update_param, released=self.update_param, description="Number of components to start with.", int_values=True)
        self.add_param_slider(label_name="Neuron Half-Size", name="half_size", minimum=1, maximum=50, moved=self.update_param, num=4, multiplier=1, pressed=self.update_param, released=self.update_param, description="Expected half-size of neurons (pixels).", int_values=True)
        
        self.main_layout.addStretch()

class WatershedWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "Watershed Parameters")

        self.controller = controller

        self.add_param_slider(label_name="Background Threshold", name="background_threshold", minimum=1, maximum=255, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Threshold for background.", int_values=True)
        self.add_param_slider(label_name="Normalization Window Size", name="window_size", minimum=2, maximum=30, moved=self.update_param, num=1, multiplier=1, pressed=self.update_param, released=self.update_param, description="Size (in pixels) of the window used to normalize brightness across the image.", int_values=True)
        
        self.main_layout.addStretch()

class ROIFilteringWidget(ParamWidget):
    def __init__(self, parent_widget, controller):
        ParamWidget.__init__(self, parent_widget, controller, "ROI Filtering Parameters")

        self.controller = controller

        # self.videos_list_widget = QWidget(self)
        # self.videos_list_layout = QHBoxLayout(self.videos_list_widget)
        # self.videos_list_layout.setContentsMargins(5, 0, 5, 0)
        # self.main_layout.addWidget(self.videos_list_widget)
        
        # self.videos_list = QListWidget(self)
        # self.videos_list.setStyleSheet(rounded_stylesheet)
        # # self.videos_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.videos_list.itemSelectionChanged.connect(self.item_selected)
        # self.videos_list_layout.addWidget(self.videos_list)

        self.add_param_slider(label_name="Imaging FPS", name="imaging_fps", minimum=1, maximum=60, moved=self.update_param, num=0, multiplier=1, pressed=self.update_param, released=self.update_param, description="Imaging frame rate (frames per second).", int_values=True)
        self.add_param_slider(label_name="Decay Time", name="decay_time", minimum=1, maximum=100, moved=self.update_param, num=1, multiplier=100, pressed=self.update_param, released=self.update_param, description="Length of a typical calcium transient (seconds).", int_values=False)
        self.add_param_slider(label_name="Minimum SNR", name="min_snr", minimum=1, maximum=500, moved=self.update_param, num=2, multiplier=100, pressed=self.update_param, released=self.update_param, description="Minimum signal to noise ratio.", int_values=False)
        self.add_param_slider(label_name="Minimum Spatial Correlation", name="min_spatial_corr", minimum=1, maximum=100, moved=self.update_param, num=3, multiplier=100, pressed=self.update_param, released=self.update_param, description="Minimum spatial correlation.", int_values=False)
        self.add_param_checkbox(label_name="Use CNN", name="use_cnn", clicked=self.toggle_use_cnn, description="Whether to use a convolutional neural network for determining which ROIs are neurons.", num=4)
        self.add_param_slider(label_name="CNN Threshold", name="cnn_threshold", minimum=1, maximum=100, moved=self.update_param, num=5, multiplier=100, pressed=self.update_param, released=self.update_param, description="Minimum CNN confidence (only relevant if using CNN).", int_values=False)
        self.add_param_slider(label_name="Minimum Area", name="min_area", minimum=1, maximum=1000, moved=self.update_param, num=6, multiplier=1, pressed=self.update_param, released=self.update_param, description="Minimum area.", int_values=True)
        self.add_param_slider(label_name="Maximum Area", name="max_area", minimum=1, maximum=1000, moved=self.update_param, num=7, multiplier=1, pressed=self.update_param, released=self.update_param, description="Maximum area.", int_values=True)
        
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

        # self.erase_rois_button = HoverButton('Select...', None, self.parent_widget.statusBar())
        # self.erase_rois_button.setHoverMessage("Select ROIs by dragging the mouse.")
        # self.erase_rois_button.setIcon(QIcon("icons/eraser_icon.png"))
        # self.erase_rois_button.setIconSize(QSize(16, 16))
        # self.erase_rois_button.clicked.connect(self.controller.erase_rois)
        # self.roi_button_layout.addWidget(self.erase_rois_button)
        # self.erase_rois_button.setEnabled(False)

        self.undo_button = HoverButton('Undo', None, self.parent_widget.statusBar())
        self.undo_button.setHoverMessage("Undo the previous action.")
        self.undo_button.setIcon(QIcon("icons/undo_icon.png"))
        self.undo_button.setIconSize(QSize(16, 16))
        self.undo_button.clicked.connect(self.controller.undo)
        self.roi_button_layout.addWidget(self.undo_button)
        self.undo_button.setEnabled(False)

        self.reset_button = HoverButton('Reset', None, self.parent_widget.statusBar())
        self.reset_button.setHoverMessage("Reset ROIs.")
        self.reset_button.setIcon(QIcon("icons/reset_icon.png"))
        self.reset_button.setIconSize(QSize(16, 16))
        self.reset_button.clicked.connect(self.controller.reset_erase)
        self.roi_button_layout.addWidget(self.reset_button)
        self.reset_button.setEnabled(False)

        # self.draw_rois_button = HoverButton('Draw', None, self.parent_widget.statusBar())
        # self.draw_rois_button.setHoverMessage("Manually draw circular ROIs.")
        # self.draw_rois_button.setIcon(QIcon("icons/draw_icon.png"))
        # self.draw_rois_button.setIconSize(QSize(16, 16))
        # self.draw_rois_button.clicked.connect(self.controller.draw_rois)
        # self.roi_button_layout.addWidget(self.draw_rois_button)

        self.roi_button_widget_2 = QWidget(self)
        self.roi_button_layout_2 = QHBoxLayout(self.roi_button_widget_2)
        self.roi_button_layout_2.setContentsMargins(5, 0, 0, 0)
        self.roi_button_layout_2.setSpacing(5)
        self.main_layout.addWidget(self.roi_button_widget_2)

        label = QLabel("Selected ROIs")
        label.setStyleSheet(SUBTITLE_STYLESHEET)
        self.roi_button_layout_2.addWidget(label)

        self.roi_button_layout_2.addStretch()

        self.erase_selected_roi_button = HoverButton('Discard', None, self.parent_widget.statusBar())
        self.erase_selected_roi_button.setHoverMessage("Discard the selected ROIs.")
        self.erase_selected_roi_button.setIcon(QIcon("icons/hide_icon.png"))
        self.erase_selected_roi_button.setIconSize(QSize(16, 16))
        self.erase_selected_roi_button.clicked.connect(self.controller.erase_selected_rois)
        self.erase_selected_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.erase_selected_roi_button)

        self.erase_all_rois_button = HoverButton('Discard All', None, self.parent_widget.statusBar())
        self.erase_all_rois_button.setHoverMessage("Discard all ROIs.")
        self.erase_all_rois_button.setIcon(QIcon("icons/hide_icon.png"))
        self.erase_all_rois_button.setIconSize(QSize(16, 16))
        self.erase_all_rois_button.clicked.connect(self.controller.erase_all_rois)
        self.roi_button_layout_2.addWidget(self.erase_all_rois_button)

        self.unerase_selected_roi_button = HoverButton('Keep', None, self.parent_widget.statusBar())
        self.unerase_selected_roi_button.setHoverMessage("Keep the selected ROIs.")
        self.unerase_selected_roi_button.setIcon(QIcon("icons/show_icon.png"))
        self.unerase_selected_roi_button.setIconSize(QSize(16, 16))
        self.unerase_selected_roi_button.clicked.connect(self.controller.unerase_selected_rois)
        self.unerase_selected_roi_button.setEnabled(False)
        self.roi_button_layout_2.addWidget(self.unerase_selected_roi_button)

        # self.lock_roi_button = HoverButton('Lock', None, self.parent_widget.statusBar())
        # self.lock_roi_button.setHoverMessage("Lock the currently selected ROI. This prevents it from being filtered out or erased.")
        # self.lock_roi_button.setIcon(QIcon("icons/lock_icon.png"))
        # self.lock_roi_button.setIconSize(QSize(16, 16))
        # self.lock_roi_button.clicked.connect(self.controller.lock_roi)
        # self.lock_roi_button.setEnabled(False)
        # self.roi_button_layout_2.addWidget(self.lock_roi_button)
        # self.lock_roi_button.setEnabled(False)

        self.merge_rois_button = HoverButton('Merge', None, self.parent_widget.statusBar())
        self.merge_rois_button.setHoverMessage("Merge the selected ROIs.")
        self.merge_rois_button.setIcon(QIcon("icons/merge_icon.png"))
        self.merge_rois_button.setIconSize(QSize(16, 16))
        self.merge_rois_button.clicked.connect(self.controller.merge_selected_rois)
        self.roi_button_layout_2.addWidget(self.merge_rois_button)
        self.merge_rois_button.setEnabled(False)

        # self.plot_traces_button = HoverButton('Plot Traces', None, self.parent_widget.statusBar())
        # self.plot_traces_button.setHoverMessage("Plot traces of the selected ROIs.")
        # self.plot_traces_button.setIcon(QIcon("icons/plot_icon.png"))
        # self.plot_traces_button.setIconSize(QSize(16, 16))
        # self.plot_traces_button.clicked.connect(self.controller.plot_traces)
        # self.roi_button_layout_2.addWidget(self.plot_traces_button)
        # self.plot_traces_button.setEnabled(False)

        # self.main_layout.addWidget(HLine())

        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(5, 5, 5, 5)
        self.button_layout.setSpacing(5)
        self.main_layout.addWidget(self.button_widget)

        self.show_rois_checkbox = QCheckBox("Show ROIs")
        self.show_rois_checkbox.setObjectName("Show ROIs")
        self.show_rois_checkbox.setChecked(False)
        self.show_rois_checkbox.clicked.connect(lambda:self.controller.show_roi_image(self.show_rois_checkbox.isChecked()))
        self.button_layout.addWidget(self.show_rois_checkbox)

        self.button_layout.addStretch()

        # self.motion_correct_button = HoverButton('Motion Correction', None, self.parent_widget.statusBar())
        # self.motion_correct_button.setHoverMessage("Go back to motion correction.")
        # self.motion_correct_button.setIcon(QIcon("icons/skip_back_icon.png"))
        # self.motion_correct_button.setIconSize(QSize(16,16))
        # self.motion_correct_button.clicked.connect(lambda:self.controller.show_motion_correction_params())
        # self.button_layout.addWidget(self.motion_correct_button)

        # self.find_rois_button = HoverButton('ROI Finding', None, self.parent_widget.statusBar())
        # self.find_rois_button.setHoverMessage("Go back to ROI segmentation.")
        # self.find_rois_button.setIcon(QIcon("icons/skip_back_icon.png"))
        # self.find_rois_button.setIconSize(QSize(16,16))
        # self.find_rois_button.clicked.connect(lambda:self.controller.show_roi_finding_params())
        # self.button_layout.addWidget(self.find_rois_button)

        self.filter_rois_button = HoverButton('Filter ROIs', None, self.parent_widget.statusBar())
        self.filter_rois_button.setHoverMessage("Automatically filter ROIs with the current parameters.")
        self.filter_rois_button.setIcon(QIcon("icons/accept_icon.png"))
        self.filter_rois_button.setIconSize(QSize(16,16))
        self.filter_rois_button.setStyleSheet('font-weight: bold;')
        self.filter_rois_button.clicked.connect(self.controller.filter_rois)
        self.button_layout.addWidget(self.filter_rois_button)

        self.process_videos_button = HoverButton('Save...', None, self.parent_widget.statusBar())
        self.process_videos_button.setHoverMessage("Save traces, ROI centroids and other ROI data.")
        self.process_videos_button.setIcon(QIcon("icons/save_icon.png"))
        self.process_videos_button.setIconSize(QSize(16,16))
        self.process_videos_button.setStyleSheet('font-weight: bold;')
        self.process_videos_button.clicked.connect(self.controller.process_all_videos)
        self.button_layout.addWidget(self.process_videos_button)

    def roi_erasing_started(self):
        self.filter_rois_button.setEnabled(False)
        # self.find_rois_button.setEnabled(False)
        # self.motion_correct_button.setEnabled(False)
        # self.enlarge_roi_button.setEnabled(False)
        # self.shrink_roi_button.setEnabled(False)
        # self.lock_roi_button.setEnabled(False)
        # self.erase_rois_button.setEnabled(False)
        self.erase_selected_roi_button.setEnabled(False)
        self.merge_rois_button.setEnabled(False)
        self.process_videos_button.setEnabled(False)
        # self.reset_button.setEnabled(False)
        # self.undo_button.setEnabled(False)
        self.param_widget.setEnabled(False)
        # self.draw_rois_button.setEnabled(False)
        # self.erase_rois_button.setText("Finished")
        self.parent_widget.tab_widget.setTabEnabled(0, False)
        self.parent_widget.tab_widget.setTabEnabled(1, False)
        self.parent_widget.tab_widget.setTabEnabled(2, False)
        # self.parent_widget.tab_widget.setTabEnabled(3, False)

    def roi_erasing_ended(self):
        self.filter_rois_button.setEnabled(True)
        # self.find_rois_button.setEnabled(True)
        # self.motion_correct_button.setEnabled(True)
        # self.erase_selected_roi_button.setEnabled(True)
        # self.merge_rois_button.setEnabled(True)
        # self.erase_rois_button.setEnabled(True)
        self.process_videos_button.setEnabled(True)
        # self.reset_button.setEnabled(True)
        # self.undo_button.setEnabled(True)
        self.param_widget.setEnabled(True)
        # self.draw_rois_button.setEnabled(True)
        # self.erase_rois_button.setText("Select...")
        self.parent_widget.tab_widget.setTabEnabled(0, True)
        self.parent_widget.tab_widget.setTabEnabled(1, True)
        self.parent_widget.tab_widget.setTabEnabled(2, True)
        # self.parent_widget.tab_widget.setTabEnabled(3, True)

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
        # self.erase_rois_button.setEnabled(False)
        self.draw_rois_button.setText("Finished")

    def roi_drawing_ended(self):
        print("ROI drawing ended.")
        
        self.filter_rois_button.setEnabled(True)
        self.find_rois_button.setEnabled(True)
        self.motion_correct_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.param_widget.setEnabled(True)
        # self.erase_rois_button.setEnabled(True)
        self.draw_rois_button.setText("Draw")

    def item_selected(self):
        selected_items = self.videos_list.selectedItems()

        if len(selected_items) > 0:
            index = self.videos_list.selectedIndexes()[0].row()
            self.controller.video_selected(index)
        else:
            index = None

    def toggle_use_cnn(self, boolean):
        self.controller.controller.params['use_cnn'] = boolean

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

    frame.setStyleSheet("color: rgba(0, 0, 0, 0.2);")

    return frame
