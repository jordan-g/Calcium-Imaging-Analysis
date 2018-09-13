from __future__ import division
from past.utils import old_div
from param_window import ParamWindow
from preview_window import PreviewWindow
from skimage.morphology import *
import utilities
import time
import json
import os
import glob
import sys
import scipy.ndimage as ndi
import scipy.signal
import numpy as np
from skimage.external.tifffile import imread, imsave
from skimage.measure import find_contours, regionprops
from skimage.filters import gaussian
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import exposure
import cv2
import matplotlib.pyplot as plt
import csv
import pdb
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from matplotlib import gridspec

# import the Qt library
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    pyqt_version = 4
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    pyqt_version = 5

if sys.version_info[0] < 3:
    python_version = 2
else:
    python_version = 3

colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]

class GUIController():
    def __init__(self, controller):
        self.controller = controller
        
        # create windows
        self.param_window   = ParamWindow(self)
        self.preview_window = PreviewWindow(self)

        self.video               = None
        self.image               = None
        self.background_mask     = None
        self.roi_overlay         = None
        self.roi_image           = None
        self.selected_mask       = None
        self.selected_mask_num   = -1
        self.rois_erased         = False
        self.trace_figure        = None
        self.figure_closed       = True
        self.selected_rois       = []
        self.manual_roi_selected = False
        self.show_rois           = False
        self.selected_video      = 0

        # initialize state variables
        self.closing                      = False
        self.performing_motion_correction = False # whether motion correction is being performed
        self.finding_rois                 = False # whether ROIs are currently being found
        self.processing_videos            = False # whether videos are currently being processed

        # initialize thread variables
        self.motion_correction_thread = None
        self.roi_finding_thread       = None
        self.video_processing_thread  = None

        # set the mode -- "loading" / "motion_correcting" / "roi_finding" / "roi_filtering"
        self.mode = "loading"

        # set references to param widgets & preview window
        self.param_widget                   = self.param_window.main_param_widget
        self.motion_correction_param_widget = self.param_window.motion_correction_widget
        self.roi_finding_param_widget       = self.param_window.roi_finding_widget
        self.roi_filtering_param_widget     = self.param_window.roi_filtering_widget

        # set the current z plane to 0
        self.z = 0

        mmap_files = glob.glob('*.mmap')
        for mmap_file in mmap_files:
            os.remove(mmap_file)

        mmap_files = glob.glob('memmap_*')
        for mmap_file in mmap_files:
            os.remove(mmap_file)

        mmap_files = glob.glob('*_temp.tif')
        for mmap_file in mmap_files:
            os.remove(mmap_file)

    def import_videos(self):
        if self.controller.mc_video is not None or (self.controller.roi_spatial_footprints is not None and self.controller.roi_spatial_footprints[0] is not None):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText("Adding videos will throw out any motion correction or ROI finding results. Continue?")
            msg.setWindowTitle("")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            retval = msg.exec_()
            if retval == QMessageBox.Cancel:
                return
            else:
                self.controller.reset_motion_correction_variables()
                self.controller.reset_roi_finding_variables(reset_rois=True)
                self.controller.reset_roi_filtering_variables(reset_rois=True)

        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff)')

            video_paths = [ str(path) for path in video_paths ]
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff)')[0]

        # import the videos (only the first video is actually opened and previewed, the rest are added to a list of videos to process)
        if video_paths is not None and len(video_paths) > 0:
            self.controller.import_videos(video_paths)

            # set z to 0 if necessary
            if self.z >= self.controller.video.shape[1]:
                self.z = 0

            # notify the param window
            self.param_window.videos_imported(video_paths)
            
            if self.controller.use_mc_video and self.controller.mc_video is not None:
                video = self.controller.mc_video
            else:
                video = self.controller.video
                
            self.show_video(video, self.controller.video_path)

            if self.selected_video is None:
                self.selected_video = 0

            # reset history variables
            self.reset_history()

            # notify the param window
            self.param_window.video_opened(max_z=self.controller.video.shape[1]-1, z=self.z)
            
            # # re-do motion correction
            # if self.controller.use_mc_video and self.controller.mc_video is not None:
            #     self.motion_correct_video()

    def video_selected(self, index, force_show=False):
        if force_show or (index is not None and index != self.selected_video):
            self.selected_video = index
            self.controller.open_video(self.controller.video_paths[index])

            if self.controller.use_mc_video and self.controller.mc_video is not None:
                self.controller.open_mc_video(self.controller.mc_video_paths[index])
                video = self.controller.mc_video
            else:
                video = self.controller.video

            if self.mode in ("loading", "motion_correcting"):
                self.show_video(video, self.controller.video_path)
            else:
                # calculate mean images
                self.controller.calculate_mean_images()

                # calculate ROI finding variables
                self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])
                
                # self.selected_rois = []
                self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked(), force_update=True)
                
                # self.preview_window.image_plot.deselect_rois()
                # self.preview_window.image_plot.erase_rois(self.controller.removed_rois[self.z])
                if self.mode == "roi_filtering":
                    self.update_trace_plot()

    def show_video(self, video, video_path):
        # calculate gamma- and contrast-adjusted video
        self.video = self.calculate_adjusted_video(video, z=self.z)

        # print(self.video.shape, self.video.dtype)

        self.preview_window.play_video(self.video, video_path, self.controller.params['fps'])

    def save_mc_video(self):
        # let the user pick where to save the video
        if pyqt_version == 4:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save Video', '{}_motion_corrected'.format(os.path.splitext(self.controller.video_path)[0]), 'Videos (*.tif *.tiff)'))
        elif pyqt_version == 5:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save Video', '{}_motion_corrected'.format(os.path.splitext(self.controller.video_path)[0]), 'Videos (*.tif *.tiff)')[0])
        if not (save_path.endswith('.tif') or save_path.endswith('.tiff')):
            save_path += ".tif"

        self.controller.save_mc_video(save_path)

    def save_rois(self):
        # let the user pick where to save the ROIs
        if pyqt_version == 4:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.controller.video_path)[0]), 'Numpy (*.npy)'))
        elif pyqt_version == 5:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROIs', '{}_rois'.format(os.path.splitext(self.controller.video_path)[0]), 'Numpy (*.npy)')[0])
        if not save_path.endswith('.npy'):
            save_path += ".npy"

        if save_path is not None and len(save_path) > 0:
            self.controller.save_rois(save_path)

    def load_rois(self):
        print("Loading ROIs")
        # let the user pick saved ROIs
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        if load_path is not None and len(load_path) > 0:
            self.controller.load_rois(load_path)

            self.param_window.tab_widget.setTabEnabled(3, True)
            self.param_window.show_rois_action.setEnabled(True)
            self.param_window.save_rois_action.setEnabled(True)

            # stop any motion correction or ROI finding process
            self.cancel_motion_correction()
            self.cancel_roi_finding()

            self.preview_window.timer.stop()

            # reset motion correction progress text
            self.param_window.update_motion_correction_progress(-1)

            # reset ROI finding progress text
            self.param_window.update_roi_finding_progress(-1)

            # reset video processing progress text
            self.param_window.update_process_videos_progress(-1)

            # self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())
            self.show_rois = True
            self.roi_finding_param_widget.show_rois_checkbox.setChecked(True)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)

            msg.setText("Do you want to compute new temporal traces? This is required if you are loading ROIs created using one set of videos onto a different set of videos.")
            msg.setWindowTitle("")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            retval = msg.exec_()
            if retval != QMessageBox.No:
                if self.controller.use_mc_video and self.controller.mc_video is not None:
                    video_paths = self.controller.mc_video_paths
                else:
                    video_paths = self.controller.video_paths

                for i in range(len(video_paths)):
                    video_path = video_paths[i]

                    video = imread(video_path)
                        
                    if len(video.shape) == 3:
                        # add a z dimension
                        video = video[:, np.newaxis, :, :]
                        
                    if i == 0:
                        final_video = video
                    else:
                        final_video = np.concatenate([final_video, video], axis=0)

                for z in range(final_video.shape[1]):
                    roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.do_cnmf(final_video[:, z, :, :], self.controller.params, self.controller.roi_spatial_footprints[z], np.ones(self.controller.roi_temporal_footprints[z].shape), np.ones(self.controller.roi_temporal_residuals[z].shape), self.controller.bg_spatial_footprints[z], np.ones(self.controller.bg_temporal_footprints[z].shape), use_multiprocessing=self.controller.use_multiprocessing)

                    self.controller.roi_spatial_footprints[z]  = roi_spatial_footprints
                    self.controller.roi_temporal_footprints[z] = roi_temporal_footprints
                    self.controller.roi_temporal_residuals[z]  = roi_temporal_residuals
                    self.controller.bg_spatial_footprints[z]   = bg_spatial_footprints
                    self.controller.bg_temporal_footprints[z]  = bg_temporal_footprints

                # calculate temporal components based on spatial components
                roi_temporal_footprints, roi_temporal_residuals, bg_temporal_footprints = utilities.calculate_temporal_components(video_paths, self.controller.roi_spatial_footprints, self.controller.roi_temporal_footprints, self.controller.roi_temporal_residuals, self.controller.bg_spatial_footprints, self.controller.bg_temporal_footprints)
                self.controller.roi_temporal_footprints = roi_temporal_footprints
                self.controller.roi_temporal_residuals  = roi_temporal_residuals
                self.controller.bg_temporal_footprints  = bg_temporal_footprints

            # show ROI filtering parameters
            self.show_roi_filtering_params(loading_rois=True)

            self.update_trace_plot()

    def remove_videos_at_indices(self, indices):
        self.controller.remove_videos_at_indices(indices)
        # if self.selected_video in indices and len(self.controller.video_paths) != 0:
        #     self.video_selected(0)

        # cancel any ongoing motion correction
        self.cancel_motion_correction()
        self.cancel_processing_videos()
        self.cancel_roi_finding()

        if len(self.controller.video_paths) == 0:
            # switch to showing motion correction params
            # self.show_motion_correction_params()

            print("All videos removed.")

            # reset variables
            self.video             = None
            self.image             = None
            self.background_mask   = None
            self.roi_overlay       = None
            self.roi_image         = None
            self.selected_mask     = None
            self.selected_mask_num = -1
            self.rois_erased       = False
            self.trace_figure      = None
            self.figure_closed     = True
            self.selected_rois     = []

            mmap_files = glob.glob('*.mmap')
            for mmap_file in mmap_files:
                os.remove(mmap_file)

            mmap_files = glob.glob('memmap_*')
            for mmap_file in mmap_files:
                os.remove(mmap_file)

            mmap_files = glob.glob('*_temp.tif')
            for mmap_file in mmap_files:
                os.remove(mmap_file)

            # set the current z plane to 0
            self.z = 0

            # reset param window & preview window to their initial states
            self.param_window.set_initial_state()
            self.preview_window.set_initial_state()
        elif 0 in indices:
            self.video_selected(0, force_show=True)

            # notify the param window
            self.param_window.video_opened(max_z=self.controller.video.shape[1]-1, z=self.z)

    def process_all_videos(self):
        save_directory = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))

        for i in range(len(self.controller.video_paths)):
            video_path = self.controller.video_paths[i]

            base_name      = os.path.basename(video_path)
            name           = os.path.splitext(base_name)[0]
            directory      = os.path.dirname(video_path)
            video_dir_path = os.path.join(save_directory, name)

            # make a folder to hold the results
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)

            if base_name.endswith('.npy'):
                video = np.load(video_path)
            elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
                video = imread(video_path)

            if len(video.shape) == 3:
                # add z dimension
                video = video[:, np.newaxis, :, :]

            # save centroids & traces

            for z in range(video.shape[1]):
                print("Calculating ROI activities for z={}...".format(z))

                centroids = np.zeros((self.controller.roi_spatial_footprints[z].shape[-1], 2))
                kept_rois = [ roi for roi in range(self.controller.roi_spatial_footprints[z].shape[-1]) if (roi not in self.controller.removed_rois[z]) or (roi in self.controller.locked_rois[z]) ]

                footprints_2d = self.controller.roi_spatial_footprints[z].toarray().reshape((video.shape[2], video.shape[3], self.controller.roi_spatial_footprints[z].shape[-1]))

                for roi in kept_rois:
                    footprint_2d = footprints_2d[:, :, roi]

                    mask = footprint_2d > 0

                    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                    if len(contours) > 0:
                        contour = contours[0]

                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                        else:
                            center_x = 0
                            center_y = 0

                        centroids[roi] = [center_x, center_y]

                temporal_footprints = self.controller.roi_temporal_footprints[z]

                if z == 0:
                    temporal_footprints = temporal_footprints[:, :self.controller.video_lengths[0]]
                else:
                    temporal_footprints = temporal_footprints[:, np.sum(self.controller.video_lengths[:z]):np.sum(self.controller.video_lengths[:z+1])]

                traces = temporal_footprints[kept_rois]
                centroids = centroids[kept_rois]
                
                # centroids, traces = utilities.calculate_centroids_and_traces(rois[z], vid[:, z, :, :])

                print("Saving CSV for z={}...".format(z))

                # roi_nums = np.unique(rois[z]).tolist()
                roi_nums = kept_rois
                # # remove ROI #0 (this is the background)
                # try:
                #     index = roi_nums.index(0)
                #     del roi_nums[index]
                # except:
                #     pass

                with open(os.path.join(video_dir_path, 'z_{}_traces.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['ROI #'] + [ "Frame {}".format(frame) for frame in range(traces.shape[1]) ])

                    for i in range(traces.shape[0]):
                        writer.writerow(['{}'.format(roi_nums[i])] + traces[i].tolist())

                with open(os.path.join(video_dir_path, 'z_{}_centroids.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['Label', 'X', 'Y'])

                    for i in range(centroids.shape[0]):
                        writer.writerow(["ROI #{}".format(roi_nums[i])] + centroids[i].tolist())

                # save ROIs
                self.controller.save_rois(os.path.join(video_dir_path, 'roi_data.npy'))

                print("Done.")

    def cancel_processing_videos(self):
        if self.video_processing_thread is not None:
            # inform the thread to stop running
            self.video_processing_thread.running = False

        # reset video processing progress variables
        self.processing_videos       = False
        self.video_processing_thread = None

    def process_videos_progress(self, percent):
        self.param_window.update_process_videos_progress(percent)

    def process_videos_finished(self):
        self.param_window.update_process_videos_progress(100)

        self.processing_videos = False

    def find_rois(self):
        # stop any motion correction or video processing process
        self.cancel_motion_correction()
        self.cancel_processing_videos()

        if not self.finding_rois:
            # cancel any ongoing ROI finding
            self.cancel_roi_finding()

            # create an ROI finding thread
            self.roi_finding_thread = ROIFindingThread(self.roi_finding_param_widget)

            if self.controller.use_mc_video and self.controller.mc_video is not None:
                video_paths = self.controller.mc_video_paths
            else:
                video_paths = self.controller.video_paths

            # set the parameters of the ROI finding thread
            self.roi_finding_thread.set_parameters(video_paths, self.controller.masks, self.background_mask, self.controller.params["invert_masks"], self.controller.params, self.controller.mc_borders, self.controller.use_mc_video, self.controller.use_multiprocessing)

            # start the thread
            self.roi_finding_thread.start()

            self.finding_rois = True

            self.roi_finding_thread.progress.connect(self.roi_finding_progress)
            self.roi_finding_thread.finished.connect(self.roi_finding_ended)

            # notify the param widget
            self.roi_finding_param_widget.roi_finding_started()
        else:
            self.cancel_roi_finding()

    def cancel_roi_finding(self):
        if self.roi_finding_thread is not None:
            # inform the thread to stop running
            self.roi_finding_thread.running = False

        # reset ROI finding progress variables
        self.finding_rois       = False
        self.roi_finding_thread = None

        self.roi_finding_param_widget.update_roi_finding_progress(-1)

    def roi_finding_progress(self, percent):
        # notify the param widget
        self.roi_finding_param_widget.update_roi_finding_progress(percent)

    def roi_finding_ended(self, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints): # TODO: create an roi_finding_ended() method for the param window
        # pass

        self.finding_rois = False
        self.selected_rois = []

        self.controller.roi_spatial_footprints  = roi_spatial_footprints
        self.controller.roi_temporal_footprints = roi_temporal_footprints
        self.controller.roi_temporal_residuals  = roi_temporal_residuals
        self.controller.bg_spatial_footprints   = bg_spatial_footprints
        self.controller.bg_temporal_footprints  = bg_temporal_footprints
        self.controller.filtered_out_rois       = [ [] for i in range(self.controller.video.shape[1]) ]

        # # notify the param widget
        # self.roi_finding_param_widget.update_roi_finding_progress(100)

        # self.finding_rois = False

        # self.controller.rois              = rois
        # self.controller.original_rois     = rois[:]
        # self.controller.roi_areas         = roi_areas
        # self.controller.roi_circs         = roi_circs
        # self.controller.filtered_out_rois = filtered_out_rois
        # self.controller.removed_rois      = filtered_out_rois[:]

        # update the param window
        self.roi_finding_param_widget.show_rois_checkbox.setDisabled(False)
        self.roi_finding_param_widget.show_rois_checkbox.setChecked(True)
        # self.roi_finding_param_widget.motion_correct_button.setEnabled(True)
        self.param_window.show_rois_action.setDisabled(False)
        self.param_window.show_rois_action.setChecked(True)
        self.param_window.save_rois_action.setDisabled(False)
        # self.roi_finding_param_widget.filter_rois_button.setDisabled(False)

        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked(), force_update=True)
        # self.controller.mc_rois = self.controller.use_mc_video

        self.roi_finding_param_widget.roi_finding_ended()

        # # create ROI image
        # rgb_image = cv2.cvtColor(utilities.normalize(self.image, self.controller.video_max), cv2.COLOR_GRAY2RGB)
        # self.roi_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.controller.rois[self.z], None, None, self.controller.filtered_out_rois[self.z], None)

        # # show the ROI image
        # self.show_roi_image(True)

        if self.trace_figure is None:
            self.trace_figure = TracePlotWindow(self)
            self.trace_figure.show()
        self.trace_figure.plot(self.controller.roi_temporal_footprints[self.z])

    def roi_finding_method_changed(self, index):
        if index == 0:
            self.roi_finding_method = "CNMF"
        else:
            self.roi_finding_method = "watershed"

        print("Using {} method to find ROIs.".format(self.roi_finding_method))

    def motion_correct_video(self):
        # stop any video processing or ROI finding process
        self.cancel_processing_videos()
        self.cancel_roi_finding()

        if not self.performing_motion_correction:
            # cancel any ongoing motion correction
            self.cancel_motion_correction()

            self.motion_correction_param_widget.motion_correct_button.setEnabled(True)

            # create a motion correction thread
            self.motion_correction_thread = MotionCorrectThread(self.motion_correction_param_widget)
            self.motion_correction_thread.progress.connect(self.motion_correction_progress)
            self.motion_correction_thread.finished.connect(self.motion_correction_ended)

            # set the parameters of the motion correction thread
            self.motion_correction_thread.set_parameters(self.controller.video_paths
            , int(self.controller.params["max_shift"]), int(self.controller.params["patch_stride"]), int(self.controller.params["patch_overlap"]), use_multiprocessing=self.controller.use_multiprocessing)

            # start the thread
            self.motion_correction_thread.start()

            self.performing_motion_correction = True

            # notify the param widget
            self.motion_correction_param_widget.motion_correction_started()
        # else:
            # self.cancel_motion_correction()

    def cancel_motion_correction(self):
        pass
        # if self.motion_correction_thread is not None:
        #     self.motion_correction_thread.running = False

        # self.performing_motion_correction = False
        # self.motion_correction_thread     = None

        # self.motion_correction_param_widget.cancelling_motion_correction()

    def motion_correction_progress(self, percent):
        # notify the param widget
        self.motion_correction_param_widget.update_motion_correction_progress(percent)

    def motion_correction_ended(self, mc_videos, mc_borders):
        # notify the param widget
        self.motion_correction_param_widget.update_motion_correction_progress(100)

        self.performing_motion_correction = False

        if len(mc_videos) != 0:
            mc_video_paths = []
            for i in range(len(mc_videos)):
                video_path = self.controller.video_paths[i]
                directory = os.path.dirname(video_path)
                filename  = os.path.basename(video_path)
                mc_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_mc.tif")
                
                # save the motion-corrected video
                imsave(mc_video_path, mc_videos[i])
                
                mc_video_paths.append(mc_video_path)
                
            self.controller.mc_video       = mc_videos[self.selected_video].astype(np.float32)
            self.controller.mc_video_paths = mc_video_paths
            self.controller.mc_borders     = mc_borders
            
            # pdb.set_trace()

            self.motion_correction_param_widget.motion_correction_ended()

            # calculate the adjusted motion-corrected video at the current z plane
            self.video = self.calculate_adjusted_video(self.controller.mc_video, z=self.z)

            # update the param window
            self.motion_correction_param_widget.use_mc_video_checkbox.setEnabled(True)
            self.motion_correction_param_widget.use_mc_video_checkbox.setChecked(True)

            self.set_use_mc_video(True)

    def show_video_loading_params(self):
        self.param_window.tab_widget.setCurrentIndex(0)
        self.mode = "loading"

        self.preview_window.timer.stop()

        # play the video
        if self.controller.video is not None:
            self.show_video(self.controller.video, self.controller.video_path)

        self.param_window.statusBar().showMessage("")

    def show_motion_correction_params(self):
        # switch to showing motion correction params
        self.param_window.tab_widget.setCurrentIndex(1)
        self.mode = "motion_correcting"

        self.preview_window.timer.stop()

        # play the video
        if self.controller.use_mc_video and self.controller.mc_video is not None:
            self.show_video(self.controller.mc_video, self.controller.video_path)
        elif self.controller.video is not None:
            self.show_video(self.controller.video, self.controller.video_path)

        self.param_window.statusBar().showMessage("")

    def show_roi_finding_params(self):
        # cancel any ongoing motion correction
        self.cancel_motion_correction()

        if self.mode in ("loading", "motion_correcting"):
            update_mean_image = True
        else:
            update_mean_image = False

        self.preview_window.timer.stop()

        self.roi_finding_param_widget.show_rois_checkbox.setChecked(self.show_rois)
        if self.controller.roi_spatial_footprints is not None and self.controller.roi_spatial_footprints[self.z] is not None:
            self.roi_finding_param_widget.show_rois_checkbox.setEnabled(True)

        # switch to showing ROI finding params
        self.param_window.tab_widget.setCurrentIndex(2)
        self.mode = "roi_finding"

        if self.controller.mc_rois != self.controller.use_mc_video:
            self.controller.find_new_rois = True

        if self.controller.find_new_rois:
            self.controller.reset_roi_finding_variables(reset_rois=True)
            self.controller.reset_roi_filtering_variables(reset_rois=True)

            # calculate mean images
            self.controller.calculate_mean_images()

            # uncheck "Show ROIs" checkbox
            # self.roi_finding_param_widget.show_rois_checkbox.setChecked(False)

            self.controller.find_new_rois = False

            # calculate ROI finding variables
            self.image           = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])
            self.background_mask = utilities.calculate_background_mask(self.image, self.controller.params['background_threshold'], self.controller.video_max)
            
        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked(), force_update=update_mean_image)

        self.param_window.statusBar().showMessage("")

        self.preview_window.setWindowTitle(os.path.basename(self.controller.video_path))

    def show_roi_filtering_params(self, loading_rois=False):
        self.roi_filtering_param_widget.show_rois_checkbox.setChecked(self.show_rois)

        self.preview_window.timer.stop()

        if loading_rois:
            # calculate mean images
            self.controller.calculate_mean_images()

            # calculate adjusted image
            self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])

            # reset history variables
            self.reset_history()

        # switch to showing ROI filtering params
        self.param_window.tab_widget.setCurrentIndex(3)
        self.mode = "roi_filtering"

        # show the ROI image
        self.roi_filtering_param_widget.setEnabled(True)
        self.roi_filtering_param_widget.show_rois_checkbox.setChecked(True)
        self.show_roi_image(self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # add the current state to the history
        self.add_to_history()

        self.param_window.statusBar().showMessage("")

    def reset_history(self, z=None):
        # initialize history variables
        if z is None:
            self.last_erased_rois           = [ [] for i in range(self.video.shape[1]) ]
            self.previous_erased_rois       = [ [] for i in range(self.video.shape[1]) ]
            self.previous_filtered_out_rois = [ [] for i in range(self.video.shape[1]) ]
            self.previous_images            = [ [] for i in range(self.video.shape[1]) ]
            self.previous_removed_rois      = [ [] for i in range(self.video.shape[1]) ]
            self.previous_locked_rois       = [ [] for i in range(self.video.shape[1]) ]
            self.previous_params            = [ [] for i in range(self.video.shape[1]) ]
        else:
            self.last_erased_rois[z]           = []
            self.previous_erased_rois[z]       = []
            self.previous_filtered_out_rois[z] = []
            self.previous_images[z]            = []
            self.previous_removed_rois[z]      = []
            self.previous_locked_rois[z]       = []
            self.previous_params[z]            = []

    def close_all(self):
        # cancel any ongoing threads
        self.cancel_motion_correction()
        self.cancel_roi_finding()
        self.cancel_processing_videos()

        self.closing = True

        # close param & preview windows
        self.param_window.close()
        self.preview_window.close()
        if self.trace_figure is not None:
            self.trace_figure.close()

        # save the current parameters
        self.save_params()

    def preview_contrast(self, contrast):
        self.controller.params['contrast'] = contrast

        if self.mode in ("loading", "motion_correcting"):
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            if self.controller.use_mc_video:
                adjusted_frame = self.calculate_adjusted_frame(self.controller.mc_video)
            else:
                adjusted_frame = self.calculate_adjusted_frame(self.controller.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("contrast", contrast)

    def preview_gamma(self, gamma):
        self.controller.params['gamma'] = gamma

        if self.mode in ("loading", "motion_correcting"):
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            if self.controller.use_mc_video:
                adjusted_frame = self.calculate_adjusted_frame(self.controller.mc_video)
            else:
                adjusted_frame = self.calculate_adjusted_frame(self.controller.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("gamma", gamma)

    def update_param(self, param, value):
        # update the parameter
        if param in self.controller.params.keys():
            self.controller.params[param] = value

        if self.mode in ("loading", "motion_correcting"):
            if param in ("contrast, gamma"):
                self.preview_window.timer.stop()

                # play the video
                if self.controller.use_mc_video and self.controller.mc_video is not None:
                    self.show_video(self.controller.mc_video, self.controller.video_path)
                else:
                    self.show_video(self.controller.video, self.controller.video_path)
            elif param == "fps":
                # update the FPS of the preview window
                self.preview_window.set_fps(self.controller.params['fps'])
            elif param == "z":
                self.z = value

                self.preview_window.timer.stop()

                # play the video
                if self.controller.use_mc_video and self.controller.mc_video is not None:
                    self.show_video(self.controller.mc_video, self.controller.video_path)
                else:
                    self.show_video(self.controller.video, self.controller.video_path)
        elif self.mode == "roi_finding":
            if param in ("contrast, gamma"):
                # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
                self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])
                self.background_mask = utilities.calculate_background_mask(self.image, self.controller.params['background_threshold'], self.controller.video_max)

                # update the ROI image using the new adjusted image
                # if self.controller.rois is not None:
                #     self.calculate_roi_image(self.z, update_overlay=False)

                # show the ROI image
                self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked(), force_update=True)
            elif param == "background_threshold":
                # calculate the background mask using the new threshold
                self.background_mask = utilities.calculate_background_mask(self.image, self.controller.params['background_threshold'], self.controller.video_max)

                # uncheck the "Show ROIs" checkbox
                self.roi_finding_param_widget.show_rois_checkbox.setChecked(False)
                self.param_window.show_rois_action.setChecked(False)

                # show the background mask
                # TODO: only show the background mask while dragging the sliderimp
                self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max)
            elif param == "z":
                self.z = value

                # calculate a contrast- and gamma-adjusted version of the new z plane's mean image
                self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])
                self.background_mask = utilities.calculate_background_mask(self.image, self.controller.params['background_threshold'], self.controller.video_max)

                # update the ROI image using the new adjusted image
                # if self.controller.rois is not None:
                #     self.calculate_roi_image(self.z, update_overlay=True)

                # show the ROI image
                self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked(), force_update=True)
        elif self.mode == "roi_filtering":
            if param in ("contrast, gamma"):
                # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
                self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])

                # update the ROI image using the new adjusted image
                # if self.controller.rois is not None:
                #     self.calculate_roi_image(self.z, update_overlay=False)

                # show the ROI image
                self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), force_update=True)
            if param == "z":
                self.z = value

                self.selected_rois = []

                # calculate a contrast- and gamma-adjusted version of the new z plane's mean image
                self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])

                # filter the ROIs at the new z plane
                # self.controller.filter_rois(z=self.z)

                # update the ROI image using the new adjusted image
                # self.calculate_roi_image(z=self.z, update_overlay=True)

                # show the ROI image
                self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=True, force_update=True)

                # add the current state to the history only if this is the first time we've switched to this z plane
                # self.add_to_history(only_if_new=True)
            elif param in ("min_area", "max_area", "min_circ", "max_circ"):
                pass

    def calculate_adjusted_video(self, video, z=None):
        print(video.shape)

        if z is not None:
            # calculate the adjusted video only at this z plane
            return utilities.adjust_gamma(utilities.adjust_contrast((video[:, z, :, :]), self.controller.params['contrast']), self.controller.params['gamma'])
        else:
            # calculate the adjusted video at all z planes
            return utilities.adjust_gamma(utilities.adjust_contrast((video), self.controller.params['contrast']), self.controller.params['gamma'])

    def calculate_adjusted_frame(self, video):
        # calculate the adjusted frame
        return utilities.adjust_gamma(utilities.adjust_contrast((video[self.preview_window.frame_num, self.z]), self.controller.params['contrast']), self.controller.params['gamma'])

    def set_use_mc_video(self, use_mc_video):
        self.controller.use_mc_video = use_mc_video

        # calculate the corresponding adjusted video and play it
        if self.controller.use_mc_video:
            self.video = self.calculate_adjusted_video(self.controller.mc_video, z=self.z)
        else:
        	self.video = self.calculate_adjusted_video(self.controller.video, z=self.z)
        
        self.preview_window.play_movie(self.video, fps=self.controller.params['fps'])

    def set_mc_current_z(self, mc_current_z):
        self.controller.mc_current_z = mc_current_z

    def set_use_multiprocessing(self, use_multiprocessing):
        self.controller.use_multiprocessing = use_multiprocessing

    def set_apply_blur(self, apply_blur):
        self.controller.apply_blur = apply_blur

        # calculate new mean images
        self.controller.calculate_mean_images()

        # calculate a contrast- and gamma-adjusted version of the current z plane's mean image
        self.image = utilities.calculate_adjusted_image(self.controller.mean_images[self.z], self.controller.params['contrast'], self.controller.params['gamma'])
        self.background_mask = utilities.calculate_background_mask(self.image, self.controller.params['background_threshold'], self.controller.video_max)

        # update the ROI image using the new adjusted image
        # if self.controller.rois is not None:
        #     self.calculate_roi_image(self.z, update_overlay=False)


        # show the ROI image
        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())

    def show_roi_image(self, show, update_overlay=True, force_update=False):
        self.show_rois = show

        # plot the ROI image (or the regular image if show is False)
        if show:
            if self.preview_window.image_plot.flat_contours is None or force_update:
                print(self.z)
                self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=255.0, update_overlay=update_overlay)
            self.preview_window.image_plot.deselect_rois()
            if len(self.selected_rois) > 0:
                self.preview_window.image_plot.select_rois(self.selected_rois)
            if len(self.controller.removed_rois[self.z]) > 0:
                self.preview_window.image_plot.erase_rois(self.controller.removed_rois[self.z])
        else:
            self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max, update_overlay=False)

        # print("Done~")

        # update the param window
        self.param_window.show_rois_action.setChecked(show)
        if self.mode == "roi_finding":
            self.roi_finding_param_widget.show_rois_checkbox.setChecked(show)
        elif self.mode == "roi_filtering":
            self.roi_filtering_param_widget.show_rois_checkbox.setChecked(show)

        # print("Done~")

    def save_roi_image(self):
        # let the user pick where to save the ROI images
        if pyqt_version == 4:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROI image', '{}_rois_z_{}'.format(os.path.splitext(self.controller.video_path)[0], self.z), 'PNG (*.png)'))
        elif pyqt_version == 5:
            save_path = str(QFileDialog.getSaveFileName(self.param_window, 'Save ROI image', '{}_rois_z_{}'.format(os.path.splitext(self.controller.video_path)[0], self.z), 'PNG (*.png)')[0])
        if not save_path.endswith('.png'):
            save_path += ".png"

        if save_path is not None and len(save_path) > 0:
            # save the ROIs image
            scipy.misc.imsave(save_path, self.roi_image)

    def set_invert_masks(self, boolean):
        self.controller.set_invert_masks(boolean)

        self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max)

    def draw_mask(self):
        if not self.preview_window.drawing_mask:
            self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max)

            # notify the preview window that we are in mask drawing mode
            self.preview_window.start_drawing_mask()

            self.selected_mask     = None
            self.selected_mask_num = -1

            # update the param widget
            self.roi_finding_param_widget.draw_mask_button.setText("Done")
            self.roi_finding_param_widget.draw_mask_button.previous_message = "Draw a mask on the image preview."
            self.roi_finding_param_widget.param_widget.setEnabled(False)
            self.roi_finding_param_widget.button_widget.setEnabled(False)
            self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)
            self.roi_finding_param_widget.draw_mask_button.setEnabled(True)
        else:
            if len(self.preview_window.mask_points) > 0:
                # update the mask points
                mask_points = self.preview_window.mask_points
                mask_points += [mask_points[0]]
                self.controller.mask_points[self.z].append(mask_points)
                mask_points = np.array(mask_points)

                # create the mask
                mask = np.zeros(self.image.shape)
                cv2.fillConvexPoly(mask, mask_points, 1)
                mask = mask.astype(np.bool)

                if self.controller.params['invert_masks']:
                    mask = mask == False

                self.controller.masks[self.z].append(mask)

                self.controller.n_masks += 1

            # notify the preview window that we are no longer in mask drawing mode
            self.preview_window.end_drawing_mask()

            self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max)

            # update the param widget
            self.roi_finding_param_widget.draw_mask_button.setText("Draw Mask")
            self.roi_finding_param_widget.draw_mask_button.previous_message = ""
            self.roi_finding_param_widget.param_widget.setEnabled(True)
            self.roi_finding_param_widget.button_widget.setEnabled(True)

    # def calculate_roi_image(self, z, update_overlay=True, newly_erased_rois=None):
    #     if update_overlay:
    #         roi_overlay = None
    #     else:
    #         roi_overlay = self.roi_overlay

    #     # create ROI image
    #     rgb_image = cv2.cvtColor(utilities.normalize(self.image, self.controller.video_max), cv2.COLOR_GRAY2RGB)
    #     self.roi_image, self.roi_overlay = utilities.draw_rois(rgb_image, self.controller.rois[z], self.selected_roi, self.controller.erased_rois[z], self.controller.filtered_out_rois[z], self.controller.locked_rois[z], newly_erased_rois=newly_erased_rois, roi_overlay=roi_overlay)

    def select_mask(self, mask_point):
        # figure out which mask is selected (if any)
        selected_mask, selected_mask_num = utilities.get_mask_containing_point(self.controller.masks[self.z], mask_point, inverted=self.controller.params['invert_masks'])

        if selected_mask is not None:
            # update the param widget
            # self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(True)

            self.selected_mask     = selected_mask
            self.selected_mask_num = selected_mask_num
        else:
            # update the param widget
            # self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)

            self.selected_mask     = None
            self.selected_mask_num = -1

        self.show_roi_image(show=self.roi_finding_param_widget.show_rois_checkbox.isChecked())

    def erase_selected_mask(self):
        if self.selected_mask is not None:
            # remove the mask
            del self.controller.masks[self.z][self.selected_mask_num]
            del self.controller.mask_points[self.z][self.selected_mask_num]

            self.selected_mask     = None
            self.selected_mask_num = -1

            # update the param widget
            # self.roi_finding_param_widget.erase_selected_mask_button.setEnabled(False)

            self.preview_window.plot_image(self.image, background_mask=self.background_mask, video_max=self.controller.video_max)

    def filter_rois(self):
        self.controller.filter_rois()

        self.selected_rois = []

        # update the ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=True)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        self.update_trace_plot()

    def draw_rois(self): # TODO: create roi_drawing_started() and roi_drawing_ended() methods for the preview window
        if not self.preview_window.drawing_rois:
            self.preview_window.drawing_rois = True

            # notify the param window
            self.param_window.roi_drawing_started()
        else:
            self.preview_window.drawing_rois = False

            # notify the param window
            self.param_window.roi_drawing_ended()

    def create_roi(self, start_point, end_point):
        self.controller.create_roi(start_point, end_point, self.z)

        # update the ROI overlay
        # utilities.add_roi_to_overlay(self.roi_overlay, self.controller.rois[self.z] == l, self.controller.rois[self.z])

        # calculate the ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=False)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # add this state to the history
        self.add_to_history()

    def create_roi_magic_wand(self, point):
        self.controller.create_roi_magic_wand(self.image, point, self.z)

        # update the ROI overlay
        # utilities.add_roi_to_overlay(self.roi_overlay, self.controller.rois[self.z] == l, self.controller.rois[self.z])

        # calculate the ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=False)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # add this state to the history
        self.add_to_history()

    def shift_rois(self, start_point, end_point):
        self.controller.shift_rois(start_point, end_point, self.z)

        self.roi_overlay  = np.roll(self.roi_overlay, y_shift, axis=0)
        self.roi_overlay  = np.roll(self.roi_overlay, x_shift, axis=1)

        # calculate the ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=False)

        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def erase_rois(self): # TODO: create roi_erasing_started() and roi_erasing_ended() methods for the preview window
        self.rois_erased = False

        if not self.preview_window.erasing_rois:
            self.selected_rois = []

            # print("starting erasing")

            self.preview_window.erasing_rois = True

            # notify the param window
            self.param_window.roi_erasing_started()
        else:
            self.preview_window.erasing_rois = False

            if len(self.selected_rois) == 1:
                self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(True)
                self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
                self.param_window.merge_rois_action.setEnabled(False)
                self.param_window.trace_rois_action.setEnabled(True)
                # self.preview_window.trace_rois_action.setEnabled(True)
            elif len(self.selected_rois) > 1:
                # self.roi_filtering_param_widget.merge_rois_button.setEnabled(True)
                # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(True)
                if not any(x in self.selected_rois for x in self.controller.removed_rois[self.z]):
                # if selected_roi not in self.controller.removed_rois[self.z]:
                    self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                    self.roi_filtering_param_widget.merge_rois_button.setEnabled(True)
                else:
                    self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
                    self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                self.param_window.erase_rois_action.setEnabled(True)
                self.param_window.merge_rois_action.setEnabled(True)
                self.param_window.trace_rois_action.setEnabled(True)
                # self.preview_window.trace_rois_action.setEnabled(True)
            else:
                self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(False)
                self.param_window.erase_rois_action.setEnabled(False)
                self.param_window.merge_rois_action.setEnabled(False)
                self.param_window.trace_rois_action.setEnabled(False)

            # notify the param window
            self.param_window.roi_erasing_ended()

            # add the current state to the history
            # self.add_to_history()

    def select_rois_near_point(self, roi_point, radius=10):
        # if not self.rois_erased:
        #     # create a new list storing the ROIs that are being erased in this erasing operation
        #     self.last_erased_rois[self.z].append([])
        #     self.rois_erased = True

        # print("selecting rois...")

        rois_selected = self.controller.select_rois_near_point(roi_point, self.z, radius=radius)

        # print("rois selected")

        # remove the ROIs
        # for i in range(len(rois_erased)-1, -1, -1):
        #     roi = rois_erased[i]
        #     if roi in self.controller.locked_rois[self.z] or roi in self.controller.erased_rois[self.z]:
        #         del rois_erased[i]

        rois_selected = [ roi for roi in rois_selected if roi not in self.selected_rois ]

        if len(rois_selected) > 0:
            # update ROI variables
            self.selected_rois += rois_selected
            # self.controller.erased_rois[self.z] += rois_erased
            # self.last_erased_rois[self.z][-1]   += rois_erased
            # self.controller.removed_rois[self.z] = self.controller.filtered_out_rois[self.z] + self.controller.erased_rois[self.z]

            # create & show the new ROI image
            # self.calculate_roi_image(z=self.z, update_overlay=False, newly_erased_rois=rois_erased)
            # self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())
            return True
        return False

    def select_roi(self, roi_point, shift_held=False): # TODO: create roi_selected() and roi_unselected() methods for the param window
        if roi_point is not None:
            # find out which ROI to select
            manual_roi_selected, selected_roi = utilities.get_roi_containing_point(self.controller.roi_spatial_footprints[self.z], self.controller.manual_roi_spatial_footprints[self.z], roi_point, self.image.shape)
        else:
            manual_roi_selected, selected_roi = False, None

        if selected_roi is not None:
            # an ROI is selected
            if shift_held:
                self.selected_rois.append(selected_roi)
            else:
                self.selected_rois = [selected_roi]
            self.manual_roi_selected = manual_roi_selected

            print("Manual ROI selected: {}".format(self.manual_roi_selected))

            # create & show the new ROI image
            # self.calculate_roi_image(z=self.z, update_overlay=False)
            # self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=False)

            # update the param window
            # self.roi_filtering_param_widget.lock_roi_button.setEnabled(True)
            # self.roi_filtering_param_widget.enlarge_roi_button.setEnabled(True)
            # self.roi_filtering_param_widget.shrink_roi_button.setEnabled(True)
            # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
            # if selected_roi in self.controller.locked_rois[self.z]:
            #     self.roi_filtering_param_widget.lock_roi_button.setText("Unlock ROI")
            # else:
            #     self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")


            if self.controller.use_mc_video and self.controller.mc_video is not None:
                video = self.controller.mc_video
            else:
                video = self.controller.video

            if len(self.selected_rois) == 1:
                self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(True)
                if selected_roi not in self.controller.removed_rois[self.z]:
                    self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                else:
                    self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
                self.param_window.merge_rois_action.setEnabled(False)
                self.param_window.trace_rois_action.setEnabled(True)
                # self.preview_window.trace_rois_action.setEnabled(True)
            elif len(self.selected_rois) > 1:
                # self.roi_filtering_param_widget.merge_rois_button.setEnabled(True)
                # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(True)
                if not any(x in self.selected_rois for x in self.controller.removed_rois[self.z]):
                # if selected_roi not in self.controller.removed_rois[self.z]:
                    self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(True)
                    self.roi_filtering_param_widget.merge_rois_button.setEnabled(True)
                else:
                    self.roi_filtering_param_widget.unerase_selected_roi_button.setEnabled(True)
                    self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                self.param_window.erase_rois_action.setEnabled(True)
                self.param_window.merge_rois_action.setEnabled(True)
                self.param_window.trace_rois_action.setEnabled(True)
                # self.preview_window.trace_rois_action.setEnabled(True)
            else:
                self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
                self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
                self.roi_filtering_param_widget.plot_traces_button.setEnabled(False)
                self.param_window.erase_rois_action.setEnabled(False)
                self.param_window.merge_rois_action.setEnabled(False)
                self.param_window.trace_rois_action.setEnabled(False)
                # self.preview_window.trace_rois_action.setEnabled(False)
        else:
            # no ROI is selected

            self.selected_rois = []

            # create & show the new ROI image
            # self.calculate_roi_image(z=self.z, update_overlay=False)
            # self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=False)

            self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
            self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
            self.roi_filtering_param_widget.plot_traces_button.setEnabled(False)
            self.param_window.erase_rois_action.setEnabled(False)
            self.param_window.merge_rois_action.setEnabled(False)
            self.param_window.trace_rois_action.setEnabled(False)
            # self.preview_window.trace_rois_action.setEnabled(False)

            # update the param window
            # self.roi_filtering_param_widget.lock_roi_button.setEnabled(False)
            # self.roi_filtering_param_widget.enlarge_roi_button.setEnabled(False)
            # self.roi_filtering_param_widget.shrink_roi_button.setEnabled(False)
            # self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")

        # self.update_trace_plot()

    def plot_traces(self):
        self.update_trace_plot()

    def update_trace_plot(self):
        if self.trace_figure is None:
            self.trace_figure = TracePlotWindow(self)
            self.trace_figure.show()

        temporal_footprints = self.controller.roi_temporal_footprints[self.z]

        if temporal_footprints is not None:
            if self.selected_video == 0:
                temporal_footprints = temporal_footprints[:, :self.controller.video_lengths[0]]
            else:
                temporal_footprints = temporal_footprints[:, np.sum(self.controller.video_lengths[:self.selected_video]):np.sum(self.controller.video_lengths[:self.selected_video+1])]

            kept_rois = [ roi for roi in range(temporal_footprints.shape[0]) if roi not in self.controller.removed_rois[self.z] ]

            roi_spatial_footprints = self.controller.roi_spatial_footprints[self.z].toarray().reshape((self.controller.video.shape[2], self.controller.video.shape[2], self.controller.roi_spatial_footprints[self.z].shape[-1])).transpose((1, 0, 2))
            mean_traces = np.zeros((temporal_footprints.shape[0], self.controller.roi_temporal_footprints[self.z].shape[-1]))

            if self.controller.use_mc_video and self.controller.mc_video is not None:
                video_paths = self.controller.mc_video_paths
            else:
                video_paths = self.controller.video_paths

            for i in range(len(video_paths)):
                video_path = video_paths[i]

                video = imread(video_path)
                    
                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]
                    
                if i == 0:
                    final_video = video
                else:
                    final_video = np.concatenate([final_video, video], axis=0)

            for i in range(len(kept_rois)):
                roi = kept_rois[i]
                mask = roi_spatial_footprints[:, :, roi] > 0

                coords = np.nonzero(mask)
                min_y = min(coords[0])
                max_y = max(coords[0])
                min_x = min(coords[1])
                max_x = max(coords[1])

                mask = mask[min_y:max_y+1, min_x:max_x+1]

                mask[mask == 0] = np.nan
                # trace_sum = np.sum(video*mask[np.newaxis, :, :], axis=(1, 2))
                # count = np.count_nonzero(mask)
                # traces[1:, i+1] = trace_sum/count

                mean_traces[roi] = np.nanmean(final_video[:, self.z, min_y:max_y+1, min_x:max_x+1]*mask[np.newaxis, :, :], axis=(1, 2))

            if self.selected_video == 0:
                mean_traces = mean_traces[:, :self.controller.video_lengths[0]]
            else:
                mean_traces = mean_traces[:, np.sum(self.controller.video_lengths[:self.selected_video]):np.sum(self.controller.video_lengths[:self.selected_video+1])]
        else:
            mean_traces = None

        self.trace_figure.plot(temporal_footprints, self.controller.removed_rois[self.z], self.selected_rois, mean_traces=mean_traces)

        if len(self.selected_rois) <= 10:
            self.preview_window.show_roi_nums()

    def add_to_history(self, only_if_new=False):
        if (not only_if_new) or len(self.previous_erased_rois[self.z]) == 0:
            print("Adding to history.")

            # only store up to 20 history states
            if len(self.previous_erased_rois[self.z]) > 20:
                del self.previous_erased_rois[self.z][0]
            if len(self.previous_filtered_out_rois[self.z]) > 20:
                del self.previous_filtered_out_rois[self.z][0]
            if len(self.previous_images[self.z]) > 20:
                del self.previous_images[self.z][0]
            if len(self.previous_removed_rois[self.z]) > 20:
                del self.previous_removed_rois[self.z][0]
            if len(self.previous_locked_rois[self.z]) > 20:
                del self.previous_locked_rois[self.z][0]

            # store the current state
            self.previous_erased_rois[self.z].append(self.controller.erased_rois[self.z][:])
            self.previous_filtered_out_rois[self.z].append(self.controller.filtered_out_rois[self.z][:])
            self.previous_removed_rois[self.z].append(self.controller.removed_rois[self.z][:])
            self.previous_locked_rois[self.z].append(self.controller.locked_rois[self.z][:])

            self.previous_images[self.z].append(self.image.copy())

            self.roi_filtering_param_widget.undo_button.setEnabled(True)
            self.roi_filtering_param_widget.reset_button.setEnabled(True)

    def undo(self):
        # unselect any ROIs
        self.select_roi(None)

        if len(self.previous_erased_rois[self.z]) > 1:
            del self.previous_erased_rois[self.z][-1]

            self.controller.erased_rois[self.z] = self.previous_erased_rois[self.z][-1][:]
        if len(self.previous_images[self.z]) > 1:
            del self.previous_images[self.z][-1]

            self.image  = self.previous_images[self.z][-1].copy()
        if len(self.previous_locked_rois[self.z]) > 1:
            del self.previous_locked_rois[self.z][-1]

            self.controller.locked_rois[self.z] = self.previous_locked_rois[self.z][-1][:]

        self.controller.removed_rois[self.z] = self.controller.filtered_out_rois[self.z] + self.controller.erased_rois[self.z]

        # create & show the new ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=False)
        # self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())
        self.preview_window.image_plot.deselect_rois()
        self.preview_window.image_plot.erase_rois(self.controller.removed_rois[self.z])

    def reset_erase(self):
        # unselect any ROIs
        self.select_roi(None)

        if len(self.previous_erased_rois[self.z]) > 0:
            self.controller.erased_rois[self.z] = self.previous_erased_rois[self.z][0][:]
        if len(self.previous_images[self.z]) > 0:
            self.image  = self.previous_images[self.z][0].copy()
        if len(self.previous_locked_rois[self.z]) > 0:
            self.controller.locked_rois[self.z] = self.previous_locked_rois[self.z][0][:]

        self.controller.removed_rois[self.z] = self.controller.filtered_out_rois[self.z][:]

        # reset the history for this z plane
        self.reset_history(z=self.z)

        # create & show the new ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=True)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

    def erase_selected_rois(self): # TODO: call roi_unselected() method of the param window
        for label in self.selected_rois:
            self.controller.erase_roi(label, self.z)

        self.preview_window.image_plot.erase_rois(self.selected_rois)

        self.selected_rois = []

        self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
        self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
        self.roi_filtering_param_widget.plot_traces_button.setEnabled(False)
        self.param_window.erase_rois_action.setEnabled(False)
        self.param_window.merge_rois_action.setEnabled(False)
        self.param_window.trace_rois_action.setEnabled(False)

        # self.update_trace_plot()

        # create & show the new ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=True)
        # self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=False)


        # update param widget
        # self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)

        # add current state to the history
        self.add_to_history()

        self.update_trace_plot()

    def unerase_selected_rois(self):
        for label in self.selected_rois:
            self.controller.unerase_roi(label, self.z)

        rois = [ roi for roi in self.selected_rois if roi not in self.controller.filtered_out_rois[self.z] ]

        self.preview_window.image_plot.unerase_rois(rois)

        self.selected_rois = []

        self.roi_filtering_param_widget.merge_rois_button.setEnabled(False)
        self.roi_filtering_param_widget.erase_selected_roi_button.setEnabled(False)
        self.roi_filtering_param_widget.plot_traces_button.setEnabled(False)
        self.param_window.erase_rois_action.setEnabled(False)
        self.param_window.merge_rois_action.setEnabled(False)
        self.param_window.trace_rois_action.setEnabled(False)

        self.add_to_history()

        self.update_trace_plot()

    def merge_selected_rois(self):
        if len(self.selected_rois) > 1:
            if isinstance(self.controller.roi_spatial_footprints[self.z], scipy.sparse.coo_matrix) or isinstance(self.controller.roi_spatial_footprints[self.z], scipy.sparse.csc_matrix):
                f = self.controller.roi_spatial_footprints[self.z].toarray()
            else:
                f = self.controller.roi_spatial_footprints[self.z]

            # pdb.set_trace()
            rois = list(range(self.controller.roi_spatial_footprints[self.z].shape[1]))
            merged_spatial_footprint  = np.sum(f[:, self.selected_rois], axis=1)
            merged_temporal_footprint = np.sum(self.controller.roi_temporal_footprints[self.z][self.selected_rois], axis=0)[np.newaxis, :]
            merged_temporal_residual  = np.sum(self.controller.roi_temporal_residuals[self.z][self.selected_rois], axis=0)[np.newaxis, :]
            # for roi in self.selected_rois:
            #     del rois[roi]

            rois = [ roi for roi in rois if roi not in self.selected_rois]

            print(rois)
            print(self.selected_rois)

            # pdb.set_trace()

            # pdb.set_trace()
            roi_spatial_footprints  = np.concatenate((f[:, rois], np.asarray(merged_spatial_footprint)[:, np.newaxis]), axis=1)
            roi_temporal_footprints = np.concatenate((self.controller.roi_temporal_footprints[self.z][rois], merged_temporal_footprint), axis=0)
            roi_temporal_residuals  = np.concatenate((self.controller.roi_temporal_residuals[self.z][rois], merged_temporal_residual), axis=0)
            bg_spatial_footprints   = self.controller.bg_spatial_footprints[self.z]
            bg_temporal_footprints  = self.controller.bg_temporal_footprints[self.z]

            # pdb.set_trace()

            print(roi_spatial_footprints.shape[1])

            if self.controller.use_mc_video and self.controller.mc_video is not None:
                video_paths = self.controller.mc_video_paths
            else:
                video_paths = self.controller.video_paths

            for i in range(len(video_paths)):
                video_path = video_paths[i]

                video = imread(video_path)
                    
                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]
                    
                if i == 0:
                    final_video = video
                else:
                    final_video = np.concatenate([final_video, video], axis=0)

            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.do_cnmf(final_video[:, self.z, :, :], self.controller.params, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, use_multiprocessing=self.controller.use_multiprocessing)

            print(roi_spatial_footprints.shape[1])
            # print("Done!")

            self.controller.roi_spatial_footprints[self.z]  = roi_spatial_footprints.tocsc()
            self.controller.roi_temporal_footprints[self.z] = roi_temporal_footprints
            self.controller.roi_temporal_residuals[self.z]  = roi_temporal_residuals
            self.controller.bg_spatial_footprints[self.z]   = bg_spatial_footprints
            self.controller.bg_temporal_footprints[self.z]  = bg_temporal_footprints

            removed_rois      = np.array(self.controller.removed_rois[self.z])
            locked_rois       = np.array(self.controller.locked_rois[self.z])
            erased_rois       = np.array(self.controller.erased_rois[self.z])
            filtered_out_rois = np.array(self.controller.filtered_out_rois[self.z])

            # update removed ROIs
            for i in sorted(self.selected_rois):
                removed_rois[removed_rois > i]           -= 1
                locked_rois[locked_rois > i]             -= 1
                erased_rois[erased_rois > i]             -= 1
                filtered_out_rois[filtered_out_rois > i] -= 1

                if i in removed_rois:
                    index = np.where(removed_rois == i)[0][0]
                    removed_rois = np.delete(removed_rois, index)
                if i in locked_rois:
                    index = np.where(locked_rois == i)[0][0]
                    locked_rois = np.delete(locked_rois, index)
                if i in erased_rois:
                    index = np.where(erased_rois == i)[0][0]
                    erased_rois = np.delete(erased_rois, index)
                if i in filtered_out_rois:
                    index = np.where(filtered_out_rois == i)[0][0]
                    filtered_out_rois = np.delete(filtered_out_rois, index)

            self.controller.removed_rois[self.z]      = list(removed_rois)
            self.controller.locked_rois[self.z]       = list(locked_rois)
            self.controller.erased_rois[self.z]       = list(erased_rois)
            self.controller.filtered_out_rois[self.z] = list(filtered_out_rois)

            # pdb.set_trace()

            self.selected_rois = []

            # print("Done!")

            self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked(), update_overlay=True, force_update=True)

            # print("Done!")

    def lock_roi(self): # TODO: create roi_locked() and roi_unlocked() methods for the param window
        if self.selected_rois[0] not in self.controller.locked_rois[self.z]:
            self.controller.locked_rois[self.z].append(self.selected_rois[0])

            # update the param widget
            self.roi_filtering_param_widget.lock_roi_button.setText("Unlock ROI")
        else:
            index = self.controller.locked_rois[self.z].index(self.selected_rois[0])
            del self.controller.locked_rois[self.z][index]

            # update the param widget
            self.roi_filtering_param_widget.lock_roi_button.setText("Lock ROI")

        # create & show the new ROI image
        # self.calculate_roi_image(z=self.z, update_overlay=False)
        self.show_roi_image(show=self.roi_filtering_param_widget.show_rois_checkbox.isChecked())

        # add current state to the history
        self.add_to_history()

    def figure_closed(self, event):
        self.figure = None

    def save_params(self):
        self.controller.save_params()

    def set_motion_correct(self, boolean):
        self.controller.motion_correct_all_videos = boolean

class MotionCorrectThread(QThread):
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video_paths, max_shift, patch_stride, patch_overlap, use_multiprocessing=True):
        self.video_paths   = video_paths
        self.max_shift     = max_shift
        self.patch_stride  = patch_stride
        self.patch_overlap = patch_overlap
        self.use_multiprocessing          = use_multiprocessing

    def run(self):
        self.running = True

        mc_videos, mc_borders = utilities.motion_correct_multiple_videos(self.video_paths, self.max_shift, self.patch_stride, self.patch_overlap, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing)

        self.finished.emit(mc_videos, mc_borders)

        self.running = False

class ROIFindingThread(QThread):
    finished = pyqtSignal(list, list, list, list, list)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False
    
    def set_parameters(self, video_paths, masks, background_mask, invert_masks, params, mc_borders, use_mc_video, use_multiprocessing):
        self.video_paths     = video_paths
        self.masks           = masks
        self.background_mask = background_mask
        self.invert_masks    = invert_masks
        self.params          = params
        self.mc_borders      = mc_borders
        self.use_mc_video    = use_mc_video
        self.use_multiprocessing    = use_multiprocessing

    def run(self):
        self.running = True

        # roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois(self.video, self.video_path, self.params, masks=self.masks, background_mask=self.background_mask, mc_borders=self.mc_borders, progress_signal=self.progress, thread=self)
        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_multiple_videos(self.video_paths, self.params, use_mc_video=self.use_mc_video, masks=self.masks, background_mask=self.background_mask, mc_borders=self.mc_borders, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing)

        self.finished.emit(roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints)

        self.running = False

class ProcessVideosThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video_paths, roi_spatial_footprints, roi_temporal_footprints, bg_spatial_footprints, bg_temporal_footprints, motion_correct, max_shift, patch_stride, patch_overlap, apply_blur, params):
        self.video_paths             = video_paths
        self.roi_spatial_footprints  = roi_spatial_footprints
        self.roi_temporal_footprints = roi_temporal_footprints
        self.bg_spatial_footprints   = bg_spatial_footprints
        self.bg_temporal_footprints  = bg_temporal_footprints
        self.motion_correct          = motion_correct
        self.max_shift               = max_shift
        self.patch_stride            = patch_stride
        self.patch_overlap           = patch_overlap
        self.params                  = params

    def run(self):
        self.running = True

        video_shape       = None

        for i in range(len(self.video_paths)):
            video_path = self.video_paths[i]

            # open video
            base_name = os.path.basename(video_path)
            if base_name.endswith('.npy'):
                video = np.load(video_path)
            elif base_name.endswith('.tif') or base_name.endswith('.tiff'):
                video = imread(video_path)

            print("Processing {}.".format(base_name))

            if len(video.shape) < 3:
                print("Skipping, this file is not a video -- not enough dimensions.")
                continue

            if len(video.shape) == 3:
                # add z dimension
                video = video[:, np.newaxis, :, :]

            if video_shape is None and not self.motion_correct:
                video_shape = video.shape

            # video = np.nan_to_num(video).astype(np.float32)

            # # figure out the dynamic range of the video
            # max_value = np.amax(video)
            # if max_value > 2047:
            #     video_max = 4095
            # elif max_value > 1023:
            #     video_max = 2047
            # elif max_value > 511:
            #     video_max = 1023
            # elif max_value > 255:
            #     video_max = 511
            # elif max_value > 1:
            #     video_max = 255
            # else:
            #     video_max = 1
            
            name           = os.path.splitext(base_name)[0]
            directory      = os.path.dirname(video_path)
            video_dir_path = os.path.join(directory, name)

            # make a folder to hold the results
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + (1/3))/len(self.video_paths)))

            if self.motion_correct:
                print("Performing motion correction...")
                mc_video, mc_borders = utilities.motion_correct(video, video_path, self.max_shift, self.patch_stride, self.patch_overlap)

                if video_shape is None:
                    video_shape = mc_video.shape

                # np.save(os.path.join(video_dir_path, '{}_motion_corrected.npy'.format(name)), mc_video)

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + (2/3))/len(self.video_paths)))

            rois = self.rois[:]

            # print(np.unique(self.rois))

            if self.motion_correct:
                vid = mc_video
            else:
                vid = video

            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_refine(vid, video_path, self.params, masks=None, background_mask=None, mc_borders=mc_borders, roi_spatial_footprints=self.roi_spatial_footprints, roi_temporal_footprints=self.roi_temporal_footprints, roi_temporal_residuals=self.roi_temporal_residuals, bg_spatial_footprints=self.bg_spatial_footprints, bg_temporal_footprints=self.bg_temporal_footprints)

            results = [ {} for z in range(video.shape[1]) ]

            for z in range(video.shape[1]):
                np.save(os.path.join(video_dir_path, 'z_{}_rois.npy'.format(z)), rois[z])

                print("Calculating ROI activities for z={}...".format(z))
                centroids, traces = utilities.calculate_centroids_and_traces(rois[z], vid[:, z, :, :])

                print("Saving CSV for z={}...".format(z))

                roi_nums = np.unique(rois[z]).tolist()
                # remove ROI #0 (this is the background)
                try:
                    index = roi_nums.index(0)
                    del roi_nums[index]
                except:
                    pass

                with open(os.path.join(video_dir_path, 'z_{}_traces.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow([''] + [ "ROI #{}".format(roi) for roi in roi_nums ])

                    for i in range(traces.shape[0]):
                        writer.writerow([i+1] + traces[i].tolist())

                with open(os.path.join(video_dir_path, 'z_{}_centroids.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['Label', 'X', 'Y'])

                    for i in range(centroids.shape[0]):
                        writer.writerow(["ROI #{}".format(roi_nums[i])] + centroids[i].tolist())

                # print("Done.")

            if not self.running:
                self.running = False

                return

            self.progress.emit(int(100.0*float(i + 1)/len(self.video_paths)))

            np.savez(os.path.join(video_dir_path, '{}_roi_traces.npz'.format(os.path.splitext(video_path)[0])), results)

        self.finished.emit()

        self.running = False

class TracePlotWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self)

        self.parent = parent

        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(0)

        self.setWindowTitle("ROIs")

        # a figure instance to plot on
        self.figure = Figure(figsize=(10, 4))

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        # self.button = QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # set the layout
        # layout = QVBoxLayout()
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        # # layout.addWidget(self.button)
        # self.setLayout(layout)

    def plot(self, roi_temporal_footprints, removed_rois=[], selected_rois=[], mean_traces=None):
        ''' plot some random stuff '''
        # random data
        # data = [random.random() for i in range(10)]

        # instead of ax.hold(False)
        self.figure.clear()

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])

        # create an axis
        ax0 = self.figure.add_subplot(gs[0])

        kept_rois = [ roi for roi in range(roi_temporal_footprints.shape[0]) if roi not in removed_rois ]

        ax0.imshow(roi_temporal_footprints[kept_rois], cmap='inferno')

        ax0.set_xlabel("Frame #")
        ax0.set_ylabel("ROI #")

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        ax1 = self.figure.add_subplot(gs[1])

        if len(selected_rois) > 0:
            max_value = np.amax(roi_temporal_footprints)

            if mean_traces is not None:
                max_value_mean = np.amax(mean_traces)

            # plot data
            for i in range(len(selected_rois)):
                print(i)
                roi = selected_rois[i]
                print(roi, roi_temporal_footprints.shape)
                y_offset = 0

                if i < len(colors):
                    color = colors[i]
                else:
                    color = colors[-1]

                color=next(ax1._get_lines.prop_cycler)['color']
                ax1.plot(roi_temporal_footprints[roi]/max_value + y_offset, c=color, label="ROI #{}".format(roi))
                if mean_traces is not None:
                    ax1.plot(mean_traces[roi]/max_value_mean + y_offset, c=color, linestyle=':')

        ax1.set_xlabel("Frame #")
        ax1.set_ylabel("Fluorescence")

        if len(selected_rois) <= 10:
            ax1.legend()

        self.figure.tight_layout()

        # refresh canvas
        self.canvas.draw()

    def closeEvent(self, ce):
        self.parent.trace_figure = None
        ce.accept()