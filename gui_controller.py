import os
import numpy as np
import skimage.external.tifffile as tifffile
import cv2
import csv
import scipy

import utilities
import param_window
from preview_window import PreviewWindow

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

class GUIController():
    def __init__(self, controller):
        self.controller = controller
        
        # create windows
        self.param_window   = param_window.ParamWindow(self)
        self.preview_window = PreviewWindow(self)

        # initialize variables
        self.reset_variables()

        self.roi_finding_mode     = "cnmf" # which algorithm to use to find ROIs -- "cnmf" / "suite2p"

        # initialize state variables
        self.closing            = False # whether the user has requested to close the application
        self.roi_finding_queued = False

        # initialize thread variables
        self.motion_correction_thread = None
        self.roi_finding_thread       = None
        self.video_processing_thread  = None

        # set the mode -- "loading" / "motion_correcting" / "roi_finding" / "roi_filtering"
        self.mode = "loading"

        # set the current z plane to 0
        self.z = 0

    def reset_variables(self):
        self.video                = None   # currently loaded video
        self.adjusted_video       = None   # gamma- and contrast-adjusted video
        self.mean_images          = []     # mean images for all z planes
        self.adjusted_mean_images = []     # gamma- and contrast-adjusted mean images
        self.selected_rois        = []     # ROIs currently selected by the user
        self.show_rois            = False  # whether to show ROIs
        self.show_zscore          = True   # whether to show z-score vs. raw fluorescence
        self.selected_video       = 0      # which video is selected
        self.group_num            = 0      # group number of currently loaded video
        self.video_max            = 1      # dynamic range of currently loaded video

    def roi_spatial_footprints(self):
        if len(self.controller.roi_spatial_footprints.keys()) > 0:
            return self.controller.roi_spatial_footprints[self.group_num][self.z]
        else:
            return None

    def roi_temporal_footprints(self):
        if len(self.controller.roi_temporal_footprints.keys()) > 0:
            return self.controller.roi_temporal_footprints[self.group_num][self.z]
        else:
            return None

    def removed_rois(self):
        if len(self.controller.removed_rois.keys()) > 0:
            return self.controller.removed_rois[self.group_num][self.z]
        else:
            return None

    def filtered_out_rois(self):
        if len(self.controller.filtered_out_rois.keys()) > 0:
            return self.controller.filtered_out_rois[self.group_num][self.z]
        else:
            return None

    def bg_temporal_footprints(self):
        if len(self.controller.bg_temporal_footprints.keys()) > 0:
            return self.controller.bg_temporal_footprints[self.group_num][self.z]
        else:
            return None

    def selected_group_video_nums(self):
        group_nums = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]

        return group_nums

    def selected_group_video_paths(self):
        video_paths   = self.video_paths()
        group_indices = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]
        group_paths = [ video_paths[i] for i in group_indices ]

        return group_paths

    def selected_group_video_lengths(self):
        group_indices = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]
        group_lengths = [ self.controller.video_lengths[i] for i in group_indices ]

        return group_lengths

    def selected_video_path(self):
        video_paths = self.video_paths()

        return video_paths[self.selected_video]

    def video_paths(self):
        if self.controller.use_mc_video and len(self.controller.mc_video_paths) > 0:
            video_paths = self.controller.mc_video_paths
        else:
            video_paths = self.controller.video_paths

        return video_paths

    def video_groups(self):
        return self.controller.video_groups

    def params(self):
        return self.controller.params

    def selected_video_mean_image(self):
        return self.mean_images[self.selected_video]

    def rois_exist(self):
        return len(self.controller.roi_spatial_footprints.keys()) > 0

    def import_videos(self):
        if len(self.controller.mc_video_paths) > 0 or len(self.controller.roi_spatial_footprints) > 0:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Warning)
            message_box.setContentsMargins(5, 5, 5, 5)

            message_box.setText("Adding videos will throw out any motion correction or ROI finding results. Continue?")
            message_box.setWindowTitle("")
            message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            return_value = message_box.exec_()
            if return_value == QMessageBox.Cancel:
                return
            else:
                self.controller.reset_motion_correction_variables()
                self.controller.reset_roi_finding_variables()
                self.controller.reset_roi_filtering_variables()

        # let user pick video file(s)
        if pyqt_version == 4:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff)')

            video_paths = [ str(path) for path in video_paths ]
        elif pyqt_version == 5:
            video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tif *.tiff)')[0]

        # import the videos
        if video_paths is not None and len(video_paths) > 0:
            self.controller.import_videos(video_paths)

            if self.video is None:
                # open the first video for previewing
                self.selected_video = 0
                self.open_video(self.controller.video_paths[0])

                # set z to 0 if necessary
                if self.z >= self.video.shape[1]:
                    self.z = 0

                self.play_video()

                self.preview_window.reset_zoom()

            # notify the param window
            self.param_window.videos_imported(video_paths)

        # make the param window the front-most window
        self.param_window.activateWindow()

    def open_video(self, video_path):
        # open the video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = tifffile.imread(video_path)
        else:
            return

        if len(self.video.shape) < 3:
            print("Error: Opened file is not a video -- not enough dimensions.")
            return

        # figure out the dynamic range of the video
        max_value = np.amax(self.video)
        if max_value > 2047:
            self.video_max = 4095
        elif max_value > 1023:
            self.video_max = 2047
        elif max_value > 511:
            self.video_max = 1023
        elif max_value > 255:
            self.video_max = 511
        elif max_value > 1:
            self.video_max = 255
        else:
            self.video_max = 1

        print("Video max: {}.".format(self.video_max))

        # set the path to the previewed video
        self.video_path = video_path

        if len(self.video.shape) == 3:
            # add a z dimension
            self.video = self.video[:, np.newaxis, :, :]

        # flip video 90 degrees to match what is shown in Fiji
        self.video = self.video.transpose((0, 1, 3, 2))

        # remove NaNs
        self.video = np.nan_to_num(self.video).astype(np.float32)

        print("Opened video with shape {}.".format(self.video.shape))

        # calculate mean images
        self.update_mean_images()

        # notify the param window and preview window
        self.param_window.video_opened(max_z=self.video.shape[1]-1, z=self.z)

    def update_mean_images(self):
        self.mean_images = [ (utilities.mean(self.video, z)/self.video_max)*self.video_max for z in range(self.video.shape[1]) ]

    def update_adjusted_mean_images(self):
        self.adjusted_mean_images = [ utilities.adjust_gamma(utilities.adjust_contrast(mean_image, self.controller.params['contrast']), self.controller.params['gamma']) for mean_image in self.mean_images ]

    def update_adjusted_video(self):
        self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.video[:, self.z, :, :], self.controller.params['contrast']), self.controller.params['gamma'])

    def adjusted_frame(self, video):
        return utilities.adjust_gamma(utilities.adjust_contrast(video[self.preview_window.frame_num, self.z, :, :], self.controller.params['contrast']), self.controller.params['gamma'])

    def video_selected(self, index, force=False):
        if index is not None and (index != self.selected_video or force):
            print("Video #{} selected.".format(index+1))

            self.selected_video = index
            self.group_num      = self.controller.video_groups[self.selected_video]

            self.open_video(self.selected_video_path())
            self.update_adjusted_mean_images()

            if self.mode in ("loading", "motion_correcting"):
                self.play_video()
            else:
                self.show_roi_image(update_overlay=True)

            self.preview_window.plot_tail_angles(self.controller.tail_angles[self.selected_video], self.controller.params['tail_data_fps'], self.controller.params['imaging_fps'])
            self.update_trace_plot()
    def play_video(self):
        # calculate gamma- and contrast-adjusted video and mean images
        self.update_adjusted_video()
        self.update_adjusted_mean_images()

        self.preview_window.play_video(self.adjusted_video, self.selected_video_path(), self.controller.params['fps'])
        self.preview_window.plot_mean_image(self.adjusted_mean_images[self.z], self.video_max)
        self.preview_window.show_plot()

    def videos_rearranged(self, old_indices, groups):
        self.controller.video_groups  = groups
        self.controller.video_paths   = [ self.controller.video_paths[i] for i in old_indices ]
        self.controller.video_lengths = [ self.controller.video_lengths[i] for i in old_indices ]

        if len(self.controller.mc_video_paths) > 0:
            self.controller.mc_video_paths = [ self.controller.mc_video_paths[i] for i in old_indices ]

        self.selected_video = old_indices.index(self.selected_video)
        self.group_num      = self.controller.video_groups[self.selected_video]

    def save_rois(self):
        self.process_all_videos()
        self.process_all_videos()

    def load_rois(self):
        # let the user pick saved ROIs
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        if load_path is not None and len(load_path) > 0:
            self.controller.load_rois(load_path, group_num=self.group_num, video_path=self.selected_video_path())

            self.param_window.rois_loaded()
            self.show_rois = True

            self.preview_window.timer.stop()

            # show ROI filtering parameters
            self.show_roi_filtering_params(update_overlay=True)

    def remove_videos_at_indices(self, indices):
        self.controller.remove_videos_at_indices(indices)

        if len(self.controller.video_paths) == 0:
            print("All videos removed.")

            # reset variables
            self.reset_variables()

            # reset param window & preview window to their initial states
            self.param_window.set_initial_state()
            self.preview_window.set_initial_state()
        else:
            # open the newest video at index 0
            self.video_selected(0, force=True)

    def process_all_videos(self):
        save_directory = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))

        video_paths = self.video_paths()

        for i in range(len(video_paths)):
            video_path = self.controller.video_paths[i]

            base_name      = os.path.basename(video_path)
            name           = os.path.splitext(base_name)[0]
            directory      = os.path.dirname(video_path)
            video_dir_path = os.path.join(save_directory, name)

            # make a folder to hold the results
            if not os.path.exists(video_dir_path):
                os.makedirs(video_dir_path)

            video = tifffile.imread(video_path)

            if len(video.shape) == 3:
                # add z dimension
                video = video[:, np.newaxis, :, :]

            group_num = self.controller.video_groups[i]

            roi_spatial_footprints  = self.controller.roi_spatial_footprints[group_num]
            roi_temporal_footprints = self.controller.roi_temporal_footprints[group_num]
            roi_temporal_residuals  = self.controller.roi_temporal_residuals[group_num]
            bg_spatial_footprints   = self.controller.bg_spatial_footprints[group_num]
            bg_temporal_footprints  = self.controller.bg_temporal_footprints[group_num]
            
            discarded_rois = self.controller.discarded_rois[group_num]
            removed_rois   = self.controller.removed_rois[group_num]
            locked_rois    = self.controller.locked_rois[group_num]

            # save centroids & traces
            for z in range(video.shape[1]):
                print("Calculating ROI activities for z={}...".format(z))

                centroids = np.zeros((roi_spatial_footprints[z].shape[-1], 2))
                kept_rois = [ roi for roi in range(roi_spatial_footprints[z].shape[-1]) if (roi not in removed_rois[z]) or (roi in locked_rois[z]) ]

                footprints_2d = roi_spatial_footprints[z].toarray().reshape((video.shape[2], video.shape[3], roi_spatial_footprints[z].shape[-1]))

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

                temporal_footprints = roi_temporal_footprints[z]

                group_indices = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == group_num ]
                group_paths   = [ self.controller.video_paths[i] for i in group_indices ]
                group_lengths = [ self.controller.video_lengths[i] for i in group_indices ]
                
                index = group_paths.index(video_path)

                if index == 0:
                    temporal_footprints = temporal_footprints[:, :group_lengths[0]]
                else:
                    temporal_footprints = temporal_footprints[:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]

                traces = temporal_footprints[kept_rois]
                centroids = centroids[kept_rois]

                print("Saving CSV for z={}...".format(z))

                with open(os.path.join(video_dir_path, 'z_{}_traces.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['ROI #'] + [ "Frame {}".format(frame) for frame in range(traces.shape[1]) ])

                    for j in range(traces.shape[0]):
                        writer.writerow(['{}'.format(kept_rois[j])] + traces[j].tolist())

                with open(os.path.join(video_dir_path, 'z_{}_centroids.csv'.format(z)), 'w') as file:
                    writer = csv.writer(file)

                    writer.writerow(['Label', 'X', 'Y'])

                    for j in range(centroids.shape[0]):
                        writer.writerow(["ROI #{}".format(kept_rois[j]+1)] + centroids[j].tolist())

                # save ROIs
                self.controller.save_rois(os.path.join(video_dir_path, 'roi_data.npy'), group_num=group_num, video_path=video_path)

                print("Done.")

    def motion_correct_and_find_rois(self):
        self.roi_finding_queued = True

        self.motion_correct_video()

    def motion_correct_video(self):
        # create a motion correction thread
        self.motion_correction_thread = MotionCorrectThread(self.param_window)

        # set the parameters of the motion correction thread
        self.motion_correction_thread.set_parameters(self.controller.video_paths, self.controller.video_groups, int(self.controller.params["max_shift"]), int(self.controller.params["patch_stride"]), int(self.controller.params["patch_overlap"]), use_multiprocessing=self.controller.use_multiprocessing)
        
        self.motion_correction_thread.progress.connect(self.motion_correction_progress)
        self.motion_correction_thread.finished.connect(self.motion_correction_ended)

        # start the thread
        self.motion_correction_thread.start()

        # notify the param window
        self.param_window.motion_correction_started()

    def motion_correction_progress(self, group_num):
        # notify the param window
        self.param_window.update_motion_correction_progress(group_num)

    def motion_correction_ended(self, mc_videos, mc_borders):
        mc_video_paths = []
        for i in range(len(mc_videos)):
            video_path    = self.controller.video_paths[i]
            directory     = os.path.dirname(video_path)
            filename      = os.path.basename(video_path)
            mc_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_mc.tif")
            
            # save the motion-corrected video
            tifffile.imsave(mc_video_path, mc_videos[i])
            
            mc_video_paths.append(mc_video_path)

        self.controller.mc_video_paths = mc_video_paths
        self.controller.mc_borders     = mc_borders

        self.param_window.motion_correction_ended()

        self.set_use_mc_video(True)

        if self.roi_finding_queued:
            self.find_rois()

    def set_use_mc_video(self, use_mc_video):
        self.controller.use_mc_video = use_mc_video

        self.open_video(self.selected_video_path())
        
        if self.mode in ("loading", "motion_correcting"):
            self.play_video()
        else:
            self.show_roi_image(update_overlay=True)

        self.param_window.set_video_paths(self.video_paths())

    def find_rois(self):
        video_paths = self.video_paths()

        # create an ROI finding thread
        self.roi_finding_thread = ROIFindingThread(self.param_window)

        # set the parameters of the ROI finding thread
        self.roi_finding_thread.set_parameters(video_paths, self.controller.video_groups, self.controller.params, self.controller.mc_borders, self.controller.use_multiprocessing, method=self.roi_finding_mode)

        self.roi_finding_thread.progress.connect(self.roi_finding_progress)
        self.roi_finding_thread.finished.connect(self.roi_finding_ended)

        # start the thread
        self.roi_finding_thread.start()

        # notify the param window
        self.param_window.roi_finding_started()

    def roi_finding_progress(self, group_num):
        # notify the param window
        self.param_window.update_roi_finding_progress(group_num)

    def roi_finding_ended(self, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints):
        self.selected_rois = []

        self.controller.roi_spatial_footprints  = roi_spatial_footprints
        self.controller.roi_temporal_footprints = roi_temporal_footprints
        self.controller.roi_temporal_residuals  = roi_temporal_residuals
        self.controller.bg_spatial_footprints   = bg_spatial_footprints
        self.controller.bg_temporal_footprints  = bg_temporal_footprints
        self.controller.filtered_out_rois       = { group_num: [ [] for z in range(self.video.shape[1]) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.discarded_rois          = { group_num: [ [] for z in range(self.video.shape[1]) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.removed_rois            = { group_num: [ [] for z in range(self.video.shape[1]) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.locked_rois             = { group_num: [ [] for z in range(self.video.shape[1]) ] for group_num in np.unique(self.controller.video_groups) }

        # notify the param window
        self.param_window.roi_finding_ended()

        self.show_rois = True

        self.show_roi_image(update_overlay=True)

        self.roi_finding_queued = False

    def show_video_loading_params(self):
        self.param_window.tab_widget.setCurrentIndex(0)

        self.mode = "loading"

        self.preview_window.timer.stop()

        # play the video
        if self.video is not None:
            self.play_video()

    def show_motion_correction_params(self):
        self.param_window.tab_widget.setCurrentIndex(1)

        self.mode = "motion_correcting"

        self.preview_window.timer.stop()

        # play the video
        if self.video is not None:
            self.play_video()

    def show_roi_finding_params(self):
        self.param_window.tab_widget.setCurrentIndex(2)

        self.mode = "roi_finding"

        self.preview_window.timer.stop()

        self.show_roi_image(update_overlay=False)

        self.preview_window.setWindowTitle(os.path.basename(self.selected_video_path()))

    def show_roi_filtering_params(self, update_overlay=False):
        self.param_window.tab_widget.setCurrentIndex(3)

        self.mode = "roi_filtering"
        
        self.preview_window.timer.stop()

        self.show_roi_image(update_overlay=update_overlay)

        self.preview_window.setWindowTitle(os.path.basename(self.selected_video_path()))

    def close_all(self):
        self.closing = True

        # close param & preview windows
        self.param_window.close()
        self.preview_window.close()

        # save the current parameters
        self.save_params()

    def preview_contrast(self, contrast):
        self.controller.params['contrast'] = contrast

        if self.mode in ("loading", "motion_correcting"):
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            adjusted_frame = self.adjusted_frame(self.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("contrast", contrast)

    def preview_gamma(self, gamma):
        self.controller.params['gamma'] = gamma

        if self.mode in ("loading", "motion_correcting"):
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            adjusted_frame = self.adjusted_frame(self.video)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        elif self.mode in ("roi_finding", "roi_filtering"):
            self.update_param("gamma", gamma)

    def update_param(self, param, value):
        print("Setting parameter '{}' to {}.".format(param, value))

        # update the parameter
        if param in self.controller.params.keys():
            self.controller.params[param] = value

        if self.mode in ("loading", "motion_correcting"):
            if param in ("contrast, gamma"):
                self.preview_window.timer.stop()

                self.play_video()
            elif param == "fps":
                # update the FPS of the preview window
                self.preview_window.set_fps(self.controller.params['fps'])
            elif param == "z":
                self.z = value

                self.selected_rois = []
                self.preview_window.clear_text_and_outline_items()
                self.update_trace_plot()

                self.preview_window.timer.stop()

                self.play_video()
        elif self.mode == "roi_finding":
            if param in ("contrast, gamma"):
                self.update_adjusted_mean_images()

                # show the ROI image
                self.show_roi_image(update_overlay=False, recreate_roi_images=True)
            elif param == "z":
                self.z = value

                self.selected_rois = []
                self.preview_window.clear_text_and_outline_items()
                self.update_trace_plot()

                # show the ROI image
                self.show_roi_image(update_overlay=True)
        elif self.mode == "roi_filtering":
            if param in ("contrast, gamma"):
                self.update_adjusted_mean_images()

                # show the ROI image
                self.show_roi_image(update_overlay=False, recreate_roi_images=True)
            if param == "z":
                self.z = value

                self.selected_rois = []
                self.preview_window.clear_text_and_outline_items()
                self.update_trace_plot()

                # show the ROI image
                self.show_roi_image(update_overlay=True)
            elif param in ("min_area", "max_area", "min_circ", "max_circ"):
                pass

    def set_use_multiprocessing(self, use_multiprocessing):
        self.controller.use_multiprocessing = use_multiprocessing

    def set_show_rois(self, show_rois):
        self.show_rois = show_rois

        self.show_roi_image(update_overlay=False, recreate_roi_images=False)

    def set_show_zscore(self, show_zscore):
        self.show_zscore = show_zscore

        self.show_roi_image(update_overlay=False)

    def show_roi_image(self, update_overlay=False, recreate_roi_images=True):
        self.preview_window.plot_image(self.adjusted_mean_images[self.z], video_max=255.0, show_rois=self.show_rois, update_overlay=update_overlay, recreate_roi_images=recreate_roi_images)

    def filter_rois(self):
        self.controller.filter_rois(group_num=self.group_num)

        self.selected_rois = []

        self.show_roi_image(update_overlay=True)

    def select_roi(self, roi_point, ctrl_held=False):
        if self.mode == "loading" or self.mode == "motion_correcting":
            return

        if roi_point is not None:
            group_num = self.controller.video_groups[self.selected_video]

            # find out which ROI to select
            selected_roi = utilities.get_roi_containing_point(self.controller.roi_spatial_footprints[self.group_num][self.z], roi_point, self.mean_images[self.z].shape)

            if selected_roi is not None:
                if ctrl_held:
                    self.selected_rois.append(selected_roi)
                else:
                    self.selected_rois = [selected_roi]

                print("ROI #{} selected.".format(selected_roi))

                if len(self.selected_rois) == 1:
                    self.param_window.single_roi_selected(discarded=selected_roi in self.removed_rois())
                elif len(self.selected_rois) > 1:
                    self.param_window.multiple_rois_selected(discarded=any(x in self.selected_rois for x in self.removed_rois()), merge_enabled=self.bg_temporal_footprints() is not None)
            else:
                # no ROI is selected

                self.selected_rois = []

                self.param_window.no_rois_selected()

    def load_tail_angles(self): # TODO: Ask the user for FPS of tail traces and calcium traces
        if pyqt_version == 4:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved tail angle data.', '', 'CSV (*.csv)')
        elif pyqt_version == 5:
            load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved tail angle data.', '', 'CSV (*.csv)')[0]

        if load_path is not None and len(load_path) > 0:
            tail_angles = np.genfromtxt(load_path, delimiter=",")[:, 1:]

            self.controller.tail_angles[self.selected_video] = tail_angles

            tail_data_fps, imaging_fps, ok = TailTraceParametersDialog.getParameters(None, self.controller.params['tail_data_fps'], self.controller.params['imaging_fps'])

            if ok:
                self.controller.params['tail_data_fps'] = tail_data_fps
                self.controller.params['imaging_fps']   = imaging_fps

                success = self.preview_window.plot_tail_angles(self.controller.tail_angles[self.selected_video], self.controller.params['tail_data_fps'], self.controller.params['imaging_fps'])

                if success:
                    self.param_window.set_imaging_fps(self.controller.params['imaging_fps'])
                else:
                    message_box = QMessageBox()
                    message_box.setIcon(QMessageBox.Critical)
                    message_box.setContentsMargins(5, 5, 5, 5)

                    message_box.setText("Could not load tail trace. Tail trace data must be at least as long (in seconds) as calcium imaging data.")
                    message_box.setWindowTitle("")
                    message_box.setStandardButtons(QMessageBox.Ok)

                    return_value = message_box.exec_()
    def discard_selected_rois(self):
        for roi in self.selected_rois:
            self.controller.discard_roi(roi, self.z, self.group_num)

        self.selected_rois = []

        self.param_window.no_rois_selected()

        self.show_roi_image()

    def discard_all_rois(self):
        self.controller.discarded_rois[self.group_num][self.z] = np.arange(self.controller.roi_spatial_footprints[self.group_num][self.z].shape[1]).tolist()
        self.controller.removed_rois[self.group_num][self.z]   = self.controller.discarded_rois[self.group_num][self.z][:]

        self.selected_rois = []

        self.param_window.no_rois_selected()

        self.show_roi_image()

    def keep_selected_rois(self):
        for roi in self.selected_rois:
            self.controller.keep_roi(roi, self.z, self.group_num)

        self.selected_rois = []

        self.param_window.no_rois_selected()

        self.show_roi_image()

    def merge_selected_rois(self):
        if len(self.selected_rois) > 1:
            roi_spatial_footprints  = self.controller.roi_spatial_footprints[self.group_num][self.z]
            roi_temporal_footprints = self.controller.roi_temporal_footprints[self.group_num][self.z]
            roi_temporal_residuals  = self.controller.roi_temporal_residuals[self.group_num][self.z]
            bg_spatial_footprints   = self.controller.bg_spatial_footprints[self.group_num][self.z]
            bg_temporal_footprints  = self.controller.bg_temporal_footprints[self.group_num][self.z]

            if isinstance(roi_spatial_footprints, scipy.sparse.coo_matrix) or isinstance(roi_spatial_footprints, scipy.sparse.csc_matrix):
                roi_spatial_footprints = roi_spatial_footprints.toarray()

            rois = list(range(roi_spatial_footprints.shape[1]))
            merged_spatial_footprint  = np.sum(roi_spatial_footprints[:, self.selected_rois], axis=1)
            merged_temporal_footprint = np.sum(roi_temporal_footprints[self.selected_rois], axis=0)[np.newaxis, :]
            merged_temporal_residual  = np.sum(roi_temporal_residuals[self.selected_rois], axis=0)[np.newaxis, :]

            rois = [ roi for roi in rois if roi not in self.selected_rois]

            roi_spatial_footprints  = np.concatenate((roi_spatial_footprints[:, rois], np.asarray(merged_spatial_footprint)[:, np.newaxis]), axis=1)
            roi_temporal_footprints = np.concatenate((roi_temporal_footprints[rois], merged_temporal_footprint), axis=0)
            roi_temporal_residuals  = np.concatenate((roi_temporal_residuals[rois], merged_temporal_residual), axis=0)

            video_paths = self.controller.video_paths_in_group(self.video_paths(), self.group_num)

            for i in range(len(video_paths)):
                video_path = video_paths[i]

                video = tifffile.imread(video_path)
                    
                if len(video.shape) == 3:
                    # add a z dimension
                    video = video[:, np.newaxis, :, :]
                    
                if i == 0:
                    final_video = video
                else:
                    final_video = np.concatenate([final_video, video], axis=0)

            roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.perform_cnmf(final_video[:, self.z, :, :], self.controller.params, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints, use_multiprocessing=self.controller.use_multiprocessing)

            self.controller.roi_spatial_footprints[self.group_num][self.z]  = roi_spatial_footprints.tocsc()
            self.controller.roi_temporal_footprints[self.group_num][self.z] = roi_temporal_footprints
            self.controller.roi_temporal_residuals[self.group_num][self.z]  = roi_temporal_residuals
            self.controller.bg_spatial_footprints[self.group_num][self.z]   = bg_spatial_footprints
            self.controller.bg_temporal_footprints[self.group_num][self.z]  = bg_temporal_footprints

            removed_rois      = np.array(self.controller.removed_rois[self.group_num][self.z])
            locked_rois       = np.array(self.controller.locked_rois[self.group_num][self.z])
            discarded_rois    = np.array(self.controller.discarded_rois[self.group_num][self.z])
            filtered_out_rois = np.array(self.controller.filtered_out_rois[self.group_num][self.z])

            # update removed ROIs
            for i in sorted(self.selected_rois):
                removed_rois[removed_rois > i]           -= 1
                locked_rois[locked_rois > i]             -= 1
                discarded_rois[discarded_rois > i]       -= 1
                filtered_out_rois[filtered_out_rois > i] -= 1

                if i in removed_rois:
                    index = np.where(removed_rois == i)[0][0]
                    removed_rois = np.delete(removed_rois, index)
                if i in locked_rois:
                    index = np.where(locked_rois == i)[0][0]
                    locked_rois = np.delete(locked_rois, index)
                if i in discarded_rois:
                    index = np.where(discarded_rois == i)[0][0]
                    discarded_rois = np.delete(discarded_rois, index)
                if i in filtered_out_rois:
                    index = np.where(filtered_out_rois == i)[0][0]
                    filtered_out_rois = np.delete(filtered_out_rois, index)

            self.controller.removed_rois[self.group_num][self.z]      = list(removed_rois)
            self.controller.locked_rois[self.group_num][self.z]       = list(locked_rois)
            self.controller.discarded_rois[self.group_num][self.z]    = list(discarded_rois)
            self.controller.filtered_out_rois[self.group_num][self.z] = list(filtered_out_rois)

            self.selected_rois = []

            self.param_window.no_rois_selected()

            self.preview_window.clear_text_and_outline_items()

            self.update_trace_plot()

            self.show_roi_image(update_overlay=True)

    def update_trace_plot(self):
        temporal_footprints = self.roi_temporal_footprints()

        group_indices = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]
        group_paths   = [ self.controller.video_paths[i] for i in group_indices ]
        group_lengths = [ self.controller.video_lengths[i] for i in group_indices ]
        
        index = group_paths.index(self.selected_video_path())

        if len(self.controller.roi_temporal_footprints.keys()) > 0:
            if index == 0:
                temporal_footprints = temporal_footprints[:, :group_lengths[0]]
            else:
                temporal_footprints = temporal_footprints[:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]
        else:
            temporal_footprints = None

        self.preview_window.plot_traces(temporal_footprints, self.selected_rois)

    def save_params(self):
        self.controller.save_params()

    def set_motion_correct(self, boolean):
        self.controller.motion_correct_all_videos = boolean

class MotionCorrectThread(QThread):
    finished = pyqtSignal(list, dict)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False

    def set_parameters(self, video_paths, groups, max_shift, patch_stride, patch_overlap, use_multiprocessing=True):
        self.video_paths         = video_paths
        self.groups              = groups
        self.max_shift           = max_shift
        self.patch_stride        = patch_stride
        self.patch_overlap       = patch_overlap
        self.use_multiprocessing = use_multiprocessing

    def run(self):
        self.running = True

        mc_videos, mc_borders = utilities.motion_correct_multiple_videos(self.video_paths, self.groups, self.max_shift, self.patch_stride, self.patch_overlap, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing)

        self.finished.emit(mc_videos, mc_borders)

        self.running = False

class ROIFindingThread(QThread):
    finished = pyqtSignal(dict, dict, dict, dict, dict)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False
    
    def set_parameters(self, video_paths, groups, params, mc_borders, use_multiprocessing, method="cnmf"):
        self.video_paths         = video_paths
        self.groups              = groups
        self.params              = params
        self.mc_borders          = mc_borders
        self.use_multiprocessing = use_multiprocessing
        self.method              = method

    def run(self):
        self.running = True

        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_multiple_videos(self.video_paths, self.groups, self.params, mc_borders=self.mc_borders, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing, method=self.method)

        self.finished.emit(roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints)

        self.running = False

class TailTraceParametersDialog(QDialog):
    def __init__(self, parent, tail_fps, imaging_fps):
        super(TailTraceParametersDialog, self).__init__(parent)

        param_layout = QVBoxLayout(self)
        param_layout.setContentsMargins(0, 0, 0, 0)

        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setSpacing(5)
        param_layout.addWidget(widget)

        label = param_window.HoverLabel("Tail trace FPS:")
        label.setHoverMessage("Tail trace frame rate (frames per second).")
        layout.addWidget(label)

        self.tail_fps_textbox = QLineEdit()
        self.tail_fps_textbox.setStyleSheet(param_window.rounded_stylesheet)
        self.tail_fps_textbox.setAlignment(Qt.AlignHCenter)
        self.tail_fps_textbox.setObjectName("Tail Trace FPS")
        self.tail_fps_textbox.setFixedWidth(60)
        self.tail_fps_textbox.setFixedHeight(20)
        self.tail_fps_textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tail_fps_textbox.setText("{}".format(tail_fps))
        layout.addWidget(self.tail_fps_textbox)

        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setSpacing(5)
        param_layout.addWidget(widget)

        label = param_window.HoverLabel("Imaging FPS:")
        label.setHoverMessage("Imaging frame rate (frames per second).")
        layout.addWidget(label)

        self.imaging_fps_textbox = QLineEdit()
        self.imaging_fps_textbox.setStyleSheet(param_window.rounded_stylesheet)
        self.imaging_fps_textbox.setAlignment(Qt.AlignHCenter)
        self.imaging_fps_textbox.setObjectName("Imaging FPS")
        self.imaging_fps_textbox.setFixedWidth(60)
        self.imaging_fps_textbox.setFixedHeight(20)
        self.imaging_fps_textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imaging_fps_textbox.setText("{}".format(imaging_fps))
        layout.addWidget(self.imaging_fps_textbox)

        param_layout.addStretch()

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        param_layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def tail_fps(self):
        return float(self.tail_fps_textbox.text())

    def imaging_fps(self):
        return float(self.imaging_fps_textbox.text())

    @staticmethod
    def getParameters(parent, tail_fps, imaging_fps):
        dialog      = TailTraceParametersDialog(parent, tail_fps, imaging_fps)
        result      = dialog.exec_()
        tail_fps    = dialog.tail_fps()
        imaging_fps = dialog.imaging_fps()

        return (tail_fps, imaging_fps, result == QDialog.Accepted)