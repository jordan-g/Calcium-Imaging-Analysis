import os
import numpy as np
import tifffile
import cv2
import csv
import scipy
import platform
from PIL import Image

import utilities
from param_window import ParamWindow
from preview_window import PreviewWindow
from cnn_training_window import CNNTrainingWindow
from dataset_editing_window import DatasetEditingWindow

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# set default parameters dictionary
DEFAULT_PARAMS = {'gamma'   : 1.0,
                  'contrast': 1.0,
                  'fps'     : 60,
                  'tail_fps': 349
                  }

# set filename for saving current parameters
PARAMS_FILENAME = "gui_params.txt"

n_colors = 20
cmap     = utilities.get_cmap(n_colors)

class GUIController():
    def __init__(self, controller):
        self.controller = controller

        self.load_params()

        self.reset_variables()

        # initialize state variables
        self.closing            = False # whether the user has requested to close the application
        self.roi_finding_queued = False
        self.drawing_mask       = False

        # initialize thread variables
        self.motion_correction_thread = None
        self.roi_finding_thread       = None
        self.video_processing_thread  = None

        # set the mode -- "loading" / "motion_correcting" / "roi_finding" / "roi_filtering"
        self.mode = "loading"

        # set the current z plane to 0
        self.z = 0

        self.show_rois     = False # whether to show ROIs
        self.show_zscore   = True  # whether to show z-score vs. raw fluorescence
        self.video_playing = True  # whether to play the video (if False, the mean image is shown)
        self.frame_offset  = 0     # offset (in calcium imaging video frames) between calcium imaging video and tail trace
        
        # create windows
        self.param_window           = ParamWindow(self)
        self.preview_window         = PreviewWindow(self)
        self.cnn_training_window    = CNNTrainingWindow(self)
        self.dataset_editing_window = DatasetEditingWindow(self)

    def reset_variables(self):
        self.video                  = None # currently loaded video
        self.adjusted_video         = None # gamma- and contrast-adjusted video (for the current z plane)
        self.mean_images            = []   # mean images for all z planes
        self.adjusted_mean_image    = None # gamma- and contrast-adjusted mean image (for the current z plane)
        self.selected_rois          = []   # ROIs currently selected by the user
        self.video_num              = None # which video is currently loaded
        self.group_num              = None # group number of currently loaded video
        self.video_max              = 1    # dynamic range of currently loaded video
        self.mask_images            = None # list of mask images for each z plane in the currently loaded video
        self.selected_mask          = None # which mask, if any, is selected
        self.roi_contours           = []   # list of contours for each ROI in the current z plane
        self.roi_overlays           = []   # list of overlays for each ROI in the current z plane
        self.kept_rois_overlay      = None # overlay containing kept ROIs
        self.removed_rois_overlay   = None # overlay containing removed ROIs
        self.tail_angle_traces      = []   # dict of tail angle traces for each video
        self.heatmap                = None # heatmap of traces for kept ROIs that are currently showing

    def load_params(self):
        if os.path.exists(PARAMS_FILENAME):
            try:
                self.gui_params = DEFAULT_PARAMS
                params = json.load(open(PARAMS_FILENAME))
                for key in params.keys():
                    self.gui_params[key] = params[key]
            except:
                self.gui_params = DEFAULT_PARAMS
        else:
            self.gui_params = DEFAULT_PARAMS

    def video_paths(self):
        if self.controller.use_mc_video and len(self.controller.mc_video_paths) > 0:
            video_paths = self.controller.mc_video_paths
        else:
            video_paths = self.controller.video_paths

        return video_paths

    def video_groups(self):
        return self.controller.video_groups

    def roi_spatial_footprints(self):
        if self.group_num in self.controller.roi_spatial_footprints.keys():
            return self.controller.roi_spatial_footprints[self.group_num][self.z]
        else:
            return None

    def roi_temporal_footprints(self):
        if self.group_num in self.controller.roi_temporal_footprints.keys():
            return self.controller.roi_temporal_footprints[self.group_num][self.z]
        else:
            return None

    def bg_spatial_footprints(self):
        if self.group_num in self.controller.bg_spatial_footprints.keys():
            return self.controller.bg_spatial_footprints[self.group_num][self.z]
        else:
            return None

    def bg_temporal_footprints(self):
        if self.group_num in self.controller.bg_temporal_footprints.keys():
            return self.controller.bg_temporal_footprints[self.group_num][self.z]
        else:
            return None

    def roi_temporal_residuals(self):
        if self.group_num in self.controller.roi_temporal_residuals.keys():
            return self.controller.roi_temporal_residuals[self.group_num][self.z]
        else:
            return None

    def filtered_out_rois(self):
        if self.group_num in self.controller.filtered_out_rois.keys():
            return self.controller.filtered_out_rois[self.group_num][self.z]
        else:
            return []

    def removed_rois(self):
        if self.group_num in self.controller.all_removed_rois.keys():
            return self.controller.all_removed_rois[self.group_num][self.z]
        else:
            return []

    def current_group_video_nums(self):
        return [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]

    def current_group_video_paths(self):
        video_paths = self.video_paths()
        group_paths = [ video_paths[i] for i in self.current_group_video_nums() ]

        return group_paths

    def selected_group_video_lengths(self):
        return [ self.controller.video_lengths[i] for i in self.current_group_video_nums() ]

    def loaded_video_path(self):
        video_paths = self.video_paths()

        return video_paths[self.video_num]

    def video_groups(self):
        return self.controller.video_groups

    def params(self):
        return self.controller.params

    def mean_image(self):
        return self.mean_images[self.z]

    def import_videos(self):
        if len(self.controller.mc_video_paths) > 0 or len(self.controller.roi_spatial_footprints.keys()) > 0:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Warning)
            message_box.setContentsMargins(5, 5, 5, 5)

            message_box.setText("Adding videos will discard any motion correction or ROI finding results. Continue?")
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
        video_paths = QFileDialog.getOpenFileNames(self.param_window, 'Select videos to process.', '', 'Videos (*.tiff *.tif)')[0]

        # import the videos
        if video_paths is not None and len(video_paths) > 0:
            video = tifffile.memmap(video_paths[0])
            num_z = video.shape[1]

            for video_path in video_paths[1:]:
                video = tifffile.memmap(video_path)
                if video.shape[1] != num_z:
                    print("All videos imported together must have the same number of z planes.")
                    return

            if len(self.controller.video_paths) == 0:
                self.preview_window.first_video_imported()
                self.param_window.first_video_imported()

            self.controller.import_videos(video_paths)

            self.tail_angle_traces += [ None for i in range(len(video_paths)) ]

            # notify the param window
            self.param_window.videos_imported(video_paths)

            if self.video is None:
                # load the first video for previewing
                self.load_video(0)

    def load_video(self, video_num):
        old_group_num = self.group_num
        old_z         = self.z

        self.video_num = video_num
        self.group_num = self.controller.video_groups[self.video_num]

        video_path = self.video_paths()[self.video_num]

        print("Loading video: {}".format(video_path))

        # load the video
        base_name = os.path.basename(video_path)
        if base_name.endswith('.tif') or base_name.endswith('.tiff'):
            self.video = tifffile.memmap(video_path)
        else:
            print("Error: Attempted to open a non-TIFF file. Only TIFF files are currently supported.")
            return

        if len(self.video.shape) < 3:
            print("Error: Opened file is not a video -- not enough dimensions.")
            return

        if self.z >= self.video.shape[1]:
            self.z = 0

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

        if len(self.video.shape) == 3:
            # add a z dimension
            self.video = self.video[:, np.newaxis, :, :]

        # flip video 90 degrees to match what is shown in Fiji
        self.video = self.video.transpose((0, 1, 3, 2))

        print("Opened video with shape {}.".format(self.video.shape))

        self.update_adjusted_video()

        # calculate mean images
        self.update_mean_images()
        
        group_changed = self.group_num != old_group_num
        z_changed     = self.z != old_z

        if group_changed:
            self.update_mask_images()
        
        if group_changed or z_changed:
            self.update_roi_contours_and_overlays()
            self.update_merged_roi_overlays()
            self.update_roi_heatmap()

        # notify the param window and preview window
        self.param_window.video_loaded(video_path)

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.preview_window.reset_zoom()

        if self.video_playing:
            self.play_video()
        else:
            self.show_mean_image()

        self.update_selected_rois_plot()
        self.update_tail_plot()

    def update_mean_images(self):
        self.mean_images = np.mean(self.video, axis=0)

        self.update_adjusted_mean_image()

    def update_mask_images(self):
        self.mask_images = [ [] for z in range(self.video.shape[1]) ]

        if self.group_num in self.controller.mask_points.keys():
            for z in range(self.video.shape[1]):
                mask_points_list = self.controller.mask_points[self.group_num][z]

                for mask_points in mask_points_list:
                    mask = self.create_mask_image(z, mask_points)
                    self.mask_images[z].append(mask)

    def update_adjusted_video(self):
        self.adjusted_video = utilities.adjust_gamma(utilities.adjust_contrast(self.video[:, self.z, :, :], self.gui_params['contrast']), self.gui_params['gamma'])

    def adjusted_frame(self, frame):
        return utilities.adjust_gamma(utilities.adjust_contrast(self.video[frame, self.z, :, :], self.gui_params['contrast']), self.gui_params['gamma'])

    def update_adjusted_mean_image(self):
        self.adjusted_mean_image = utilities.adjust_gamma(utilities.adjust_contrast(self.mean_images[self.z], self.gui_params['contrast']), self.gui_params['gamma'])
    
    def update_roi_contours_and_overlays(self):
        if self.roi_spatial_footprints() is not None:
            roi_spatial_footprints = self.roi_spatial_footprints().toarray()
 
            roi_spatial_footprints = roi_spatial_footprints.reshape((self.video.shape[2], self.video.shape[3], roi_spatial_footprints.shape[-1])).transpose((1, 0, 2))
            
            self.roi_contours  = [ None for i in range(roi_spatial_footprints.shape[-1]) ]
            
            self.roi_overlays = np.zeros((roi_spatial_footprints.shape[-1], self.video.shape[2], self.video.shape[3], 4)).astype(np.uint8)

            for i in range(roi_spatial_footprints.shape[-1]):
                maximum = np.amax(roi_spatial_footprints[:, :, i])

                mask = (roi_spatial_footprints[:, :, i] > 0).copy()
                
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

                color = cmap(i % n_colors)[:3]
                color = [255*color[0], 255*color[1], 255*color[2]]

                overlay = np.zeros((self.video.shape[2], self.video.shape[3], 4)).astype(np.uint8)
                overlay[mask, :-1] = color
                overlay[mask, -1] = 255.0*roi_spatial_footprints[mask, i]/maximum
                self.roi_overlays[i] = overlay

                self.roi_contours[i] = contours
        else:
            self.roi_overlays = []
            self.roi_contours = []

    def update_merged_roi_overlays(self):
        roi_spatial_footprints = self.roi_spatial_footprints()

        if roi_spatial_footprints is not None:
            kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in self.removed_rois() ]
            
            if len(kept_rois) > 0:
                a = Image.fromarray(self.roi_overlays[kept_rois[0]])
                for roi in kept_rois[1:]:
                    b = Image.fromarray(self.roi_overlays[roi])
                    a.alpha_composite(b)
                self.kept_rois_overlay = np.asarray(a)
            else:
                self.kept_rois_overlay = None

            removed_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi in self.removed_rois() ]
            
            if len(removed_rois) > 0:
                a = Image.fromarray(self.roi_overlays[removed_rois[0]])
                for roi in removed_rois[1:]:
                    b = Image.fromarray(self.roi_overlays[roi])
                    a.alpha_composite(b)
                self.removed_rois_overlay = np.asarray(a)
            else:
                self.removed_rois_overlay = None
        else:
            self.kept_rois_overlay    = None
            self.removed_rois_overlay = None

    def update_roi_heatmap(self):
        roi_spatial_footprints = self.roi_spatial_footprints()

        self.heatmap = None

        if roi_spatial_footprints is not None:
            kept_rois = [ roi for roi in range(roi_spatial_footprints.shape[-1]) if roi not in self.removed_rois() ]

            if len(kept_rois) > 0:
                heatmap = self.roi_temporal_footprints()[kept_rois]
                
                video_lengths = self.selected_group_video_lengths()

                index = self.current_group_video_paths().index(self.loaded_video_path())

                if index == 0:
                    heatmap = heatmap[:, :video_lengths[0]]
                else:
                    heatmap = heatmap[:, np.sum(video_lengths[:index]):np.sum(video_lengths[:index+1])]

                if heatmap.shape[0] > 0:
                    if self.show_zscore:
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

                    if self.show_zscore:
                        heatmap[heatmap > 3]  = 3
                        heatmap[heatmap < -2] = -2

                    self.heatmap = heatmap.T

        self.preview_window.update_heatmap_plot(self.heatmap)

    def update_selected_rois_plot(self):
        temporal_footprints = self.roi_temporal_footprints()

        if temporal_footprints is not None:
            group_indices = [ i for i in range(len(self.controller.video_paths)) if self.controller.video_groups[i] == self.group_num ]
            
            if self.controller.use_mc_video and len(self.controller.mc_video_paths) > 0:
                group_paths   = [ self.controller.mc_video_paths[i] for i in group_indices ]
            else:
                group_paths   = [ self.controller.video_paths[i] for i in group_indices ]

            group_lengths = [ self.controller.video_lengths[i] for i in group_indices ]
            
            index = group_paths.index(self.loaded_video_path())

            if len(self.controller.roi_temporal_footprints.keys()) > 0:
                if index == 0:
                    temporal_footprints = temporal_footprints[:, :group_lengths[0]]
                else:
                    temporal_footprints = temporal_footprints[:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]
            else:
                temporal_footprints = None

        if temporal_footprints is not None:
            if self.show_zscore:
                temporal_footprints = (temporal_footprints - np.mean(temporal_footprints, axis=1)[:, np.newaxis])/np.std(temporal_footprints, axis=1)[:, np.newaxis]

        self.preview_window.plot_traces(temporal_footprints, self.selected_rois)

    def update_tail_plot(self):
        self.preview_window.plot_tail_angles(self.tail_angle_traces[self.video_num], self.gui_params['tail_fps'], self.controller.params['imaging_fps'])

    def play_video(self):
        self.preview_window.show_plot()
        self.preview_window.play_video()

    def show_mean_image(self):
        self.preview_window.show_mean_image()

    def videos_rearranged(self, old_indices, groups):
        self.controller.video_groups  = groups
        self.controller.video_paths   = [ self.controller.video_paths[i] for i in old_indices ]
        self.controller.video_lengths = [ self.controller.video_lengths[i] for i in old_indices ]

        if len(self.controller.mc_video_paths) > 0:
            self.controller.mc_video_paths = [ self.controller.mc_video_paths[i] for i in old_indices ]

        self.video_num = old_indices.index(self.video_num)
        self.group_num = self.controller.video_groups[self.video_num]

    def save_rois(self):
        self.save_all_rois()

    def load_rois(self):
        # let the user pick saved ROIs
        load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved ROI data.', '', 'Numpy (*.npy)')[0]

        if load_path is not None and len(load_path) > 0:
            self.controller.load_rois(load_path, group_num=self.group_num, video_path=self.loaded_video_path())

            self.param_window.tab_widget.setTabEnabled(3, True)

            self.show_rois = True
            self.param_window.show_rois_action.setEnabled(True)
            self.param_window.show_rois_action.setChecked(True)
            self.preview_window.show_rois_checkbox.setEnabled(True)
            self.preview_window.show_rois_checkbox.setChecked(True)
            self.param_window.save_rois_action.setEnabled(True)

            self.update_roi_contours_and_overlays()
            self.update_merged_roi_overlays()
            self.update_roi_heatmap()

            self.selected_rois = []
            self.preview_window.clear_outline_items()

            self.update_selected_rois_plot()
            self.update_tail_plot()
            
            self.preview_window.create_text_items()

            # show ROI filtering parameters
            self.show_roi_filtering_params()

    def remove_videos_at_indices(self, indices):
        self.controller.remove_videos_at_indices(indices)

        for i in range(len(indices)-1, -1, -1):
            del self.tail_angle_traces[i]

        if len(self.controller.video_paths) == 0:
            print("All videos removed.")

            # reset variables
            self.reset_variables()

            # reset param window & preview window to their initial states
            self.param_window.set_initial_state()
            self.preview_window.set_initial_state()

            self.video_playing = True
            self.preview_window.play_video_checkbox.setEnabled(True)
            self.preview_window.play_video_checkbox.setChecked(True)
            self.param_window.play_video_action.setEnabled(True)
            self.param_window.play_video_action.setChecked(True)
        else:
            # open the newest video at index 0
            self.load_video(0)

    def remove_group(self, group):
        self.controller.remove_group(group)

        if len(self.controller.video_paths) == 0:
            print("All videos removed.")

            # reset variables
            self.reset_variables()

            # reset param window & preview window to their initial states
            self.param_window.set_initial_state()
            self.preview_window.set_initial_state()

            self.play_video_bool = True
            self.preview_window.play_video_checkbox.setEnabled(True)
            self.preview_window.play_video_checkbox.setChecked(True)
            self.param_window.play_video_action.setEnabled(True)
            self.param_window.play_video_action.setChecked(True)
        else:
            # open the newest video at index 0
            self.video_selected(0, force=True)

    def add_group(self, group):
        self.controller.add_group(group)

    def save_all_rois(self):
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

            video = tifffile.memmap(video_path)

            if len(video.shape) == 3:
                # add z dimension
                video = video[:, np.newaxis, :, :]

            group_num = self.controller.video_groups[i]

            roi_spatial_footprints  = self.controller.roi_spatial_footprints[group_num]
            roi_temporal_footprints = self.controller.roi_temporal_footprints[group_num]
            roi_temporal_residuals  = self.controller.roi_temporal_residuals[group_num]
            bg_spatial_footprints   = self.controller.bg_spatial_footprints[group_num]
            bg_temporal_footprints  = self.controller.bg_temporal_footprints[group_num]
            
            manually_removed_rois = self.controller.manually_removed_rois[group_num]
            all_removed_rois      = self.controller.all_removed_rois[group_num]
            locked_rois           = self.controller.locked_rois[group_num]

            # save centroids & traces
            for z in range(video.shape[1]):
                print("Calculating ROI activities for z={}...".format(z))

                centroids = np.zeros((roi_spatial_footprints[z].shape[-1], 2))
                kept_rois = [ roi for roi in range(roi_spatial_footprints[z].shape[-1]) if (roi not in all_removed_rois[z]) or (roi in locked_rois[z]) ]

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
        # # create a motion correction thread
        # self.motion_correction_thread = MotionCorrectThread(self.param_window)

        # # set the parameters of the motion correction thread
        # self.motion_correction_thread.set_parameters(self.controller.video_paths, self.controller.video_groups, int(self.controller.params["max_shift"]), int(self.controller.params["patch_stride"]), int(self.controller.params["patch_overlap"]), use_multiprocessing=self.controller.use_multiprocessing)
        
        # self.motion_correction_thread.progress.connect(self.motion_correction_progress)
        # self.motion_correction_thread.finished.connect(self.motion_correction_ended)

        # # start the thread
        # self.motion_correction_thread.start()

        # # notify the param window
        # self.param_window.motion_correction_started()

        mc_video_paths, mc_borders = utilities.motion_correct_multiple_videos(self.controller.video_paths, self.controller.video_groups, int(self.controller.params["max_shift"]), int(self.controller.params["patch_stride"]), int(self.controller.params["patch_overlap"]), progress_signal=None, thread=None, use_multiprocessing=self.controller.use_multiprocessing)

        self.motion_correction_ended(mc_video_paths, mc_borders)

    def motion_correction_progress(self, group_num):
        # notify the param window
        self.param_window.update_motion_correction_progress(group_num)

    def motion_correction_ended(self, mc_video_paths, mc_borders):
        self.controller.mc_video_paths = mc_video_paths
        self.controller.mc_borders     = mc_borders

        self.param_window.motion_correction_ended()

        self.set_use_mc_video(True)

        if self.roi_finding_queued:
            self.show_roi_finding_params()
            self.find_rois()

    def set_use_mc_video(self, use_mc_video):
        self.controller.use_mc_video = use_mc_video

        print(self.video_paths())
        print(len(self.video_paths()))
        
        self.param_window.set_video_paths(self.video_paths())

        self.load_video(self.video_num)

        self.update_adjusted_video()
        self.update_adjusted_mean_image()

        if self.mode in ("loading", "motion_correcting"):
            self.play_video()
        else:
            self.show_mean_image()

    def set_roi_finding_mode(self, mode):
        self.controller.roi_finding_mode = mode

    def find_rois(self):
        video_paths = self.video_paths()

        # # create an ROI finding thread
        # self.roi_finding_thread = ROIFindingThread(self.param_window)

        # # set the parameters of the ROI finding thread
        # self.roi_finding_thread.set_parameters(video_paths, self.controller.video_groups, self.controller.params, self.controller.mc_borders, self.controller.use_multiprocessing, method=self.controller.roi_finding_mode, mask_points=self.controller.mask_points)

        # self.roi_finding_thread.progress.connect(self.roi_finding_progress)
        # self.roi_finding_thread.finished.connect(self.roi_finding_ended)

        # # start the thread
        # self.roi_finding_thread.start()

        # # notify the param window
        # self.param_window.roi_finding_started()

        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_multiple_videos(video_paths, self.controller.video_groups, self.controller.params, mc_borders=self.controller.mc_borders, progress_signal=None, thread=None, use_multiprocessing=self.controller.use_multiprocessing, method=self.controller.roi_finding_mode, mask_points=self.controller.mask_points)

        self.roi_finding_ended(roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints)

    def roi_finding_progress(self, group_num):
        # notify the param window
        self.param_window.update_roi_finding_progress(group_num)

    def roi_finding_ended(self, roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints):
        self.controller.roi_spatial_footprints  = roi_spatial_footprints
        self.controller.roi_temporal_footprints = roi_temporal_footprints
        self.controller.roi_temporal_residuals  = roi_temporal_residuals
        self.controller.bg_spatial_footprints   = bg_spatial_footprints
        self.controller.bg_temporal_footprints  = bg_temporal_footprints

        self.controller.filtered_out_rois     = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.manually_removed_rois = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.all_removed_rois      = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.controller.video_groups) }
        self.controller.locked_rois           = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.controller.video_groups) }

        # notify the param window
        self.param_window.roi_finding_ended()

        self.show_rois = True
        self.param_window.show_rois_action.setEnabled(True)
        self.param_window.show_rois_action.setChecked(True)
        self.preview_window.show_rois_checkbox.setEnabled(True)
        self.preview_window.show_rois_checkbox.setChecked(True)
        self.param_window.save_rois_action.setEnabled(True)

        self.update_roi_contours_and_overlays()
        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()
        self.update_tail_plot()

        if self.video_playing:
            self.play_video()
        else:
            self.show_mean_image()

        self.roi_finding_queued = False

    def show_video_loading_params(self):
        self.param_window.tab_widget.setCurrentIndex(0)

        self.mode = "loading"

    def show_motion_correction_params(self):
        self.param_window.tab_widget.setCurrentIndex(1)

        self.mode = "motion_correcting"

    def show_roi_finding_params(self):
        self.param_window.tab_widget.setCurrentIndex(2)

        self.mode = "roi_finding"

    def show_roi_filtering_params(self):
        self.param_window.tab_widget.setCurrentIndex(3)

        self.mode = "roi_filtering"

    def close_all(self):
        self.closing = True

        # close param & preview windows
        self.param_window.close()
        self.preview_window.close()
        self.cnn_training_window.close()

        # save the current parameters
        self.save_params()

    def preview_contrast(self, contrast):
        self.gui_params['contrast'] = contrast

        if self.video_playing:
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            adjusted_frame = self.adjusted_frame(self.preview_window.frame_num)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        else:
            # calculate a contrast- and gamma-adjusted version of the current mean image
            self.update_adjusted_mean_image()

            # show the adjusted mean image
            self.preview_window.show_frame(self.adjusted_mean_image)

    def preview_gamma(self, gamma):
        self.gui_params['gamma'] = gamma

        if self.video_playing:
            self.preview_window.timer.stop()

            # calculate a contrast- and gamma-adjusted version of the current frame
            adjusted_frame = self.adjusted_frame(self.preview_window.frame_num)

            # show the adjusted frame
            self.preview_window.show_frame(adjusted_frame)
        else:
            # calculate a contrast- and gamma-adjusted version of the current mean image
            self.update_adjusted_mean_image()

            # show the adjusted mean image
            self.preview_window.show_frame(self.adjusted_mean_image)

    def update_param(self, param, value):
        print("Setting parameter '{}' to {}.".format(param, value))

        # update the parameter
        if param in self.controller.params.keys():
            self.controller.params[param] = value
        elif param in self.gui_params.keys():
            self.gui_params[param] = value

        if param in ("contrast, gamma"):
            self.update_adjusted_video()
            self.update_adjusted_mean_image()

            if self.video_playing:
                self.play_video()
            else:
                self.show_mean_image()
        elif param == "fps":
            # update the FPS of the preview window
            self.preview_window.set_fps(self.gui_params['fps'])

            if self.video_playing:
                self.play_video()
            else:
                self.show_mean_image()
        if param == "z":
            self.z = value

            self.update_adjusted_video()
            self.update_adjusted_mean_image()

            self.update_roi_contours_and_overlays()
            self.update_merged_roi_overlays()
            self.update_roi_heatmap()

            self.selected_rois = []
            self.preview_window.clear_outline_items()

            self.update_selected_rois_plot()
            self.update_tail_plot()

            if self.video_playing:
                self.play_video()
            else:
                self.show_mean_image()

    def set_use_multiprocessing(self, use_multiprocessing):
        self.controller.use_multiprocessing = use_multiprocessing
        self.param_window.loading_widget.use_multiprocessing_checkbox.setChecked(use_multiprocessing)
        self.param_window.motion_correction_widget.use_multiprocessing_checkbox.setChecked(use_multiprocessing)
        self.param_window.roi_finding_widget.use_multiprocessing_checkbox.setChecked(use_multiprocessing)

    def set_show_rois(self, show_rois):
        print("Setting show ROIs from {} to {}.".format(self.show_rois, show_rois))
        self.show_rois = show_rois

        self.param_window.show_rois_action.setChecked(show_rois)
        self.preview_window.show_rois_checkbox.setChecked(show_rois)

        self.cnn_training_window.set_show_rois(show_rois)

        if not self.video_playing:
            self.show_mean_image()

    def set_play_video(self, video_playing):
        print("Setting play video from {} to {}.".format(self.video_playing, video_playing))
        self.video_playing = video_playing

        self.param_window.play_video_action.setChecked(video_playing)
        self.preview_window.play_video_checkbox.setChecked(video_playing)

    def set_show_zscore(self, show_zscore):
        self.show_zscore = show_zscore

        if self.show_zscore:
            self.preview_window.roi_trace_viewbox.setYRange(-2, 3)
            self.preview_window.roi_trace_viewbox.setLabel('left', "Z-Score")
        else:
            self.preview_window.roi_trace_viewbox.setYRange(0, 1)
            self.preview_window.roi_trace_viewbox.setLabel('left', "Fluorescence")

        self.update_roi_heatmap()

        self.update_selected_rois_plot()

    def filter_rois(self):
        mean_images = [ (utilities.mean(self.video, z)/self.video_max)*self.video_max for z in range(self.video.shape[1]) ]
        mean_images = [ utilities.adjust_gamma(utilities.adjust_contrast(mean_image, self.gui_params['contrast']), self.gui_params['gamma']) for mean_image in mean_images ]

        self.controller.filter_rois(mean_images, self.group_num)

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()
        self.update_tail_plot()

        if self.video_playing:
            self.play_video()
        else:
            self.show_mean_image()

    def select_single_roi(self, roi):
        self.selected_rois = [roi]

        self.param_window.single_roi_selected(discarded=roi in self.removed_rois())
        self.preview_window.single_roi_selected(roi)

        self.update_selected_rois_plot()

    def select_roi(self, roi_point, ctrl_held=False):
        if self.mode == "loading" or self.mode == "motion_correcting":
            return

        if roi_point is not None:
            if len(self.controller.roi_spatial_footprints) > 0:
                # find out which ROI to select
                selected_roi = utilities.get_roi_containing_point(self.controller.roi_spatial_footprints[self.group_num][self.z], roi_point, self.mean_images[self.z].shape)

                print("Selected ROI: {}".format(selected_roi))

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
                    self.preview_window.clear_outline_items()

                    self.param_window.no_rois_selected()

                    self.preview_window.no_rois_selected()

    def load_tail_angles(self):
        load_path = QFileDialog.getOpenFileName(self.param_window, 'Select saved tail angle data.', '', 'CSV (*.csv)')[0]

        if load_path is not None and len(load_path) > 0:
            tail_angles = np.genfromtxt(load_path, delimiter=",")

            tail_angles = np.nanmean(tail_angles[:, -3:], axis=-1)

            tail_angles -= np.mean(tail_angles[:100])

            self.tail_angle_traces[self.video_num] = tail_angles

            tail_fps, imaging_fps, ok = TailTraceParametersDialog.getParameters(None, self.gui_params['tail_fps'], self.controller.params['imaging_fps'])

            if ok:
                self.gui_params['tail_fps']           = tail_fps
                self.controller.params['imaging_fps'] = imaging_fps

                success = self.preview_window.plot_tail_angles(self.tail_angle_traces[self.video_num], self.gui_params['tail_fps'], self.controller.params['imaging_fps'])

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

        self.update_tail_plot()

    def erase_selected_rois(self):
        nonerased_rois = [ roi for roi in range(self.roi_spatial_footprints().shape[-1]) if roi not in self.selected_rois ]
        self.controller.roi_spatial_footprints[self.group_num][self.z] = self.controller.roi_spatial_footprints[self.group_num][self.z][:, nonerased_rois]
        self.controller.roi_temporal_footprints[self.group_num][self.z] = self.controller.roi_temporal_footprints[self.group_num][self.z][nonerased_rois]
        self.controller.roi_temporal_residuals[self.group_num][self.z] = self.controller.roi_temporal_residuals[self.group_num][self.z][nonerased_rois]

        self.roi_contours = [ self.roi_contours[i] for i in nonerased_rois ]
        self.roi_overlays = [ self.roi_overlays[i] for i in nonerased_rois ]

        for roi in sorted(self.selected_rois, reverse=True):
            if roi in self.controller.all_removed_rois[self.group_num][self.z]:
                i = self.controller.all_removed_rois[self.group_num][self.z].index(roi)
                del self.controller.all_removed_rois[self.group_num][self.z][i]

            if roi in self.controller.manually_removed_rois[self.group_num][self.z]:
                i = self.controller.manually_removed_rois[self.group_num][self.z].index(roi)
                del self.controller.manually_removed_rois[self.group_num][self.z][i]

            if roi in self.controller.locked_rois[self.group_num][self.z]:
                i = self.controller.locked_rois[self.group_num][self.z].index(roi)
                del self.controller.locked_rois[self.group_num][self.z][i]

            self.controller.all_removed_rois[self.group_num][self.z] = [ (a if a < roi else a-1) for a in self.controller.all_removed_rois[self.group_num][self.z] ]
            self.controller.manually_removed_rois[self.group_num][self.z] = [ (a if a < roi else a-1) for a in self.controller.manually_removed_rois[self.group_num][self.z] ]
            self.controller.locked_rois[self.group_num][self.z] = [ (a if a < roi else a-1) for a in self.controller.locked_rois[self.group_num][self.z] ]

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

    def discard_selected_rois(self):
        for roi in self.selected_rois:
            self.controller.discard_roi(roi, self.z, self.group_num)

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

    def discard_all_rois(self):
        self.controller.manually_removed_rois[self.group_num][self.z] = np.arange(self.controller.roi_spatial_footprints[self.group_num][self.z].shape[1]).tolist()
        self.controller.filtered_out_rois[self.group_num][self.z]     = []
        self.controller.all_removed_rois[self.group_num][self.z]      = self.controller.manually_removed_rois[self.group_num][self.z][:]

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

    def keep_selected_rois(self):
        for roi in self.selected_rois:
            self.controller.keep_roi(roi, self.z, self.group_num)

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

    def keep_all_rois(self):
        self.controller.manually_removed_rois[self.group_num][self.z] = []
        self.controller.filtered_out_rois[self.group_num][self.z]     = []
        self.controller.all_removed_rois[self.group_num][self.z]      = []

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

    def save_selected_roi_traces(self):
        save_path = QFileDialog.getSaveFileName(self.param_window, 'Enter CSV filename.', '', 'CSV (*.csv)')[0]

        if save_path is not None and len(save_path) > 0:
            traces = self.controller.roi_temporal_footprints[self.group_num][self.z][self.selected_rois]

            with open(save_path, 'ab') as f:
                np.savetxt(f, traces, delimiter=',')

    def save_roi_images(self):
        save_directory = str(QFileDialog.getExistingDirectory(self.param_window, "Select Directory"))

        if save_directory is not None and len(save_directory) > 0:
            gSig = [8, 8]
            dims = (512, 512)
            patch_size = 50
            A = self.controller.roi_spatial_footprints[self.group_num][self.z]
            half_crop = np.minimum(
                    gSig[0] * 4 + 1, patch_size), np.minimum(gSig[1] * 4 + 1, patch_size)
            dims = np.array(dims)
            coms = [scipy.ndimage.center_of_mass(
                mm.toarray().reshape(dims, order='F')) for mm in A.tocsc().T]
            coms = np.maximum(coms, half_crop)
            coms = np.array([np.minimum(cms, dims - half_crop)
                             for cms in coms]).astype(np.int)
            crop_imgs = [mm.toarray().reshape(dims, order='F')[com[0] - half_crop[0]:com[0] + half_crop[0],
                                                               com[1] - half_crop[1]:com[1] + half_crop[1]] for mm, com in zip(A.tocsc().T, coms)]
            final_crops = np.array([cv2.resize(
                im / np.linalg.norm(im), (patch_size, patch_size)) for im in crop_imgs])

            images = (final_crops*255.0).astype(np.uint8)

            for i in range(final_crops.shape[0]):
                cv2.imwrite(os.path.join(save_directory, 'roi_{}.png'.format(i)), images[i])

    def merge_selected_rois(self):
        if len(self.selected_rois) > 1:
            print(self.controller.roi_temporal_footprints[self.group_num][self.z].shape)
            all_rois_pre = np.arange(self.controller.roi_temporal_footprints[self.group_num][self.z].shape[0])

            video_paths = self.controller.video_paths_in_group(self.video_paths(), self.group_num)

            roi_spatial_footprints, roi_temporal_footprints = utilities.merge_rois(self.selected_rois, self.roi_spatial_footprints(), self.roi_temporal_footprints(), self.bg_spatial_footprints(), self.bg_temporal_footprints(), self.roi_temporal_residuals(), video_paths, self.z, self.controller.params)

            self.controller.roi_spatial_footprints[self.group_num][self.z]  = roi_spatial_footprints
            self.controller.roi_temporal_footprints[self.group_num][self.z] = roi_temporal_footprints

            all_removed_rois      = self.controller.all_removed_rois[self.group_num][self.z]
            locked_rois           = self.controller.locked_rois[self.group_num][self.z]
            manually_removed_rois = self.controller.manually_removed_rois[self.group_num][self.z]
            filtered_out_rois     = self.controller.filtered_out_rois[self.group_num][self.z]

            mapping = {}

            for i in sorted(self.selected_rois, reverse=True):
                all_rois_pre = np.delete(all_rois_pre, i)

                if i in all_removed_rois:
                    index = all_removed_rois.index(i)

                    del all_removed_rois[index]

                if i in locked_rois:
                    index = locked_rois.index(i)
                    
                    del locked_rois[index]

                if i in manually_removed_rois:
                    index = manually_removed_rois.index(i)
                    
                    del manually_removed_rois[index]

                if i in filtered_out_rois:
                    index = filtered_out_rois.index(i)
                    
                    del filtered_out_rois[index]

            for i in range(len(all_rois_pre)):
                mapping[all_rois_pre[i]] = i

            for i in range(len(all_removed_rois)):
                all_removed_rois[i] = mapping[all_removed_rois[i]]

            for i in range(len(locked_rois)):
                locked_rois[i] = mapping[locked_rois[i]]

            for i in range(len(manually_removed_rois)):
                manually_removed_rois[i] = mapping[manually_removed_rois[i]]

            for i in range(len(filtered_out_rois)):
                filtered_out_rois[i] = mapping[filtered_out_rois[i]]

            self.controller.all_removed_rois[self.group_num][self.z]      = all_removed_rois
            self.controller.locked_rois[self.group_num][self.z]           = locked_rois
            self.controller.manually_removed_rois[self.group_num][self.z] = manually_removed_rois
            self.controller.filtered_out_rois[self.group_num][self.z]     = filtered_out_rois

            self.update_roi_contours_and_overlays()
            self.update_merged_roi_overlays()
            self.update_roi_heatmap()

            self.selected_rois = []
            self.preview_window.clear_outline_items()

            self.update_selected_rois_plot()

            if not self.video_playing:
                self.show_mean_image()

    def save_params(self):
        self.controller.save_params()

    def set_motion_correct(self, boolean):
        self.controller.motion_correct_all_videos = boolean

    def toggle_mask_mode(self):
        self.drawing_mask = not self.drawing_mask

        if self.drawing_mask:
            self.param_window.mask_drawing_started()
            self.preview_window.uncheck_play_video()
            self.preview_window.play_video_checkbox.setEnabled(False)
            if platform.system() == "Darwin":
                key = "⌘"
            else:
                key = "Ctrl"
            self.preview_window.set_default_statusbar_message("Draw masks in the left plot. Hold {} and click to add a mask point. Click a mask to select it, and right click to delete it.".format(key))
        else:
            self.param_window.mask_drawing_ended()
            self.preview_window.reset_default_statusbar_message()
            self.preview_window.play_video_checkbox.setEnabled(True)

            if self.preview_window.temp_mask_item is not None:
                if len(self.preview_window.mask_points) >= 3:
                    self.create_mask(self.preview_window.mask_points)
                    mask_item = self.preview_window.create_mask_item(self.preview_window.mask_points)
                    self.preview_window.mask_items.append(mask_item)
                    self.preview_window.left_image_viewbox.addItem(mask_item)
                    self.preview_window.mask_points = []

    def create_mask(self, mask_points):
        self.controller.add_mask(mask_points, self.z, self.video.shape[1], self.group_num)

        mask = self.create_mask_image(self.z, mask_points)

        if self.mask_images is None:
            self.mask_images = [ [] for z in range(self.video.shape[1]) ]

        self.mask_images[self.z].append(mask)

    def delete_mask(self, mask_num):
        self.controller.delete_mask(mask_num, self.z, self.group_num)

        del self.mask_images[self.z][mask_num]

    def create_mask_image(self, z, mask_points):
        # create mask image
        mask_points = np.array(mask_points + [mask_points[0]]).astype(int)

        mask = np.zeros(self.mean_images[z].shape).astype(np.uint8)
        cv2.fillConvexPoly(mask, mask_points, 1)
        mask = mask.astype(np.bool)

        if self.controller.params['invert_masks']:
            mask = mask == False

        return mask

    def mask_points(self):
        if self.group_num in self.controller.mask_points.keys():
            return self.controller.mask_points[self.group_num][self.z]
        else:
            return []

    def test_cnn_on_data(self):
        predictions, final_crops = utilities.test_cnn_on_data(self.roi_spatial_footprints(), self.adjusted_mean_image, self.controller.params['half_size'])

        filtered_out_rois = [ i for i in range(predictions.shape[0]) if predictions[i, 0] < self.controller.params['cnn_accept_threshold'] ]

        self.controller.filtered_out_rois[self.group_num][self.z] = filtered_out_rois

        self.controller.all_removed_rois[self.group_num][self.z] = self.controller.filtered_out_rois[self.group_num][self.z] + self.controller.manually_removed_rois[self.group_num][self.z]

        self.update_merged_roi_overlays()
        self.update_roi_heatmap()

        self.selected_rois = []
        self.preview_window.clear_outline_items()

        self.update_selected_rois_plot()

        if not self.video_playing:
            self.show_mean_image()

        self.cnn_training_window.update_with_predictions(predictions)

    def pick_data_to_train_cnn(self):
        self.cnn_training_window.show()

        cropped_images, cropped_overlays = self.create_cropped_images_and_overlays(self.cnn_training_window.crop_size)

        self.cnn_training_window.refresh(cropped_images, cropped_overlays, show_rois=self.preview_window.show_rois_checkbox.isChecked())

    def create_cropped_images_and_overlays(self, crop_size):
        _, cropped_images, cropped_overlays = utilities.preprocess_spatial_footprints(self.roi_spatial_footprints(), self.adjusted_mean_image, crop_size, roi_overlays=self.roi_overlays)

        cropped_images /= np.amax(cropped_images, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]

        return cropped_images, cropped_overlays

    def train_cnn_on_data(self, positive_rois, negative_rois):
        learning_rate, batch_size, ok = CNNTrainingParametersDialog.getParameters(None, self.gui_params['tail_fps'], self.controller.params['imaging_fps'])
        if ok:
            utilities.train_cnn_on_data(self.roi_spatial_footprints(), self.adjusted_mean_image, positive_rois, negative_rois, self.controller.params['half_size'], learning_rate, weight_decay)

    def reset_cnn(self):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Warning)
        message_box.setContentsMargins(5, 5, 5, 5)

        message_box.setText("Are you sure you want to reset the CNN to an untrained state?")
        message_box.setWindowTitle("")
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        return_value = message_box.exec_()
        if return_value == QMessageBox.Cancel:
            return
        else:
            _ = utilities.load_model(reset=True)

    def edit_dataset(self):
        self.dataset_editing_window.show()

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

        mc_video_paths, mc_borders = utilities.motion_correct_multiple_videos(self.video_paths, self.groups, self.max_shift, self.patch_stride, self.patch_overlap, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing)

        self.finished.emit(mc_video_paths, mc_borders)

        self.running = False

class ROIFindingThread(QThread):
    finished = pyqtSignal(dict, dict, dict, dict, dict)
    progress = pyqtSignal(int)

    def __init__(self, parent):
        QThread.__init__(self, parent)

        self.running = False
    
    def set_parameters(self, video_paths, groups, params, mc_borders, use_multiprocessing, method="cnmf", mask_points=[]):
        self.video_paths         = video_paths
        self.groups              = groups
        self.gui_params          = params
        self.mc_borders          = mc_borders
        self.use_multiprocessing = use_multiprocessing
        self.method              = method
        self.mask_points         = mask_points

    def run(self):
        self.running = True

        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_multiple_videos(self.video_paths, self.groups, self.gui_params, mc_borders=self.mc_borders, progress_signal=self.progress, thread=self, use_multiprocessing=self.use_multiprocessing, method=self.method, mask_points=self.mask_points)

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

        label = QLabel("Tail trace FPS:")
        label.setToolTip("Tail trace frame rate (frames per second).")
        layout.addWidget(label)

        self.tail_fps_textbox = QLineEdit()
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

        label = QLabel("Imaging FPS (per plane):")
        label.setToolTip("Imaging frame rate (frames per second).")
        layout.addWidget(label)

        self.imaging_fps_textbox = QLineEdit()
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

class CNNTrainingParametersDialog(QDialog):
    def __init__(self, parent, learning_rate, batch_size):
        super(CNNTrainingParametersDialog, self).__init__(parent)

        param_layout = QVBoxLayout(self)
        param_layout.setContentsMargins(0, 0, 0, 0)

        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setSpacing(5)
        param_layout.addWidget(widget)

        label = QLabel("Learning rate:")
        label.setToolTip("Learning rate for training the network.")
        layout.addWidget(label)

        self.learning_rate_textbox = QLineEdit()
        self.learning_rate_textbox.setAlignment(Qt.AlignHCenter)
        self.learning_rate_textbox.setFixedWidth(60)
        self.learning_rate_textbox.setFixedHeight(20)
        self.learning_rate_textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.learning_rate_textbox.setText("{}".format(learning_rate))
        layout.addWidget(self.learning_rate_textbox)

        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setSpacing(5)
        param_layout.addWidget(widget)

        label = QLabel("Batch size:")
        label.setToolTip("Batch size to use when training the network.")
        layout.addWidget(label)

        self.batch_size_textbox = QLineEdit()
        self.batch_size_textbox.setAlignment(Qt.AlignHCenter)
        self.batch_size_textbox.setFixedWidth(60)
        self.batch_size_textbox.setFixedHeight(20)
        self.batch_size_textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.batch_size_textbox.setText("{}".format(batch_size))
        layout.addWidget(self.batch_size_textbox)

        param_layout.addStretch()

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        param_layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def learning_rate(self):
        return float(self.learning_rate_textbox.text())

    def batch_size(self):
        return float(self.batch_size_textbox.text())

    @staticmethod
    def getParameters(parent, learning_rate, batch_size):
        dialog        = CNNTrainingParametersDialog(parent, learning_rate, batch_size)
        result        = dialog.exec_()
        learning_rate = dialog.learning_rate()
        batch_size    = dialog.batch_size()

        return (learning_rate, batch_size, result == QDialog.Accepted)