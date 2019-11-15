import os
import json
import numpy as np
import tifffile
import cv2
import csv

import utilities

# set default parameters dictionary
DEFAULT_PARAMS = {'use_patches'          : True,
                  'max_shift'            : 6,
                  'patch_stride'         : 48,
                  'patch_overlap'        : 24,
                  'cnmf_patch_size'      : 20,
                  'cnmf_patch_stride'    : 10,
                  'max_merge_area'       : 150,
                  'imaging_fps'          : 3,
                  'decay_time'           : 0.4,
                  'autoregressive_order' : 0,
                  'num_bg_components'    : 2,
                  'merge_threshold'      : 0.8,
                  'num_components'       : 400,
                  'init_method'          : 'greedy_roi',
                  'min_df_f'             : 1,
                  'artifact_decay_speed' : 1,
                  'half_size'            : 4,
                  'use_cnn'              : False,
                  'min_snr'              : 1.3,
                  'min_spatial_corr'     : 0.8,
                  'use_cnn'              : False,
                  'cnn_accept_threshold' : 0.5,
                  'cnn_reject_threshold' : 0.1,
                  'min_area'             : 10,
                  'max_area'             : 100,
                  'diameter'             : 10,
                  'sampling_rate'        : 3,
                  'connected'            : True,
                  'neuropil_basis_ratio' : 6,
                  'neuropil_radius_ratio': 3,
                  'inner_neuropil_radius': 2,
                  'min_neuropil_pixels'  : 350,
                  'invert_masks'         : False
                  }

# set filename for saving current parameters
PARAMS_FILENAME = "params.txt"

class Controller():
    def __init__(self):
        # load parameters
        if os.path.exists(PARAMS_FILENAME):
            try:
                self.params = DEFAULT_PARAMS
                params = json.load(open(PARAMS_FILENAME))
                for key in params.keys():
                    self.params[key] = params[key]
            except:
                self.params = DEFAULT_PARAMS
        else:
            self.params = DEFAULT_PARAMS

        # initialize other variables
        self.video_paths   = [] # paths of all videos to process
        self.video_lengths = [] # lengths (# of frames) of all videos
        self.video_groups  = [] # groups that videos belong to

        # initialize all variables
        self.reset_variables()
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables()
        self.reset_roi_filtering_variables()

    def reset_variables(self):
        self.use_mc_video        = False  # whether to use the motion-corrected video for finding ROIs
        self.use_multiprocessing = True   # whether to use multi-processing
        self.roi_finding_mode    = "cnmf" # which algorithm to use to find ROIs -- "cnmf" / "suite2p"

    def reset_motion_correction_variables(self):
        self.mc_video_paths = [] # paths of all motion-corrected videos
        self.mc_borders     = {} # borders of all motion-corrected videos

    def reset_roi_finding_variables(self):
        self.roi_spatial_footprints  = {}
        self.roi_temporal_footprints = {}
        self.roi_temporal_residuals  = {}
        self.bg_spatial_footprints   = {}
        self.bg_temporal_footprints  = {}
        self.filtered_out_rois       = {}
        self.mask_points             = {}

    def reset_roi_filtering_variables(self):
        self.manually_removed_rois = {}
        self.all_removed_rois      = {}
        self.locked_rois           = {}

    def import_videos(self, video_paths):
        # add the new video paths to the currently loaded video paths
        self.video_paths += video_paths

        # assign a group number to the new videos
        if len(self.video_groups) > 0:
            group_num = np.amax(np.unique(self.video_groups)) + 1
        else:
            group_num = 0

        # store video lengths and group numbers
        for video_path in video_paths:
            video = tifffile.memmap(video_path)
            self.video_lengths.append(video.shape[0])
            self.video_groups.append(group_num)

            if len(video.shape) > 3:
                num_z = video.shape[1]
            else:
                num_z = 1

        # initialize mask points list
        self.mask_points[group_num] = [ [] for z in range(num_z) ]

    def save_rois(self, save_path, group_num=None, video_path=None):
        if group_num is None:
            # set video paths
            if self.use_mc_video and len(self.mc_video_paths) > 0:
                video_paths = self.mc_video_paths
            else:
                video_paths = self.video_paths

            # create a dictionary to hold the ROI data
            roi_data = {'roi_spatial_footprints' : self.roi_spatial_footprints,
                        'roi_temporal_footprints': self.roi_temporal_footprints,
                        'roi_temporal_residuals' : self.roi_temporal_residuals,
                        'bg_spatial_footprints'  : self.bg_spatial_footprints,
                        'bg_temporal_footprints' : self.bg_temporal_footprints,
                        'filtered_out_rois'      : self.filtered_out_rois,
                        'manually_removed_rois'  : self.manually_removed_rois,
                        'all_removed_rois'       : self.all_removed_rois,
                        'locked_rois'            : self.locked_rois,
                        'video_paths'            : video_paths,
                        'masks'                  : self.mask_points}
        else:
            group_indices = [ i for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]
            group_lengths = [ self.video_lengths[i] for i in group_indices ]
            group_paths   = [ self.video_paths[i] for i in group_indices ]

            index = group_paths.index(video_path)

            roi_spatial_footprints = self.roi_spatial_footprints[group_num]
            bg_spatial_footprints  = self.bg_spatial_footprints[group_num]
            filtered_out_rois      = self.filtered_out_rois[group_num]
            manually_removed_rois  = self.manually_removed_rois[group_num]
            all_removed_rois       = self.all_removed_rois[group_num]
            locked_rois            = self.locked_rois[group_num]
            masks                  = self.mask_points[group_num]
            if index == 0:
                roi_temporal_footprints = [ self.roi_temporal_footprints[group_num][z][:, :group_lengths[0]] for z in range(len(roi_spatial_footprints)) ]
                roi_temporal_residuals  = [ self.roi_temporal_residuals[group_num][z][:, :group_lengths[0]] for z in range(len(roi_spatial_footprints)) ]
                bg_temporal_footprints  = [ self.bg_temporal_footprints[group_num][z][:, :group_lengths[0]] for z in range(len(roi_spatial_footprints)) ]
            else:
                roi_temporal_footprints = [ self.roi_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])] for z in range(len(roi_spatial_footprints)) ]
                roi_temporal_residuals  = [ self.roi_temporal_residuals[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])] for z in range(len(roi_spatial_footprints)) ]
                bg_temporal_footprints  = [ self.bg_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])] for z in range(len(roi_spatial_footprints)) ]

            roi_data = {'roi_spatial_footprints' : roi_spatial_footprints,
                        'roi_temporal_footprints': roi_temporal_footprints,
                        'roi_temporal_residuals' : roi_temporal_residuals,
                        'bg_spatial_footprints'  : bg_spatial_footprints,
                        'bg_temporal_footprints' : bg_temporal_footprints,
                        'filtered_out_rois'      : filtered_out_rois,
                        'manually_removed_rois'  : manually_removed_rois,
                        'all_removed_rois'       : all_removed_rois,
                        'locked_rois'            : locked_rois,
                        'video_paths'            : [video_path],
                        'masks'                  : masks}

        # save the ROI data
        np.save(save_path, roi_data)
    
    def save_all_rois(self, save_directory):
        # set video paths
        if self.use_mc_video and len(self.mc_video_paths) > 0:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths
            
        for i in range(len(video_paths)):
            video_path = self.video_paths[i]

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

            group_num = self.video_groups[i]

            roi_spatial_footprints  = self.roi_spatial_footprints[group_num]
            roi_temporal_footprints = self.roi_temporal_footprints[group_num]
            roi_temporal_residuals  = self.roi_temporal_residuals[group_num]
            bg_spatial_footprints   = self.bg_spatial_footprints[group_num]
            bg_temporal_footprints  = self.bg_temporal_footprints[group_num]
            
            manually_removed_rois = self.manually_removed_rois[group_num]
            all_removed_rois      = self.all_removed_rois[group_num]
            locked_rois           = self.locked_rois[group_num]

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

                group_indices = [ i for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]
                group_paths   = [ self.video_paths[i] for i in group_indices ]
                group_lengths = [ self.video_lengths[i] for i in group_indices ]
                
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
                self.save_rois(os.path.join(video_dir_path, 'roi_data.npy'), group_num=group_num, video_path=video_path)

                print("Done.")

    def load_rois(self, load_path, group_num=None, video_path=None):
        # load the saved ROIs
        roi_data = np.load(load_path, allow_pickle=True)

        # extract the dictionary
        roi_data = roi_data[()]

        if group_num is None:
            # set ROI variables
            self.roi_spatial_footprints  = roi_data['roi_spatial_footprints']
            self.roi_temporal_footprints = roi_data['roi_temporal_footprints']
            self.roi_temporal_residuals  = roi_data['roi_temporal_residuals']
            self.bg_spatial_footprints   = roi_data['bg_spatial_footprints']
            self.bg_temporal_footprints  = roi_data['bg_temporal_footprints']
            self.filtered_out_rois       = roi_data['filtered_out_rois']
            if 'manually_removed_rois' in roi_data.keys():
                self.manually_removed_rois = roi_data['manually_removed_rois']
            else:
                self.manually_removed_rois = roi_data['discarded_rois']
            if 'all_removed_rois' in roi_data.keys():
                self.all_removed_rois = roi_data['all_removed_rois']
            else:
                self.all_removed_rois = roi_data['removed_rois']
            self.locked_rois = roi_data['locked_rois']
            if 'masks' in roi_data.keys():
                self.mask_points = roi_data['masks']
            else:
                self.mask_points = {}
                if len(self.video_paths) > 0:
                    # get number of z planes
                    video = tifffile.memmap(self.video_paths[0])
                    if len(video.shape) > 3:
                        num_z = video.shape[1]
                    else:
                        num_z = 1

                    for group_num in np.unique(self.video_groups):
                        self.mask_points[group_num] = [ [] for z in range(num_z) ]
        else:
            roi_spatial_footprints  = roi_data['roi_spatial_footprints']
            roi_temporal_footprints = roi_data['roi_temporal_footprints']
            roi_temporal_residuals  = roi_data['roi_temporal_residuals']
            bg_spatial_footprints   = roi_data['bg_spatial_footprints']
            bg_temporal_footprints  = roi_data['bg_temporal_footprints']
            filtered_out_rois       = roi_data['filtered_out_rois']
            if 'manually_removed_rois' in roi_data.keys():
                manually_removed_rois = roi_data['manually_removed_rois']
            else:
                manually_removed_rois = roi_data['discarded_rois']
            if 'all_removed_rois' in roi_data.keys():
                all_removed_rois = roi_data['all_removed_rois']
            else:
                all_removed_rois = roi_data['removed_rois']
            locked_rois = roi_data['locked_rois']
            if 'masks' in roi_data.keys():
                masks = roi_data['masks']
            else:
                if len(self.video_paths) > 0:
                    # get number of z planes
                    video = tifffile.memmap(self.video_paths[0])
                    if len(video.shape) > 3:
                        num_z = video.shape[1]
                    else:
                        num_z = 1

                    masks = [ [] for z in range(num_z) ]

            self.roi_spatial_footprints[group_num]  = roi_spatial_footprints
            self.bg_spatial_footprints[group_num]   = bg_spatial_footprints
            self.filtered_out_rois[group_num]       = filtered_out_rois
            self.manually_removed_rois[group_num]   = manually_removed_rois
            self.all_removed_rois[group_num]        = all_removed_rois
            self.locked_rois[group_num]             = locked_rois
            self.mask_points[group_num]             = masks
            self.roi_temporal_footprints[group_num] = roi_temporal_footprints
            self.roi_temporal_residuals[group_num]  = roi_temporal_residuals
            self.bg_temporal_footprints[group_num]  = bg_temporal_footprints

            # group_indices = [ i for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]
            # group_lengths = [ self.video_lengths[i] for i in group_indices ]

            # if self.use_mc_video and len(self.mc_video_paths) > 0:
            #     group_paths = [ self.mc_video_paths[i] for i in group_indices ]
            # else:
            #     group_paths = [ self.video_paths[i] for i in group_indices ]

            # index = group_paths.index(video_path)

            # if group_num not in self.roi_temporal_footprints.keys():
            #     self.roi_temporal_footprints[group_num] = [ np.zeros((self.roi_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]
            #     self.roi_temporal_residuals[group_num]  = [ np.zeros((self.roi_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]
            #     self.bg_temporal_footprints[group_num]  = [ np.zeros((self.bg_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]

            # print(self.roi_temporal_footprints.keys())
            # print(group_lengths)
            # print(self.roi_spatial_footprints[group_num][z].shape)
            # print(roi_temporal_residuals[z].shape)

            # if index == 0:
            #     for z in range(len(roi_spatial_footprints)):
            #         # print(z, roi_spatial_footprints[z].shape)
            #         # print(z, bg_spatial_footprints[z].shape)
            #         # print(z, roi_temporal_footprints[z].shape)
            #         # print(z, roi_temporal_residuals[z].shape)
            #         # print(z, bg_temporal_footprints[z].shape)
            #         self.roi_temporal_footprints[group_num][z][:, :group_lengths[0]] = roi_temporal_footprints[z]
            #         self.roi_temporal_residuals[group_num][z][:, :group_lengths[0]]  = roi_temporal_residuals[z]
            #         self.bg_temporal_footprints[group_num][z][:, :group_lengths[0]]  = bg_temporal_footprints[z]
            # else:
            #     for z in range(len(roi_spatial_footprints)):
            #         self.roi_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])] = roi_temporal_footprints[z]
            #         self.roi_temporal_residuals[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]  = roi_temporal_residuals[z]
            #         self.bg_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]  = bg_temporal_footprints[z]

        self.find_new_rois = False

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths, lengths and groups at the indices, in reverse order
            index = indices[i]

            if self.video_groups.count(self.video_groups[index]) == 1:
                # if this is the last video in the group, remove the group
                self.remove_group(self.video_groups[index], remove_videos=False)

            del self.video_paths[index]
            del self.video_lengths[index]
            del self.video_groups[index]

            if len(self.mc_video_paths) > 0:
                del self.mc_video_paths[index]

        if len(self.video_paths) == 0:
            # reset variables
            self.reset_variables()
            self.reset_motion_correction_variables()
            self.reset_roi_finding_variables()
            self.reset_roi_filtering_variables()

    def add_group(self, group_num):
        print("Adding group {}.".format(group_num))

        video_paths = [ self.video_paths[i] for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]

        if len(video_paths) > 0:
            # get number of z planes
            video = tifffile.memmap(video_paths[0])
            if len(video.shape) > 3:
                num_z = video.shape[1]
            else:
                num_z = 1

            self.mask_points[group_num] = [ [] for z in range(num_z) ]

    def remove_group(self, group, remove_videos=True):
        if group in self.mc_borders.keys():
            del self.mc_borders[group]
        if group in self.roi_spatial_footprints.keys():
            del self.roi_spatial_footprints[group]
        if group in self.roi_temporal_footprints.keys():
            del self.roi_temporal_footprints[group]
        if group in self.roi_temporal_residuals.keys():
            del self.roi_temporal_residuals[group]
        if group in self.bg_spatial_footprints.keys():
            del self.bg_spatial_footprints[group]
        if group in self.bg_temporal_footprints.keys():
            del self.bg_temporal_footprints[group]
        if group in self.filtered_out_rois.keys():
            del self.filtered_out_rois[group]
        if group in self.mask_points.keys():
            del self.mask_points[group]
        if group in self.manually_removed_rois.keys():
            del self.manually_removed_rois[group]
        if group in self.all_removed_rois.keys():
            del self.all_removed_rois[group]
        if group in self.locked_rois.keys():
            del self.locked_rois[group]

        if remove_videos:
	        video_indices = self.video_indices_in_group(self.video_paths, group)

	        self.remove_videos_at_indices(video_indices)

    def video_paths_in_group(self, video_paths, group_num):
        return [ video_paths[i] for i in range(len(video_paths)) if self.video_groups[i] == group_num ]

    def video_indices_in_group(self, video_paths, group_num):
        return [ i for i in range(len(video_paths)) if self.video_groups[i] == group_num ]

    def motion_correct(self):
        mc_videos, mc_borders = utilities.motion_correct_multiple_videos(self.video_paths, self.video_groups, self.params['max_shift'], self.params['patch_stride'], self.params['patch_overlap'], use_multiprocessing=self.use_multiprocessing)

        mc_video_paths = []
        for i in range(len(mc_videos)):
            video_path    = self.video_paths[i]
            directory     = os.path.dirname(video_path)
            filename      = os.path.basename(video_path)
            mc_video_path = os.path.join(directory, os.path.splitext(filename)[0] + "_mc.tif")
            
            # save the motion-corrected video
            tifffile.imsave(mc_video_path, mc_videos[i])
            
            mc_video_paths.append(mc_video_path)

        self.mc_video_paths = mc_video_paths
        self.mc_borders     = mc_borders

        self.use_mc_video = True

    def find_rois(self):
        # set video paths
        if self.use_mc_video and len(self.mc_video_paths) > 0:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        roi_spatial_footprints, roi_temporal_footprints, roi_temporal_residuals, bg_spatial_footprints, bg_temporal_footprints = utilities.find_rois_multiple_videos(video_paths, self.video_groups, self.params, mc_borders=self.mc_borders, use_multiprocessing=self.use_multiprocessing, method=self.roi_finding_mode)

        self.roi_spatial_footprints  = roi_spatial_footprints
        self.roi_temporal_footprints = roi_temporal_footprints
        self.roi_temporal_residuals  = roi_temporal_residuals
        self.bg_spatial_footprints   = bg_spatial_footprints
        self.bg_temporal_footprints  = bg_temporal_footprints

        self.filtered_out_rois     = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.video_groups) }
        self.manually_removed_rois = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.video_groups) }
        self.all_removed_rois      = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.video_groups) }
        self.locked_rois           = { group_num: [ [] for z in range(len(roi_spatial_footprints[group_num])) ] for group_num in np.unique(self.video_groups) }

    def filter_rois(self, mean_images, group_num):
        # set video paths
        if self.use_mc_video and len(self.mc_video_paths) > 0:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        # only use videos in the given group
        video_paths = self.video_paths_in_group(video_paths, group_num)

        # filter out ROIs and update the removed ROIs
        self.filtered_out_rois[group_num] = utilities.filter_rois(video_paths, self.roi_spatial_footprints[group_num], self.roi_temporal_footprints[group_num], self.roi_temporal_residuals[group_num], self.bg_spatial_footprints[group_num], self.bg_temporal_footprints[group_num], mean_images, self.params)
        
        # keep locked ROIs
        for z in range(len(self.filtered_out_rois[group_num])):
            self.manually_removed_rois[group_num][z] = []
            self.filtered_out_rois[group_num][z] = [ roi for roi in self.filtered_out_rois[group_num][z] if roi not in self.locked_rois[group_num][z] ]
            self.all_removed_rois[group_num][z]      = self.filtered_out_rois[group_num][z] + self.manually_removed_rois[group_num][z]

    def discard_roi(self, roi, z, group_num):
        # add to discarded ROIs list
        self.manually_removed_rois[group_num][z].append(roi)
        self.all_removed_rois[group_num][z] = self.filtered_out_rois[group_num][z] + self.manually_removed_rois[group_num][z]

        # remove from locked ROIs if it's there
        if roi in self.locked_rois[group_num][z]:
            i = self.locked_rois[group_num][z].index(roi)
            del self.locked_rois[group_num][z][i]

    def keep_roi(self, roi, z, group_num):
        # remove from discared ROIs or filtered out ROIs list if it's there
        if roi in self.manually_removed_rois[group_num][z]:
            i = self.manually_removed_rois[group_num][z].index(roi)
            del self.manually_removed_rois[group_num][z][i]
        elif roi in self.filtered_out_rois[group_num][z]:
            i = self.filtered_out_rois[group_num][z].index(roi)
            del self.filtered_out_rois[group_num][z][i]

        # add to locked ROIs list
        if roi not in self.locked_rois[group_num][z]:
            self.locked_rois[group_num][z].append(roi)

        self.all_removed_rois[group_num][z] = self.filtered_out_rois[group_num][z] + self.manually_removed_rois[group_num][z]

    def add_mask(self, mask_points, z, num_z, group_num):
        if len(mask_points) >= 3:
            if group_num not in self.mask_points.keys():
                self.mask_points[group_num] = [ [] for z in range(num_z) ]
                
            self.mask_points[group_num][z].append(mask_points)

    def delete_mask(self, mask_num, z, group_num):
        if mask_num < len(self.mask_points[group_num][z]):
            del self.mask_points[group_num][z][mask_num]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))
