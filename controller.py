import os
import json
import numpy as np
import skimage.external.tifffile as tifffile

import utilities

# set default parameters dictionary
DEFAULT_PARAMS = {'gamma'                : 1.0,
                  'contrast'             : 1.0,
                  'fps'                  : 60,
                  'z'                    : 0,
                  'max_shift'            : 6,
                  'patch_stride'         : 48,
                  'patch_overlap'        : 24,
                  'window_size'          : 7,
                  'background_threshold' : 10,
                  'invert_masks'         : False,
                  'soma_threshold'       : 0.8,
                  'min_area'             : 10,
                  'max_area'             : 100,
                  'min_circ'             : 0,
                  'max_circ'             : 2,
                  'imaging_fps'          : 30,
                  'decay_time'           : 0.4,
                  'autoregressive_order' : 0,
                  'num_bg_components'    : 2,
                  'merge_threshold'      : 0.8,
                  'num_components'       : 400,
                  'half_size'            : 4,
                  'use_cnn'              : False,
                  'min_snr'              : 1.3,
                  'min_spatial_corr'     : 0.8,
                  'use_cnn'              : False,
                  'cnn_threshold'        : 0.5,
                  'diameter'             : 10,
                  'sampling_rate'        : 3,
                  'connected'            : True,
                  'neuropil_basis_ratio' : 6,
                  'neuropil_radius_ratio': 3,
                  'inner_neuropil_radius': 2,
                  'min_neuropil_pixels'  : 350
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
        self.tail_angles   = [] # tail angle traces for all videos

        # initialize all variables
        self.reset_variables()
        self.reset_motion_correction_variables()
        self.reset_roi_finding_variables()
        self.reset_roi_filtering_variables()

    def reset_variables(self):
        self.use_mc_video        = False # whether to use the motion-corrected video for finding ROIs
        self.use_multiprocessing = True  # whether to use multi-processing

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

    def reset_roi_filtering_variables(self):
        self.discarded_rois = {}
        self.removed_rois   = {}
        self.locked_rois    = {}

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
            video = tifffile.imread(video_path)
            self.video_lengths.append(video.shape[0])
            self.video_groups.append(group_num)
            self.tail_angles.append(None)

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
                        'discarded_rois'         : self.discarded_rois,
                        'removed_rois'           : self.removed_rois,
                        'locked_rois'            : self.locked_rois,
                        'video_paths'            : video_paths}
        else:
            group_indices = [ i for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]
            group_lengths = [ self.video_lengths[i] for i in group_indices ]
            group_paths   = [ self.video_paths[i] for i in group_indices ]

            index = group_paths.index(video_path)

            roi_spatial_footprints = self.roi_spatial_footprints[group_num]
            bg_spatial_footprints  = self.bg_spatial_footprints[group_num]
            filtered_out_rois      = self.filtered_out_rois[group_num]
            discarded_rois         = self.discarded_rois[group_num]
            removed_rois           = self.removed_rois[group_num]
            locked_rois            = self.locked_rois[group_num]
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
                        'discarded_rois'         : discarded_rois,
                        'removed_rois'           : removed_rois,
                        'locked_rois'            : locked_rois,
                        'video_paths'            : [video_path]}

        # save the ROI data
        np.save(save_path, roi_data)

    def load_rois(self, load_path, group_num=None, video_path=None):
        # load the saved ROIs
        roi_data = np.load(load_path)

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
            self.discarded_rois          = roi_data['discarded_rois']
            self.removed_rois            = roi_data['removed_rois']
            self.locked_rois             = roi_data['locked_rois']
        else:
            roi_spatial_footprints  = roi_data['roi_spatial_footprints']
            roi_temporal_footprints = roi_data['roi_temporal_footprints']
            roi_temporal_residuals  = roi_data['roi_temporal_residuals']
            bg_spatial_footprints   = roi_data['bg_spatial_footprints']
            bg_temporal_footprints  = roi_data['bg_temporal_footprints']
            filtered_out_rois       = roi_data['filtered_out_rois']
            discarded_rois          = roi_data['discarded_rois']
            removed_rois            = roi_data['removed_rois']
            locked_rois             = roi_data['locked_rois']

            self.roi_spatial_footprints[group_num] = roi_spatial_footprints
            self.bg_spatial_footprints[group_num]  = bg_spatial_footprints
            self.filtered_out_rois[group_num]      = filtered_out_rois
            self.discarded_rois[group_num]         = discarded_rois
            self.removed_rois[group_num]           = removed_rois
            self.locked_rois[group_num]            = locked_rois

            group_indices = [ i for i in range(len(self.video_paths)) if self.video_groups[i] == group_num ]
            group_lengths = [ self.video_lengths[i] for i in group_indices ]
            group_paths   = [ self.video_paths[i] for i in group_indices ]

            index = group_paths.index(video_path)

            if group_num not in self.roi_temporal_footprints.keys():
                self.roi_temporal_footprints[group_num] = [ np.zeros((self.roi_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]
                self.roi_temporal_residuals[group_num]  = [ np.zeros((self.roi_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]
                self.bg_temporal_footprints[group_num]  = [ np.zeros((self.bg_spatial_footprints[group_num][z].shape[1], np.sum(group_lengths))) for z in range(len(roi_spatial_footprints)) ]

            if index == 0:
                for z in range(len(roi_spatial_footprints)):
                    self.roi_temporal_footprints[group_num][z][:, :group_lengths[0]] = roi_temporal_footprints[z]
                    self.roi_temporal_residuals[group_num][z][:, :group_lengths[0]]  = roi_temporal_residuals[z]
                    self.bg_temporal_footprints[group_num][z][:, :group_lengths[0]]  = bg_temporal_footprints[z]
            else:
                for z in range(len(roi_spatial_footprints)):
                    self.roi_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])] = roi_temporal_footprints[z]
                    self.roi_temporal_residuals[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]  = roi_temporal_residuals[z]
                    self.bg_temporal_footprints[group_num][z][:, np.sum(group_lengths[:index]):np.sum(group_lengths[:index+1])]  = bg_temporal_footprints[z]

        self.find_new_rois = False

    def remove_videos_at_indices(self, indices):
        # sort the indices in increasing order
        indices = sorted(indices)

        for i in range(len(indices)-1, -1, -1):
            # remove the video paths, lengths and groups at the indices, in reverse order
            index = indices[i]
            del self.video_paths[index]
            del self.video_lengths[index]
            del self.video_groups[index]
            del self.tail_angles[index]

            if len(self.mc_video_paths) > 0:
                del self.mc_video_paths[index]

        if len(self.video_paths) == 0:
            # reset variables
            self.reset_variables()

    def video_paths_in_group(self, video_paths, group_num):
        return [ video_paths[i] for i in range(len(video_paths)) if self.video_groups[i] == group_num ]

    def filter_rois(self, group_num):
        # set video paths
        if self.use_mc_video and len(self.mc_video_paths) > 0:
            video_paths = self.mc_video_paths
        else:
            video_paths = self.video_paths

        # only use videos in the given group
        video_paths = self.video_paths_in_group(video_paths, group_num)

        # filter out ROIs and update the removed ROIs
        self.filtered_out_rois[group_num] = utilities.filter_rois(video_paths, self.roi_spatial_footprints[group_num], self.roi_temporal_footprints[group_num], self.roi_temporal_residuals[group_num], self.bg_spatial_footprints[group_num], self.bg_temporal_footprints[group_num], self.params)
        
        # keep locked ROIs
        for z in range(len(self.filtered_out_rois[group_num])):
            self.filtered_out_rois[group_num][z] = [ roi for roi in self.filtered_out_rois[group_num][z] if roi not in self.locked_rois[group_num][z] ]
            self.removed_rois[group_num][z]      = self.filtered_out_rois[group_num][z] + self.discarded_rois[group_num][z]

    def discard_roi(self, roi, z, group_num):
        # add to discarded ROIs list
        self.discarded_rois[group_num][z].append(roi)
        self.removed_rois[group_num][z] = self.filtered_out_rois[group_num][z] + self.discarded_rois[group_num][z]

        # remove from locked ROIs if it's there
        if roi in self.locked_rois[group_num][z]:
            i = self.locked_rois[group_num][z].index(roi)
            del self.locked_rois[group_num][z][i]

    def keep_roi(self, roi, z, group_num):
        # remove from discared ROIs or filtered out ROIs list if it's there
        if roi in self.discarded_rois[group_num][z]:
            i = self.discarded_rois[group_num][z].index(roi)
            del self.discarded_rois[group_num][z][i]
        elif roi in self.filtered_out_rois[group_num][z]:
            i = self.filtered_out_rois[group_num][z].index(roi)
            del self.filtered_out_rois[group_num][z][i]

            # add to locked ROIs list
            if roi not in self.locked_rois[group_num][z]:
                self.locked_rois[group_num][z].append(roi)

        self.removed_rois[group_num][z] = self.filtered_out_rois[group_num][z] + self.discarded_rois[group_num][z]

    def save_params(self):
        json.dump(self.params, open(PARAMS_FILENAME, "w"))
