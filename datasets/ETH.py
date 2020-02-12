# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import pickle
import open3d

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
import os.path as path

# Dataset parent class
from datasets.common import Dataset
from datasets.ThreeDMatch import rotate


class ETHDataset(Dataset):
    def __init__(self, input_threads=8, load_test=False):
        Dataset.__init__(self, 'ETH')
        # Type of task conducted on this dataset
        self.network_model = 'descriptor'

        # Number of input threads
        self.num_threads = input_threads

        # Load test set or train set?
        self.load_test = load_test

        ##########################
        # Parameters for the files
        ##########################

        # Initiate containers
        self.anc_points = {'test': []}
        self.ids_list = {'test': []}

        if self.load_test:
            self.prepare_geometry_registration_eth()
        else:
            exit(-1)

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        ################
        # Def generators
        ################

        def random_balanced_gen():

            # Initiate concatenation lists
            anc_points_list = []
            pos_points_list = []
            anc_keypts_list = []
            pos_keypts_list = []
            backup_anc_points_list = []
            backup_pos_points_list = []
            ti_list = []
            ti_list_pos = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'train' or split == 'val':
                exit(-1)

            elif split == 'test':
                gen_indices = np.arange(self.num_test)

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            print(gen_indices)
            for p_i in gen_indices:
                anc_id = self.ids_list[split][p_i]
                pos_id = self.ids_list[split][p_i]
                anc_ind = self.ids_list[split].index(anc_id)
                pos_ind = self.ids_list[split].index(pos_id)
                anc_points = self.anc_points[split][anc_ind].astype(np.float32)
                pos_points = self.anc_points[split][pos_ind].astype(np.float32)

                # back up point cloud
                backup_anc_points = anc_points
                backup_pos_points = pos_points

                anc_keypts = np.array([])
                pos_keypts = np.array([])

                # Add data to current batch
                anc_points_list += [anc_points]
                anc_keypts_list += [anc_keypts]
                pos_points_list += [pos_points]
                pos_keypts_list += [pos_keypts]
                backup_anc_points_list += [backup_anc_points]
                backup_pos_points_list += [backup_pos_points]
                ti_list += [p_i]
                ti_list_pos += [p_i]

                yield (np.concatenate(anc_points_list + pos_points_list, axis=0),  # anc_points
                       np.concatenate(anc_keypts_list, axis=0),  # anc_keypts
                       np.concatenate(pos_keypts_list, axis=0),
                       np.array(ti_list + ti_list_pos, dtype=np.int32),  # anc_obj_index
                       np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]),  # anc_stack_length
                       np.array([anc_id, pos_id]),
                       np.concatenate(backup_anc_points_list + backup_pos_points_list, axis=0)
                       )
                anc_points_list = []
                pos_points_list = []
                anc_keypts_list = []
                pos_keypts_list = []
                backup_anc_points_list = []
                backup_pos_points_list = []
                ti_list = []
                ti_list_pos = []

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        # gen_types = (tf.float32, tf.int32, tf.int32, tf.int32,  tf.float32, tf.int32, tf.int32, tf.int32)
        # gen_shapes = ([None, 3], [None], [None], [None], [None, 3], [None], [None], [None])
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config,
                                                         anc_points,
                                                         stacked_features,
                                                         stack_lengths,
                                                         batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points]

        return tf_map

    def prepare_geometry_registration_eth(self):
        scene_list = [
            'gazebo_summer',
            'gazebo_winter',
            'wood_autmn',
            'wood_summer',
        ]
        self.num_test = 0
        for scene in scene_list:
            ply_path = 'data/ETH/{}'.format(scene)
            keypts_path = os.path.join(ply_path, '01_Keypoints/')
            pcd_list = [filename for filename in os.listdir(ply_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = open3d.read_point_cloud(join(ply_path, ind))
                pcd = open3d.voxel_down_sample(pcd, voxel_size=0.0625)

                # with open(os.path.join(keypts_path, f'Hokuyo_{i}_Keypoints.txt'), 'r') as f:
                #     keypts_id = np.array([int(x.replace("\n", "")) for x in f.readlines()])
                # keypts_location = np.asarray(pcd.points)[keypts_id]

                # if use predefined keypoints location, find the keypoint indices
                # kdtree = open3d.KDTreeFlann(pcd)
                # keypts_id = []
                # for j in range(keypts_location.shape[0]):
                #     _, id, _ = kdtree.search_knn_vector_3d(keypts_location[j], 1)
                #     keypts_id.append(id[0])
                # # Load points and labels
                # keypts_id = np.array(keypts_id)

                points = np.asarray(pcd.points)
                print(points.shape)

                self.anc_points['test'] += [points]
                self.ids_list['test'] += [scene + '/' + ind]
        return
