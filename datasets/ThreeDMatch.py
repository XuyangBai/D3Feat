# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import pickle
import open3d

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
import os.path as path

# Dataset parent class
from datasets.common import Dataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def rotate(points, num_axis=1):
    if num_axis == 1:
        theta = np.random.rand() * 2 * np.pi
        axis = np.random.randint(3)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)
    elif num_axis == 3:
        for axis in [0, 1, 2]:
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
            R[:, axis] = 0
            R[axis, :] = 0
            R[axis, axis] = 1
            points = np.matmul(points, R)
    else:
        exit(-1)
    return points


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class ThreeDMatchDataset(Dataset):
    """
    Class to handle ThreeDMatch dataset for dense keypoint detection and feature description task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8, voxel_size=0.03, load_test=False):
        Dataset.__init__(self, 'ThreeDMatch')

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        self.network_model = 'descriptor'

        # Number of input threads
        self.num_threads = input_threads

        # Load test set or train set?
        self.load_test = load_test

        # voxel size
        self.downsample = voxel_size

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.root = 'data/3DMatch/'

        # Initiate containers
        self.anc_points = {'train': [], 'val': [], 'test': []}
        self.keypts = {'train': [], 'val': [], 'test': []}
        self.anc_to_pos = {'train': {}, 'val': {}, 'test': {}}
        self.ids_list = {'train': [], 'val': [], 'test': []}

        if self.load_test:
            self.prepare_geometry_registration()
        else:
            self.prepare_3dmatch_ply(split='train')
            self.prepare_3dmatch_ply(split='val')

    def prepare_3dmatch_ply(self, split='train'):
        """
        Load pre-generated point cloud, keypoint correspondence(the indices) to save time. 
        Construct the self.anc_to_pos dictionary.
        """

        print('\nPreparing ply files')
        pts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_points.pkl')
        keypts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_keypts.pkl')

        if exists(pts_filename) and exists(keypts_filename):
            with open(pts_filename, 'rb') as file:
                data = pickle.load(file)
                self.anc_points[split] = [*data.values()]
                self.ids_list[split] = [*data.keys()]
            with open(keypts_filename, 'rb') as file:
                self.keypts[split] = pickle.load(file)
        else:
            print("PKL file not found.")
            return

        for idpair in self.keypts[split].keys():
            anc = idpair.split("@")[0]
            pos = idpair.split("@")[1]
            # add (key -> value)  anc -> pos 
            if anc not in self.anc_to_pos[split].keys():
                self.anc_to_pos[split][anc] = [pos]
            else:
                self.anc_to_pos[split][anc] += [pos]
        if split == 'train':
            self.num_train = len(list(self.anc_to_pos[split].keys()))
            print("Num_train", self.num_train)
        else:
            self.num_val = len(list(self.anc_to_pos[split].keys()))
            print("Num_val", self.num_val)
        return

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}

        # Reset potentials
        self.potentials[split] = np.random.rand(len(self.anc_points[split])) * 1e-3

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
            if split == 'train':
                gen_indices = np.random.permutation(self.num_train)
                # gen_indices = np.arange(self.num_train)

            elif split == 'val':
                gen_indices = np.random.permutation(self.num_val)
                # gen_indices = np.arange(self.num_val)

            elif split == 'test':
                gen_indices = np.arange(self.num_test)

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            print(gen_indices)
            # Generator loop
            for p_i in gen_indices:

                if split == 'test':
                    anc_id = self.ids_list[split][p_i]
                    pos_id = self.ids_list[split][p_i]
                else:
                    anc_id = list(self.anc_to_pos[split].keys())[p_i]
                    import random
                    if random.random() > 0.5:
                        pos_id = self.anc_to_pos[split][anc_id][0]
                    else:
                        pos_id = random.choice(self.anc_to_pos[split][anc_id])

                anc_ind = self.ids_list[split].index(anc_id)
                pos_ind = self.ids_list[split].index(pos_id)
                anc_points = self.anc_points[split][anc_ind].astype(np.float32)
                pos_points = self.anc_points[split][pos_ind].astype(np.float32)
                # back up point cloud
                backup_anc_points = anc_points
                backup_pos_points = pos_points

                n = anc_points.shape[0] + pos_points.shape[0]

                if split == 'test':  # for test, use all 5000 the anc_keypts
                    anc_keypts = np.array([])
                    pos_keypts = np.array([])
                    # add rotation to test on Rotated3DMatch
                    # anc_points = rotate(anc_points, num_axis=3)
                    # pos_points = rotate(pos_points, num_axis=3)
                else:
                    if anc_points.shape[0] > 80000 or pos_points.shape[0] > 80000:
                        continue
                    if anc_points.shape[0] < 2000 or pos_points.shape[0] < 2000:
                        continue
                    anc_keypts = self.keypts[split][f'{anc_id}@{pos_id}'][:, 0]
                    pos_keypts = self.keypts[split][f'{anc_id}@{pos_id}'][:, 1]
                    if split == 'train':
                        selected_ind = np.random.choice(min(len(anc_keypts), len(pos_keypts)), config.keypts_num, replace=True)
                    else:
                        selected_ind = np.random.choice(min(len(anc_keypts), len(pos_keypts)), config.keypts_num, replace=True)
                    anc_keypts = anc_keypts[selected_ind]
                    pos_keypts = pos_keypts[selected_ind] + len(anc_points)

                    # if split == 'train':
                    #     # training does not need this keypts 
                    #     anc_keypts = np.random.choice(len(anc_points), 200)
                    #     pos_keypts = np.random.choice(len(anc_points), 200)
                    # else:
                    #     # find the correspondence by nearest neighbors sourch.
                    #     anc_keypts = np.random.choice(len(anc_points), 400)
                    #     pos_pcd = open3d.PointCloud()
                    #     pos_pcd.points = open3d.utility.Vector3dVector(pos_points)
                    #     kdtree = open3d.geometry.KDTreeFlann(pos_pcd)
                    #     pos_ind = []
                    #     anc_ind = []
                    #     for pts, i in zip(anc_points[anc_keypts], anc_keypts):
                    #         _, ind, dis = kdtree.search_knn_vector_3d(pts, 1)
                    #         if dis[0] < 0.001 and ind[0] not in pos_ind and i not in anc_ind:
                    #             pos_ind.append(ind[0])
                    #             anc_ind.append(i)
                    #             if len(anc_ind) >= config.keypts_num:
                    #                 break

                    #     anc_keypts = np.array(anc_ind)
                    #     pos_keypts = np.array(pos_ind)
                    #     pos_keypts = pos_keypts + len(anc_points)

                    # # No matter how many num_keypts are used for training, test only use 64 pair.
                    # if len(anc_keypts) >= config.keypts_num:
                    #     if split == 'train':
                    #         selected_ind = np.random.choice(range(len(anc_keypts)), config.keypts_num, replace=False)
                    #     else:
                    #         selected_ind = np.random.choice(range(len(anc_keypts)), 64, replace=False)
                    #     anc_keypts = anc_keypts[selected_ind]
                    #     pos_keypts = pos_keypts[selected_ind]
                    # else: # if can not build enough correspondence, then skip this fragments pair.
                    #     continue

                    # data augmentations: noise
                    anc_noise = np.random.rand(anc_points.shape[0], 3) * config.augment_noise
                    pos_noise = np.random.rand(pos_points.shape[0], 3) * config.augment_noise
                    anc_points += anc_noise
                    pos_points += pos_noise
                    # data augmentations: rotation
                    anc_points = rotate(anc_points, num_axis=config.augment_rotation)
                    pos_points = rotate(pos_points, num_axis=config.augment_rotation)

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

    def prepare_geometry_registration(self):
        """
        Prepare the point cloud and keypoints indices (if use predefined keypoints) for testing (geometric registration)
        """
        scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        self.num_test = 0
        for scene in scene_list:
            self.test_path = f'{self.root}/fragments/{scene}'
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = open3d.read_point_cloud(join(self.test_path, ind))
                pcd = open3d.voxel_down_sample(pcd, voxel_size=0.03)

                # keypts_location = np.fromfile(join(self.test_path, ind.replace("ply", "keypts.bin")), dtype=np.float32)
                # num_keypts = int(keypts_location[0])
                # keypts_location = keypts_location[1:].reshape([num_keypts, 3])

                # # find the keypoint indices selected by 3DMatch.
                # kdtree = open3d.KDTreeFlann(pcd)
                # keypts_id = []
                # for j in range(keypts_location.shape[0]):
                #     _, id, _ = kdtree.search_knn_vector_3d(keypts_location[j], 1)
                #     keypts_id.append(id[0])
                # keypts_id = np.array(keypts_id)

                # Load points and labels
                points = np.array(pcd.points)

                self.anc_points['test'] += [points]
                self.ids_list['test'] += [scene + '/' + ind]
        return
