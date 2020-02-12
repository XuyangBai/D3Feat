#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle ThreeDMatch dataset in a class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import pickle
import open3d
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
import os.path as path

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


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
        return points
    else:
        theta = np.random.rand() * 2 * np.pi
        axis = 0
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)

        theta = np.random.rand() * 2 * np.pi
        axis = 1
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)

        theta = np.random.rand() * 2 * np.pi
        axis = 2
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)
        return points


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class ThreeDMatchDataset(Dataset):
    """
    Class to handle ThreeDMatch dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8, load_test=False):
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

        ##########################
        # Parameters for the files
        ##########################

        # Initiate containers
        self.anc_points = {'train': [], 'val': [], 'test': []}
        self.pos_points = {'train': [], 'val': [], 'test': []}
        # actually this keypoints file is not used in training process
        self.anc_keypts = {'train': [], 'val': [], 'test': []}
        self.pos_keypts = {'train': [], 'val': [], 'test': []}
        self.idpair_list = {'train': [], 'val': [], 'test': []}
        self.anc_to_pos = {'train': {}, 'val': {}, 'test': {}}
        self.ids_list = {'train': [], 'val': [], 'test': []}

        if self.load_test:
            self.prepare_geometry_registration()
        else:
            self.prepare_3dmatch_ply(split='train')
            self.prepare_3dmatch_ply(split='val')

    def prepare_3dmatch_ply(self, split='train'):

        print('\nPreparing ply files')
        t0 = time.time()

        # pos_keypts_filename = join(pklpath, '{}_{:.3f}_pos_keypts.pkl'.format(split, 0.03))
        # pos_points_filename = join(self.writepath, '{}_{:.3f}_pos_points.pkl'.format(split, self.downsample))
        if split == 'train':
            self.files = {'train': [], 'test': [], 'val': []}
            self.DATA_FILES = {
                'train': '../FCGF/config/train_3dmatch.txt',
                'val': '../FCGF/config/val_3dmatch.txt',
                'test': '../FCGF/config/test_3dmatch.txt'
            }
            self.root = '../FCGF/data/3DMatch/threedmatch'
        import glob
        subset_names = open(self.DATA_FILES[split]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % 0.3
            fnames_txt = glob.glob(self.root + "/" + fname)
            import pdb
            pdb.set_trace()
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files[split].append([fname[0], fname[1]])

        for idpair in self.files[split]:
            anc = idpair[0]
            pos = idpair[1]
            if anc not in self.anc_to_pos[split].keys():
                self.anc_to_pos[split][anc] = [pos]
            else:
                self.anc_to_pos[split][anc] += [pos]

        if split == 'train':
            self.num_train = len(self.anc_to_pos[split].keys())
            print("Num_train", self.num_train)
        else:
            self.num_val = len(self.anc_to_pos[split].keys())
            print("Num_test", self.num_val)
        import pdb
        pdb.set_trace()

        if exists(anc_points_filename) and exists(ids_pair_filename) and exists(anc_keypts_filename):
            with open(anc_points_filename, 'rb') as file:
                self.anc_points[split] = pickle.load(file)

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
                    # anc_id = list(self.anc_to_pos[split].keys())[p_i]
                    # import random
                    # if random.random() > 0.75:
                    # pos_id = self.anc_to_pos[split][anc_id][0]
                    # else:
                    # pos_id = random.choice(self.anc_to_pos[split][anc_id])
                    # anc_id = self.files[split][p_i][0]
                    anc_id = [*self.anc_to_pos[split].keys()][p_i]
                    import random
                    if random.random() > 0.5:
                        pos_id = self.anc_to_pos[split][anc_id][0]
                        # print("***", anc_id, pos_id)
                    else:
                        pos_id = random.choice(self.anc_to_pos[split][anc_id])
                        # print("+++", anc_id, pos_id)
                    file0 = os.path.join(self.root, anc_id)
                    file1 = os.path.join(self.root, pos_id)
                    data0 = np.load(file0)
                    data1 = np.load(file1)
                    xyz0 = data0["pcd"]
                    xyz1 = data1["pcd"]
                    anc_pcd = open3d.PointCloud()
                    anc_pcd.points = open3d.utility.Vector3dVector(xyz0)
                    anc_pcd = open3d.voxel_down_sample(anc_pcd, voxel_size=0.03)
                    pos_pcd = open3d.PointCloud()
                    pos_pcd.points = open3d.utility.Vector3dVector(xyz1)
                    pos_pcd = open3d.voxel_down_sample(pos_pcd, voxel_size=0.03)
                    anc_points = np.asarray(anc_pcd.points)
                    pos_points = np.asarray(pos_pcd.points)

                # anc_ind = self.ids_list[split].index(anc_id)
                # pos_ind = self.ids_list[split].index(pos_id)
                # anc_points = self.anc_points[split][anc_ind].astype(np.float32)
                # pos_points = self.anc_points[split][pos_ind].astype(np.float32)
                # back up point cloud
                backup_anc_points = anc_points
                backup_pos_points = pos_points

                n = anc_points.shape[0] + pos_points.shape[0]

                if split == 'test':  # for test, use all 5000 the anc_keypts
                    anc_keypts = self.anc_keypts[split][anc_ind].astype(np.int32)
                    pos_keypts = self.pos_keypts[split][pos_ind].astype(np.int32)
                    assert (np.array_equal(anc_keypts, pos_keypts))
                    anc_keypts = anc_keypts
                    pos_keypts = anc_keypts + len(anc_points)
                    # pos_keypts = self.pos_keypts[split][ind].astype(np.int32) + len(anc_points)
                    assert (np.array_equal(anc_points, pos_points))
                    assert (np.array_equal(anc_keypts, pos_keypts - len(anc_points)))
                else:
                    # here the anc_keypts and pos_keypts is useless
                    # anc_keypts = np.random.choice(len(anc_points), 10)
                    # pos_keypts = np.random.choice(len(pos_points), 10)
                    # if anc_points.shape[0] > 60000 or pos_points.shape[0] > 60000:
                    # continue
                    if anc_points.shape[0] < 2000 or pos_points.shape[0] < 2000:
                        continue

                    if split == 'fake train':
                        # training does not need this keypts 
                        anc_keypts = np.random.choice(len(anc_points), 200)
                        pos_keypts = np.random.choice(len(anc_points), 200)
                    else:
                        anc_keypts = np.random.choice(len(anc_points), 400)
                        pos_pcd = open3d.PointCloud()
                        pos_pcd.points = open3d.utility.Vector3dVector(pos_points)
                        kdtree = open3d.geometry.KDTreeFlann(pos_pcd)
                        pos_ind = []
                        anc_ind = []
                        for pts, i in zip(anc_points[anc_keypts], anc_keypts):
                            _, ind, dis = kdtree.search_knn_vector_3d(pts, 1)
                            if dis[0] < 0.001 and ind[0] not in pos_ind and i not in anc_ind:
                                pos_ind.append(ind[0])
                                anc_ind.append(i)
                                if len(anc_ind) >= config.keypts_num:
                                    break

                        anc_keypts = np.array(anc_ind)
                        pos_keypts = np.array(pos_ind)
                        pos_keypts = pos_keypts + len(anc_points)

                    # No matter how many num_keypts are used for training, test only use 64 pair.
                    if len(anc_keypts) >= config.keypts_num:
                        if split == 'train':
                            selected_ind = np.random.choice(range(len(anc_keypts)), config.keypts_num, replace=False)
                        else:
                            selected_ind = np.random.choice(range(len(anc_keypts)), 64, replace=False)
                        anc_keypts = anc_keypts[selected_ind]
                        pos_keypts = pos_keypts[selected_ind]
                    else:  # if can not build enough correspondence, then skip this fragments pair.
                        continue

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
                # print("\t Yield ", anc_id, pos_id)
                anc_points_list = []
                pos_points_list = []
                anc_keypts_list = []
                pos_keypts_list = []
                backup_anc_points_list = []
                backup_pos_points_list = []
                ti_list = []
                ti_list_pos = []
                import time
                # time.sleep(0.3)

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

    def prepare_geometry_registration(self):
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
            self.test_path = 'data/3DMatch/fragments/{}'.format(scene)
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = open3d.read_point_cloud(join(self.test_path, ind))
                pcd = open3d.voxel_down_sample(pcd, voxel_size=0.03)

                keypts_location = np.fromfile(join(self.test_path, ind.replace("ply", "keypts.bin")), dtype=np.float32)
                num_keypts = int(keypts_location[0])
                keypts_location = keypts_location[1:].reshape([num_keypts, 3])

                # find the keypoint indices
                kdtree = open3d.KDTreeFlann(pcd)
                keypts_id = []
                for j in range(keypts_location.shape[0]):
                    _, id, _ = kdtree.search_knn_vector_3d(keypts_location[j], 1)
                    keypts_id.append(id[0])
                # Load points and labels
                points = np.array(pcd.points)
                keypts_id = np.array(keypts_id)

                self.anc_points['test'] += [points]
                self.pos_points['test'] += [points]
                self.anc_keypts['test'] += [keypts_id]
                self.pos_keypts['test'] += [keypts_id]
                self.idpair_list['test'] += "{0},{1}".format(scene + '/' + ind, scene + '/' + ind)
                self.anc_to_pos['test'][ind] = [ind]
                self.ids_list['test'] += [scene + '/' + ind]
        return
