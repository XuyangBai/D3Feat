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
from utils.mesh import rasterize_mesh

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

        # Path of the folder containing ply files
        self.rgbdpath = '../Point_Cloud_Descriptor/data/3DMatch/rgbd_fragments'
        # self.rgbdpath = '../s102/data/3DMatch/rgbd_fragments'
        self.writepath = 'data/3DMatch/'
        # self.num_train = 6579
        # self.num_test = 1000
        
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

        if split == 'val':
            split = 'test'
        with open(os.path.join(self.rgbdpath, 'scene_list_{0}.txt'.format(split))) as f:
            scene_list = f.readlines()

        for scene in scene_list:
            scene = scene.replace("\n", "")
            for seq in sorted(os.listdir(os.path.join(self.rgbdpath, scene))):
                if not seq.startswith('seq'):
                    continue
                scene_path = os.path.join(self.rgbdpath, scene + '/{}'.format(seq))
                ids = [scene + "/{}/".format(seq) + str(filename.split(".")[0]) for filename in os.listdir(scene_path) if filename.endswith('ply')]
                ids = sorted(ids, key=lambda x: int(x.split("_")[-1]))
                if split == 'test':
                    self.ids_list['val'] += ids 
                else:
                    self.ids_list[split] += ids
                print("Scene {0}, seq {1}: num ply: {2}".format(scene, seq, len(ids)))

        # ply_path = join(self.plypath, '{:s}_ply'.format(split))
        # if not exists(ply_path):
        #     makedirs(ply_path)
        pklpath = join(self.rgbdpath, 'backup_pkl_5000_fast')
        anc_points_filename = join(self.rgbdpath, '{}_{:.3f}_points.pkl'.format(split, 0.03)) # aligned point cloud 
        anc_keypts_filename = join(pklpath, '{}_{:.3f}_anc_keypts.pkl'.format(split, 0.03))   # keypoint indices for each point cloud
        ids_pair_filename = join(pklpath, '{}_{:.3f}_anc_ids.pkl'.format(split, 0.03))        # ids_pair : {id1}_{id2}

        if split == 'test':
            split = 'val'

        # pos_keypts_filename = join(pklpath, '{}_{:.3f}_pos_keypts.pkl'.format(split, 0.03))
        # pos_points_filename = join(self.writepath, '{}_{:.3f}_pos_points.pkl'.format(split, self.downsample))
        
        if exists(anc_points_filename) and exists(ids_pair_filename) and exists(anc_keypts_filename):
            with open(anc_points_filename, 'rb') as file:
                self.anc_points[split] = pickle.load(file)
                # self.pos_points[split] = pickle.load(file)
            with open(anc_keypts_filename, 'rb') as file:
                self.anc_keypts[split] = pickle.load(file)
            # with open(pos_keypts_filename, 'rb') as file:
            #     self.pos_keypts[split] = pickle.load(file)
            with open(ids_pair_filename, 'rb') as file:
                self.idpair_list[split] = pickle.load(file)
            for idpair in self.idpair_list[split]:
                anc = idpair.split(",")[0]
                pos = idpair.split(",")[1]
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
                print("Num_test", self.num_val)
            assert len(self.ids_list[split]) == len(self.anc_points[split])
            # keys = list(self.anc_to_pos[split].keys())
            return

        # anc_pts_filename = join(self.writepath, '{}_{:.3f}_anc_pts.pkl'.format(split, 0.03))
        # anc_keypts_filename = join(self.writepath, '{}_{:.3f}_anc_keypts.pkl'.format(split, 0.03))
        # pos_pts_filename = join(self.writepath, '{}_{:.3f}_pos_pts.pkl'.format(split, 0.03))
        # pos_keypts_filename = join(self.writepath, '{}_{:.3f}_pos_keypts.pkl'.format(split, 0.03))
        # ids_list_filename = join(self.writepath, '{}_{:.3f}_anc_ids.pkl'.format(split, 0.03))
        # if split == 'test':
        #     split = 'val'

        # if exists(anc_pts_filename) and exists(pos_pts_filename) and exists(pos_keypts_filename) and exists(anc_keypts_filename):
        #     with open(anc_pts_filename, 'rb') as file:
        #         self.anc_points[split] = pickle.load(file)
        #     with open(pos_pts_filename, 'rb') as file:
        #         self.pos_points[split] = pickle.load(file)
        #     with open(pos_keypts_filename, 'rb') as file:
        #         self.pos_keypts[split] = pickle.load(file)
        #     with open(anc_keypts_filename, 'rb') as file:
        #         self.anc_keypts[split] = pickle.load(file)
        #     with open(ids_list_filename, 'rb') as file:
        #         self.ids_list = pickle.load(file)
        #     print("Load pkl from disk")
        #     if split == 'train':
        #         self.num_train = len(self.anc_points[split])
        #         print("Num_train", self.num_train)
        #         # nums = []
        #         # for pcd in self.anc_points[split]:
        #         #     print(pcd.shape)
        #         #     nums.append(pcd.shape[0])
        #         # import pdb 
        #         # pdb.set_trace()
        #     else:
        #         self.num_val = len(self.anc_points['val'])
        #         print("Num_val", self.num_val)
        #     return

        # invalid_id = []
        # for i in range(len(self.ids_list)):
        #     anc_id = self.ids_list[i]
        #     anc_scene = anc_id.split("/")[0]
        #     pos_id = self.ids_list[min(i+1, len(self.ids_list) - 1)]
        #     pos_scene = pos_id.split("/")[0]
        #     if pos_scene != anc_scene:
        #         pos_id = self.ids_list[max(i-1, 0)]
        #     anc_pcd, keypts = load_ply(self.rgbdpath, anc_id, aligned=True, downsample=0.03, return_keypoints=128)
        #     pos_pcd = load_ply(self.rgbdpath, pos_id, aligned=True, downsample=0.03)
        
        #     anc_ref_pts = np.array(anc_pcd.points)[keypts]
        #     kdtree = open3d.geometry.KDTreeFlann(pos_pcd)
        #     pos_ind = []
        #     anc_ind = []
        #     for pts, j in zip(anc_ref_pts, keypts):
        #         _, idx, dis = kdtree.search_knn_vector_3d(pts, 1)
        #         if dis[0] < 0.03 and idx[0] not in pos_ind and j not in anc_ind:
        #             pos_ind.append(idx[0])
        #             anc_ind.append(j)
        #     # remove invalid pairs.
        #     if len(pos_ind) == 0:
        #         invalid_id.append(anc_id)
        #         continue
        #     print(len(pos_ind))
        #     unaligned_anc_pcd = load_ply(self.rgbdpath, anc_id, aligned=False, downsample=0.03)
        #     unaligned_pos_pcd = load_ply(self.rgbdpath, pos_id, aligned=False, downsample=0.03)
        #     self.anc_points[split] += [np.asarray(unaligned_anc_pcd.points)]
        #     self.pos_points[split] += [np.asarray(unaligned_pos_pcd.points)]
        #     self.anc_keypts[split] += [np.array(anc_ind)]
        #     self.pos_keypts[split] += [np.array(pos_ind)]
        #     print('preparing {:s} ply: {:.1f}%'.format(split, 100 * i / len(self.ids_list)))
        # print('Done in {:.1f}s'.format(time.time() - t0))

        # for ind in invalid_id:
        #     self.ids_list.remove(ind)
        #     print("Remove", ind)
        # with open(anc_pts_filename, 'wb') as file:
        #     pickle.dump(self.anc_points[split], file)
        # with open(pos_pts_filename, 'wb') as file:
        #     pickle.dump(self.pos_points[split], file)
        # with open(pos_keypts_filename, 'wb') as file:
        #     pickle.dump(self.pos_keypts[split], file)
        # with open(anc_keypts_filename, 'wb') as file:
        #     pickle.dump(self.anc_keypts[split], file)
        # with open(ids_list_filename, 'wb') as file:
        #     pickle.dump(self.ids_list, file)
        
        # if split == 'train':
        #     self.num_train = len(self.ids_list)
        #     print("Num_train", self.num_train)
        # else:
        #     self.num_val = len(self.ids_list)
        #     print("Num_val", self.num_val)

    def load_subsampled_clouds(self, subsampling_parameter):
        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        if self.load_test:
            print("\n Load Test Point")
            t0 = time.time()
            filename_test = join(self.path, 'test_{:.3f}_record.pkl'.format(subsampling_parameter))
            split_path = join(self.path, '{:s}_ply'.format('test'))
            names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
            # names = np.sort(names)
            # TODO: sort the name so that the return order is the same to the id.
            names = sorted(names, key=lambda x: int(x.split("_")[-1]))
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                if subsampling_parameter > 0:
                    sub_points = grid_subsampling(points.astype(np.float32), sampleDl=subsampling_parameter)
                    self.input_points['test'] += [sub_points]
                else:
                    self.input_points['test'] += [points]
            # Save for later use
            with open(filename_test, 'wb') as file:
                pickle.dump(self.input_points['test'], file)
            lengths = [p.shape[0] for p in self.input_points['test']]
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))
            return
        ################
        # Training files
        ################

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading training points')
        filename_train = join(self.path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))
        filename_val = join(self.path, 'val_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename_train) and exists(filename_val):
            with open(filename_train, 'rb') as file:
                self.input_points['training'] = pickle.load(file)
            with open(filename_val, 'rb') as file:
                self.input_points['val'] = pickle.load(file)
        # Else compute them from original points
        else:

            # Collect training file names
            split_path = join(self.path, '{:s}_ply'.format('train'))
            names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                if subsampling_parameter > 0:
                    sub_points = grid_subsampling(points.astype(np.float32), sampleDl=subsampling_parameter)
                    self.input_points['training'] += [sub_points]
                else:
                    self.input_points['training'] += [points]
            # Save for later use
            with open(filename_train, 'wb') as file:
                pickle.dump(self.input_points['training'], file)

            lengths = [p.shape[0] for p in self.input_points['training']]
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))
            t0 = time.time()

            # Collect validation file names
            split_path = join(self.path, '{:s}_ply'.format('val'))
            names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                if subsampling_parameter > 0:
                    sub_points = grid_subsampling(points.astype(np.float32), sampleDl=subsampling_parameter)
                    self.input_points['val'] += [sub_points]
                else:
                    self.input_points['val'] += [points]
            # Save for later use
            with open(filename_val, 'wb') as file:
                pickle.dump(self.input_points['val'], file)

            lengths = [p.shape[0] for p in self.input_points['val']]
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))
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
                    
                if split == 'test': # for test, use all 5000 the anc_keypts 
                    anc_keypts = self.anc_keypts[split][anc_ind].astype(np.int32)
                    pos_keypts = self.pos_keypts[split][pos_ind].astype(np.int32)
                    assert (np.array_equal(anc_keypts, pos_keypts))
                    anc_keypts = anc_keypts
                    pos_keypts = anc_keypts + len(anc_points)
                    # pos_keypts = self.pos_keypts[split][ind].astype(np.int32) + len(anc_points)
                    assert(np.array_equal(anc_points, pos_points))
                    assert(np.array_equal(anc_keypts, pos_keypts - len(anc_points)))
                    # add rotation to test on Rotated3DMatch
                    # anc_points = rotate(anc_points, num_axis=3)
                    # pos_points = rotate(pos_points, num_axis=3)
                else:
                    # here the anc_keypts and pos_keypts is useless
                    # anc_keypts = np.random.choice(len(anc_points), 10)
                    # pos_keypts = np.random.choice(len(pos_points), 10)
                    if anc_points.shape[0] > 60000 or pos_points.shape[0] > 60000:
                        continue
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
                    else: # if can not build enough correspondence, then skip this fragments pair.
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

                # In case batch is full, yield it and reset it
                # if batch_n + n > self.batch_limit and batch_n > 0:
                # if batch_n > 0 and n > 0:
                #     yield (np.concatenate(anc_points_list + pos_points_list, axis=0), # anc_points
                #            np.concatenate(anc_keypts_list, axis=0),     # anc_keypts
                #            np.concatenate(pos_keypts_list, axis=0),
                #            np.array(ti_list + ti_list_pos, dtype=np.int32),       # anc_obj_index
                #            np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]), # anc_stack_length 
                #     )
                #     print("\t", anc_id, pos_id)
                #     anc_points_list = []
                #     pos_points_list = []
                #     anc_keypts_list = []
                #     pos_keypts_list = []
                #     ti_list = []
                #     ti_list_pos = []
                #     batch_n = 0


                # Update batch size
                # batch_n += n

                # yield (np.concatenate(anc_points_list, axis=0), # anc_points
                #         np.concatenate(anc_keypts_list, axis=0),     # anc_keypts
                #         np.array(ti_list, dtype=np.int32),       # anc_obj_index
                #         np.array([tp.shape[0] for tp in anc_points_list]), # anc_stack_length 
                        
                #         np.concatenate(pos_points_list, axis=0), # pos_points
                #         np.concatenate(pos_keypts_list, axis=0), # pos_keypts 
                #         np.array(ti_list_pos, dtype=np.int32),   # pos_obj_index 
                #         np.array([tp.shape[0] for tp in pos_points_list]), # anc_stack_length 
                #         )
                yield (np.concatenate(anc_points_list + pos_points_list, axis=0), # anc_points
                        np.concatenate(anc_keypts_list, axis=0),     # anc_keypts
                        np.concatenate(pos_keypts_list, axis=0),    
                        np.array(ti_list + ti_list_pos, dtype=np.int32),       # anc_obj_index
                        np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]), # anc_stack_length 
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
                # time.sleep(0.4)

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        # gen_types = (tf.float32, tf.int32, tf.int32, tf.int32,  tf.float32, tf.int32, tf.int32, tf.int32)
        # gen_shapes = ([None, 3], [None], [None], [None], [None, 3], [None], [None], [None])
        gen_types = (tf.float32, tf.int32, tf.int32,  tf.int32, tf.int32, tf.string, tf.float32)
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

        # def tf_map(anc_points, anc_keypts, obj_inds, stack_lengths, pos_points, pos_keypts, obj_inds_pos, stack_lengths_pos):
        #     """
        #     anc_points 和 pos_points 是没有align过的point cloud
        #     但我们需要先用matrix将两个point cloud align好然后再去找matching pairs

        #     From the input point cloud, this function compute all the point clouds at each layer, the neighbors
        #     indices, the pooling indices and other useful variables.
        #     :param stacked_points: Tensor with size [None, 3] where None is the total number of points
        #     :param labels: Tensor with size [None] where None is the number of batch
        #     :param stack_lengths: Tensor with size [None] where None is the number of batch
        #     """
        #     # Get batch indice for each point
        #     batch_inds = self.tf_get_batch_inds(stack_lengths)

        #     # First add a column of 1 as feature for the network to be able to learn 3D shapes
        #     stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
        #     # data augmentation
        #     # noise = tf.random_normal(tf.shape(anc_points), stddev=config.augment_noise)
        #     # anc_points = anc_points + noise
        #     # Get the whole input list
        #     anchor_input_list = self.tf_descriptor_input(config,
        #                                           anc_points,
        #                                           stacked_features,
        #                                           stack_lengths,
        #                                           batch_inds)

        #     # # Add scale and rotation for testing
        #     # positive_input_list += [rots, obj_inds]
        #     batch_inds_pos = self.tf_get_batch_inds(stack_lengths_pos)
        #     stacked_features_pos = tf.ones((tf.shape(pos_points)[0], 1), dtype=tf.float32)
        #     # data augmentation
        #     # noise = tf.random_normal(tf.shape(pos_points), stddev=config.augment_noise)
        #     # pos_points = pos_points + noise
        #     positive_input_list = self.tf_descriptor_input(config, 
        #                                                 pos_points, 
        #                                                 stacked_features_pos,
        #                                                 stack_lengths_pos,
        #                                                 batch_inds_pos)

        #     return anchor_input_list + [anc_keypts] + positive_input_list + [pos_keypts]

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

