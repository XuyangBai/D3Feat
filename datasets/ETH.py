# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import pickle
import open3d
open3d.set_verbosity_level(open3d.VerbosityLevel.Error)
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
from datasets.ThreeDMatch import rotate

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling



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
        self.anc_points = {'train': [], 'val': [], 'test': []}
        self.anc_keypts = {'test': []}
        self.idpair_list = {'train': [], 'val': [], 'test': []}
        self.anc_to_pos = {'train': {}, 'val': {}, 'test': {}}
        self.ids_list = {'train': [], 'val': [], 'test': []}
        self.trans = {'train': {}, 'val': {}, 'test': {}}

        if self.load_test:
            self.prepare_geometry_registration_eth()
        else:
            self.prepare_eth_ply(split='train')
            self.prepare_eth_ply(split='val')
    

    def prepare_eth_ply(self, split='train'):
        if split == 'train':
            scene_list = ['gazebo_summer', 'gazebo_winter']
        if split == 'val':
            scene_list = ['wood_summer', 'wood_autmn']
        
        for scene in scene_list:
            scene_path = os.path.join("data/ETH/", scene)
            ids = [scene + '/' + str(filename.split(".")[0]) for filename in os.listdir(scene_path) if filename.endswith('ply')]
            ids = sorted(ids, key=lambda x: int(x.split("_")[-1]))  
            self.ids_list[split] += ids 

            pcd_list = [filename for filename in os.listdir(scene_path) if filename.endswith('ply')]
            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = open3d.read_point_cloud(join(scene_path, ind))
                pcd = open3d.voxel_down_sample(pcd, voxel_size=0.04)
                points = np.asarray(pcd.points)
                self.anc_points[split] += [points]

            # overlap_file = os.path.join(scene_path, 'overlapMatrix.csv')
            # overlap = np.genfromtxt(overlap_file, delimiter=',')
            # for i in range(overlap.shape[0]):
            #     for j in range(overlap.shape[1]):
            #         if i != j and overlap[i, j] > 0.3:
            #             anc_id = self.ids_list[split][i]
            #             pos_id = self.ids_list[split][j]
            #             self.idpair_list[split] += [anc_id + ',' + pos_id]
            #             if anc_id not in self.anc_to_pos[split].keys():
            #                 self.anc_to_pos[split][anc_id] = [pos_id]
            #             else:
            #                 self.anc_to_pos[split][anc_id] += [pos_id]

            from geometric_registration.utils import loadlog
            gt_trans = loadlog(scene_path)
            self.trans[split][scene] = gt_trans
            for pair in list(gt_trans.keys()):
                i = int(pair.split('_')[0])
                j = int(pair.split('_')[1])
                anc_id = scene + '/' + 'Hokuyo_' + str(i)
                pos_id = scene + '/' + 'Hokuyo_' + str(j)
                self.idpair_list[split] += [anc_id + ',' + pos_id]
                if anc_id not in self.anc_to_pos[split].keys():
                    self.anc_to_pos[split][anc_id] = [pos_id]
                else:
                    self.anc_to_pos[split][anc_id] += [pos_id]


        if split == 'train':
            self.num_train = len(self.idpair_list[split])
            print("Num train:", len(self.idpair_list[split]))
        else:
            self.num_val = len(self.idpair_list[split])
            print("Num val:", len(self.idpair_list[split]))


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

            # Generator loop
            print(gen_indices)
            for p_i in gen_indices:
                
                if split == 'test':
                    anc_id = self.ids_list[split][p_i]
                    pos_id = self.ids_list[split][p_i]
                    anc_ind = self.ids_list[split].index(anc_id)
                    pos_ind = self.ids_list[split].index(pos_id)
                    anc_points = self.anc_points[split][anc_ind].astype(np.float32)
                    pos_points = self.anc_points[split][pos_ind].astype(np.float32)

                else:
                    print(split, self.idpair_list[split][p_i])
                    anc_id = self.idpair_list[split][p_i].split(",")[0]
                    pos_id = self.idpair_list[split][p_i].split(",")[1]
                
                    anc_ind = self.ids_list[split].index(anc_id)
                    pos_ind = self.ids_list[split].index(pos_id)
                    anc_points = self.anc_points[split][anc_ind].astype(np.float32)
                    pos_points = self.anc_points[split][pos_ind].astype(np.float32)

                    scene = anc_id.split("/")[0]
                    anc_num = anc_id.split("_")[-1].replace(".ply", "")
                    pos_num = pos_id.split("_")[-1].replace(".ply", "")
                    if int(anc_num) < int(pos_num):
                        key = anc_num + '_' + pos_num
                        trans = self.trans[split][scene][key]
                        pos_pcd = open3d.PointCloud()
                        pos_pcd.points = open3d.utility.Vector3dVector(pos_points)
                        pos_pcd.transform(trans)
                        pos_points = np.array(pos_pcd.points)
                    else:
                        key = pos_num + '_' + anc_num
                        trans = self.trans[split][scene][key]
                        anc_pcd = open3d.PointCloud()
                        anc_pcd.points = open3d.utility.Vector3dVector(anc_points)
                        anc_pcd.transform(trans)
                        anc_points = np.array(anc_pcd.points)
                
                # back up point cloud
                backup_anc_points = anc_points
                backup_pos_points = pos_points

                n = anc_points.shape[0] + pos_points.shape[0]
                    
                if split == 'test': # for test, use all 5000 the anc_keypts 
                    anc_keypts = self.anc_keypts[split][anc_ind].astype(np.int32)
                    pos_keypts = anc_keypts + len(anc_points)
                    # pos_keypts = self.pos_keypts[split][ind].astype(np.int32) + len(anc_points)
                    assert(np.array_equal(anc_points, pos_points))
                    assert(np.array_equal(anc_keypts, pos_keypts - len(anc_points)))
                else:
                    print(split, self.idpair_list[split][p_i], anc_points.shape[0], pos_points.shape[0])
                    # here the anc_keypts and pos_keypts is useless
                    # anc_keypts = np.random.choice(len(anc_points), 10)
                    # pos_keypts = np.random.choice(len(pos_points), 10)
                    if anc_points.shape[0] > 60000:
                        anc_points = anc_points[np.random.choice(anc_points.shape[0], 60000, replace=False)]
                    if pos_points.shape[0] > 60000:
                        pos_points = pos_points[np.random.choice(pos_points.shape[0], 60000, replace=False)]
                    if anc_points.shape[0] < 2000 or pos_points.shape[0] < 2000:
                        continue

                    anc_keypts = np.random.choice(len(anc_points), 1000)
                    pos_pcd = open3d.PointCloud()
                    pos_pcd.points = open3d.utility.Vector3dVector(pos_points)
                    kdtree = open3d.geometry.KDTreeFlann(pos_pcd)
                    pos_ind = []
                    anc_ind = []
                    for pts, i in zip(anc_points[anc_keypts], anc_keypts):
                        _, ind, dis = kdtree.search_knn_vector_3d(pts, 1)
                        if dis[0] < 0.03 and ind[0] not in pos_ind and i not in anc_ind:
                            pos_ind.append(ind[0])
                            anc_ind.append(i)
                            if len(anc_ind) >= config.keypts_num:
                                break
                    anc_keypts = np.array(anc_ind)
                    pos_keypts = np.array(pos_ind)
                    pos_keypts = pos_keypts + len(anc_points)
                    print(split, self.idpair_list[split][p_i], len(anc_ind))
                    
                    # No matter how many num_keypts are used for training, test only use 64 pair.
                    if len(anc_keypts) >= config.keypts_num:
                        selected_ind = np.random.choice(range(len(anc_keypts)), config.keypts_num, replace=False)
                        anc_keypts = anc_keypts[selected_ind]
                        pos_keypts = pos_keypts[selected_ind]
                    else: # if can not build enough correspondence, then skip this fragments pair.
                        continue
                        print("Can build corr for:", self.idpair_list[split][p_i])
                    

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

                yield (np.concatenate(anc_points_list + pos_points_list, axis=0), # anc_points
                        np.concatenate(anc_keypts_list, axis=0),     # anc_keypts
                        np.concatenate(pos_keypts_list, axis=0),    
                        np.array(ti_list + ti_list_pos, dtype=np.int32),       # anc_obj_index
                        np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]), # anc_stack_length 
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
                import time

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
                # with open(os.path.join(keypts_path, f'Hokuyo_{i}_Keypoints.txt'), 'r') as f:
                #     keypts_id = np.array([int(x.replace("\n", "")) for x in f.readlines()])
                # keypts_location = np.asarray(pcd.points)[keypts_id]

                keypts_id = np.array([])
                pcd = open3d.voxel_down_sample(pcd, voxel_size=0.0625)

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
                self.anc_keypts['test'] += [keypts_id]
                self.ids_list['test'] += [scene + '/' + ind]
        return