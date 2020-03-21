# Basic libs
import os
import tensorflow as tf
import numpy as np
import time
import glob
import random
import pickle
import copy
import open3d

# Dataset parent class
from datasets.common import Dataset
from datasets.ThreeDMatch import rotate

kitti_icp_cache = {}
kitti_cache = {}


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = open3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = open3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


class KITTIDataset(Dataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'data/kitti/config/train_kitti.txt',
        'val': 'data/kitti/config/val_kitti.txt',
        'test': 'data/kitti/config/test_kitti.txt'
    }
    TEST_RANDOM_ROTATION = False
    IS_ODOMETRY = True
    MAX_TIME_DIFF = 3

    def __init__(self, input_threads=8, first_subsampling_dl=0.30, load_test=False):
        Dataset.__init__(self, 'KITTI')
        self.network_model = 'descriptor'
        self.num_threads = input_threads
        self.load_test = load_test
        self.root = 'data/kitti/'
        self.icp_path = 'data/kitti/icp'
        self.voxel_size = first_subsampling_dl
        self.matching_search_voxel_size = first_subsampling_dl * 1.5

        # Initiate containers
        self.anc_points = {'train': [], 'val': [], 'test': []}
        self.files = {'train': [], 'val': [], 'test': []}

        if self.load_test:
            self.prepare_kitti_ply('test')
        else:
            self.prepare_kitti_ply(split='train')
            self.prepare_kitti_ply(split='val')

    def prepare_kitti_ply(self, split='train'):
        max_time_diff = self.MAX_TIME_DIFF
        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1
        # for dirname in subset_names:
        #     drive_id = int(dirname)
        #     inames = self.get_all_scan_ids(drive_id)
        #     for start_time in inames:
        #         for time_diff in range(2, max_time_diff):
        #             pair_time = time_diff + start_time
        #             if pair_time in inames:
        #                 self.files[split].append((drive_id, start_time, pair_time))
        if split == 'train':
            self.num_train = len(self.files[split])
            print("Num_train", self.num_train)
        elif split == 'val':
            self.num_val = len(self.files[split])
            print("Num_val", self.num_val)
        else:
            # pair (8, 15, 58) is wrong.
            self.files[split].remove((8, 15, 58))
            self.num_test = len(self.files[split])
            print("Num_test", self.num_test)

        for idx in range(len(self.files[split])):
            drive = self.files[split][idx][0]
            filename = self._get_velodyne_fn(drive, self.files[split][idx][1])
            xyzr = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            xyz = xyzr[:, :3]
            self.anc_points[split] += [xyz]

    def get_batch_gen(self, split, config):
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
                # gen_indices = np.random.permutation(self.num_test)
                gen_indices = np.arange(self.num_test)

            print(gen_indices)
            # Generator loop
            for p_i in gen_indices:

                if split == 'test':
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(split, p_i)
                    if flag == False:
                        continue
                else:
                    aligned_anc_points, aligned_pos_points, anc_points, pos_points, matches, trans, flag = self.__getitem__(split, p_i)
                    if flag == False:
                        continue

                anc_id = str(self.files[split][p_i][0]) + "@" + str(self.files[split][p_i][1])
                pos_id = str(self.files[split][p_i][0]) + "@" + str(self.files[split][p_i][2])
                # the backup_points shoule be in the same coordinate
                backup_anc_points = aligned_anc_points
                backup_pos_points = aligned_pos_points
                if split == 'test':
                    anc_keypts = np.array([])
                    pos_keypts = np.array([])
                else:
                    # input to the network should be in different coordinates
                    anc_keypts = matches[:, 0]
                    pos_keypts = matches[:, 1]
                    selected_ind = np.random.choice(range(len(anc_keypts)), config.keypts_num, replace=False)
                    anc_keypts = anc_keypts[selected_ind]
                    pos_keypts = pos_keypts[selected_ind]
                    pos_keypts += len(anc_points)

                if split == 'train' or split == 'val':
                    # data augmentations: noise
                    anc_noise = np.random.rand(anc_points.shape[0], 3) * config.augment_noise
                    pos_noise = np.random.rand(pos_points.shape[0], 3) * config.augment_noise
                    anc_points += anc_noise
                    pos_points += pos_noise
                    # data augmentations: rotation
                    anc_points = rotate(anc_points, num_axis=config.augment_rotation)
                    pos_points = rotate(pos_points, num_axis=config.augment_rotation)
                    # data augmentations: scale
                    scale = config.augment_scale_min + (config.augment_scale_max - config.augment_scale_min) * random.random()
                    anc_points = scale * anc_points
                    pos_points = scale * pos_points
                    # data augmentations: translation
                    anc_points = anc_points + np.random.uniform(-config.augment_shift_range, config.augment_shift_range, 3)
                    pos_points = pos_points + np.random.uniform(-config.augment_shift_range, config.augment_shift_range, 3)

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
                       np.concatenate(backup_anc_points_list + backup_pos_points_list, axis=0),
                       np.array(trans)
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

        # Generator types and shapes
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3], [4, 4])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):
        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points, trans):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config,
                                                         anc_points,
                                                         stacked_features,
                                                         stack_lengths,
                                                         batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points, trans]

        return tf_map

    def get_all_scan_ids(self, drive_id):
        if self.IS_ODOMETRY:
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        else:
            fnames = glob.glob(self.root + '/' + self.date +
                               '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def __getitem__(self, split, idx):
        drive = self.files[split][idx][0]
        t0, t1 = self.files[split][idx][1], self.files[split][idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                if self.IS_ODOMETRY:
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                else:
                    M = self.get_position_transform(positions[0], positions[1], invert=True).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           open3d.registration.TransformationEstimationPointToPoint(),
                                                           open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                # open3d.draw_geometries([pcd0, pcd1])
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        trans = M2

        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)
        pcd0 = open3d.voxel_down_sample(pcd0, self.voxel_size)
        pcd1 = open3d.voxel_down_sample(pcd1, self.voxel_size)
        unaligned_anc_points = np.array(pcd0.points)
        unaligned_pos_points = np.array(pcd1.points)

        # Get matches
        # if True:
        if split == 'train' or split == 'val':
            matching_search_voxel_size = self.matching_search_voxel_size
            matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
            if len(matches) < 1024:
                # raise ValueError(f"{drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
                print(f"Not enought corr: {drive}, {t0}, {t1}, {len(matches)}/{len(pcd0.points)}")
                return (None, None, None, None, None, None, False)
        else:
            matches = np.array([])

        # align the two point cloud into one corredinate system.
        matches = np.array(matches)
        pcd0.transform(trans)
        anc_points = np.array(pcd0.points)
        pos_points = np.array(pcd1.points)

        return (anc_points, pos_points, unaligned_anc_points, unaligned_pos_points, matches, trans, True)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in kitti_cache:
                kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return kitti_cache[data_path]
            else:
                return kitti_cache[data_path][indices]
        else:
            data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
            odometry = []
            if indices is None:
                fnames = glob.glob(self.root + '/' + self.date +
                                   '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
                indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            for index in indices:
                filename = os.path.join(data_path, '%010d%s' % (index, ext))
                if filename not in kitti_cache:
                    kitti_cache[filename] = np.genfromtxt(filename)
                    odometry.append(kitti_cache[filename])

            odometry = np.array(odometry)
            return odometry

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0
        else:
            lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

            R = 6378137  # Earth's radius in metres

            # convert to metres
            lat, lon = np.deg2rad(lat), np.deg2rad(lon)
            mx = R * lon * np.cos(lat)
            my = R * lat

            times = odometry.T[-1]
            return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        else:
            fname = self.root + \
                    '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
                        drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
