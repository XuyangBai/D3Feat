#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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
import open3d
import tensorflow as tf
import numpy as np
import os
import sys
import logging
from os import makedirs
from os.path import exists, join
import time
from datasets.KITTI import make_open3d_point_cloud, make_open3d_feature
import json


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
    xyz0_est = xyz0 @ est[:3, :3].transpose() + est[:3, 3]
    xyz0_gth = xyz0 @ gth[:3, :3].transpose() + gth[:3, 3]
    dists = np.clip(np.sqrt(np.power(xyz0_est - xyz0_gth, 2).sum(1)), a_min=0, a_max=max_dist)
    if weight is not None:
        dists = weight * dists
    return dists.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.avg = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff


class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
            self.experiment_str = restore_snap.split("_")[-1][:8] + "-" + restore_snap.split("-")[-1]
        # for i, var in enumerate(my_vars):
        # print(i, var.name)
        # for v in my_vars:
        #     if 'kernel_points' in v.name:
        #         rescale_op = v.assign(tf.multiply(v, 0.10 / 0.03))
        #         self.sess.run(rescale_op)

        # Add a softmax operation for predictions
        # self.prob_logits = tf.nn.softmax(model.logits)

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def generate_descriptor(self, model, dataset):
        self.sess.run(dataset.test_init_op)

        use_random_points = False
        if use_random_points:
            self.experiment_str = self.experiment_str + '-rand'
        else:
            self.experiment_str = self.experiment_str + '-pred'
        descriptor_path = f'geometric_registration/D3Feat_{self.experiment_str}/descriptors'
        keypoint_path = f'geometric_registration/D3Feat_{self.experiment_str}/keypoints'
        score_path = f'geometric_registration/D3Feat_{self.experiment_str}/scores'
        if not exists(descriptor_path):
            makedirs(descriptor_path)
        if not exists(keypoint_path):
            makedirs(keypoint_path)
        if not exists(score_path):
            makedirs(score_path)

        t = []
        for i in range(dataset.num_test):
            stime = time.time()
            ops = [model.anchor_inputs, model.out_features, model.out_scores, model.anc_id]
            [inputs, features, scores, anc_id] = self.sess.run(ops, {model.dropout_prob: 1.0})
            t += [time.time() - stime]

            if use_random_points:
                num_points = inputs['in_batches'][0].shape[0] - 1
                # keypts_ind = np.random.choice(np.arange(num_points), num_keypts)
                keypts_loc = inputs['backup_points'][:]
                anc_features = features[:]
            else:
                # selecet keypoint based on scores
                scores_first_pcd = scores[inputs['in_batches'][0][:-1]]
                selected_keypoints_id = np.argsort(scores_first_pcd, axis=0)[:].squeeze()
                keypts_score = scores[selected_keypoints_id]
                keypts_loc = inputs['backup_points'][selected_keypoints_id]
                anc_features = features[selected_keypoints_id]

            scene = anc_id.decode("utf-8").split("/")[0]
            num_frag = int(anc_id.decode("utf-8").split("_")[-1][:-4])
            descriptor_path_scene = join(descriptor_path, scene)
            keypoint_path_scene = join(keypoint_path, scene)
            score_path_scene = join(score_path, scene)
            if not exists(descriptor_path_scene):
                os.mkdir(descriptor_path_scene)
            if not exists(keypoint_path_scene):
                os.mkdir(keypoint_path_scene)
            if not exists(score_path_scene):
                os.mkdir(score_path_scene)

            np.save(join(descriptor_path_scene, 'cloud_bin_{}.D3Feat'.format(num_frag)), anc_features.astype(np.float32))
            np.save(join(keypoint_path_scene, 'cloud_bin_{}'.format(num_frag)), keypts_loc.astype(np.float32))
            np.save(join(score_path_scene, 'cloud_bin_{}'.format(num_frag)), keypts_score.astype(np.float32))
            print("Generate cloud_bin_{0} for {1}".format(num_frag, scene))
            print("*" * 40)

        print("Avergae Feature Extraction Time:", np.mean(t))

    def test_kitti(self, model, dataset):
        self.sess.run(dataset.test_init_op)

        use_random_points = False
        if use_random_points:
            num_keypts = 5000
            icp_save_path = f'geometric_registration_kitti/D3Feat_{self.experiment_str}-rand{num_keypts}'
        else:
            num_keypts = 250
            icp_save_path = f'geometric_registration_kitti/D3Feat_{self.experiment_str}-pred{num_keypts}'
        if not exists(icp_save_path):
            makedirs(icp_save_path)

        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

        success_meter, loss_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        feat_timer, reg_timer = Timer(), Timer()

        for i in range(dataset.num_test):
            feat_timer.tic()
            ops = [model.anchor_inputs, model.out_features, model.out_scores, model.anc_id, model.pos_id, model.accuracy]
            [inputs, features, scores, anc_id, pos_id, accuracy] = self.sess.run(ops, {model.dropout_prob: 1.0})
            feat_timer.toc()
            # print(accuracy, anc_id)

            stack_lengths = inputs['stack_lengths']
            first_pcd_indices = np.arange(stack_lengths[0])
            second_pcd_indices = np.arange(stack_lengths[1]) + stack_lengths[0]
            # anc_points = inputs['points'][0][first_pcd_indices]
            # pos_points = inputs['points'][0][second_pcd_indices]
            # anc_features = features[first_pcd_indices]
            # pos_features = features[second_pcd_indices]
            # anc_scores = scores[first_pcd_indices]
            # pos_scores = scores[second_pcd_indices]
            if use_random_points:
                anc_keypoints_id = np.random.choice(stack_lengths[0], num_keypts)
                pos_keypoints_id = np.random.choice(stack_lengths[1], num_keypts) + stack_lengths[0]
                anc_points = inputs['points'][0][anc_keypoints_id]
                pos_points = inputs['points'][0][pos_keypoints_id]
                anc_features = features[anc_keypoints_id]
                pos_features = features[pos_keypoints_id]
                anc_scores = scores[anc_keypoints_id]
                pos_scores = scores[pos_keypoints_id]
            else:
                scores_anc_pcd = scores[first_pcd_indices]
                scores_pos_pcd = scores[second_pcd_indices]
                anc_keypoints_id = np.argsort(scores_anc_pcd, axis=0)[-num_keypts:].squeeze()
                pos_keypoints_id = np.argsort(scores_pos_pcd, axis=0)[-num_keypts:].squeeze() + stack_lengths[0]
                anc_points = inputs['points'][0][anc_keypoints_id]
                anc_features = features[anc_keypoints_id]
                anc_scores = scores[anc_keypoints_id]
                pos_points = inputs['points'][0][pos_keypoints_id]
                pos_features = features[pos_keypoints_id]
                pos_scores = scores[pos_keypoints_id]

            pcd0 = make_open3d_point_cloud(anc_points)
            pcd1 = make_open3d_point_cloud(pos_points)
            feat0 = make_open3d_feature(anc_features, 32, anc_features.shape[0])
            feat1 = make_open3d_feature(pos_features, 32, pos_features.shape[0])

            reg_timer.tic()
            filename = anc_id.decode("utf-8") + "-" + pos_id.decode("utf-8").split("@")[-1] + '.npz'
            if os.path.exists(join(icp_save_path, filename)):
                data = np.load(join(icp_save_path, filename))
                T_ransac = data['trans']
                print(f"Read from {join(icp_save_path, filename)}")
            else:

                distance_threshold = dataset.voxel_size * 1.0
                ransac_result = open3d.registration.registration_ransac_based_on_feature_matching(
                    pcd0, pcd1, feat0, feat1, distance_threshold,
                    open3d.registration.TransformationEstimationPointToPoint(False), 4, [
                        open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        open3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                    ],
                    open3d.registration.RANSACConvergenceCriteria(50000, 1000)
                    # open3d.registration.RANSACConvergenceCriteria(4000000, 10000)
                )
                # print(ransac_result)
                T_ransac = ransac_result.transformation.astype(np.float32)
                np.savez(join(icp_save_path, filename),
                         trans=T_ransac,
                         anc_pts=anc_points,
                         pos_pts=pos_points,
                         anc_scores=anc_scores,
                         pos_scores=pos_scores
                         )
            reg_timer.toc()

            T_gth = inputs['trans']
            # loss_ransac = corr_dist(T_ransac, T_gth, anc_points, pos_points, weight=None, max_dist=1)
            loss_ransac = 0
            rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
            rre = np.arccos((np.trace(T_ransac[:3, :3].transpose() @ T_gth[:3, :3]) - 1) / 2)

            if rte < 2:
                rte_meter.update(rte)

            if not np.isnan(rre) and rre < np.pi / 180 * 5:
                rre_meter.update(rre * 180 / np.pi)

            if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
                success_meter.update(1)
            else:
                success_meter.update(0)
                logging.info(f"{anc_id} Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")

            loss_meter.update(loss_ransac)

            if (i + 1) % 10 == 0:
                logging.info(
                    f"{i+1} / {dataset.num_test}: Feat time: {feat_timer.avg}," +
                    f" Reg time: {reg_timer.avg}, Loss: {loss_meter.avg}, RTE: {rte_meter.avg}," +
                    f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}" +
                    f" ({success_meter.avg * 100} %)"
                )
                feat_timer.reset()
                reg_timer.reset()

        logging.info(
            f"Total loss: {loss_meter.avg}, RTE: {rte_meter.avg}, var: {rte_meter.var}," +
            f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
            f"/ {success_meter.count} ({success_meter.avg * 100} %)"
        )
