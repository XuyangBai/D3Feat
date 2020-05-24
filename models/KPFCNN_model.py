#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Segmentation model
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
from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import sys
import numpy as np
import shutil
import os

# Convolution functions
from models.D3Feat import assemble_FCNN_blocks
from utils.loss import cdist, LOSS_CHOICES


# ----------------------------------------------------------------------------------------------------------------------
#
#           Model Class
#       \*****************/
#


class KernelPointFCNN:

    def __init__(self, flat_inputs, config):
        """
        Initiate the model
        :param flat_inputs: List of input tensors (flatten)
        :param config: configuration class
        """

        # Model parameters
        self.config = config
        self.tensorboard_root = ''
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path == None:
                # self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
                self.saving_path = time.strftime('results/Log_%m%d%H%M')
                if self.config.is_test:
                    experiment_id = "D3Feat" + time.strftime('%m%d%H%M') + "test"
                else:
                    experiment_id = "D3Feat" + time.strftime('%m%d%H%M')
                snapshot_root = 'snapshot/%s' % experiment_id
                os.makedirs(snapshot_root, exist_ok=True)
                tensorboard_root = 'tensorboard/%s' % experiment_id
                os.makedirs(tensorboard_root, exist_ok=True)
                shutil.copy2(os.path.join('.', 'training_3DMatch.py'), os.path.join(snapshot_root, 'train.py'))
                shutil.copy2(os.path.join('.', 'utils/trainer.py'), os.path.join(snapshot_root, 'trainer.py'))
                shutil.copy2(os.path.join('.', 'models/D3Feat.py'), os.path.join(snapshot_root, 'model.py'))
                shutil.copy2(os.path.join('.', 'utils/loss.py'), os.path.join(snapshot_root, 'loss.py'))
                self.tensorboard_root = tensorboard_root
            else:
                self.saving_path = self.config.saving_path
            if not exists(self.saving_path):
                makedirs(self.saving_path)

        ########
        # Inputs
        ########
        # Sort flatten inputs in a dictionary
        with tf.variable_scope('anchor_inputs'):
            self.anchor_inputs = dict()
            self.anchor_inputs['points'] = flat_inputs[:config.num_layers]
            self.anchor_inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.anchor_inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.anchor_inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.anchor_inputs['features'] = flat_inputs[ind]
            ind += 1
            self.anchor_inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.anchor_inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.anchor_inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            # self.anchor_inputs['augment_scales'] = flat_inputs[ind]
            # ind += 1
            # self.anchor_inputs['augment_rotations'] = flat_inputs[ind]
            # ind += 1
            # self.anchor_inputs['object_inds'] = flat_inputs[ind]
            # ind += 1
            self.anchor_inputs['stack_lengths'] = flat_inputs[ind]
            ind += 1
            self.anc_keypts_inds = tf.squeeze(flat_inputs[ind])
            ind += 1
            self.pos_keypts_inds = tf.squeeze(flat_inputs[ind])
            ind += 1
            self.anc_id = flat_inputs[ind][0]
            self.pos_id = flat_inputs[ind][1]
            ind += 1
            self.anchor_inputs['backup_points'] = flat_inputs[ind]
            if config.dataset == 'KITTI':
                ind += 1
                self.anchor_inputs['trans'] = flat_inputs[ind]
            # self.object_inds = self.anchor_inputs['object_inds']            
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        ########
        # Layers
        ########

        # Create layers
        # with tf.device('/gpu:%d' % config.gpu_id):
        with tf.variable_scope('KernelPointNetwork', reuse=False) as scope:
            self.out_features, self.out_scores = assemble_FCNN_blocks(self.anchor_inputs, self.config, self.dropout_prob)
            anc_keypts = tf.gather(self.anchor_inputs['backup_points'], self.anc_keypts_inds)
            self.keypts_distance = cdist(anc_keypts, anc_keypts, metric='euclidean')
            # self.anchor_keypts_inds, self.positive_keypts_inds, self.keypts_distance = self.anc_key, self.pos_key, self.keypts_distance

        # show all the trainable vairble
        # all_trainable_vars = tf.trainable_variables()
        # for i in range(len(all_trainable_vars)):
        #     print(i, all_trainable_vars[i])
        ########
        # Losses
        ########

        with tf.variable_scope('loss'):
            # calculate the distance between anchor and positive in feature space.
            positiveIDS = tf.range(tf.size(self.anc_keypts_inds))
            positiveIDS = tf.reshape(positiveIDS, [tf.size(self.anc_keypts_inds)])
            self.anc_features = tf.gather(self.out_features, self.anc_keypts_inds)
            self.pos_features = tf.gather(self.out_features, self.pos_keypts_inds)
            dists = cdist(self.anc_features, self.pos_features, metric='euclidean')
            self.dists = dists
            # find false negative pairs (within the safe radius).
            same_identity_mask = tf.equal(tf.expand_dims(positiveIDS, axis=1), tf.expand_dims(positiveIDS, axis=0))
            distance_lessthan_threshold_mask = tf.less(self.keypts_distance, config.safe_radius)
            false_negative_mask = tf.logical_and(distance_lessthan_threshold_mask, tf.logical_not(same_identity_mask))

            # calculate the contrastive loss using the dist
            self.desc_loss, self.accuracy, self.ave_d_pos, self.ave_d_neg = LOSS_CHOICES['circle_loss'](self.dists, 
                                                                                                      positiveIDS, 
                                                                                                      pos_margin=0.1, 
                                                                                                      neg_margin=1.4, 
                                                                                                      false_negative_mask=false_negative_mask)

            # calculate the score loss.
            if config.det_loss_weight != 0:
                self.anc_scores = tf.gather(self.out_scores, self.anc_keypts_inds)
                self.pos_scores = tf.gather(self.out_scores, self.pos_keypts_inds)
                self.det_loss = LOSS_CHOICES['det_loss'](self.dists, self.anc_scores, self.pos_scores, positiveIDS)
                self.det_loss = tf.scalar_mul(self.config.det_loss_weight, self.det_loss)
            else:
                self.det_loss = tf.constant(0, dtype=self.desc_loss.dtype)

            # if the number of correspondence is less than half of keypts num, then skip
            enough_keypts_num = tf.constant(0.5 * config.keypts_num)
            condition = tf.less_equal(enough_keypts_num, tf.cast(tf.size(self.anc_keypts_inds), tf.float32))

            def true_fn():
                return self.desc_loss, self.det_loss, self.accuracy, self.ave_d_pos, self.ave_d_neg

            def false_fn():
                return tf.constant(0, dtype=self.desc_loss.dtype), \
                       tf.constant(0, dtype=self.det_loss.dtype), \
                       tf.constant(-1, dtype=self.accuracy.dtype), \
                       tf.constant(0, dtype=self.ave_d_pos.dtype), \
                       tf.constant(0, dtype=self.ave_d_neg.dtype), \

            self.desc_loss, self.det_loss, self.accuracy, self.ave_d_pos, self.ave_d_neg = tf.cond(condition, true_fn, false_fn)

            # Get L2 norm of all weights
            regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
            self.regularization_loss = self.config.weights_decay * tf.add_n(regularization_losses)
            self.loss = self.desc_loss + self.det_loss + self.regularization_loss

        tf.summary.scalar('desc loss', self.desc_loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('det loss', self.det_loss)
        tf.summary.scalar('d_pos', self.ave_d_pos)
        tf.summary.scalar('d_neg', self.ave_d_neg)
        self.merged = tf.summary.merge_all()
        if self.tensorboard_root != '':
            self.train_writer = tf.summary.FileWriter(self.tensorboard_root + '/train/')
            self.val_writer = tf.summary.FileWriter(self.tensorboard_root + '/val/')

        return

    def regularization_losses(self):

        #####################
        # Regularizatizon loss
        #####################

        # Get L2 norm of all weights
        regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
        self.regularization_loss = self.config.weights_decay * tf.add_n(regularization_losses)

        ##############################
        # Gaussian regularization loss
        ##############################

        gaussian_losses = []
        for v in tf.global_variables():
            if 'kernel_extents' in v.name:
                # Layer index
                layer = int(v.name.split('/')[1].split('_')[-1])

                # Radius of convolution for this layer
                conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** (layer - 1))

                # Target extent
                target_extent = conv_radius / 1.5
                gaussian_losses += [tf.nn.l2_loss(v - target_extent)]

        if len(gaussian_losses) > 0:
            self.gaussian_loss = self.config.gaussian_decay * tf.add_n(gaussian_losses)
        else:
            self.gaussian_loss = tf.constant(0, dtype=tf.float32)

        #############################
        # Offsets regularization loss
        #############################

        offset_losses = []

        if self.config.offsets_loss == 'permissive':

            for op in tf.get_default_graph().get_operations():
                if op.name.endswith('deformed_KP'):
                    # Get deformed positions
                    deformed_positions = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_positions / conv_radius

                    # Loss will be zeros inside radius and linear outside radius
                    # Mean => loss independent from the number of input points
                    radius_outside = tf.maximum(0.0, tf.norm(KP_locs, axis=2) - 1.0)
                    offset_losses += [tf.reduce_mean(radius_outside)]


        elif self.config.offsets_loss == 'fitting':

            for op in tf.get_default_graph().get_operations():

                if op.name.endswith('deformed_d2'):
                    # Get deformed distances
                    deformed_d2 = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Get the distance to closest input point
                    KP_min_d2 = tf.reduce_min(deformed_d2, axis=1)

                    # Normalize KP locations to be independant from layers
                    KP_min_d2 = KP_min_d2 / (KP_extent ** 2)

                    # Loss will be the square distance to closest input point.
                    # Mean => loss independent from the number of input points
                    offset_losses += [tf.reduce_mean(KP_min_d2)]

                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_KP = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_KP / KP_extent

                    # Point should not be close to each other
                    for i in range(self.config.num_kernel_points):
                        other_KP = tf.stop_gradient(tf.concat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], axis=1))
                        distances = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(other_KP - KP_locs[:, i:i + 1, :]), axis=2))
                        repulsive_losses = tf.reduce_sum(tf.square(tf.maximum(0.0, 1.5 - distances)), axis=1)
                        offset_losses += [tf.reduce_mean(repulsive_losses)]

        elif self.config.offsets_loss != 'none':
            raise ValueError('Unknown offset loss')

        if len(offset_losses) > 0:
            self.offsets_loss = self.config.offsets_decay * tf.add_n(offset_losses)
        else:
            self.offsets_loss = tf.constant(0, dtype=tf.float32)

        return self.offsets_loss + self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)
