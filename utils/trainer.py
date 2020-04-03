#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
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
import tensorflow as tf
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import psutil
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Add training ops
        self.add_train_ops(model)

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        """
        print('*************************************')
        sum = 0
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='KernelPointNetwork'):
            #print(var.name, var.shape)
            sum += np.prod(var.shape)
        print('total parameters : ', sum)
        print('*************************************')

        print('*************************************')
        sum = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork'):
            #print(var.name, var.shape)
            sum += np.prod(var.shape)
        print('total parameters : ', sum)
        print('*************************************')
        """

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto(allow_soft_placement=False)
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            exclude_vars = ['softmax', 'head_unary_conv', '/fc/', 'offset']
            restore_vars = my_vars
            for exclude_var in exclude_vars:
                restore_vars = [v for v in restore_vars if exclude_var not in v.name]
            # restore_vars = restore_vars[:-1]
            restorer = tf.train.Saver(restore_vars)
            restorer.restore(self.sess, restore_snap)
            print("Model restored.")

    def add_train_ops(self, model):
        """
        Add training ops on top of the model
        """

        ##############
        # Training ops
        ##############

        with tf.variable_scope('optimizer'):

            # Learning rate as a Variable so we can modify it
            self.learning_rate = tf.Variable(model.config.learning_rate, trainable=False, name='learning_rate')

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, model.config.momentum)

            # Training step op
            gvs = optimizer.compute_gradients(model.loss)
            # my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
            # gvs = optimizer.compute_gradients(model.loss, my_vars[-1:])

            if model.config.grad_clip_norm > 0:

                # Get gradient for deformable convolutions and scale them
                scaled_gvs = []
                for grad, var in gvs:
                    if 'offset_conv' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    if 'offset_mlp' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    else:
                        scaled_gvs.append((grad, var))

                # Clipping each gradient independantly
                capped_gvs = [(tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for grad, var in scaled_gvs]

                # Clipping the whole network gradient (problematic with big network where grad == inf)
                # capped_grads, global_norm = tf.clip_by_global_norm([grad for grad, var in gvs], model.config.grad_clip_norm)
                # vars = [var for grad, var in gvs]
                # capped_gvs = [(grad, var) for grad, var in zip(capped_grads, vars)]

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            else:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)

        ############
        # Result ops
        ############

        # # Add the Op to compare the logits to the labels during evaluation.
        # with tf.variable_scope('results'):
        #
        #     if len(model.config.ignored_label_inds) > 0:
        #         #  Boolean mask of points that should be ignored
        #         ignored_bool = tf.zeros_like(model.labels, dtype=tf.bool)
        #         for ign_label in model.config.ignored_label_inds:
        #             ignored_bool = tf.logical_or(ignored_bool, model.labels == ign_label)
        #
        #         #  Collect logits and labels that are not ignored
        #         inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        #         new_logits = tf.gather(model.logits, inds, axis=0)
        #         new_labels = tf.gather(model.labels, inds, axis=0)
        #
        #         #  Reduce label values in the range of logit shape
        #         reducing_list = tf.range(model.config.num_classes, dtype=tf.int32)
        #         inserted_value = tf.zeros((1,), dtype=tf.int32)
        #         for ign_label in model.config.ignored_label_inds:
        #             reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        #             new_labels = tf.gather(reducing_list, new_labels)
        #
        #         # Metrics
        #         self.correct_prediction = tf.nn.in_top_k(new_logits, new_labels, 1)
        #         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #         self.prob_logits = tf.nn.softmax(new_logits)
        #
        #     else:
        #
        #         # Metrics
        #         self.correct_prediction = tf.nn.in_top_k(model.logits, model.labels, 1)
        #         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #         self.prob_logits = tf.nn.softmax(model.logits)

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, model, dataset, debug_NaN=False):
        """
        Train the model on a particular dataset.
        """

        if debug_NaN:
            # Add checking ops
            self.check_op = tf.add_check_numerics_ops()

        # Parameters log file
        if model.config.saving:
            model.parameters_log()

        # Save points of the kernel to file
        self.save_kernel_points(model, 0)

        if model.config.saving:
            # Training log file
            with open(join(model.saving_path, 'training.txt'), "w") as file:
                file.write('Steps desc_loss det_loss train_accuracy d_pos d_neg time memory\n')

            # Killing file (simply delete this file when you want to stop the training)
            if not exists(join(model.saving_path, 'running_PID.txt')):
                with open(join(model.saving_path, 'running_PID.txt'), "w") as file:
                    file.write('Launched with PyCharm')

        # Train loop variables
        t0 = time.time()
        self.training_step = 0
        self.training_epoch = 0
        mean_dt = np.zeros(2)
        last_display = t0
        epoch_n = 1
        mean_epoch_n = 0

        # self.validation(model, dataset)

        # Initialise iterator with train data
        self.sess.run(dataset.train_init_op)

        ave_d_neg_buf = []
        ave_d_pos_buf = []
        desc_loss_buf = []
        det_loss_buf = []
        accuracy_buf = []
        # Start loop
        while self.training_epoch < model.config.max_epoch:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.train_op,
                       model.merged,
                       model.desc_loss,
                       model.det_loss,
                       #    model.regularization_loss,
                       #    model.l2_loss,
                       model.accuracy,
                       model.dists,
                       model.ave_d_pos,
                       model.ave_d_neg,
                       model.anchor_inputs,
                       model.out_scores,
                       model.out_features,
                       model.anc_keypts_inds,
                       model.pos_keypts_inds
                       ]

                # If NaN appears in a training, use this debug block
                if debug_NaN:
                    all_values = self.sess.run(ops + [self.check_op] + list(dataset.flat_inputs), {model.dropout_prob: 0.5})
                    _, L_desc, L_det, acc, dists = all_values[1:6]
                    if np.isnan(L_det) or np.isnan(L_desc):
                        input_values = all_values[7:]
                        self.debug_nan(model, input_values, probs)
                        a = 1 / 0

                else:
                    # Run normal
                    _, merged, L_desc, L_det, acc, dists, ave_d_pos, ave_d_neg, inputs, scores, features, anc_key, pos_key = self.sess.run(ops, {
                        model.dropout_prob: 0.5})
                    if self.training_step % 1000 == 0:
                        print("Anchor Score:", scores[anc_key].squeeze())
                        print("Positive Score:", scores[pos_key].squeeze())
                    if L_desc != 0:
                        desc_loss_buf.append(L_desc)
                    if acc > 0:
                        accuracy_buf.append(acc)
                    if L_det != 0:
                        det_loss_buf.append(L_det)
                    if ave_d_pos != 0:
                        ave_d_pos_buf.append(ave_d_pos)
                    if ave_d_neg != 0:
                        ave_d_neg_buf.append(ave_d_neg)

                t += [time.time()]

                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} L_out={:5.3f} L_det={:5.3f} Acc={:4.2f} d_pos:{:4.3f} d_neg:{:4.3f}' \
                              '---{:8.2f} ms/batch (Averaged)'
                    print(message.format(self.training_step,
                                         L_desc,
                                         L_det,
                                         acc,
                                         ave_d_pos,
                                         ave_d_neg,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

                # Log file
                if model.config.saving:
                    process = psutil.Process(os.getpid())
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:.3f} {:.3f} {:.2f} {:.2f} {:.2f} {:.3f} {:.1f}\n'
                        file.write(message.format(self.training_step,
                                                  L_desc,
                                                  L_det,
                                                  acc,
                                                  ave_d_pos,
                                                  ave_d_neg,
                                                  t[-1] - t0,
                                                  process.memory_info().rss * 1e-6))

                # Check kill signal (running_PID.txt deleted)
                if model.config.saving and not exists(join(model.saving_path, 'running_PID.txt')):
                    break

                if epoch_n > model.config.epoch_steps:
                    raise tf.errors.OutOfRangeError(None, None, '')

            except tf.errors.OutOfRangeError:
                # Tensorboard
                mean_det_loss = np.mean(det_loss_buf)
                mean_desc_loss = np.mean(desc_loss_buf)
                mean_accuracy = np.mean(accuracy_buf)
                mean_d_pos = np.mean(ave_d_pos_buf)
                mean_d_neg = np.mean(ave_d_neg_buf)
                summary = tf.Summary()
                summary.value.add(tag="desc_loss", simple_value=mean_desc_loss)
                summary.value.add(tag="det_loss", simple_value=mean_det_loss)
                summary.value.add(tag="accuracy", simple_value=mean_accuracy)
                summary.value.add(tag="d_pos", simple_value=mean_d_pos)
                summary.value.add(tag="d_neg", simple_value=mean_d_neg)
                model.train_writer.add_summary(summary, self.training_epoch + 1)
                desc_loss_buf = []
                accuracy_buf = []
                det_loss_buf = []
                ave_d_pos_buf = []
                ave_d_neg_buf = []

                # End of train dataset, update average of epoch steps
                mean_epoch_n += (epoch_n - mean_epoch_n) / (self.training_epoch + 1)
                epoch_n = 0
                self.int = int(np.floor(mean_epoch_n))
                model.config.epoch_steps = int(np.floor(mean_epoch_n))
                if model.config.saving:
                    model.parameters_log()

                # Snapshot
                if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:

                    # Tensorflow snapshot
                    snapshot_directory = join(model.saving_path, 'snapshots')
                    if not exists(snapshot_directory):
                        makedirs(snapshot_directory)
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_epoch + 1)

                    # Save points
                    self.save_kernel_points(model, self.training_epoch)

                # Update learning rate
                if self.training_epoch in model.config.lr_decays:
                    op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                               model.config.lr_decays[self.training_epoch]))
                    self.sess.run(op)

                # Increment
                self.training_epoch += 1

                # Validation
                self.validation(model, dataset)

                # Reset iterator on training data
                self.sess.run(dataset.train_init_op)

            except tf.errors.InvalidArgumentError as e:

                import pdb
                pdb.set_trace()
                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

            # Increment steps
            self.training_step += 1
            epoch_n += 1

        # Remove File for kill signal
        if exists(join(model.saving_path, 'running_PID.txt')):
            remove(join(model.saving_path, 'running_PID.txt'))
        self.sess.close()

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------
    def validation(self, model, dataset):
        self.sess.run(dataset.val_init_op)

        desc_loss_buf = []
        det_loss_buf = []
        accuracy_buf = []
        ave_d_pos_buf = []
        ave_d_neg_buf = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (model.desc_loss,
                       model.det_loss,
                       model.accuracy,
                       model.ave_d_pos,
                       model.ave_d_neg,
                       model.dists,
                       model.out_scores,
                       model.anc_keypts_inds,
                       model.pos_keypts_inds
                       )
                desc_loss, det_loss, accuracy, ave_d_pos, ave_d_neg, dists, scores, anc_key, pos_key = self.sess.run(ops, {model.dropout_prob: 1.0})
                if desc_loss != 0:
                    desc_loss_buf.append(desc_loss)
                t += [time.time()]
                if det_loss != 0:
                    det_loss_buf.append(det_loss)
                if accuracy > 0:
                    accuracy_buf.append(accuracy)
                if ave_d_pos != 0:
                    ave_d_pos_buf.append(ave_d_pos)
                if ave_d_neg != 0:
                    ave_d_neg_buf.append(ave_d_neg)
                t += [time.time()]

                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))
                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))
            except tf.errors.OutOfRangeError:
                break

        # Print instance mean
        mean_desc_loss = np.mean(desc_loss_buf)
        mean_det_loss = np.mean(det_loss_buf)
        mean_accuracy = np.mean(accuracy_buf)
        mean_d_pos = np.mean(ave_d_pos_buf)
        mean_d_neg = np.mean(ave_d_neg_buf)
        summary = tf.Summary()
        summary.value.add(tag="desc_loss", simple_value=mean_desc_loss)
        summary.value.add(tag="det_loss", simple_value=mean_det_loss)
        summary.value.add(tag="accuracy", simple_value=mean_accuracy)
        summary.value.add(tag="d_pos", simple_value=mean_d_pos)
        summary.value.add(tag="d_neg", simple_value=mean_d_neg)
        model.val_writer.add_summary(summary, self.training_epoch)
        print('{:s} Epoch {:3d}: desc_loss = {:.3f} det_loss = {:.3f} accuracy = {:.2f}%  d_pos = {:.3f} d_neg = {:.3f}'.format(model.config.dataset,
                                                                                                                     self.training_epoch,
                                                                                                                     mean_desc_loss,
                                                                                                                     mean_det_loss,
                                                                                                                     mean_accuracy * 100,
                                                                                                                     mean_d_pos,
                                                                                                                     mean_d_neg,))
        if model.config.saving:
            process = psutil.Process(os.getpid())
            with open(join(model.saving_path, 'training.txt'), "a") as file:
                message = '{:s} Epoch {:3d}: desc_loss = {:.3f} det_loss = {:.3f} accuracy = {:.2f}% d_pos = {:.3f} d_neg = {:.3f}\n'
                file.write(message.format(model.config.dataset,
                                          self.training_epoch,
                                          mean_desc_loss,
                                          mean_det_loss,
                                          mean_accuracy * 100,
                                          mean_d_pos,
                                          mean_d_neg)
                           )
        return

    # Saving methods
    # ------------------------------------------------------------------------------------------------------------------

    def save_kernel_points(self, model, epoch):
        """
        Method saving kernel point disposition and current model weights for later visualization
        """

        if model.config.saving:

            # Create a directory to save kernels of this epoch
            kernels_dir = join(model.saving_path, 'kernel_points', 'epoch{:d}'.format(epoch))
            if not exists(kernels_dir):
                makedirs(kernels_dir)

            # Get points
            all_kernel_points_tf = [v for v in tf.global_variables() if 'kernel_points' in v.name
                                    and v.name.startswith('KernelPoint')]
            all_kernel_points = self.sess.run(all_kernel_points_tf)

            # Get Extents
            if False and 'gaussian' in model.config.convolution_mode:
                all_kernel_params_tf = [v for v in tf.global_variables() if 'kernel_extents' in v.name
                                        and v.name.startswith('KernelPoint')]
                all_kernel_params = self.sess.run(all_kernel_params_tf)
            else:
                all_kernel_params = [None for p in all_kernel_points]

            # Save in ply file
            for kernel_points, kernel_extents, v in zip(all_kernel_points, all_kernel_params, all_kernel_points_tf):

                # Name of saving file
                ply_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.ply'
                ply_file = join(kernels_dir, ply_name)

                # Data to save
                if kernel_points.ndim > 2:
                    kernel_points = kernel_points[:, 0, :]
                if False and 'gaussian' in model.config.convolution_mode:
                    data = [kernel_points, kernel_extents]
                    keys = ['x', 'y', 'z', 'sigma']
                else:
                    data = kernel_points
                    keys = ['x', 'y', 'z']

                # Save
                write_ply(ply_file, data, keys)

            # Get Weights
            all_kernel_weights_tf = [v for v in tf.global_variables() if 'weights' in v.name
                                     and v.name.startswith('KernelPointNetwork')]
            all_kernel_weights = self.sess.run(all_kernel_weights_tf)

            # Save in numpy file
            for kernel_weights, v in zip(all_kernel_weights, all_kernel_weights_tf):
                np_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.npy'
                np_file = join(kernels_dir, np_name)
                np.save(np_file, kernel_weights)

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def show_memory_usage(self, batch_to_feed):

        for l in range(self.config.num_layers):
            neighb_size = list(batch_to_feed[self.in_neighbors_f32[l]].shape)
            dist_size = neighb_size + [self.config.num_kernel_points, 3]
            dist_memory = np.prod(dist_size) * 4 * 1e-9
            in_feature_size = neighb_size + [self.config.first_features_dim * 2 ** l]
            in_feature_memory = np.prod(in_feature_size) * 4 * 1e-9
            out_feature_size = [neighb_size[0], self.config.num_kernel_points, self.config.first_features_dim * 2 ** (l + 1)]
            out_feature_memory = np.prod(out_feature_size) * 4 * 1e-9

            print('Layer {:d} => {:.1f}GB {:.1f}GB {:.1f}GB'.format(l,
                                                                    dist_memory,
                                                                    in_feature_memory,
                                                                    out_feature_memory))
        print('************************************')

    def debug_nan(self, model, inputs):
        """
        NaN happened, find where
        """

        print('\n\n------------------------ NaN DEBUG ------------------------\n')

        # First save everything to reproduce error
        file1 = join(model.saving_path, 'all_debug_inputs.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(inputs, f1)

        # Then print a list of the trainable variables and if they have nan
        print('List of variables :')
        print('*******************\n')
        all_vars = self.sess.run(tf.global_variables())
        for v, value in zip(tf.global_variables(), all_vars):
            nan_percentage = 100 * np.sum(np.isnan(value)) / np.prod(value.shape)
            print(v.name, ' => {:.1f}% of values are NaN'.format(nan_percentage))

        print('Inputs :')
        print('********')

        # Print inputs
        nl = model.config.num_layers
        for layer in range(nl):
            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / np.prod(pools.shape)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / np.prod(upsamples.shape)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if model.config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        # Print inputs
        for layer in range(nl):
            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            max_n = np.max(pools)
            nums = np.sum(pools < max_n - 0.5, axis=-1)
            print('min pools =>', np.min(nums))

            max_n = np.max(upsamples)
            nums = np.sum(upsamples < max_n - 0.5, axis=-1)
            print('min upsamples =>', np.min(nums))

        print('\nFinished\n\n')
        time.sleep(0.5)
