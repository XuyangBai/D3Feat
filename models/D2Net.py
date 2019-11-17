from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer  # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break
    
    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:
            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, F[layer]), axis=1)

    backup_features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)
    # TODO: whether to use relu
    # features = tf.nn.relu(features)
    # features = tf.square(features)
    # return features

    # Soft Detection Module
    neighbor = inputs['neighbors'][0] # [n_points, n_neighbors]
    in_batches = inputs['in_batches']
    first_pcd_indices = in_batches[0]
    second_pcd_indices = in_batches[1]
    statcked_length = inputs['stack_lengths']
    first_pcd_length = statcked_length[0]
    second_pcd_length = statcked_length[1]

    # add a fake point in the last row for shadow neighbors
    shadow_features = tf.zeros_like(features[:1, :])
    features = tf.concat([features, shadow_features], axis=0)
    shadow_neighbor = tf.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
    neighbor = tf.concat([neighbor, shadow_neighbor], axis=0)

    # if training is False:
    # TODO: normalize the feature to avoid overflow
    # option 1 : use max per sample to norm (D2Net idea)
    point_cloud_feature0 = tf.reduce_max(tf.gather(features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0, 
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1], 
        axis=0) # [n_points, 1]
    features = tf.divide(features, max_per_sample + 1e-6)
    # option 2: L2 normalization 
    # features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)
    # option 3: use max per neighbor to norm.
    # neighbor_max = tf.reduce_max(tf.gather(features, neighbor, axis=0), axis=1)
    # features = 0.1 * (features - neighbor_max)
    # features = tf.Print(features, [features], message='features')

    # https://arxiv.org/pdf/1604.08859.pdf only minus mean
    neighbor_features = tf.gather(features, neighbor, axis=0) # [n_points, n_neighbors, 64]
    neighbor_features_sum = tf.reduce_sum(neighbor_features,axis=-1) # [n_points, n_neighbors]
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True) # [n_points, 1]
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32) # [n_points, 64]
    local_max_score = tf.math.softplus(features - mean_features)  # [n_points, 64]
    # local_max_score = tf.Print(local_max_score, [local_max_score], message='local_max_score without sigma')

    # minus mean and divide by var
    # neighbor_features = tf.gather(features, neighbor, axis=0) # [n_points, n_neighbors, 64]
    # neighbor_features_sum = tf.reduce_sum(neighbor_features,axis=-1) # [n_points, n_neighbors]
    # neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True) # [n_points, 1]
    # neighbor_num = tf.maximum(neighbor_num, 1)
    # mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32) # [n_points, 64]
    # var_features = tf.reduce_sum(tf.square(neighbor_features - tf.expand_dims(mean_features, 1)), axis=1) / tf.cast(neighbor_num, tf.float32) # [n_points, 64]
    # # features = features / (tf.math.reduce_std(neighbor_features, axis=1) + 1e-6) # [n_points, 64]
    # local_max_score = tf.math.softplus((features - mean_features) / (1e-6 + tf.sqrt(var_features)))
    # local_max_score = tf.Print(local_max_score, [local_max_score], message='local_max_score with sigma')



    # # calculate the local maximum score
    # exp = tf.exp(features)  # [n_points, 64]
    # neighbor_exp = tf.gather(exp, neighbor, axis=0)  # [n_points, n_neighbors, 64]
    # max_exp = tf.reduce_max(neighbor_exp, axis=1)
    # local_max_score = exp / (max_exp + 1e-6)
    # sum_exp = tf.reduce_sum(neighbor_exp, axis=1)
    # local_max_score = exp / (sum_exp + 1e-6)

    # calculate the depth-wise max score
    depth_wise_max = tf.reduce_max(features, axis=1, keepdims=True)  # [n_points, 1]
    depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]
    # depth_wise_max_score = tf.Print(depth_wise_max_score, [depth_wise_max_score], message='depth_wise_max_score')

    all_score = local_max_score * depth_wise_max_score
    # use the max score among channel to be the score of a single point. 
    score = tf.reduce_max(all_score, axis=1, keepdims=True)  # [n_points, 1]
    # score = tf.Print(score, [score, tf.count_nonzero(score), tf.reduce_sum(tf.cast(tf.is_nan(score), tf.int32))], message='score before hard selection')


    # # point cloud level normalization
    # # each point's score will be divided by the summation [or max]of all the points' score in the same point cloud.
    # # I use max because the score will be in [0, 1]
    # point_cloud_score0 = tf.reduce_max(tf.gather(score, first_pcd_indices, axis=0))
    # point_cloud_score1 = tf.reduce_max(tf.gather(score, second_pcd_indices, axis=0))
    # pc_score = tf.concat([
    #     tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_score0, 
    #     tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_score1], 
    #     axis=0) # [n_points, 1]
    # # TODO: whether do point cloud level normalization.
    # score = tf.divide(score, pc_score + 1e-12)

    # hard selection + soft selection
    # local_max = tf.reduce_max(neighbor_features, axis=1)
    # is_local_max = tf.equal(features, local_max)
    # is_local_max = tf.Print(is_local_max, [tf.reduce_sum(tf.cast(is_local_max, tf.int32))], message='num of local max')
    # detected = tf.reduce_max(tf.cast(is_local_max, tf.float32), axis=1, keepdims=True)
    # score = score * detected
    # score = tf.Print(score, [score, tf.count_nonzero(score)], message='score')
    
    return backup_features, score[:-1, :]
    # keypts_anc, keypts_pos = select_topK(score, features, inputs, config)

    # anc_keypts = tf.gather(inputs['backup_points'], keypts_anc)
    # from loss import cdist
    # keypts_distance = cdist(anc_keypts, anc_keypts, metric='euclidean')
    # return backup_features, score[:-1, :], keypts_anc, keypts_pos, keypts_distance
    
    # TODO: select topK from pcd1 and find the corresponding points in pcd2, return the index.

# from tensorflow.contrib import autograph
# @autograph.convert()
# def select_topK(score, features, inputs, config):
#     in_batches = inputs['in_batches']
#     first_pcd_indices = in_batches[0]
#     second_pcd_indices = in_batches[1]
#     statcked_length = inputs['stack_lengths']
#     first_pcd_length = statcked_length[0]

#     # Select all local max point in the left point cloud to find correspondence.
#     num_local_max = tf.reduce_sum(tf.cast(tf.greater(tf.gather(score, first_pcd_indices, axis=0), 0), dtype=tf.int32))
#     num_local_max = tf.Print(num_local_max, [num_local_max], message="Num of Max Local in Left Point Cloud Pair")    
#     num_local_max = tf.minimum(num_local_max, 3000) # to avoid out of memory
#     # TODO: not topK, random K
#     keypts_indices = tf.boolean_mask(tf.expand_dims(first_pcd_indices, 1), tf.greater(tf.gather(score, first_pcd_indices, axis=0), 0))
#     keypts_indices = tf.gather(keypts_indices, tf.random_uniform([num_local_max], minval=0, maxval=num_local_max - 1, dtype=tf.int32), axis=0)
#     # _, keypts_indices = tf.nn.top_k(tf.squeeze(tf.gather(score, first_pcd_indices, axis=0)), k=num_local_max)
#     # here the point cloud has been aligned.
#     shadow_point = tf.ones_like(inputs['backup_points'][:1, :]) * 1e6
#     backup_points = tf.concat([inputs['backup_points'], shadow_point], axis=0)
#     anc_pts = tf.gather(backup_points, first_pcd_indices)
#     pos_pts = tf.gather(backup_points, second_pcd_indices)

#     train_pts = tf.gather(anc_pts, keypts_indices, axis=0)
#     keypts_indices = tf.reshape(keypts_indices, [num_local_max, 1])
#     AA = tf.reduce_sum(train_pts * train_pts, 1)
#     BB = tf.reduce_sum(pos_pts * pos_pts, 1)
#     AA = tf.reshape(AA, [-1, 1])
#     BB = tf.reshape(BB, [-1, 1])
#     dis_matrix = AA - 2 * tf.matmul(train_pts, tf.transpose(pos_pts)) + tf.transpose(BB)
#     # diffs = tf.expand_dims(train_pts, axis=1) - tf.expand_dims(pos_pts, axis=0) # [k, second_pcd_length, 3]
#     # dis_matrix = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12) # [k, second_pcd_length]
#     query_idx = tf.argmin(dis_matrix, axis=1) # [k, ]
#     query_idx = tf.reshape(query_idx, [num_local_max, 1])
#     query_idx = tf.add(query_idx, tf.cast(first_pcd_length, tf.int64))
#     distance = tf.reduce_min(dis_matrix, axis=1)
#     distance = tf.reshape(distance, [num_local_max, 1])
#     mask_distance = tf.less_equal(distance, 0.001)
#     mask_score = tf.greater(tf.gather(score, query_idx, axis=0), 0) # if the point is local max, then the score must be larger than 0
#     mask_distance = tf.reshape(mask_distance, [num_local_max, 1])
#     mask_score = tf.reshape(mask_score, [num_local_max, 1])
#     mask = tf.logical_and(mask_score, mask_distance)
#     # num0 = tf.reduce_sum(tf.cast(mask_distance, tf.int32))
#     # num0 = tf.Print(num0, [num0], message="num distance")
#     # num1 = tf.reduce_sum(tf.cast(mask_score, tf.int32))
#     # num1 = tf.Print(num1, [num1], message="num score")
#     keypts_indices_anc = tf.boolean_mask(keypts_indices, mask)
#     keypts_indices_pos = tf.boolean_mask(query_idx,  mask)
#     actual_num_corr = tf.size(keypts_indices_anc)
#     # actual_num_corr = tf.Print(actual_num_corr, [actual_num_corr], message="Correspondence Pair")
#     # Max Num of Correspondence.
#     if tf.size(keypts_indices_anc) > config.keypts_num:
#         indices = tf.random_uniform([config.keypts_num], minval=0, maxval=actual_num_corr - 1, dtype=tf.int32) # to add randomness
#         keypts_indices_anc = tf.gather(keypts_indices_anc, indices, axis=0)
#         keypts_indices_pos = tf.gather(keypts_indices_pos, indices, axis=0)
#     return keypts_indices_anc, keypts_indices_pos