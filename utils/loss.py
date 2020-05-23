# loss.py ---
#
# Filename: loss.py
# Description: Code is the courtesy of Alexander Hermans and is based on the
#              https://github.com/VisualComputingInstitute/triplet-reid paper. If you use it please consider citing.
# Author: Alexander Hermans
# Project: In Defense of the Triplet Loss for Person Re-Identification
# Version: 1.0

# Code:

import numbers
import numpy as np
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))


cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))


def desc_loss(dists, pids, pos_margin=0.1, neg_margin=1.4, false_negative_mask=None):
    """Computes the contrastive loss.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        pos_margin, neg_margin (float): the margin for contrastive loss
        false_negative_mask (2D tensor): A boolean matrix to indicate the false negative within the safe_radius.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("desc_loss"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        if false_negative_mask is not None: 
            negative_mask = tf.logical_and(negative_mask, tf.logical_not(false_negative_mask))
            negative_mask.set_shape([None, None])
        # positive_mask = tf.logical_xor(same_identity_mask,
        #                              tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists * tf.cast(same_identity_mask, tf.float32), axis=1)
        # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])), (dists, negative_mask), tf.float32)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)
        # Another way of achieving the same, though more hacky:
        # closest_negative_col = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)
        # closest_negative_row = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=0)
        # closest_negative = tf.minimum(closest_negative_col, closest_negative_row)

        # # calculate average negative to monitor the training
        # average_negative = tf.map_fn(lambda x: tf.reduce_mean(tf.boolean_mask(x[0], x[1])), (dists, negative_mask), tf.float32)
        average_negative = tf.reduce_mean(dists * tf.cast(negative_mask,  tf.float32)) * tf.cast(tf.size(pids), tf.float32) / (tf.cast(tf.size(pids), tf.float32) - 1.0)
        # average_diff = tf.reduce_mean(furthest_positive - average_negative)
        diff = furthest_positive - closest_negative
        accuracy = tf.reduce_sum(tf.cast(tf.greater_equal(0., diff), tf.float32)) / tf.cast(tf.shape(diff)[0], tf.float32)

        # contrastive loss
        diff = tf.maximum(furthest_positive - pos_margin, 0.0) + tf.maximum(neg_margin - closest_negative, 0.0)
        return tf.reduce_mean(diff), accuracy, tf.reduce_mean(furthest_positive), tf.reduce_mean(average_negative)


def circle_loss(dists, pids, pos_margin=0.1, neg_margin=1.4, false_negative_mask=None):
    """Computes the circle loss proposed in https://arxiv.org/abs/2002.10857.
        Note that my implementation is slightly different: I use distance instead of cosine similarity && I set the optimal and margin to be the same.
        In my case, the circle loss gives better convergence, the performance is similar with original contrastive loss + hard mining

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        pos_margin, neg_margin (float): the margin for contrastive loss
        false_negative_mask (2D tensor): A boolean matrix to indicate the false negative within the safe_radius.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("desc_loss"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        if false_negative_mask is not None: 
            negative_mask = tf.logical_and(negative_mask, tf.logical_not(false_negative_mask))
            negative_mask.set_shape([None, None])
            
        furthest_positive = tf.reduce_max(dists * tf.cast(same_identity_mask, tf.float32), axis=1)
        # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])), (dists, negative_mask), tf.float32)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)   
        average_negative = tf.reduce_mean(dists * tf.cast(negative_mask,  tf.float32)) * tf.cast(tf.size(pids), tf.float32) / (tf.cast(tf.size(pids), tf.float32) - 1.0)
        diff = furthest_positive - closest_negative
        accuracy = tf.reduce_sum(tf.cast(tf.greater_equal(0., diff), tf.float32)) / tf.cast(tf.shape(diff)[0], tf.float32)
        
        # circle loss
        log_scale = 25
        # log_scale = tf.Variable(10.0, trainable=True)
        # log_scale = tf.Print(log_scale, [log_scale],  'log_scale')
        pos_optimal = pos_margin
        neg_optimal = neg_margin
        # use LSE to approximate the max function:  LSE(pos - pos_margin) <==> max(pos - pos_margin) <===> furthest_positive - pos_margin
        # lse_positive = tf.map_fn(lambda x: tf.math.reduce_logsumexp(log_scale * (tf.boolean_mask(x[0], x[1]) - pos_margin) * tf.maximum(0., tf.stop_gradient(tf.boolean_mask(x[0], x[1]) - pos_optimal))), 
        #                          elems=(dists, same_identity_mask), 
        #                          dtype=tf.float32
        #                          )
        lse_positive = log_scale * (furthest_positive - pos_margin) # this is because we only have one positive.
        # lse_positive = tf.Print(lse_positive, [lse_positive], 'lse_positive')
        # use LSE to approximate the max function:  LSE(neg_margin - neg) <==> max(neg_margin - neg) <===> neg_margin - closest_negative
        # lse_negative = tf.map_fn(lambda x: tf.math.reduce_logsumexp(log_scale * (neg_margin - tf.boolean_mask(x[0], x[1])) * tf.maximum(0., tf.stop_gradient(neg_optimal - tf.boolean_mask(x[0], x[1])))), 
        #                          elems=(dists, negative_mask), 
        #                          dtype=tf.float32
        #                          )
        # there are some tricks here, we add 1e8 to eliminate the effect of positve and false negative fairs.
        neg = dists + 1e8 * tf.cast(false_negative_mask, tf.float32) + 1e8 * tf.cast(same_identity_mask, tf.float32)
        lse_negative = tf.math.reduce_logsumexp(log_scale * (neg_margin - neg) * tf.maximum(0.0, tf.stop_gradient(neg_optimal - neg)), axis=-1)
        # lse_negative = tf.Print(lse_negative, [lse_negative], 'lse_negative')
        # circle loss idea -> triplet loss
        loss = tf.math.softplus(lse_positive + lse_negative) / log_scale
        # # circle  loss idea -> contrastive loss
        # loss = (tf.math.softplus(lse_positive) + tf.math.softplus(lse_negative)) / log_scale
        return tf.reduce_mean(loss), accuracy, tf.reduce_mean(furthest_positive), tf.reduce_mean(average_negative)
    

def det_loss(dists, score1, score2, pids):
    with tf.name_scope("det_loss"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1), tf.expand_dims(pids, axis=0))

        furthest_positive = tf.reduce_max(dists * tf.cast(same_identity_mask, tf.float32), axis=1)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = tf.expand_dims(furthest_positive - closest_negative, 1)
        score_loss = tf.reduce_mean(diff * (score1 + score2 + 1e-6))
        # score_loss = tf.Print(score_loss, [score_loss], message="score_loss")
        return score_loss


LOSS_CHOICES = {
    'circle_loss': circle_loss,
    'desc_loss': desc_loss,
    'det_loss': det_loss,
}
